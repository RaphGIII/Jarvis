"""The only way a provider reaches the network -- and the FREE-mode wall.

Providers in :mod:`gateway.providers` hold no HTTP code and no credentials.
They build a request body and hand it to :class:`Transport` together with a
:class:`Ticket`.  Tickets are issued by :meth:`Transport.issue`, and that is
the one place the zero-cost guarantee lives:

* FREE mode + metered route  -> :class:`ZeroCostViolation`, before any bytes move
* metered route, no reservation -> :class:`ReservationRequired`, likewise

A ticket names the provider, so a body for one provider cannot be posted to
another's endpoint, and the URL must sit under the provider's configured base
URL, so a provider implementation cannot quietly talk to a different host.

Credentials are read from the store at send time and attached as headers here.
They are never returned to the caller, and every error body passes through
:func:`gateway.secrets.redact` before it is stored or raised.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from gateway.budget import Reservation
from gateway.config import ProviderConfig
from gateway.health import GatewayError, ProviderStatus, classify_http
from gateway.modes import ChatMode, CostClass
from gateway.secrets import CredentialStore, redact


class ZeroCostViolation(RuntimeError):
    """FREE mode: a metered provider call was requested.  Nothing was sent."""


class ReservationRequired(RuntimeError):
    """A metered call was requested without a budget reservation.  Nothing was sent."""


@dataclass(frozen=True)
class StreamTimeouts:
    """The four bounds of one streamed call, in seconds; ``total`` 0 = no ceiling."""

    connect: float = 10.0
    first_token: float = 20.0
    idle: float = 20.0
    total: float = 150.0

    def to_dict(self) -> dict[str, float]:
        return {"connect": self.connect, "first_token": self.first_token, "idle": self.idle, "total": self.total}


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    provider: str
    role: str
    mode: ChatMode
    cost_class: CostClass
    reservation_id: str
    issued_at: float = field(default_factory=time.time)


@dataclass
class HttpReply:
    status: int
    data: dict[str, Any]
    latency_seconds: float
    raw_length: int = 0


class Transport:
    def __init__(self, credentials: CredentialStore, *, opener: Callable[..., Any] | None = None) -> None:
        self.credentials = credentials
        self._opener = opener or urllib.request.urlopen
        self.issued: int = 0
        self.refused: list[dict[str, Any]] = []

    # -- the guard -----------------------------------------------------------

    def issue(self, *, provider: ProviderConfig, role: str, mode: ChatMode | str, cost_class: CostClass,
              reservation: Reservation | None) -> Ticket:
        mode = ChatMode.parse(mode)
        if mode is ChatMode.FREE and cost_class is CostClass.METERED:
            self.refused.append({"provider": provider.name, "role": role, "mode": mode.value, "why": "zero-cost mode"})
            raise ZeroCostViolation(f"FREE mode: {provider.name} ({role}) is metered; the call was not made")
        if cost_class is CostClass.METERED and (reservation is None or reservation.reserved_eur <= 0.0):
            self.refused.append({"provider": provider.name, "role": role, "mode": mode.value, "why": "no reservation"})
            raise ReservationRequired(f"{provider.name} ({role}) is metered and no budget was reserved; the call was not made")
        if provider.secret and not self.credentials.has(provider.secret):
            raise GatewayError(ProviderStatus.AUTHENTICATION_ERROR, "no credential configured", role=role, provider=provider.name)
        self.issued += 1
        return Ticket(ticket_id=uuid.uuid4().hex[:10], provider=provider.name, role=role, mode=mode, cost_class=cost_class,
                      reservation_id=reservation.reservation_id if reservation else "")

    # -- sending ---------------------------------------------------------------

    def _check_url(self, provider: ProviderConfig, url: str) -> None:
        base = urllib.parse.urlparse(provider.base_url)
        target = urllib.parse.urlparse(url)
        if not base.netloc or target.scheme != base.scheme or target.netloc != base.netloc:
            raise ValueError(f"request URL {redact(url, self.credentials)!r} is not under provider {provider.name}'s base URL")

    def _request_headers(self, provider: ProviderConfig, headers: dict[str, str] | None, auth: str) -> dict[str, str]:
        request_headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "ZEUS/1.0"}
        request_headers.update(headers or {})
        if provider.secret:
            secret = self.credentials.get(provider.secret)
            if auth == "bearer":
                request_headers["Authorization"] = f"Bearer {secret}"
            elif auth == "x-api-key":
                request_headers["x-api-key"] = secret
            elif auth == "x-goog-api-key":
                request_headers["x-goog-api-key"] = secret
        return request_headers

    def _http_error(self, exc: urllib.error.HTTPError, ticket: Ticket, provider: ProviderConfig) -> GatewayError:
        text = ""
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        text = redact(text, self.credentials)
        status_kind = classify_http(int(exc.code), text, provider_kind=provider.kind)
        retry_after = None
        try:
            header = exc.headers.get("Retry-After") if exc.headers else None
            retry_after = float(header) if header else None
        except (TypeError, ValueError):
            retry_after = None
        if retry_after is None:
            match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', text) or re.search(r"retry in (\d+(?:\.\d+)?)s", text, re.I)
            if match:
                retry_after = float(match.group(1))
        return GatewayError(status_kind, f"HTTP {exc.code}: {text[:500]}", role=ticket.role, provider=provider.name,
                            http_status=int(exc.code), retry_after_seconds=retry_after)

    @staticmethod
    def _timed_out(exc: BaseException) -> bool:
        """Whether a urllib/socket failure is the provider not answering in time."""

        if isinstance(exc, TimeoutError):
            return True
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            return True
        return "timed out" in str(exc).lower()

    def _transport_error(self, exc: BaseException, ticket: Ticket, provider: ProviderConfig, limit: float, *, prefix: str = "") -> GatewayError:
        if self._timed_out(exc):
            return GatewayError(ProviderStatus.TIMEOUT, f"{prefix}no answer within {limit:.0f}s", role=ticket.role, provider=provider.name)
        return GatewayError(ProviderStatus.PROVIDER_UNAVAILABLE, prefix + redact(str(exc), self.credentials)[:300], role=ticket.role,
                            provider=provider.name)

    @staticmethod
    def _socket_of(response: Any) -> Any:
        """The socket under an HTTP response, for re-arming its timeout per phase; None for bodies that have none."""

        for path in (("fp", "raw", "_sock"), ("_sock",), ("sock",)):
            obj: Any = response
            try:
                for name in path:
                    obj = getattr(obj, name)
            except AttributeError:
                continue
            if obj is not None and hasattr(obj, "settimeout"):
                return obj
        return None

    def post_sse(self, ticket: Ticket, provider: ProviderConfig, url: str, body: dict[str, Any], *,
                 headers: dict[str, str] | None = None, auth: str = "bearer", timeout: float | None = None,
                 timeouts: StreamTimeouts | None = None):
        """POST and read the reply as Server-Sent Events: yields ``(event, data)`` as they arrive.

        The same ticket, URL and credential rules as :meth:`post_json`; an
        HTTP failure before the stream opens is classified the same way.
        Closing the generator closes the connection.

        ``timeouts`` gives the stream its four bounds -- the response headers
        (connect), the first event, the gap between events (idle), and the
        whole request -- each armed on the socket for exactly its phase.
        Observed live before this existed: one 120 s silence bound covered
        everything, so a provider that held the connection 115 s before a
        503, or stalled after three chunks, kept the owner waiting two
        minutes.  Without ``timeouts`` the single ``timeout`` (or the
        provider's) bounds every phase, as before.
        """

        if ticket.provider != provider.name:
            raise ValueError(f"ticket for {ticket.provider} used with provider {provider.name}")
        self._check_url(provider, url)
        request_headers = self._request_headers(provider, headers, auth)
        request_headers["Accept"] = "text/event-stream"
        payload = json.dumps(body).encode("utf-8")
        limit = float(timeout or provider.timeout_seconds)
        phases = timeouts or StreamTimeouts(connect=limit, first_token=limit, idle=limit, total=0.0)
        started = time.perf_counter()

        def remaining() -> float:
            return (phases.total - (time.perf_counter() - started)) if phases.total > 0 else float("inf")

        phase = "connect"
        bound = phases.connect if phases.total <= 0 else max(0.1, min(phases.connect, phases.total))
        try:
            request = urllib.request.Request(url, data=payload, headers=request_headers, method="POST")
            response = self._opener(request, timeout=bound)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, ticket, provider) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise self._transport_error(exc, ticket, provider, bound, prefix=f"{phase}: ") from None
        sock = self._socket_of(response)

        def arm(seconds: float) -> None:
            nonlocal bound
            left = remaining()
            bound = seconds if left == float("inf") else max(0.1, min(seconds, left))
            if sock is not None:
                try:
                    sock.settimeout(bound)
                except Exception:  # noqa: BLE001 - a body without a real socket keeps the connect bound
                    pass

        try:
            if not hasattr(response, "__iter__"):
                # Not a stream: the whole body, once.  The adapter reads it as
                # the completed reply it is.
                raw_body = response.read()
                text = raw_body.decode("utf-8", errors="replace") if isinstance(raw_body, (bytes, bytearray)) else str(raw_body)
                yield "", text
                return
            phase = "first token"
            arm(phases.first_token)
            event, data_lines = "", []
            for raw in response:
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.rstrip("\r\n")
                if line == "":
                    if data_lines:
                        yield event, "\n".join(data_lines)
                        if remaining() <= 0:
                            raise GatewayError(ProviderStatus.TIMEOUT, f"total ceiling of {phases.total:.0f}s reached", role=ticket.role,
                                               provider=provider.name)
                        phase = "idle"
                        arm(phases.idle)
                    event, data_lines = "", []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                yield event, "\n".join(data_lines)
        except (TimeoutError, OSError) as exc:
            raise self._transport_error(exc, ticket, provider, bound, prefix=f"stream interrupted ({phase}): ") from None
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass

    def post_json(self, ticket: Ticket, provider: ProviderConfig, url: str, body: dict[str, Any], *,
                  headers: dict[str, str] | None = None, auth: str = "bearer", timeout: float | None = None) -> HttpReply:
        if ticket.provider != provider.name:
            raise ValueError(f"ticket for {ticket.provider} used with provider {provider.name}")
        self._check_url(provider, url)
        request_headers = self._request_headers(provider, headers, auth)
        payload = json.dumps(body).encode("utf-8")
        started = time.perf_counter()
        limit = float(timeout or provider.timeout_seconds)
        try:
            request = urllib.request.Request(url, data=payload, headers=request_headers, method="POST")
            with self._opener(request, timeout=limit) as response:
                raw = response.read()
                status = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = ""
            text = redact(text, self.credentials)
            status_kind = classify_http(int(exc.code), text, provider_kind=provider.kind)
            retry_after = None
            try:
                header = exc.headers.get("Retry-After") if exc.headers else None
                retry_after = float(header) if header else None
            except (TypeError, ValueError):
                retry_after = None
            if retry_after is None:
                # Gemini puts the delay in the body: "retryDelay": "23s".
                match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', text) or re.search(r"retry in (\d+(?:\.\d+)?)s", text, re.I)
                if match:
                    retry_after = float(match.group(1))
            raise GatewayError(status_kind, f"HTTP {exc.code}: {text[:500]}", role=ticket.role, provider=provider.name,
                               http_status=int(exc.code), retry_after_seconds=retry_after) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise self._transport_error(exc, ticket, provider, limit) from None
        latency = time.perf_counter() - started
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise GatewayError(ProviderStatus.TASK_FAILURE, "provider returned a non-JSON body", role=ticket.role,
                               provider=provider.name, http_status=status) from None
        if not isinstance(data, dict):
            data = {"_": data}
        return HttpReply(status=status, data=data, latency_seconds=latency, raw_length=len(raw))
