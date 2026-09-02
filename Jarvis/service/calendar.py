"""The local-first calendar: real events, real times, no cloud required.

Events live in one JSON file under the state root.  Times are stored as ISO
strings WITH the local UTC offset, so an event survives a timezone change
with its wall-clock meaning intact and .ics export carries real instants.

The German datetime parser turns "Trag morgen um 14 Uhr Lernen ein" into a
typed event: relative days (heute/morgen/übermorgen), weekdays, explicit
dates (5.9., 05.09.2026, 5. September), times (14 Uhr, 14:30, 14 Uhr 30)
and durations (für 30 Minuten, für zwei Stunden).  What it cannot parse it
reports as missing — the caller asks ONE question instead of guessing.

External providers (Google, Outlook) can plug in later as importers/
exporters; the local model stays the source of truth.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

WEEKDAYS = {"montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3,
            "freitag": 4, "samstag": 5, "sonnabend": 5, "sonntag": 6,
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6}
MONTHS = {"januar": 1, "februar": 2, "maerz": 3, "märz": 3, "april": 4, "mai": 5,
          "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
          "november": 11, "dezember": 12}

_TIME = re.compile(r"\b(?:um\s+)?(?P<h>\d{1,2})(?::(?P<m1>\d{2})|\s*uhr(?:\s+(?P<m2>\d{1,2}))?)\b", re.I)
_DUR = re.compile(r"\bf(?:ue|ü)r\s+(?P<n>\d+|eine?|zwei|drei|vier)\s+(?P<unit>minuten?|stunden?)\b|\b(?P<n2>\d+)\s+(?P<unit2>minuten?|stunden?)\s+lang\b", re.I)
_DATE_NUM = re.compile(r"\b(?:am\s+)?(?P<d>\d{1,2})\.(?P<mo>\d{1,2})\.(?P<y>\d{2,4})?(?!\d)")
_DATE_NAME = re.compile(r"\b(?:am\s+)?(?P<d>\d{1,2})\.\s*(?P<mo>januar|februar|maerz|märz|april|mai|juni|juli|august|september|oktober|november|dezember)\b", re.I)
_REL = re.compile(r"\b(heute|morgen|(?:ue|ü)bermorgen|today|tomorrow)\b", re.I)
_WD = re.compile(r"\b(?:am\s+|n(?:ae|ä)chsten\s+)?(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonnabend|sonntag|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_WORDNUM = {"eine": 1, "ein": 1, "einer": 1, "zwei": 2, "drei": 3, "vier": 4}

#: Words that are command scaffolding, not the event's name.
_SCAFFOLD = re.compile(
    r"\b(trag|traeg\w*|trage|eintragen|ein|erstelle?|lege?|leg|anlegen|fest|frei|termin\w*|kalender|calendar|"
    r"erinnere|erinnerung|mich|bitte|zeus|jarvis|einen|eine|ein|f(?:ue|ü)r|an|am|um|uhr|minuten?|stunden?|lang|"
    r"neuen|neue|schreib\w*|notier\w*|mir|in|den|meinem|meinen)\b", re.I)


def _local_tz():
    return datetime.now().astimezone().tzinfo


def parse_event(text: str, *, now: datetime | None = None) -> dict[str, Any]:
    """A typed event proposal from German words; `missing` names what is not there."""

    now = now or datetime.now(_local_tz())
    body = str(text or "").strip()
    lowered = body.lower()
    day: date | None = None
    consumed: list[tuple[int, int]] = []

    m = _DATE_NUM.search(lowered)
    if m:
        year = int(m.group("y") or now.year)
        if year < 100:
            year += 2000
        try:
            day = date(year, int(m.group("mo")), int(m.group("d")))
            consumed.append(m.span())
        except ValueError:
            day = None
    if day is None:
        m = _DATE_NAME.search(lowered)
        if m:
            try:
                day = date(now.year, MONTHS[m.group("mo").lower()], int(m.group("d")))
                if day < now.date():
                    day = day.replace(year=now.year + 1)
                consumed.append(m.span())
            except (ValueError, KeyError):
                day = None
    if day is None:
        m = _REL.search(lowered)
        if m:
            word = m.group(1).lower().replace("ü", "ue")
            offset = {"heute": 0, "today": 0, "morgen": 1, "tomorrow": 1, "uebermorgen": 2}[word]
            day = now.date() + timedelta(days=offset)
            consumed.append(m.span())
    if day is None:
        m = _WD.search(lowered)
        if m:
            target = WEEKDAYS[m.group(1).lower()]
            ahead = (target - now.weekday()) % 7 or 7
            day = now.date() + timedelta(days=ahead)
            consumed.append(m.span())

    start_time = None
    m = _TIME.search(lowered)
    if m:
        hour = int(m.group("h"))
        minute = int(m.group("m1") or m.group("m2") or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            # "morgen" before a bare "um 8": afternoon words could refine this,
            # but a stated hour is taken literally — no silent guessing
            start_time = (hour, minute)
            consumed.append(m.span())

    minutes = 60
    m = _DUR.search(lowered)
    if m:
        raw = (m.group("n") or m.group("n2") or "1").lower()
        n = int(raw) if raw.isdigit() else _WORDNUM.get(raw, 1)
        unit = (m.group("unit") or m.group("unit2") or "").lower()
        minutes = n * (60 if unit.startswith("stunde") else 1)
        consumed.append(m.span())

    # the title is what remains after removing datetime phrases and scaffolding
    chars = list(body)
    for a, b in consumed:
        for i in range(a, min(b, len(chars))):
            chars[i] = " "
    remainder = re.sub(r"\s+", " ", _SCAFFOLD.sub(" ", "".join(chars))).strip(" .,!?-–")
    title = remainder or ""

    missing: list[str] = []
    if day is None:
        missing.append("date")
    if start_time is None:
        missing.append("time")
    if not title:
        missing.append("title")

    start = end = None
    if day is not None and start_time is not None:
        start_dt = datetime(day.year, day.month, day.day, start_time[0], start_time[1], tzinfo=_local_tz())
        start = start_dt.isoformat()
        end = (start_dt + timedelta(minutes=minutes)).isoformat()
    return {"title": title, "start": start, "end": end, "duration_minutes": minutes,
            "missing": missing, "timezone": str(_local_tz())}


class CalendarStore:
    """Owner events in one JSON file; every mutation persists immediately."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._events is None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._events = list(raw.get("events") or [])
            except (OSError, ValueError):
                self._events = []
        return self._events

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"events": self._events or []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, *, title: str, start: str, end: str = "", timezone: str = "",
               location: str = "", notes: str = "", project_id: str = "",
               reminder_minutes: int | None = None, source: str = "owner") -> dict[str, Any]:
        if not title.strip():
            raise ValueError("an event needs a title")
        start_dt = datetime.fromisoformat(start)  # validates; raises on garbage
        end_iso = end or (start_dt + timedelta(hours=1)).isoformat()
        datetime.fromisoformat(end_iso)
        event = {
            "id": "ev_" + uuid.uuid4().hex[:10], "title": title.strip(),
            "start": start, "end": end_iso, "timezone": timezone or str(_local_tz()),
            "location": location, "notes": notes, "project_id": project_id,
            "reminder_minutes": reminder_minutes, "reminded_at": "",
            "source": source, "created_at": datetime.now(_local_tz()).isoformat(),
        }
        with self._lock:
            self._load().append(event)
            self._save()
        return dict(event)

    def update(self, event_id: str, **changes: Any) -> dict[str, Any] | None:
        allowed = {"title", "start", "end", "timezone", "location", "notes",
                   "project_id", "reminder_minutes", "reminded_at"}
        with self._lock:
            for event in self._load():
                if event["id"] == event_id:
                    for key, value in changes.items():
                        if key in allowed:
                            event[key] = value
                    for key in ("start", "end"):
                        if event.get(key):
                            datetime.fromisoformat(event[key])
                    self._save()
                    return dict(event)
        return None

    def delete(self, event_id: str) -> bool:
        with self._lock:
            events = self._load()
            kept = [e for e in events if e["id"] != event_id]
            if len(kept) == len(events):
                return False
            self._events = kept
            self._save()
            return True

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            for event in self._load():
                if event["id"] == event_id:
                    return dict(event)
        return None

    def list(self, *, start: str = "", end: str = "", query: str = "") -> list[dict[str, Any]]:
        lo = datetime.fromisoformat(start) if start else None
        hi = datetime.fromisoformat(end) if end else None
        needle = query.strip().lower()
        out = []
        with self._lock:
            for event in self._load():
                try:
                    ev_start = datetime.fromisoformat(event["start"])
                except ValueError:
                    continue
                if lo is not None and ev_start < lo:
                    continue
                if hi is not None and ev_start >= hi:
                    continue
                if needle and needle not in (event.get("title", "") + " " + event.get("notes", "") + " " + event.get("location", "")).lower():
                    continue
                out.append(dict(event))
        out.sort(key=lambda e: e["start"])
        return out

    # -- reminders -------------------------------------------------------

    def due_reminders(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        """Events whose reminder window has opened and was not yet announced."""

        now = now or datetime.now(_local_tz())
        due = []
        with self._lock:
            for event in self._load():
                minutes = event.get("reminder_minutes")
                if minutes is None or event.get("reminded_at"):
                    continue
                try:
                    start = datetime.fromisoformat(event["start"])
                except ValueError:
                    continue
                fire_at = start - timedelta(minutes=int(minutes))
                if fire_at <= now < start + timedelta(minutes=5):
                    event["reminded_at"] = now.isoformat()
                    due.append(dict(event))
            if due:
                self._save()
        return due

    # -- .ics ------------------------------------------------------------

    @staticmethod
    def _ics_stamp(iso: str) -> str:
        dt = datetime.fromisoformat(iso)
        return dt.astimezone().strftime("%Y%m%dT%H%M%S")

    def export_ics(self) -> str:
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//ZEUS//Calendar//DE"]
        with self._lock:
            for e in self._load():
                lines += ["BEGIN:VEVENT", f"UID:{e['id']}@zeus.local",
                          f"DTSTART:{self._ics_stamp(e['start'])}",
                          f"DTEND:{self._ics_stamp(e['end'])}",
                          "SUMMARY:" + e.get("title", "").replace("\n", " ")]
                if e.get("location"):
                    lines.append("LOCATION:" + e["location"].replace("\n", " "))
                if e.get("notes"):
                    lines.append("DESCRIPTION:" + e["notes"].replace("\n", "\\n"))
                lines.append("END:VEVENT")
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines) + "\r\n"

    def import_ics(self, text: str, *, source: str = "ics") -> int:
        """Minimal VEVENT import: DTSTART/DTEND/SUMMARY(/LOCATION/DESCRIPTION)."""

        count = 0
        current: dict[str, str] = {}
        for raw in str(text or "").replace("\r\n ", "").splitlines():
            line = raw.strip()
            if line == "BEGIN:VEVENT":
                current = {}
            elif line == "END:VEVENT":
                try:
                    start = datetime.strptime(current.get("DTSTART", "")[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_local_tz())
                    end = datetime.strptime(current.get("DTEND", "")[:15], "%Y%m%dT%H%M%S").replace(tzinfo=_local_tz())
                    self.create(title=current.get("SUMMARY", "(ohne Titel)"), start=start.isoformat(), end=end.isoformat(),
                                location=current.get("LOCATION", ""), notes=current.get("DESCRIPTION", ""), source=source)
                    count += 1
                except (ValueError, KeyError):
                    pass
            elif ":" in line:
                key, _, value = line.partition(":")
                current[key.split(";", 1)[0].upper()] = value
        return count
