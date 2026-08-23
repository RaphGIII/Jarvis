from __future__ import annotations

import os
import urllib.request

from brain.providers import ProviderError
from brain.router import BrainRouter, RemoteBrainUnavailable


def configure_default_local_brain() -> None:
    """Use local Ollama by default while still allowing explicit provider overrides."""

    if not os.getenv("JARVIS_BRAIN_PROVIDER"):
        os.environ["JARVIS_BRAIN_PROVIDER"] = "openai_compatible"

    if os.environ.get("JARVIS_BRAIN_PROVIDER") == "openai_compatible":
        os.environ.setdefault(
            "JARVIS_BRAIN_BASE_URL",
            "http://127.0.0.1:11434",
        )
        os.environ.setdefault("JARVIS_BRAIN_MODEL", "qwen3:4b-instruct")
        os.environ.setdefault("JARVIS_BRAIN_API_KEY", "ollama-local")

        if "127.0.0.1:11434" in os.environ["JARVIS_BRAIN_BASE_URL"]:
            os.environ.setdefault(
                "JARVIS_BRAIN_REASONING_EFFORT",
                "none",
            )
            os.environ.setdefault(
                "JARVIS_BRAIN_STRUCTURED_MODE",
                "response_format",
            )


def local_endpoint_online() -> bool:
    base_url = os.getenv("JARVIS_BRAIN_BASE_URL", "").rstrip("/")

    if not base_url:
        return False

    try:
        with urllib.request.urlopen(
            base_url + "/v1/models",
            timeout=2.0,
        ) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def print_banner(router: BrainRouter) -> None:
    status = router.status()
    online = local_endpoint_online()

    print()
    print("JARVIS ONLINE")
    print("--------------------------------")
    print("Core:        READY")
    print(
        f"Fast Brain:  {status['fast_model'] or 'unknown'} "
        f"{'ONLINE' if online else 'OFFLINE'}"
    )
    print(f"Provider:    {status['fast_provider']}")
    print(
        "Cloud Brain: "
        + ("READY" if status["build_remote_loaded"] else "OFF")
    )
    print("--------------------------------")
    print("Commands: /status  /route <goal>  /chat <text>  /quit")
    print()


def print_status(router: BrainRouter) -> None:
    status = router.status()
    status["endpoint_online"] = local_endpoint_online()

    for key, value in status.items():
        print(f"{key}: {value}")


def main() -> None:
    configure_default_local_brain()
    router = BrainRouter()

    print_banner(router)

    while True:
        try:
            raw = input("jarvis> ")
        except (EOFError, KeyboardInterrupt):
            print("\nJarvis> Goodbye.")
            return

        command = raw.strip()

        if not command:
            continue

        if command.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Jarvis> Goodbye.")
            return

        if command == "/status":
            print()
            print_status(router)
            print()
            continue

        if command.startswith("/route"):
            goal = command.removeprefix("/route").strip()

            if not goal:
                goal = input("goal> ").strip()

            decision = router.route(goal)
            print(f"\nJarvis> {decision.tier.value}: {decision.reason}\n")
            continue

        if command.startswith("/chat"):
            command = command.removeprefix("/chat").strip()

            if not command:
                command = input("message> ").strip()

        try:
            answer, decision = router.respond(command)
            print(f"\nJarvis> {answer}")
            print(f"[{decision.tier.value}]\n")

        except RemoteBrainUnavailable as exc:
            print(f"\nJarvis> {exc}")
            print("Jarvis> No cloud compute was started and no paid request was made.\n")

        except ProviderError as exc:
            print(f"\nJarvis> Brain provider error: {exc}\n")

        except Exception as exc:
            print(f"\nJarvis> Runtime error: {type(exc).__name__}: {exc}\n")
