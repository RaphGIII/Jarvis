"""Answer questions about this system from the registries, not from the model.

"Which of your capabilities are verified?" is a question about a file on disk.
A language model asked it will produce a fluent, plausible, entirely invented
list, because that is what a language model does with a question it has no way
to look up -- and the user has no way to tell that answer from a real one.

So these questions never reach the model.  They are answered by reading the
capability registry and the project store and rendering what is there, which
means the answer agrees with the Projects panel and with Diagnostics by
construction rather than by coincidence.

An empty registry produces "nothing is registered", not silence and not a
guess.  That is the honest answer and it is the one the system currently owes:
the registry on this machine holds zero capabilities, and every previous way of
asking produced prose suggesting otherwise.
"""

from __future__ import annotations

from typing import Any


def answer(core: Any, text: str, *, language: str = "", acquisition: bool = False) -> str:
    """Render a system-state question from real records."""

    german = language.startswith("de")
    lowered = f" {(text or '').lower()} "

    if acquisition:
        return _cannot_acquire(core, german)

    developer_words = ("entwickler", "developer", "selfdev", "self-development",
                       "selbstentwicklung", "programmier")
    if any(word in lowered for word in developer_words):
        return _developer(core, german)

    project_words = ("projekt", "project")
    if any(word in lowered for word in project_words):
        return _projects(core, german)
    return _capabilities(core, german)


def _cannot_acquire(core: Any, german: bool) -> str:
    """"Learn to do X" -- said plainly, then what is actually installed.

    The alternative is a conversational answer, and a model asked to learn
    something will happily report that it now can. That is a present-tense
    capability claim rather than a claim about a completed action, so the claim
    guard does not see it; the only reliable answer is to not ask the model.
    """

    head = (
        "Ich kann mir ueber den Chat keine neue Faehigkeit beibringen -- dafuer gibt es "
        "die Capability-Akquise, und sie laeuft nicht aus einer Unterhaltung heraus. "
        "Nichts wurde gestartet.\n\n"
        if german
        else "I cannot acquire a new capability from a chat turn. That runs through the "
        "capability-acquisition pipeline, not through conversation. Nothing was started.\n\n"
    )
    return head + _capabilities(core, german)


def _developer(core: Any, german: bool) -> str:
    """"What can you do as a developer?" -- the wiring, then the record.

    Counted from the same stores the Activity and Projects panels read, so this
    answer cannot drift away from them.
    """

    try:
        missions = list(core.selfdev_store.list())
    except Exception:
        missions = []
    try:
        projects = core.list_projects()
    except Exception:
        projects = []
    phases = "UNDERSTAND -> INVESTIGATE -> BUILD_LOCAL -> VERIFY -> ESCALATE -> PROMOTE"
    head = (
        "Als Entwickler arbeite ich an mir selbst ueber eine feste Kette:\n\n"
        f"  {phases}\n\n"
        "Jede Aenderung entsteht in einem eigenen Worktree, wird durch die "
        "Akzeptanzkommandos gepruft und erst danach promotet. Ohne bestandene "
        "Pruefung wird nichts uebernommen.\n\n"
        f"Im Speicher: {len(missions)} Self-Development-Mission(en), "
        f"{len(projects)} Projekt(e).\n\n"
        if german
        else
        "As a developer I work on myself through a fixed chain:\n\n"
        f"  {phases}\n\n"
        "Every change is built in its own worktree, checked by the acceptance "
        "commands, and only then promoted. Nothing lands without a passing check.\n\n"
        f"On record: {len(missions)} self-development mission(s), {len(projects)} project(s).\n\n"
    )
    return head + _capabilities(core, german)


def _projects(core: Any, german: bool) -> str:
    projects = core.list_projects()
    if not projects:
        return (
            "Es sind keine Projekte gespeichert."
            if german
            else "There are no projects in the store."
        )
    head = (
        f"{len(projects)} Projekt(e) im Speicher, dieselbe Quelle wie die Projects-Ansicht:"
        if german
        else f"{len(projects)} project(s) in the store -- the same source the Projects panel reads:"
    )
    lines = [head, ""]
    for project in projects:
        name = project.get("title") or project.get("goal") or project["id"]
        lines.append(f"  - {name}  [{project.get('state', '?')}]  id {project['id']}")
    return "\n".join(lines)


def _capabilities(core: Any, german: bool) -> str:
    report = core.capability_report()
    if report.get("error"):
        return (
            f"Die Capability-Registry ist nicht lesbar: {report['error']}"
            if german
            else f"The capability registry could not be read: {report['error']}"
        )

    active = report.get("active") or []
    disabled = report.get("disabled") or []
    if not active and not disabled:
        return (
            "In der Capability-Registry ist nichts eingetragen -- keine verifizierten "
            f"Faehigkeiten. Registry: {report.get('path', '')}\n\n"
            "Was ich ohne Registry-Eintrag ausfuehren kann, ist fest verdrahtet: "
            "Datei schreiben, Datei lesen, Projekt anlegen. Jede davon liefert einen Beleg."
            if german
            else "The capability registry is empty -- there are no verified capabilities. "
            f"Registry: {report.get('path', '')}\n\n"
            "What I can execute without a registry entry is fixed: write a file, read a file, "
            "create a project. Each one produces a receipt."
        )

    lines = [
        (
            f"{len(active)} aktive, {len(disabled)} deaktivierte Eintraege in der Registry "
            f"({report.get('path', '')}):"
            if german
            else f"{len(active)} active and {len(disabled)} disabled entries in the registry "
            f"({report.get('path', '')}):"
        ),
        "",
    ]
    for item in active:
        validated = item.get("validation_status") or {}
        mark = "verified" if validated.get("verified") else "active, verification not recorded"
        lines.append(f"  - {item.get('capability_id', '?')} v{item.get('version', '?')}  [{mark}]")
    for item in disabled:
        reason = (item.get("validation_status") or {}).get("disabled_reason", "")
        lines.append(f"  - {item.get('capability_id', '?')}  [disabled] {reason}")
    return "\n".join(lines)
