"""Deterministic project operations with independent verification.

A typed :class:`~service.intents.ActionIntent` for a project comes in; a
receipt comes out.  Creation goes through the same engine the CLI uses
(``kernel.start_project``), then the record is reloaded from a *fresh* store
on disk and compared against the contract: the title the owner asked for,
exactly the tasks named, the parent and importance requested, and
visibility through the same listing the interface reads.  EXECUTION_VERIFIED
(the store wrote a file) is kept apart from GOAL_SATISFIED (the file says
what the owner asked for); both are in the receipt.

The owner-facing sentence is composed from the receipt, concisely
("Erledigt. Projekt „M1“ ist angelegt – mit drei Aufgaben."); the evidence
stays in the receipt for Activity and the Beleg link.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from runtime.receipts import Receipt, Verification, failed
from service.intents import ActionIntent

_NUMBERS_DE = {0: "keinen", 1: "einer", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs", 7: "sieben", 8: "acht", 9: "neun", 10: "zehn"}
_NUMBERS_EN = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


class ProjectOperations:
    def __init__(self, core: Any) -> None:
        self.core = core

    # -- helpers ---------------------------------------------------------

    def _store(self) -> Any:
        from projects.store import ProjectStore

        return ProjectStore(Path(self.core.kernel.state_root) / "projects")

    def _find(self, reference: str) -> Any:
        """A project by id or title (exact title first, then the kernel's fuzzy lookup)."""

        if not reference:
            return None
        if reference == "__last__":
            last = getattr(self.core, "_last_project_id", "")
            return self._find(last) if last else None
        projects = self.core.kernel.projects
        try:
            if projects.exists(reference):
                return projects.load(reference)
        except Exception:  # noqa: BLE001
            pass
        wanted = reference.strip().lower()
        try:
            for project in projects.list_projects():
                if (project.title or "").strip().lower() == wanted:
                    return project
            for project in projects.list_projects():
                if wanted in (project.title or "").lower() or wanted in (project.goal or "").lower():
                    return project
        except Exception:  # noqa: BLE001
            pass
        try:
            return self.core.kernel.resolve_project(reference)
        except Exception:  # noqa: BLE001
            return None

    # -- execution -------------------------------------------------------

    def execute(self, intent: ActionIntent, *, request: str = "") -> Receipt:
        started = time.perf_counter()
        handlers = {
            "project.create": self.create, "project.rename": self.rename, "project.add_tasks": self.add_tasks,
            "project.archive": self.archive, "project.delete": self.delete,
        }
        handler = handlers.get(intent.operation)
        if handler is None:
            receipt = failed(intent.operation, "service.project_ops", f"no executor for {intent.operation}", request=request)
        else:
            try:
                receipt = handler(intent, request)
            except Exception as exc:  # noqa: BLE001 - an executor bug is a failed action, never a silent success
                receipt = failed(intent.operation, "service.project_ops", f"{type(exc).__name__}: {exc}", request=request)
        return Receipt(kind=receipt.kind, executor=receipt.executor, ok=receipt.ok, request=request or receipt.request, detail=receipt.detail,
                       evidence=receipt.evidence, verifications=receipt.verifications, id=receipt.id, at=receipt.at,
                       duration_seconds=time.perf_counter() - started)

    def create(self, intent: ActionIntent, request: str) -> Receipt:
        args = intent.arguments
        title = str(args.get("title") or intent.target or "").strip()
        if not title:
            return failed("project.create", "projects.store", "no project title was given", request=request)
        goal = str(args.get("goal") or title).strip()
        tasks = [str(t).strip() for t in (args.get("tasks") or []) if str(t).strip()]
        parent = self._find(str(args.get("parent") or "")) if args.get("parent") else None
        if args.get("parent") and parent is None:
            return failed("project.create", "projects.store", f"no parent project named {args.get('parent')!r}", request=request, parent=args.get("parent"))
        # A second project with the same title is almost never wanted: say so instead of duplicating.
        existing = None
        try:
            for p in self.core.kernel.projects.list_projects():
                if (p.title or "").strip().lower() == title.lower() and str(getattr(p.state, "value", p.state)).lower() not in {"abandoned"}:
                    existing = p
                    break
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None:
            return failed("project.create", "projects.store", f"a project titled {title!r} already exists ({existing.id}); nothing was created",
                          request=request, project_id=existing.id, title=title, duplicate=True)

        project = self.core.kernel.start_project(goal, title=title)
        meta = project.metadata
        if parent is not None:
            meta["parent_id"] = parent.id
            meta["parent_title"] = parent.title
        if args.get("importance"):
            meta["importance"] = str(args["importance"])
        if args.get("deadline"):
            meta["deadline"] = str(args["deadline"])
        if args.get("description"):
            meta["owner_request"] = str(args["description"])[:500]
        meta["origin"] = "owner_request"
        for task in tasks:
            project.add_task(task)
        self.core.kernel.projects.save(project)
        self.core._last_project_id = project.id

        # -- independent reload from a fresh store ---------------------------
        store = self._store()
        reloaded = store.try_load(project.id)
        path = store.path_for(project.id)
        listed = [p for p in store.list_projects() if p.id == project.id]
        verifications = [
            Verification("project file written to disk", path.is_file(), observed=f"{path} ({path.stat().st_size} bytes)" if path.is_file() else f"{path} missing", expected=str(path)),
            Verification("reloaded from a fresh store", reloaded is not None and reloaded.id == project.id,
                          observed=f"id={reloaded.id} title={reloaded.title!r}" if reloaded is not None else "the store returned nothing", expected=f"id={project.id}"),
            Verification("title is what the owner asked for", reloaded is not None and (reloaded.title or "").strip() == title,
                          observed=repr(reloaded.title) if reloaded is not None else "-", expected=repr(title)),
            Verification("visible through the Projects API the UI uses", bool(listed), observed=f"{len(store.list_projects())} project(s) listed", expected=f"a listing containing {project.id}"),
        ]
        if tasks:
            persisted = [t.title for t in (reloaded.tasks if reloaded is not None else [])]
            verifications.append(Verification(f"exactly {len(tasks)} intended task(s) persist", persisted == tasks, observed=str(persisted), expected=str(tasks)))
        if parent is not None:
            verifications.append(Verification("parent recorded", reloaded is not None and reloaded.metadata.get("parent_id") == parent.id,
                                              observed=str(reloaded.metadata.get("parent_id") if reloaded is not None else "-"), expected=parent.id))
        if args.get("importance"):
            verifications.append(Verification("importance recorded", reloaded is not None and reloaded.metadata.get("importance") == args["importance"],
                                              observed=str(reloaded.metadata.get("importance") if reloaded is not None else "-"), expected=str(args["importance"])))
        ok = all(v.passed for v in verifications)
        return Receipt(kind="project.create", executor="projects.store", ok=ok, request=request,
                       detail=f"created project {title!r} as {project.id}" + (f" with {len(tasks)} task(s)" if tasks else "") if ok else f"project {title!r} was not persisted correctly",
                       evidence={"project_id": project.id, "title": project.title, "goal": project.goal, "tasks": tasks, "parent_id": parent.id if parent else "",
                                 "importance": args.get("importance", ""), "deadline": args.get("deadline", ""), "path": str(path), "workspace": str(project.workspace)},
                       verifications=tuple(verifications))

    def rename(self, intent: ActionIntent, request: str) -> Receipt:
        project = self._find(intent.target)
        new_title = str(intent.arguments.get("title") or "").strip()
        if project is None:
            return failed("project.rename", "projects.store", f"no project named {intent.target!r}", request=request)
        if not new_title:
            return failed("project.rename", "projects.store", "no new title was given", request=request)
        old = project.title
        project.title = new_title
        project.metadata.setdefault("renamed", []).append({"from": old, "to": new_title, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.core.kernel.projects.save(project)
        self.core._last_project_id = project.id
        reloaded = self._store().try_load(project.id)
        verifications = [Verification("title persisted", reloaded is not None and reloaded.title == new_title, observed=repr(reloaded.title) if reloaded else "-", expected=repr(new_title))]
        return Receipt(kind="project.rename", executor="projects.store", ok=all(v.passed for v in verifications), request=request,
                       detail=f"renamed project {old!r} to {new_title!r}", evidence={"project_id": project.id, "title": new_title, "previous_title": old},
                       verifications=tuple(verifications))

    def add_tasks(self, intent: ActionIntent, request: str) -> Receipt:
        project = self._find(intent.target)
        tasks = [str(t).strip() for t in (intent.arguments.get("tasks") or []) if str(t).strip()]
        if project is None:
            return failed("project.add_tasks", "projects.store", f"no project named {intent.target!r}", request=request)
        if not tasks:
            return failed("project.add_tasks", "projects.store", "no tasks were named", request=request)
        before = len(project.tasks)
        for task in tasks:
            project.add_task(task)
        self.core.kernel.projects.save(project)
        self.core._last_project_id = project.id
        reloaded = self._store().try_load(project.id)
        titles = [t.title for t in reloaded.tasks][before:] if reloaded is not None else []
        verifications = [Verification(f"{len(tasks)} task(s) persist on the project", titles == tasks, observed=str(titles), expected=str(tasks))]
        return Receipt(kind="project.add_tasks", executor="projects.store", ok=all(v.passed for v in verifications), request=request,
                       detail=f"added {len(tasks)} task(s) to {project.title!r}", evidence={"project_id": project.id, "title": project.title, "tasks": tasks},
                       verifications=tuple(verifications))

    def archive(self, intent: ActionIntent, request: str) -> Receipt:
        project = self._find(intent.target)
        if project is None:
            return failed("project.archive", "projects.store", f"no project named {intent.target!r}", request=request)
        project.metadata["importance"] = "ARCHIVED"
        self.core.kernel.projects.save(project)
        reloaded = self._store().try_load(project.id)
        verifications = [Verification("importance is ARCHIVED", reloaded is not None and reloaded.metadata.get("importance") == "ARCHIVED",
                                      observed=str(reloaded.metadata.get("importance") if reloaded else "-"), expected="ARCHIVED")]
        return Receipt(kind="project.archive", executor="projects.store", ok=all(v.passed for v in verifications), request=request,
                       detail=f"archived project {project.title!r}", evidence={"project_id": project.id, "title": project.title}, verifications=tuple(verifications))

    def delete(self, intent: ActionIntent, request: str) -> Receipt:
        project = self._find(intent.target)
        if project is None:
            return failed("project.delete", "projects.store", f"no project named {intent.target!r}", request=request)
        title, pid = project.title, project.id
        self.core.kernel.projects.delete(pid, remove_workspace=False)
        store = self._store()
        verifications = [Verification("project no longer in the store", not store.exists(pid), observed="still present" if store.exists(pid) else "gone", expected="gone")]
        return Receipt(kind="project.delete", executor="projects.store", ok=all(v.passed for v in verifications), request=request,
                       detail=f"deleted project {title!r} ({pid}); its workspace was kept", evidence={"project_id": pid, "title": title}, verifications=tuple(verifications))


# --------------------------------------------------------------------------
# The owner-facing sentence
# --------------------------------------------------------------------------

def compose_concise(receipt: Receipt, *, language: str = "de") -> str:
    """One natural sentence from the receipt.  Evidence stays in Activity."""

    de = (language or "de").startswith("de")
    ev = receipt.evidence or {}
    title = str(ev.get("title") or "")
    if not receipt.ok:
        reason = receipt.detail
        if ev.get("duplicate"):
            return (f"Ein Projekt „{title}“ gibt es schon – ich habe kein zweites angelegt." if de
                    else f"A project “{title}” already exists – I did not create a second one.")
        return (f"Das konnte ich nicht ausführen: {reason}. Ich habe nichts verändert." if de
                else f"I could not do that: {reason}. Nothing was changed.")
    if not receipt.verified:
        failures = "; ".join(v.check for v in receipt.failures) or "Prüfung fehlgeschlagen"
        return (f"Ausgeführt, aber nicht bestätigt ({failures}). Ich werte das nicht als Erfolg." if de
                else f"Executed, but not confirmed ({failures}). I am not treating that as success.")
    if receipt.kind == "project.create":
        tasks = list(ev.get("tasks") or [])
        n = len(tasks)
        extra = ""
        if n:
            word = _NUMBERS_DE.get(n, str(n)) if de else _NUMBERS_EN.get(n, str(n))
            extra = (f" – mit {'einer Aufgabe' if n == 1 else word + ' Aufgaben'}" if de else f" – with {word} task{'s' if n != 1 else ''}")
        if ev.get("parent_id"):
            extra += (" unter dem übergeordneten Projekt" if de else " under its parent project")
        return (f"Erledigt. Projekt „{title}“ ist angelegt{extra}." if de else f"Done. Project “{title}” is created{extra}.")
    if receipt.kind == "project.rename":
        return (f"Erledigt. Das Projekt heißt jetzt „{title}“." if de else f"Done. The project is now called “{title}”.")
    if receipt.kind == "project.add_tasks":
        n = len(ev.get("tasks") or [])
        return (f"Erledigt. {n} Aufgabe{'n' if n != 1 else ''} zu „{title}“ hinzugefügt." if de else f"Done. Added {n} task{'s' if n != 1 else ''} to “{title}”.")
    if receipt.kind == "project.archive":
        return (f"Erledigt. „{title}“ ist archiviert." if de else f"Done. “{title}” is archived.")
    if receipt.kind == "project.delete":
        return (f"Erledigt. „{title}“ ist gelöscht; der Arbeitsordner bleibt erhalten." if de else f"Done. “{title}” is deleted; its workspace was kept.")
    return ("Erledigt." if de else "Done.") + f" {receipt.detail}"
