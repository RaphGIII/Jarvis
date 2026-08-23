"""Durable storage for projects.

Projects have to survive a crashed process, a closed terminal and a rebooted
machine, so the storage rules are deliberately dull:

* One JSON document per project, written via a temp file and an atomic replace,
  so a process killed mid-save leaves the previous good version intact rather
  than a half-written one.
* No database, no daemon, no schema migration step.  A project is inspectable
  and repairable with a text editor, which matters a great deal when the thing
  writing them is an autonomous agent.
* Unknown fields are dropped on load rather than raising, so a project written
  by an older version of Jarvis still opens after an upgrade.

JSON was chosen over SQLite precisely because the failure mode of a corrupt file
should be "one project needs a look", not "the store will not open".
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from projects.models import Project, ProjectState


class ProjectStore:
    """A directory of project documents."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def path_for(self, project_id: str) -> Path:
        return self.root / f"{project_id}.json"

    def workspace_for(self, project: Project) -> Path:
        """The project's own directory, created on demand.

        Kept beside the document and under the store root so that deleting a
        project is one recursive delete, and so nothing a project builds can end
        up scattered across the machine.
        """

        if project.workspace:
            workspace = Path(project.workspace)
        else:
            workspace = self.root / "workspaces" / project.id
            project.workspace = str(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    # -- reading ---------------------------------------------------------

    def exists(self, project_id: str) -> bool:
        return self.path_for(project_id).exists()

    def load(self, project_id: str) -> Project:
        path = self.path_for(project_id)
        if not path.exists():
            raise KeyError(f"no such project: {project_id}")
        return Project.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def try_load(self, project_id: str) -> Project | None:
        try:
            return self.load(project_id)
        except (KeyError, json.JSONDecodeError, OSError):
            return None

    def iter_projects(self) -> Iterator[Project]:
        for path in sorted(self.root.glob("*.json")):
            try:
                yield Project.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError, TypeError):
                # A single unreadable document must not hide every other project.
                continue

    def list_projects(self, *, state: ProjectState | None = None, limit: int | None = None) -> list[Project]:
        projects = [item for item in self.iter_projects() if state is None or item.state is state]
        projects.sort(key=lambda item: item.updated_at, reverse=True)
        return projects[:limit] if limit else projects

    def active(self) -> list[Project]:
        return [item for item in self.list_projects() if not item.state.terminal]

    def find(self, query: str, *, limit: int = 5) -> list[Project]:
        """Rank projects by term overlap with the goal, title and requirements.

        Deliberately keyword-based: resolving "keep working on the chess thing"
        must not require an embedding model to be loaded, and must work offline.
        """

        terms = _terms(query)
        if not terms:
            return []
        scored: list[tuple[float, Project]] = []
        for project in self.iter_projects():
            haystack = " ".join(
                [project.goal, project.title, *[item.text for item in project.requirements], project.kind]
            )
            overlap = len(terms & _terms(haystack))
            if not overlap:
                continue
            score = overlap / len(terms)
            if not project.state.terminal:
                score += 0.25  # prefer work that is still live
            scored.append((score, project))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [project for _, project in scored[:limit]]

    # -- writing ---------------------------------------------------------

    def save(self, project: Project) -> Path:
        project.updated_at = datetime.now(timezone.utc).isoformat()
        path = self.path_for(project.id)
        payload = json.dumps(project.to_dict(), indent=2, sort_keys=True, default=str)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)  # atomic on both POSIX and Windows
        return path

    def delete(self, project_id: str, *, remove_workspace: bool = False) -> None:
        project = self.try_load(project_id)
        self.path_for(project_id).unlink(missing_ok=True)
        if remove_workspace and project and project.workspace:
            workspace = Path(project.workspace)
            # Only ever recurse into a workspace this store owns.
            if workspace.is_relative_to(self.root) and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    def summary(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        for project in self.iter_projects():
            by_state[project.state.value] = by_state.get(project.state.value, 0) + 1
        return {"root": str(self.root), "total": sum(by_state.values()), "by_state": by_state}


def _terms(text: str) -> set[str]:
    import re

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "you", "was", "are",
        "der", "die", "das", "und", "ein", "eine", "mit", "fuer", "für", "den", "dem", "des",
        "make", "build", "create", "please", "want", "need", "should", "would", "can",
    }
    words = {word for word in re.split(r"[^a-z0-9]+", text.lower()) if len(word) > 2}
    return words - stopwords
