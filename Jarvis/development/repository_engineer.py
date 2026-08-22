from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SelfImprovementGoal:
    objective: str
    success_criteria: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=lambda: ["."])
    protected_paths: list[str] = field(default_factory=list)
    tests: list[list[str]] = field(default_factory=list)
    benchmark: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepositoryCommandResult:
    command: list[str]
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepositoryCandidateResult:
    status: str
    worktree: str
    diff: str = ""
    tests: list[RepositoryCommandResult] = field(default_factory=list)
    protected_pristine: bool = True
    rationale: str = ""
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    error: str = ""

    @property
    def success(self) -> bool:
        return self.status == "SELF_IMPROVEMENT_CANDIDATE_READY"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tests"] = [item.to_dict() for item in self.tests]
        data["success"] = self.success
        return data


class SelfImprovementMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, payload: dict[str, Any]) -> None:
        payload = {**payload, "saved_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RepositoryEngineer:
    """Autonomous repository candidate builder with worktree isolation and deterministic promotion evidence."""

    def __init__(
        self,
        *,
        brain: Any,
        worktree_root: str | Path,
        memory: SelfImprovementMemory | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.brain = brain
        self.worktree_root = Path(worktree_root).resolve()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.memory = memory or SelfImprovementMemory(self.worktree_root / "self_improvement_trajectories.jsonl")
        self.timeout_seconds = timeout_seconds

    def improve(
        self,
        repository_path: str | Path,
        goal: SelfImprovementGoal,
        acceptance_commands: list[list[str]] | None = None,
    ) -> RepositoryCandidateResult:
        source = Path(repository_path).resolve()
        worktree = self._create_worktree(source)
        trajectory: dict[str, Any] = {
            "goal": goal.to_dict(),
            "repository": str(source),
            "worktree": str(worktree),
            "events": [],
        }
        try:
            context = self._repository_context(worktree, goal)
            trajectory["events"].append({"stage": "understand", "payload": context})
            proposal = self._request_patch(goal, context)
            trajectory["events"].append({"stage": "proposal", "payload": _redact_large(proposal)})
            self._apply_proposal(worktree, source, goal, proposal)
            tests = self._run_acceptance(worktree, acceptance_commands or goal.tests)
            self._git(worktree, ["add", "-N", "."], check=False)
            diff = self._git(worktree, ["diff"], check=False).stdout
            protected_pristine = self._protected_pristine(source, worktree, goal.protected_paths)
            all_tests_pass = all(item.success for item in tests) if tests else True
            status = "SELF_IMPROVEMENT_CANDIDATE_READY" if all_tests_pass and protected_pristine and diff.strip() else "SELF_IMPROVEMENT_CANDIDATE_REJECTED"
            result = RepositoryCandidateResult(
                status=status,
                worktree=str(worktree),
                diff=diff,
                tests=tests,
                protected_pristine=protected_pristine,
                rationale=str(proposal.get("analysis", "")),
                error="" if status == "SELF_IMPROVEMENT_CANDIDATE_READY" else "candidate lacked deterministic acceptance evidence",
            )
            trajectory["events"].append({"stage": "evaluate", "payload": result.to_dict()})
            self.memory.record({"trajectory_id": result.trajectory_id, **trajectory, "outcome": result.to_dict()})
            return result
        except Exception as exc:
            result = RepositoryCandidateResult(
                status="SELF_IMPROVEMENT_CANDIDATE_REJECTED",
                worktree=str(worktree),
                protected_pristine=False,
                error=str(exc),
            )
            trajectory["events"].append({"stage": "failed", "payload": result.to_dict()})
            self.memory.record({"trajectory_id": result.trajectory_id, **trajectory, "outcome": result.to_dict()})
            return result

    def project_tree(self, repository_path: str | Path, *, limit: int = 120) -> list[str]:
        root = Path(repository_path).resolve()
        files = []
        for path in sorted(root.rglob("*")):
            if len(files) >= limit:
                break
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                files.append(path.relative_to(root).as_posix())
        return files

    def read_file(self, repository_path: str | Path, relative_path: str, *, max_chars: int = 6000) -> str:
        path = self._safe_path(Path(repository_path).resolve(), Path(repository_path).resolve(), relative_path, SelfImprovementGoal(""))
        return path.read_text(encoding="utf-8")[:max_chars]

    def search_text(self, repository_path: str | Path, query: str, *, limit: int = 20) -> list[str]:
        root = Path(repository_path).resolve()
        matches = []
        for relative in self.project_tree(root):
            path = root / relative
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if query in text:
                matches.append(relative)
            if len(matches) >= limit:
                break
        return matches

    def _create_worktree(self, repository_path: Path) -> Path:
        worktree = self.worktree_root / f"candidate_{uuid.uuid4().hex[:10]}"
        worktree.parent.mkdir(parents=True, exist_ok=True)
        if (repository_path / ".git").exists() and self._git_root(repository_path) == repository_path.resolve():
            completed = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                cwd=repository_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode == 0:
                return worktree.resolve()
        shutil.copytree(repository_path, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        self._run_git_raw(worktree, ["init"], check=True)
        self._run_git_raw(worktree, ["config", "user.email", "jarvis@example.invalid"], check=True)
        self._run_git_raw(worktree, ["config", "user.name", "Jarvis RepositoryEngineer"], check=True)
        self._run_git_raw(worktree, ["add", "."], check=True)
        self._run_git_raw(worktree, ["commit", "-m", "baseline"], check=True)
        return worktree.resolve()

    def _repository_context(self, worktree: Path, goal: SelfImprovementGoal) -> dict[str, Any]:
        files = self.project_tree(worktree)
        snippets = {}
        for relative in files[:20]:
            if relative.endswith((".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml")):
                try:
                    snippets[relative] = (worktree / relative).read_text(encoding="utf-8")[:4000]
                except UnicodeDecodeError:
                    continue
        return {
            "tree": files,
            "snippets": snippets,
            "tests": goal.tests,
            "allowed_paths": goal.allowed_paths,
            "protected_paths": goal.protected_paths,
        }

    def _request_patch(self, goal: SelfImprovementGoal, context: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Return JSON only. Role: RepositoryEngineer Implementer. Produce a repository improvement candidate.\n"
            "Use complete file replacements. Do not alter protected paths. Final promotion requires deterministic tests.\n"
            "Schema: {\"analysis\":\"...\",\"files\":[{\"path\":\"...\",\"content\":\"...\"}],\"new_files\":[],\"deleted_files\":[]}\n"
            f"SelfImprovementGoal:\n{json.dumps(goal.to_dict(), indent=2, sort_keys=True)}\n"
            f"Repository context:\n{json.dumps(context, indent=2, sort_keys=True)}"
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analysis": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
                "new_files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
                "deleted_files": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["analysis", "files"],
        }
        if hasattr(self.brain, "generate_structured"):
            raw = self.brain.generate_structured(prompt, schema, max_tokens=4000, temperature=0.2, top_p=0.9)
        elif hasattr(self.brain, "generate_coding"):
            raw = self.brain.generate_coding(prompt, max_tokens=4000, temperature=0.2, top_p=0.9)
        else:
            raw = self.brain.generate(prompt, max_tokens=4000, temperature=0.2, top_p=0.9)
        data = json.loads(_extract_json(raw))
        if not isinstance(data.get("files"), list):
            raise ValueError("repository proposal did not contain files")
        return data

    def _apply_proposal(self, worktree: Path, source: Path, goal: SelfImprovementGoal, proposal: dict[str, Any]) -> None:
        for item in [*proposal.get("files", []), *proposal.get("new_files", [])]:
            if not isinstance(item, dict):
                continue
            path = self._safe_path(worktree, source, str(item.get("path", "")), goal)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(item.get("content", "")), encoding="utf-8")
        for relative in proposal.get("deleted_files", []):
            path = self._safe_path(worktree, source, str(relative), goal)
            if path.exists():
                path.unlink()

    def _run_acceptance(self, worktree: Path, commands: list[list[str]]) -> list[RepositoryCommandResult]:
        results = []
        for command in commands:
            completed = subprocess.run(command, cwd=worktree, capture_output=True, text=True, timeout=self.timeout_seconds, shell=False)
            results.append(
                RepositoryCommandResult(
                    command=list(command),
                    success=completed.returncode == 0,
                    stdout=completed.stdout[-4000:],
                    stderr=completed.stderr[-4000:],
                    return_code=completed.returncode,
                )
            )
        return results

    def _safe_path(self, worktree: Path, source: Path, relative_path: str, goal: SelfImprovementGoal) -> Path:
        raw = Path(relative_path)
        normalized = raw.as_posix()
        if not normalized or raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"unsafe repository path: {relative_path}")
        if _path_matches(normalized, goal.protected_paths):
            raise ValueError(f"protected repository path: {relative_path}")
        if goal.allowed_paths and not _path_matches(normalized, goal.allowed_paths):
            raise ValueError(f"path not allowed for this goal: {relative_path}")
        candidate = (worktree / raw).resolve(strict=False)
        root = worktree.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"path escapes worktree: {relative_path}")
        live_candidate = (source / raw).resolve(strict=False)
        if live_candidate.exists() and live_candidate.is_symlink():
            raise ValueError(f"refuse to edit symlinked live path: {relative_path}")
        return candidate

    def _protected_pristine(self, source: Path, worktree: Path, protected_paths: list[str]) -> bool:
        for relative in protected_paths:
            source_path = (source / relative).resolve(strict=False)
            candidate_path = (worktree / relative).resolve(strict=False)
            if source_path.exists() != candidate_path.exists():
                return False
            if source_path.is_file() and candidate_path.is_file() and source_path.read_bytes() != candidate_path.read_bytes():
                return False
        return True

    def _git(self, worktree: Path, args: list[str], *, check: bool) -> subprocess.CompletedProcess:
        return self._run_git_raw(worktree, args, check=check)

    def _run_git_raw(self, cwd: Path, args: list[str], *, check: bool) -> subprocess.CompletedProcess:
        import os

        env = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(cwd.resolve().parent)
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=self.timeout_seconds, check=check, env=env)

    def _git_root(self, cwd: Path) -> Path | None:
        completed = self._run_git_raw(cwd, ["rev-parse", "--show-toplevel"], check=False)
        if completed.returncode != 0:
            return None
        return Path(completed.stdout.strip()).resolve()


def _path_matches(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    normalized = path.strip("/")
    for pattern in patterns:
        clean = pattern.strip("/")
        if clean in {"", "."}:
            return True
        if normalized == clean or normalized.startswith(clean + "/"):
            return True
    return False


def _extract_json(text: str) -> str:
    import re

    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


def _redact_large(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    for key in ["files", "new_files"]:
        redacted[key] = [
            {"path": item.get("path"), "content_hash": str(hash(item.get("content", "")))}
            for item in payload.get(key, [])
            if isinstance(item, dict)
        ]
    return redacted
