"""Derive the catalog from the code.  Deterministic; no model involved.

    python -m catalog.build            # rewrite catalog/*.yaml|json|md
    python -m catalog.build --stdout   # print the map

Sources, all read from the repository itself:

* every ``*.py`` under the product packages (not tests, data, build
  artefacts or virtualenvs): first docstring line, public classes with their
  public methods and signatures, public functions with signatures, imports
  of other product modules, line count
* ``tests/test_*.py``: which modules each test file imports or names
* ``service/http.py``: the JSON API routes
* ``ui/**/*.js`` and ``ui/index.html``: the page's module graph
* ``data/jarvis/capabilities/registry.json``: the capabilities ZEUS has
  (ids, family, health, permissions) -- runtime state, included as facts
  about what exists, never edited here

The output is stable across runs (sorted keys, no timestamps), so a diff of
the catalog is a diff of the architecture.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO = Path(__file__).resolve().parent.parent
CATALOG_DIR = REPO / "catalog"
CATALOG_FILES = ("ZEUS_MAP.yaml", "INTERFACES.yaml", "DEPENDENCY_GRAPH.json", "ARCHITECTURE.md")
#: Runtime facts (which capabilities exist, their health) are written beside the
#: catalog for engineers to read, but are not part of the staleness check: they
#: change while ZEUS runs, not when its code does.
RUNTIME_FILES = ("CAPABILITIES.yaml",)

#: Directories that are not product code.
EXCLUDED_DIRS = frozenset({"tests", "data", "build", "dist", "__pycache__", "catalog", "pytest-current", "pytest-temp",
                           "pytest-v03-final", "node_modules"})

#: What each package owns, in one line.  Kept here rather than in every
#: ``__init__`` because several packages predate the habit of writing one;
#: a package with an ``__init__`` docstring uses that instead.
PACKAGE_PURPOSES: dict[str, str] = {
    "agent": "legacy agent loop",
    "brain": "model tiers, providers and the local Ollama client",
    "capabilities": "the capability registry, resolver, acquisition and health",
    "core": "composition root (kernel) and identity",
    "deployment": "promotion of a verified candidate into the live tree, and rollback",
    "developer": "developer tooling",
    "development": "the autonomous software engineer (local coder), code index, QA",
    "devices": "paired devices and their gateway",
    "environments": "sandboxes for generated code",
    "evaluator": "evaluation of generated skills",
    "experts": "engineers behind the expert gateway: Codex CLI, Claude Code CLI, metered API engineers",
    "gateway": "the model gateway: roles, chat modes, budget, privacy, routing, providers, persona",
    "imagegen": "local image generation",
    "jarvis": "entry points: serve, CLI, window, verify_ui, measurements",
    "knowledge": "the knowledge graph and library",
    "learning": "adaptive rules and feedback",
    "memory": "conversation memory",
    "owner": "the owner core: protected documents, security gate, protected paths",
    "persona": "persona profiles and language",
    "projects": "the project engine and store",
    "research": "web research tools",
    "runtime": "runtime services: activity, cost policy, preferences, secrets, receipts",
    "sandbox": "sandboxing",
    "senses": "perception",
    "service": "the product service: core, HTTP API, intents, routing, self-development, isolation",
    "skills": "skill packaging",
    "speech": "speech recognition, synthesis, wake word",
    "tools": "tool registry and builtin tools",
    "training": "training data and wake-word training",
    "voice": "voice pipeline",
    "zeus_supervisor": "the supervisor process outside the mutable code",
}


# ---------------------------------------------------------------------------
# Python modules
# ---------------------------------------------------------------------------

@dataclass
class ModuleInfo:
    path: str
    module: str
    purpose: str = ""
    lines: int = 0
    classes: list[dict[str, Any]] = field(default_factory=list)
    functions: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    syntax_error: str = ""

    @property
    def public_count(self) -> int:
        return len(self.classes) + len(self.functions)


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        text = line.strip()
        if text:
            return text[:160]
    return ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:  # noqa: BLE001
        args = "..."
    returns = ""
    if node.returns is not None:
        try:
            returns = " -> " + ast.unparse(node.returns)
        except Exception:  # noqa: BLE001
            returns = ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args}){returns}"


def _product_modules(repo: Path) -> set[str]:
    names: set[str] = set()
    for entry in repo.iterdir():
        if entry.is_dir() and entry.name not in EXCLUDED_DIRS and not entry.name.startswith(".") and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
    return names


def _module_name(repo: Path, path: Path) -> str:
    rel = path.relative_to(repo).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_python(repo: Path) -> Iterable[Path]:
    for entry in sorted(repo.iterdir()):
        if entry.is_file() and entry.suffix == ".py":
            yield entry
        elif entry.is_dir() and entry.name not in EXCLUDED_DIRS and not entry.name.startswith("."):
            for path in sorted(entry.rglob("*.py")):
                if any(part in EXCLUDED_DIRS or part.startswith(".") for part in path.relative_to(repo).parts[:-1]):
                    continue
                yield path


def analyse_module(repo: Path, path: Path, product: set[str]) -> ModuleInfo:
    info = ModuleInfo(path=path.relative_to(repo).as_posix(), module=_module_name(repo, path))
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        info.syntax_error = str(exc)
        return info
    info.lines = source.count("\n") + 1
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        info.syntax_error = f"line {exc.lineno}: {exc.msg}"
        return info
    info.purpose = _first_line(ast.get_docstring(tree))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    methods.append(_signature(item))
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:  # noqa: BLE001
                    pass
            info.classes.append({"name": node.name, "bases": bases, "doc": _first_line(ast.get_docstring(node)), "methods": methods})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            info.functions.append({"signature": _signature(node), "doc": _first_line(ast.get_docstring(node))})
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in product:
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if top in product:
                imports.add(node.module)
    info.imports = sorted(imports)
    return info


def _tests_by_module(repo: Path, modules: dict[str, ModuleInfo]) -> None:
    tests_dir = repo / "tests"
    if not tests_dir.is_dir():
        return
    by_name = {info.module: info for info in modules.values()}
    for test in sorted(tests_dir.glob("test_*.py")):
        try:
            text = test.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = test.relative_to(repo).as_posix()
        for name, info in by_name.items():
            if not name:
                continue
            if re.search(rf"(?m)^\s*(?:from\s+{re.escape(name)}\b|import\s+{re.escape(name)}\b)", text):
                info.tests.append(rel)


# ---------------------------------------------------------------------------
# Routes, UI, capabilities
# ---------------------------------------------------------------------------

_ROUTE = re.compile(r'^\s*"(/api/[^"]+)"\s*:', re.M)


def api_routes(repo: Path) -> list[str]:
    path = repo / "service" / "http.py"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return sorted(set(_ROUTE.findall(text)))


def ui_modules(repo: Path) -> list[dict[str, Any]]:
    ui = repo / "ui"
    out: list[dict[str, Any]] = []
    if not ui.is_dir():
        return out
    for path in sorted(ui.rglob("*.js")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        purpose = ""
        match = re.match(r"\s*/\*\s*(.+?)(?:\n|\*/)", text)
        if match:
            purpose = match.group(1).strip()[:160]
        imports = sorted(set(re.findall(r"""from\s+["']\.{1,2}/([^"']+)["']""", text)))
        out.append({"path": path.relative_to(repo).as_posix(), "purpose": purpose, "lines": text.count("\n") + 1, "imports": imports})
    return out


def capabilities(repo: Path) -> list[dict[str, Any]]:
    path = repo / "data" / "jarvis" / "capabilities" / "registry.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("capabilities", data) if isinstance(data, dict) else data
    rows = list(items.values()) if isinstance(items, dict) else list(items)
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            "id": str(row.get("capability_id", "")), "family": str(row.get("family", "")),
            "health": str(row.get("health_state") or row.get("health", "")),
            "permissions": list(row.get("permissions_required") or []),
            "goal_types": list(row.get("goal_types") or [])[:6],
            "implementation": str(row.get("implementation_path", "")),
        })
    return sorted(out, key=lambda r: r["id"])


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

def build_catalog(repo: Path | None = None) -> dict[str, Any]:
    """Everything, as plain data.  ``write_catalog`` renders it to files."""

    repo = Path(repo or REPO).resolve()
    product = _product_modules(repo)
    modules: dict[str, ModuleInfo] = {}
    for path in _iter_python(repo):
        info = analyse_module(repo, path, product)
        modules[info.path] = info
    _tests_by_module(repo, modules)

    packages: dict[str, dict[str, Any]] = {}
    for info in modules.values():
        top = info.path.split("/", 1)[0] if "/" in info.path else "(root)"
        pkg = packages.setdefault(top, {"purpose": "", "modules": []})
        if info.path.endswith("__init__.py") and info.purpose:
            pkg["purpose"] = info.purpose
        pkg["modules"].append({
            "path": info.path, "purpose": info.purpose, "lines": info.lines, "public": info.public_count,
            "tests": info.tests, **({"syntax_error": info.syntax_error} if info.syntax_error else {}),
        })
    for name, pkg in packages.items():
        if not pkg["purpose"]:
            pkg["purpose"] = PACKAGE_PURPOSES.get(name, "")
        pkg["modules"].sort(key=lambda m: m["path"])

    interfaces: dict[str, Any] = {}
    for info in sorted(modules.values(), key=lambda m: m.path):
        if not info.classes and not info.functions:
            continue
        interfaces[info.path] = {
            "module": info.module,
            "classes": [{"name": c["name"], **({"bases": c["bases"]} if c["bases"] else {}), **({"doc": c["doc"]} if c["doc"] else {}),
                         "methods": c["methods"]} for c in info.classes],
            "functions": [{"signature": f["signature"], **({"doc": f["doc"]} if f["doc"] else {})} for f in info.functions],
        }

    by_module = {info.module: info.path for info in modules.values() if info.module}
    edges: dict[str, list[str]] = {}
    for info in modules.values():
        targets = []
        for imported in info.imports:
            target = by_module.get(imported)
            if target is None:
                # `from service import core` style: the package init, or a
                # submodule named later in the statement we cannot see here.
                target = by_module.get(imported + ".__init__") or by_module.get(imported)
            if target is None:
                candidate = imported.replace(".", "/") + ".py"
                target = candidate if candidate in modules else (imported.replace(".", "/") + "/__init__.py")
            if target in modules and target != info.path:
                targets.append(target)
        edges[info.path] = sorted(set(targets))
    dependents: dict[str, list[str]] = {path: [] for path in modules}
    for source, targets in edges.items():
        for target in targets:
            dependents.setdefault(target, []).append(source)
    for key in dependents:
        dependents[key].sort()

    routes = api_routes(repo)
    ui = ui_modules(repo)
    caps = capabilities(repo)
    entry_points = {
        "serve": "jarvis/serve.py" if (repo / "jarvis" / "serve.py").exists() else "",
        "cli": "jarvis/cli.py" if (repo / "jarvis" / "cli.py").exists() else "",
        "http_api": "service/http.py", "core": "service/core.py", "kernel": "core/kernel.py",
        "supervisor": "zeus_supervisor" if (repo / "zeus_supervisor").is_dir() else "",
        "ui": "ui/index.html", "tests": "python -m pytest tests/ -q",
        "ui_gate": "python -m jarvis.verify_ui", "catalog": "python -m catalog.build / python -m catalog.check",
    }

    map_doc = {
        "schema_version": 1,
        "packages": dict(sorted(packages.items())),
        "entry_points": {k: v for k, v in entry_points.items() if v},
        "api_routes": routes,
        "ui_modules": ui,
        "totals": {"modules": len(modules), "packages": len(packages), "api_routes": len(routes), "ui_modules": len(ui),
                   "tests": len(list((repo / "tests").glob("test_*.py"))) if (repo / "tests").is_dir() else 0},
    }
    graph_doc = {"schema_version": 1, "nodes": sorted(modules), "imports": edges, "dependents": dependents}
    return {"map": map_doc, "interfaces": {"schema_version": 1, "modules": interfaces}, "graph": graph_doc,
            "capabilities": {"schema_version": 1, "note": "runtime facts from the capability registry; not part of the staleness check",
                             "capabilities": caps}}


def render_architecture(catalog: dict[str, Any]) -> str:
    """A compact human overview: the first thing an engineer reads."""

    map_doc = catalog["map"]
    graph = catalog["graph"]
    lines = ["# ZEUS Architecture (generated by `python -m catalog.build`; do not edit)", ""]
    totals = map_doc["totals"]
    lines.append(f"{totals['modules']} modules in {totals['packages']} packages, {totals['api_routes']} JSON API routes, "
                 f"{totals['ui_modules']} UI modules, {totals['tests']} test files; capabilities in CAPABILITIES.yaml.")
    lines.append("")
    lines.append("## Entry points")
    for key, value in map_doc["entry_points"].items():
        lines.append(f"- **{key}**: `{value}`")
    lines.append("")
    lines.append("## Packages")
    for name, pkg in map_doc["packages"].items():
        count = len(pkg["modules"])
        total_lines = sum(m["lines"] for m in pkg["modules"])
        lines.append(f"- **{name}** — {pkg['purpose'] or '(no purpose recorded)'} ({count} modules, {total_lines} lines)")
    lines.append("")
    lines.append("## Most depended-on modules")
    ranked = sorted(graph["dependents"].items(), key=lambda kv: -len(kv[1]))[:15]
    for path, deps in ranked:
        if deps:
            lines.append(f"- `{path}` ← {len(deps)} modules")
    lines.append("")
    lines.append("## How to work with this")
    lines.append("1. Read `catalog/ZEUS_MAP.yaml` for the module you need; `catalog/INTERFACES.yaml` for its contracts.")
    lines.append("2. `catalog/DEPENDENCY_GRAPH.json` `dependents[path]` lists what a change can break; run those modules' tests.")
    lines.append("3. Run the module's tests (`tests` per module in the map), then `python -m catalog.build` before committing.")
    lines.append("")
    return "\n".join(lines)


def render_files(catalog: dict[str, Any]) -> dict[str, str]:
    dump = dict(sort_keys=False, allow_unicode=True, width=120)
    return {
        "ZEUS_MAP.yaml": yaml.safe_dump(catalog["map"], **dump),
        "INTERFACES.yaml": yaml.safe_dump(catalog["interfaces"], **dump),
        "DEPENDENCY_GRAPH.json": json.dumps(catalog["graph"], indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        "ARCHITECTURE.md": render_architecture(catalog),
        "CAPABILITIES.yaml": yaml.safe_dump(catalog["capabilities"], **dump),
    }


def write_catalog(repo: Path | None = None, *, out_dir: Path | None = None) -> list[Path]:
    repo = Path(repo or REPO).resolve()
    out_dir = Path(out_dir or (repo / "catalog"))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_files(build_catalog(repo)).items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


def fingerprint(catalog: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--stdout" in args:
        sys.stdout.write(render_files(build_catalog())["ARCHITECTURE.md"])
        return 0
    for path in write_catalog():
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
