"""From the catalog to the narrow context an engineer needs.

    EngineeringSpec (request, impacted files)
        -> catalog
        -> impact analysis (direct dependents, tests)
        -> module manifests + contracts + dependencies + tests, within a budget

The engineer reads this first and asks for source only where it will edit.
Tens of thousands of tokens, not hundreds of thousands: the budget is a
character count and the sections are ordered by how much they matter, so what
gets cut is the least relevant, never the module being changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from catalog.build import CATALOG_DIR, REPO, build_catalog


@dataclass
class EngineeringContext:
    files: list[str]
    dependents: dict[str, list[str]]
    tests: list[str]
    text: str
    chars: int
    sections: list[str] = field(default_factory=list)
    #: catalog_tokens (map, dependents, dependencies, overview), interface_tokens
    #: (contracts of the impacted modules), test_tokens (the tests to run).
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def dependents_count(self) -> int:
        return sum(len(v) for v in self.dependents.values())

    def to_dict(self) -> dict[str, Any]:
        return {"files": self.files, "dependents": self.dependents, "tests": self.tests, "chars": self.chars,
                "sections": self.sections, "tokens": dict(self.tokens)}


def load_catalog(repo: Path | None = None, *, out_dir: Path | None = None, rebuild_if_missing: bool = True) -> dict[str, Any]:
    repo = Path(repo or REPO).resolve()
    out_dir = Path(out_dir or (repo / "catalog" if repo != REPO else CATALOG_DIR))
    try:
        map_doc = yaml.safe_load((out_dir / "ZEUS_MAP.yaml").read_text(encoding="utf-8"))
        interfaces = yaml.safe_load((out_dir / "INTERFACES.yaml").read_text(encoding="utf-8"))
        graph = json.loads((out_dir / "DEPENDENCY_GRAPH.json").read_text(encoding="utf-8"))
        if all(isinstance(doc, dict) for doc in (map_doc, interfaces, graph)):
            return {"map": map_doc, "interfaces": interfaces, "graph": graph}
    except (OSError, ValueError):
        pass
    if rebuild_if_missing:
        return build_catalog(repo)
    raise FileNotFoundError(f"no catalog under {out_dir}")


def _module_row(catalog: dict[str, Any], path: str) -> dict[str, Any] | None:
    for pkg in catalog["map"].get("packages", {}).values():
        for module in pkg.get("modules", []):
            if module["path"] == path:
                return module
    return None


def _contract_text(catalog: dict[str, Any], path: str, *, max_methods: int = 40) -> str:
    entry = catalog["interfaces"].get("modules", {}).get(path)
    if not entry:
        return ""
    lines = [f"### {path}"]
    for cls in entry.get("classes", []):
        bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
        lines.append(f"class {cls['name']}{bases}" + (f"  # {cls['doc']}" if cls.get("doc") else ""))
        for method in cls.get("methods", [])[:max_methods]:
            lines.append(f"    {method}")
        if len(cls.get("methods", [])) > max_methods:
            lines.append(f"    ... {len(cls['methods']) - max_methods} more methods")
    for fn in entry.get("functions", []):
        lines.append(fn["signature"] + (f"  # {fn['doc']}" if fn.get("doc") else ""))
    return "\n".join(lines)


def build_context(files: list[str], *, request: str = "", repo: Path | None = None, catalog: dict[str, Any] | None = None,
                  budget_chars: int = 60_000, out_dir: Path | None = None) -> EngineeringContext:
    """The narrow context for a change touching ``files``.

    Order of inclusion (what survives a tight budget first): the impacted
    modules' manifests and contracts, their direct dependents (what the change
    can break), the tests that belong to them, then the direct dependencies'
    contracts, then a one-line map of the packages.
    """

    repo = Path(repo or REPO).resolve()
    catalog = catalog or load_catalog(repo, out_dir=out_dir)
    graph = catalog["graph"]
    files = [str(f).replace("\\", "/") for f in files if str(f).strip()]
    known = [f for f in files if f in set(graph.get("nodes", []))]

    dependents = {f: list(graph.get("dependents", {}).get(f, [])) for f in known}
    tests: list[str] = []
    for f in known:
        row = _module_row(catalog, f)
        for test in (row or {}).get("tests", []):
            if test not in tests:
                tests.append(test)
    for f in known:
        for dep in dependents[f]:
            row = _module_row(catalog, dep)
            for test in (row or {}).get("tests", []):
                if test not in tests and len(tests) < 12:
                    tests.append(test)

    sections: list[tuple[str, str]] = []
    head = ["## Impacted modules"]
    for f in known:
        row = _module_row(catalog, f) or {}
        head.append(f"- `{f}` — {row.get('purpose', '')} ({row.get('lines', '?')} lines, {row.get('public', 0)} public names); "
                    f"depended on by {len(dependents[f])} module(s)")
    unknown = [f for f in files if f not in known]
    for f in unknown:
        head.append(f"- `{f}` — not a catalogued Python module (a UI file, a test, or new)")
    sections.append(("impacted", "\n".join(head)))

    contracts = ["## Contracts of the impacted modules (keep these working)"]
    for f in known:
        text = _contract_text(catalog, f)
        if text:
            contracts.append(text)
    sections.append(("contracts", "\n".join(contracts)))

    if any(dependents.values()):
        dep_lines = ["## Direct dependents (what this change can break)"]
        for f in known:
            for dep in dependents[f][:25]:
                row = _module_row(catalog, dep) or {}
                dep_lines.append(f"- `{dep}` imports `{f}` — {row.get('purpose', '')}")
        sections.append(("dependents", "\n".join(dep_lines)))

    if tests:
        sections.append(("tests", "## Tests that belong to this change (run these)\n" + "\n".join(f"- `{t}`" for t in tests)))

    deps_text = ["## Contracts of direct dependencies (called by the impacted modules)"]
    seen: set[str] = set()
    for f in known:
        for dep in graph.get("imports", {}).get(f, [])[:12]:
            if dep in seen or dep in known:
                continue
            seen.add(dep)
            text = _contract_text(catalog, dep, max_methods=15)
            if text:
                deps_text.append(text)
    if len(deps_text) > 1:
        sections.append(("dependencies", "\n".join(deps_text)))

    packages = catalog["map"].get("packages", {})
    overview = ["## The system in one screen"] + [f"- **{name}**: {pkg.get('purpose', '')}" for name, pkg in packages.items() if pkg.get("purpose")]
    entry = catalog["map"].get("entry_points", {})
    if entry:
        overview.append("Entry points: " + ", ".join(f"{k}=`{v}`" for k, v in entry.items()))
    sections.append(("overview", "\n".join(overview)))

    kept: list[str] = []
    names: list[str] = []
    used = 0
    tokens = {"catalog_tokens": 0, "interface_tokens": 0, "test_tokens": 0}
    from gateway.estimate import estimate_tokens

    for name, text in sections:
        if used + len(text) + 2 > budget_chars:
            if name in {"impacted", "contracts"}:
                text = text[: max(0, budget_chars - used - 40)] + "\n... [truncated to budget]"
            else:
                continue
        kept.append(text)
        names.append(name)
        used += len(text) + 2
        kind = "interface_tokens" if name in {"contracts", "dependencies"} else ("test_tokens" if name == "tests" else "catalog_tokens")
        tokens[kind] += estimate_tokens(text)
    body = "\n\n".join(kept)
    if request:
        body = f"# Engineering context for: {request.strip()[:300]}\n\n{body}"
    return EngineeringContext(files=files, dependents=dependents, tests=tests, text=body, chars=len(body), sections=names, tokens=tokens)
