"""Is the committed catalog what the code says?  Exit 1 when it is not.

    python -m catalog.check

The check rebuilds the catalog in memory and compares it with the files on
disk, then says *what* is stale: modules that appeared or vanished, contracts
whose signatures changed, tests that moved, routes added.  A stale catalog is
a verification failure -- the self-development verification regenerates the
catalog inside the candidate so an engineer's change never leaves it stale,
and the test suite holds the human to the same standard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from catalog.build import CATALOG_FILES, REPO, build_catalog


def _load_existing(out_dir: Path) -> dict[str, Any] | None:
    try:
        map_doc = yaml.safe_load((out_dir / "ZEUS_MAP.yaml").read_text(encoding="utf-8"))
        interfaces = yaml.safe_load((out_dir / "INTERFACES.yaml").read_text(encoding="utf-8"))
        graph = json.loads((out_dir / "DEPENDENCY_GRAPH.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not all(isinstance(doc, dict) for doc in (map_doc, interfaces, graph)):
        return None
    return {"map": map_doc, "interfaces": interfaces, "graph": graph}


def _module_paths(catalog: dict[str, Any]) -> set[str]:
    return set(catalog["graph"].get("nodes", []))


def diff_catalogs(existing: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Human-readable stale items, empty when the catalog is current."""

    problems: list[str] = []
    old_nodes, new_nodes = _module_paths(existing), _module_paths(fresh)
    for path in sorted(new_nodes - old_nodes):
        problems.append(f"module not in catalog: {path}")
    for path in sorted(old_nodes - new_nodes):
        problems.append(f"catalog lists a module that no longer exists: {path}")

    old_if = existing["interfaces"].get("modules", {})
    new_if = fresh["interfaces"].get("modules", {})
    for path in sorted(set(old_if) | set(new_if)):
        if path not in old_nodes & new_nodes:
            continue
        before, after = old_if.get(path), new_if.get(path)
        if before != after:
            problems.append(f"interface changed: {path}")

    old_imports = existing["graph"].get("imports", {})
    new_imports = fresh["graph"].get("imports", {})
    for path in sorted(old_nodes & new_nodes):
        if old_imports.get(path, []) != new_imports.get(path, []):
            problems.append(f"dependencies changed: {path}")

    old_tests = {m["path"]: m.get("tests", []) for pkg in existing["map"].get("packages", {}).values() for m in pkg.get("modules", [])}
    new_tests = {m["path"]: m.get("tests", []) for pkg in fresh["map"].get("packages", {}).values() for m in pkg.get("modules", [])}
    for path in sorted(old_nodes & new_nodes):
        if old_tests.get(path, []) != new_tests.get(path, []):
            problems.append(f"tests changed: {path}")

    old_purpose = {m["path"]: m.get("purpose", "") for pkg in existing["map"].get("packages", {}).values() for m in pkg.get("modules", [])}
    new_purpose = {m["path"]: m.get("purpose", "") for pkg in fresh["map"].get("packages", {}).values() for m in pkg.get("modules", [])}
    for path in sorted(old_nodes & new_nodes):
        if old_purpose.get(path, "") != new_purpose.get(path, ""):
            problems.append(f"purpose changed: {path}")

    if existing["map"].get("api_routes", []) != fresh["map"].get("api_routes", []):
        problems.append("api routes changed")
    if [m["path"] for m in existing["map"].get("ui_modules", [])] != [m["path"] for m in fresh["map"].get("ui_modules", [])]:
        problems.append("ui module list changed")
    if existing["map"].get("entry_points", {}) != fresh["map"].get("entry_points", {}):
        problems.append("entry points changed")
    return problems


def check_catalog(repo: Path | None = None, *, out_dir: Path | None = None) -> tuple[bool, list[str]]:
    repo = Path(repo or REPO).resolve()
    out_dir = Path(out_dir or (repo / "catalog"))
    missing = [name for name in CATALOG_FILES if not (out_dir / name).is_file()]
    if missing:
        return False, [f"catalog file missing: {name}" for name in missing]
    existing = _load_existing(out_dir)
    if existing is None:
        return False, ["catalog files are unreadable"]
    fresh = build_catalog(repo)
    # Structure is what must be current: modules, contracts, dependencies,
    # tests, purposes, routes, entry points.  Line counts and runtime facts
    # (capabilities, health) may drift without making the catalog stale.
    problems = diff_catalogs(existing, fresh)
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    ok, problems = check_catalog()
    if ok:
        print("CATALOG_OK")
        return 0
    print("CATALOG_STALE")
    for line in problems[:60]:
        print(f"  - {line}")
    if len(problems) > 60:
        print(f"  ... and {len(problems) - 60} more")
    print("run: python -m catalog.build")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
