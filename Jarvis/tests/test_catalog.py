"""The ZEUS Catalog: derived from the code, checked against it, narrowed for engineers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from catalog.build import REPO, build_catalog, render_files, write_catalog
from catalog.check import check_catalog, diff_catalogs
from catalog.context import build_context


@pytest.fixture
def small_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "service").mkdir(parents=True)
    (root / "brain").mkdir()
    (root / "tests").mkdir()
    (root / "ui").mkdir()
    (root / "service" / "__init__.py").write_text('"""The service package."""\n', encoding="utf-8")
    (root / "service" / "core.py").write_text(
        '"""The core: answers requests."""\n\nfrom brain.model import Model\n\n\nclass Core:\n    """The product core."""\n\n'
        '    def answer(self, text: str) -> str:\n        return text\n\n    def _private(self):\n        pass\n\n\n'
        'def helper(x: int, *, flag: bool = False) -> int:\n    """A helper."""\n    return x\n', encoding="utf-8")
    (root / "service" / "http.py").write_text(
        '"""HTTP."""\nroutes = {\n    "/api/status": 1,\n    "/api/message": 2,\n}\n', encoding="utf-8")
    (root / "brain" / "__init__.py").write_text("", encoding="utf-8")
    (root / "brain" / "model.py").write_text('"""A model."""\n\n\nclass Model:\n    def generate(self, prompt: str) -> str:\n        return prompt\n',
                                             encoding="utf-8")
    (root / "tests" / "test_core.py").write_text("from service.core import Core\n\n\ndef test_x():\n    assert Core\n", encoding="utf-8")
    (root / "ui" / "app.js").write_text('/* the app */\nimport { x } from "./core/dom.js";\n', encoding="utf-8")
    return root


def test_the_catalog_describes_modules_contracts_dependencies_and_tests(small_repo):
    catalog = build_catalog(small_repo)
    nodes = set(catalog["graph"]["nodes"])
    assert {"service/core.py", "service/http.py", "brain/model.py", "service/__init__.py"} <= nodes
    core = catalog["interfaces"]["modules"]["service/core.py"]
    assert core["classes"][0]["name"] == "Core" and core["classes"][0]["methods"] == ["def answer(self, text: str) -> str"]
    assert core["functions"][0]["signature"] == "def helper(x: int, *, flag: bool=False) -> int"
    assert "_private" not in json.dumps(core), "private names are not contracts"
    assert catalog["graph"]["imports"]["service/core.py"] == ["brain/model.py"]
    assert catalog["graph"]["dependents"]["brain/model.py"] == ["service/core.py"]
    service_pkg = catalog["map"]["packages"]["service"]
    core_row = next(m for m in service_pkg["modules"] if m["path"] == "service/core.py")
    assert core_row["tests"] == ["tests/test_core.py"] and core_row["purpose"] == "The core: answers requests."
    assert service_pkg["purpose"] == "The service package."
    assert catalog["map"]["api_routes"] == ["/api/message", "/api/status"]
    assert catalog["map"]["ui_modules"][0]["path"] == "ui/app.js" and catalog["map"]["ui_modules"][0]["imports"] == ["core/dom.js"]


def test_the_rendering_is_stable_and_round_trips(small_repo):
    first = render_files(build_catalog(small_repo))
    second = render_files(build_catalog(small_repo))
    assert first == second
    written = write_catalog(small_repo)
    assert {p.name for p in written} >= {"ZEUS_MAP.yaml", "INTERFACES.yaml", "DEPENDENCY_GRAPH.json", "ARCHITECTURE.md", "CAPABILITIES.yaml"}
    assert yaml.safe_load((small_repo / "catalog" / "ZEUS_MAP.yaml").read_text(encoding="utf-8"))["totals"]["modules"] == 5
    ok, problems = check_catalog(small_repo)
    assert ok, problems


def test_a_changed_signature_a_new_module_and_a_moved_test_make_the_catalog_stale(small_repo):
    write_catalog(small_repo)
    (small_repo / "service" / "core.py").write_text(
        (small_repo / "service" / "core.py").read_text(encoding="utf-8").replace("def answer(self, text: str) -> str",
                                                                                   "def answer(self, text: str, fast: bool) -> str"),
        encoding="utf-8")
    (small_repo / "service" / "new_thing.py").write_text('"""New."""\n\n\ndef go():\n    pass\n', encoding="utf-8")
    (small_repo / "tests" / "test_core.py").rename(small_repo / "tests" / "test_core_moved.py")
    ok, problems = check_catalog(small_repo)
    assert not ok
    assert "interface changed: service/core.py" in problems
    assert "module not in catalog: service/new_thing.py" in problems
    assert "tests changed: service/core.py" in problems
    write_catalog(small_repo)
    assert check_catalog(small_repo)[0]


def test_line_count_drift_does_not_make_the_catalog_stale(small_repo):
    write_catalog(small_repo)
    path = small_repo / "brain" / "model.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# a comment\n# another\n", encoding="utf-8")
    ok, problems = check_catalog(small_repo)
    assert ok, problems


def test_the_engineering_context_is_narrow_and_ordered_by_relevance(small_repo):
    write_catalog(small_repo)
    context = build_context(["brain/model.py"], request="make generate faster", repo=small_repo, budget_chars=60_000)
    assert context.sections[:2] == ["impacted", "contracts"]
    assert "class Model" in context.text and "def generate(self, prompt: str) -> str" in context.text
    assert context.dependents == {"brain/model.py": ["service/core.py"]}
    assert "service/core.py" in context.text and "imports `brain/model.py`" in context.text
    assert context.tests == ["tests/test_core.py"], "the dependent's tests belong to the change"
    tight = build_context(["brain/model.py"], repo=small_repo, budget_chars=600)
    assert tight.chars <= 700 and tight.sections[0] == "impacted"
    assert "overview" not in tight.sections


def test_the_committed_catalog_is_current():
    """A stale catalog is a verification failure: run `python -m catalog.build` before committing."""

    ok, problems = check_catalog(REPO)
    assert ok, "catalog is stale:\n" + "\n".join(problems[:20])
