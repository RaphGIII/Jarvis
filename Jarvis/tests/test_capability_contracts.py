"""Manifest v2: typed semantic contracts, compact and machine-readable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capabilities.contracts import SemanticContract, infer_contract, normalise_token
from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest, SkillSpecification
from capabilities.registry import CapabilityRegistry


def test_tokens_are_normalised_and_validated():
    contract = SemanticContract.from_dict({
        "goals": ["Understand Recent Loss"], "events": ["chess_game_finished"], "consumes": ["Active Chess-Session"],
        "produces": ["chess.game_record"], "effects": ["chess_game_available_for_analysis"],
        "risk_class": "harmless", "latency_class": "medium", "cost_class": "free",
    })
    assert contract.goals == ["understand_recent_loss"] and contract.consumes == ["active_chess_session"]
    assert contract.requires == {"active_chess_session"}
    assert contract.provides == {"chess.game_record", "chess_game_available_for_analysis"}
    assert contract.validate() == []
    assert SemanticContract.from_dict({"risk_class": "explosive"}).validate() == [
        "risk_class 'explosive' not in ('harmless', 'reversible', 'irreversible')"]
    assert normalise_token("  Chess Game Record ") == "chess_game_record"


def test_contracts_serialise_compactly_and_round_trip():
    contract = SemanticContract.from_dict({"consumes": ["a"], "produces": ["b"], "latency_class": "slow"})
    data = contract.to_dict()
    assert data == {"consumes": ["a"], "produces": ["b"], "latency_class": "slow"}, "defaults and empties are omitted"
    assert SemanticContract.from_dict(data) == contract


def test_a_manifest_carries_its_contract_through_the_registry(tmp_path: Path):
    registry = CapabilityRegistry(tmp_path / "registry.json")
    manifest = CapabilityManifest(
        "chess.capture_game", "Records the game just played.", lifecycle=CapabilityLifecycle.ACTIVE.value,
        health={"state": "healthy", "health": CapabilityHealth.HEALTHY.value},
        contract={"consumes": ["active_chess_session"], "produces": ["chess_game_record"],
                  "effects": ["chess_game_available_for_analysis"], "events": ["chess_game_finished"],
                  "related_projects": ["Schach Training"], "domain": "chess", "risk_class": "harmless"},
    )
    registry.register(manifest)
    again = CapabilityRegistry(tmp_path / "registry.json").get("chess.capture_game")
    assert again is not None
    contract = again.semantic_contract()
    assert not contract.inferred and contract.provides == {"chess_game_record", "chess_game_available_for_analysis"}
    assert contract.related_projects == ["Schach Training"] and contract.events == ["chess_game_finished"]
    stored = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["capabilities"]["chess.capture_game"]
    assert stored["contract"]["consumes"] == ["active_chess_session"]
    assert stored["effective_contract"] == stored["contract"]


def test_a_legacy_manifest_gets_an_inferred_contract_and_says_so():
    legacy = CapabilityManifest("archive.zip.create", "Package a folder into a zip.", permissions_required=["fs.write"],
                                goal_types=["ARCHIVE_FOLDER"], preconditions=["folder exists"], latency_class="local",
                                tests_location="/x/test_capability.py")
    contract = legacy.semantic_contract()
    assert contract.inferred
    assert contract.produces == ["archive.zip.create.result"]
    assert contract.goals == ["archive_folder"] and contract.preconditions == ["folder_exists"]
    assert contract.risk_class == "reversible" and contract.permissions == ["fs.write"]
    assert legacy.to_dict()["effective_contract"]["inferred"] is True


def test_manifest_validation_rejects_a_malformed_contract():
    bad = CapabilityManifest("x.y", "desc", contract={"produces": ["ok"], "cost_class": "priceless"})
    assert any("cost_class" in error for error in bad.validate())
    registry_error = None
    try:
        CapabilityRegistry(Path(__file__).with_name("_unused_registry.json")).register(bad)
    except ValueError as exc:
        registry_error = str(exc)
    finally:
        Path(__file__).with_name("_unused_registry.json").unlink(missing_ok=True)
    assert registry_error and "cost_class" in registry_error


def test_a_specification_passes_its_contract_into_the_manifest():
    spec = SkillSpecification(
        capability_id="chess.classify_errors", objective="Classify the mistakes in an analysed game.",
        acceptance_criteria=["returns categories"], public_tests=[{"input": {"analysis": {}}, "expected_keys": ["categories"]}],
        metadata={"contract": {"consumes": ["chess_engine_analysis"], "produces": ["chess_error_profile_update"],
                               "goals": ["understand_recent_loss", "improve_chess"]}},
    )
    manifest = spec.to_manifest()
    assert manifest.semantic_contract().goals == ["understand_recent_loss", "improve_chess"]
    assert manifest.semantic_contract().requires == {"chess_engine_analysis"}


def test_install_reads_the_contract_the_engineer_declared_next_to_the_code(tmp_path: Path):
    from capabilities.service import _declared_contract

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "contract.json").write_text(json.dumps({"consumes": ["chess_game_record"], "produces": ["chess_engine_analysis"],
                                                         "latency_class": "slow"}), encoding="utf-8")
    assert _declared_contract(workspace) == {"consumes": ["chess_game_record"], "produces": ["chess_engine_analysis"], "latency_class": "slow"}
    (workspace / "contract.json").write_text("{not json", encoding="utf-8")
    assert _declared_contract(workspace) == {}
    (workspace / "contract.json").write_text(json.dumps({"produces": ["x"], "risk_class": "nope"}), encoding="utf-8")
    assert _declared_contract(workspace) == {}, "an invalid contract is not installed; the capability still is"
