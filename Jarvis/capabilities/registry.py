from __future__ import annotations

import json
from pathlib import Path

from capabilities.models import CapabilityManifest


#: Vocabulary that every capability contract contains, so sharing it says
#: nothing about whether two capabilities are related. Drawn from the words
#: that actually produced a false match between a music provider and a
#: screen-capture goal, plus the ordinary English around them.
#:
#: The second block is the vocabulary of *building and repairing* capabilities,
#: which is not a subject either. A capability's stored keywords are derived
#: from the goal that produced it, and the goal that produces version 1.0.4 of
#: anything is a repair brief -- so `music.provider.spotify` came to be indexed
#: under *defect, existing, implementation, rebuild, repair, working*, and
#: answered "rebuild the existing implementation because of a defect and repair
#: the working code" with a music player. Every capability in the registry
#: acquires these words, so they can only ever produce a false match.
#:
#: Exported rather than private: the same words must be excluded where keywords
#: are *stored*, not only where they are matched, or the registry keeps filling
#: up with terms that are then filtered out on every single query.
BOILERPLATE = frozenset("""
accept accepts accepted actually and anything are because been before being
call called caller cannot check checked checks computer current currently
declare declared error errors exist exists expected fail failed failure false
file files first from give given has have how implement implementation
input into its just key keys machine match matched must name named never not
now only optional other out output package packages path paths payload
python read reads report reported reports require required return returned
returns run running same say says shape should size some standard
str string success successful take takes than that the them then there these
this those true unless use used uses using value values what when where
which while will with without work works write writes written you your
dict bool int list none true false main test tests

attempt attempted attempts broken build building built capability capabilities
change changed changes create created creating defect defects existing feature
fix fixed fixes fixing implement implementing improve improved make making
missing new provider rebuild rebuilding rebuilt reusable repair repaired
repairing reimplement replace replacing rewrite rewriting version versions
work working thing something anything everything currently what which
""".split())

#: The old private name, kept because the module has been imported under it.
_BOILERPLATE = BOILERPLATE

#: How many of a capability's *own* subject terms must appear in the query
#: before it is considered a candidate at all. One is enough when the terms
#: are distinctive, which is the point of scoring on keywords rather than prose.
_MIN_SUBJECT_HITS = 1

#: Below this length a term is too short to be safely matched as part of
#: another word: "png" inside "opening" would be a hit, and a false one.
_COMPOUND_MINIMUM = 6


def _compound_hits(query_terms: set[str], subject: set[str]) -> int:
    """Matches where one word contains another, for languages that glue.

    German writes "Bildschirmfoto" where English writes "screen photo", so a
    keyword of *bildschirm* never equals a query term of *bildschirmfoto* and
    exact matching misses it entirely. The user said the right word; the
    tokeniser disagreed about where it ended.

    Only for terms long enough that containment means something. Applied to
    short ones it manufactures matches -- *png* is inside *opening* -- which is
    the failure this scoring was just repaired for.
    """

    hits = 0
    for term in query_terms:
        if len(term) < _COMPOUND_MINIMUM or term in subject:
            continue
        for candidate in subject:
            if len(candidate) >= _COMPOUND_MINIMUM and (candidate in term or term in candidate):
                hits += 1
                break
    return hits


class CapabilityRegistry:
    """Persistent registry for installed Jarvis capabilities."""

    schema_version = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, CapabilityManifest] = {}
        self._load()

    def has(self, capability_id: str) -> bool:
        manifest = self._records.get(capability_id)
        return bool(manifest and manifest.status == "active")

    def get(self, capability_id: str) -> CapabilityManifest | None:
        return self._records.get(capability_id)

    def find(self, query: str, *, limit: int = 5) -> list[CapabilityManifest]:
        """Capabilities whose *subject* matches the query.

        Scored on the identifier and the declared keywords, not on the
        description.  A capability's description is its contract -- payload
        keys, return shapes, failure rules -- written in the same house style
        for every capability, so two unrelated capabilities share most of it.

        Measured, when this scored on descriptions: a goal about capturing the
        screen as a PNG matched the Spotify music provider on 39 of its 90
        terms, for a score of 0.43. The shared terms were *accept, actually,
        cannot, checked, current, error, exists, failure, file, match, must,
        name, never, not* -- every one of them generic English, not one of them
        about music or screens. The mission concluded a screen-capture
        capability was already installed and reported it as acquired.

        Term overlap was standing in for "is about the same thing", and stopped
        being that as soon as descriptions grew. The identifier and the
        keywords are what a capability declares itself to be *for*, so those are
        what get matched; the description can only break a tie.
        """

        query_terms = self._terms(query) - _BOILERPLATE
        if not query_terms:
            return []
        scored: list[tuple[float, float, CapabilityManifest]] = []
        for manifest in self._records.values():
            if manifest.status != "active":
                continue
            subject = self._terms(manifest.capability_id.replace(".", " ")) | {
                term
                for keyword in (manifest.creation_metadata.get("keywords") or [])
                for term in self._terms(str(keyword))
            }
            subject -= _BOILERPLATE
            hits = len(query_terms & subject) + _compound_hits(query_terms, subject)
            if manifest.capability_id.lower() in query.lower():
                hits += 4
            if hits < _MIN_SUBJECT_HITS:
                # Nothing this capability says it is for appears in the query.
                continue
            described = self._terms(manifest.description) - _BOILERPLATE
            tiebreak = len(query_terms & described) / max(1, len(query_terms))
            scored.append((hits / max(1, len(query_terms)), tiebreak, manifest))
        scored.sort(key=lambda item: (item[0], item[1], item[2].capability_id), reverse=True)
        return [manifest for _, _, manifest in scored[:limit]]

    def register(self, manifest: CapabilityManifest) -> CapabilityManifest:
        errors = manifest.validate()
        if errors:
            raise ValueError("; ".join(errors))
        existing = self._records.get(manifest.capability_id)
        if existing and existing.status == "active" and existing.version == manifest.version:
            raise ValueError(f"Capability already registered at version {manifest.version}: {manifest.capability_id}")
        self._records[manifest.capability_id] = manifest
        self._save()
        return manifest

    def disable(self, capability_id: str, reason: str = "") -> CapabilityManifest:
        manifest = self._records[capability_id]
        manifest.status = "disabled"
        manifest.validation_status = {**manifest.validation_status, "disabled_reason": reason}
        self._save()
        return manifest

    def restore(self, capability_id: str, reason: str = "") -> CapabilityManifest:
        """Put a disabled capability back into service.

        The counterpart to :meth:`disable`, and it exists because a repair
        disables what it is about to rebuild -- so that the resolver stops
        handing out something known to be defective while the rebuild runs. If
        the rebuild then fails, the old behaviour left the registry with a
        verified capability disabled and nothing in its place: one failed
        repair, and the system could no longer play music at all.

        A capability with a known defect is worse than one without and better
        than none. Why it was disabled stays on the record, and what put it
        back is recorded next to it.
        """

        manifest = self._records[capability_id]
        manifest.status = "active"
        manifest.validation_status = {
            **manifest.validation_status,
            "restored_reason": reason or "a repair did not produce a verified replacement",
        }
        self._save()
        return manifest

    def all(self) -> list[CapabilityManifest]:
        return list(self._records.values())

    def _load(self) -> None:
        if not self.path.exists():
            self._records = {}
            self._save()
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 0)) > self.schema_version:
            raise ValueError("Unsupported capability registry schema version.")
        self._records = {
            capability_id: CapabilityManifest.from_dict(payload)
            for capability_id, payload in (data.get("capabilities") or {}).items()
        }

    def _save(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "capabilities": {capability_id: manifest.to_dict() for capability_id, manifest in sorted(self._records.items())},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term for term in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(term) > 2}
