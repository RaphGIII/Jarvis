from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from brain.json_utils import lenient_json_loads
from capabilities.models import (
    CapabilityHealth,
    CapabilityManifest,
    CapabilityResolution,
    CapabilityResolutionStatus,
    GoalEnvelope,
)
from capabilities.registry import ADDRESS_TERMS, BOILERPLATE, CapabilityRegistry, _alphanumeric_parts


STRONG_THRESHOLD = 0.72
MODERATE_THRESHOLD = 0.48


@dataclass(frozen=True)
class CandidateScore:
    manifest: CapabilityManifest
    confidence: float
    reason: str
    typed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.manifest.capability_id,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "typed": self.typed,
            "health": self.manifest.health_state().value,
            "lifecycle": self.manifest.lifecycle_state().value,
        }


class CapabilityResolver:
    """Fast local resolver for installed capabilities.

    It does not write code, invent tools or perform deep planning. The output is
    a routing decision: FOUND, AMBIGUOUS, MISSING or BROKEN. Existing callers
    still receive legacy ``status`` strings (``available`` / ``missing``) so the
    current acquisition suite can migrate gradually.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        brain: Any | None = None,
        confidence_threshold: float = STRONG_THRESHOLD,
        *,
        moderate_threshold: float = MODERATE_THRESHOLD,
        require_healthy: bool = False,
        enable_brain_resolution: bool = False,
    ) -> None:
        self.registry = registry
        self.brain = brain
        self.confidence_threshold = confidence_threshold
        self.moderate_threshold = moderate_threshold
        self.require_healthy = require_healthy
        self.enable_brain_resolution = enable_brain_resolution

    def resolve(self, goal: str | GoalEnvelope) -> CapabilityResolution:
        envelope = goal if isinstance(goal, GoalEnvelope) else GoalEnvelope.from_text(str(goal))
        candidates = self._rank(envelope)
        if candidates:
            best = candidates[0]
            if best.manifest.is_broken():
                return self._result(
                    CapabilityResolutionStatus.BROKEN,
                    best,
                    envelope,
                    "Candidate matched, but its lifecycle or health is BROKEN.",
                    candidates,
                )
            if self.require_healthy and best.manifest.health_state() is not CapabilityHealth.HEALTHY:
                return self._missing(
                    envelope,
                    "Best candidate is installed, but not currently HEALTHY.",
                    candidates,
                )
            if len(candidates) > 1:
                second = candidates[1]
                if second.confidence >= self.moderate_threshold and best.confidence - second.confidence < 0.12:
                    return self._ambiguous(envelope, candidates)
            if best.confidence >= self.confidence_threshold:
                return self._result(
                    CapabilityResolutionStatus.FOUND,
                    best,
                    envelope,
                    best.reason,
                    candidates,
                )
            if best.confidence >= self.moderate_threshold:
                if len(candidates) > 1 and candidates[1].confidence >= self.moderate_threshold:
                    return self._ambiguous(envelope, candidates)
                if "registry subject match" in best.reason or best.typed:
                    return self._result(
                        CapabilityResolutionStatus.FOUND,
                        best,
                        envelope,
                        best.reason,
                        candidates,
                    )

        if self.enable_brain_resolution:
            qwen = self._qwen_resolve(envelope)
            if qwen is not None:
                return qwen
        return self._missing(envelope, "No installed capability matched the request.", candidates)

    def _rank(self, envelope: GoalEnvelope) -> list[CandidateScore]:
        query = envelope.normalized_goal
        registry_candidates = {item.capability_id: item for item in self.registry.find(query, limit=8)}
        registry_hit_ids = set(registry_candidates)
        for manifest in self.registry.all():
            if not manifest.is_active():
                continue
            if self._typed_compatible(envelope, manifest):
                registry_candidates.setdefault(manifest.capability_id, manifest)

        scored = []
        for manifest in registry_candidates.values():
            score = self._score(envelope, manifest, registry_hit=manifest.capability_id in registry_hit_ids)
            if score.confidence >= 0.18:
                scored.append(score)
        scored.sort(key=lambda item: (item.confidence, item.typed, item.manifest.capability_id), reverse=True)
        return scored

    def _typed_compatible(self, envelope: GoalEnvelope, manifest: CapabilityManifest) -> bool:
        if envelope.goal_type and manifest.goal_types and envelope.goal_type not in manifest.goal_types:
            return False
        if envelope.target_type and manifest.target_types and envelope.target_type not in manifest.target_types:
            return False
        return bool(
            (envelope.goal_type and envelope.goal_type in manifest.goal_types)
            or (envelope.target_type and envelope.target_type in manifest.target_types)
        )

    def _score(self, envelope: GoalEnvelope, manifest: CapabilityManifest, *, registry_hit: bool = False) -> CandidateScore:
        reasons: list[str] = []
        score = 0.0
        typed = False
        if registry_hit:
            score += 0.32
            reasons.append("registry subject match")
        if envelope.goal_type and manifest.goal_types:
            if envelope.goal_type in manifest.goal_types:
                score += 0.34
                typed = True
                reasons.append("goal_type matches")
            else:
                return CandidateScore(manifest, 0.0, "goal_type differs", typed=False)
        if envelope.target_type and manifest.target_types:
            if envelope.target_type in manifest.target_types:
                score += 0.2
                typed = True
                reasons.append("target_type matches")
            else:
                return CandidateScore(manifest, 0.0, "target_type differs", typed=typed)

        query_terms = _terms(envelope.normalized_goal)
        positive_terms = _manifest_terms(manifest)
        anti_score = _best_overlap(query_terms, [_terms(text) for text in manifest.anti_examples])
        example_score = _best_overlap(query_terms, [_terms(text) for text in manifest.examples + manifest.aliases])
        # Two ways of asking "is this request about this capability", and the
        # request is about it if EITHER is high.
        #
        # Dividing by the query's length alone punishes a request for saying
        # more than the minimum: "wie lautet die sha256 Pruefsumme von
        # zeus_acceptance.txt?" contains every subject word the capability
        # declares, and scored 0.462 -- under the threshold -- because it also
        # named a file and asked politely. Coverage of the CAPABILITY's own
        # subject is the other direction of the same question, and it is the
        # one that does not degrade as the sentence grows.
        shared = len(query_terms & positive_terms)
        subject_score = max(
            shared / max(1, len(query_terms)),
            shared / max(1, len(positive_terms)),
        )
        if example_score:
            score += min(0.24, example_score * 0.24)
            reasons.append("examples/aliases match")
        if subject_score:
            score += min(0.24, subject_score * 0.3)
            reasons.append("subject terms match")
        if manifest.capability_id.lower() in envelope.normalized_goal.lower():
            score += 0.5
            reasons.append("capability_id named")
        if manifest.semantic_signature and any(term in manifest.semantic_signature for term in query_terms):
            score += 0.04
            reasons.append("semantic signature hint")
        if anti_score >= max(0.45, example_score + 0.1):
            score *= 0.2
            reasons.append("anti-example is closer than examples")
        if manifest.health_state() is CapabilityHealth.BROKEN and score >= self.moderate_threshold:
            # A broken capability that really does match is reported as BROKEN
            # rather than quietly skipped, because "this is the thing you want
            # and it is defective" is what makes a repair possible.
            #
            # Only when it really does match. Lifting the score unconditionally
            # promoted every broken capability that shared one word with the
            # request to the top of the ranking, so a single defective entry in
            # the registry would answer BROKEN to nearly everything and send
            # every request into a repair of the wrong thing.
            score = max(score, self.confidence_threshold)
            reasons.append("matched but health is BROKEN")
        return CandidateScore(manifest, max(0.0, min(1.0, score)), ", ".join(reasons) or "weak local signal", typed=typed)

    def _qwen_resolve(self, envelope: GoalEnvelope) -> CapabilityResolution | None:
        if self.brain is None or not self.registry.all():
            return None
        catalog = [
            {
                "capability_id": manifest.capability_id,
                "description": manifest.description,
                "goal_types": manifest.goal_types,
                "target_types": manifest.target_types,
                "examples": manifest.examples[:4],
                "anti_examples": manifest.anti_examples[:4],
                "health": manifest.health_state().value,
                "lifecycle": manifest.lifecycle_state().value,
                "input_schema": manifest.input_schema,
                "output_schema": manifest.output_schema,
            }
            for manifest in self.registry.all()
            if manifest.is_active()
        ]
        prompt = (
            "Return JSON only. Decide if one installed capability can satisfy the user goal.\n"
            "You are a cheap local classifier, not an engineer. Do not invent a new capability.\n"
            "Schema: {\"result\":\"FOUND|AMBIGUOUS|MISSING|BROKEN\",\"capability_id\":\"...\",\"reason\":\"...\",\"confidence\":0.0}\n"
            f"GoalEnvelope: {json.dumps(envelope.to_dict(), sort_keys=True)}\n"
            f"Installed capabilities: {json.dumps(catalog, sort_keys=True)}"
        )
        try:
            raw = self.brain.generate(prompt, max_tokens=300, temperature=0.0, top_p=1.0)
            data = lenient_json_loads(_extract_json(raw))
        except Exception:
            return None
        result = str(data.get("result") or data.get("status") or "").strip().upper()
        capability_id = str(data.get("capability_id") or "")
        confidence = _clamp_float(data.get("confidence", 0.0), 0.0, 1.0)
        manifest = self.registry.get(capability_id) if capability_id else None
        if result in {"FOUND", "AVAILABLE"} and manifest is not None and confidence >= self.confidence_threshold:
            return CapabilityResolution("available", capability_id, str(data.get("reason", "")), confidence, manifest, result="FOUND", goal=envelope)
        if result == "BROKEN" and manifest is not None:
            return CapabilityResolution("broken", capability_id, str(data.get("reason", "")), confidence, manifest, result="BROKEN", goal=envelope)
        if result == "AMBIGUOUS":
            return CapabilityResolution("ambiguous", capability_id or None, str(data.get("reason", "")), confidence, manifest, result="AMBIGUOUS", goal=envelope)
        return CapabilityResolution("missing", capability_id or None, str(data.get("reason", "No suitable capability.")), confidence, manifest, result="MISSING", goal=envelope)

    @staticmethod
    def _result(
        result: CapabilityResolutionStatus,
        candidate: CandidateScore,
        envelope: GoalEnvelope,
        reason: str,
        candidates: list[CandidateScore],
    ) -> CapabilityResolution:
        legacy = {
            CapabilityResolutionStatus.FOUND: "available",
            CapabilityResolutionStatus.AMBIGUOUS: "ambiguous",
            CapabilityResolutionStatus.MISSING: "missing",
            CapabilityResolutionStatus.BROKEN: "broken",
        }[result]
        return CapabilityResolution(
            legacy,
            candidate.manifest.capability_id,
            reason,
            candidate.confidence,
            candidate.manifest,
            result=result.value,
            goal=envelope,
            candidates=[item.to_dict() for item in candidates[:5]],
        )

    @staticmethod
    def _missing(envelope: GoalEnvelope, reason: str, candidates: list[CandidateScore] | None = None) -> CapabilityResolution:
        return CapabilityResolution("missing", reason=reason, result=CapabilityResolutionStatus.MISSING.value, goal=envelope,
                                    candidates=[item.to_dict() for item in (candidates or [])[:5]])

    @staticmethod
    def _ambiguous(envelope: GoalEnvelope, candidates: list[CandidateScore]) -> CapabilityResolution:
        best = candidates[0]
        return CapabilityResolution(
            "ambiguous",
            best.manifest.capability_id,
            "Multiple plausible capabilities matched; ask one clarification instead of guessing.",
            best.confidence,
            best.manifest,
            result=CapabilityResolutionStatus.AMBIGUOUS.value,
            goal=envelope,
            candidates=[item.to_dict() for item in candidates[:5]],
        )


def _manifest_terms(manifest: CapabilityManifest) -> set[str]:
    text_parts = [
        manifest.capability_id.replace(".", " "),
        manifest.name,
        manifest.family,
        manifest.description,
        " ".join(manifest.goal_types),
        " ".join(manifest.target_types),
        " ".join(manifest.examples),
        " ".join(manifest.aliases),
        " ".join(str(item) for item in manifest.creation_metadata.get("keywords") or []),
    ]
    return _terms(" ".join(text_parts))


def _terms(text: str) -> set[str]:
    # The address term goes before the expansion, not only after it. "Zeus" is
    # filtered at the end, but ``_singular`` had already turned it into "zeu"
    # -- a token that is in no filter list, appears in every request the owner
    # speaks, and matches nothing. Every score was being divided by one more
    # term than the request actually contained.
    raw = {
        term
        for term in re.split(r"[^a-z0-9]+", _fold(text))
        if len(term) > 2 and term not in ADDRESS_TERMS
    }
    expanded = set(raw)
    synonyms = {
        "actual": {"non", "empty"},
        "content": {"text", "line"},
        "string": {"text"},
        "strings": {"text"},
        "many": {"count"},
        "number": {"count"},
        "amount": {"count"},
        "common": {"mode", "frequency"},
        "frequent": {"mode", "frequency"},
        "filename": {"file", "name"},
        "filenames": {"file", "name"},
        "dictionary": {"dict", "record"},
        "dictionaries": {"dict", "record"},
        "spiel": {"play", "music", "musik"},
        "spiele": {"play", "music", "musik"},
        "spielen": {"play", "music", "musik"},
        "lied": {"song", "track", "music"},
        "musik": {"music", "play"},
        "oeffne": {"open"},
        "offne": {"open"},
        "ordner": {"folder", "directory"},
        "verzeichnis": {"folder", "directory"},
        "groesste": {"largest", "biggest"},
        "groesster": {"largest", "biggest"},
        "groessten": {"largest", "biggest"},
    }
    for term in list(raw):
        expanded.add(_singular(term))
        expanded.update(synonyms.get(term, set()))
        expanded.update(_alphanumeric_parts(term))
    return {term for term in expanded if len(term) > 2} - BOILERPLATE - ADDRESS_TERMS


def _best_overlap(query_terms: set[str], examples: list[set[str]]) -> float:
    if not query_terms or not examples:
        return 0.0
    return max((len(query_terms & sample) / max(1, len(query_terms | sample)) for sample in examples), default=0.0)


def _singular(term: str) -> str:
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("es") and len(term) > 4:
        return term[:-2]
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def _extract_json(text: str) -> str:
    match = re.search(r"(\{.*\})", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))


def _fold(text: str) -> str:
    import unicodedata

    lowered = (text or "").lower().replace("ß", "ss")
    for source, target in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        lowered = lowered.replace(source, target)
    return "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
