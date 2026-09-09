from __future__ import annotations

import json
from pathlib import Path

from capabilities.models import CapabilityHealth, CapabilityLifecycle, CapabilityManifest


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
#: The third block is the German container nouns -- *Datei, Ordner, Pfad,
#: Dokument*. Their English equivalents (*file, files, path, paths*) were
#: already here for exactly this reason, and their absence produced a live
#: false match on 2026-09-08: asked to "bestimme die Entropie in Bits pro Byte
#: der Datei X", ZEUS answered with the SHA-256 checksum of that file, at
#: confidence 0.50, because *datei* was the only word the request and the
#: capability shared. Every capability that works on a file says *Datei*; a
#: word every capability says cannot tell two of them apart. Worse, the wrong
#: answer was then learned as one of the checksum capability's phrasings, so
#: the next wrong match would have been easier than the last.
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

aber alle allem allen aller alles auch auf aus bei beim damit dann dass dein
deine deinem deinen deiner dem den denn der des dessen dich die dies diese
diesem diesen dieser dieses dir doch dort durch ein eine einem einen einer
eines einfach einmal etwas fuer gegen gerne haben habe hast hat hatte hier
ich ihm ihn ihr ihre ihrem ihren ihrer im in ist jede jedem jeden jeder jedes
kann kannst kein keine keinem keinen keiner koennen koennte lerne lernen mach
mache machen macht mal man mehr mein meine meinem meinen meiner mich mir mit
muss nach nicht noch nur oder ohne schon sehr sein seine seinem seinen seiner
sich sie sind so soll sollen sollte um und uns unser unter viel vom von vor
war waren was wenn werde werden wie wieder wird wurde zu zum zur

datei dateien dateipfad dokument dokumente verzeichnis verzeichnisse pfad pfade
""".split())

#: The old private name, kept because the module has been imported under it.
_BOILERPLATE = BOILERPLATE

#: How the owner addresses the assistant.  A goal sentence starts with it,
#: so it ended up as a stored keyword of a learned capability and matched
#: every later request that started the same way -- a word counter was put
#: on the plan for "Zeus, store this in Knowledge" on that word alone.
ADDRESS_TERMS = frozenset({"zeus", "jarvis", "hey", "ok", "okay", "hallo", "hi", "bitte", "please"})

#: After this many consecutive verified failures a capability is FAILING and
#: the resolver stops offering it first.
FAILING_AFTER = 2

#: How many of a capability's *own* subject terms must appear in the query
#: before it is considered a candidate at all. One is enough when the terms
#: are distinctive, which is the point of scoring on keywords rather than prose.
_MIN_SUBJECT_HITS = 1

#: A term derived from prose rather than declared as a keyword has to be long
#: enough to mean something on its own.
_DERIVED_MINIMUM = 4

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


def _subject_sentence(description: str) -> str:
    """The opening sentence of a description -- what the capability is about.

    A description here is one sentence of subject followed by a contract:
    payload keys, return shapes, dry-run rules, "it must ACTUALLY WORK". The
    contract is written in the same house style for every capability, so it is
    the part that makes two unrelated capabilities look alike -- which is the
    whole defect this module was repaired for.

    Removing the known contract words was not enough on its own. Feeding whole
    descriptions back in for keyword-less capabilities immediately produced the
    old false match again on seventeen terms -- *action, afterwards,
    description, directly, doing, dry, every, function, happening, honour,
    index, outside, perform, raise, would* -- none of them in the boilerplate
    list and none of them about music or screens. Any list of such words is a
    list of the ones seen so far.

    Where the subject is is not a matter of taste: a capability states what it
    is for and then says how it must behave. Reading the first sentence is a
    rule about structure rather than about vocabulary, so it does not need
    maintaining as the house style grows.
    """

    import re

    text = " ".join(str(description or "").split())
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s", text)
    sentence = text[: match.start()] if match else text
    return sentence[:300]


class CapabilityRegistry:
    """Persistent registry for installed Jarvis capabilities."""

    schema_version = 2

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

        query_terms = self._terms(query) - BOILERPLATE
        if not query_terms:
            return []
        scored: list[tuple[float, float, CapabilityManifest]] = []
        for manifest in self._records.values():
            if not manifest.is_active():
                continue
            keywords = {
                term
                for keyword in (manifest.creation_metadata.get("keywords") or [])
                for term in self._terms(str(keyword))
            } - ADDRESS_TERMS
            # Declared aliases and examples are the capability's own statement
            # of the words it answers to, and after a verified run they also
            # hold the owner's real phrasings (``learn_alias``). Without them a
            # paraphrase of a request that already worked is indistinguishable
            # from a request for something that does not exist.
            declared = {
                term
                for phrase in (manifest.aliases + manifest.examples)
                for term in self._terms(str(phrase))
            } - ADDRESS_TERMS
            subject = self._terms(manifest.capability_id.replace(".", " ")) | keywords | declared
            subject -= BOILERPLATE | ADDRESS_TERMS
            anti = {
                term
                for phrase in manifest.anti_examples
                for term in self._terms(str(phrase))
            } - BOILERPLATE - ADDRESS_TERMS
            subject -= anti - keywords
            if not keywords and not declared:
                # Nothing declared. The identifier is then the only thing this
                # capability says it is for, and an identifier can be a code
                # name: `custom.scale` shares no word with "double an integer
                # from the request payload", so a capability that had just been
                # built, verified and registered could not be found a second
                # later, and the caller rebuilt it from scratch.
                #
                # Measured: `e26f723` fixed a false positive and introduced this
                # false negative, and three v04 tests caught it -- the second
                # call re-acquiring instead of reusing is what they assert.
                #
                # The description is allowed back in only here, and only through
                # the same boilerplate filter. That filter is what made the
                # original false positive impossible: the music provider and the
                # screen-capture goal shared *accept, actually, cannot, checked,
                # current, error, exists, failure, file, match, must, name,
                # never, not* and nothing else, and every one of those is gone
                # before this line runs.
                #: Only terms long enough to be distinctive. A declared keyword of
                #: any length is a choice -- *png*, *vlc*, *api* are all real --
                #: but a three-letter word pulled out of prose is not chosen by
                #: anyone, and *for* alone was enough to match a music provider
                #: to a screen-capture goal once the description came back in.
                subject |= {
                    term
                    for term in self._terms(_subject_sentence(manifest.description)) - BOILERPLATE
                    if len(term) >= _DERIVED_MINIMUM
                }
            hits = len(query_terms & subject) + _compound_hits(query_terms, subject)
            if manifest.capability_id.lower() in query.lower():
                hits += 4
            if hits < _MIN_SUBJECT_HITS:
                # Nothing this capability says it is for appears in the query.
                continue
            described = self._terms(manifest.description) - BOILERPLATE
            tiebreak = len(query_terms & described) / max(1, len(query_terms))
            score = hits / max(1, len(query_terms))
            if manifest.health_state() is CapabilityHealth.BROKEN:
                score *= 0.5  # demoted, not hidden: a repair can restore it
            scored.append((score, tiebreak, manifest))
        scored.sort(key=lambda item: (item[0], item[1], item[2].capability_id), reverse=True)
        return [manifest for _, _, manifest in scored[:limit]]

    def register(self, manifest: CapabilityManifest) -> CapabilityManifest:
        self._backfill(manifest)
        errors = manifest.validate()
        if errors:
            raise ValueError("; ".join(errors))
        existing = self._records.get(manifest.capability_id)
        if existing and existing.status == "active" and existing.version == manifest.version:
            raise ValueError(f"Capability already registered at version {manifest.version}: {manifest.capability_id}")
        self._records[manifest.capability_id] = manifest
        self._deduplicate(manifest)
        self._save()
        return manifest

    @staticmethod
    def _backfill(manifest: CapabilityManifest) -> CapabilityManifest:
        """Fill the v2 fields a v1 record (or a terse caller) left empty."""

        if not manifest.family:
            manifest.family = manifest.capability_id.split(".", 1)[0]
        if not manifest.implementation_path and manifest.source_location:
            manifest.implementation_path = manifest.source_location
        if not manifest.runtime_dependencies:
            manifest.runtime_dependencies = list(manifest.dependencies)
        if not manifest.semantic_signature:
            manifest.semantic_signature = manifest.compute_semantic_signature()
        if not manifest.lifecycle:
            manifest.lifecycle = manifest.lifecycle_state().value
        if manifest.validation_status.get("verified") and not manifest.health:
            manifest.health = {"state": "healthy", "health": CapabilityHealth.HEALTHY.value}
        return manifest

    def _deduplicate(self, manifest: CapabilityManifest) -> list[str]:
        """One subject, one live capability.

        Two capabilities that declare the same semantic signature -- same
        family, goal types, target types, effects and dependencies -- are the
        same thing under two names, and a resolver facing both can only guess.
        The newcomer wins and inherits the aliases and examples of what it
        replaces, so the paraphrases the older one had already learned are not
        lost with it. Nothing is deleted: the superseded record stays readable
        with its history and says what replaced it.
        """

        if not manifest.declares_subject:
            return []
        signature = manifest.semantic_signature or manifest.compute_semantic_signature()
        superseded: list[str] = []
        for other in list(self._records.values()):
            if other.capability_id == manifest.capability_id:
                continue
            if not other.is_active() or not other.declares_subject:
                continue
            if (other.semantic_signature or other.compute_semantic_signature()) != signature:
                continue
            other.lifecycle = CapabilityLifecycle.SUPERSEDED.value
            other.status = "deprecated"
            other.validation_status = {**other.validation_status, "superseded_by": manifest.capability_id}
            manifest.aliases = _merge(manifest.aliases, other.aliases, limit=48)
            manifest.examples = _merge(manifest.examples, other.examples, limit=48)
            superseded.append(other.capability_id)
        if superseded:
            manifest.supersedes = _merge(manifest.supersedes, superseded, limit=24)
        return superseded

    def learn_alias(self, capability_id: str, phrase: str) -> CapabilityManifest | None:
        """Record a real owner phrasing that this capability satisfied.

        Generalization, and the cheapest form of it: the next paraphrase of a
        request that already worked resolves on stored evidence instead of on
        a fresh guess. Only a phrase that actually reached a successful run
        with nothing contradicting it gets here, so the alias list is a record
        of what worked, not of what a model thought might.

        The phrase is stored with its particulars removed. Storing it verbatim
        makes the file name part of what the capability answers to, and it then
        answers everything about that file: measured, a checksum capability
        that had learned "…die Datei zeus_acceptance.txt" scored 0.57 on
        "oeffne zeus_acceptance.txt" and would have run a checksum for a
        request to open the file.
        """

        from capabilities.generalize import generalize

        manifest = self._records.get(capability_id)
        phrase = " ".join(generalize(str(phrase or "")).goal.split())[:160]
        if manifest is None or not phrase:
            return manifest
        known = {item.strip().lower() for item in manifest.aliases}
        if phrase.strip().lower() in known:
            return manifest
        # Newest first, so a full list evicts the oldest phrasing rather than
        # silently refusing to learn any more: ``_merge`` truncates at the
        # limit, and appending would have made the forty-ninth wording -- and
        # every wording after it -- unlearnable without any sign of it.
        manifest.aliases = _merge([phrase], manifest.aliases, limit=48)
        self._save()
        return manifest

    def note_execution(self, capability_id: str, ok: bool, detail: str = "", *,
                       repair: bool = False, kind: str = "") -> CapabilityManifest | None:
        """Record one real execution outcome on the manifest's runtime health.

        Policy: a failure sets DEGRADED at once and FAILING after
        ``FAILING_AFTER`` in a row; a success after failures sets DEGRADED
        (one good call is not a clean bill), a second consecutive success
        HEALTHY.  Persisted, so the state survives a restart.

        ``kind`` is :mod:`runtime.failure_kind`'s reading of the failure, and
        it overrides the streak in one direction only: a ``DEFECT`` -- a
        traceback out of the capability's own module -- is FAILING on the
        first occurrence, because waiting for a second identical crash to
        confirm the first is a second failed owner request bought for nothing.
        The streak still governs everything the classifier is not sure about.
        """

        import time as _time

        manifest = self._records.get(capability_id)
        if manifest is None:
            return None
        health = manifest.health_view()
        now = _time.strftime("%Y-%m-%dT%H:%M:%S")
        health["calls"] = int(health.get("calls", 0)) + 1
        health["last_used"] = now
        if ok:
            streak = int(health.get("consecutive_ok", 0)) + 1
            health["consecutive_ok"] = streak
            health["consecutive_failures"] = 0
            health["last_ok_at"] = now
            healthy_now = streak >= 2 or health.get("state") in {"unverified", "healthy"}
            health["state"] = "healthy" if healthy_now else "degraded"
            health["health"] = CapabilityHealth.HEALTHY.value if healthy_now else CapabilityHealth.AT_RISK.value
            if healthy_now:
                # The defect is over, so the record of it stops being current.
                # A repaired capability that kept quoting the error of the
                # version before it told the owner, a day later, that
                # ``hex_digest`` was still broken.
                health["last_error"] = ""
                health["last_failure_kind"] = ""
        else:
            failures = int(health.get("consecutive_failures", 0)) + 1
            health["consecutive_failures"] = failures
            health["consecutive_ok"] = 0
            health["last_error_at"] = now
            health["last_error"] = str(detail)[:300]
            broken = failures >= FAILING_AFTER or str(kind).upper() == "DEFECT"
            health["state"] = "failing" if broken else "degraded"
            health["health"] = CapabilityHealth.BROKEN.value if broken else CapabilityHealth.AT_RISK.value
            health["failure_count"] = int(health.get("failure_count", 0)) + 1
            if kind:
                health["last_failure_kind"] = str(kind)[:40]
        if repair:
            health.setdefault("repairs", []).append({"at": now, "ok": ok, "detail": str(detail)[:200]})
            health["repairs"] = health["repairs"][-10:]
        manifest.health = health
        manifest.failure_count = int(health.get("failure_count", manifest.failure_count or 0) or 0)
        calls = int(health.get("calls", 0) or 0)
        if calls:
            manifest.success_rate = max(0.0, min(1.0, (calls - manifest.failure_count) / calls))
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
        found = int(data.get("schema_version", 0))
        if found > self.schema_version:
            raise ValueError("Unsupported capability registry schema version.")
        self._records = {
            capability_id: CapabilityManifest.from_dict(payload)
            for capability_id, payload in (data.get("capabilities") or {}).items()
        }
        if found < self.schema_version:
            # A v1 record has no family, signature, lifecycle or implementation
            # path. Reading it is not enough: the resolver ranks on exactly
            # those fields, so a registry that was never rewritten would keep
            # every pre-migration capability permanently unrankable.
            for manifest in self._records.values():
                self._backfill(manifest)
            self._save()

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
        """Subject words, folded so one German word is one term.

        ``ü`` is alphanumeric, so without folding "Prüfsumme" and
        "Pruefsumme" are two different tokens that never match each other --
        and the resolver, which does fold, disagreed with the registry, which
        did not. Measured live: a capability indexed under *prüfsumme* could
        not be found by a request written *Pruefsumme*.
        """

        from capabilities.models import _fold

        folded = "".join(ch if ch.isalnum() else " " for ch in _fold(text))
        found = {term for term in folded.split() if len(term) > 2}
        for term in list(found):
            found |= _alphanumeric_parts(term)
        return found


def _merge(first: list[str], second: list[str], *, limit: int = 32) -> list[str]:
    """Union of two string lists, order-stable and case-insensitively unique."""

    out: list[str] = []
    seen: set[str] = set()
    for item in list(first) + list(second):
        text = " ".join(str(item or "").split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _alphanumeric_parts(term: str) -> set[str]:
    """``sha256`` also means ``sha`` and ``256``.

    A hyphen makes the boundary visible to a tokeniser and nothing else does:
    "SHA-256" becomes {sha, 256} while "sha256" stays one opaque token, so the
    same request written two ordinary ways shares no vocabulary at all.
    Measured: a checksum capability learned from "SHA-256-Pruefsumme" scored
    0.40 against "sha256 Fingerprint" -- under the threshold, and answered as
    if it did not exist.

    Only where the boundary is real. A term with no letter/digit transition is
    returned unchanged.
    """

    import re as _re

    parts = {piece for piece in _re.findall(r"[a-z]+|[0-9]+", term) if len(piece) > 2}
    return parts if len(parts) > 1 else set()
