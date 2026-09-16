"""Which memory writes are protected: ZEUS's personality, its creator and owner, and the owner's own person.

Saving an ordinary note ("merke dir: Klausur am 3. Oktober") must stay frictionless.  But an
utterance that asks ZEUS to remember or overwrite something about

  * ZEUS itself -- its personality, character, name, rules, identity ("du bist ab jetzt ...",
    "deine Persönlichkeit", "nenn dich ..."),
  * its creator and owner ("dein Schöpfer", "Raphael hat dich gebaut", "dein Besitzer ist ..."),
  * Raphael as a person -- facts about him ("über mich", "ich bin ...", "mein Geburtstag", "meine
    Adresse", family, health, passwords),

is a protected write.  It is held until the owner confirms it with the password (scope
PROTECTED_MEMORY), and both the hold and the release are audited.  A model's opinion never
enters this decision: the classification is a deterministic reading of the owner's words, and
the password is checked by the security gate, not by a prompt.

Rule (exact):  protected  <=>  (SAVE_VERB and PROTECTED_SUBJECT)  or  IDENTITY_DIRECTIVE

  SAVE_VERB          speicher(e/n), merk(e) dir, merke (dir), behalte, notier(e), überschreib(e),
                     ersetz(e) ... in deinem gedächtnis, vergiss (löschen aus dem Gedächtnis),
                     remember, save, store, overwrite, memorize, note that
  PROTECTED_SUBJECT  personality:  persönlichkeit, charakter, wesen, verhalten, deine regel(n), dein name,
                                   deine identität, wer du bist, dein stil, wie du sprichst/antwortest
                     creator/owner: raphael, schöpfer, erschaffer, entwickler, ersteller, erbauer, besitzer,
                                   owner, creator, wer dich gebaut/erschaffen hat, dein mensch
                     owner person: über mich, ich bin/heiße, mein(e) name/geburtstag/adresse/alter/wohnort/
                                   telefon/passwort/familie/freundin/frau/kind/krankheit/allergie/arzt/
                                   bankdaten, meine daten, von mir
  IDENTITY_DIRECTIVE du bist (ab jetzt|jetzt|nun) ..., du heißt (ab jetzt) ..., dein name ist ...,
                     nenn dich ..., ab (jetzt|sofort) bist du ..., du wurdest von ... gebaut/erschaffen,
                     dein schöpfer/besitzer ist ..., vergiss (raphael|deinen schöpfer|wer du bist)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SCOPE = "PROTECTED_MEMORY"

_SAVE = re.compile(
    r"\b(speicher(?:e|n|st)?|merk(?:e|st)?\s+(?:dir|es\s+dir|dir\s+das)|merke\b|behalt(?:e)?\s+(?:dir\s+)?(?:das\s+)?(?:im\s+(?:gedächtnis|kopf))?"
    r"|notier(?:e|st)?|[üu]berschreib(?:e|st)?|ersetz(?:e|t)?\s+(?:in\s+deinem\s+gedächtnis|dein\s+wissen)|schreib(?:e)?\s+(?:dir\s+)?auf"
    r"|vergiss(?:t)?|l[öo]sch(?:e)?\s+(?:aus\s+deinem\s+gedächtnis|dein\s+wissen\s+über)"
    r"|remember|memori[sz]e|save|store|overwrite|note\s+that|keep\s+in\s+mind|forget)\b",
    re.I,
)
_PERSONALITY = re.compile(
    r"\b(pers[öo]nlichkeit|charakter|wesen|dein(?:e|en|em)?\s+(?:verhalten|regel(?:n)?|name|identit[äa]t|stil|art|ton|sprache|rolle|aufgabe|zweck)"
    r"|wer\s+du\s+bist|wie\s+du\s+(?:sprichst|antwortest|redest|dich\s+verh[äa]ltst)|verhalte?\s+dich|antworte\s+(?:ab\s+jetzt\s+)?immer"
    r"|sprich\s+(?:ab\s+jetzt\s+)?immer|your\s+(?:personality|character|name|identity|rules|behaviou?r|style)|who\s+you\s+are)\b",
    re.I,
)
_CREATOR = re.compile(
    r"\b(raphael|sch[öo]pfer|erschaffer|entwickler|ersteller|erbauer|besitzer|eigent[üu]mer|owner|creator|maker"
    r"|wer\s+dich\s+(?:gebaut|erschaffen|entwickelt|programmiert|gemacht)\s+hat|dein\s+mensch|deine\s+herkunft|who\s+(?:made|built|created)\s+you)\b",
    re.I,
)
_OWNER_PERSON = re.compile(
    r"\b([üu]ber\s+mich|ich\s+(?:bin|hei[ßs]e|wohne|arbeite|studiere|habe\s+(?:eine?|keine?)|leide|nehme)"
    r"|mein(?:e|en|em|er|es)?\s+(?:name|vorname|nachname|geburtstag|geburtsdatum|adresse|anschrift|alter|wohnort|telefon(?:nummer)?|handy(?:nummer)?"
    r"|passw(?:ort|örter)|pin|konto|bank(?:daten|verbindung)|iban|kreditkarte|familie|eltern|mutter|vater|bruder|schwester|freundin|freund|frau|mann|partner(?:in)?"
    r"|kind(?:er)?|sohn|tochter|krankheit|diagnose|allergie(?:n)?|medikament(?:e)?|arzt|[äa]rztin|therapie|gesundheit|gewicht|blutgruppe|religion|daten|beruf|arbeitgeber|gehalt|einkommen)"
    r"|von\s+mir|about\s+me|my\s+(?:name|birthday|address|age|phone|password|family|health|doctor|bank|salary|job)|i\s+am\b|i'm\b)",
    re.I,
)
_RELATIONSHIP = re.compile(
    r"\b(du\s+(?:bist|wirst|sollst)\s+(?:auch\s+|ab\s+jetzt\s+|jetzt\s+)?(?:mein|meine|ein|eine)\s+\w*(?:freund|begleiter|partner|vertraute|berater|coach|mentor|lehrer|kumpel)"
    r"|du\s+(?:auch\s+|ab\s+jetzt\s+|jetzt\s+)?(?:mein|meine|ein|eine)\s+\w*(?:freund|begleiter|partner|vertraute|berater|coach|mentor|lehrer|kumpel)\w*\s+(?:bist|wirst|sein\s+sollst)"
    r"|h[öo]rst\s+(?:mir\s+)?zu|zuh[öo]rst|gibst\s+(?:mir\s+)?rat|f[üu]r\s+mich\s+da|unterst[üu]tzt\s+mich|unsere\s+beziehung"
    r"|you\s+are\s+(?:also\s+)?my\s+(?:friend|companion|coach|mentor)|listen\s+to\s+me|give\s+me\s+advice)\b",
    re.I,
)
_DIRECTIVE = re.compile(
    r"(\bdu\s+bist\s+(?:ab\s+jetzt|jetzt|nun|von\s+nun\s+an|k[üu]nftig)\b|\bdu\s+hei[ßs]t\s+(?:ab\s+jetzt\s+|jetzt\s+|nun\s+)?\S+|\bdein\s+name\s+ist\b"
    r"|\bnenn(?:e)?\s+dich\b|\bab\s+(?:jetzt|sofort|heute)\s+bist\s+du\b|\bdu\s+wurdest\s+von\b|\bdein(?:e)?\s+(?:sch[öo]pfer|besitzer|erschaffer|entwickler|ersteller|owner|creator)\s+(?:ist|hei[ßs]t|war|sind)\b"
    r"|\bvergiss(?:t)?\s+(?:raphael|deinen\s+sch[öo]pfer|wer\s+du\s+bist|deine\s+pers[öo]nlichkeit|deine\s+identit[äa]t|deinen\s+namen)"
    r"|\byou\s+are\s+now\b|\byour\s+name\s+is\b|\bfrom\s+now\s+on\s+you\s+are\b|\bcall\s+yourself\b|\byou\s+were\s+(?:made|built|created)\s+by\b)",
    re.I,
)


@dataclass
class MemoryWriteClass:
    protected: bool
    save_verb: bool
    subjects: list[str] = field(default_factory=list)
    directive: bool = False

    @property
    def reason(self) -> str:
        if not self.protected:
            return "ordinary"
        parts = list(self.subjects)
        if self.directive:
            parts.append("identity directive")
        return ", ".join(parts) or "protected"

    def to_dict(self) -> dict:
        return {"protected": self.protected, "save_verb": self.save_verb, "subjects": list(self.subjects),
                "directive": self.directive, "reason": self.reason, "scope": SCOPE}


def classify_memory_write(text: str, *, title: str = "", implicit_save: bool = False) -> MemoryWriteClass:
    """The exact rule from the module docstring, applied to the owner's words (and a note's title).

    ``implicit_save``: the caller IS a write (a note or knowledge primitive), so the verb is given.
    """

    body = f"{title or ''}\n{text or ''}".strip()
    if not body:
        return MemoryWriteClass(False, False)
    save_verb = implicit_save or bool(_SAVE.search(body))
    subjects: list[str] = []
    if _PERSONALITY.search(body):
        subjects.append("personality")
    if _CREATOR.search(body):
        subjects.append("creator/owner")
    if _OWNER_PERSON.search(body):
        subjects.append("owner person")
    if _RELATIONSHIP.search(body):
        subjects.append("relationship")
    directive = bool(_DIRECTIVE.search(body))
    protected = (save_verb and bool(subjects)) or directive
    return MemoryWriteClass(protected, save_verb, subjects, directive)


def is_protected_memory_write(text: str, *, title: str = "") -> bool:
    return classify_memory_write(text, title=title).protected
