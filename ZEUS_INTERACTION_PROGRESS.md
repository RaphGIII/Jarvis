# ZEUS — Human interaction, execution reliability, product quality (sprint of 2026-09-01)

Branch `adaptive-brain-v1`. Baseline and rollback point: tag
`zeus-baseline-interaction-20260901` = `58a37cb` (the known-good revision the
supervisor had recorded at the start of the sprint).

The boundary this sprint repairs is the one the owner uses:

    HUMAN INTENT → SPEECH → UNDERSTANDING → ACTION → VERIFICATION → RESPONSE

Everything below was found by reading the live code path, reproduced with a
test, fixed, and re-verified.  Where a claim rests on a measurement the file
holding it is named.  Where something is the owner's to confirm (a real
microphone), it says so.

---

## 1. Ghost sentences — the causes, and the defence that replaced trust

Five independent ways a sentence nobody said could reach the transcript were
found.  None of them was a single bug; together they explain "old/random
phrases appear and ZEUS starts talking about them".

| # | source | mechanism (before) | fix |
|---|---|---|---|
| 1 | Whisper on silence/noise | `condition_on_previous_text` defaulted on, no `no_speech`/`log_prob` reading; a 2 s noise segment came back as a plausible sentence and the only gate was "≥ 2 words" | `speech/utterance.py`: audio is *measured* before the recogniser runs (RMS, peak, speech-like energy per 20 ms frame against the recording's own floor); silence/fan/keyboard are refused before Whisper is even asked; Whisper's own `no_speech_probability`, `avg_logprob`, `compression_ratio`, words-per-speech-second, language plausibility are read and each has a named check |
| 2 | the interim transcript on screen | `VoiceService.transcribe` published TRANSCRIPT *before* the gate ruled; the UI rendered it as an interim note and never removed it — a rejected hallucination stayed visible | TRANSCRIPT is published only with the verdict; the chat renders USER turns only from `user_message` events |
| 3 | replayed history | the SSE stream replays the last 100 events on connect; a page refresh (`since=0`) re-emitted old `user_message`, `message` **and `speech` events — the browser re-played ZEUS's last answers aloud** | `_stream_events` marks replay; the UI renders replayed turns dimmed under "— earlier —", never plays replayed audio, never re-opens views |
| 4 | ZEUS hearing itself | barge-in interrupted speech on every wake; audio already handed to the client kept playing; nothing knew a recording overlapped ZEUS's own voice | `VoiceService` owns speaking state (`note_spoken`, `speaking_until`, `spoken_recently`); the listener learns from `/api/voice/interrupt` whether speech was interrupted and says so with the audio (`X-Jarvis-Interrupted`); the gate rejects a transcript that overlaps playback **and** resembles what was just said (`self-echo`), while a different sentence during playback still barges in |
| 5 | microphone backlog | the listener blocks in the POST for as long as transcription takes; the device keeps recording; that backlog (often ZEUS's answer starting) was scored for the wake word and fed the next pre-roll | `CaptureLoop` requests a drain when a session ends; `WakeListener` discards `stream.read_available` before listening again |

A sixth path was closed by construction: `thought_action("tell_me_more")`
sent ZEUS's own sentence through `send_message` as an owner message.  Every
USER turn now needs provenance (`send_message` refuses a `meta.source`
outside `USER_SOURCES`); a thought carries `source=thought_inbox` /
`zeus_thought` and renders as INSIGHT.

### One utterance = one authoritative event

`UtteranceEvidence`: `utterance_id` (from the listener: `<session>-u1`),
`session_id`, `source` (`microphone` / `ui_mic`), wake score, timing, audio
evidence, device measurements, recogniser quality, raw and normalised
transcript, confidence, `speaking_overlap`.  `UtteranceLedger` refuses a
second execution of the same utterance id, the same audio bytes, or the same
envelope within 120 s.  `send_message` is idempotent by `request_id` (the UI
sends one per press; the utterance id is the request id for speech).

`tests/test_utterance_gate.py` (22 tests): silence / fan / keyboard clicks
with a recogniser that *always* returns a sentence → 0 user messages, 0
receipts, 0 missions; Whisper doubt, implausible decoding, repetition loops,
impossible word rates; replay by id / bytes / envelope; duplicate text; a
message without provenance; self-echo vs. barge-in; a thought never a user
message.

### Diagnostics

Every utterance emits one `voice_trace` DIAGNOSTIC (recorded in Activity as
`voice.accepted` / `voice.rejected`) with WAKE → AUDIO → STT → VERDICT and
every check's observation; Activity renders the chain expandable.  A
rejected utterance is its own Activity group, never filed under the previous
request.

---

## 2. STT — measured, then changed

`data/acceptance_evidence/V2_acoustic_probe_*.json` — 14 German commands
synthesised with Piper (`de_DE-thorsten`, **not the owner's voice**) plus
noise/fast/slow variants, decoded by the real faster-whisper worker with the
new options (`condition_on_previous_text=False`, single temperature,
`hallucination_silence_threshold`, bounded `hotwords`).

| model / beam | mean similarity (clean) | accepted | median latency* |
|---|---|---|---|
| base / 1 | 0.69 | 8/14 | 2.3 s |
| base / 4 | 0.89 | 11/14 | 2.4 s |
| **small / 1** | **0.96** | **13/14** (25/26 with variants) | 3.7 s |
| small / 2 | 0.94 | 14/14 | 3.7 s (but a 20 s hallucination on keyboard clicks) |

\* measured while the full test suite ran on the same CPU; absolute numbers
are pessimistic, the ranking is not.

Decisions: `small`, beam 1, one decoding pass.  The default temperature
*fallback* was re-decoding bad audio up to six times (10–15 s per silent or
noisy utterance) — it is off; the gate rejects instead.  The only miss with
`small` is "Zeus, spiel Rammstein" → "Spielramstein" (one merged token,
rejected as a fragment rather than executed wrongly).  Silence, fan and
keyboard were rejected in every configuration — before Whisper ran.

Entity hints: `_stt_hotwords()` builds a bounded (≤ 28) list from the
product vocabulary, current project titles, capability names and the
owner's learned vocabulary; it reaches Whisper as `hotwords`, never as prompt
text that could leak into the transcript.  "Stockfish" survived every
variant because of it.

### Normalisation without lying

`speech/normalize.py`: punctuation, sentence capitalisation, canonical
casing of known entities, near-miss correction only when the heard token is
≥ 5 letters, shares its first letter and is ≥ 0.8 similar to a known entity,
and the owner's heard→meant rules applied to the exact heard form.  Both
transcripts are kept (`raw_transcript`, `normalized`) with the replacement
list; the chat shows „gehört: …“ under a spoken turn when something changed.

---

## 3. Understanding — semantic purpose before domain parsers

`service/intents.py`: every request becomes one `TopIntent`
(CONVERSATION, ACTION, PROJECT_OPERATION, MISSION, SELF_DEVELOPMENT,
CORRECTION, KNOWLEDGE_OPERATION, SYSTEM_CONTROL, CLARIFICATION) and, for
deterministic operations, a typed `ActionIntent` (operation, verb,
object_type, target, arguments, constraints, success_criteria,
forbidden_effects, confidence, consequence, missing).  The existing router
(`service/routing.py`) still decides self-development / acquisition /
owner-core first; the new stage sits between it and the legacy keyword
classifier, and `core._answer` dispatches on it.

Natural project creation, without a model (`tests/test_intents.py`):

    Erstelle ein neues Projekt.                        → CLARIFICATION "Wie soll das Projekt heißen?"
    Mach mir ein Projekt für M1.                        → create M1
    Lege ein neues Projekt Biochemie an.                → create Biochemie
    Ich möchte ein neues Projekt für meinen Hausbau.    → create Hausbau
    Erstell unter ZEUS ein Teilprojekt Voice.           → create Voice, parent ZEUS
    Kannst du mir ein Projekt für Biochemie anlegen?    → create Biochemie
    Erstelle ein Projekt M1 und leg drei Aufgaben an: … → create M1 + 3 tasks
    Erstelle ein wichtiges Projekt Hausbau bis Freitag. → create, importance FOCUS, deadline Freitag
    Was ist ein Projekt?                                → CONVERSATION
    Verbessere deine Projektansicht.                    → SELF_DEVELOPMENT (router)
    Das Projekt heißt nicht Bio sondern Biochemie.      → rename Bio → Biochemie
    Zeus, öffne das Stockfish-Projekt.                  → open Stockfish
    Lösche das Projekt Test Alpha.                      → delete (IRREVERSIBLE → asks once)

Also: list/open projects and views (SYSTEM_CONTROL emits `open_view`; the
UI opens it), knowledge saves, stop, screenshots, "Nein, ich meinte X"
(CORRECTION), and "Finde den Fehler und repariere dich" (SELF_DEVELOPMENT —
the action router may not swallow it).

### Action must act

`is_action_request()` (imperative or polite/wished imperative with an
action verb, not a question about a concept).  In `_answer`, an action
request that the legacy classifier read as conversation goes to the
executor anyway; in `_answer_by_executing`, a declined plan for an action
request ends in one concise question or a plain "Das kann ich so nicht
ausführen: …", never in prose — creative requests (poem, joke, summary)
excepted.  `tests/test_actionability.py::test_an_action_request_never_degrades_into_prose`
pins six requests through the real request path.

### Confidence + consequence

`_needs_confirmation`: a *spoken* request with low speech confidence is not
executed — ZEUS asks „Ich bin nicht sicher, ob ich dich richtig verstanden
habe: „…“ – soll ich das machen?“ and „ja“ runs the parked intent; an
IRREVERSIBLE intent (delete) asks once whatever the source; medium
confidence + harmless executes.  Speech uncertainty and action safety are
separate axes in the code (`Consequence`).

---

## 4. Execution — typed, verified, concise

`service/project_ops.py`: create (title, goal, tasks, parent, importance,
deadline, description), rename, add_tasks, archive, delete.  Creation goes
through the kernel, then the record is reloaded from a **fresh store on
disk** and compared against the contract: file exists, reload by id, title
exactly as asked, listed by the Projects API, exactly the intended tasks,
parent and importance recorded.  GOAL_SATISFIED is the contract, emitted as
`goal: SATISFIED/NOT satisfied` with ACTION_EXECUTED / EXECUTION_VERIFIED /
GOAL_SATISFIED kept apart.  A second project with the same title is refused
("gibt es schon"), not duplicated.

The owner reads one sentence: „Erledigt. Projekt „M1“ ist angelegt – mit
drei Aufgaben.“ / „Das konnte ich nicht ausführen: X. Ich habe nichts
verändert.“  The receipt with every check stays in Activity and behind
"Beleg"; the chat receipt is compact (one line, checks behind „3 Prüfungen“).

The planner path (`actions.plan` → `project.create`) now routes into the
same typed executor, so model-extracted project creations get the same
verification.

`tests/test_actionability.py::test_project_create_end_to_end_with_three_tasks_and_cleanup`:
"Erstelle ein neues Projekt namens Test Alpha und leg die Aufgaben Eins,
Zwei und Drei an." → PROJECT_OPERATION, project.create, verified receipt,
exactly three tasks, visible via `list_projects`, GOAL_SATISFIED, concise
answer, no model call; the TEST-titled record is hidden from the default
galaxy and removed by the test.

### Spotify semantics

`MusicRequest.kind` (track | artist | album | playlist | top_track | any)
and `artist` from the sentence; verification checks the artist field for an
ARTIST request ("Rammstein" playing "Sonne" passes), the title (+ named
artist) for a TRACK request, either for ANY.

---

## 5. Corrections

Chat: „Nein, ich meinte Stockfish.“ after a spoken request finds the
look-alike token in the last transcript („Starkfisch“), stores a bounded
vocabulary rule (`data/jarvis/voice/vocabulary.json`, ≤ 300 entries, exact
heard form only), records an `STT_CORRECTION` in the correction memory and
re-runs the corrected request with `source=correction_rerun`.  Without a
look-alike it becomes an intent/entity correction attached to the last
receipt.

Korrigieren dialog: categories MISHEARD / WRONG_INTENT / WRONG_TARGET /
WRONG_RESULT / INCOMPLETE / PRONUNCIATION / OTHER → STT_CORRECTION /
INTENT_ERROR / ENTITY_RESOLUTION_ERROR / VERIFICATION_DEFECT /
PARAMETER_ERROR / PRONUNCIATION / OWNER_PREFERENCE.  MISHEARD learns
heard→meant from „X → Y“, „nicht X sondern Y“ or „ich meinte Y“ (looked up
in the transcript).  The protected personality and policy are never touched
from here.

---

## 6. Personality — the live prompt source

Found: the Ollama provider attached `config.system_prompt()` to **every**
chat — identity preamble + personality + the legacy engineering job sheet
("Your job is to: 1. Understand the user's goal … 4. Decide which
capabilities are required") — and the conversation prompt was sent as the
*user* message on top of it.  A 4B model read the job sheet and the
"you are a system" honesty line as a script: „Keine Wahrnehmung, kein
Gefühl. Nur Aufgaben, die du mir gibst.“

Now: `_compose_messages` splits the fixed-order conversation prompt into a
system message (identity, protected core, honesty invariants, owner
dials, corrections, task style) and a user message (transcript + the
owner's words); `OllamaBrainProvider` takes a per-call `system` that
replaces the provider default.  The job sheet stays with planners.
`persona/smalltalk.identity_answer` answers „Wer bist du?“ / „Was bist du?“
/ „Wie heißt du?“ deterministically as Zeus („Ich bin Zeus – dein
persönlicher Assistent. …“); literal consciousness/technical questions still
go to the model with the personality.  `test_conversation_sends_the_personality_as_the_system_message`
pins that "Your job is to" never reaches conversation.

---

## 7. Thoughts

A thought is never the owner's: `_say_pending_thought` delivers with
`meta.source=zeus_thought`; the inbox's "Tell me more" carries
`source=thought_inbox`; the chat renders both as INSIGHT (amber), and the
NOTIFICATION of a new thought as INSIGHT too.  Delivery stays dial-gated
(low → silent, medium → inbox, high → after an answer).

---

## 8. Project Galaxy

`ui/views/projects.js` rebuilt: layered star systems (health-tinted corona
and halo, diffraction spikes for FOCUS/ACTIVE, progress arc, BLOCKED ring,
lock/pin marks), orbit rings with subprojects / missions / capability
satellites, Knowledge as a drifting nebula, thoughts as pulses on their
relations, three parallax star layers, a sparse particle drift, eased
camera, semantic zoom (far: systems; mid: subprojects + satellites; near:
missions + thoughts), labels with collision avoidance.  Systems are laid out
on a golden-angle spiral scaled to the canvas and relaxed apart (orbit radii
included) — the whole canvas is used.  Right-click context menu: Focus,
Open, Pin, Lock, Hide, Archive, Importance, Create mission, Ask Zeus, Local
graph; importance chips (all / focus / active / blocked); legend and hint.
Owner drag persists (`OWNER_POSITIONED`), lock/release, multi-select, box
select and group move as before.  Inspector: goal, owner's words, parent,
deadline, progress, health, importance, last activity, current mission,
blockers, next action, subprojects, capabilities, thoughts, related
Knowledge, documents (artifacts), decisions, acceptance, ZEUS notes.

Backend: importance level `TEST`; legacy records whose title reads as a
test/probe default to TEST; TEST and ARCHIVED are hidden unless "show
everything"; `parent_id` hierarchy → `subproject_of` edges; project detail
carries metadata, artifacts, decisions, findings, blockers.

---

## 9. Tests

New: `tests/test_utterance_gate.py` (22), `tests/test_intents.py` (30),
`tests/test_actionability.py` (20).  Adjusted: the voice tests post
speech-like audio (`tests/_audio.py`) because the gate now measures the
samples; a legacy assertion that the project id appears in the owner's
sentence was replaced (the id lives in the receipt).

Full suite: see the final report section below.

---

## 10. Live verification (ZEUS.exe, revision b82d91f)

Process shape after a clean start: 1 core, 1 listener, 1 speech worker, 1
supervisor; owner wake model `1b540fcadd731255` loaded at threshold 0.55
(`/api/voice/wake`); restart through the supervisor 11–13 s; known-good
re-verified at `b82d91f` (2026-09-01T16:16:21Z).

`data/acceptance_evidence/V3_live_interaction.json` — through `/api/message`:

| request | outcome |
|---|---|
| same `request_id` twice | second refused `{"duplicate": true}` |
| `source: whisper_ghost` | refused: "no provenance" |
| "Zeus, wie geht es dir?" | „Mir geht's gut. Systeme laufen. Womit fange ich an?“ 0.25 s |
| "Zeus, erstelle ein neues Projekt namens Sprachtest Live und leg die Aufgaben Eins, Zwei und Drei an." | PROJECT_OPERATION → project.create, verified, 3 tasks, GOAL SATISFIED, „Erledigt. Projekt „Sprachtest Live“ ist angelegt – mit drei Aufgaben.“ 1.46 s, no model call |
| "Zeus, öffne dieses Projekt." | project.open → `open_view` projects/{id}, „Projekt „Sprachtest Live“ ist offen.“ 0.29 s |
| "Zeus, öffne meine Projekte." | project.list → view opened, the titles named, 0.44 s |
| "Das Projekt heißt nicht Sprachtest Live sondern Sprachtest Live Zwei." | project.rename, verified, 0.43 s |
| "Was ist ein Projekt?" | CONVERSATION → FAST_LOCAL answer, 4.1 s |

`data/acceptance_evidence/V3_live_voice_path.json` — real audio posted to
`/api/voice/utterance` exactly as the listener posts it (wake score, session,
utterance id, device evidence):

| audio | user messages | verdict |
|---|---|---|
| 2 s silence | 0 | rejected before Whisper: silence |
| 2 s fan noise | 0 | rejected before Whisper: no speech energy |
| 2 s keyboard clicks | 0 | rejected before Whisper: 0.24 s speech energy |
| Piper „Zeus, wie geht es dir?“ | 1 | accepted, answered, 4.9 s end to end |
| the same bytes again | 0 | rejected: replay (identical audio 4.8 s ago) |
| Piper „…erstelle ein neues Projekt namens Sprachtest Audio.“ | 1 | heard „Sprachtist Audio“ → project created (a Whisper mishear on a synthetic voice; „Nein, ich meinte Sprachtest“ now renames instead of creating twice) |
| Piper „Zeus, wer bist du?“ | 1 | „Ich bin Zeus – dein persönlicher Assistent. …“ |
| ZEUS's own last answer, synthesised, with `interrupted=speech` | 0 | rejected: self-echo 0.73 |

The live listener log during the session shows one false wake (score 0.83)
that ended `no_speech_after_wake` — nothing entered the conversation.

UI (headless Edge over CDP, no extension available): chat renders the
replayed history dimmed under „— earlier —“, spoken turns as „YOU · 🎙“
with wake tag and confidence; Projects renders the galaxy (2 owner systems
after the TEST rule, 15 missions, 5 capability satellites, Knowledge nebula,
4 hidden), Activity shows VOICE REJECTED groups and the accepted chain; **no
console errors** on either page.

Test artefacts created during verification („Sprachtest Live Zwei“,
„Sprachtist Audio“) were set to TEST + hidden, not deleted.

### Found on the way, not fixed here

The frozen supervisor hung in preflight when Ollama was not running (its
`ollama serve` `Popen` never returned; no child process appeared; 12 min).
Starting Ollama by hand and relaunching ZEUS.exe took the "already running"
path and worked.  This is in `zeus_supervisor/preflight.py`, i.e. inside the
frozen exe, and outside this sprint's scope; it is recorded here so the next
session does not rediscover it.

`tests/test_environment.py::test_the_fingerprint_keys_are_all_actually_probed`
fails on the baseline tree as well when Ollama is not reachable
(`ollama.models` never probed) — environment-dependent, not a regression.

---

## Final report

**CURRENT HEAD** `b82d91f` · **PUSHED HEAD** `b82d91f` (origin/adaptive-brain-v1) · **KNOWN GOOD** `b82d91f`, verified live 2026-09-01T16:16:21Z · **ROLLBACK** tag `zeus-baseline-interaction-20260901` = `58a37cb`.

**FULL SUITE** 1996 passed, 6 skipped, 1 xpassed, 1 failed (the pre-existing, Ollama-dependent environment test above) in 12:47.

1. *Can silence/noise still produce ghost user sentences?* — Not through any path found: silence, fan noise and keyboard clicks are refused before Whisper runs (live: 0 user messages); Whisper's own doubt, repetition and impossible word rates are refused after; the interim transcript is gone; replayed history is never a new turn. Residual: a real voice saying real words that Whisper mishears is still a *mishear*, not a ghost — and it is now visible („gehört: …“) and correctable.
2. *Can duplicate/stale transcripts execute twice?* — No: utterance id, audio bytes and envelope are refused by the ledger (live: replay refused), `request_id` makes `send_message` idempotent (live: duplicate refused), a session's microphone backlog is discarded.
3. *Can ZEUS reliably understand natural German project-creation requests?* — Yes for the paraphrase set (12 forms + tasks/parent/importance/deadline, `tests/test_intents.py`), deterministic, no model; a missing title asks one question.
4. *Does an actionable request actually cause an action?* — Yes: project operations execute typed and verified; other action requests reach the executor, a mission, one question or a plain "cannot" — never prose (`test_an_action_request_never_degrades_into_prose`).
5. *Does ZEUS know when it misunderstood speech?* — Partly: it knows when the evidence is weak (confidence level from Whisper's signals and the audio; low → asks before acting) and when a correction arrives; it cannot know a confident mishear by itself — that is what the visible raw transcript and the correction path are for.
6. *Can owner corrections improve future recognition?* — Yes: „Nein, ich meinte X“ and Korrigieren/MISHEARD store a bounded heard→meant rule that the normaliser applies to the exact heard form and the hotword list carries to Whisper; after a spoken create it renames instead of duplicating.
7. *Can ZEUS distinguish its own thoughts from owner speech?* — Yes by construction: a USER turn needs provenance; thoughts carry `zeus_thought`/`thought_inbox` and render as INSIGHT; the live self-echo test rejected ZEUS's own words.
8. *Is the personality now actually applied to FAST_LOCAL?* — Yes: the conversation prompt is the system message (verified by test that "Your job is to" never reaches it); „Wer bist du?“ answered as Zeus live.
9. *Is Project Galaxy materially more sophisticated in the real UI?* — Yes (screenshots taken live: layered star systems, orbit rings, satellites, nebula, parallax, spread layout, context menu, chips, no console errors). Visual taste is the owner's to judge.
10. *Is ZEUS now ready for owner-only microphone acceptance?* — Technically yes: the live product runs the committed revision with one listener/worker/core/supervisor, the owner wake model loaded, and the whole audio→verdict→action→answer chain exercised with real audio. The owner's spoken tests (§35 of the brief) remain the owner's — nothing acoustic with the owner's voice was claimed here.
