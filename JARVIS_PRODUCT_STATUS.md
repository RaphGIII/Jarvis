# Jarvis Product Status

Living document. Every "validated" claim names the measurement or test behind
it; anything not measured says so.

**Hardware:** Windows 10, GTX 1070 8 GB, older i7, Ollama pinned Pascal-compatible.
**Models:** `qwen3:4b-instruct` (FAST_LOCAL), `qwen2.5-coder:7b-instruct-q4_K_M`
(BUILD_LOCAL), `whisper-base` int8 CPU (STT), `piper de_DE-thorsten-medium`
(TTS), `openwakeword hey_jarvis` (wake).

**Start it:** `python -m jarvis.serve` → prints a tokenised loopback URL.
**Hands-free:** `.venv-speech\Scripts\python -m speech.listener --token <token>`

---

## Measured on this machine

| What | Measurement |
|---|---|
| Conversation, first token | **0.35–0.55 s** warm |
| Conversation throughput | **77 tok/s** (qwen3:4b) |
| Text → first spoken audio | **1.49 s** (against a 15.7 s answer) |
| **Spoken question → first audio** | **3.16 s** warm (54.07 s cold) |
| Transcription, 2.5 s utterance | **1.90 s** warm (9.44 s cold) |
| TTS realtime factor | **0.087** (11× faster than playback) |
| STT realtime factor | **0.26** base / 0.79 small — base also *more* accurate |
| Wake word, correct pronunciation | **0.995**; unrelated speech **0.000** |
| Wake word CPU | 3–5 ms per 80 ms frame (~5% of one core) |
| Startup warm-up | 15.7 s, in the background |
| UI page load / status | **61 ms** |
| Expert job, end to end | 244 s, independently verified |
| Capability acquisition (music) | **736 s, acquired** — see caveat below |
| Non-live test suite | **~950 passing** |

---

## DONE and VALIDATED ON REAL HARDWARE

- **Cost policy** — channel-based, every metered channel off by default, a
  credential is never consent, and `fallbacks_for()` structurally cannot name a
  metered channel even under a permissive policy. 33 adversarial tests.
- **Expert gateway** — policy checked before a provider is consulted; Jarvis
  re-runs acceptance itself; `verified` never consults the provider's report.
  Claude Code adapter never passes `--bare` and scrubs six billing-related
  variables from the child environment. *Proved live*: the expert was refused
  permission to run pytest, said so, and Jarvis' own re-run decided the outcome.
- **Core as a service** — no UI imports, HTTP+SSE on stdlib only, token-authed,
  loopback by default.
- **Event bus** — publishing never blocks; a stalled subscriber loses its oldest
  events rather than applying backpressure; replay catches up a late client.
- **The eye** — canvas, no assets, continuous parameters eased toward per-state
  targets. All 11 states.
- **Local speech, end to end** — whisper + Piper in an isolated venv behind a
  JSON-over-stdio worker; asymmetric phrase chunker; barge-in within 100 ms.
- **Voice over the service boundary** — client captures and plays, core
  transcribes/thinks/synthesises, audio returned by reference.
- **Hands-free** — openWakeWord + energy endpointing in a separate process:
  the reference device client, arriving early because wake needed it.
- **Deterministic repository navigation** — AST role classification so real code
  outranks help text. 20 adversarial fixtures.
- **Escalation from evidence** — counted failures, *distinct* diagnoses, and a
  measured per-task-class history. No "difficulty" field for a model to fill in.
- **Desktop tools** — discovery (SAFE) separated from action (HIGH); every side
  effect supports `dry_run`; `.msi/.ps1/.vbs` never launched, `file://` refused.
- **Knowledge ingestion + starfield** — documents split at their own headings,
  the author's `[[wikilinks]]` preserved as edges, re-scanning updates rather
  than duplicates.
- **Persona and language** — one identity across backends; language detected
  without a model and sticky against short utterances.
- **Static capability checking** — catches undefined names in branches the tests
  never reach.
- **Stall/timeout hardening** — deadlines, heartbeats, `jarvis.doctor`.

## IN PROGRESS

- **Autonomous self-development (A).** Passed live in 172 s after the navigation
  fix — the first pass in five attempts. Needs three consecutive passes.
- **Capability acquisition (F).** One clean acquisition; the static check that
  the near-miss motivated has not yet been through a full live run.

## NOT STARTED

- Device gateway proper (identity, pairing, per-device revocation)
- WebSocket audio transport for a remote device
- TV / kiosk mode
- Browser / research agent
- Complex guided project proof (chess pipeline)
- UI self-modification through the development pipeline
- Notifications and proactive behaviour
- Unified configuration layer; boot-at-login
- Permission tiers as a first-class object

## BLOCKED

Nothing is blocked on the user.

---

## Honest limitations

- **Scenario A is 1/5 lifetime.** One pass after the fix is evidence the fix
  helps, not evidence of reliability.
- **The acquired music capability passed every check and was still weak.**
  Reading the generated code found a `media_control(...)` call — an undefined
  name — sitting in a branch the dry run never enters, and every branch returned
  a "Dry run:" message even when `dry_run` was false. The runtime checks were
  satisfied without music playing. `capabilities/static_check.py` now catches the
  undefined name; **"the dry run proves the real path works" is still not
  something this system can claim**, and that is the honest limit of
  dry-run-based verification for side-effecting capabilities.
- **STT on synthetic speech is imperfect** — "mein Jarvis-Projekt" came back as
  "meinen Jarvisprojekts". Vocabulary biasing fixed the worse failure
  ("Jahresprojekt"). Real microphone accuracy has still not been measured.
- **Wake word is English-pronunciation only.** "YAR-vis" scores 0.04.
- **Status is eventually consistent** — an honest health check costs a real
  generation, so the badge reads `STARTING` until the first probe lands.
- **One shared token, loopback only.** Correct for this machine, insufficient
  the moment a second device is involved.
- **Speech assets are ~1 GB** and gitignored; a fresh machine needs setup.
