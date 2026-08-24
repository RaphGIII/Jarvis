# Jarvis Product Status

Living document. Every "validated" claim names the measurement or test behind
it; anything not measured says so.

**Hardware:** Windows 10, GTX 1070 8 GB, older i7, Ollama pinned Pascal-compatible.
**Models:** `qwen3:4b-instruct` (FAST_LOCAL), `qwen2.5-coder:7b-instruct-q4_K_M`
(BUILD_LOCAL), `whisper-base` int8 CPU, `piper de_DE-thorsten-medium`,
`openwakeword hey_jarvis`.

**Start it:** `python -m jarvis.serve` → tokenised loopback URL
**Hands-free:** `.venv-speech\Scripts\python -m speech.listener --token <token>`
**A device:** `python -m devices.client --url <url> --pair --core-token <token>`

---

## Measured on this machine

| What | Measurement |
|---|---|
| Conversation, first token | **0.35–0.55 s** warm |
| Conversation throughput | **77 tok/s** |
| Text → first spoken audio | **1.49 s** (against a 15.7 s answer) |
| **Spoken question → first audio** | **3.16 s** warm (54.07 s cold) |
| Transcription, 2.5 s utterance | **1.90 s** warm |
| TTS realtime factor | **0.087** |
| STT realtime factor | **0.26** base / 0.79 small — base also *more* accurate |
| Wake word | **0.995** correct pronunciation, **0.000** unrelated speech |
| Wake word CPU | 3–5 ms per 80 ms frame |
| Audio meter, silence vs 0.3 tone | **0.02 → 0.30** (the playback proof) |
| Research: source ranking | docs.python.org **100** > Stack Overflow **60**, live |
| Startup warm-up | 15.7 s, background |
| UI page load | **61 ms** |
| Expert job, end to end | 244 s, independently verified |
| Stockfish 17.1 | verified (`e2e4`, +34cp) |
| Non-live test suite | **~1200 passing** |

---

## DONE and VALIDATED ON REAL HARDWARE

**Safety and cost**
- Channel-based cost policy; every metered channel off by default; a credential
  is never consent; `fallbacks_for()` structurally cannot name a metered channel.
- Expert gateway: policy checked before a provider is consulted, Jarvis re-runs
  acceptance itself, `verified` never consults the provider's report. Claude Code
  adapter proved on a real job — the expert was refused permission to run pytest,
  said so, and Jarvis' own re-run decided the outcome.
- Codex adapter written; **not verified** (CLI absent) and it says so.

**The product**
- Core as a service, no UI imports, HTTP+SSE on stdlib only, token-authed.
- Event bus a stalled client cannot wedge; replay for late clients.
- The eye: canvas, no assets, 11 states, continuous parameters.
- Local speech end to end, asymmetric phrase chunker, barge-in in 100 ms slices.
- Hands-free wake word in a separate process.
- **DeviceGateway**: per-device credentials, human-approved pairing, six-digit
  codes expiring in 5 min, tokens collectable once, revocation in isolation,
  presence separate from pairing, bounded per-device command queues.
- **Reference device client**: standard library only, proved across real
  process boundaries.
- TV/kiosk as a flag on the same document. Three clients demonstrated on one core.
- Projects view; knowledge starfield; persona and language detection.

**Knowledge**
- Ingestion: documents split at their own headings, `[[wikilinks]]` preserved,
  re-scanning updates rather than duplicates, PDFs via pypdf.
- **Natural-language graph operations** with a closed vocabulary, ambiguity that
  does nothing loudly, and confirmation gates on destructive edits.
- **Research agent**: no paid API, deterministic domain ranking, every finding
  carries a verbatim quote verified to be in the document, contradictions
  surfaced rather than resolved, honest when offline.

**Autonomous development**
- Deterministic repository navigation by AST role. Scenario A passed live in
  172 s after it, the first pass in five attempts.
- Escalation from counted evidence, never self-assessment.
- Expert memory: verified lessons only, failed approaches recorded alongside
  what worked, recalled *before* the next escalation.
- **UI self-development**: isolated worktree → health check → promote → roll
  back. `jarvis.verify_ui` catches all seven deliberate breakages tested.
- Capability acceptance: tests, contract, implemented, **static** (undefined
  names in unreached branches), and **audible** for playback capabilities.

---

## IN PROGRESS

- **Music capability.** Five live attempts. The bar has risen each time and the
  system now catches what it previously accepted — see limitations.
- **Guided chess project.** Fixtures, ground truth and Stockfish are in place;
  the project itself has not been run yet.

## NOT STARTED

- WebSocket audio transport for a remote device
- Notifications and proactive behaviour
- Unified configuration layer; boot-at-login
- Permission tiers as a first-class object

## BLOCKED

Nothing is blocked on the user.

---

## Honest limitations

- **The music capability is not finished.** Attempt 2 was reported "acquired"
  and the code was a fake — every branch returned a "Dry run:" message and
  nothing played. That is what motivated the static check and the audio meter,
  and both now catch it. Attempts 3–5 fail *correctly* rather than passing
  falsely. The bar is right; the 7B model has not yet cleared it.
- **Scenario A is 1/5 lifetime.** One pass after the navigation fix is evidence
  the fix helps, not evidence of reliability.
- **`winsound.Beep` is inaudible to the meter.** It bypasses the session mixer,
  so a capability using it would fail the audible check despite a human hearing
  it. `PlaySound` and system sounds read 0.278–0.955.
- **STT on synthetic speech is imperfect** — "mein Jarvis-Projekt" → "meinen
  Jarvisprojekts". Real microphone accuracy has still not been measured.
- **Wake word is English-pronunciation only.** "YAR-vis" scores 0.04.
- **The Codex adapter has never run.** Written against documented behaviour.
- **Chess fixtures are rendered, not photographed.** Real photographs are the
  harder problem and the honest next step.
- **One core token plus per-device tokens, loopback only.** No TLS yet.
