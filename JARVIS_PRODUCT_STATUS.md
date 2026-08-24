# Jarvis Product Status

Living document. Updated as subsystems land. Every "validated" claim below names
the measurement or the test that supports it; anything not measured says so.

**Hardware this is measured on:** Windows 10, GTX 1070 8 GB, older i7, 
Ollama pinned to a Pascal-compatible build.
**Local models:** `qwen3:4b-instruct` (FAST_LOCAL), `qwen2.5-coder:7b-instruct-q4_K_M` (BUILD_LOCAL),
`whisper-base` int8 CPU (STT), `piper de_DE-thorsten-medium` (TTS).

**Start it:** `python -m jarvis.serve` → prints a tokenised loopback URL.

---

## Measured on this machine

| What | Measurement | Where |
|---|---|---|
| Conversation, first token | **0.35–0.55 s** warm | `brain/ollama.py` `generate_stream` |
| Conversation throughput | **77 tok/s** (qwen3:4b) | same |
| Voice: first token | **0.47 s** | `speech/pipeline.py` metrics |
| Voice: first phrase ready | **0.97 s** | same |
| **Voice: first audio out** | **1.49 s** (answer was 15.7 s long) | same |
| TTS realtime factor | **0.087** (11× faster than playback) | Piper, CPU |
| STT realtime factor | **0.26** (whisper-base), 0.79 (small) | measured comparison |
| UI page load / status | **61 ms** | `service/core.py` background probes |
| Expert job, end to end | 244 s, independently verified | `experts/` real run |
| **Spoken question -> first audio** | **3.16 s** warm (54.07 s cold) | full HTTP chain |
| Transcription of a 2.5 s utterance | **1.90 s** warm (9.44 s cold) | same |
| Startup warm-up | 15.7 s, in the background | `JarvisCore.warm()` |
| Non-live test suite | **822 passing** | `pytest tests/` |

---

## DONE and VALIDATED ON REAL HARDWARE

- **Cost policy.** Channel-based (`local` / `subscription_cli` / `paid_api` /
  `usage_credits` / `runpod` / `browser_ai_automation`). Every metered channel
  off by default; `require()` raises rather than returning false. A credential in
  the environment is never treated as consent. Quota exhaustion cannot reach a
  metered channel — `fallbacks_for()` cannot name one even under a permissive
  policy. 33 tests, adversarial.
- **Expert gateway.** Policy checked before a provider is consulted; acceptance
  commands re-run by Jarvis afterwards; `verified` consults that evidence and
  never the provider's own report. Claude Code adapter built against the
  installed CLI (2.1.241), never passes `--bare` (it forces API-key auth), and
  scrubs `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_BASE_URL`/Bedrock/
  Vertex/`OPENAI_API_KEY` from the child environment.
  *Proved on a real job*: the expert was refused permission to run pytest itself,
  said so, and Jarvis' independent re-run (`exit=0`) established the result.
- **Core as a service.** `JarvisCore` owns state and events and imports no UI.
  HTTP + SSE over stdlib only (no dependencies added). Token-authenticated,
  loopback by default, warns when bound wider.
- **Event bus.** Publishing never blocks; a subscriber that stops draining loses
  its oldest events rather than applying backpressure — a paused browser tab
  cannot freeze an autonomous build. Replay buffer means a page opened
  mid-mission renders the truth.
- **The eye.** Canvas, no assets, no dependencies. Continuous parameters eased
  toward per-state targets, so transitions are one entity changing behaviour
  rather than clips cutting. All 11 required states.
- **Streaming generation.** `generate_stream` on the Ollama provider; request
  body shared with the blocking path so the two cannot disagree about `num_ctx`.
- **Local speech.** Whisper + Piper in an isolated venv behind a JSON-over-stdio
  worker. Phrase chunker with abbreviation/decimal/list-marker handling and an
  asymmetric first-phrase rule. Barge-in stops audio within 100 ms slices.
- **Deterministic repository navigation.** AST role classification
  (`DOCUMENTATION < CONSTANT_TEXT < MODULE_CODE < FUNCTION_CODE < FUNCTION_DATA
  < CONTROL_FLOW`) so real code outranks help text mentioning the same words.
  20 adversarial fixtures where plain search picks wrong by construction.
- **Stall/timeout hardening.** Deadlines, heartbeats, `python -m jarvis.doctor`,
  hard wall-clock limits. A timed-out call becomes evidence for DIAGNOSE.

- **Voice over the service boundary.** The client captures and plays; the core
  transcribes, thinks and synthesises. Audio is posted as raw bytes and returned
  by reference (a bounded store) rather than base64 inside the event stream.
  Speaking to Jarvis enters voice mode; typing alone never makes it talk back.
  One generation feeds both the screen and the speaker, so they cannot drift.
  *Measured end to end over HTTP*: spoken question in, first audio out in 3.16 s.

## DONE, not yet validated end to end on hardware

- **Diagnosis-repetition escalation.** Fingerprinted diagnoses quoted back to the
  model so a retry is a genuinely different request. Unit-tested; the live
  capability run that motivated it has not been repeated since.
- **Unified capability acceptance.** `capability_checks()` is now the single
  definition the loop is graded on and the verifier re-runs independently.
  Unit-tested; needs a live F run to confirm the pass rate moves.
- **Knowledge graph export/inspect** (`export()`, `node_detail()`) — tested, but
  no UI consumes it yet.

## IN PROGRESS

- **Autonomous reliability.** Scenario A passed live in 172 s after the
  navigation fix — the first pass in five attempts. Needs the three consecutive
  passes the brief requires. E and F need re-measuring against the new code.

## NOT STARTED

- Wake word, VAD, continuous conversation mode, push-to-talk, mic privacy toggle
- Escalation controller (complexity signals, historical performance)
- Expert experience memory (problem class → successful architecture → reuse)
- Knowledge graph UI (starfield), file/note/PDF ingestion
- Projects view in the UI
- Desktop/file/media tool pack; music capability
- Browser/research agent
- Persona system as a first-class configurable object (currently one prompt)
- Multilingual switching and persistence (the model answers in-language already)
- Device gateway, reference client, TV mode
- Notifications, configuration layer, boot/persistence, permission tiers
- Complex guided project proof; UI self-modification

## BLOCKED

Nothing is blocked on the user.

---

## Honest limitations

- **Scenario A is 1/5 lifetime**, and the one pass came after the navigation
  fix. One pass is evidence the fix helps, not evidence of reliability.
- **STT accuracy on synthetic speech is imperfect.** A Piper→Whisper round trip
  of "mein Jarvis-Projekt" came back as "meinen Jarvisprojekts" — the words are
  right, the grammar is not. Feeding whisper an `initial_prompt` of domain
  vocabulary fixed the worse failure ("Jahresprojekt"). Real microphone input is
  the case that matters and has still not been measured.
- **Status is eventually-consistent.** Health is probed in the background
  because an honest check costs a real generation (~80 s cold); the badge shows
  `STARTING` until the first probe lands.
- **No wake word yet**, so voice is press-to-record rather than hands-free.
- **Cold start is 15.7 s.** Warming runs in the background at startup, so the
  cost is only visible if the first question arrives within that window.
- The speech venv, whisper models and Piper voices total ~1 GB and are
  gitignored — a fresh machine needs the setup step.
