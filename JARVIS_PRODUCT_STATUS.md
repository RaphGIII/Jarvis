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

## THE CHESS PROOF -- ACCEPTED

Four requirements, given one at a time to a persistent project. **328 steps,
all eight acceptance criteria green, `stop_reason: ACCEPTED`.**

The end-to-end check reads a PNG, derives the FEN, and analyses it with the real
Stockfish binary:

```
PIPELINE_OK 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1 a1a8
```

`a1a8` is checkmate, verified independently.

| Requirement | Result |
|---|---|
| board geometry | **BUILD_LOCAL** -- 266 s, 13 steps, real CV, one failure diagnosed and repaired |
| position -> FEN | **escalated** -- 3 local failures, expert wrote it, Jarvis verified it |
| Stockfish | **escalated** -- 8 local failures, expert wrote it, Jarvis verified it |
| pipeline | **escalated** -- 9 local runs got most of it, expert fixed the last line |

### What BUILD_LOCAL did and did not do

It passed requirement 1 outright, including a real VERIFY-fail -> DIAGNOSE ->
repair -> VERIFY-pass cycle. `detect_board` uses greyscale thresholding, contour
detection and a bounding box, and works on both the 64px fixtures and the 80px
ones with margins, so nothing is hard-coded.

It got most of requirement 4: it wrote `pipeline.py`, imported the three real
modules, and passed the static check. What defeated it was one line -- it called
`position(path, x, y, square)` where `position(path)` takes a single argument,
having invented the signature rather than reading the module it was importing.
It then diagnosed that mismatch correctly and, four runs running, chose to
change the accepted module rather than the caller.

It never cleared requirements 2 and 3. On requirement 3 the failure is stark:
`engine.py` came out **byte-identical** after a 40-step run with a working
repair loop, having used `--usi` (the *Shogi* protocol) and invented UCI
commands, while the requirement text named `chess.engine` in as many words.

### Escalation

Each escalation followed the same path, and no step of it was a shortcut:

* the **controller** decided from counted evidence (`local pass rate 0% over N
  attempts`), not from anyone judging the model to look stuck;
* the **cost policy** was checked before a provider was chosen -- `paid_api`,
  `usage_credits`, `runpod` and browser automation all false, `is_free: true`;
* the **expert's report was never the verdict**. In both later escalations the
  expert said plainly that its own attempts to execute anything were refused by
  its permission layer, so Jarvis's independent re-run was the first actual
  execution;
* the **lesson** was stored with the failures as well as the working pattern.

The performance ledger now says something useful, including about escalation:

```
general/build_local  17 attempts, 0 passed, 0.0
general/expert        2 attempts, 2 passed, 1.0
vision/build_local    6 attempts, 0 passed, 0.0
vision/expert         3 attempts, 1 passed, 0.33
```

`vision/expert` at 0.33 is the honest part: escalation is not a magic word
either.

### Sixteen infrastructure defects, none chess-specific

The point of running this rather than reasoning about it was the defects it
surfaced. All sixteen are general, and all are fixed with regression tests.

1. An acceptance check whose answer key is reachable is not a check.
2. The project engine never learned what the capability path already knew about
   tools not being importable from generated source.
3. A missing file passed both guards vacuously.
4. The repair budget was counted over project lifetime.
5. `No module named 'engine'` read as "install a package" when it meant "write
   the file".
6. A Windows path mangled by Python's own escaping.
7. A check wired to a different requirement than the one that needed it.
8. Progress measured by `success` when `productive` was already recorded.
9. `attempts` was a lifetime counter, so a long project lost the ability to
   repair anything and reported it as considered judgement.
10. Verification recorded before an expert changed the workspace was still
    believed, so a whole failure budget went on repairing an already-fixed bug.
11. DIAGNOSE focused on the failure, making a wrong-file mistake self-reinforcing.
12. `acceptance[:8]` dropped exactly the criteria being worked on.
13. Nothing protected modules that a passing check depends on.
14. The protected-path refusal named what was forbidden, not what to do instead.
15. `reopenings` was a lifetime counter -- the third of three.
16. Reopening chose by recency, a proxy for relevance that breaks with length.

Two patterns account for most of them. Seven are **a mechanical fact left to
inference from an ambiguous signal**. Five are **a decision made from a proxy
that was easier to reach than the real signal** -- `success` for `productive`,
lifetime counts for run-length ones three times over, recency for relevance. A
proxy is right when a project is small and quietly stops being right as it grows.

### The rule this proof produced

The sharpest measurement of the four days:

* The mangled Stockfish path, unchanged across three runs, was **fixed inside
  one run** once a static check named it: *"this path contains a
  carriage-return escape"*.
* The wrong engine protocol **never moved in eight runs**, because the
  acceptance check could only report `chess.InvalidMoveError: invalid uci:
  'None'` -- a symptom three steps from the cause.

**A check that names the defect gets repaired; a check that merely fails does
not.** That is a design constraint to build to, not a defect to close. It also
settles the retrieval question honestly: the requirement text already said to
use `chess.engine` and to write the path with forward slashes. One instruction
was followed only once a check restated it mechanically; the other never was.
More prose for the model to ignore was not the missing piece.

## IN PROGRESS

- **Music capability.** Six live attempts, none passing. The bar has risen each
  time and the system now catches what it previously accepted — see limitations.


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
  and both now catch it. Attempts 3–6 fail *correctly* rather than passing
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
