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

## THE CHESS PROOF -- where it actually got to

Four requirements, given one at a time to a persistent project on BUILD_LOCAL.

| Requirement | Result |
|---|---|
| board geometry | **ACCEPTED on BUILD_LOCAL** -- 266 s, 13 steps, real CV, one failure diagnosed and repaired |
| position -> FEN | **ACCEPTED via escalation** -- failed 3x locally, expert wrote it, Jarvis verified it independently |
| Stockfish | **FAILED 6x on BUILD_LOCAL** -- see below; on the escalation path |
| pipeline | not reached |

**Requirement 1 is a genuine local pass.** `detect_board` uses greyscale
thresholding, contour detection and a bounding box, deriving square size as
width/8. It works on both the 64px fixtures and the 80px ones with margins, so
nothing is hard-coded. It included a real VERIFY-fail -> DIAGNOSE -> repair ->
VERIFY-pass cycle.

**Requirement 2 failed three times locally, and each failure taught something:**

1. Returned a hard-coded starting FEN. Caught only because there are four
   fixtures and only one is the starting position.
2. Read the answers out of `fixtures.json` -- and *passed*, because my acceptance
   command compared against the same file. **My design's defect**, identical in
   shape to the capability that returned "Dry run: would play music".
   Fixed by holding the ground truth out and adding an oracle guard.
3. Called `image_region(...)` -- a Jarvis tool -- from inside `position.py`. The
   same error the capability path hit twice with `media_folders`. The project
   engine had none of the three defences the capability path already had; it
   does now.
4. A later attempt produced no file at all while three of four criteria still
   read green, because a missing file passed both guards vacuously. Fixed.

It was then escalated, which is the architecture working as designed: the
controller decided from counted evidence, the cost policy confirmed the channel
was free, the expert worked in the project's own workspace, hit its 900 s budget
mid-tidy-up, and **Jarvis re-ran the acceptance commands itself** and found all
three green. The expert's report was not the verdict.

**Requirement 3 failed six times, and this is the honest part.**

The requirement is reachable: Stockfish 17.1 and python-chess are both installed
and working here, and a correct `analyse()` is four lines around
`chess.engine.SimpleEngine.popen_uci`, which the requirement text names
explicitly. What BUILD_LOCAL produced instead, across three separate attempts:

* the executable path written as an ordinary quoted string full of backslashes,
  so `\r` became a carriage return and `\t` a tab and the path could not
  exist -- while the requirement says in as many words to use forward slashes
  or a raw string;
* `--usi`, which is the *Shogi* protocol, driven through raw `subprocess.Popen`
  instead of the `chess.engine` it was told to use;
* invented UCI commands (`analyse`), no `uci`/`isready` handshake and no `go`,
  so no bestmove line was ever going to be printed.

Attempts 4-6 were also fighting a frozen repair loop (defect 9 below), so the
loop deserved one fair attempt after that fix before any conclusion was drawn.

**Infrastructure defects found by running this proof -- nine, none chess-specific:**

1. An acceptance check whose answer key is reachable is not a check.
2. The project engine never learned what the capability path already knew about
   tools not being importable from generated source.
3. A missing file passed both guards vacuously.
4. The repair budget was counted over project lifetime, so a persistent project
   became *less* able to fix things as it aged.
5. `No module named 'engine'` was read as "install a package" when it meant
   "write the file".
6. A Windows path mangled by Python's own escaping: right in the source,
   non-existent on disk.
7. A check that existed but was wired to a different requirement than the one
   that needed it.
8. Progress was measured by `success` (every criterion green) when `productive`
   (a criterion newly green) was already being recorded and was the right signal.
9. `attempts` was a run-length budget used as a lifetime counter. Every task
   eventually reached three attempts, reopening skips exhausted tasks, and the
   project permanently lost the ability to repair anything -- while reporting it
   as "no further repair avenue is available", which reads like judgement rather
   than a frozen counter.

Seven of the nine are one mistake wearing different clothes: **a mechanical fact
left to inference from an ambiguous signal.** The other two are wiring -- a check
that could not be seen by the loop that needed it, and a signal recorded but
never read.

**What the seventh attempt showed, once the loop could actually repair.**

Attempt 7 was the first run with a working reopen path, and the difference is
sharp enough to be worth stating as a rule.

* The mangled Stockfish path -- unchanged across attempts 4, 5 and 6 -- was
  **fixed within 25 steps**. A static check named that defect in machine-readable
  terms: "this path contains a carriage-return escape".
* The wrong engine protocol -- `--usi`, which is Shogi -- has **not moved in
  seven attempts**. The acceptance check can only say that the assertion failed,
  and what it reports is `chess.InvalidMoveError: invalid uci: 'None'`: a symptom
  three steps removed from the cause. Stockfish's own complaint never arrives,
  because the generated code pipes its stderr and then discards it.

So the loop repairs what a check *names* and flounders on what a check merely
*fails*. That is not a defect to fix so much as a design constraint to build to:
the return on writing a check that says what is wrong, rather than that
something is wrong, is most of the difference between a loop that self-repairs
and one that spins.

It also settles the retrieval question honestly. The requirement text already
said, in as many words, to use `chess.engine` and to write the path with forward
slashes. One of those instructions was followed only once a check restated it
mechanically; the other has never been followed. Handing the model more text to
ignore is not the missing piece.

**Attempt 8 removed the last confound.** Attempt 7 stopped on STEP_LIMIT, so
"exceeds capacity" would have been a conclusion drawn from a truncated run. It
was given 40 steps instead of 25, with the repair loop working and the task
reopen budget available.

`engine.py` came out **byte-identical** -- not one character changed in 40 steps
-- and the tasks ended at `reopenings=2`, the deliberate bound, so the repair
path was exercised to its limit rather than frozen. 195 steps stand on this
project.

That is the brief's threshold met without ambiguity: a working loop, an adequate
budget, eight attempts, zero movement. Classification: **model capacity**, not
retrieval, decomposition, tooling, interface contract or verification. It went
to the ExpertGateway on the same path `position.py` took.

## IN PROGRESS

- **Music capability.** Six live attempts, none passing. The bar has risen each
  time and the system now catches what it previously accepted — see limitations.
- **Guided chess project.** Two of four requirements accepted; the third has had
  seven BUILD_LOCAL attempts and is on the escalation path. See the chess proof
  section above.

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
