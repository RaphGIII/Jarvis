# ZEUS Owner-Experience / Adaptive-Voice / Galaxy-OS / Feedback / Security Sprint

Date: 2026-09-01 · Branch `adaptive-brain-v1` · Baseline (rollback): tag
`zeus-baseline-os-20260901` = `afbf52a`, live and READY before this sprint.

This file records what was built, what was measured, and what the evidence
is. Live sections are filled from the running ZEUS, not from unit tests.

## Voice: wake without focus, no false `too_short`, canonical wake word

- **Focus independence (root cause, fixed):** the Chromium `--app` shell ran
  with the default autoplay policy and background throttling, so audio and
  timers degraded whenever the window lost focus. `jarvis/window.py` now
  launches the shell with `--autoplay-policy=no-user-gesture-required`,
  `--disable-background-timer-throttling`,
  `--disable-backgrounding-occluded-windows`,
  `--disable-renderer-backgrounding`. Wake detection itself always ran in the
  listener process (independent of the window); what broke with focus was the
  audible/visible reaction.
- **"Diese Seite übersetzen" popup:** gone via `--disable-features=Translate,
  TranslateUI` plus `<html lang="de" translate="no">` and the `notranslate`
  meta. Only our own shell is configured; no global Chrome settings touched.
- **`too_short` ghosts (root cause, fixed):** live listener.log showed nine
  sessions dying at 0.08–0.96 s: the wake-word tail or a blip armed CAPTURING,
  then the endpointer closed on the following silence. Two changes in
  `speech/listener.py`: speech onset now needs `min_voiced_frames=3`
  sustained frames, and a short capture inside the
  `total_listen_seconds=8.0` budget **re-arms** (CAPTURING → LISTENING)
  instead of ending the session. Regression tests: `tests/test_wake_rearm.py`
  (7 tests: tail-blip survival, mid-grace re-arm, sustained onset, single
  clean end, finite budget, immediate speech unchanged).
- **Canonical wake word:** the detector decides "Zeus" was spoken; Whisper's
  spelling of the tail token ("Zeus", "Seus", "Deus"…) is stripped from the
  transcript when it sits inside the wake tail window with low probability
  (`speech/wake_segment.py`); raw and final transcripts are both preserved in
  the utterance evidence.
- **STT device (measured, decided):** CUDA on the GTX 1070 is a dead end
  today: int8/int8_float16/float16 are refused by ctranslate2 on Pascal
  ("not efficient"), float32 constructs in ~50 s and then fails at the first
  transcribe (`cublas64_12.dll` missing). CPU int8 does the German corpus at
  similarity 0.96, median 3.66 s. `speech/worker.py`: `auto` now means CPU;
  only an explicit `ZEUS_STT_DEVICE=cuda` pays a CUDA attempt (remembered on
  failure). This also removes a ~50 s worker-start penalty the auto-CUDA
  chain would have cost on every boot.
- **Wake feedback:** Voice Studio now has "✓ war Zeus / ✗ war nicht Zeus /
  ich sagte Zeus — nicht gehört" wired to `/api/feedback kind=wake`; three
  consistent reports form an insight through the adaptation store.

## Feedback → adaptation (bounded, scoped, reversible)

- 👍/👎/Korrigieren under every ZEUS answer (`ui/views/chat.js`, visually
  restrained `.fb` row), categories TOO_SHORT … BAD_PRONUNCIATION/OTHER.
- `runtime/adaptation.py`: rules carry domain/scope/weight/source/confidence;
  source priority OWNER_RULE 100 > EXPLICIT_CORRECTION 80 > EXPLICIT_RATING
  60 > REPEATED_BEHAVIOR 40 > INFERRED 20; RATING_STEP 0.34 (≈3 ratings to
  full weight), WEIGHT_LIMIT 2.0, 60-day half-life decay, one rule per scope.
  Context classifier keeps scopes apart: "technical explanations longer"
  never lengthens action confirmations (tested).
- Rules are injected into `_compose_prompt` only for the matching context;
  the owner inspects/disables/deletes them in Owner → "Gelernt aus deinem
  Feedback" and in the Persönlichkeit view; own rules can be added and
  always outrank inferred ones.
- Action feedback (`kind=action`, RESULT_WAS_SUCCESSFUL/…) feeds the verifier
  ledger; 3 consistent reports → INSIGHT with a SelfDev proposal hook.
- Tests: `tests/test_adaptation.py` (10).

## Security: Owner Security Gate

- `owner/security_gate.py`: scrypt (n=2^15, r=8, p=1) with unique salt,
  verifier wrapped with Windows DPAPI (CryptProtectData) at
  `data/jarvis/owner/auth.json`; constant-time compare; 5-failure/60 s
  backoff; scoped, short-lived, memory-only tokens (PERSONALITY_EDIT 5 min,
  SELFDEV_PROMOTE, PROJECT_DELETE, …). Locked on every restart. Only this
  deterministic module emits OWNER_AUTHORIZED; no model ever sees the
  password (it exists only in `/api/auth/*` request bodies, which are never
  logged).
- Gated operations: personality core/preferences apply, release promote,
  project delete (voice "Ja" on a delete now answers with a password prompt
  notification instead of deleting). UI: `ui/core/authgate.js` modal (manual
  typing, no autofill), needs_auth retry flow in `app.js`.
- Recovery (documented): local access + deleting auth.json resets the gate —
  no security questions, matching the threat model (protects against remote
  and model-initiated actions, not against the owner at the keyboard).
- Tests: `tests/test_security_gate.py` (9).

## File Galaxy (extends the loved Project Galaxy — same engine)

- `ui/views/projects.js` exports the Galaxy engine; additive hooks only
  (custom hue, onDoubleClick/onContext, addGraph/removeSubtree for dynamic
  sub-graphs, orbit freeze under the cursor, persistDrag switch). Projects
  rendering unchanged.
- `service/filesystem.py`: bounded real listings (os.scandir, MAX 400
  entries), TTL cache, **live** ReadDirectoryChangesW watchers (recursive,
  debounced 0.6 s, overflow → targeted rescan), metadata-only
  categorisation, explorer.exe opening with path validation. Nothing is ever
  invented; missing paths are errors.
- `ui/views/files.js`: drives → folders → subfolders → files with semantic
  zoom (double-click enters, deep zoom expands in place, zooming out
  collapses), breadcrumbs, camera memory per root, category legend, pins and
  visual-only hiding, context menu (Enter Galaxy / Explorer / Copy Path /
  Pin / Hide / Inspect / Ask Zeus), project↔folder links from real
  workspaces, immersive full-screen mode. D:\ is primary.
- Performance (measured, this machine): drives 21.6 ms; `D:\` cold 763 ms /
  warm 0.03 ms; `C:\` cold 8.5 ms; live watcher events arrive debounced
  (create/rename/delete each seen in `tests/test_filesystem.py`, 9 tests,
  including a real RDCW round-trip and burst batching ≤6 events for 30
  files).

## Knowledge redesign (depth, not a galaxy)

- `ui/views/knowledge.js`: strata — DOMÄNEN (most-connected hubs, max 9) →
  THEMEN (neighbours) → KONZEPTE & BEFUNDE (findings/lessons/decisions at
  the bottom), breadcrumb descent, search jumps, plate size/degree bars from
  real link counts, dark scientific look (`.kboard/.kplate`). Never all
  nodes at once; the starfield overlay remains available.

## Persönlichkeit control centre

- `ui/views/personality.js`: the applied pipeline IDENTITÄT → KERN →
  EHRLICHKEIT → PRÄFERENZEN → ADAPTIVE REGELN → AUFGABENREGELN → EFFEKTIV,
  each stage clickable; protected core read-only with password-gated editing
  via Owner; adaptive rule editor; the literal prompt blocks in model order;
  a configuration inspector naming where each layer lives and who may write
  it; security panel embedded.

## Activity: corrections without rewriting history

- `ui/views/activity.js`: ✎ transcript editor and 👎 per request. Backend
  `activity_correct` appends to `data/jarvis/owner/activity_corrections.jsonl`
  (append-only), original evidence immutable, corrections feed the
  vocabulary (heard→meant) and optionally re-run the request.

## Media: intent vs resolution vs playback (live bug, fixed)

- Live receipt `rcpt_b77ab7a7d2b0` (today 20:08): owner asked "Rammstein
  ohne mich", provider resolved to the track "Ohne dich" (Rammstein,
  track_id 4aFC7Mes…), Windows reported it Playing — and the old title-only
  check still called it a failure. `service/music.py` now separates the
  three facts: a "the resolved track is playing" verification (provider
  output vs Windows), an intent check that lets query words match title OR
  artist when no artist was named explicitly, and a headline that says
  "playing: Ohne dich - Rammstein (resolved from 'Rammstein ohne mich')".
  Wrong resolutions still fail (regression test). `tests/test_music.py`: 90
  pass including 2 new.

## Test state

- Targeted groups: wake re-arm 7, adaptation 10, security gate 9, filesystem
  9, voice session 12 → 47 passed. Music 90 passed.
- Full suite at the integration boundary: (filled in below when the run
  completes).

## Live acceptance (§48–53)

(to be filled from the running, newly released ZEUS)

## Release

(to be filled: commit, push, promote, restart, revision match)
