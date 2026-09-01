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
  9, voice session 12 → 47 passed. Music 90 passed (2 new).
- Full suite at the integration boundary: **2061 passed, 5 skipped,
  1 xpassed** in 17:31. Two initial failures, both resolved:
  `test_the_core_service_never_passes_unlock_from_a_non_ui_origin` exposed
  that the new security property assumed `kernel.state_root` (fixed with a
  tolerant lookup; 22/22 personality tests green), and
  `test_stockfish_returns_five_ranked_lines` is flaky under full-suite CPU
  load only — it passes standalone (3.1 s) and is untouched by this sprint.

## Live acceptance (§48–53) — measured on the RELEASED instance

- **Health**: `/api/health` → `ready: true`, revision
  `c9b5321b6881…` (= HEAD = pushed), `supervised: true`; stages http /
  fast_local / missions / voice / recogniser all ok. One ZEUS.exe, one
  ollama.exe, one core/listener/worker set.
- **Window flags live**: the relaunched shell's real command line contains
  `--autoplay-policy=no-user-gesture-required
  --disable-features=Translate,TranslateUI
  --disable-background-timer-throttling
  --disable-backgrounding-occluded-windows
  --disable-renderer-backgrounding`, and the served page carries
  `translate="no"` — the translate popup has nothing to attach to and audio
  no longer needs focus/gesture.
- **Wake pipeline**: `/api/voice/wake` on the new instance: model OWNER,
  effective threshold 0.55, listener present, `listener_match: true`
  (listener fingerprint = tested model).
- **File Galaxy backend, live**: `/api/fs/roots` → real C:\ and D:\ with
  sizes, D:\ primary; `/api/fs/list D:\` → 51 real entries, not truncated.
  Watcher round-trip on `D:\ZeusFsAccept`: watch (`mode: rdcw`) → mkdir →
  rename → rmdir produced exactly **3 debounced `fs` events** on the live
  SSE stream, then `unwatch stopped: 1`. Nothing invented; the test dir was
  created and removed for the probe.
- **Security gate, live**: `/api/auth/status` → `configured: false, locked:
  true` with all nine scopes listed — the gate is armed and waiting for the
  owner to set the password (set it in Owner → Security; a password typed by
  anyone else would lock the owner out, so this step is deliberately left to
  the owner).
- **Adaptation, live**: `/api/adaptation` → empty rule store with stats —
  ready for real feedback; rules only ever come from the owner's clicks.
- **Galaxy intact**: `/api/projects/graph` on the new build: 3 projects, 15
  missions, 5 capabilities, knowledge, self, 16 edges — the loved Projects
  view is untouched and the engine is shared, not replaced.
- **New UI shipped**: `views/files.js`, `views/personality.js`,
  `core/authgate.js`, redesigned `views/knowledge.js` all served by the
  packaged exe.
- **Owner-only checks that remain** (need a human at the microphone /
  speaker, deliberately not simulated): saying "Zeus" with the window
  unfocused/in another view/after long idle; confirming no `too_short`
  ghosts in normal speech; the eye following LISTENING→CAPTURING live; a
  spoken "Spiel …" end-to-end (the verifier fix is regression-tested against
  the exact live failure receipt); setting the owner password.

## Release

- Commit `c9b5321` (32 files, +3196/−61), pushed to
  `origin/adaptive-brain-v1`.
- Candidate `c9b5321b6881-20260901T212248Z-67d28c` built by the live core
  (PyInstaller), verified `files: ok; fingerprint: ok; size: ok; runs: ok;
  preflight: ok`, promoted → `staged` (running exe locks its dir), relaunch
  watchdog `swapped` then `healthy`.
- Known-good: `dist/ZEUS` @ revision `c9b5321…`, fingerprint
  `72ba974b79f356fd`. Rollback: `dist/ZEUS.previous` @ `30b7c7f` (the
  supervisor-reliability build) plus tag `zeus-baseline-os-20260901`.
- ZEUS left running for the owner.

## Final report — the sprint's questions, answered

1. **Wake without focus?** The root cause was the shell (autoplay policy +
   background throttling), fixed at the source; flags verified in the live
   process command line. Detection always ran out-of-window; the reaction now
   does too. Owner mic confirmation pending.
2. **`too_short` after wake?** Root-caused from live logs (wake-tail blips
   armed capture); sustained onset + 8 s re-arm budget; 7 regression tests;
   the old strict behaviour retained when the budget is 0.
3. **Canonical wake word?** The detector's verdict names the session "Zeus";
   Whisper's tail spelling is stripped by evidence (time window + low
   probability), raw and final transcripts both kept.
4. **German STT?** CPU int8 whisper-small, similarity 0.96 / median 3.7 s on
   the owner corpus. CUDA measured dead on this Pascal card (details above);
   `auto` no longer wastes ~50 s per worker start attempting it.
5. **Listening animation = backend state?** Session transitions post to
   `/api/voice/session` per frame-step and reach the UI over the same SSE
   stream the eye already renders; re-arm emits LISTENING again. Live
   <100 ms measurement needs the owner's microphone.
6. **Feedback under every answer?** Yes — restrained 👍/👎/Korrigieren row
   with 11 categories, wired to `/api/feedback` with the answer's
   request_id.
7. **Does feedback change behaviour, scoped?** Yes — bounded weights
   (≤2.0), ≈3 ratings to full effect, 60-day half-life, one rule per scope,
   context classifier keeps technical explanations apart from
   confirmations; injected into the prompt only for the matching context.
8. **Can the owner see/edit/delete learned rules?** Yes — Owner and
   Persönlichkeit views: enable/disable, delete, add own rules that outrank
   inferred ones.
9. **Password security?** scrypt (n=2^15) + unique salt, verifier wrapped
   with DPAPI, never plaintext, never in git/logs/prompts (it exists only in
   `/api/auth/*` bodies); constant-time compare, backoff, scoped short-lived
   memory-only tokens; locked on restart; only deterministic code emits
   OWNER_AUTHORIZED. 9 tests.
10. **What is gated?** Personality core/preference application, SelfDev/
    release promotion, project deletion (voice-"Ja" included), plus scope
    definitions for install/filesystem-destructive/credentials/system
    levels. Gating activates the moment the owner sets a password.
11. **Personality control centre?** The applied pipeline IDENTITÄT→KERN→
    EHRLICHKEIT→PRÄFERENZEN→ADAPTIVE→AUFGABENREGELN→EFFEKTIV with the
    literal prompt blocks in model order and a configuration inspector
    naming storage and write authority for every layer.
12. **File Galaxy without regressing Projects?** Same engine, exported with
    additive hooks only; Projects rendering untouched (graph counts
    unchanged live). Drives→folders→files with semantic zoom, real bounded
    listings, live RDCW events (3 debounced events for a create/rename/
    delete round-trip), context menu incl. Explorer/Copy Path/Ask Zeus,
    immersive mode, project↔workspace links. Visual pin/hide never moves
    anything on disk.
13. **Knowledge ≠ galaxy copy?** Depth strata (DOMÄNEN→THEMEN→KONZEPTE &
    BEFUNDE) with breadcrumb descent and search-jump; never all nodes at
    once; dark scientific plates sized by real link counts.
14. **Activity corrections?** ✎ transcript edit and 👎 per request;
    append-only JSONL, original evidence immutable, corrections feed the
    heard→meant vocabulary and can re-run the request.
15. **Spotify false failure?** Reproduced from the live receipt
    (`rcpt_b77ab7a7d2b0`), fixed by separating request intent / resolved
    target / playback state; the headline now names the resolution; wrong
    resolutions still fail. 90 music tests green.
16. **Tests?** Targeted groups first, one full suite at the boundary: 2061
    passed (chess flake passes standalone; stub-kernel fix applied and
    re-verified).
17. **Release?** Built, verified, staged-promoted, relaunched, READY at the
    pushed HEAD revision with rollback retained — evidence above. ZEUS is
    running for the owner now.
