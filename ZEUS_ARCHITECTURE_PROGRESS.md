# ZEUS architectural acceleration — progress

Durable state for the architecture sprint (2026-08-27). No secrets in this file.

Vocabulary as in `ZEUS_CORE_BOOTSTRAP_PROGRESS.md`: IMPLEMENTED (code exists) → TESTED (suite green) → LIVE VERIFIED (observed through the real product, checked by something other than the code under test).

## Current position

| | |
|---|---|
| CURRENT HEAD | see the last line of this file |
| KNOWN GOOD | `259d673` (`data/jarvis/supervisor/known_good.json`, written by the supervisor after a live READY) |
| VERIFIED BRANCH | `adaptive-brain-v1` |
| WIP BRANCH | `wip/lifecycle-repair-20260827` (`ab8c801`) — an interrupted lifecycle attempt; **not merged**, read for ideas only |
| HUMAN ACTION REQUIRED | wake-word recordings (unchanged from the bootstrap sprint) |

## Reality found at the start (Rule Zero)

- `adaptive-brain-v1` = `origin/adaptive-brain-v1` = `259d673`, clean. The desktop-window SelfDev (`f0e2e51863`) is that commit.
- Tag `zeus-known-good-desktop-20260827` points at `a27f934`, which is **not** on this branch; the supervisor's own known-good pointer says `259d673`. The pointer is what rollback uses.
- The failed lifecycle SelfDev (`3bc1c105a1`) **was** routed to SELF_DEVELOPMENT; the Spotify misroute happened on the *next* two attempts, whose only trace is the final Activity row: `state.error: the Spotify search failed: Spotify replied 400 Bad Request {"error": {"status": 400, "message": "Query e…`. Cause found in code: `service/intent.classify` consulted `service/music.understand` *first*, and its `_PLAY` regex matched the bare word `start` inside "Wenn ich ZEUS.exe starte"; the query extractor then took the rest of the paragraph as the track name.
- Isolation before this sprint: BUILD/VERIFY ran in a worktree (actually a `copytree` + `git init`, because ZEUS lives in `repo/Jarvis` and the engineer only used `git worktree` at a git root); the candidate's health check inherited the full environment (`JARVIS_STATE_ROOT` would have pointed it at production); the expert CLI had an unrestricted shell; nothing ever removed a worktree (141 dead candidates in `%TEMP%\jarvis_selfdev`, 4 registered in `.git/worktrees`); promotion copied files without a journal.

## Phase 1 — hierarchical top-level router — LIVE VERIFIED

`service/routing.py` (new): every request is read as an OPERATION (act / modify / learn / ask / correct / research) on an OBJECT (world / self / self_core / self_capability), scored from stems and shapes rather than phrases. `route()` returns a `TopLevelIntent` ∈ {CONVERSATION, REAL_WORLD_ACTION, SELF_DEVELOPMENT, CAPABILITY_ACQUISITION, CAPABILITY_REPAIR, OWNER_CONFIG_CHANGE, OWNER_CORRECTION, PROJECT, RESEARCH, COMPLEX_MISSION} with confidence, the reading, overruled signals (`conflicts`) and the owner corrections consulted. `service/intent.classify` now runs it first; **domain parsers (music) are only consulted when the top level is REAL_WORLD_ACTION or CONVERSATION**. Owner corrections carrying an `intent` override (`service/corrections._INTENT_WORDS`) force the route before it is chosen. Provider guard: `routing.looks_like_prose` refuses paragraphs both in `music.understand` and in `MusicService.run` (receipt `guard=prose`), and `start`/`mach … an` only mean "play" next to a music word. Core emits a `routed:` TOOL event (Activity) with the whole decision; a SelfDev mission stores it in `mission.routing`. New intents `OWNER_CONFIG` (deferred to the Owner Settings transaction, never started as a mission) and `CORRECTION` (pointed at Korrigieren).

Tests: `tests/test_routing.py` (26) incl. the two live paragraphs, object-vs-operation pairs ("Play a song" / "Improve how you choose songs", "Take a screenshot" / "Improve your screenshot function", "Open Activity" / "Change your Activity view", "Change your core personality"), forced routes from corrections, registry-name repairs, the prose guard at both layers.

Live (2026-08-27 13:30Z, product started through the supervisor from source at `259d673`+working tree): the full lifecycle paragraph via `/api/message` → Activity `routed: self_development (high) -> self_development` (self_score 11, world_score 0, `zeus.exe` among the self-references), **zero `music.*` events**, mission `c9ed4e1231` created with `routing` stored, cancelled through `/api/selfdev/cancel` at UNDERSTAND → `CANCELLED`, HEAD and `git status` unchanged. Router latency 118.9 ms on the first call (includes the registry read). Evidence: `Jarvis/data/acceptance_evidence/R1_router_selfdev_live.json`.

## Phase 2 — hard SelfDev isolation — TESTED, sweep LIVE VERIFIED

`service/isolation.py` (new):
- `CandidateWorkspace`: `git worktree add --detach` from the **git root**, always outside it, candidate ZEUS dir = `<worktree>/Jarvis`; registry in `data/jarvis/selfdev/worktrees.json`; `env()` = allow-listed environment with `PYTHONPATH` pinned to the candidate and `JARVIS_STATE_ROOT`/`JARVIS_CONFIG_ROOT` pointed *inside* it (`ZEUS_SUPERVISED` and credentials removed); `release()` keeps the diff as `data/jarvis/selfdev/evidence/<mission>.patch` then removes the worktree; `reap()` at startup collects every candidate not belonging to a kept mission.
- `LiveTreeGuard`: fingerprint of the live tree before the mission, `check()` after BUILD / VERIFY / ESCALATE. Contamination = a changed live file byte-identical to the candidate's copy; restored from git (new files deleted), the mission fails with "isolation breach", the owner's own edits are never touched.
- `SelfDevRunner`: creates the workspace at BUILD and hands it to `RepositoryEngineer.improve(worktree=…)` (new keyword; the engineer never chooses a path); `_run` uses the workspace env inside the candidate; `_finish` releases on every exit (success, failure, exception, cancel) except a verified-but-unpromoted candidate, which is kept for `/api/selfdev/resume`; `cancel_requested` persists across saves and stops the mission at the next phase boundary (and expires the engineer's deadline).
- Expert (`experts/claude_code.py`): `--disallowedTools` withholds `cd`/`pushd`/`Set-Location`, shared-state git verbs (`worktree stash reset checkout switch branch push gc config commit rebase merge clean`) and `powershell/pwsh/cmd` — the tool menu, not a sentence in the prompt.
- Promoter (`deployment/promotion.py`): per-file atomic replace; `JOURNAL.json` in the snapshot dir (`applying` → `applied` → `committed` / `rolled_back`); `recover_interrupted()` at startup restores any snapshot whose journal never reached `committed`. `is_clean` prefix bug (`str.lstrip` used as a prefix strip) fixed.
- Core: `_sweep_selfdev_isolation()` at warm (reap + recover); missions left mid-flight by a restart are marked FAILED "interrupted"; `/api/selfdev/cancel`, `/api/selfdev/resume`; SelfDev repository derived from `__file__`, never from an environment variable.

Tests: `tests/test_isolation.py` (11) — real git repo with ZEUS in a subdirectory; failing candidate → live tree byte-identical (`git status`, HEAD); a build that writes candidate bytes into production → detected, restored, mission failed, error event; crashing build → candidate released, diff kept; cancel → CANCELLED, released; `JARVIS_STATE_ROOT` pointed at production → the candidate's health command still writes only inside the candidate.

Live: startup sweep on the running product removed the 5 stale candidate worktrees (`isolation sweep: 5 stale candidate(s) removed, 0 interrupted promotion(s) restored` in Activity; `git worktree list` shows only the main tree).

## Phase 3 — desktop/supervisor lifecycle — LIVE VERIFIED (all six cases)

`service/desktop.py` (new): the window is a *managed* Chromium `--app` window, found by what it is (a top-level `Chrome_WidgetWin` window titled exactly `ZEUS`; a browser tab of the same page carries the browser's name), never by a remembered pid. `show()` focuses/restores or launches and measures time-to-visible; `hide()` hides without ending the engine; `close()` ends it; `_ensure_single` closes duplicates; a 0.25 s beacon watcher answers a second `ZEUS.exe` while the API is not up yet. `service/processes.py` (new): counts from `Win32_Process` (command line + parent), collapsing the venv launcher pair (`Scripts\python.exe` spawns the real interpreter with an identical command line) and ignoring shells whose command *string* mentions a role. `service/lifecycle.py`: `attach_desktop`, `window(show|hide|close|status)`, `process_counts`, `request_quit` (window → speech worker → listeners/workers → shutdown), `leave(final)` (a restart keeps the window, a shutdown closes it), `readiness()` (UI/CORE/AI/VOICE/FULL). `/api/window`, `/api/window/show`, `/api/window/hide`, `/api/processes`, `/api/quit`. Supervisor: a second instance POSTs `/api/window/show` (beacon fallback) and exits 0; shutdown sweeps `speech.listener`/`speech.worker`; CLI `zeus quit` / `zeus show`. `serve.py` sweeps orphaned workers/listeners before the core starts and closes the speech engine on exit. Tests: `tests/test_desktop_lifecycle.py` (11, fakes for Win32).

Live (2026-08-27 13:44Z, `data/acceptance_evidence/L3_lifecycle_all.json`, counted from the process table by a script outside the product):

| case | result |
|---|---|
| A fresh start | core 1, listener 1, worker 1, supervisor 1, window 1 (the 2/2 in the raw file is the venv launcher pair, fixed in the counter afterwards) |
| B second `ZEUS.exe` (supervisor) with window open | exit 0 in 0.70 s; window focused in 0.029 s (0.16 s end to end); same core pid; no second runtime |
| C window hidden → second invocation | visible after **1.045 s**, same core pid |
| D owner presses X | window gone in 1.0 s, core alive and READY; second invocation reopens in 1.10 s, same core |
| F `/api/restart` (self-update path) | READY in 9.6 s, **same window hwnd before and after**, one of everything |
| E "ZEUS vollständig beenden" (`/api/quit`) | port 8420 closed after 1.65 s, **zero** core/listener/worker/supervisor/window after 1.22 s |

Defect found on the way: the supervisor's stale-listener sweep matched its own PowerShell (`'*speech.listener*'` is in that process's command line) and reported "killed 1" on every boot; excluded now.

## Phase 4 — the app feels like ZEUS — LIVE VERIFIED

Title `ZEUS`; own `AppUserModelID` (`ZEUS.Desktop`, via `SHGetPropertyStoreForWindow`/`PKEY_AppUserModel_ID` — own taskbar group, not Edge's) and own icon (`ui/zeus.ico`, generated, original synthetic eye; `WM_SETICON`), both applied on the window handle after it appears (`identity: {app_id: true, icon: true}` live); no browser opens; hide/restore 0.038 s; `ZEUS_WINDOW_MODE=kiosk|fullscreen|maximized` for a TV/panel. Frontend unchanged.

## Phase 5 — startup performance — LIVE VERIFIED

Measured before: supervisor preflight ran a *real generation* before launching the core (30 s), then the core loaded the model again (28 s): window at ~33 s, READY at ~63 s. Changes: the boot preflight no longer generates (`preflight_generation: false`; the core's READY *is* a generation in the process that answers); stable checks (python, ollama binary, speech venv) come from a fingerprint cache (`data/jarvis/supervisor/preflight_cache.json`, invalidated by mtime/size of the interpreter, the venv, the config, the ollama binary); the supervisor opens the window at T0 onto its status page, which now polls `/api/health` with JS (a meta-refresh into the port hand-over gap landed on Edge's error page and produced a second window) and the core reuses that window. `/api/health.readiness` separates UI_READY / CORE_READY / AI_READY / VOICE_READY / FULL_READY; `ready` is still only a real generation.

`data/acceptance_evidence/S5_boot_timeline.json`, cold boot after a full quit, FAST_LOCAL still resident in Ollama:

| mark | before | after |
|---|---|---|
| T1 window visible | ~33 s | **1.37 s** |
| T2/T3 core HTTP, deterministic actions | ~35 s | 7.7 s |
| T4/T5 FAST_LOCAL generation ok (READY) | ~63 s | 8.7 s (+~28 s when Ollama has to load the model) |
| T6/T7 voice, FULL_READY | ~78 s | 20.3 s |
| warm reopen (hidden → visible) | n/a | 0.038 s; relaunch after X 1.05–1.10 s |

Not done: the 6 s between core launch and HTTP bind is kernel construction (imports, registries) and is the next startup lever; BUILD_LOCAL is never loaded at startup (unchanged).

## Test status

`tests/test_routing.py tests/test_isolation.py tests/test_selfdev.py tests/test_promotion.py tests/test_music.py tests/test_corrections.py tests/test_action_receipts.py`: 262 passed. `tests/test_desktop_lifecycle.py tests/test_supervisor.py tests/test_desktop_window.py`: 41 passed.

## Performance

| | |
|---|---|
| router (first call, live) | 118.9 ms |
| mission creation after `/api/message` | 0.41 s |
| cold boot to READY (supervisor from source) | 35 s (unchanged) |

## Next highest-value task

Phase 6: the self-updating release build (the supervisor/launcher changes above are only live from source until the exe is rebuilt).
