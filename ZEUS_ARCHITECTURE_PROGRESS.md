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

## Test status

`tests/test_routing.py tests/test_isolation.py tests/test_selfdev.py tests/test_promotion.py tests/test_music.py tests/test_corrections.py tests/test_action_receipts.py`: 262 passed.

## Performance

| | |
|---|---|
| router (first call, live) | 118.9 ms |
| mission creation after `/api/message` | 0.41 s |
| cold boot to READY (supervisor from source) | 35 s (unchanged) |

## Next highest-value task

Phase 3: desktop/supervisor lifecycle (B/C/D/E/F cases), then the self-updating release build.
