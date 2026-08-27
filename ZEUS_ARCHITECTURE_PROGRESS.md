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

## Phase 6 — self-updating release build — LIVE VERIFIED, chaos-tested

`deployment/release.py` (new): `ReleaseManager` with `dist/ZEUS` (known-good, the folder the owner excluded from the antivirus — never touched by ZEUS), `dist/ZEUS.previous` (rollback artifact), `dist/candidates/<rev>-<ts>-<id>/ZEUS` (complete onedir builds). `launcher_fingerprint()` hashes `zeus_supervisor/*.py` + icon + Python + PyInstaller versions; `needs_rebuild()` compares it with the fingerprint the known-good exe was built from, so a source-only promotion never rebuilds and a launcher change is never left unbuilt. `build_candidate()` → PyInstaller into the candidate dir with commit + fingerprint in `VERSION.json`; `verify_candidate()` = files, fingerprint = source, size, `ZEUS.exe --version`, `ZEUS.exe check` (the full preflight from inside the frozen program) → `VERIFIED.json`; `promote()` refuses unverified candidates, renames known-good → previous (works while the old exe runs), copies the candidate in, keeps `PROMOTED.json`; `rollback()` explicit; `sign()` documents that no code signing happens (real signing needs an owner-obtained certificate; the `signtool` step is named there). `zeus_supervisor/relaunch.py` (new, stdlib, frozen into the exe): the watchdog the supervisor spawns on a `relaunch` control request — waits for the old supervisor to exit (the instance mutex), starts the promoted exe detached, watches `/api/health` for READY from a new pid, otherwise restores `ZEUS.previous`, parks the failed release as `dist/ZEUS.failed.<ts>` and starts the restored one; receipts in `data/jarvis/supervisor/releases.jsonl`. `/api/release`, `/api/release/build` (background, then verify), `/api/release/verify`, `/api/release/promote` (+relaunch), `/api/release/rollback` (confirm required). Tests: `tests/test_release.py` (7, fake builder; watchdog rollback and healthy paths).

Live (`data/acceptance_evidence/R6_release_selfupdate_live.json`): through the running product's API, **ZEUS built its own candidate exe and verified it in 68.5 s** (`files ok; fingerprint ok; size ok; runs ok; preflight ok`), promoted it (previous kept), asked for a relaunch; the source-run supervisor stepped aside, the watchdog started `dist/ZEUS/ZEUS.exe` 0.8 s later and saw READY at `7ba351b` after 39.5 s; process table afterwards: supervisor = `ZEUS.exe`, one core/listener/worker. `dist/ZEUS/VERSION.json` now carries revision + `launcher_fingerprint`.

Chaos (`data/acceptance_evidence/C30_release_chaos_broken_exe.json`): the promoted `ZEUS.exe` replaced by garbage bytes, the watchdog run exactly as the supervisor runs it → `WinError 216` on start → previous release restored and started within **4.6 s**, failed release parked, ZEUS READY again (71.6 s, the old exe still ran the preflight generation), known-good = the previous revision. The launcher cannot be bricked by a failed build.

Found and fixed on the way: after a relaunch the core opened a second window because the early window was mid-navigation (title momentarily empty); `DesktopWindow.find` now falls back to *visible* main windows of our own profile's engine process. First version of that fallback sent WM_CLOSE to Chromium's hidden helper windows and closed the engine — only visible windows are ever acted on now (test pinned). The product now runs from the frozen `dist/ZEUS/ZEUS.exe` built by ZEUS.

## Phase 7 + 8 — the UI as one operating environment; Owner Settings — LIVE VERIFIED (DevTools-driven)

The frontend is now an ES-module graph with no build step (`ui/app.js` entry; `ui/core/{dom,api,bus,state,views}.js`; `ui/views/{chat,activity,projects,missions,knowledge,corrections,diagnostics,owner,release,capabilities,voice,palette}.js`; `ui/voice/{mic,playback}.js`; `ui/zeus.css` design system). Three modes on one page: **Presence** (large eye, readiness under it, conversation; the eye goes 372 px → 108 px on the first message), **Workspace** (rail / central pane / optional inspector, hash-routed, restorable), **Mission Control** (missions with phase, executors, isolation reports, verifier, kept diff, cancel/resume, plus the persistent HUD driven only by real `progress` events). Command palette (Ctrl+K: ask, open any view, hide window, restart, quit, build a release, reduce motion) and universal search (`/api/search` over projects, missions, capabilities, corrections, knowledge, activity, receipts). Keyboard: Ctrl+K, Ctrl+P, Ctrl+Shift+P, Ctrl+M, Ctrl+,, Esc (closes palette → drawer → graph → inspector → workspace → interrupts). Per-viewer preferences (last view, echo actions, reduced motion) in `localStorage`, wrapped in try/catch. UI errors surface as a red toast instead of a silent console line.

Views: **Activity** (day-grouped, kind filter, search, technical expansion with routing evidence, inspector), **Projects** (constellation canvas from real project/mission relations; deep view answering what/where/now/blocking/next with health derived from task status, dependency chain, timeline), **Knowledge** (list by degree, node inspector with forward links and backlinks, starfield overlay), **Korrekturen** (kinds kept apart: owner preference / owner correction / technical; inspect scope + provenance + use count; edit, disable, delete, change scope; the Korrigieren dialog now offers **„Jetzt korrigiert erneut ausführen“** and „Nur diesmal“ — `correction_save(rerun=True)` re-sends the original request through the router, which reads the correction first), **Diagnostics** (the doctor's 16 checks with remedies, resources, process counts on demand, router evidence link; the measured probe is a separate button that says its cost), **Owner** (five documents, propose → diff → confirm → audit, rollback per audit entry, protected paths; secret-shaped keys masked and not editable), **Versions** (running/known-good/previous, supervisor receipts, ZEUS.exe candidates with verify/promote/rollback), **Capabilities** (status, contract, verification strategy, Improve/Repair through the router), **Voice Studio** (settings, live meter, wake-word wizard: record 15× „Zeus“ + 10 phrases in the product → `/api/voice/wake/record`, train → `/api/voice/wake/train` (speech venv), test → `/api/voice/wake/test`).

`jarvis/verify_ui.py` now checks every module asset is present, well-formed and served, that every relative `import` resolves, and the new element ids — a broken import turns it red (`UI_BROKEN` was observed once during the work on a regex the checker could not parse). Backend added: `/api/doctor` (`service/doctor.py`, 16 deterministic checks, never probes a tier — `tests/test_doctor.py` pins that), `/api/search`, `/api/selfdev/diff`, `/api/capabilities/report`, `/api/voice/wake*`.

Live (2026-08-27 14:44Z, headless Edge driven over the DevTools protocol against the running product, `data/acceptance_evidence/UI7_cdp_report.json` + `UI7_*.png`): every view mounts with **zero console errors**; hero readiness shows UI/CORE/AI/VOICE all lit; Diagnostics reports HEALTHY (16 checks, 3.4 s); palette search for "spotify" returns the capability, the owner's own correction about the misroute, and the failed receipts; a chat message flips the page to compact mode with the eye at 108 px and the answer rendered. Browser-extension automation was unavailable (extension not connected), so the evidence is DevTools + screenshots rather than a hand-driven session.

Not done / honest limits: the knowledge graph is empty on this machine (nothing ingested yet), so Knowledge shows its empty state; the visual canvas, document reader, research workspace, decision ledger, memory inspector, notification center, human-action queue, automation view, device map, split view, snapshots and "since last visit" are **not built** — the view registry and inspector are the architecture they slot into (one file each under `ui/views/`, registered in `app.js`).

## Phase 28 — live SelfDev acceptance #3 through the new product — LIVE VERIFIED, with a defect it exposed

Request through `/api/message` (2026-08-27 14:47:38Z): *"Zeus, zeige in deiner Kopfzeile rechts neben der Uptime die Anzahl der aktuell aktiven Missionen … Die Zahl kommt aus /api/selfdev … Entwickle und teste das selbst."* Mission `f16dc926c6`: routed `self_development (high)`, candidate in an isolated worktree, **BUILD_LOCAL produced the change alone** (946 s, no expert), verify 7.9 s, promote 13.6 s → commit `5d401da`, restart READY at the promoted revision (38 s), conversation resumed. Total **988 s (16.5 min)** vs 1316 s / 2155 s for #2 / #1; isolation reports for BUILD and VERIFY: live tree unchanged.

The honest reading: the candidate was `<div id="missionCount"></div>` and nothing else — every acceptance check passed (kernel imports, the interface serves, targeted tests) and the request was **not** implemented. That is exactly the EXECUTION_VERIFIED ≠ GOAL_SATISFIED distinction of Phase 13, caught in the wild. Fixed generically (`c2470d3`): the SelfDev verifier now runs a deterministic *shape* check (a request that names data or behaviour cannot be met by markup alone) and asks FAST_LOCAL to read the diff against the request, recorded as MODEL_INFERENCE that gates escalation to the expert and never promotes on its own (`mission.verification_goal`).

## Phases 10–15, 21–24 — the durable core (merged from `dev/sprint-2`, `2a9cb11`)

- **Mission Engine** (`runtime/mission_engine.py`): one JSON per mission with goal, interpretation, constraints, acceptance criteria, evidence, hypotheses, tasks with dependencies, completed work, failed approaches, phase, next action, blockers, owner-input-required; phases UNDERSTAND … COMPLETE with checked transitions, COMPLETE only with proof; pause/resume/cancel through flags that survive saves; `brief()` is the compact summary a new process reads instead of a transcript; `mark_interrupted()` at startup. `/api/missions` lists engine + SelfDev + acquisition checkpoints in one shape; `/api/mission`, `/api/mission/{cancel,pause,resume}`. Tests: `tests/test_mission_engine.py`.
- **Evidence core** (`runtime/evidence.py`): CLAIM / OWNER_STATEMENT / MODEL_INFERENCE / TOOL_OBSERVATION / EXTERNAL_SOURCE / EXECUTION_RECEIPT / VERIFIED_FACT; `Verifier.confirm` refuses when the observer is the writer; `verdict()` answers execution_verified and goal_satisfied separately (goal = None unless a goal check or the owner says so); receipts become evidence with their checks intact.
- **Composition + general planner** (`service/composer.py`): a menu of typed primitives (built-in actions, music, `timer.start`, `note.create`, `file.open`, `knowledge.search`, `say`, `window.hide`, every registered capability from its manifest) → FAST_LOCAL returns typed steps referencing the menu by name; anything off the menu is a gap, never an instruction; device requirements checked against the device context; sequential execution, first failure stops, receipts → mission evidence; a missing primitive is named and only that is offered for acquisition. Wired in front of the single-action path for compound requests. **Live**: *"erstelle die Datei plan.txt … und starte danach einen Timer auf 1 Minute"* → composition `file.write` + `timer.start` → mission `m_974e85507c` COMPLETE in 30 s with two execution receipts (file read back by an independent check; timer registered), `plan.txt` on disk, timer fired a notification a minute later (`data/acceptance_evidence/C11_composition_live.json`). The first live attempt found two defects, both fixed: `lern\w*` matched the noun "Lernplan" (router), and a timer was refused for lack of a speaker right after a restart because the device context confused "ZEUS's voice has warmed" with "this desktop has a speaker".
- **RESEARCH route** now runs the existing `ResearchAgent` (sources, findings, contradictions) as a research mission and answers with sources and freshness.
- **Verified experience** (`development/experience.py`): after every SelfDev mission a compact entry (subsystem, goal, files that mattered, search path, failed hypotheses, what worked, tests, verifier, timings, expert use) is kept; the next mission's goal text carries the matching entries' summary lines; `compare()` reports investigate/build/total time, model calls and expert use over similar tasks. `/api/experience`.
- **Backup** (`runtime/backup.py`): zip + manifest with sha256 per file for projects, missions, corrections, capabilities, knowledge, preferences, experience, devices, receipts/activity ledgers; the secret store and credential-shaped files are excluded and listed; `verify` re-hashes; `restore` needs `confirm` and keeps the current state aside. `/api/backup{,/create,/verify,/restore}`.
- **Device context** (`runtime/device_context.py`): device id/type/name/room, inputs, outputs, screen, speaker, microphone, capabilities; owner facts via `/api/device/context/set`; the composer consults `available`.
- **Barge-in** (`speech/listener.py`): the wake word POSTs `/api/stop` before recording, so ZEUS stops speaking the moment the owner says "Zeus" (`--no-barge-in` to disable). Streaming TTS already existed (`VoiceService.speak_stream` synthesises sentence by sentence as tokens arrive). Not live-verified acoustically: the machine's microphone records silence during playback without a person.
- **Doctor** (`service/doctor.py`, `/api/doctor`): 16 deterministic checks (supervisor, core, FAST_LOCAL, BUILD_LOCAL, expert, Ollama, GPU, voice processes, duplicates, window, revision, release, capability registry, mission stores, pending rollback, isolation); never probes a tier (`tests/test_doctor.py` pins it).
- **"Learn to do X" from the chat** starts a capability-acquisition mission (engine kind `capability`) with a fresh id derived from the goal and the goal's terms as keywords — the registry's term matcher once answered a word-count goal with the Spotify provider, so acquisition never "looks up" what it was asked to build. Live acquisition #1 (`m_c2c7dd3c9e`, 15:26Z) is in BUILD_LOCAL at the time of writing; see the final report line.

## Phase 30 — failure / rollback chaos matrix

| case | how it was exercised | outcome | evidence |
|---|---|---|---|
| coding failure (candidate changes files, verifier rejects) | `tests/test_isolation.py::test_a_failing_candidate_leaves_the_live_tree_byte_identical` | live tree byte-identical, candidate released, diff kept | test |
| candidate writes into the live tree | `test_a_build_that_writes_into_the_live_tree_is_caught_and_undone` | contamination detected by byte identity, restored, mission FAILED "isolation breach" | test |
| verifier failure on a real mission | live mission `f16dc926c6`'s acceptance was too weak; now shape + goal-inference checks | fixed generically (`c2470d3`) | selfdev record |
| broken candidate startup (release) | garbage `ZEUS.exe` promoted, watchdog run as the supervisor runs it | previous release restored and started in 4.6 s | `C30_release_chaos_broken_exe.json` |
| broken commit promoted (core) | bootstrap sprint Gate O; unchanged machinery | supervisor revert to known-good in 0.5 s | `ZEUS_CORE_BOOTSTRAP_PROGRESS.md` |
| killed child worker (BUILD crashes) | `test_a_crashing_build_still_releases_the_candidate` | FAILED record, candidate released, diff kept | test |
| cancelled SelfDev | live `c9ed4e1231` cancelled at UNDERSTAND; `test_cancel_stops_the_mission_at_the_next_phase_and_releases_it` | CANCELLED, worktree released, tree clean | `R1_router_selfdev_live.json` |
| expert unavailable | `tests/test_selfdev.py` (gateway status without expert) + `Doctor._expert` warning | mission fails honestly with "no verified candidate"; startup unaffected | test |
| interrupted mission (process death mid-phase) | `MissionEngine.mark_interrupted` at warm; SelfDev missions mid-flight marked FAILED "interrupted"; promotion journal `recover_interrupted` | resumable record / restored snapshot | tests `test_mission_engine.py`, `test_promotion.py` |
| duplicate ZEUS.exe invocation | live cases B/C/D | window focused/reopened, one runtime | `L3_lifecycle_all.json` |
| half-applied promotion | `deployment/promotion.Journal` + `recover_interrupted` at startup | snapshot restored, journal says `recovered` | code + doctor `rollback` check |

Known-good runtime survived every case; no real user data was used.

## Status of the remaining numbered phases

- **16 ExpertGateway**: already an internal service — `experts/gateway.py` + `experts/claude_code.py` run the subscription CLI as a subprocess with `--permission-mode acceptEdits`, `--add-dir <worktree>` and now `--disallowedTools` for `cd`/shared-git verbs; called by SelfDev after counted local failure and by acquisition through the escalation controller; the gateway re-runs acceptance itself (`gateway.verify`); its status is a cheap check that never blocks startup (doctor `expert`). The owner launched nothing by hand in any live run of this sprint. Not built: a Codex adapter (the provider interface is where it goes).
- **17 Research**: `ResearchAgent` (queries, sources with authority, findings, contradictions, freshness) existed; the RESEARCH route now runs it as a mission and answers with sources. No per-site scraping.
- **18 Knowledge graph**: `knowledge/graph.py` (typed nodes/edges, provenance) existed; UI backlinks/forward links/local graph added; the graph is empty on this machine (nothing ingested). Missions/decisions are not yet written into it as nodes — the next step is `MissionEngine` → `KnowledgeGraph` projection (MISSION produced CAPABILITY, DECISION supersedes DECISION).
- **19 Resource arbitration**: unchanged from the bootstrap sprint (one GPU; BUILD_LOCAL never loaded at startup; diagnostics never probe). Not built: a scheduler that queues a BUILD_LOCAL call behind an interactive FAST_LOCAL turn.
- **20 Background autonomy**: pause/resume/cancel/inspect for engine missions; cancel/resume for SelfDev; conversation stays open during every mission (live: chat answered while `f16dc926c6` built).
- **23 Secrets**: `data/jarvis/secrets` (existing store) stays outside backups and out of the expert's environment (`_METERED_CREDENTIALS` scrubbed); Owner UI masks secret-shaped keys. Not done: Windows DPAPI-backed storage; the listener still passes the core token on its command line (visible in the process table) — a token file would fix it.
- **25 Self-repair**: the repair loop from the acquisition sprint is unchanged; `Doctor._capabilities` flags disabled capabilities; a `RepairMission` that runs on a detected regression (not on an owner sentence) is not built.
- **26 Owner-only boundaries**: unchanged and re-checked — no paid API, no cloud, no antivirus change (grep of this sprint's diff for `avira|exclusion|Defender` finds only prose).

## Phase 28 — live SelfDev acceptance #4 — PROMOTED, THEN REVERTED BY REVIEW (a verifier defect found and fixed)

Request (16:05:08Z): the header mission count again, now stated fully (source `/api/missions?status=active`, refresh every 15 s, empty at zero). Mission `10089dfaa4`: BUILD_LOCAL alone, 1062 s build, **the new shape check passed** (`service/core.py` + `ui/index.html`), the diff-reading inference said *"partly"*, targeted tests + acceptance green in 79 s, promoted as `e4c9cbd`, restart READY, total **1176 s (19.6 min)** — cheaper than #3 (988 s) is *not* the reading, because the candidate was **defective**: the coder replaced the line `def _answer_by_capability(...)` with `def update_mission_count(self)` (calling an unimported `requests` against a relative URL), so an existing method vanished and its body hung off the new one; nothing on the acceptance path exercised that method. The change was caught by reading the diff, not by ZEUS, and reverted (`3786f69`); ZEUS is READY at the revert.

Generic fix (`_structure_preserved` in `service/selfdev.py`): from the AST of the baseline (`git show HEAD:./<file>`) and the candidate, every function/class definition that existed must still exist unless the request asks for its removal; syntax errors are named. And the inference gate now treats *"partly"* like *"no"*: such a candidate goes to the expert for completion and is never promoted on the local tier's word. Test: `tests/test_isolation.py::test_a_candidate_that_hijacks_a_definition_is_rejected` (with the removal-asked exception). Verified-experience entries now exist for #4 (the first mission that ended under the new code); the "fewer retries with experience" measurement needs the next mission.

Efficiency across the four live self-developments: 2155 s → 1316 s → 988 s → 1176 s; expert used in #1, #2, not in #3, #4. The local tier can now land edits on this repository (it could not in #1/#2) — what it lands is not yet trustworthy without the stronger verifier above.

## Phase 29 — live capability acquisition — LIVE VERIFIED

Through the chat (16:26Z start): *"Zeus, lerne, wie man die Wörter in einer Textdatei zählt: Eingabe ist ein Dateipfad, Ausgabe die Anzahl der Wörter, Zeilen und Zeichen."* → routed `capability_acquisition (high)` → engine mission `m_c2c7dd3c9e` (kind capability) → acquisition pipeline with a fresh id derived from the goal → INVESTIGATE (4 tool calls) → DECOMPOSE (8 tasks) → EXECUTE task by task on BUILD_LOCAL → VERIFY 2/4 → DIAGNOSE/repair loop thrashing on an edit anchor (the known local-tier limit) → the 1800 s local budget ran out → **expert** → verified → registered as `learned.ausgabe_dateipfad_zeilen` v1.0.0 (active). Total **1906 s (31.8 min)**, 1 local attempt, with expert. The chat received "Gelernt und verifiziert … Ab jetzt nutze ich es direkt."

Second invocation, *"Zeus, zähle die Wörter, Zeilen und Zeichen in der Datei plan.txt."*: the first try exposed two defects, both fixed generically — the single-action planner preferred `file.read` over the learned capability (the composer now runs for a single-step request when a registered capability's keywords match), and the owner's earlier correction "notes go into notizen/" was applied to a *read* (directory overrides now apply to writes only, preventing that over-generalisation); a third defect was in the new capability step executor (it read a field the outcome does not have). After the fixes: composition chose `capability:learned.ausgabe_dateipfad_zeilen`, the workspace-relative path was resolved, the capability ran in 0.15 s and answered `word_count=1, line_count=1, character_count=8` for `plan.txt` (whose content is "Lernplan") — mission COMPLETE, one FAST_LOCAL planner call, no build. Evidence: `data/acceptance_evidence/A29_capability_acquisition_live.json`.

## Phase 31 — performance budgets (measured this session)

| item | measured | budget |
|---|---|---|
| desktop reopen (hidden → visible) | 0.038 s | ≤ 2 s |
| relaunch after X | 1.05–1.10 s | ≤ 2 s |
| cold window display (exe start → window) | 1.37 s | ≤ 2 s |
| cold boot → core HTTP | 7.7 s | ≤ 10 s (next lever: kernel construction) |
| cold boot → AI READY (model resident) | 8.7 s | ≤ 15 s; +~28 s when Ollama loads the model |
| router latency | 118.9 ms first call (registry read), sub-ms warm | ≤ 200 ms |
| simple deterministic action (file write, verified) | ~0.4 s incl. planner call | ≤ 2 s |
| composed 2-step mission | 30 s wall (incl. 25 s polling wait; steps themselves < 1 s) | — |
| capability reuse (learned capability, second invocation) | 0.15 s execution, ~2 s incl. planner | ≤ 5 s |
| mission creation after `/api/message` | 0.41 s | ≤ 1 s |
| BUILD_LOCAL wall time (SelfDev #3) | 946 s | — |
| SelfDev end to end (#1 / #2 / #3) | 2155 s / 1316 s / 988 s | — |
| capability acquisition (#1 this sprint) | 1906 s | — |
| release build + verify (PyInstaller) | 68.5 s | — |
| relaunch into a promoted exe → READY | 39.5 s | — |
| full test suite | 1781 passed, 5 skipped, 1 xpassed in 14:32 (one failure fixed afterwards; see final line) | — |

## Human actions still needed

1. **Wake-word recordings**: open Voice Studio → "Say „Zeus“ 15 times" → "10 other phrases" → Train. Everything else is in the product; only the owner's voice is missing. (The synthetic model is loaded and the listener says "listening for 'zeus'", but false activations per hour are too high for all-day use.)
2. **Acoustic check of barge-in and the voice round trip**: say "Zeus" while ZEUS speaks; the listener now POSTs `/api/stop` on the wake word. The machine's headset microphone records silence during playback, so this cannot be verified without a person.
3. **Spotify**: not re-run (Spotify was not playing); the provider capability is registered and unchanged.
4. Optional: a code-signing certificate, if SmartScreen warnings on `ZEUS.exe` matter (`ReleaseManager.sign` documents the step).

## Exact remaining blockers (engineering truth)

- **The local coder cannot yet implement most features alone.** SelfDev #3 was the first promotion by BUILD_LOCAL alone — and its candidate was one empty `<div>`. Acquisition #1 needed the expert after 30 minutes of anchor failures. The expert (subscription CLI) is used automatically and re-verified, so the owner never launches Claude by hand; but without it, self-development stops honestly at "no verified candidate". A larger local coder (the earlier measured lever) remains the single most valuable change.
- **Goal verification is still shallow.** The new shape check + model inference catches markup-only candidates; it does not prove the feature works. Feature-level acceptance (a browser check of the actual element, a test the mission writes for itself) is the next verifier step.
- **Knowledge graph is empty** on this machine; missions/decisions are not yet projected into it.
- **Voice**: owner recordings (above); streaming TTS exists, barge-in exists, neither acoustically verified here.
- **UI**: the operating-environment skeleton and 12 views exist; visual canvas, document reader, research workspace, decision ledger, memory inspector, notification center, human-action queue, automation view, device map, split view are stubs-by-architecture (registry + inspector), not built.

## A self-development nobody asked for — found during the final test run, closed structurally

While the final full suite ran, commit `546d43f` "ZEUS self-development: Erzaehl mir von deinen Faehigkeiten als Entwickler" appeared on the branch. `tests/test_action_receipts.py:610` says that sentence to a *test* core (temporary state root, stub model). The new router read "Entwickler" as the modify stem `entwickl` next to "deinen", started a SelfDev mission, the test core's runner built in a worktree of the **live** repository (`selfdev_repository()` derives from `__file__`), BUILD_LOCAL (the stub) failed, the **real expert** was called, it wrote a correct fix (describe-verbs are questions, `entwickl(?!er)`, a self-description is a registry READ), the verifier passed it and it was promoted into the live tree — all from a unit test. The expert's change is kept (reviewed; the routing tests pass with it). The structural hole is closed: `_selfdev_allowed()` — only a core whose state root lies inside the repository (the installed product) may develop the product; any other core gets "self-development is not available here" and nothing is created (`test_only_the_installed_product_develops_the_product`). Same gate on `resume_selfdev`.

## Test status

`tests/test_routing.py tests/test_isolation.py tests/test_selfdev.py tests/test_promotion.py tests/test_music.py tests/test_corrections.py tests/test_action_receipts.py`: 262 passed. `tests/test_desktop_lifecycle.py tests/test_supervisor.py tests/test_desktop_window.py tests/test_release.py`: 49 passed.

## Performance

| | |
|---|---|
| router (first call, live) | 118.9 ms |
| mission creation after `/api/message` | 0.41 s |
| cold boot to READY (supervisor from source) | 35 s (unchanged) |

## Next highest-value task

Feature-level acceptance for SelfDev (a browser check of the element the request names); mission/decision projection into the knowledge graph; a bigger local coder.

---

**Final (2026-08-27 ~16:55 local):** HEAD `133b91d` = `origin/adaptive-brain-v1`; the product runs from `dist/ZEUS/ZEUS.exe` (launcher built by ZEUS at `7ba351b`, fingerprint `489920eb5a974c3b`) with the core at `133b91d`, one core / listener / worker / supervisor / window. Full suite before the last two commits (gate + expert routing fix): 1794 passed, 5 skipped, 1 xpassed, 0 failed in 11:51; the gate and routing suites re-ran green afterwards (138 passed).
