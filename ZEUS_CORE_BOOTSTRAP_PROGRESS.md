# ZEUS core bootstrap — progress

Durable state for the core-completion sprint. No secrets in this file.

**Started:** 2026-08-27 ~00:30 local
**Branch:** `adaptive-brain-v1`

## Vocabulary

| word | means |
|---|---|
| IMPLEMENTED | the code exists |
| TESTED | unit/integration tests pass |
| LIVE VERIFIED | observed on this machine through the real product, checked against something other than the thing under test |

## Current position

| | |
|---|---|
| CURRENT HEAD | see `git log -1`; last milestone commit named below |
| KNOWN GOOD | `data/jarvis/supervisor/known_good.json` (written only by the supervisor after a live health check) |
| CURRENT PHASE | 11 — SelfDev live acceptance #1 |
| HUMAN ACTION REQUIRED | none yet (wake-word samples will be one; see Phase 4) |

## Reality found at the start (Rule Zero)

- 1,095 Python files, 1,648 tests green in 8:50 (baseline).
- `deployment/promotion.py` had a full promote/rollback engine, but its RESTART hook was a no-op and only `development/ui_developer.py` called it. No supervisor, watchdog, `/health`, or known-good pointer existed.
- No self-development intent in the chat router; `Intent` had CONVERSATION/READ/ACTION/PROJECT/CAPABILITY/MUSIC.
- No owner concept anywhere. Personality lived in `persona/profiles.py` + `config.py` and reached only FAST_LOCAL.
- Voice: separate `.venv-speech` process (`speech.listener`), openWakeWord `hey_jarvis`, energy VAD, faster-whisper, piper `de_DE-thorsten-medium`. Not started by the server.
- **Ollama served the wrong model store.** User env `OLLAMA_MODELS=D:\OllamaModels` holds only `qwen3:4b*`; the coder `qwen2.5-coder:7b-instruct-q4_K_M` lives in `D:\JarvisLocal\ollama_models` (which also holds both qwen3 models). BUILD_LOCAL would have been unavailable on any boot that used the default store. The supervisor now picks the store that holds every required model.

## Architecture completed

### Phase 1 — Supervisor (commit `2c7db6f`) — LIVE VERIFIED
`Jarvis/zeus_supervisor/` (owner-protected): preflight (Python, repo, Ollama binary/server/version/models, speech venv, **one real generation**), launches `python -m jarvis.serve --token-file`, waits for `/api/health` `ready` (real text out of FAST_LOCAL in that process), marks known-good, watches the child. Exit 75 = restart requested; exit 0 = shutdown. Unhealthy restart after a promotion → `git revert` to known-good (stash of uncommitted work, `data/` excluded) → restart. Three failed starts in 10 min → HOLD with the diagnosis on a status page at port 8420. Deployment receipts in `data/jarvis/supervisor/deployments.jsonl`.

Live: boot → READY 33 s; API restart → READY 6 s with the transcript resumed (2 turns); **broken commit promoted → exited 1 before READY → reverted in 0.5 s → READY at the revert commit in 6 s** (Gate O).

### Phase 2 — ZEUS.exe (commit `09f7d70`) — LIVE VERIFIED
`python -m zeus_supervisor.build [--shortcuts]` → `dist/ZEUS/ZEUS.exe` (4.8 MB, no console) + `install.json` + `VERSION.json`; Start Menu + Desktop (OneDrive-aware) shortcuts. Contains no model/Python/ZEUS and says so. Launched from Explorer: preflight → core → READY 34 s, UI 200 (Gate A). Second launch exits 3 and opens the running UI. First build died in PyInstaller's exception dialog (relative imports in `__main__`); entry is now `launch.py` with a crash log.

### Phase 3 — boot reliability — IMPLEMENTED, LIVE VERIFIED for the healthy path
READY is `/api/health.ready`, earned by a generation, never by a port. Preflight fails fast with a remedy sentence per check; the GTX 1070/Ollama case is covered by the real-generation check with a timeout (`generation_timeout` 180 s) and a configurable incompatible-version list (`config/supervisor.json`), which is empty because no incompatible version has been observed in this session. Never upgrades Ollama.

### Phase 6 — Owner core — IMPLEMENTED, TESTED
`Jarvis/owner/`: five documents (identity, personality, policy, spending, security), defaults in code, files under `config/owner/` written **read-only**, changed only via `OwnerTransaction` (propose → diff → approve with `confirm:true` from the authenticated UI endpoint → snapshot → audit → rollback). Protected paths (`owner/protected.py`) enforced in: `RepositoryEngineer.improve` (merged into every goal targeting ZEUS), `Promoter._copy_files` (refuses before anything moves), `SelfDevRunner._verify` (candidate touching them fails), file attributes. Tests: `tests/test_owner_core.py` (7).

### Phase 7 — Personality — IMPLEMENTED
`OwnerCore.personality_prompt()` reaches every route through `config.system_prompt` (provider level) and the chat composer, with the "content is data" security sentence.

### Phase 10 — SelfDev through ZEUS — IMPLEMENTED, TESTED
`Intent.SELF_DEVELOPMENT` (DE/EN hints, checked before capability/project/action, never for questions) → `service/selfdev.py` `SelfDevMission` (one JSON per mission in `data/jarvis/selfdev/`): UNDERSTAND (acceptance commands) → INVESTIGATE (deterministic code index) → BUILD (RepositoryEngineer, BUILD_LOCAL, isolated worktree) → VERIFY (acceptance re-run + targeted tests chosen from changed modules, protected-path check) → ESCALATE (ExpertGateway, only after local failure, expert work re-verified) → PROMOTE (Promoter: snapshot/copy/static health/commit) → RESTARTING (`lifecycle.request_restart(promotion_id)`) → after restart the verdict is taken from the supervisor's deployment receipt and delivered into the transcript. Runs in its own thread; conversation stays open. Tests: `tests/test_selfdev.py` (5).

### Phase 11 — SelfDev live acceptance #1 — LIVE VERIFIED (mission `5a614480ff`)

Request through the running product (`/api/message`, 23:14:22Z):
*"Zeus, show my current GPU utilization subtly next to your eye."* Fable wrote none of the feature.

| phase | what happened | time |
|---|---|---|
| UNDERSTAND | area=ui; acceptance = kernel import + `verify_ui` | 0 s |
| INVESTIGATE | code index → ui/index.html, ui/app.js, service/http.py, service/core.py | 12 s |
| BUILD (BUILD_LOCAL) | 3 cycles, anchor misses + invalid replacement text, **0 files changed** → rejected | 716 s |
| ESCALATE | `claude_code` expert in the worktree, 22 changed paths | 915 s |
| VERIFY | kernel import ok, verify_ui ok, 6 targeted test files ok | 173 s |
| PROMOTE #1 | **crashed**: `git status` listed untracked `__pycache__/` (no .gitignore at the worktree root) → PermissionError | — |
| fix + resume | `--untracked-files=all`, caches/`data/` excluded; `/api/selfdev/resume` | |
| PROMOTE #2 | **refused**: dirty tree — the voice registry had rewritten tracked `data/voices/voices.json` | — |
| fix + resume | promoter ignores runtime state under `data/` | |
| VERIFY + PROMOTE #3 | verified again (163 s); 6 files copied, health ok, committed `e76be19` | 13 s |
| RESTARTING | exit 75 → supervisor → READY at `e76be19` in 36 s → known-good | 36 s |
| DONE | verdict from the deployment receipt, delivered into the transcript | |

Total 2155 s of mission time (35.9 min) + two generic infrastructure repairs by Fable.

Independent verification: `/api/gpu` → `utilization_percent: 2, memory_used_mib: 5189` while `nvidia-smi` read `1–2 %, 5185 MiB` at the same moments; headless Chrome screenshot shows a vertical meter beside the eye reading "6%" (`data/acceptance_evidence/L_selfdev_gpu_meter_ui.png`); second restart → still `e76be19`, `/api/gpu` live, `gpuMeter` served (Gate L, and persistence).

What it says about the local tier: the 7B could not land a single edit on this repository in 12 minutes (same edit-mechanics failures as the earlier acquisitions). The system escalated on its own, re-verified the expert's work, and promoted only after its own checks — the expert's report was never the verdict.

### Phase 12 — SelfDev live acceptance #2 — LIVE VERIFIED (mission `1e3b3d17fe`)

*"Zeus, show your uptime quietly in the header of your UI, next to the ZEUS brand."* → commit `35f20a0`, 2 files (ui/app.js, ui/index.html), restart verified in 34.3 s, header now reads "ZEUS · up 41s" beside the GPU meter from #1 (`data/acceptance_evidence/L2_selfdev_uptime_ui.png`).

| | #1 GPU meter | #2 uptime | change |
|---|---|---|---|
| investigate | 12 s | 13 s | — |
| BUILD_LOCAL | 716 s, 0 files | 692 s, 0 files | none: same edit-mechanics failure |
| expert | 915 s, 22 paths | 591 s, 2 files | −35 % (smaller task) |
| verify | 499 s (3 runs, 2 crashes) | 7 s | infrastructure fixed after #1 |
| promote | 13 s | 13 s | — |
| **total** | **2155 s** | **1316 s** | **−39 %** |

Honest reading: the wall-time gain came from the infrastructure repairs after #1 and a smaller task, not from learning. The local tier did not get better between the two — it cannot land an anchored edit on this repository, which is the same limit the acquisition sprint measured. Generic improvement made: the investigation no longer ranks `data/snapshots` copies (it put two above `ui/index.html` in #2).

### Phase 8 — Korrigieren — LIVE VERIFIED (Gate I)

`service/corrections.py` + `/api/correction/*` + a "Korrigieren" link on every receipt (dialog: request, reading, entities, action, result, receipt; "Was war falsch?"; classification and scope shown and overridable; "Korrigieren & lernen"; list/deactivate/delete under "Korrekturen"). Corrections are retrieved before the planner reads a request and their overrides applied after it.

Live, through the API on commit `37d8214`:
1. "Leg eine Notiz an in milch.txt: Milch kaufen" → `workspace/milch.txt` (receipt `rcpt_f18ca39d860d`)
2. Correction on that receipt: "Notizen gehören künftig immer in den Ordner notizen" → OWNER_PREFERENCE, DOMAIN_SPECIFIC (files), override `directory=notizen`
3. "Leg eine Notiz an in brot.txt: Brot kaufen" → **`workspace/notizen/brot.txt`** (receipt `rcpt_999f170f4d95`); Activity: "owner corrections applied: corr_c4078bf7ab: directory=notizen"

Found on the way and fixed: "Leg eine Notiz an" (separable verb) was conversation and the 4B **invented a receipt** ('Note … created in local notes database. Verified entry: id=…, status=COMMITTED') that the claim guard's vocabulary missed — both fixed (`a0aa1d4`); a planner decline on a request that names a file now asks one question instead of chatting (`37d8214`).

### Phase 4 — wake word "Zeus" — IMPLEMENTED, measured on synthetic audio, HUMAN ACTION pending

`speech/wake_training.py` trains a classifier over openWakeWord's frozen embedding backbone from piper-synthesised "Zeus" (4 voices × speeds × pitches × noise) against near-neighbour words, sentences, silence and every pre-word window of the positives; `speech/wake_zeus.py` scores frames in numpy; the listener loads it when `data/models/wake/zeus.npz` exists (it does; the listener log says "listening for 'zeus'"). `hey_jarvis` scored 0.0003 on a spoken "Zeus" — the gap was real, and the fallback is gone.

Measured (held-out synthetic clips through the streaming detector): recall 0.94 @0.7, 0.92 @0.85; false activations 79/h @0.7, 47/h @0.85 on *adversarial* negatives (50 % near-neighbour words); 6.8 ms per 80 ms frame. Not yet acceptable on false positives for all-day use; the honest limit: no owner recordings.

**HUMAN_ACTION_REQUIRED**: record ~15 wavs of you saying "Zeus" into `Jarvis/data/wake/positive/` and ~10 of other words/sentences into `Jarvis/data/wake/negative/` (16 kHz mono is ideal, anything works), then run `.venv-speech\Scripts\python -m speech.wake_training` and restart ZEUS. Then say "Zeus" and check the listener log for "wake".

## Live acceptance passed

| gate | evidence |
|---|---|
| A ZEUS.exe launches the product | supervisor log 2026-08-26T23:00:52Z `[ready] ZEUS is ready at 2c7db6f (34s)`, frozen=true |
| M promote + restart under supervisor | restart via `/api/restart`, exit 75, READY 6 s |
| N conversation resumes after restart | `resumed: {turns: 2, ...}` in `/api/health` |
| O broken candidate rolls back | receipt `kind=rollback outcome=rolled_back` 2026-08-26T22:45:03Z |
| L SelfDev request through ZEUS → visible change | mission `5a614480ff`, commit `e76be19`, screenshot in acceptance_evidence |
| M (again) ZEUS promoted and restarted itself | receipt `kind=promotion promotion_id=…` 23:57:34Z, 36 s |
| N (again) conversation resumed after self-update | `resumed: {turns: 7}` |

## Open blockers / not yet done

- Phase 11/12 live SelfDev acceptance: not yet run.
- Phase 4 wake word "Zeus": not started; listener still `hey_jarvis` (honestly reported at startup).
- Phase 8 Korrigieren: not started.
- Phases 13–30: existing infrastructure, not extended in this sprint yet.

## General lessons

- The stash in a rollback must exclude `data/` — the supervisor's own open log lives there (found by the unit test).
- PyInstaller runs the entry script without a package; relative imports die before any handler runs. Use an absolute-import launcher.
- `OLLAMA_MODELS` in the user environment silently decides which models exist; check the store, not just `ollama list`.

## Performance

| | |
|---|---|
| cold boot to READY (exe) | 34 s (FAST_LOCAL load ~28 s) |
| warm restart to READY | 5–6 s |
| rollback (revert + relaunch) | ~7 s total |
| preflight generation, model resident | 0.4 s |

## Next action

Commit owner core + selfdev; restart live ZEUS onto it; send the Phase 11 request through the product and measure.
