# ZEUS — supervisor/Ollama boot reliability (2026-09-01, follow-up sprint)

Scope: the frozen supervisor's Ollama preflight hang, and everything the live
verification of the fix surfaced.  Voice/intent/execution/personality/galaxy
systems from the interaction sprint were not touched.

## The original hang, exactly

`dist/ZEUS/ZEUS-error.log` (from the live incident):

    File "zeus_supervisor\preflight.py", line 279, in _check_ollama_server
        [self._ollama_exe, "serve"], env=env,
    AttributeError: 'Preflight' object has no attribute '_ollama_exe'

Two defects composed:

1. the `ollama.binary` check stored the found path as a **side effect**
   (`self._ollama_exe = found`); when that check came from the preflight
   *fingerprint cache*, the side effect never ran, and with Ollama not
   already running the server check raised;
2. the frozen crash handler showed a **modal `MessageBoxW`** with no console
   — the process sat at an invisible dialog with the instance lock already
   released, indistinguishable from a hang (observed 12+ min), and a second
   launch could then start a stray core that held port 8420.

(Contrary to the initial report, the old `Popen`/poll loop itself was
bounded; the hang was the crash path around it.  `ollama serve` is now
handled as a long-running service anyway — see below.)

## The new lifecycle (`zeus_supervisor/ollama.py`)

`OllamaService` owns Ollama for the life of the supervisor:

* **states** RUNNING / STARTING / UNAVAILABLE / FAILED / MISSING, each with
  the actual reason, in `status.json`, the preflight report and the doctor;
* **detached spawn** (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
  CREATE_NO_WINDOW`, no inherited pipes, `OLLAMA_MODELS` from the configured
  store — `D:\JarvisLocal\ollama_models` discovered as before); never
  `wait()`/`communicate()`; the handle is kept only for the pid and to
  notice an early death;
* **readiness = the HTTP API**, polled within `ollama_start_timeout`
  (45 s default);
* **idempotent**: an answering API is left alone; an existing `ollama`
  process (tray app, previous supervisor, slow start) is waited for, never
  duplicated (`tasklist`, bytes decoded by hand — see below); one spawn at a
  time behind a lock;
* **storm-proof**: 30 s cool-down between spawns, at most 3 spawns per
  10 min, then FAILED with "run `ollama serve` and read its error";
* `ollama serve`'s own output goes to
  `data/jarvis/supervisor/logs/ollama.log` (rotated), so a server that
  cannot load a model leaves its reason somewhere readable.

Boot: a failed preflight **holds and retries every 30 s** — the status page
answers, `POST /api/quit` (with the token) or a control-file shutdown ends
it, and the boot resumes by itself when the checks pass.  Running: the
supervisor checks `/api/version` every 15 s (no model is touched) and
recovers a dead Ollama in one background thread within the same budget.
Diagnostics (`service/doctor.py`) observe with `start_services=False` and
never spawn.  Every preflight check is wrapped: an exception is a failed
check with the text, never a crash; the crash dialog itself now uses
`MessageBoxTimeoutW` (30 s).

## What the live verification surfaced (and fixed)

1. **`tasklist` broke Unicode**: with no `ollama.exe` present it prints a
   localised OEM message (0x81 = "ü" in cp850); `text=True` under cp1252
   killed the reader thread, `stdout` came back `None`, and the process
   probe raised — a bounded HOLD, but state B could not recover.  The probe
   now decodes bytes itself and never raises.
2. **A false source rollback**: a cold-started Ollama loads qwen3:4b on the
   first request (71 s measured; `load_duration` 70.7 s), the core's single
   warm-up probe had already failed, the supervisor read "not READY in
   300 s" as a bad revision and `git revert`ed a good commit (b35c08b —
   re-applied as bc082e6).  Now: a core that answers its health report and
   only lacks the conversation model is an **environment failure** — retried
   on the same revision, never reverted (`_environmental_failure`); when the
   supervisor itself started Ollama the READY window grows by
   `ollama_cold_ready_extra` (300 s); the core's warm-up retries the probe
   up to 4×.
3. **Promotion while running**: Windows refuses to rename `dist/ZEUS` while
   ZEUS.exe runs from it, so the product could never promote its own exe.
   A blocked promotion is now **staged** (`dist/ZEUS.staged` + a pointer in
   the control directory) and the relaunch watchdog — which already runs
   after the old supervisor exits — performs the swap, writes
   `PROMOTED.json` (`"swapped_by": "relaunch watchdog"`) and starts the new
   release, restoring the previous one if it never gets READY.

## Live evidence (`data/acceptance_evidence/V4_supervisor_ollama_live.json`, exe `30b7c7f`)

| scenario | result |
|---|---|
| A: Ollama stopped, cold `ZEUS.exe` start | supervisor started `ollama serve` (11 s to RUNNING, pid recorded, correct store); first warm probe hit a provider error during the cold model load; at 600 s the **environmental branch** retried the same revision — no rollback, no hold — READY at 273 s on the second start; 1 core / 1 listener / 1 worker / 1 supervisor / 1 ollama |
| B: quit + relaunch, Ollama running | "already running", **no second Ollama** (same pid 278832), READY |
| C1/C2: `ZEUS.exe check` with a bad URL + missing/real exe (env overrides, owner config untouched) | exit 1 in 16.1 s / 13.6 s with `FAILED: ollama serve did not answer … within 7s/9s` |
| C3/C4: boot with the bad config | phase `error` with the remedy on the status page in 13.3 s — responsive, retrying; `POST /api/quit` ended it in 6 s; nothing left running |
| D: `taskkill ollama.exe` under a running ZEUS | watchdog noticed within 15 s, restarted it (new pid, `started_by_supervisor`, spawns=1) — **32.3 s** to RUNNING; AI_READY true again; a request answered; still exactly one of everything |

Unit/targeted tests: `tests/test_supervisor_ollama.py` (26) covering the nine
required scenarios plus the cached-check regression, the raising-check wrap,
the no-spawn diagnostics, the environmental-rollback rule, the cold budget
and the log-path spawner; `tests/test_release.py` gained the staged-swap
test.  The previously Ollama-dependent
`test_environment.py::test_the_fingerprint_keys_are_all_actually_probed`
passes with the managed Ollama up.

## Operational notes

* The known-good **exe** lineage: `7ba351b` (old, with the hang) →
  `9fa88b4` → `30b7c7f`; each previous release is kept
  (`dist/ZEUS.previous`), and the release history (`releases.jsonl`)
  records every build/verify/stage/swap.
* Cold-start worst case observed end to end (Ollama stopped, models not
  resident): ~16 min including one bounded internal retry — loud in the
  status page the whole time, never silent, never held, never rolled back.
  A warm machine boots in 13–19 s.
* The supervisor never stops Ollama: it treats it as a shared local service;
  duplicates are prevented on the start side.
