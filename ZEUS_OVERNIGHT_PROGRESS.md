# ZEUS overnight sprint

Durable state for a long autonomous run. No credentials or secrets in this file.

**Started:** 2026-08-25 ~22:30 local
**Branch:** `adaptive-brain-v1`

---

## Vocabulary

These are not synonyms and are never used as such below.

| word | means |
|---|---|
| IMPLEMENTED | the code exists |
| TESTED | unit/integration tests pass |
| LIVE VERIFIED | observed on this machine, through the real product, checked against something other than the thing being tested |
| PROMOTED | registered in the capability registry after independent verification |

---

## Current position

**Commit:** `5008fdf` — Every transport command was silently doing nothing
**Milestone:** Priority 1 — Spotify repair loop
**Human action required:** none

---

## Priority 1 — Spotify

### Where it stands

| | state |
|---|---|
| capability `music.provider.spotify` | PROMOTED, active, v1.0.3 |
| acquisition | LIVE VERIFIED (autonomous: 3 local failures -> escalation -> expert -> ZEUS re-verified 6/6 gates) |
| `play` from a paused player | LIVE VERIFIED |
| `play` while already playing | **DEFECT — under repair** |
| transport (pause/resume/next) | LIVE VERIFIED via SMTC |
| `current` | LIVE VERIFIED, read from Windows |
| restart persistence | not yet re-run since the defect was found |

### The defect, precisely

```
Spotify PAUSED  -> spotify:track:<id> switches in under 2 seconds   OK
Spotify PLAYING -> URI ignored, Spotify continues its own queue     FAILS
```

Four consecutive requests while playing produced four different wrong tracks
(Africa/TOTO, Sweet Child O' Mine, Highway to Hell, Livin' On A Prayer) — the
player was running its own queue throughout and never honoured the handoff.

Every previously successful play happened from a paused state. That was not
noticed at the time because nothing recorded the *prior* player state; the
defect report now carries it.

### Second Spotify defect (open)

`play` takes ~213 s. The capability spawns a fresh PowerShell per operation and
`awaitplay` blocks up to 30 s with a retry loop. Two minutes for "play a song"
is not shippable. Not yet handed to the repair loop.

---

## Defects found and fixed this session

| commit | defect | how it was found |
|---|---|---|
| `f087a40` | my own anchor hint told a 688-line file to rewrite itself; the model produced a 37-line sketch and only the shrink guard prevented data loss | reading what a real repair did |
| `8804d67` | repair brief appended the defect *after* the full spec, so the planner read a build brief and planned a build (7 "implement X" tasks) | reading DECOMPOSE output |
| `d0bba19` | anchor misses reported a similarity score, which cannot be edited; model never used `read_file` in 6 attempts | 6 failed attempts at 0.47 similarity |
| `a9968e5` | a 120 s timeout was read as "the capability is broken", retiring a verified capability and rebuilding for 30 min | acceptance run cascade |
| `5008fdf` | **two `-Command` flags** meant every transport command silently did nothing and returned the unchanged state | direct PowerShell worked, module did not |

`5008fdf` is the one worth remembering: it corrupted a live diagnosis. Testing
"does pausing first fix the handoff", the pause never happened, and the
conclusion drawn would have been the exact opposite of the truth.

---

## Measurements so far

### Capability acquisition (Spotify, first acquisition)

| phase | wall clock |
|---|---|
| BUILD_LOCAL attempt 1 | 22.5 min — FAILURE_LIMIT, 4/5 gates |
| BUILD_LOCAL attempt 2 | 30.4 min — STEP_LIMIT, 2/5 gates (regressed) |
| BUILD_LOCAL attempt 3 | 30.4 min — TIME_LIMIT |
| escalation decision | from counted evidence, not a timer |
| expert | 28.6 min |
| ZEUS re-verification | 2.2 min |
| **total** | **~114 min** |

### Repair cycles (same capability, one constant wrong)

| cycle | local | outcome |
|---|---|---|
| 1 | 35 min | expert edited blind, broke static + tests, left the bug |
| 2 | 34 min | expert fixed search, broke playback |
| 3 | 13.5 min | expert: "I changed one thing, in one function" -> **6/6 gates, promoted** |

**The ladder got faster because it remembered.** First acquisition needed three
local attempts before escalating; by the repair cycles the performance ledger
showed 0% local success over N attempts at `capability`, so the controller
escalated after one — ~60 min saved per cycle.

### Local model on this task class

`capability/build_local`: 6 attempts, 0 passed. The 7B diagnoses correctly and
cannot land an anchor in a 688-line file; it never once used `read_file` first.

---

## Next actions

1. Hand the play-while-playing defect to the repair loop (in progress).
2. Live acceptance A–G.
3. Priority 2: instrument and optimise the acquisition pipeline.
