# ZEUS overnight sprint

Durable state for a long autonomous run. No credentials or secrets in this file.

**Started:** 2026-08-25 ~22:30 local
**Branch:** `adaptive-brain-v1` — pushed to origin after each milestone

---

## Vocabulary

Not synonyms, and never used as such below.

| word | means |
|---|---|
| IMPLEMENTED | the code exists |
| TESTED | unit/integration tests pass |
| LIVE VERIFIED | observed on this machine, through the real product, checked against something other than the thing being tested |
| PROMOTED | registered in the capability registry after independent verification |

---

## Current position

**Commit:** `b4b27e2`
**Milestone:** Priority 1 acceptance running (A–E passed); Priority 2 measured
**Human action required:** none

---

## Priority 1 — Spotify

`music.provider.spotify` v1.0.4 — PROMOTED, active, seven gates green.

### Live acceptance (real HTTP path, verdicts read from Windows SMTC)

| | check | evidence |
|---|---|---|
| A | play a requested track | `Lose Yourself - Eminem [Playing]` — 186 s |
| B | **replace a track while playing** | `'Lose Yourself' -> 'Bohemian Rhapsody'` — 60 s |
| C | pause | `Bohemian Rhapsody - Queen [Paused]` |
| D | resume | `Bohemian Rhapsody - Queen [Playing]` |
| E | next | `'Bohemian Rhapsody' -> 'God Save the Queen'` |
| F | what's playing | running |
| G | restart + reuse, no reacquisition | running |

B is the defect that took four repair cycles. It is fixed and verified.

### The defect, and why six gates missed it

```
Spotify PAUSED  -> spotify:track:<id> switches in under 2 s   OK
Spotify PLAYING -> URI ignored, the player continues its queue FAILED
```

`playback` proved resume and pause. `search` proved a name resolves to a track.
Neither proved *start this track now, while something else is playing* — which
is what a user actually asks for. Each gate was individually reasonable; between
them they covered every state except the one the defect lived in.

A seventh gate now starts from a playing state deliberately. It names no track:
it asks the provider to search for a common word, takes the result as the
expectation, and requires that to be what Windows reports. Verified against
v1.0.4: `SWITCH_OK Feel Good Inc. -> Love Me Not`.

### Open Spotify issue

First play costs ~186 s cold, ~60 s warm. The capability spawns a fresh
PowerShell per operation and its await loop blocks up to 30 s. Not shippable
latency; not yet handed to the repair loop.

---

## Priority 2 — measurements

### Where acquisition time actually goes

12 capability attempts on record: **189 model calls, 190 tool calls, 162.7 minutes.**

| phase | calls | share |
|---|---|---|
| EXECUTE | 70 | 37% |
| DIAGNOSE | 59 | 31% |
| VERIFY | 42 | 22% |
| INVESTIGATE | 9 | 4.8% |
| DECOMPOSE | 7 | 3.7% |
| REPLAN | 2 | 1.1% |

**38% of model calls were productive. 62% produced no progress.**

### The dominant recoverable cost

| | |
|---|---|
| EXECUTE steps | 70, of which **22 failed (31%)** |
| edit-mechanics share of those failures | **77%** (anchor missed 9, anchor ambiguous 5, shrink refused 2, invalid code 2, no-op 1) |
| failed EXECUTE immediately followed by DIAGNOSE | **21 of 22 (95%)** |
| **total attributable to failed edits** | 22 + 21 = **43 calls = 23% of everything** |

Nearly a quarter of every acquisition was spent failing to apply an edit and
then explaining that failure to itself. That is the bottleneck, and the three
edit fixes this sprint target exactly it.

### A correction to my own assumption

I called the environment cache the highest-leverage optimisation. **The data
says INVESTIGATE is 4.8% of model calls** — the cache saves very little time.
Its real value is preventing *wrong* environment assumptions, which caused four
of six earlier music failures. Worth keeping, wrongly justified.

---

## Defects found and fixed this sprint

| commit | defect | how found |
|---|---|---|
| `f087a40` | my own anchor hint told a 688-line file to rewrite itself; the model produced a 37-line sketch, only the shrink guard prevented data loss | reading what a real repair did |
| `8804d67` | repair brief put the defect *after* the spec, so the planner planned a rebuild (7 "implement X" tasks) | reading DECOMPOSE output |
| `d0bba19` | anchor misses reported a similarity score, which cannot be edited; `read_file` unused in 6 attempts | 6 failures at 0.47 similarity |
| `a9968e5` | a 120 s timeout read as "capability broken", retiring a verified capability | acceptance cascade |
| `5008fdf` | **two `-Command` flags** — every transport command silently did nothing | direct PowerShell worked, module did not |
| `b63a9c3` | SQLite cross-thread error killed a BUILD_LOCAL attempt instantly; the mission counted it as a model failure and escalated over it | mission reports what raised |
| `68d6c58` | six gates, none covering play-while-playing | the defect survived all of them |

### The pattern

Four of these fed the escalation controller conclusions it had not earned.
`ok=False` meaning both "behaved wrongly" and "never finished"; a crashed
attempt counted as a model failure; a restart re-deriving evidence *and*
recording it as new failures. The ladder reasons from counts, so anything that
manufactures a count corrupts the decision to escalate.

`5008fdf` is the one to remember: it corrupted a live diagnosis. Testing "does
pausing first fix the handoff", the pause never happened, and the conclusion
would have been the exact opposite of the truth.

---

## Subsystems connected that were built and orphaned

Four, now wired to production for the first time:

- `EscalationController` + `PerformanceLedger` — escalation decided from counts
- `ExpertMemory` — verified lessons recalled before a mission spends anything
- `MissionStore` (new) — checkpoints so an interruption is not a restart

Each was correctly written and imported by nothing outside its own tests. Their
absence looked exactly like the system working.

---

## Escalation, measured

| | |
|---|---|
| expert escalations so far | 4 |
| times the expert could execute anything | **0** |
| times ZEUS's re-run was the first real execution | **4** |

Every escalation has produced a variant of *"command execution is blocked in
this session"*. The rule that the expert's report is never the verdict has paid
for itself four times out of four.

The ladder also got faster by remembering: the first acquisition needed three
local attempts before escalating; later repairs escalated after one, because
the ledger showed 0% local success for the class. ~60 min saved per cycle.

---

## Next actions

1. Finish acceptance F and G.
2. Run `jarvis.measure_pipeline --heavy` on a free GPU for latency + eviction numbers.
3. Priority 3: one cross-domain capability, measured against the Spotify baseline
   above (189 calls / 162.7 min / 23% wasted on edit mechanics).
