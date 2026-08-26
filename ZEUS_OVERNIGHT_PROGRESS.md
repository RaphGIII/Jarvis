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

**Commit:** `e26f723` + this working tree
**Milestone:** Priority 3 PASSED — `system.screen.capture` acquired, verified and promoted
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

## Priority 3 — cross-domain capability: `system.screen.capture`

The benchmark asks ZEUS to acquire a capability in a domain with nothing
reusable from the Spotify work. Gates, both run from outside the capability:

| gate | what it refuses |
|---|---|
| `artifact` | a path with no file, a `.png` that is not a PNG, a file found rather than written, a near-empty one |
| `dimensions` | an image whose pixel size is not this display's, read from Windows *inside the checking process* |

Run 1 (commit `e26f723`, before the resolver fix) reported `acquired: True` in
0.11 s having built nothing: resolution matched the goal to
`music.provider.spotify`. That is the defect `e26f723` fixed.

### PASSED — run 5, `system.screen.capture` v1.0.0 PROMOTED

| measurement | |
|---|---|
| wall clock | **47.6 min** (2858 s) |
| BUILD_LOCAL calls | **37** (25.7 min of generation) |
| FAST_LOCAL calls | 12 |
| model calls, total | 49 over 48 engine steps |
| local attempts (retries) | **1** of a permitted 3 |
| ExpertGateway required | **yes** — `claude_code`, 1005 s |
| verification steps | 6 gates, all re-run locally after the expert |
| **second-use latency** | **resolve 0.000 s, execute 0.459 s, 0 model calls** |

Local attempt 1 spent its whole 1800 s budget and did not converge. The ladder
escalated after one attempt rather than three, on ledger evidence: *0% local
success over 7 attempts at 'vision'*. The expert **could not execute anything**
— *"I could not execute Python in this session"* — the fifth such report out of
five escalations. It wrote code; ZEUS re-ran the gates itself and promoted on
its own verification, which is the rule that has now paid for itself five times.

What was built is a 416-line stdlib GDI capture (`BitBlt` through `ctypes`, PNG
encoded with `zlib` + `struct`), with `mss` only as a fallback if importable.
It carries its own guard against a uniform-colour capture.

**Verified independently of the acquisition**, through the registry, into a path
that did not exist beforehand:

```
resolve('capture what is on the screen and save it as a png') -> ['system.screen.capture']
execute -> ok=True in 0.38s   1920x1080, 66014 bytes, written 0.1s ago
distinct colours: 553          (a blank or placeholder has one)
```

Windows was asked for the display size inside the checking process, and the PNG
dimensions were read from the IHDR chunk with `struct` — no imaging library, so
nothing is shared with the thing under test. Opening the image shows this
session's own terminal. Spotify does not answer a screen request.

### What the failing runs exposed — six general defects

Runs 2–4 each failed differently, and each failure named a defect in the
pipeline rather than in the capability.

**`find_program` answered a narrower question than it was asked.** It ran
`shutil.which` and nothing else, so:

```
find_program({"name": "pyautogui"}) -> {"found": false, "path": ""}
find_program({"name": "mss"})       -> {"found": false, "path": ""}
```

Both are installed and importable here. "Is there an executable on PATH called
X" was standing in for "is X available", the same substitution as description
overlap standing in for relevance. It now answers about availability, with
`kind` ∈ `executable | python_package | absent`, no `path` for a package
because there is nothing to hand to `subprocess`, and an `answer` sentence
saying which of those it is.

**The briefing claimed to know what was not installed.** It listed twenty
probed packages and then said *"Anything else is NOT installed"*. `mss` was
installed and on no list. The list is now what was *probed*, anything else is
unknown, and `find_program` is named as what answers for it. Six more
candidates added to the probe (`mss`, `pyperclip`, `keyboard`, `pyscreeze`,
`pygetwindow`, `pytesseract`), all present here and none previously mentioned
to any capability.

**A traceback names where an exception was raised, not where the mistake was.**
`main.py` called `sct.shot(mon=monitor)` with a dict where mss wants an index,
so mss raised inside `site-packages/mss/base.py`. Three consecutive diagnoses
said, correctly quoting the evidence, that the defect was `monitor =
monitors[mon]` — a line in a library the project cannot edit, and every anchored
edit against it missed. `_foreign_frame_notice` now names the deepest frame
that is *inside the workspace*, and names the library as `mss` rather than as
`base.py`, which identifies nothing.

**A rejected edit reported its syntax error as if it were in the file.** The
message read *"the edit would leave test_capability.py unparseable: ... at line
6"* — true, and read as *line 6 of that file is broken*. **Eight consecutive
identical diagnoses** hunting a syntax error in a file that had none, while the
real defect sat untouched in `main.py`. It now leads with `NOTHING WAS WRITTEN`,
says the file on disk still parses, and says line 6 belongs to the text the
model sent.

**Guesses were evicting observations out of `KNOWN FACTS`.** DIAGNOSE records
each diagnosis as a finding; the brief shows the last eight findings; a stuck
loop produces nothing else. Within four cycles every fact the tools had
established was gone. Measured: the `research` tool returned the correct mss
call, and by the time EXECUTE wrote the line, all eight `KNOWN FACTS` were
repetitions of one wrong diagnosis. A diagnosis is a guess, it is already in the
prompt twice, and it no longer appears under a heading that says *facts*.

**A task that never exhausts could be reopened for ever.** `max_reopenings`
exists to stop a task oscillating between DONE and reopened, and was only ever
consulted for tasks that had *also* run out of attempts. A task that succeeds
on its first attempt is never exhausted. Measured: the implementation task was
abandoned after three attempts, which left *"Check if 'pyautogui' is
installed"* — one tool call, always succeeds, changes nothing — as the most
recent finished task, and the loop reopened and re-completed it for the rest of
the run. Every reopening is now counted.

**And DECOMPOSE was planning lookups as tasks.** *"Check if 'mss' is
installed"* is not work: INVESTIGATE already answered it. Worse, a lookup task
always succeeds, which is what made it the thing a repair reopened. The planner
is now told that a task must change something in the workspace.

Also removed: the `vision / build_local / succeeded / 0.005 s` row the original
false positive wrote to the performance ledger. The escalation controller
reasons from counts, and nothing was built.

### The pattern, again

Five of the six are the same shape as the defect this benchmark was written to
catch: something answering a question narrower than the one asked, and the
answer being read as the wider one. `shutil.which` for *is it available*.
*Would leave unparseable* for *is broken*. A diagnosis under a heading that says
*known facts*. Term overlap for *is about the same thing*. The proxy is right
while things are small and quietly stops being right as they grow.

---

## Next actions

1. Finish Spotify acceptance F and G.
2. Run `jarvis.measure_pipeline --heavy` on a free GPU for latency + eviction numbers.
3. Optimise generic acquisition speed; reduce model calls that produce no progress.
4. Checkpoint/resume so a capability mission survives a restart.
5. Verified-trajectory reuse, so the second acquisition in a domain is cheaper.
