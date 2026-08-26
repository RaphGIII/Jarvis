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

**Commit:** `d00599d` + this working tree
**Milestone:** three capabilities promoted; Spotify 9/9 live; suite green (1637 passed, 0 failed)
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
| F | what's playing | `God Save the Queen`, matched against Windows — 4 s |
| G | restart + reuse, no reacquisition | `Du hast - Rammstein [Playing]` — 14 s, 0 acquisitions |

Re-run end to end against **v1.0.5**: **9/9**. The timings are the repair,
measured through the product rather than through the capability:

| | v1.0.4 | v1.0.5 |
|---|---|---|
| A play a requested track | 186 s | **18 s** |
| B replace a track while playing | 60 s | **16 s** |
| G second use, after a restart | 61 s | **14 s** |

B is the defect that took four repair cycles. It is fixed and verified.

F failed once, and it failed in the measuring instrument. The harness read
Windows, asked, and compared the answer against the reading it had taken
several seconds earlier; Spotify auto-advanced in between, so the two were
describing different moments. A verdict reached against a stale observation —
this time in the ruler. It now pauses first so the queue cannot advance, reads
before and after, and requires the answer to name a track that was genuinely
current while ZEUS was looking.

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

### The latency defect — found, handed to ZEUS, repaired, verified

`music.provider.spotify` **v1.0.5**, promoted after ZEUS re-ran all eight gates
itself. 24.3 minutes, 13 model calls, one local attempt, escalated once.

Measured before, through the product and then through each layer:

| | |
|---|---|
| "what's playing", through the product | 154.7 s cold, 30.2 s warm |
| `run({'action': 'current'})` | 151 s cold, 26.5 s warm |
| `tools.media_session.read()` — the same question | **1.3 s** |
| the WinRT work, timed inside the PowerShell script | **0.3 s** |

So the capability spent between twenty and a hundred and fifty times what the
work costs, and it was not the model: only FAST_LOCAL is ever loaded during
that turn, and it answers ordinary chat in 1.1 s.

The cause was measured, not guessed. Stage timings taken *inside* PowerShell —
`Add-Type` 0.11 s, resolve the WinRT types 0.16 s, `RequestAsync` 0.19 s,
`GetSessions` 0.20 s, `GetPlaybackInfo` 0.22 s, `TryGetMediaPropertiesAsync`
0.24 s, emit 0.30 s — around **8.8 s of wall clock**. The stopwatch starts after
parsing, so the missing time is spent before the first line runs: the bridge
handed PowerShell the whole 5 kB script with `-EncodedCommand`, and PowerShell
re-parses that string on every invocation.

**This exact defect was measured and fixed once already**, in `2f3a4a2`, in
`tools/media_session.py` — 4.4 s a call down to 1.1 s by running the script from
a file with a `param()` block. The capability was written before that commit and
could not see the lesson, because the lesson lived in a Jarvis module a
model-authored capability has no access to. It is now a capability constraint,
which is the only place a fact like that reaches the code that needs it.

The gate was written before the repair and proved to bite against the defect:
`a warm current took 26.5s; the budget is 8.0s and Windows answers in 0.3s`. It
checks correctness alongside the clock, from Windows, so a capability cannot
pass it by answering faster from somewhere other than the operating system.

Measured after, independently, through the registry:

| | before | after |
|---|---|---|
| `run({'action': 'current'})` | 26.5 s warm | **1.7 s** |
| "what's playing", through the product | 30.2 s warm | **5.1 s** |
| — cold | 154.7 s | **5.0 s** |
| ordinary chat (FAST_LOCAL) | 1.1 s | 1.5 s |
| `status` | 0.3 s, no model call | 0.3 s, no model call |

The expert reported its own attempt as **failed**. ZEUS re-ran the gates and
promoted on its own verification — the fifth time in five that the rule has
been the difference between a verdict and a claim.

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

### A seventh, found only because a second capability existed

With two capabilities in the registry, one query still resolved wrongly:

    "rebuild the existing implementation because of a defect and repair the
     working code"  ->  music.provider.spotify

Keywords are derived from the goal that produced the capability. The goal that
*creates* one describes it; the goal that produces v1.0.4 of one is a repair
brief, and a repair brief opens with the defect -- deliberately, because
`8804d67` fixed a planner that rebuilt instead of repairing when it did not.
So the music provider was indexed under *defect, existing, implementation,
rebuild, repair, working*, and every capability that is ever repaired acquires
the same six.

The vocabulary resolution ignores is now also the vocabulary that is never
stored, so a word that could not contribute to a match cannot be recorded as
one. 12/12 on the live registry, including four queries that must resolve to
nothing.

This one is worth noting for how it was found: it was invisible with one
capability installed. The benchmark did not test it; having done the benchmark
made it testable.

### The two local models cannot both be resident, and that is hardware

Six tier changes in the passing acquisition cost about **303 seconds — 11% of
the run** — reloading a model that had been resident moments earlier.

| from → to | first call after the change |
|---|---|
| BUILD → FAST | 88.9 s, 34.8 s, 34.5 s (on 210-character prompts) |
| FAST → BUILD | 112.9 s, 74.0 s, 79.5 s |

Steady-state, the same tiers cost 3.0 s and 37.5 s. So roughly 150 seconds on
each side of the swap is pure reloading.

Tested rather than assumed, at two context sizes:

```
num_ctx 8192   coder 5.12 GB resident -> load qwen3:4b -> coder EVICTED
num_ctx 4096   coder 4.74 GB resident -> load qwen3:4b -> coder EVICTED
```

4.74 + 3.18 GB is 7.9 GB on an 8 GB card that already owes ~0.7 GB to the
display. They do not fit at any context size, so this is not a configuration
mistake and cannot be tuned away. The remedy, if it is worth taking, is
behavioural — do not interleave the tiers inside one mission — and it is
recorded here rather than acted on, because the measurement does not yet say
which FAST_LOCAL calls those were.

### Four tests were red on arrival; all four are green

The full suite was run at the promotion boundary and found four failures. Three
of them were this sprint's own doing.

**`e26f723` traded a false positive for a false negative.** Bisected: the three
`test_autonomous_engineering_v04` failures pass at `b4b27e2`, `b6b26d9`,
`2f3a4a2` and `7e76681`, and fail at `e26f723` — the commit that fixed the worst
false positive in the project, landed without the full suite being run.

Matching moved to the identifier and the declared keywords, which is right, and
an identifier can be a code name. `custom.scale` shares no word with *"double an
integer x from the request payload"*, so a capability that declared no keywords
became invisible the moment it was registered. All three tests assert the same
thing: that the second call reuses what the first one built. It was rebuilding
from scratch what it had finished building a second earlier, and reporting the
rebuild as a success. The same failure the benchmark exists to measure, from
the other side — a false positive answers with the wrong capability, a false
negative pays for the right one twice.

Three defences now, in order: match on what a capability *declares*; fall back
to the description's **first sentence** only, because a capability states what
it is for and then states how it must behave; and never on a derived term
shorter than four characters. The middle rule exists because whole descriptions
brought the original false match straight back on seventeen shared contract
words, and the last because after that `for` alone was still enough.

**And the voice failure was a question being read as a command.** *"Wie geht es
weiter?"* — four words, inside `MAX_TRANSPORT_WORDS` — matched `weiter` and was
classified as *resume*. Live, that resumed playback, found no verified provider
and started acquiring a Spotify capability: an autonomous build begun in answer
to something nobody asked. An interrogative opening now rules out a bare
transport verb, which needs no threshold — a sentence that begins by asking is
asking. *"Was ist der nächste Schritt?"* stops being a skip too.

Found because of a fix made along the way: an exception in the answering thread
used to vanish. `send_message` returns `{"ok": True, "accepted": text}` the
moment the thread starts — that is what lets Jarvis speak before it has finished
thinking — so everything after that is out of the request's reach. The exception
reached a daemon thread's default handler, printed to a stderr nobody reads, and
the user simply never got an answer. Accepted, no reply, no error. It now
publishes an ERROR event carrying the traceback and delivers a reply saying so,
and still returns immediately. It named both missing attributes in the voice
stub on the first run.

### The pattern, again

Five of the six are the same shape as the defect this benchmark was written to
catch: something answering a question narrower than the one asked, and the
answer being read as the wider one. `shutil.which` for *is it available*.
*Would leave unparseable* for *is broken*. A diagnosis under a heading that says
*known facts*. Term overlap for *is about the same thing*. The proxy is right
while things are small and quietly stops being right as they grow.

---

## Priority 3b — a third domain, and the first acquisition BUILD_LOCAL finished alone

`archive.zip.create` v1.0.0 — PROMOTED. Package a folder into a zip: no WinRT,
no display, no player, reachable from the standard library in a few lines. That
was the point. The ledger recorded **0% local success over nine attempts**, and
a task the 7B model should be able to finish alone is the one that says whether
that number is about the model or about the pipeline.

| | screen capture | zip archive |
|---|---|---|
| wall clock | 47.6 min | **7.5 min** |
| model calls | 49 | **11** |
| engine steps | 48 | 13 |
| BUILD_LOCAL prompt | 24,155 chars | **17,381 chars** |
| tier changes | 6 (~303 s of reloading) | **0** |
| ExpertGateway | required | **not reached** |
| second use | 0.459 s, 0 model calls | 0.0 s, 0 model calls |

Six times faster, and it never left the local tier. The prompt reduction shows
up exactly where it was measured to be — 24,155 → 17,381 characters is the 28%
this sprint removed.

Verified from outside, on a folder built *after* the capability was registered
so nothing about it could have been anticipated: five files including a unicode
name and five levels of nesting, an empty directory, every byte identical, and
a missing source folder answered with `{'ok': False, 'error': 'Source folder
does not exist'}` rather than a traceback.

Resolution with three capabilities installed:
`find("zip a folder into an archive") -> ['archive.zip.create']`.

### A gap in my own gate

The independent check found something the gate did not: it reports
`'files': 1` while archiving five. The contract in the goal asks for a count,
the archive is correct, and the number beside it is not.

The gate proved the *artifact* and never checked the numbers the capability
reported *about* the artifact — the same shape of omission as the six Spotify
gates that each covered a state and between them missed play-while-playing.
Recorded here rather than quietly fixed, because a gate found wanting is worth
more written down than corrected in silence.

---

## Acquisition speed — what was measured, and what was done

The passing run: 48 engine steps, 49 model calls, 47.6 minutes.

| phase | steps | share | failed |
|---|---|---|---|
| EXECUTE | 17 | 35% | 4 |
| DIAGNOSE | 16 | 33% | 0 |
| VERIFY | 13 | 27% | 13 |
| INVESTIGATE | 1 | 2% | 0 |
| DECOMPOSE | 1 | 2% | 0 |

VERIFY costs **no model call at all** — it is the deterministic gate runner. So
the model budget is EXECUTE and DIAGNOSE, and on this GPU **prefill dominates
each of them**: 37 BUILD_LOCAL calls at 41.6 s on prompts of 24,000 characters.

Four changes, each from a measurement rather than an intuition:

| change | effect |
|---|---|
| the goal was in every prompt twice | brief 29.5 k → 16.6 k chars |
| `python -c` check scripts pasted in full | −5,385 chars a call |
| DECOMPOSE planning lookups as tasks | 2 fewer EXECUTE steps, and no lookup task left for a repair to reopen |
| a task that never exhausts could be reopened for ever | a whole run's tail of no-op EXECUTEs, gone |

Total prompt, same project and state: **41,095 → 29,610 characters, −28%**,
with nothing removed the model could act on.

---

## Next actions

1. `jarvis.measure_pipeline --heavy` on a free GPU, for the numbers this sprint
   did not need but the roadmap does.
2. Find which FAST_LOCAL calls happen *inside* an acquisition. Six tier changes
   cost 303 s and the models cannot coexist, so the only remedy is not to
   interleave them — and the measurement does not yet say which calls those
   were. Their prompts were 210 characters, which is the clue.
3. A third capability, in a third domain, against the numbers above. The first
   cross-domain acquisition needed one local attempt and an escalation; the
   question is whether the reusable pattern now recorded makes the second one
   cheaper.
4. `main.py` for the Spotify provider is 708 lines and its repair spent eight
   consecutive anchor failures before escalating. `replace_definition` exists
   now; whether a local attempt can use it is untested against a real repair.
