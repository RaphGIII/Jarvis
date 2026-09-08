# ZEUS — Capability-first / Codex-teaching architecture

Continues `0399019` (WIP, Codex hit its usage limit mid-migration). Nothing from
that commit was reverted; what was wrong in it is named below with the evidence.

## The architecture

```
owner request
      │
      ▼
CapabilityResolver ──── FOUND + HEALTHY ──▶ execute locally.  No engineer.
      │                                     No availability probe.
      │                                     No model call.
      ├──── AMBIGUOUS ─────────────────────▶ one clarification
      │
      ├──── BROKEN ────┐
      └──── MISSING ───┴──▶ is Codex available?
                              │  no  ─▶ queue the request, say so
                              └─ yes ─▶ Codex engineers a GENERIC capability
                                        in an isolated workspace
                                          ▼
                                        verify here, independently
                                          ▼
                                        register ACTIVE / HEALTHY
                                        codex_required = false
                                          ▼
                                        the original request resumes
                                          ▼
                                        every equivalent request after this
                                        one is answered locally, forever
```

Codex is the engineer. It is never the runtime.

## What `0399019` already had

Codex's two commits (`3444ced`, `0399019`) built most of the vocabulary:

- `CapabilityLifecycle` / `CapabilityHealth` / `CapabilityResolutionStatus`
  enums, and a `CapabilityManifest` widened with `family`, `goal_types`,
  `target_types`, `examples`, `anti_examples`, `aliases`, `side_effects`,
  `verification`, `latency_class`, `runtime_brain`, `codex_required`,
  `source`, `supersedes`, `semantic_signature`.
- `CapabilityResolver`: typed, model-free ranking producing
  FOUND / AMBIGUOUS / MISSING / BROKEN with candidates as evidence.
- `capabilities/codex.py`: `CodexAvailabilityCache`, `StaticCodexAvailability`,
  `CapabilityRequestQueue`.
- `AcquisitionMission.run(codex_first=True)` and `_run_codex_first`: check
  Codex, queue when unavailable, otherwise engineer in an isolated workspace
  and go through the existing verification and registration gates.
- Registry schema bumped to 2; health writes both the legacy `state` and the
  new `health`.
- `CapabilityAcquisitionRuntime` wired to consult Codex before generating.
- `experts/codex.py` promoted from "written, not verified" to the selected
  provider, with metered credentials removed from the child environment.
- `Jarvis/tests/test_capability_first_architecture.py` — 9 tests, all passing.

## What it did not have

Each of these is a thing the migration needs to be true, and was not.

**1. The production request path never used the resolver.**
`JarvisCore._answer_by_capability` called `CapabilityService.resolve()`, which
was still `registry.find()` plus a knowledge-graph fallback returning a
manifest or `None`. FOUND / BROKEN / AMBIGUOUS could not be distinguished, so
a broken capability and an absent one led to the same place.

**2. The semantic planner outranked the registry.**
`capability.missing` from FAST_LOCAL went straight to acquisition without ever
asking the registry. A capability learned an hour earlier was re-acquired.
Worse, measured live on this machine: asked *"welchen sha256 Fingerprint hat
die Datei zeus_acceptance.txt?"*, FAST_LOCAL answered `file.open` with
confidence **0.95** — a confident wrong reading of a request the registry could
answer exactly. The model has never seen the registry, so it cannot be the
thing that decides whether a capability exists.

**3. `semantic_signature` could not detect a duplicate.**
It was computed from `capability_id` (unique by construction) and from
`aliases` (which change every time the capability is used). Deduplication was
therefore unreachable code.

**4. The Codex adapter did not work.**
`codex exec --full-auto` is rejected outright by the installed
`codex-cli 0.153.4` ("unexpected argument '--full-auto' found"). It exits in
under four seconds with an empty summary, which in the acquisition log is
indistinguishable from an expert that tried and failed — and that is exactly
what the first live run recorded. `--skip-git-repo-check` is also required,
because a capability workspace is a plain directory and without it the CLI
refuses to start. The commit had removed the adapter's own honest
"never verified against a real CLI" admission without verifying it.

**5. Availability was a version string.**
`codex --version` reads a binary on disk and knows nothing about the account
behind it, so availability reported READY while every real call came back
"You've hit your usage limit". The request was then reported as a failed build
rather than queued.

**6. A capability learned from one request was indexed under that request.**
Keywords came from the raw owner sentence, so a checksum capability learned
from *"…der Datei zeus_acceptance.txt"* was stored under `zeus_acceptance` and
`txt`, and answered every request naming that file — including "open it".

**7. Mojibake** in three German strings (`FÃ¤higkeit`, `zuverlÃ¤ssig`,
`AusfÃ¼hren`) and one stale test asserting a flag the commit had deleted.

## What changed

### Routing

- `CapabilityService.resolve_request()` — the routing decision, using
  `CapabilityResolver`; the knowledge graph is a candidate source only.
  `resolve()` remains as the narrow "give me a manifest" answer.
- `JarvisCore._route_capability_goal()` — one gate in front of every capability
  request. FOUND executes; AMBIGUOUS asks once; BROKEN repairs; MISSING teaches.
- `JarvisCore._dispatch_known_capability()` — runs **before** the semantic
  planner, on the action path and on the read path (a question can be a request
  for something to happen). A healthy, clearly-matching capability whose every
  required input is present in the sentence is dispatched with no model in the
  loop at all. This is what makes a paraphrase work, and it removes a 20–90 s
  planner call from every known-capability request. It declines everything else.
- `capability.missing` from the planner now goes through the resolver instead of
  straight to acquisition — at both confidence levels, because even an unsure
  "I have no tool for this" is a claim about the registry. So does "lerne X",
  which now runs what it already has.
- The BROKEN score boost only applies to a candidate that already matched.
  Lifting it unconditionally meant one defective registry entry would answer
  BROKEN to nearly every request and send them into a repair of the wrong thing.

### Learning and generalization

- `capabilities/generalize.py` — the particulars of a request (paths,
  filenames, quoted literals, drive letters) are replaced by what they are an
  example of before anything is built, and are dropped from the indexed
  vocabulary. Two requests about two different files now generalize to one
  capability.
- `CapabilityRegistry.learn_alias()` — a phrasing that reached a successful run
  is stored, generalized, as something the capability answers to. Measured on
  the live registry: "wie lautet die sha256 Pruefsumme von X?" goes from MISSING
  to FOUND after the first real run, while "öffne X", "wie spät ist es?" and
  "spiel Rammstein" stay MISSING. Newest-first, so a full list evicts the oldest
  phrasing rather than silently refusing to learn any more.
- `CapabilityRegistry.find()` now indexes aliases and examples, and subtracts
  anti-example vocabulary.
- Alphanumeric-boundary splitting in both tokenisers: `sha256` also means
  `sha` and `256`. Without it "SHA-256-Prüfsumme" and "sha256" share no
  vocabulary, and the paraphrase scored 0.40 — under threshold.
- `compute_semantic_signature()` rebuilt on the subject (family, goal types,
  target types, effects, dependencies), excluding the identifier and aliases;
  capabilities that declare no subject keep a unique signature and are never
  deduplicated. `_deduplicate()` supersedes a same-subject predecessor and
  inherits its learned phrasings.
- `suggest_id()` folds umlauts before splitting — `Prüfsumme` was becoming the
  fragments `pr` and `fsumme`, and a live capability registered as
  `local.berechne.sha.256_fsumme`.

### The engineer

- `experts/codex.py`: `--sandbox workspace-write --skip-git-repo-check -C <ws>`,
  verified against the installed CLI; `stdin=DEVNULL` (the CLI otherwise waits
  on a terminal that a service does not have); quota exhaustion remembered with
  the reset time parsed out of the message, so availability stops saying READY.
- The environment handed to Codex drops every metered credential **and**
  everything whose name says it carries a secret.
- `_run_codex_first` sends Codex the capability contract (the project's own
  constraints and acceptance commands). It was previously judged against gates
  it had never been shown.
- A quota refusal mid-run is now a queued request, not a failed build.

### Repair, queue, evidence, UI

- `_start_capability_repair_for_request` — a BROKEN resolution repairs the named
  capability from its own source, with the recorded defect as the brief.
- `CapabilityRequestQueue.resolve()` and last-write-wins `list()`; a queued
  request is closed when an acquisition serves it. `/api/capabilities/requests`.
- Every routing decision is emitted as `capability_routing` evidence — result,
  capability, confidence, reason, runners-up, whether an engineer was consulted,
  and the dispatch latency — and rendered in Activity.
- Capability Center shows lifecycle, health, `codex_required`, runtime brain,
  source, goal/target types, examples, learned aliases, semantic signature,
  supersedes, success rate, and the queue.

### Bugs found on the way

- `_capability_payload` fed the whole sentence to the path resolver: `FILENAME`
  allows spaces, so *"berechne die SHA-256-Pruefsumme der Datei bericht.txt"*
  matched in one piece and was looked up as a path. The request was answered
  with "tell me which file" about a file the owner had named.
- `_install` copied `.jarvis_tmp` — the workspace-local TEMP that the WIP commit
  pointed every verification subprocess at — into the installed capability. One
  live install failed with a `shutil.Error` while pytest deleted those files
  underneath it, and a verified, correct capability was thrown away.
- "Nothing external to check" was being counted as "the external check failed":
  every correct value-returning capability was marked AT_RISK on its first
  successful call, and the whole system went to ERROR after doing what was asked.
  VERIFIED, RAN and FAILED are now three outcomes, not two.
- `_output_schema_of` — a capability's declared `OUTPUT_SCHEMA` is now read the
  way its `INPUT_SCHEMA` already was. Every manifest previously claimed
  `{"ok": boolean}`, which is true of every possible return value.
- A capability that returns a value and no prose left the owner with the single
  word "ran". What it produced is the answer, so it is said out loud.
- A capability that names its input something the payload mapping does not
  recognise (`source_document`, `target`) was unusable. When exactly one
  required string slot is empty and the request names exactly one file, which
  goes where is not a guess; with two empty slots it still refuses.
- `created_by` and `source` were hardcoded to `codex` on every capability,
  including locally built ones. `built_by` is now passed from the run, so
  "which capabilities did an engineer actually write" is answerable.

## The live acceptance

Run on 2026-09-08 against the live state root, through the ordinary HTTP
request path, with the real Codex CLI. The capability chosen was one ZEUS
genuinely did not have: the SHA-256 checksum of a file. Verified absent first —
the resolver answered MISSING for all three phrasings before anything ran.

**Teaching.** `Zeus, berechne mir die SHA-256-Pruefsumme der Datei
zeus_acceptance.txt`

```
04:52:53  capability routing: MISSING - (0.00, 18ms)
04:52:54  generalized for reuse: berechne SHA-256-Prüfsumme einer Datei
04:52:54  acquisition/codex_availability: READY
04:52:55  acquisition/codex: engineering local.berechne.sha.256_pruefsumme
                             in isolated workspace …/proj_75af489e07
05:56:21  acquisition/expert: completed
05:56:21  acquisition/verify: re-running the capability checks here
05:56:23  acquisition/promote: registered after independent verification
05:56:23  answer: SHA-256-Pruefsumme von zeus_acceptance.txt:
          281a25ebee87b3beb4b325f72442e4387c8fa5b32824649750d2ae26366920db
```

The digest is correct. No `build_local` step ran: Codex was asked first and was
the only engineer involved. What it wrote is generic — chunked reading, a dry
run, an alias input key, clean errors — and it is indexed under *berechne,
pruefsumme, sha*, not under the file it was first asked about.

**Registered as** `local.berechne.sha.256_pruefsumme`, ACTIVE, HEALTHY,
`codex_required=false`, `runtime_brain=none`, `created_by=codex`.

**Restart, engineer removed.** ZEUS stopped, `codex` replaced on PATH by a shim
that logs every invocation and exits non-zero, ZEUS restarted. Then a different
formulation — a question rather than an imperative, different verb, different
preposition:

```
Zeus, wie lautet die sha256 Pruefsumme von zeus_acceptance.txt?

05:28:16  capability routing: FOUND local.berechne.sha.256_pruefsumme
                              (0.71, 2ms, before the planner)
                              codex_checked=False
05:28:16  action.verified  SHA-256-Pruefsumme von zeus_acceptance.txt:
          281a25ebee87b3beb4b325f72442e4387c8fa5b32824649750d2ae26366920db

CODEX INVOCATIONS SERVING THE REQUEST: 0
```

The shim is genuinely in effect: a request for something still missing (file
entropy) does reach it, fails, and is queued — one logged invocation, proving
the count of zero above is a measurement and not an artefact of a broken shim.

**What generalization does and does not do.** It carries across *phrasings of
the same subject*: "berechne mir die SHA-256-Prüfsumme der Datei X" and "wie
lautet die sha256 Prüfsumme von X?" resolve to the same capability. It does not
invent synonyms: "welchen sha256 **Fingerprint** hat die Datei X?" resolves only
once the capability has seen that word. That is the honest boundary, and it is
where it should be — the alternative is a system that answers requests it has
no evidence it understands.

### Found by running it

Six defects the offline tests did not surface, each fixed and pinned:

1. **A failed acquisition attempt was offered to the planner as an owner
   project.** Asked to compute a checksum, FAST_LOCAL was handed the title
   `local.berechne.sha.256_fsumme` — ZEUS's own record of failing to build the
   thing being asked for — chose `project.open` at confidence **0.98**, and
   answered *"Projekt local.berechne.sha.256_fsumme ist offen"*. The grounding
   check that exists to catch invented targets waved it through, because the
   target really did exist. `owner_projects()` now separates the two,
   everywhere the distinction matters.
2. **A read-only capability was failed for the age of its input.** The
   freshness check ("produced just now, not found") assumes a reported path is
   an artifact; a checksum reports the file it was asked about. Correct digest,
   FAILED receipt, capability marked AT_RISK. It now only applies to a path the
   capability chose itself.
3. **`Prüfsumme` was indexed as `fsumme`.** Splitting on "not `[a-z0-9]`"
   treats an umlaut as a separator, in `suggest_id` and in `_keywords_from`.
4. **The registry tokeniser did not fold umlauts while the resolver did**, so
   a capability indexed under *prüfsumme* could not be found by a request
   written *Pruefsumme*.
5. **`zeus` was singularised into `zeu`.** The address term is filtered at the
   end of tokenisation, but the singulariser had already produced a token in no
   filter list, present in every request the owner speaks, matching nothing —
   diluting every score. The paraphrase that should have resolved came in at
   0.479 against a 0.48 bar.
6. **Sharing only the word *Datei* was enough to match.** Asked to "bestimme
   die Entropie in Bits pro Byte der Datei X", ZEUS answered with the file's
   SHA-256 checksum at confidence 0.50 — and then *learned the entropy request
   as one of the checksum capability's phrasings*, so the next wrong match would
   have been easier than the last. The German container nouns are now in
   `BOILERPLATE` where their English equivalents already were.

Two aliases created by defect 6 were removed from the live registry; a queued
request whose goal was the confirmation phrase "Ja, bitte lerne das." was
withdrawn, and a goal that names no subject is now refused before an engineer
is involved.

### Measured

| | |
|---|---|
| Known-capability dispatch | **2.1 ms** to resolve, answered inside the same second |
| Codex acquisition, end to end | **215 s** (request to answer), of which 206 s was Codex |
| Codex invocations serving a learned capability | **0** |
| Registry resolution, whole registry | 2–23 ms |

## The repair loop, live

Run on 2026-09-08 on the capability Codex had taught earlier the same day. A
deliberate, reversible, deterministic fault was introduced into the installed
implementation -- `digest.hexdigest()` -> `digest.hex_digest()`, one call in
`_sha256_file`, which raises `AttributeError` on every real hash and leaves the
dry run untouched. A byte-exact backup was taken first.

```
request 1   FOUND  -> action.failed  AttributeError: ... has no attribute 'hex_digest'
            health HEALTHY -> AT_RISK      (consecutive_failures 1)
request 2   ->  action.failed
            health AT_RISK -> BROKEN       (consecutive_failures 2, FAILING_AFTER)
request 3   capability routing: BROKEN local.berechne.sha.256_pruefsumme (0.72, 1ms)
            acquisition/retire: disabled ... AttributeError: ...
            acquisition/codex_availability: READY
            acquisition/codex: engineering local.berechne.sha.256_pruefsumme
            acquisition/expert: completed (116 s)
            acquisition/verify: re-running the capability checks here
            acquisition/promote: registered after independent verification
            action.verified  SHA-256-Pruefsumme von zeus_acceptance.txt: 281a25eb...920db
```

The defect ZEUS recorded from the real failure was handed to Codex as the brief,
and the workspace was seeded from the broken version rather than from a blank
skeleton. What came back was the minimal correct fix: **73 lines, 0 content
differences from the pre-fault original.** Same `capability_id`, version
1.0.0 -> 1.0.1, no second capability registered, no phrase-specific workaround,
health back to HEALTHY, the capability's own five tests passing.

Then the engineer was removed: ZEUS stopped, `codex` replaced on PATH by the
logging shim, ZEUS restarted. A wording never used before --
*"welchen sha256 Wert hat die Datei X?"* -- resolved FOUND in 3.8 ms and ran the
repaired capability. **Zero Codex invocations.**

### Found by running it

**The brief could not be delivered.** `codex` on Windows is `codex.CMD`, and
cmd.exe truncates a command line at 8191 characters. The brief was being passed
as an argument. An acquisition brief of 7.4 KB got through; the repair brief of
the same capability -- the contract plus the defect plus the repair rules --
did not, dying in two seconds with `Die Befehlszeile ist zu lang.` The bigger
the job, the more certain the failure, which is exactly backwards. The brief now
goes on stdin, which is the CLI's own documented path and has no limit.

**A failure with no output said nothing.** The adapter reported stderr, and a
process that dies before writing anything has none -- so the log read
`expert: failed:` and stopped. That is the same line the `--full-auto` defect
produced, and it cost a separate investigation each time. The exit code, the
duration and the fact that there was no output are now the evidence, and the
acquisition step prints the blocker when the summary is empty.

**A broken capability was silently replaced by an unrelated one.** While health
was AT_RISK, the composer replanned around the failing checksum capability, ran
`file.read` plus a line/word counter, and reported *"Ziel erreicht ... GOAL
SATISFIED"* for a request that asked for a SHA-256 checksum. Fixed below.

## Replan semantic equivalence

`EXECUTION_VERIFIED` answers "did what we ran work". After a replan that is a
different question from "is what we ran the thing that was asked for", and only
the first one was ever being asked. Two true statements -- the steps ran, the
steps verified -- were being used to justify a third that was false.

The recorded plan, replayed exactly, before the fix:

```
GoalEvaluation(executed=True, execution_verified=True, goal_satisfied=True, reasons=[])
```

and after:

```
goal_satisfied=False
reasons=["the replacement is not the goal: 'Zeus, wie lautet die sha256
          Pruefsumme von zeus_acceptance.txt?' needs FILE_CHECKSUM, and what
          ran was file.read, capability:learned.ausgabe_dateipfad_zeilen"]
```

`replan_semantic_equivalence()` in `service/composer.py` runs before
`GOAL_SATISFIED` can be set, and only for a plan that actually replaced
something. It checks four things:

1. **the goal family is still served** -- the family comes from the step that
   failed (the planner had already decided what this goal needed and named it),
   falling back to the goal's own words only when that step is a capability
   whose identifier is a code name;
2. **the target is preserved** -- the file, path or drive the owner named is
   the one the replacement operated on;
3. **the output contract shows up** -- the family's result keys are present in
   the evidence, wherever the receipt exposes any;
4. constraints and required outcomes, which the surrounding gate already held.

`GOAL_FAMILIES` covers FILE_CHECKSUM, FILESYSTEM_SIZE, PLAY_MEDIA, APP_OPEN,
WEB_FETCH, CALENDAR_CREATE and IMAGE_GENERATE, with a held-out substitution per
family in `tests/test_replan_semantic_equivalence.py` -- a checksum answered by
a line count, a largest-folder question by a listing, playback by a web search,
an app by a web page, a page fetch by memory, a calendar entry by a note, an
image by a search and a file write.

Two things it deliberately does **not** do. A family it cannot classify raises
no complaint: a semantic guard that refuses everything it does not recognise is
an outage. And a first plan is left alone -- that plan *is* the planner's
reading of the goal, and second-guessing it with a keyword table would overrule
the better signal with the worse one. Only a replacement has something concrete
to stay equivalent to.

When no equivalent alternative exists the mission is BLOCKED with the reason and
the owner is told plainly: *"Steps ran, but the goal is NOT met: …"*. It never
degrades into a different task.

Re-run through production on the same request that produced the false positive,
with the fault re-introduced to recreate the AT_RISK window:

```
11:11:24  mission VERIFY: 1 done, 1 failed; EXECUTION_VERIFIED=False GOAL_SATISFIED=False
11:11:24  goal: NOT satisfied — required step capability:local.berechne.sha.256_pruefsumme
          failed: AttributeError…; the replacement is not the goal: … needs
          FILE_CHECKSUM, and what ran was file.read
11:11:24  mission FAILED
          answer: "Nicht geschafft — capability:… ist fehlgeschlagen"
```

The capability was then restored byte-exactly and earned HEALTHY back over three
real verified executions.

## Zero paid API

`CostPolicy` permits `local_model` and `subscription_cli`; `paid_api`,
`usage_credits` and `runpod` are denied and `fallbacks_for(SUBSCRIPTION_CLI)`
returns local inference only. The gateway registers exactly one provider. A
spent allowance is a state, never a licence to spend money — asserted in
`test_a_spent_subscription_never_falls_back_to_a_metered_channel`.

## Owner password

There is no plaintext to reach: the record is a scrypt verifier under a random
salt, DPAPI-wrapped on Windows, and the password exists only inside the frame
that checks it. Asserted by writing a real password through
`SecurityGate.setup()` and then searching every file under the state root for
it. What the sandbox adds on top: Codex may only *write* inside the capability
workspace, its brief names only `main.py` and `test_capability.py`, and its
environment carries no credential. It can read the disk like any local process —
stated plainly rather than claimed otherwise.

## Known limitations

Several tests construct `JarvisCore` on the default state root and write into
the owner's live `activity.jsonl`. Pre-existing, unrelated to this sprint, and
it means the full suite must not run during a live acceptance. Worth fixing.

`test_voice_service.py::test_posting_audio_requires_the_token` aborted its
connection once while other work ran concurrently on this machine, and passes
on its own. A flake, not a regression — recorded rather than quietly re-run.

The generalizer is syntactic: it recognises paths, filenames, quoted literals
and drive letters. A particular expressed some other way ("the third file in my
downloads") still reaches the engineer as part of the goal.

A confirmation ("Ja, bitte lerne das.") does not replay the offer it confirms;
the sentence itself became the goal. Now refused rather than built, but the
confirmation flow that dropped the pending text is untouched and still wrong.
