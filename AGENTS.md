# AGENTS.md — Jarvis Core V1 Autonomous Build (persistent memory)

This file is repository-specific memory for the ongoing "JARVIS CORE V1"
mission. Read this before doing fresh reconnaissance — it should save you
from repeating work already done in prior sessions.

## Repo layout

- Git root: `/projects/repo` (this file lives here).
- Jarvis source: `/projects/repo/Jarvis`.
- Protected baseline tag (NEVER move/delete): `jarvis-pre-autonomy-20260822`
  == commit `d23e8edf9ad27949b50fbc48bb65b2fbf257b31f` (also `origin/main`).
- Active development branch: `jarvis-core-v1-autonomous`.
- `Jarvis/data/benchmark_runs/**` contains large pre-existing generated
  worktree artifacts that are tracked in git from before this mission
  started (not something this session created) — leave them alone unless
  a task specifically concerns them.

## Test commands

```bash
cd /projects/repo/Jarvis
python -m pytest tests/ -q         # full suite; ~50-60s
```

Baseline as of this session: **186 passed, 5 skipped** (collected count
grows as tests are added — don't assume a fixed number, always re-verify).
The mission brief mentions "178 collected" from an earlier point in time;
that number is stale, verify freshly each session.

## Live end-to-end testing against a local LLM

There's a local OpenAI-compatible inference server used for real
(non-mocked) capability-acquisition tests, typically at
`http://127.0.0.1:8123`. Point the brain provider at it via env vars:

```bash
export JARVIS_BRAIN_PROVIDER=openai_compatible
export JARVIS_BRAIN_BASE_URL=http://127.0.0.1:8123
export JARVIS_BRAIN_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct
export JARVIS_BRAIN_TIMEOUT=600   # small local models are slow; default 60s times out
```

Then drive `runtime.capability_runtime.CapabilityAcquisitionRuntime.handle_goal(...)`
directly from a throwaway script (see pattern below). Use
`environments.coding.sandbox_backend.LocalTestSandboxBackend()` for the
backend and a fresh `data/<scratch>/` directory for `data_dir`/`skills_root`
so runs don't collide; delete the scratch dir when done (it's untracked,
matches nothing special, just don't leave junk around).

`Qwen2.5-0.5B-Instruct` (non-coder) could not reliably produce a single
valid JSON file bundle even after 4 attempts — prefer
`Qwen2.5-Coder-0.5B-Instruct` for local smoke tests of the code-generation
path. Model weights get cached under
`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/`.

**Important finding**: weak/small local models are an excellent bug-finding
tool. They reliably hit prompt/schema edge cases that mocks and strong
models (Claude) never trigger, because they take LLM instructions much
more literally and fail more creatively. When auditing the capability
pipeline, prefer testing against a real small local model over adding more
mocks.

## Bugs found & fixed via live small-model testing (this mission)

1. **Placeholder echo bug** (`development/software_engineer.py`, repair
   prompt schema): used the descriptive string `"content":"complete
   corrected file"` as a JSON-schema example instead of a generic `"..."`
   placeholder (the implementation prompt already used `"..."` correctly).
   Weak models parroted the placeholder text back verbatim as the file's
   *content* instead of generating real code. Fixed by using `"..."`
   consistently. Lesson: **never put a plausible-looking natural-language
   phrase in a JSON-schema example sent to a small LLM** — use an
   unambiguous placeholder token instead.

2. **No retry loop in `SkillSpecificationGenerator`**
   (`capabilities/specification.py`): unlike
   `AutonomousSoftwareEngineer._request_valid_bundle` (which retries
   `1 + structured_regeneration_attempts` times with feedback on failure),
   the spec generator gave up after a single malformed/invalid response and
   silently fell back to a hardcoded generic spec whose public test was
   `{"input": {"text": "hello"}, "expected_keys": ["result"]}` — completely
   unrelated to most goals (e.g. "double an integer"), which made the
   acquisition **structurally unwinnable** regardless of implementation
   quality. Fixed by adding a bounded (`attempts=3` default), feedback-driven
   retry loop mirroring the existing pattern, and by pointing the prompt at
   the requirement that public-test keys must literally match `payload`
   keys the implementation will use.

3. **Missing structural validation on `public_tests` entries**
   (`capabilities/models.py: SkillSpecification.validate()`): a test case
   like `{"value": "10"}` (no `"input"` key at all) passed validation, then
   got rendered by `capabilities/workspace.py:_render_public_tests` into a
   harness that calls `main.run({})` — never exercising real behavior, and
   for implementations that (correctly) raise on missing required keys,
   causing spurious hard failures. Added a check that every public-test
   case has an `"input"` dict and at least one assertion mechanism
   (`expected` / `expected_keys` / `raises`).

All three were fixed in commit `89d6f23` (see `git log` on
`jarvis-core-v1-autonomous`) with tests in
`Jarvis/tests/test_skill_specification_generator.py`.

## Known-good architecture behavior (do not "fix" this)

When a local 0.5B model produces a **lazy placeholder stub**
(e.g. `return {'result': 'result_value'}`) that happens to satisfy a loose
public test (only checking key presence, not value), the **reviewer +
hidden verifier correctly reject it** and the capability is NOT promoted
(`success: false`, `promoted: false`). This is the acceptance-gate safety
net working as designed per the mission's Quality Gates ("Promotion must
require deterministic evidence"). Don't weaken the reviewer/hidden-verifier
gate to make weak-model runs "succeed" — that would be fabricating success.
If a goal fails end-to-end only because the *model* is too weak to write
correct logic within the repair budget (not because of a pipeline bug),
that's an expected, honest outcome for a 0.5B model, not a regression.

## Mission status snapshot (update this section as you make progress)

See the full mission brief for the 20-point requirements list and 8
black-box acceptance tests (TEST 1-8). As of this session:
- Capability-acquisition pipeline bug-hunting via live small-model runs:
  3 real bugs found and fixed (see above), full regression suite green.
  Committed as `89d6f23`.
- TEST 2 (research-based capability acquisition): `capabilities/research.py`
  (`CapabilityResearcher`) added, wired into `capabilities/specification.py`
  and `runtime/capability_runtime.py`, tests in
  `tests/test_capability_research.py`. Committed as `468d2a7`.
- TEST 4 (self-development against the real Jarvis repo, live Qwen2.5-Coder
  -0.5B-Instruct local LLM): ran `runtime/self_developer.py` end-to-end
  against `/projects/repo/Jarvis` with the goal "make pyflakes clean" on 6
  real files with genuine unused-import/unused-var/f-string lint issues.
  The run executed the full loop (preflight -> investigate x4 rounds -> plan
  -> 4 patch/repair cycles -> targeted tests each cycle) and correctly
  self-rejected (`SELF_DEVELOPMENT_CANDIDATE_REJECTED`) because the 0.5B
  model could never produce a pyflakes-clean bundle within the cycle
  budget — an honest failure of model capability, not a pipeline bug (see
  "Known-good architecture behavior" above; same principle applies here).
  Isolation held perfectly: `git status` on `/projects/repo/Jarvis` showed
  **zero changes** to the real source tree even though the candidate
  worktree ended up with corrupted test files.
  - **Real bug found and fixed via this run**: an accidental CLI
    `--resume` invocation (from a process that got killed and manually
    resumed with different args, omitting `--protected-path tests`)
    revealed that `RepositoryEngineer.improve()` computed a
    `goal_fingerprint` at `WORKTREE_CREATED` but never validated/used it on
    resume — a resumed run silently trusted whatever `SelfImprovementGoal`
    the caller reconstructed instead of the originally-checkpointed one, so
    forgetting a flag like `--protected-path` on resume silently dropped
    that permission boundary for the rest of the run (this is exactly how
    the candidate worktree's `tests/*.py` files got clobbered — the LLM
    was allowed to "repair" by rewriting test files once `protected_paths`
    was gone). Fixed by persisting the full goal dict alongside the
    fingerprint and always reloading the checkpointed goal (with its
    tests/full_tests/benchmarks) on resume, ignoring the CLI-reconstructed
    one. Regression test:
    `tests/test_self_developer_production_gate.py::test_self_developer_resume_keeps_original_goal_protected_paths`.
    Also fixed the 6 real pyflakes findings by hand (unused imports in
    `capabilities/executor.py`, `capabilities/registry.py`,
    `development/software_engineer.py`, `runtime/jarvis_runtime.py`; unused
    var in `environments/coding/reward.py`; f-string-no-placeholder in
    `runtime/self_developer.py`). Committed as `a57d0bc`. Full suite:
    186 passed, 5 skipped, no regressions.
  - **Operational lesson**: when driving `self_developer.py` interactively
    across multiple shell turns, always resume with the *exact same* CLI
    flags (or better, rely on the fix above which now makes this safe
    regardless). A single self-developer run can take 15-20+ minutes
    end-to-end against a 0.5B local model (each LLM call is slow); launch
    it backgrounded (`nohup ... &`) and poll
    `<run_dir>/self_developer_checkpoint.json`'s `last_stage` field rather
    than blocking a terminal call on it.
- Broader mission checklist items not yet re-verified end-to-end this
  session: capability registry persistence/reuse across restart (TEST 1,
  5), multi-file software test (TEST 3), permission gate (TEST 6),
  provider-failure/resume (TEST 7 — related code exists, see
  `_provider_failure_payload`/`RepositoryStage.PAUSED`/`resume_command` in
  `development/repository_engineer.py`, but not freshly re-demonstrated),
  local-only mode (TEST 8). Check `runtime/`, `capabilities/registry.py`,
  `development/repository_engineer.py`, and existing tests under
  `Jarvis/tests/` for what's already implemented before assuming anything
  is missing.
