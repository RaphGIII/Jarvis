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

Baseline as of this session: **177 passed, 5 skipped** (collected count
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
- Broader mission checklist (capability registry persistence, self-
  development worktree flow, permission gates, provider-failure/resume,
  local-only mode demonstration, etc.) has NOT been re-verified end-to-end
  in this session — check `runtime/`, `capabilities/registry.py`,
  `development/repository_engineer.py`, and existing tests under
  `Jarvis/tests/` for what's already implemented before assuming anything
  is missing.
