# Jarvis Autonomous Core — build report

**Target machine:** Windows 10, Intel i7-class, 8 CPUs, 16 GiB RAM, NVIDIA GTX 1070 (8 GiB, driver 561.09), Ollama 0.30.10, Python 3.14.3
**Branch:** `adaptive-brain-v1`
**Baseline before this work:** commit `0c5444d`

---

## 1. What changed, in one paragraph

Jarvis could not previously develop software without remote inference, its patch
engine had regressed to a single edit dialect that broke 13 tests, and its status
line reported models as online that could not generate a single token. It now
runs a persistent, autonomous project loop on the local `qwen2.5-coder:7b`
model, edits its own repository and other repositories through a deterministic
atomic edit engine, acquires and registers new capabilities, promotes verified
changes into itself with automatic rollback, and remembers what worked across
restarts. No paid service is enabled, and none is required.

---

## 2. Final architecture

```
                       ┌──────────────────────────────────────┐
                       │  jarvis/cli.py — the console         │  one client, not the system
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │  core/kernel.py — composition root   │  the future service boundary
                       └───┬──────────┬──────────┬────────────┘
                           │          │          │
        ┌──────────────────▼───┐  ┌───▼──────┐  ┌▼───────────────────────┐
        │ brain/               │  │ tools/   │  │ projects/              │
        │  tiers, router,      │  │ registry │  │  models, store, engine │
        │  providers, ollama,  │  │ builtin  │  │  (the control loop)    │
        │  resources           │  │ web      │  └──┬─────────────────────┘
        └──────────────────────┘  └────┬─────┘     │
                                       │           │
                    ┌──────────────────▼───────────▼──────────────┐
                    │ development/edit_engine.py                  │  the ONLY writer
                    └──────────────────┬──────────────────────────┘
                                       │
   ┌───────────────────┬───────────────┼───────────────┬────────────────────┐
   │ development/      │ capabilities/ │ deployment/   │ knowledge/         │
   │ repository_       │ service       │ promotion     │ graph, memory      │
   │ engineer          │               │               │                    │
   │ (self-dev)        │ (acquire/run) │ (promote/     │ (thought palace)   │
   │                   │               │  rollback)    │                    │
   └───────────────────┴───────────────┴───────────────┴────────────────────┘
                    persona/  ·  training/dataset_export.py
```

### The control loop

`projects/engine.py` runs

```
INVESTIGATE → DECOMPOSE → EXECUTE → VERIFY → DIAGNOSE → REPLAN → … → ACCEPT
```

and stops for exactly four reasons, matching the brief: acceptance proved, user
cancelled, budget exhausted, or a genuine external blocker. Phase selection is
deterministic rather than model-chosen, because a weak model asked to pick its
own next phase loops between planning and re-planning without executing.

A project is **accepted only when an acceptance criterion with a runnable
command exits zero**. A criterion with no runnable check can never be satisfied,
so the system cannot declare victory on its own say-so. Everything durable —
goal, requirements, acceptance criteria, tasks, findings, decisions,
experiments, blockers, artifacts, full trajectory — is saved after every step,
so a killed process loses at most one step.

---

## 3. Design decisions that carry weight

**The model proposes; deterministic code disposes.** Every effect on the world
goes through the tool registry, which is the single place enforcing
permissions, timeouts, output caps and the audit trail. Every byte written goes
through the edit engine, which is the single place enforcing containment,
protected paths, atomicity and budgets.

**Make the wrong thing unrepresentable, not discouraged.** This was learned the
hard way, three times:

- Told plainly not to rewrite a 189-line file, the model did it sixteen times
  running — because the schema still offered a `content` field and constrained
  decoding will use whatever the schema describes. Removing the field fixed it
  immediately.
- Told in an error message to use `write_file` instead of fighting an anchor,
  the model ignored the advice eight times running. Withdrawing `apply_edits`
  from the offered tool list fixed it.
- The tool that reports whether a program is installed was called `which`, and
  the model called `shutil.which` and then subscripted the result as
  `player["found"]` — the shape *the tool* returns. It is now `find_program`.

**A failed attempt is not a failed task, and a failed task is not a failed
project.** Recoverable edit failures are re-prompted with the current file
content; if corrections run out, one *cycle* is spent, not the run. Policy
violations are the exception and stop the run immediately.

**Repair what is deterministically repairable.** A replacement written
flush-left into an indented anchor is re-indented automatically when that is the
only thing wrong — by a wide margin the most common way a small model breaks an
anchored edit. The repair is timid: Python only, only when the verbatim result
does not parse, only when re-indenting makes it parse.

**Verification never trusts the thing being verified.** Capability acquisition
runs the project loop, and then a *separate* check decides whether the result is
real. The seeded skeleton carries a marker that fails both gates, because
without it the scaffold passed its own verification and a capability that did
nothing registered as working.

---

## 4. Models and configuration

| Tier | Model | Provider | Enabled | Paid |
|---|---|---|---|---|
| `FAST_LOCAL` | `qwen3:4b-instruct` | ollama | yes | no |
| `BUILD_LOCAL` | `qwen2.5-coder:7b-instruct-q4_K_M` | ollama | yes | no |
| `VISION_LOCAL` | — | ollama | no | no |
| `EMBEDDING_LOCAL` | — | ollama | no | no |
| `SELF_HOSTED` | — | openai_compatible | no | no |
| `EXPERT_CLOUD` | — | openai_compatible | **no** | yes |

Configuration is layered: defaults → `Jarvis/config/models.json` → environment
(`JARVIS_<TIER>_<FIELD>`, e.g. `JARVIS_BUILD_LOCAL_MODEL`). Nothing in the
autonomous engine names a model.

The Ollama provider uses the **native `/api/chat` endpoint**, not the
OpenAI-compatible one, because the compatible dialect silently drops `num_ctx`,
`keep_alive` and schema-constrained decoding — the three settings that decide
whether an 8 GiB card stays usable.

### Honest availability

A tier is reported `ONLINE` only after (1) the endpoint answers, (2) the model
is present in its catalog, and (3) a real generation succeeds. This was not
pedantry: verified on this machine, Ollama answers `GET /v1/models` with HTTP
200 while the requested model is absent, so the previous check reported a model
that could not run a single token as online. The three failure modes are now
`PROVIDER_UNREACHABLE`, `MODEL_MISSING` and `GENERATION_FAILED`.

---

## 5. Measured performance on this machine

`python -m jarvis.tune_resources` runs real generations at each candidate
context size. Measured, not assumed:

| Tier | ctx | tok/s | load | VRAM free |
|---|---|---|---|---|
| BUILD_LOCAL | 4096 | 31.85 | 45.2 s | 2939 MiB |
| BUILD_LOCAL | 8192 | 32.65 | 46.6 s | 2579 MiB |
| BUILD_LOCAL | 12288 | 32.26 | 46.7 s | 2347 MiB |
| BUILD_LOCAL | 16384 | 32.54 | 46.4 s | 2257 MiB |
| **BUILD_LOCAL** | **24576** | **32.48** | 46.7 s | **1801 MiB** ← chosen |
| BUILD_LOCAL | 32768 | 32.30 | 46.8 s | 1345 MiB (below reserve) |
| FAST_LOCAL | 4096 | 55.89 | 28.9 s | 4426 MiB |
| FAST_LOCAL | 8192 | 53.13 | 27.7 s | 3763 MiB |
| FAST_LOCAL | 12288 | 56.80 | 28.0 s | 3180 MiB |
| **FAST_LOCAL** | **16384** | **56.34** | 28.3 s | **2596 MiB** ← chosen |
| FAST_LOCAL | 24576 | 55.67 | 29.3 s | 1432 MiB (below reserve) |
| FAST_LOCAL | 32768 | 30.94 | 30.6 s | 1154 MiB (throughput collapse) |

**Chosen policy:** BUILD_LOCAL 24576, FAST_LOCAL 16384, one concurrent
generation, 1638 MiB VRAM reserved for the desktop (20% of the card).

The first version of the chooser took 32768 for both, because 31 tok/s cleared
its absolute floor. Losing 45% of throughput is memory pressure, so a candidate
must now also retain 85% of the tier's best measured throughput, and VRAM is
reserved in proportion to the card rather than as a flat gigabyte.

---

## 6. Startup

```powershell
cd D:\Jarvis_recovery_20260823\repo\Jarvis

# Prerequisites, once
ollama pull qwen3:4b-instruct
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Measure this machine, once (a few minutes; loads each model per context size)
python -m jarvis.tune_resources

# The console
python -m jarvis

# One-shot self-development against this repository
python -m jarvis.self_develop --repo . --tier BUILD_LOCAL ^
  --goal "<what to change>" --allowed-path <path> --protected-path tests ^
  --test-command "python -m pytest -q" --max-cycles 4

# Tests
python -m pytest tests/ -q -m "not live"     # deterministic
python -m pytest tests/ -q -m live           # needs Ollama answering
python -m jarvis.record_evidence             # the real-hardware scenarios
```

Console commands: `/status`, `/projects`, `/project`, `/work`, `/new`, `/say`,
`/capabilities`, `/learn`, `/use`, `/remember`, `/recall`, `/persona`, `/tune`,
`/help`, `/quit`.

---

## 7. Acceptance tests

Suite: `Jarvis/tests/test_acceptance.py`, named A–O after the brief.

Suite: `Jarvis/tests/test_acceptance.py`, named A–O after the brief.
Deterministic tests run against scripted models; the ones needing real
inference are marked `live` and read evidence recorded by
`python -m jarvis.record_evidence` into `Jarvis/data/acceptance_evidence/`.

```
python -m pytest tests/ -q -m "not live"     →  547 passed, 5 deselected
python -m pytest tests/test_acceptance.py -q -m "not live"  →  28 passed
```

| # | Requirement | Result | Evidence |
|---|---|---|---|
| **A** | Local self-patch | **pass** (deterministic) + live, see below | `test_A_self_patch_produces_a_verified_candidate`; `A_self_patch_live.json` |
| **B** | Bad patch recovery | **pass** | `test_B_an_invalid_first_patch_does_not_end_the_mission`, `test_B_recovery_covers_every_way_a_local_model_gets_an_edit_wrong` |
| **C** | Test failure recovery | **pass** | `test_C_valid_but_wrong_code_is_diagnosed_from_evidence_and_repaired`, `test_C_the_diagnosis_sees_the_real_test_output` |
| **D** | Multi-file development | **pass** | `test_D_a_goal_spanning_two_source_files_succeeds`, `test_D_a_multi_file_edit_is_all_or_nothing` |
| **E** | New project | **pass** (deterministic **and** live) | `test_E_a_new_application_is_built_in_an_isolated_workspace`; `E_new_project_live.json` |
| **F** | Capability acquisition | **pass** (deterministic); live: see below | `test_F_a_missing_capability_is_acquired_verified_registered_and_reusable` |
| **G** | Complex project | **pass** | `test_G_a_multi_component_pipeline_is_built_and_verified`, `test_G_requirements_accumulate_across_interactions` |
| **H** | Protected path | **pass** | `test_H_a_protected_file_is_refused_and_stays_byte_identical`, `test_H_the_repository_engineer_rejects_a_protected_edit` |
| **I** | Atomicity | **pass** | `test_I_a_failed_multi_edit_leaves_no_partial_mutation`, `test_I_atomicity_holds_for_every_rejection_kind` |
| **J** | Promotion | **pass** | `test_J_a_verified_candidate_is_promoted_into_the_installation`, `test_J_the_real_health_check_exercises_the_whole_kernel` |
| **K** | Rollback | **pass** | `test_K_a_failing_health_check_rolls_back_automatically`, `test_K_a_rollback_that_itself_fails_is_escalated` |
| **L** | Persistence | **pass** | `test_L_projects_memory_and_capabilities_all_survive_a_restart`, `test_L_a_paused_project_can_be_picked_up_later` |
| **M** | No cloud | **pass** | `test_M_*` ×3; no paid LLM or search credential exists in this environment at all (`Research_live.json`) |
| **N** | Real hardware | **pass**, with caveats below | `N_build_local_probe.json`, `A_self_patch_live.json`, `E_new_project_live.json`, `F_capability_live.json` |
| **O** | Responsiveness | **pass** | `test_O_*` ×3, asserted against the measurements in `config/resources.json` |

---

## 8. Security model

| Control | Where |
|---|---|
| Workspace containment, `..`/absolute/symlink refusal | `edit_engine.PathPolicy`, `tools.builtin.resolve_readable` |
| Protected paths, non-retryable | `edit_engine.PathPolicy`, `RepositoryEngineer._assert_changed_files_allowed` |
| Atomic all-or-nothing writes | `edit_engine.EditEngine._commit` |
| Change budgets (ops, lines, file size, shrink) | `edit_engine.EditBudget` |
| Tool risk levels + approval gate | `tools.registry.ToolPolicy` |
| Executable allow-list | `tools.builtin.DEFAULT_COMMAND_ALLOWLIST` |
| Process timeouts and output caps | `tools.registry`, `tools.builtin.run_command` |
| Credentials excluded from subprocesses | `_safe_command_env`, `safe_environment` |
| Credentials redacted from logs and datasets | `AuditLog.redact`, `dataset_export.redact` |
| Dependencies into a project venv, never system Python | `tools.builtin.install_packages` |
| Loopback/private addresses refused | `tools.web.DocumentFetcher` |
| Audit trail of every tool call | `tools.registry.AuditLog` |
| Audit trail of every promotion | `deployment.promotion.PromotionAudit` |

Self-modification additionally runs in an isolated git worktree; the live tree is
never the working surface.

---

## 9. Recovery procedure

**A promotion went wrong.** Rollback is automatic on a failed verify, restart or
health check. If the rollback itself failed the outcome is `ROLLBACK_FAILED`,
never swallowed:

```powershell
python -c "from deployment.promotion import PromotionAudit; import json; ^
  print(json.dumps(PromotionAudit('data/promotions.jsonl').history(limit=5), indent=2))"
git -C . status
git -C . reset --hard <known_good_revision>   # printed in the audit record
```

Snapshots of every promoted file are under `Jarvis/data/snapshots/<id>/`.

**A project is stuck.** `/project <id>` shows tasks, blockers and the last
evidence. Projects are plain JSON under the state root and can be edited by
hand. `/work <id>` resumes.

**A capability misbehaves.** Capabilities run in a subprocess and cannot take
Jarvis down. Disable one with
`CapabilityRegistry(...).disable("<id>", "reason")`; the installed copy stays on
disk for inspection.

**Jarvis will not start.** `python -m jarvis.tune_resources --show` and
`python -c "from core.kernel import JarvisKernel; print(JarvisKernel().status())"`
report state without needing the console. Deleting
`Jarvis/config/resources.json` reverts to conservative VRAM-derived defaults.

---

## 10. Migrating inference to a server

Nothing in the autonomous engine names a model or a host, so the migration is
configuration:

1. Run the inference server on the target machine (Ollama, vLLM, llama.cpp).
2. Point the tier at it — either in `Jarvis/config/models.json`:
   ```json
   {"tiers": {"BUILD_LOCAL": {
       "provider": "openai_compatible",
       "model": "qwen2.5-coder-32b-instruct",
       "base_url": "http://jarvis-server:8000",
       "context_window": 32768, "enabled": true}}}
   ```
   or per-run: `JARVIS_BUILD_LOCAL_BASE_URL=... JARVIS_BUILD_LOCAL_MODEL=...`
3. `python -m jarvis.tune_resources --tier BUILD_LOCAL` on the server.
4. Nothing else changes. `provider_for_spec` is the only place that maps a
   provider name to an implementation; adding a backend is one branch there.

For the portable-client topology, `JarvisKernel` already exposes the surface an
API would: create a project, work on it, report status, list and run
capabilities. Tool execution is a separate concern with its own policy object,
so tools can later run near the files while inference happens elsewhere.

---

## 11. Known limitations

These are real and stated plainly.

1. **The 7B model is the binding constraint, not the architecture.** It
   succeeds reliably on focused tasks (a one-line self-patch, a two-file
   utility) and is marginal on tasks needing several interacting components in
   one run. The loop recovers from its mistakes; it cannot supply competence the
   model does not have. A larger model is a configuration change.

2. **Semantic retrieval is lexical by default.** It bridges inflection
   (`play`/`plays`) but not vocabulary (`music`/`audio`). Capabilities therefore
   declare the words people will use. `OllamaEmbedder` is implemented for when
   an embedding model is configured, but none is pulled here.

3. **Side-effecting capabilities are verified by dry run.** Acquisition can
   prove the player was found, the file exists and the command is well-formed.
   It cannot prove sound came out of the speakers. This is stated rather than
   papered over.

4. **Research is keyless and therefore fragile.** DuckDuckGo HTML scraping
   degrades to an empty result list on a layout change rather than raising.
   Offline, research tools report a clean `offline` failure and the loop
   continues on local knowledge.

5. **`VISION_LOCAL` is wired but unconfigured.** The tier, provider selection
   and health probe exist; no vision model is pulled on this machine, so nothing
   image-based has been exercised end to end.

6. **Dataset export produces material, not a fine-tune.** Deliberate: the brief
   forbids online weight updates, and a supervised run belongs offline and
   reviewed.

7. **One machine, one topology.** The service boundaries are drawn and the
   kernel surface is serialisable, but no HTTP transport has been written, so
   the remote split is designed rather than demonstrated.

8. **The Windows environment needed two workarounds** that are documented in
   code: the default pytest temp root on this machine has an ACL that denies
   `scandir` (see `Jarvis/conftest.py`), and an unrelated top-level `tests`
   package in user site-packages shadows candidates' own modules unless
   `PYTHONPATH` pins the workspace first.

---

## 12. Next improvements, in priority order

1. **A larger BUILD_LOCAL.** Everything else is downstream of model competence.
   A 14B–32B coder on a bigger card, or on the home server, is the single
   highest-leverage change, and costs one config edit.
2. **An embedding model for `EMBEDDING_LOCAL`**, removing the vocabulary
   limitation in retrieval.
3. **The HTTP service boundary**, turning the kernel surface into an API so the
   portable-client topology can be exercised rather than merely designed.
4. **Structured AST edits** for renames and signature changes, where a
   text anchor is inherently the wrong tool.
5. **Parallel candidate exploration** — several approaches in separate
   worktrees, judged on deterministic evidence. Requires either a second GPU or
   the server, given the measured one-model-at-a-time constraint.
6. **A first supervised fine-tune** from the exported repair trajectories, then
   measuring whether the loop needs fewer corrections.
7. **`VISION_LOCAL` end to end**, which is the prerequisite for the screen-capture
   class of project in the brief.
