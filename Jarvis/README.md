# JARVIS

JARVIS keeps the existing Qwen-based brain as a high-level reasoner and adds a
separate developmental learning substrate. The foundation model proposes and
reasons; the learning layer stores experience, learns compact models, evaluates
skills, and tracks competence.

## HOW JARVIS LEARNS

### POMDP

JARVIS is modeled as a partially observable decision process:

```text
M = (S, A, O, T, R, gamma)
```

The real state `S` is not directly available. JARVIS receives observations `O`,
chooses actions `A`, observes consequences, and receives reward `R`. The internal
belief state is represented as:

```text
b_t = f_theta(o_1...o_t, a_1...a_(t-1), memory)
```

The current implementation exposes this boundary through `BeliefState`,
`LatentState`, and `BeliefStateEstimator`, so a future trainable recurrent or
transformer belief model can replace the simple encoder without changing the rest
of the learning system.

### State Representation

Numeric observations can be encoded by `ObservationEncoder`, a PyTorch
`nn.Module`:

```text
observation x -> encoder f_theta -> latent vector z in R^d
```

`LATENT_DIM` defaults to 256. `ObservationAutoencoder` provides a small
reconstruction objective when observations are numeric:

```text
L_reconstruction = ||x - x_hat||^2
```

Contrastive representation learning is prepared through an InfoNCE-style loss:

```text
L = -log(exp(sim(z_i,z_pos)/tau) / sum_j exp(sim(z_i,z_j)/tau))
```

### Experience

The central data unit is `Transition`:

```text
(o_t, z_t, a_t, r_t, o_(t+1), z_(t+1))
```

`Trajectory` groups transitions for full tasks. Dataset builders can transform
experience into SFT, preference, RL, world-model, and skill-learning formats.

### Reward

Rewards are multi-objective and stored as separate components:

```text
R_total =
w_task     * R_task
+ w_user   * R_user
+ w_acc    * R_accuracy
+ w_eff    * R_efficiency
+ w_novel  * R_novelty
+ w_learn  * R_learning
- w_error  * R_error
- w_risk   * R_risk
```

This avoids spreading hard-coded rewards through agent code.

### Replay

`ReplayBuffer` supports uniform replay plus prioritized experience replay:

```text
p_i = (|delta_i| + epsilon)^alpha
P(i) = p_i / sum_k p_k
w_i = (N * P(i))^(-beta)
```

Weights are normalized by the max sampled weight.

### World Model

`WorldModel` is a trainable PyTorch MLP:

```text
(z_t, action_embedding) -> z_(t+1)^pred
```

It can also predict reward. The objective is:

```text
L_world = lambda_transition * MSE(z_next_pred, z_next)
        + lambda_reward * MSE(r_pred, r_actual)
```

The demo in `training/world_model_demo.py` trains on synthetic transitions and
verifies that loss decreases.

### Policy And Value

`NeuralPolicy` implements `pi_theta(a | z)`. `NeuralValueFunction` implements
`V_phi(z)`, and `QNetwork` provides a small `Q_phi(z,a)` module. TD error is:

```text
delta_t = r_t + gamma * V(z_(t+1)) - V(z_t)
```

The demo in `training/learning_demo.py` trains policy/value parameters on a
synthetic reward task and reports reward before and after training.

### Intrinsic Motivation

Intrinsic reward infrastructure includes:

```text
R_novelty = distance(z_t, nearest_known_state)
R_curiosity = ||WorldModel(z_t,a_t) - z_(t+1)||^2
R_progress = error_old - error_new
```

Learning progress is clamped at zero so chaotic states do not stay attractive
only because they are unpredictable.

### Uncertainty

`UncertaintyEstimate` combines model uncertainty, memory uncertainty,
disagreement, and novelty:

```text
U = w_model*U_model + w_memory*U_memory + w_disagreement*U_disagreement + w_novelty*U_novelty
```

This is exposed as a decision signal for exploration policies.

### Skills And Options

Repeated successful trajectories can be converted into `DiscoveredSkill` objects
and then into hierarchical RL `Option` structures:

```text
omega = (I_omega, pi_omega, beta_omega)
```

The first implementation is heuristic and non-parametric; it does not pretend to
be neural skill discovery yet.

### Curriculum

`CurriculumManager` selects tasks in the zone of proximal development:

```text
0.60 < P(success) < 0.85
```

Difficulty combines normalized steps, tools, uncertainty, and novelty.
`DevelopmentalStageEvaluator` maps measured capability scores to stages from
`NEWBORN` through `SELF_IMPROVING`.

### Continual Learning

`ContinualLearner` tracks replay ratio, consolidation cadence, and
stability/plasticity balance. Forgetting is measured as:

```text
Forgetting(task_i) = max_previous_score(task_i) - current_score(task_i)
```

`MemoryConsolidator` turns repeated episodic patterns into semantic and
procedural candidates. Low-strength memories should be archived first, not
deleted blindly.

### Meta-Learning

`SelfModel` tracks capability estimates:

```text
attempts, successes, success_rate, mean_reward, uncertainty, trend
```

Trends use exponential moving averages:

```text
EMA_t = alpha*x_t + (1-alpha)*EMA_(t-1)
```

`LearningStrategy` and `MetaLearner` prepare future comparison of learning
strategies using:

```text
J(strategy) = mean_future_reward - lambda_compute * compute_cost
```

### Self-Improvement Foundation

`developer/self_programming.py` defines `CodeCandidate`, `FitnessResult`, and
`Experiment` for future controlled code-improvement experiments. These are only
data models. They do not autonomously edit production code.

### No Fake Learning

Memory/statistics updates are labeled as memory or statistics. Parameter
learning only refers to modules whose PyTorch weights change through gradient
descent: the observation encoder/autoencoder, policy/value networks, and world
model.

## JARVIS RUNTIME

`runtime/JarvisRuntime` is the first integrated living loop. `main.py` is now an
interface layer; the runtime owns the coding environment, observation adapter,
action generator, action selector, replay buffer, persistent experience store,
world model, policy, value function, reward engine, self model, and scheduler.

The CLI supports:

```text
/chat
/train
/eval
/status
/learning
/brain
/exit
```

Normal chat still uses Qwen, but Qwen is loaded lazily. Unit tests and coding
learning do not load the foundation model.

## REAL-WORLD LEARNING LOOP

The controlled v0.1 environment is `CodingWorld`. It operates only inside a
sandbox workspace and exposes a discrete action space:

```text
LIST_FILES, READ_FILE, SEARCH_TEXT, WRITE_FILE, PATCH_FILE,
RUN_TESTS, RUN_PYTHON, INSPECT_ERROR, FINISH
```

No arbitrary shell is exposed. File paths are rejected if they are absolute,
contain `..`, escape the sandbox, or traverse symlinks. Test subprocesses run
with timeout and captured output.

The closed loop is:

```text
Qwen or heuristic generator
  |
  v
ActionCandidates
  |
  v
Policy + WorldModel + risk/cost scoring
  |
  v
Action
  |
  v
CodingWorld
  |
  v
Objective Reward
  |
  v
Replay + SQLite Store
  |
  v
WorldModel / Value / Policy training
  |
  +---- back to future action selection
```

Mathematically:

```text
z_t = Encoder(Adapter(o_t))

pi_theta(a | z_t) = Policy(z_t)

z_next_pred, r_pred = WorldModel(z_t, action_embedding(a_t))

environment executes a_t

transition = (z_t, a_t, r_t, z_(t+1), done)

delta_t = r_t + gamma V_phi(z_(t+1)) - V_phi(z_t)

L_value = delta_t^2

L_policy = -log pi_theta(a_t | z_t) * stop_gradient(delta_t)

L_world = lambda_state MSE(z_next_pred, z_(t+1))
        + lambda_reward MSE(r_pred, r_t)
```

Replay priority is updated from actual learning error:

```text
priority = (|TD_error| + lambda_prediction * world_prediction_error + epsilon)^alpha
```

### Persistent Experience

`learning/experience/persistent_store.py` stores transitions in SQLite:

```text
task_id, episode_id, step, observation, action, reward components,
next observation, success, TD error, prediction error, priority,
timestamps, model versions
```

The in-memory `ReplayBuffer` remains the fast sampler; SQLite preserves
experience across process restarts.

### Coding Demo

Run:

```bash
python -m training.coding_learning_demo
```

The demo runs an identical before/after benchmark around controlled training
episodes. It reports episodes, success rate, mean reward, world-model loss,
value loss, policy loss, replay size, persistent experience count, and capability
trend.

The logs explicitly distinguish:

```text
PARAMETERS UPDATED:
WorldModel, Policy, ValueFunction

NOT UPDATED:
Qwen foundation model
```

## SEMANTIC PERSISTENT CODING AGENT V0.2

The productive `JarvisRuntime` can now use Qwen as the high-level action
generator when a `JarvisBrain` is injected. Qwen remains frozen: it proposes
structured `ActionCandidate` objects, but it does not receive rewards, hidden
tests, expected solutions, or optimizer updates. The heuristic generator remains
available for unit tests, fallback, and debugging.

The v0.2 runtime adds semantic state and semantic action representations:

```text
                    QWEN (frozen)
                         |
                         v
                Action Candidates
                         |

Observation -> Semantic Encoder -> z_t
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
              Policy        Q(z,a)      World Model
                +-------------+-------------+
                              |
                              v
                        Action Selection
                              |
                              v
                        Coding Sandbox
                              |
                              v
                         Real Outcome
                              |
                              v
                            Reward
                              |
                              v
                          Experience
                         /          \
                     SQLite        Replay
                                      |
                                      v
                              Gradient Training
                                      |
                                      v
                         learned small networks
```

### Semantic State

`ObservationAdapter` now emits both numeric features and a structured text view
of the task, workspace tree, source excerpts, test output, errors, previous
actions, and budget state. The text is encoded through the `SemanticTextEncoder`
interface:

```text
Task description
Source excerpts
Test output
Error message
Previous actions
Workspace summary
        |
        v
SemanticTextEncoder
        |
        v
semantic vector
        |
numeric features
        |
        v
ProjectionEncoder
        |
        v
z_t
```

Production can use `QwenHiddenStateTextEncoder`, which reuses the already loaded
local Qwen model in inference mode with hidden-state pooling and content-hash
caching. Tests use `DeterministicTextEncoder`, so unit tests do not load Qwen.

### Semantic Actions

`SemanticActionEncoder` represents concrete actions, not only action types. It
combines:

```text
action type one-hot
path text embedding
arguments text embedding
patch/diff text embedding
cost features
        |
        v
ActionProjection
        |
        v
a_t
```

This lets two `PATCH_FILE` actions with different diffs receive different
representations, losses, and future values.

### Concrete Action Value Learning

`ActionValueNetwork` learns `Q(z_t, a_t)` for each concrete candidate. Action
selection scores every Qwen candidate with policy prior, action value, world
model prediction, uncertainty/risk, information gain, cost, and confidence:

```text
score(a_i) =
  w_policy * log(pi(action_type_i | z_t) + eps)
+ w_q      * Q(z_t, a_i)
+ w_world  * predicted_reward(z_t, a_i)
+ w_info   * information_gain
- w_risk   * risk
- w_cost   * estimated_cost
```

Training mode may explore by epsilon-greedy or softmax selection over candidate
scores. Evaluation mode is deterministic and greedy.

### Losses

Stored replay entries include the raw observation feature vectors and the
reconstructable action features. During training, the current encoders are used
again, so the projection encoders receive gradients:

```text
z_t = E(o_t)
a_t = A(candidate_t)

Q_target =
  r_t + gamma V_target(z_next)

L_Q =
  (Q(z_t, a_t) - Q_target)^2

L_value =
  (V(z_t) - Q_target)^2

L_world =
  MSE(z_next_pred, z_next)
  + lambda_r MSE(r_pred, r_t)

L_policy =
  -log pi(action_type_t | z_t) * stop_gradient(Q_target - V(z_t))
```

Replay priority is updated after learning:

```text
priority_i =
  (abs(TD_error) + lambda_world * prediction_error + epsilon)^alpha
```

### Persistent Runtime

The runtime stores continuing learning state below:

```text
data/runtime/
data/checkpoints/
data/experience/
data/tensorboard/
```

Startup loads the latest valid checkpoint, optimizer state where available, the
SQLite experience store, and a deduplicated warm replay sample containing recent,
high-priority, failed, and successful transitions. Checkpoints are separated into
`latest` and `best`; a candidate is promoted to `best` only when holdout metrics
improve without safety regression.

### Train, Eval, And TensorBoard

Training and evaluation are separated:

```text
TRAIN:
experience collection, exploration, replay writes, optimizer.step()

EVAL:
model.eval(), torch.no_grad(), greedy actions, no replay contamination,
no curriculum update, no optimizer.step()
```

Run the v0.2 demo:

```bash
python -m training.coding_brain_v02_demo
```

TensorBoard logs are written under `data/tensorboard`:

```bash
tensorboard --logdir data/tensorboard
```

Logged metrics include train/eval reward, success rate, steps, replay size,
world loss, value loss, Q loss, policy loss, and capability trends.

### Sandbox Policy

Productive generated code execution requires `DockerSandboxBackend`. The Docker
backend is configured for no network, non-root execution, CPU and memory limits,
PID limits, timeouts, read-only hidden verifier mounts, no privileged mode, no
Docker socket, and a reduced environment. If a compatible container runtime is
not available, unsafe code execution is disabled instead of falling back to host
execution.

`LocalTestSandboxBackend` exists only for explicit unit tests and controlled
developer tests. It is not the production fallback.

### Parameter Status

```text
Foundation model:
Qwen: FROZEN

Trainable:
ObservationProjection: LEARNING
ActionProjection: LEARNING
WorldModel: LEARNING
ValueFunction: LEARNING
ActionValueNetwork: LEARNING
NeuralPolicy: LEARNING
```

## JARVIS V0.4 Capability Acquisition MVP

v0.4 adds a dedicated autonomous software-development engine for capability
acquisition. Greenfield capability creation no longer depends on the v0.3
low-level candidate-action loop. Qwen is used as a frozen high-level software
author: it returns complete structured file bundles and, on failure, complete
structured repaired files. Python orchestration controls materialization,
testing, verification, promotion, and reuse.

The v0.3 Policy/Q/Value/WorldModel stack remains intact for legacy experiments
and diagnostics. For v0.4 capability acquisition, learned components operate in
shadow mode and do not select production build actions.

Lifecycle:

```text
USER GOAL
  -> CapabilityResolver
  -> missing capability
  -> SkillSpecification
  -> executable contract
  -> staging workspace
  -> AutonomousSoftwareEngineer
  -> complete implementation bundle
  -> deterministic materialization
  -> automatic public tests
  -> Jarvis-owned internal adversarial tests
  -> independent reviewer
  -> failure-driven full-file repair loop
  -> external hidden verifier
  -> optional blind generalization repair
  -> versioned promotion
  -> CapabilityRegistry
  -> execute original request
  -> second call uses installed capability directly
```

The software engineering state machine is:

```text
UNDERSTAND -> PLAN -> IMPLEMENT -> TEST -> DIAGNOSE -> REVISE
           -> TEST -> VERIFY -> COMPLETE
```

On failure, it enters `FAILED` after the repair budget is exhausted or after an
unsafe generated path/protected file attempt.

### Internal QA and Review

The v0.4 engineering path now separates roles while keeping Qwen frozen:

```text
Architect:      compiles SkillSpecification into an executable contract
Implementer:    writes complete file bundles
TestEngineer:   creates Jarvis-owned public-contract tests outside the editable tree
Reviewer:       approves/rejects and may recommend extra black-box tests
Repairer:       receives exact public/internal failures and reviewer findings
Orchestrator:   runs tests, protects paths, verifies hidden acceptance, promotes
```

The generated internal QA suite is stored outside the implementation workspace
and mounted/read from there during execution. It includes ordinary cases,
boundary cases, empty inputs, duplicate/order/tie cases where relevant, and
small deterministic metamorphic checks. The implementation can read neither
hidden verifier code nor hidden expected outputs.

Capability promotion now requires all of:

```text
public_success
internal_verification_success
reviewer_approved
hidden_success
protected files pristine
manifest validates
permission policy passes
```

If public tests, internal QA, and reviewer approval pass but the hidden verifier
fails, Jarvis may run a bounded blind-generalization repair. The repair prompt
contains only "external acceptance verification failed" plus the visible
contract, current implementation, public/internal results, and reviewer
findings. Hidden verifier source, inputs, expected outputs, and traceback remain
secret.

The persistent registry is stored as JSON and records:

```text
capability_id, description, version, status, entrypoint,
input_schema, output_schema, permissions_required, dependencies,
source_location, tests_location, creation_metadata, validation_status
```

Staged skills are created under `skills/_staging/<capability>/<candidate>/`.
Promoted skills are copied into
`skills/installed/<capability_id>/<version>/` and registered only after:

```text
syntax/build succeeds
public tests pass
hidden verifier passes
protected files remain pristine
manifest validates
permission policy passes
```

Hidden verifier workspaces are separate from the editable skill workspace. The
software engineer can see public tests and staged source files, but not hidden
verifier source or hidden expected outputs.

Permission policy is conservative in v0.4. Safe local capabilities can be
developed and tested automatically. Network, browser, credential, email, or
other side-effect permissions are blocked with a structured permission decision
instead of being silently granted.

Development memory is stored in JSONL and records task/spec fingerprints,
architecture plan, generated implementation, failures, diagnoses, repairs,
final code, public/internal/reviewer/hidden/promotion/execution/second-call
results, token usage, normal repair cycles, and blind repair cycles. Public
test success alone is not treated as successful engineering memory. Final
successful examples are those that survive the full lifecycle; failed and
partial experiences remain useful repair memory. This is practical engineering
memory, not Qwen fine-tuning.

Acquisition trajectories are appended to
`data/capabilities/acquisition_trajectories.jsonl` by default. Each record
contains the goal, gap detection result, specification, plan, implementation,
public test result, repair history, hidden verification, promotion decision,
original execution result, second-call reuse result, and final outcome.

## Autonomous Repository Engineering

The same high-level engineering engine is available for safe repository
self-improvement experiments through `development.RepositoryEngineer`.

```text
SelfImprovementGoal
  -> isolated git worktree
  -> targeted repository context
  -> structured multi-file proposal
  -> path/protected-file validation
  -> targeted tests / full acceptance commands
  -> git diff + deterministic evidence
  -> SELF_IMPROVEMENT_CANDIDATE_READY
```

The live checkout is not overwritten. Candidates are produced in disposable
worktrees, protected files are compared against the source checkout, and final
promotion still requires explicit human approval. Repository engineering
trajectories are persisted as JSONL for future SFT/LoRA/preference/RL data, but
Qwen is not trained in this milestone.

Run the mock v0.4 demo without loading Qwen:

```bash
python -m training.capability_acquisition_v04_demo --mock-brain
```

Cheapest real-Qwen smoke path through an OpenAI-compatible endpoint:

```bash
python -m training.capability_acquisition_v04_demo --brain-provider openai_compatible --task-count 3 --benchmark-dir data/benchmark_runs/v04_qwen_software_engineer_smoke_01
```

Safe repository self-improvement experiment with real Qwen:

```bash
python -m training.self_improvement_demo --real-brain --brain-provider openai_compatible --goal "Improve semantic capability reuse" --benchmark-dir data/benchmark_runs/self_improvement_qwen_smoke_01
```
