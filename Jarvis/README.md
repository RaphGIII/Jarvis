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
