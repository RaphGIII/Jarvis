# Jarvis Architecture

What the pieces are, why the boundaries are where they are, and which decisions
were forced by evidence rather than taste.

---

## Shape

```
                        ┌──────────────────────────────────┐
   browser / TV ───SSE──│                                  │
   CLI          ───HTTP─│        JARVIS CORE SERVICE       │
   future device ───────│    (service/core.py — no UI)     │
                        └───────────────┬──────────────────┘
                                        │
   ┌──────────┬──────────┬──────────────┼──────────┬──────────┬──────────┐
   │          │          │              │          │          │          │
Conversation Projects Capabilities  Knowledge  Brain      Expert    Speech
 + Persona    Engine    Engine        Graph     Router    Gateway   Pipeline
                │          │              │        │          │          │
                └──────────┴──────┬───────┴────────┘          │          │
                                  │                           │          │
                            Tool Runtime              CostPolicy    venv worker
                         (workspace + desktop)        (gate)        (whisper/piper)
```

The rule that keeps this honest: **`service/core.py` may not import anything
from a UI, and no core method formats anything for display.** It returns data
and publishes events. Every client — the browser, the CLI, a future HDMI box —
calls the same methods and renders them differently.

---

## Decisions that were forced, and by what

### Deterministic navigation, not model search
Four consecutive live self-patch runs failed identically: asked to add an exit
word to the CLI, a 7B model edited the help string four times. That is not a
reasoning failure — the help line `/quit /exit /bye   leave` genuinely matches
the goal more densely than the code implementing it, so lexical retrieval ranked
it first.

`development/code_index.py` classifies every line by **syntactic role**, ordered
so that "rank real code above documentation" is a sort rather than a pile of
heuristics:

```
DOCUMENTATION < CONSTANT_TEXT < MODULE_CODE < FUNCTION_CODE < FUNCTION_DATA < CONTROL_FLOW
```

Both candidates are string literals, so "is it a string?" ranks the target no
higher. What separates them is what the string is *doing* — a module-level
constant is help text; a set literal inside a branch test is control flow.

*Software finds the code; the model reasons about it.*

### One acceptance bar, executed twice
The capability loop was graded on 2 criteria while 4 decided pass/fail. A
control loop optimises what it is graded on, so it optimised the two it could
see and something it had never been shown decided the outcome. Both live
failures ended `contract=ok, implemented=FAILED`.

`capability_checks()` is now the single definition. The loop takes each check as
a criterion it can watch fail; `_verify` re-runs the same commands from a clean
process afterwards. Identical bar, independent execution.

### Channels, not vendors, for cost
See `JARVIS_COST_POLICY.md`. The short version: what matters is not who receives
the request but how it is paid for, and `fallbacks_for()` structurally cannot
name a metered channel.

### SSE, not WebSockets
Nothing web-related is installed and the brief forbids touching global packages.
What the UI needs is a server-to-client event stream with input arriving as
ordinary requests — exactly SSE's shape. A WebSocket dependency would have
bought bidirectionality nobody uses. Isolated behind `service/http.py`, so
swapping later touches one file.

### Speech in a virtualenv behind a pipe
faster-whisper drags in ctranslate2; Piper drags in onnxruntime. Neither belongs
in the Python that runs the project engine. The venv solves dependencies and
creates a process boundary; `speech/worker.py` sits on the far side speaking
JSON over stdio — the same shape a network transport takes when speech moves
onto a device.

### The client captures audio, the core thinks
The browser records and uploads; the core transcribes, thinks and synthesises.
That split is the future topology, not a convenience: the HDMI box and the phone
will use exactly this protocol, so the browser is the first device client rather
than a special case.

### Escalation from counts, never from self-assessment
A model's estimate of its own difficulty comes from the same weights that are
about to fail. `experts/escalation.py` has no "difficulty" field. It counts
verified failures, *distinct* diagnoses (three different explanations means the
loop is still learning; three identical ones means it is stuck), abandoned
tasks, scope, and measured historical pass rate per task class.

---

## Module map

| Package | Responsibility |
|---|---|
| `service/` | Core, event bus, state machine, HTTP+SSE, voice endpoints |
| `brain/` | Model tiers, catalog, Ollama provider (native `/api/chat`), routing, resource tuning |
| `projects/` | The GOAL→INVESTIGATE→…→ACCEPT control loop, persistence |
| `development/` | Repository engineer, edit engine, **code index** |
| `capabilities/` | Acquire / verify / register / execute installed skills |
| `knowledge/` | Graph (SQLite, typed nodes and edges), ingestion, memory |
| `experts/` | Job/result contracts, gateway, Claude Code adapter, escalation |
| `speech/` | Contracts, worker, engine bridge, phrase chunker, streaming pipeline |
| `tools/` | Workspace tools, web tools, **desktop tools** |
| `persona/` | Personas, invariant rules, language detection |
| `runtime/` | Deadlines, heartbeats, **cost policy**, checkpoints |
| `ui/` | One HTML page, the eye, the starfield — no build step |

---

## Invariants

1. **Nothing is verified by a model's own report.** Acceptance is a command that
   exits zero. This applies identically to the local model, the expert, and the
   seeded skeleton — each of which has been caught trying otherwise.
2. **Every wait is bounded.** Deadlines propagate into calls; heartbeats
   separate liveness from progress; `python -m jarvis.doctor` reports
   alive/stalled/dead/finished.
3. **A timeout is evidence.** It becomes a failed step that DIAGNOSE acts on,
   not a hang.
4. **The identity is Jarvis.** The backend is recorded on every turn and shown
   only in diagnostics.
5. **Deleting every cloud key changes nothing** about local operation.

---

## Where this goes next

The service boundary is already the device boundary. `service/core.py` holds no
assumption that its client is local, and `speech/` already speaks a
serialisable protocol across a process boundary. Moving the core to a home
server and the microphone to a small box is a transport change, not a redesign
— see `JARVIS_DEVICE_ROADMAP.md`.
