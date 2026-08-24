# Jarvis User Guide

---

## Start it

```bash
cd D:\Jarvis_recovery_20260823\repo\Jarvis
python -m jarvis.serve
```

It prints a URL containing a one-time token and opens your browser:

```
  Jarvis is running.

    http://127.0.0.1:8420/?token=…
```

Models load in the background while the page renders. The badge reads
`STARTING` for the first ~15 seconds, then `LOCAL` or `EXPERT AVAILABLE`.

**Useful flags**

| Flag | Effect |
|---|---|
| `--no-browser` | Don't open a browser |
| `--port 0` | Pick any free port |
| `--no-warm` | Skip preloading (first question will be slow) |
| `--no-speech` | Skip loading whisper and the voice |
| `--host 0.0.0.0` | Expose beyond this machine — **read the warning it prints** |

Bound to loopback by default. The token is required on every request.

---

## Talking to it

**Type** in the box at the bottom. Enter sends, Shift+Enter makes a new line,
**Esc interrupts** whatever Jarvis is doing.

**Speak** by clicking the microphone. Click again to stop recording; Jarvis
transcribes, answers, and speaks the reply. Speaking is what enters voice mode —
typing alone never makes Jarvis talk back.

It answers in the language you use. German, English, French, Spanish and Italian
are detected automatically; the language only switches on a confident detection,
so saying "ok" in a German conversation will not flip it.

---

## The eye

The eye is Jarvis' state, not decoration.

| Look | Meaning |
|---|---|
| Slow, dim, blue | idle |
| Wide, bright, pulsing with your voice | listening |
| Fast spin, tight pupil | thinking |
| Bright, breathing with speech | speaking |
| Green, steady rotation | working on a project |
| Violet | researching |
| Amber, almost still | waiting for you |
| Red, jittering | error |
| Nearly dark | offline |

---

## Projects

Describe something to build and Jarvis opens a durable project rather than
replying with a description of what it would build. **Projects** in the top bar
lists them with their goal, state, tasks and recent activity; opening one shows
acceptance criteria (✓ when satisfied) and the last 25 steps.

"Continue this project" resumes it. Projects survive restarts — this works days
later.

---

## Knowledge

**Knowledge** opens the starfield. Nodes are stars sized by how connected they
are; links are lines.

- **drag** to pan, **scroll** to zoom (toward the cursor)
- **click** a star to inspect it
- **search** filters and dims the rest
- from a selected node: **Ask Jarvis**, **Read aloud**, **Expand** its neighbourhood

**Feeding it files.** Point Jarvis at a file or a folder:

```
POST /api/knowledge/ingest   {"path": "C:/Users/you/Notes"}
```

Markdown, text, source code and PDFs are read. Documents are split at their own
headings, so a long note becomes several retrievable pieces rather than one
lump. `[[wikilinks]]` and relative markdown links become real edges. Re-scanning
a folder **updates** it — nothing is duplicated, and a deleted section
disappears from the graph.

---

## When local isn't enough

Jarvis tries locally first, always. It escalates to the subscription expert only
on evidence: several verified failures, the same diagnosis repeating, tasks
exhausted, or a measured poor track record on that class of work. You can ask
for it explicitly.

**It will never spend money you did not agree to.** See
`JARVIS_COST_POLICY.md`. If the subscription runs out, the badge reads
`EXPERT QUOTA EXHAUSTED` and work continues locally or waits for the reset —
there is no path from "out of quota" to "billed per token".

---

## Diagnostics

**Diagnostics** shows the truth: which model answered, model health, expert
quota, the cost policy, event sequence, and current state. Ordinary
conversation never volunteers which backend is running — you are talking to
Jarvis — but ask and it will tell you.

**Activity** streams tool calls and project progress into the conversation.

Is a long run stuck?

```bash
python -m jarvis.doctor --run <state-dir>
```

Reports `ALIVE`, `STALLED`, `DEAD` or `FINISHED` with the last beat and the last
*progress* — which are different things. A model thinking for four minutes is
alive and not progressing; that is normal and distinguishable from a wedge.

---

## First-time setup on a new machine

Ollama with the two models:

```bash
ollama pull qwen3:4b-instruct
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

Speech (optional; ~1 GB, gitignored deliberately):

```bash
python -m venv .venv-speech
.venv-speech\Scripts\python -m pip install faster-whisper piper-tts sounddevice pypdf
# a Piper voice into Jarvis/data/voices/ (.onnx and .onnx.json together)
```

Jarvis works without any of the speech stack — it simply will not hear or speak.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Badge stuck on `STARTING` | The first honest health check is a real generation (~80 s cold). Wait, then check Diagnostics. |
| Badge says `OFFLINE` | Ollama is not running, or the model is not pulled. Diagnostics names which. |
| First question is slow | Started with `--no-warm`, or asked within the warm-up window. |
| Microphone button does nothing | The browser needs permission, and `http://127.0.0.1` (not a LAN IP) to grant it. |
| Jarvis mishears a name | Add it to `SpeechConfig.vocabulary` — this is what fixed "Jarvis" being heard as "Jahres". |
| A project seems stuck | `python -m jarvis.doctor`. Liveness and progress are reported separately. |
