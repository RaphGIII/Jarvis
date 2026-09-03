# ZEUS Overnight Autonomy Sprint — Truth Report (2026-09-03)

## Revisions

| What | Value |
|---|---|
| HEAD | `1d8c571` |
| PUSHED | `1d8c571` (origin/adaptive-brain-v1) ✓ |
| KNOWN GOOD (dist/ZEUS) | `c9b5321` exe — **still carries the visible-Ollama-console bug** |
| CANDIDATE | `1d8c571bd4e5-20260903T024656Z-76ecaa`, built + **VERIFIED**, waiting for the owner's SELFDEV_PROMOTE password (Release view) |
| RUNNING | ZEUS.exe (old launcher) executing repo code `1d8c571`, FULL_READY |
| Full suite | **2115 passed, 0 failed** (15:41) |

## STARTUP — measured, three real cold starts (ZEUS + Ollama stopped)

**Console windows.** The live offender was `ollama serve` spawned with
`DETACHED_PROCESS`: a detached serve has NO console, so every
`llama-server.exe` runner child allocated a VISIBLE one. Measured with an
EnumWindows watcher across full boots:
- old frozen exe: **4–5 new console windows** per cold start (ollama.EXE + llama-server.exe) — the owner's report confirmed, reproduced 3×.
- supervisor **from source with the fix** (hidden console the runners inherit): **0 new console windows** through boot + model load. The fix ships with the verified candidate; until the owner promotes it, the old exe keeps the old behavior.

**Stages** (cold, Ollama stopped): shell/boot page at T0 · CORE 22–39 s ·
INTERACTIVE (core+AI) 39–116 s · voice/full 100–201 s. The spread is the
Ollama cold 4B model load — that is the bottleneck, not ZEUS code; warm
restarts reach AI_READY in ~17 s.

**No reconnecting/offline exposure.** New `INTERACTIVE_READY` level
(core + conversation model). The main UI is covered by a boot veil (in the
markup from the first byte: orb, rings, staged system lights, honest phases
— "Lokale Intelligenz wird geladen", "… wird wiederhergestellt") and
dissolves only at INTERACTIVE_READY; voice finishes warming behind it with
its own light. CDP-verified: phase reached "ZEUS ONLINE", veil lifted.

**Ollama recovery** was already supervised (state machine STOPPED→…→READY,
watchdog + bounded respawn); the core now also retries a failed conversation
once with a reduced prompt and fails as a human sentence.

## OBSERVABILITY — the WorkItem system

`service/jobs.py`: every long action is a Job (QUEUED→…→COMPLETED/FAILED/
CANCELLED, phases, progress, timings, JSONL history that survives restarts),
emitted live as `job` events. Left dock (⚡ tab): active jobs with phase +
progress, recently finished with image thumbnails, recent conversations.
Eye: background-work orbit dot, radiation emissions, stronger default glow.
Acknowledgments: image/web jobs answer "Bin dran …" immediately.
Cancel semantics (§55): "Stopp" = speech; "Stopp die Bilderzeugung" cancels
image jobs (mid-diffusion honestly reported as running out; cancelled
results suppressed); "Stopp alles" asks first.

## IMAGE — perceived-delay root cause fixed

Root cause: the one-shot worker paid **200–370 s model load per image**.
v2 keeps a persistent worker (model in CPU RAM, GPU only during the
generation window). Measured live:
- first image: 383 s total (one-time CPU load 348 s, visible as "Modell lädt" phase)
- second image: **4.6 s request→completed** (to_gpu 0.9 s, generate ~1.2 s, save, GPU freed)
- eviction decided at the to_gpu moment (489 MiB free < 3800 → FAST_LOCAL unloaded), **restored in 10.5 s immediately after**, and a model question asked during the GPU window was queued with "Das Bildmodell nutzt gerade die Grafikkarte. Deine Frage ist vorgemerkt…" then auto-answered. "Wie spät ist es?" answered instantly throughout. No silent timeout.
- result surfaces in ZEUS: "Bild fertig." notification + dock thumbnail (`/api/image/file`, path-fenced) + answer with path, ~1 s after the file lands.

## INTELLIGENCE

- "Zähle wieviele Subordner der Ordner Jarvis auf D: hat" → real filesystem
  op: bounded 2-level search found 6 candidates → ONE question ("Meinst du
  D:\Jarvis oder …?") → real count („D:\Jarvis" hat 5 direkte Unterordner).
  Live-verified. Never again "I cannot access D:".
- The conversation prompt now carries the tool roster (filesystem, apps,
  web, Spotify, Kalender, Projekte, Wissen, PDF, Bild, Screenshots, SelfDev)
  plus a no-confabulation rule for unknown terms (the "Wischbär" class).
- Web follow-up: search/research store the result context; "Nimm einen
  wichtigen Artikel davon und fass ihn zusammen" fetched the REAL page
  (2336 readable chars), summarized in 6 s, cited the source — live.
  Summarization retries on a shrinking-context ladder (full→½→¼).
- Raw ProviderError lines no longer reach the chat (friendly German; detail
  in Activity).

## STT (honest)

Owner corpus is still EMPTY — recording is owner-only; the wizard now has
**129 phrases** across every required category plus custom phrases.
Engine benchmark on THIS machine (synthetic Piper audio — latency/RAM real,
accuracy indicative, NOT owner WER): small int8 CPU median 3.3 s (stays
production), base 1.2 s but far weaker, medium best accuracy at 9.7 s median
— too slow for live voice on this CPU. Owner WER before/after requires the
recorded corpus (`Voice Studio → Spracherkennung trainieren`, then
Benchmark). Evidence: `Jarvis/data/acceptance_evidence/stt_engine_bench.json`.

## UI / P1

Recent-conversations archive with async FAST_LOCAL titles/summaries (live:
"Pfadunsicherheit", "Keine Termine morgen"), restore into the chat, archived
on /api/new and final shutdown; candidate facts stay in the record for
inspection, never silently promoted. Themes COSMOS/REACTOR/VOID/NEBULA/MONO
+ intensity OFF/LOW/NORMAL/HIGH (instant, persisted; CDP-verified NEBULA
accent #a98fff + eye shift 62). Creation defaults (folders + naming
templates) under Owner, honoured by the image pipeline; explicit request
always wins. Idle preload primes projects/calendar/jobs.

## TV (LG webOS)

Real protocol: SSDP discovery + SSAP websocket pairing (TV-side
confirmation, stored client-key), volume/mute/launch/open-url/toast,
Wake-on-LAN reported as ATTEMPT. "Zeig dich auf dem Fernseher" opens the
existing ?tv=1 kiosk UI over the LAN — refused with instructions while the
server is loopback-bound (opt-in `ZEUS_LAN=1`). Discovery at 03:20 found no
TV (off/asleep) — Devices UX is ready; physical pairing is owner-only.

## LANGUAGE COACH

Session mode in the chat ("Zeus, lass uns 10 Minuten Französisch üben"):
opener + per-turn structured evaluation (one main correction, score, next
line), learner model with spaced-repetition vocabulary and level
progression, summary saved to Wissen/Sprachen. Live 3-turn session verified
(state, persistence, summary); per-turn evaluation costs ~20–30 s under
load — usable, not snappy.

## SOAK

51 minutes logged so far (1-minute samples): ready=true throughout, exactly
one ollama process, stable RAM (~0.9–1.1 GB python total), no duplicate
listeners, no job leaks; a detached monitor keeps logging for ~6 h to
`%LOCALAPPDATA%\Temp\claude\…\scratchpad\soak_log.jsonl`.

## The 18 answers

1. **Zero startup consoles?** In the NEW code: yes — measured 0 on a full from-source cold start. On the owner's current exe: no (4–5) until the verified candidate is promoted.
2. **Main UI hidden until usable?** Yes — boot veil until INTERACTIVE_READY; no RECONNECTING/OFFLINE exposure.
3. **Ollama recovery without ZEUS restart?** Supervisor watchdog respawns it (pre-existing, verified state machine); core-side one-retry recovery added. A cold-load window still takes ~1 min.
4. **Owner sees active work?** Yes — job events, dock, eye work-orbit; verified live during a real generation.
5. **Immediate acknowledgment?** Yes for image/web jobs ("Bin dran …", measured <2 s).
6. **Image result surfaces immediately?** Yes — notification + thumbnail + answer ~1 s after file save.
7. **Conversation survives image GPU use?** Yes — deterministic answers run; model questions queue with an honest sentence and auto-run after the 10.5 s restore.
8. **ZEUS knows it has filesystem access?** Yes — live count of D:\Jarvis; prompt-level tool roster.
9. **Clarifies ambiguity?** Yes — one question among real candidates, live.
10. **Recovers instead of abandoning?** Yes — web summary source-fallback + shrinking retries; conversation retry; jobs FAILED with reasons, never silence.
11. **STT measurably better on owner recordings?** NOT CLAIMED — corpus empty; measurement machinery ready (129 phrases, benchmark, reports).
12. **Conversations + summaries persistent?** Yes — archived, summarized, restorable, on disk.
13. **Default paths/naming configurable?** Yes — Owner → Standard-Pfade & Namen, honoured live by imagegen.
14. **Themes/animations configurable?** Yes — 5 themes + intensity, instant, persisted.
15. **LG webOS real, not Bluetooth fiction?** Yes — SSDP/SSAP/WoL implemented; no TV was awake to pair (owner-only).
16. **Coach has persistent learner state?** Yes — learner_französisch.json with session history written during the live test.
17. **Repeated desktop behavior → owner-approved shortcut?** Partially — the opt-in observer with patterns/suggestions exists from the prior sprint; automatic workflow-capability creation from a suggestion is NOT built yet.
18. **Ready for owner acceptance?** The running product (new code) yes; the exe swap that removes the last visible consoles and ships the new boot page waits on the owner's password.
