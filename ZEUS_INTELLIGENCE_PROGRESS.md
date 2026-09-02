# ZEUS Intelligence Core Sprint — 2026-09-02/03

Sprint: semantic control plane, zero-friction startup, view keep-alive,
calendar, owner STT corpus, local image generation.

## Revisions

| What | Value |
|---|---|
| HEAD (all work committed) | `b67d9a1` + this record |
| PUSHED HEAD | **not pushed** — the session's push was blocked by the permission classifier; run `git push origin adaptive-brain-v1` |
| KNOWN GOOD (dist/ZEUS) | `c9b5321` exe (launches the new repo code live) |
| STAGED CANDIDATE | `b67d9a147678-20260902T221647Z-90ae97`, built + verified; **promotion awaits the owner's password** (SELFDEV_PROMOTE gate) — approve in the Release view |
| RUNNING CORE | `b67d9a1`, FULL_READY after a full physical product restart |
| Full suite | 2109 passed, 6 skipped, 1 xpassed, **0 failed** (12:55) |

## Intelligence

**Old flaw (reproduced live):** `registry.find` scored on term overlap with an
English-only stopword list, so the German function word "einer" (stored as a
learned capability keyword) matched "Öffne Wikipedia" to
`learned.ausgabe_dateipfad_zeilen`; and `CapabilityService.resolve`'s
knowledge-graph fallback had no semantic gate at all, which ran
`music.provider.spotify` for "Schalte das Licht im Wohnzimmer aus".

**New control plane:** `service/semantic.py` — one schema-constrained
FAST_LOCAL call (`generate_structured`, closed operation enum: a wrong tool is
*unrepresentable*). Deterministic parsers remain fast paths producing the same
typed `ActionIntent`; the semantic planner runs when they don't resolve, and
the dispatcher itself recovers semantically (app miss → canonical site →
web.open; unknown → ONE question whose answer becomes a persistent owner
alias). `_target_grounded` refuses planner-invented targets. Low-confidence
`capability.missing` names the gap and OFFERS acquisition (yes/no), never
falling back to lexical guessing. German stopwords joined the boilerplate
filter and are excluded at keyword-write time; the graph fallback now requires
a shared content word with the capability's own declared subject.

**Held-out paraphrases, live:** "Bring mich zu Wikipedia." → semantic
web.open 0.99; "Ich brauche Wikipedia." → openable-wish → web.open 0.99;
"Mach Spotify auf." → app.open 0.99. Evidence:
`Jarvis/data/acceptance_evidence/S37_semantic_live.json`.

**Teach-by-explanation:** `service/aliases.py` (folded, possessive-stripped
keys). Live: "Merk dir: Testplaner ist D:\ZEUS_Wissen\Testplaner.txt" →
"Öffne meinen Testplaner." opens the file directly. Aliases feed the STT
hotwords and the planner context.

## Startup

- **0 new console windows** across a full ZEUS.exe shutdown + boot + 150 s
  watch (previously nvidia-smi popped one every ~3 s). Every background spawn
  now passes `CREATE_NO_WINDOW` (gpu probe, speech worker, expert CLIs,
  tasklist, capability runner, media session, pdf extract, taskkill).
- Boot stages measured: core/UI 9.6 s · window title 15.5 s · AI 17.2 s ·
  voice/full 58.4 s.
- `serve.py` binds the port BEFORE the two PowerShell orphan sweeps (they ran
  first and left the window on a dead port for seconds).
- The supervisor's boot page was the ERR_CONNECTION_REFUSED source: it
  `location.reload()`ed into the port-handover gap and stranded Edge on the
  browser error page. Rewritten as the cosmic boot experience (orb + orbital
  rings + progressive system illumination, `lang="de" translate="no"`),
  updating in place and navigating only when the core answers — keeping the
  token query. **Ships with the staged exe.**

## Views

- `ui/core/views.js`: keep-alive parking (suspend/resume, `display:contents`
  wrapper, open-race guard, animated loading shell — no more black voids).
- CDP-verified live: leaving Projects for Chat or Files and returning reuses
  the SAME Galaxy instance with the exact camera (x123/y77/z1.31); rebuilds
  happen only when data changed (digest check / fs-watch stale flag /
  knowledge bus events).

## Calendar (new)

`service/calendar.py` + `/api/calendar/*` + Kalender view (month grid,
agenda, editor, .ics export/import, reminder minute-beat). Live: "Trag morgen
um 14 Uhr einen Testtermin für 30 Minuten ein." → parsed, persisted,
verified, listed after a restart, deleted cleanly.

## Owner STT

`speech/corpus.py` (verified-transcript corpus, WER/CER/entity metrics,
40-phrase script) + `speech/benchmark.py` (candidate faster-whisper models on
the corpus, in the speech venv) + Voice-Studio wizard "Spracherkennung
trainieren". Hotwords now include owner aliases; Wikipedia/YouTube/Kalender
joined the entity list. **The corpus is empty until the owner records — no
synthetic accuracy claims are made.**

## Image generation (new)

`imagegen/generate.py` (SD-Turbo fp16, one-shot process in
`D:\JarvisLocal\venv-image`, torch 2.7.1+cu118 — CUDA works on the GTX 1070)
+ `service/imagegen.py` + `/api/image/*` + intent "Erzeuge mir ein Bild von
…". Three real images generated: 8.5–13.4 s generation, 3359–3575 MiB VRAM
peak; the one-shot model load costs 3–5 min (documented; a persistent worker
is the known next step). When free VRAM < 3800 MiB the chat model is unloaded
first and the answer says so. Output lands in `D:\ZEUS_Wissen\Bilder`.

## The 13 answers

1. **Semantic understanding of unseen requests?** YES — live: three held-out
   paraphrases through the planner (0.99 each), plus grounding guard.
2. **"Öffne Wikipedia" can't reach a file capability?** YES — deterministic
   recovery opens de.wikipedia.org; the stopword match is dead and pinned by
   tests.
3. **Intelligent recovery when a tool fails?** YES — app miss → site → open;
   alias miss → one question → learned.
4. **Missing capability → clarification/acquisition?** YES — live offer with
   confirmation; the Spotify-for-a-light-switch path is closed.
5. **Zero startup CMD flashes?** YES — measured 0 across a full boot.
6. **Dead localhost during launch?** Core-side gap shrunk now; the boot-page
   fix that removes it entirely is in the staged exe (owner promote).
7. **Translation prompts gone?** UI already had notranslate; the boot page
   (the trigger) now has it too — ships with the staged exe; owner confirms.
8. **Spatial views survive tab switches?** YES — CDP-verified same instance +
   camera.
9. **STT measurably better?** NOT CLAIMED — the measurement system exists;
   numbers require the owner's recordings.
10. **Owner can build a verified corpus?** YES — wizard live in Voice Studio.
11. **Local offline image generation?** YES — three real images, metrics
    recorded; model cached locally.
12. **Calendar persists real events?** YES — created by sentence, survived a
    restart, deleted cleanly.
13. **ZEUS.exe ready for owner acceptance?** Candidate `b67d9a1…` built and
    verified; promotion is password-gated — approve it in Release, then the
    relaunch watchdog swaps it in.
