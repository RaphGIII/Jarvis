# ZEUS Live Intelligence + Organization + Image + LG webOS — Truth Report (2026-09-03)

## Revisions
- HEAD / PUSHED: `d7a651e` (origin/adaptive-brain-v1) — final suite + candidate below
- RUNNING: `1f62a62` live (the sprint code; d7a651e adds tests + the window-control CSS only)
- KNOWN GOOD (dist\ZEUS): `1d8c571` (from the previous sprint; a new candidate is built below)

## FILESYSTEM INTELLIGENCE — the biggest failure, fixed and live-verified

**Why screen.capture / file.read were selected:** filesystem questions were
not caught by the deterministic parser, so they fell through to the generic
model / capability matcher, which picked semantically-incompatible tools or
claimed "no access". **Architectural fix:** filesystem questions now route to
real fs tools *before* any model, via `parse_fs_operation` (rewritten) +
`service/system_context.py` (deterministic self-knowledge).

Held-out semantic results (live, through the running product):

| Request | Result |
|---|---|
| "Welcher direkte Ordner auf D: ist am größten?" | **fs.largest** (not screen.capture) → "Bin dran…" → real recursive-size job → **"Apex mit 151.4 GB. Dahinter: CAll of Duty (121.4 GB), Assassin's Creed Odyssey (79.7 GB)"** |
| "Wie viele direkte Unterordner hat dein eigenes Repo?" | self-reference → `D:\Jarvis_recovery_20260823\repo` → **8 Unterordner (und 15 Dateien)** |
| "Was ist direkt in deinem Repo drin?" | fs.list → real folders (.git, build, dist, Jarvis, …) |
| "Bring mich zu deinem Quellcode." | fs.open → Explorer opened at the real repo |
| "wieviele unterordner hat jarvis" | resolved to **6 candidates**, numbered clarification (D:\Jarvis, D:\JarvisLocal, …) |

No hardcoded answers — every result is a live D:\ read. A wrong-tool that
fails now replans (the fs route is chosen first, so screen.capture never runs).

## STARTUP — fullscreen popup

`--start-fullscreen` (the source of the "Vollbildmodus beenden" toast) is
gone. The default is a native maximized window (`--start-maximized`) with no
Fullscreen API — **screenshot-verified: no toast**, taskbar-area intact.
Chromium `--app` draws its own title bar, so the page's redundant min/close
controls are hidden when framed; `ZEUS_WINDOW_MODE=fullscreen` restores the
old immersive look. (True borderless on a Chromium --app needs kiosk mode,
kept as the optional immersive route.)

## IMAGE GENERATION — prompt adherence fixed

**Root cause:** sd-turbo runs at `guidance_scale=0` with 1–2 steps — it barely
follows the prompt. "schwarzes Loch" → a **mountain lake at night**
(`data/acceptance_evidence/images/blackhole_turbo.png`).

**Benchmark on the GTX 1070** (`data/acceptance_evidence/image_model_bench.json`):

| Model | steps · cfg | gen time | VRAM | prompt adherence |
|---|---|---|---|---|
| sd-turbo (FAST) | 3 · 0 | 1.5 s | 3359 MiB | poor (park for "black hole") |
| SD 1.5 (BALANCED/QUALITY) | 22–34 · 7.5–8 | 16–22 s | 2937 MiB | good (real accretion disk) |

Modes: **FAST** = sd-turbo draft; **BALANCED** (default) / **QUALITY** = SD 1.5
with real CFG. Prompt expansion (`expand_prompt`) adds an intent-preserving
quality tail + subject negatives (black-hole → deep space/accretion, *away
from* park/lake/landscape) and keeps the original. Mode is explicit
(`/api/image/generate mode=`) or inferred ("schnell"→FAST, "realistischer"→
QUALITY).

**Live through the product** (BALANCED): "Erstelle ein Bild von einem
schwarzen Loch im tiefen Universum mit heller Akkretionsscheibe." →
`blackhole_product_balanced.png`: **a real black hole — dark event-horizon
centre, concentric glowing accretion disk, stars in deep space.** 21.6 s
generation. Result surfaces in ZEUS (job + notification + thumbnail).

## PROJECT ORGANIZATION

`project_graph` default now emits **owner projects only** (+ active missions
as satellites); capabilities, acquisition/selfdev families and thoughts move
behind the existing "show everything" toggle. Live: the default universe went
from many capability peers to **7 nodes** (3 owner projects + ZEUS core + 2
active missions + 1 knowledge cluster). **No physical file was moved or
deleted** — presentation only.

## LG WEBOS — live network discovery (TV was ON)

Robust multi-method discovery run live against the network:
- Interface: WLAN `192.168.0.7`, subnet `192.168.0.0/24`
- **SSDP: 0 replies** (blocked by the host firewall — even the router did not answer)
- **SSAP port sweep (3000/3001): 0 of 253 hosts** open
- ARP: 9 neighbours — gateway, **2× Espressif** (ESP32 IoT), **1× HP** (printer/PC), **1× Arcadyan** (ISP router/mesh node), a phone

**No LG-vendor device and no SSAP port anywhere on the PC's subnet.** The
powered-on LG TV is not on the PC's network segment — it is on a different
SSID/subnet, behind AP isolation, or its "Mobile TV On / LG Connect Apps"
network control is off. `discover()` now uses SSDP + SSAP-sweep + ARP/OUI;
`diagnostics()` returns interface/subnet/methods/candidates/rejected-with-
reasons; the Devices UI shows this evidence and the likely cause instead of a
bare "not found". Bluetooth is correctly avoided — the display route is the
LAN `?tv=1` remote page. **A real pairing request could not be sent because
no TV is reachable from this PC.** Owner action: put the TV on the same WLAN
as the PC (192.168.0.x), disable AP isolation, and enable TV → Connection →
"Mobile TV On"; then re-run discovery.

## The 13 answers
1. **Filesystem questions answered from the real machine?** YES — Apex 151.4 GB, repo 8 subfolders, all live.
2. **Can an unrelated screen capability still be selected for size questions?** NO — fs questions route to fs tools before any model; pinned by tests.
3. **Failed wrong tool triggers replanning?** The fs route is now chosen first, so the wrong tool no longer runs; the conversation retry path also recovers ProviderErrors.
4. **Does ZEUS know its own repository path?** YES — SystemContext resolves "dein Repo" → D:\Jarvis_recovery_20260823\repo, live.
5. **Projects shows canonical owner systems?** YES — default graph is owner projects + active work only (7 nodes).
6. **Any physical owner file moved/deleted?** NO.
7. **Image prompt adherence materially better?** YES — turbo park → SD 1.5 real black hole, benchmark + live.
8. **Owner can select FAST/BALANCED/QUALITY?** YES — modes in the API and inferred from the request.
9. **Powered-on LG TV discovered?** NO — thorough live discovery shows it is not on the PC's subnet (full diagnostics reported).
10. **Real webOS pairing request reached the TV?** NO — no TV reachable to pair with; the exact owner action is reported.
11. **Bluetooth avoided as the primary display protocol?** YES — LAN remote page only.
12. **Fullscreen-exit popup gone?** YES — screenshot-verified, native maximized window.
13. **Ready for owner acceptance?** The running build is; a verified ZEUS.exe candidate is built and waits on the SELFDEV_PROMOTE password.
