# Jarvis Device Roadmap

The target topology, what already supports it, and what is genuinely left.

---

## Where this is going

```
                      JARVIS BRAIN
                 home server (owned)
                 GPU, models, memory,
                 projects, knowledge
                          │
                 encrypted LAN / WAN
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     Desktop           Phone           Jarvis Box
     browser          browser          mic / speaker / camera
                                              │
                                            HDMI
                                              │
                                             TV
```

The box needs no GPU. It captures audio, plays audio, shows a screen and holds
a connection. Everything expensive stays on the server.

---

## What already supports this

This is not aspiration — the boundary exists and is load-bearing today.

**The core has no UI.** `service/core.py` may not import anything from a UI and
never formats for display. Every client calls the same methods. Adding a device
client adds no core code.

**The browser is already a device client.** It captures an utterance, POSTs the
audio, and plays audio the core returns by reference. That is exactly the
protocol a box would use — chosen for that reason, not for convenience.

**Speech already crosses a process boundary.** `speech/worker.py` speaks JSON
over stdio because whisper and Piper need their own virtualenv. Moving that
boundary from a pipe to a socket is a transport change; the contracts in
`speech/contracts.py` carry no assumption about where the implementation runs.

**Audio is transported by reference.** Synthesised speech goes into a bounded
store and the event carries a URL. A device fetches what it needs when it needs
it, and a slow link delays audio rather than every other event behind it.

**State is one small shared vocabulary.** Eleven states in `service/state.py`,
owned by the core. A box with an LED ring renders the same states the eye does,
with no second source of truth about what Jarvis is doing.

**The event bus assumes many clients, none reliable.** Publishing never blocks;
a subscriber that stops draining loses its oldest events rather than applying
backpressure. A device on a flaky connection cannot stall an autonomous build.

---

## What is left

### 1. Device gateway — identity, pairing, authentication
Today there is one shared token, correct for loopback and insufficient the
moment a second machine is involved. Needed: per-device identity, a pairing
flow, per-device revocation, and heartbeat/presence.

### 2. Transport
SSE plus HTTP POST is right for a browser on localhost. Over a network a device
also wants to *push* a continuous audio stream. Likeliest shape: keep SSE for
events, add a WebSocket for bidirectional audio. `service/http.py` is the only
file that should need to change.

### 3. Wake word and VAD on the device
The box must decide locally when to open the microphone; streaming continuous
audio to a server is wasteful and a privacy problem. openWakeWord and Silero VAD
both run comfortably on a Pi-class device — neither is evaluated yet.

### 4. TV / kiosk mode
The UI is already responsive and canvas-based. A TV needs larger type, no
pointer affordances, and a rotating idle view. Same frontend, a display mode —
not a second UI.

### 5. Transport security
Loopback needs none. A LAN needs TLS and a real device credential; a WAN needs a
tunnel rather than an open port. Deliberately unimplemented — shipping a
half-considered auth scheme is worse than shipping none while the surface is
loopback-only.

---

## Hardware sketch

Nothing here is bought yet.

| Part | Enough | Why |
|---|---|---|
| Compute | Pi 5 (4 GB) or N100 mini-PC | Wake word + VAD + audio + browser. No inference. |
| Microphone | ReSpeaker 2-Mic HAT, or any USB array | Far-field pickup matters more than the codec |
| Speaker | Small powered / amplified HAT | Voice, not music |
| Display | HDMI to the TV | The UI is already a web page |
| Camera | Optional USB | Frames go to the server for vision |

The server is this machine today. A GTX 1070 runs a 7B coder and a 4B
conversational model; anything larger is a GPU upgrade, not a rewrite —
`brain/tiers.py` selects models by role, so a 70B on a bigger card is
configuration.

---

## Order of work

1. Device identity and pairing (the gateway proper)
2. Reference client — a Python process on another machine, no hardware
3. WebSocket audio transport
4. Wake word and VAD, evaluated on real hardware
5. TV mode
6. TLS and remote access
7. Physical build

Steps 1–3 need no hardware at all, and the reference client is what proves the
boundary is real rather than merely intended.
