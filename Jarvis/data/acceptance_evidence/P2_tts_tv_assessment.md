# P2-Bewertung: Stimme (TTS) und LG Smart TV — Stand 2026-09-02

Ehrliche technische Bewertung, gemessen bzw. geprüft auf DIESER Maschine
(GTX 1070 / Pascal, CPU-STT, Ollama-Modelle auf derselben GPU).

## Stimme (TTS)

Ist-Zustand: ZEUS spricht über **Piper** (ONNX, CPU) aus `data/voices/`;
aktiv war `de_DE-thorsten-medium` (22 kHz-Klasse, RTF « 1, sehr niedrige
Latenz). Der Stack ist also bereits lokal und schnell — die Frage ist nur
die Stimmqualität.

Real umsetzbare Optionen, bewertet:

| Option | Qualität | Latenz | Aufwand | Urteil |
|---|---|---|---|---|
| **Piper `de_DE-thorsten-high`** (heruntergeladen, liegt in `data/voices/`) | deutlich voller als medium; männlich, ruhig, souverän | **gemessen: RTF 0,63 CPU** (2,1 s für 3,4 s Audio; Streaming-Chunks ⇒ gefühlt < 1 s) | getan — im Voice Studio auswählbar | **Empfehlung, sofort nutzbar** |
| Piper andere DE-Stimmen (karlsson, pavoque) | Geschmackssache, eher flacher | wie oben | nur Download | Alternative |
| **Coqui XTTS v2** (lokal, GPU) | sehr hoch, klonbar | 1–3 s, **konkurriert mit den LLMs um die 1070**; Pascal-Kompatibilität fraglich (siehe CUDA-STT-Befund) | hoch (venv, VRAM-Management) | auf dieser Hardware **nicht empfohlen** |
| **Kokoro-82M** (ONNX) | exzellent EN, **Deutsch schwach** | gut | mittel | für DE nicht geeignet |
| **edge-tts** (Microsoft Neural, z. B. `de-DE-ConradNeural` – tief/ruhig) | beste Qualität | ~0,3–0,8 s (Cloud) | gering (pip) | **nur als Opt-in**: Cloud-Abhängigkeit; Spending/Offline-Prinzip beachten |

Entscheidung in diesem Sprint: `thorsten-high` real installiert und
verifiziert (Latenzmessung oben). Nicht als Default umgestellt — die
Stimmwahl ist Owner-Geschmack und im Voice Studio ein Klick.

## LG Smart TV

Geprüft: SSDP-Discovery im LAN (M-SEARCH, 4 s) — **kein WebOS-/DLNA-Gerät
hat geantwortet**. Ein LG-TV war zum Prüfzeitpunkt also nicht erreichbar
(aus, anderes Netz, oder Multicast blockiert). Deshalb Analyse statt PoC:

1. **Was real geht (wenn der TV im LAN ist):**
   - *Einschalten:* Wake-on-LAN (Magic Packet an die TV-MAC; im TV muss
     „LG Connect Apps“/„Einschalten über WLAN“ aktiv sein). Zuverlässig, trivial zu bauen.
   - *Steuern:* WebOS „SSAP“-Websocket (Port 3000/3001, einmaliges
     Pairing-Popup am TV). Lautstärke, App-Start, Eingang, Toast-Nachrichten.
   - *ZEUS-Ansicht anzeigen:* Der TV-Browser kann die bestehende
     `?tv=1`-Ansicht laden. **Voraussetzung und einzige echte Hürde:** der
     Core bindet heute bewusst nur an `127.0.0.1`. Nötig wäre ein
     Owner-Opt-in „LAN-Freigabe“ (Bind an LAN-IP + Token bleibt Pflicht) —
     eine Sicherheitsentscheidung, die nicht nebenbei fallen sollte.
2. **Was nicht bzw. nur begrenzt geht:**
   - *Bluetooth:* am TV nur Audio/HID — **keine UI-Projektion**. Klare Absage.
   - *Miracast:* Windows kann casten, aber verlässliche Automation
     (headless starten/stoppen) ist fragil; nicht empfohlen als Produktpfad.
   - *HDMI:* physisch immer möglich, ist aber Verkabelung, keine Integration.
3. **Bester praktischer Pfad:** WoL + SSAP-Pairing + TV-Browser auf die
   `?tv=1`-Seite (nach expliziter LAN-Freigabe). Aufwand: klein, aber erst
   sinnvoll, wenn der TV im LAN sichtbar ist — Discovery erneut laufen
   lassen, wenn der TV an ist.
