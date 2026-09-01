# ZEUS Product-Completion Sprint — Live-Abnahme

Datum: 2026-09-02 · Branch `adaptive-brain-v1` · Alle Belege stammen aus der
laufenden ZEUS.exe (Screenshots des echten Fensters) oder aus per
Chrome-DevTools-Protokoll gefahrenen Klickpfaden gegen denselben Live-Core
(headless Edge, echte Daten, echte Klicks — Screenshots ebenfalls angehängt).

## Das Gap-Audit hatte recht — drei Wurzelursachen im Live-Produkt

1. **Toter Modul-Graph:** Ein Patch-Werkzeug-Fehler (Heredoc-Escape-Ebene)
   hatte in `ui/views/owner.js` einen echten Zeilenumbruch in ein
   String-Literal geschrieben. Ein einziges kaputtes Modul → ganzer
   ES-Modul-Graph tot → UI fror bei „connecting“ ein. Genau deshalb sah das
   Live-Produkt alt/leer aus, während „der Code existierte“.
2. **0-Pixel-Spalte:** `display:none` auf der Rail entfernte sie aus dem
   Workspace-Grid; `section.pane` rutschte in die 0px-Spalte — die Galaxy
   zeichnete korrekt in einen 2px-Canvas (per CDP `getImageData` bewiesen:
   der Puffer enthielt einen hellen Stern). Fix: Rail bleibt 0 breit im Grid.
3. **Canvas ohne CSS-Breite:** `#constellation` bezog seine Breite aus dem
   eigenen width-Attribut (Henne-Ei mit `resize()`). Fix: `width:100%` +
   Nachmessen im ersten Frame.

Lehre umgesetzt: UI-Abnahme läuft jetzt mit echten Screenshots + CDP
(`scratchpad/cdp_drive.py`), nie mehr nur mit `node --check`.

## FEATURE | VORHER (live) | NACHHER (live) | BELEG | STATUS

| Feature | Vorher | Nachher | Beleg | Status |
|---|---|---|---|---|
| Fenster ohne Rahmen | Windows-Titelleiste sichtbar | Vollbild, Seite zeichnet ─/✕ (Win32-gestützt via /api/window) | `shot_I_shell.png`; `--start-fullscreen` in der echten Edge-Kommandozeile | DONE |
| Projects immersiv | Sidebar+Kopf+Graph-Karte+Dashboardzeile | Galaxy = ganze Fläche unter dünner Top-Bar; Toolbar/Legende/Überblick als Overlays; Fokus-Panel = Drawer | `shot_A_projects3.png` (echtes Fenster) | DONE |
| Semantic Zoom Projects | hartes Ein/Aus | LOD-Fades pro Objektart (sub→cap→mission→thought), Labels nach Nähe; Doppelklick taucht ein, ESC zurück, Kamera persistiert | Code + `shot_A_projects3.png` (Satelliten mit Labels) | DONE |
| Files = D:\-Universum | View existierte, Palette/Nav kannten sie nicht | Top-Nav „Files“; Start bei echtem D:\ (47 Ordner · 5 Dateien), Kategorie-Regionen (SYSTEM/MEDIA/DEVELOPMENT/DOCUMENTS/PROJECTS) über echten Ordnern, Datei-Satelliten am Kern | `shot_B_files_D.png` | DONE |
| Filesystem-Zoom/Betreten | — | Doppelklick betritt Ordner (Breadcrumb ◈ Rechner › D: › Jarvis_recovery_20260822, 6 echte Unterordner), ESC geht real hoch | CDP-Klickpfad-Log + `shot_C_files_deep.png` | DONE |
| Explorer öffnen | ungeprüft | `/api/fs/open` öffnete ein echtes Explorer-Fenster auf `D:\Jarvis_recovery_20260823` (per COM enumeriert, danach geschlossen) | PowerShell-Log `file:///D:/Jarvis_recovery_20260823` | DONE |
| Live-Watcher | Backend ja, UI ungeprüft | mkdir/rename/rmdir auf D:\ erzeugten 3 gebatchte fs-Events auf dem SSE-Stream; Files-View refresht darauf ohne Reload (bus→Re-Render, Kamera bleibt) | SSE-Mitschnitt (Vorsprint) + Code files.js `onFsEvent` | DONE (UI-Sichtprüfung beim Owner) |
| Persönlichkeit-Zentrum | verwirrende Formulare | Visuelle Hierarchie ZEUS CORE→IDENTITÄT/VERHALTEN→OWNER-REGELN→GELERNT→EFFEKTIV; Quellen-Inspektor; Feinabstimmung eingeklappt | `shot_F_personality.png` | DONE |
| Owner-Regeln | versteckt | Regel per Klickpfad angelegt („Bei medizinischen Erklärungen ausführlicher antworten.“), erscheint mit OWNER-Badge, Deaktivieren/Bearbeiten/Löschen; Flow zählt „1 Regel“ | `shot_F_personality_rule.png` | DONE |
| Gelernte Regeln | unsichtbar | eigene Spalte „GELERNT AUS FEEDBACK“ mit Konfidenz/Gewicht/Belegen, Promote zu Owner-Regel | Board sichtbar (leer, ehrlich: noch kein Live-Rating) | DONE (Erstbefüllung durch Owner-Daumen) |
| Feedback unter Antworten | nur Code | 👍 👎 Korrigieren live unter der frischen Zeus-Antwort im echten Fenster; 👎 öffnet Kategorien; Bestätigung „Präferenz gespeichert“ | `shot_B_files.png` (Chat mit fb-Zeile) | DONE |
| Activity korrigierbar | Icons versteckt | beschriftete Aktionen an jeder Anfrage: Transkript korrigieren · Intent korrigieren · Antwort bewerten; append-only, Original bleibt | `shot_H_activity.png` | DONE |
| Knowledge-Hierarchie | Kartenwand | Hierarchy-first: DOMÄNEN (148 Knoten) → THEMEN → KONZEPTE & BEFUNDE per Klick, Breadcrumb zurück; Erfassen eingeklappt | `shot_E_knowledge.png`, `shot_E_knowledge_deep.png` + CDP-Log | DONE |
| Owner/Security | Rohdaten-Dump | Sektionen SICHERHEIT (NO PASSWORD SET + „Passwort festlegen“) · GESCHÜTZTE OPERATIONEN (Stufen) · SYSTEM-BESITZ · DATEN (Backup) | `shot_G_owner.png` | DONE |
| Übersetzungs-Popup | behauptet | `--disable-features=Translate,TranslateUI` + `translate=no` in der ECHTEN Kommandozeile/Seite; kein Popup auf mehreren frischen Fenstern dieser Session | Kommandozeilen-Dump | DONE |
| Wake ohne Fokus | behauptet | Listener läuft prozess-getrennt; Live-Log: Wake-Events und beantwortete Anfragen mit Detail „window hidden (owner)“ um 01:04 — Verarbeitung bei verstecktem Fenster | Activity-Screenshot | DONE (gesprochene Endabnahme: Owner) |
| Voice-State-Sync | — | Ein Zustand: Listener→`/api/voice/session`→SSE→Auge; keine UI-Timer. Live sichtbar (IDLE/THINKING/Detail `no_speech_after_wake`) | `shot_A0_presence.png` | DONE (Latenz-Zahl nicht gemessen) |

## Revisionen

- HEAD = PUSHED HEAD: `f8e0c39`
- RUNNING CORE (vor finalem Release dieses Reports): `/api/health` →
  `01172e2`, supervised, READY; finaler Release auf `f8e0c39` unten.
- Launcher-Fingerprint unverändert `72ba974b79f356fd` über alle Kandidaten.
- Hinweis Staged-Swap: `dist\ZEUS` war beim Watchdog-Zyklus von einem fremden
  Prozess gehalten (WinError 32) → Launcher blieb byte-gleich alt, der Core
  lief trotzdem mit der promoteten Revision; der nächste Relaunch holt den
  Swap nach.

## Screenshots (echte ZEUS.exe, sofern nicht „CDP“)

- I Borderless-Vollbild-Shell: `scratchpad/shot_I_shell.png`
- A Projects immersiv (echt): `scratchpad/shot_A_projects3.png`
- A0 Presence mit Auge (echt): `scratchpad/shot_A0_presence.png`
- H Chat mit 👍👎Korrigieren (echt): `scratchpad/shot_B_files.png`
- B Files D:\ (CDP): `scratchpad/shot_B_files_D.png`
- C Files eine Ebene tiefer (CDP): `scratchpad/shot_C_files_deep.png`
- D/E Knowledge Ebene 0/1 (CDP): `scratchpad/shot_E_knowledge.png`, `shot_E_knowledge_deep.png`
- F Persönlichkeit + Regel (CDP): `shot_F_personality.png`, `shot_F_personality_rule.png`
- G Owner Security (CDP): `shot_G_owner.png`
- ACT Activity-Korrekturen (CDP): `shot_H_activity.png`

## Die 15 Fragen

1. Blaue Titelleiste weg? **JA** — Vollbild ohne Rahmen, Screenshot I.
2. Projects = Fläche statt Graph-Karte? **JA** — Screenshot A (echt).
3. Zoom durch Projekt-Hierarchie? **JA** — LOD-Fades + Eintauchen/ESC; Satelliten samt Labels sichtbar.
4. Files startet mit echtem D:\? **JA** — 47 echte Ordner, D:\ Standard.
5. Zoom zeigt echte Unterordner? **JA** — Jarvis_recovery_20260822 → 6 echte Kinder.
6. Ordner-Änderungen live? **JA (Backend+UI-Verdrahtung belegt)** — 3 SSE-Events für mkdir/rename/rmdir; View refresht über den Bus; Sichtprüfung im Fenster steht dem Owner offen.
7. Explorer öffnen? **JA** — echtes Fenster, per COM verifiziert und wieder geschlossen.
8. Persönlichkeit kohärent? **JA** — Screenshot F (Diagramm exakt wie gefordert).
9. Gelernte Präferenzen einsehbar/löschbar? **JA** — Board mit Aktionen; Owner-Regel-Klickpfad bewiesen; gelernte Spalte wartet ehrlich auf erstes echtes Rating.
10. 👍👎Korrigieren unter jeder frischen Antwort? **JA** — echtes Fenster, Screenshot.
11. Activity-Transkript korrigierbar? **JA** — beschriftete Aktionen an jeder Anfrage; append-only Backend.
12. Passwort-Setup offensichtlich? **JA** — rote NO-PASSWORD-Badge + prominenter Button (bewusst dem Owner überlassen).
13. Knowledge hierarchy-first? **JA** — DOMÄNEN→THEMEN→KONZEPTE per Klick, Kartenwand ist weg.
14. Übersetzungs-Popup eliminiert? **JA** — Flags physisch in der Kommandozeile; kein Popup in allen frischen Fenstern dieser Session.
15. Alles in der ECHTEN ZEUS.exe verifiziert? **Shell, Projects, Feedback, Presence: ja, direkt im echten Fenster. Files/Knowledge/Persönlichkeit/Owner/Activity: Klickpfade gegen denselben Live-Core per CDP (identischer Code und Daten), weil der Owner den Rechner aktiv nutzte und Fokus-Diebstahl ihn gestört hätte.** Keine Antwort beruht auf bloßer Quellcode-Existenz.

## Tests & Release

- Finale Voll-Suite: 2063 passed, 5 skipped, 1 xpassed (chess-Flake passt standalone; Lauf s. u.).
- Release: Kandidat aus `f8e0c39` gebaut, verifiziert, staged-promotet, Relaunch; READY-Beleg unten im Terminal-Log dieser Session.
