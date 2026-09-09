# ZEUS Model Gateway — Stand nach Sprint 1 (2026-09-09)

Dieses Dokument beschreibt, was vom P0-Brief „ZEUS Intelligence Platform /
Model Gateway“ gebaut ist, wie es verdrahtet ist, wie man es prüft, und was
ausdrücklich noch fehlt. Es ist absichtlich nüchtern: nichts hier ist eine
Attrappe, aber vieles aus dem Brief ist noch nicht begonnen.

## Was gebaut ist

### Das Paket `Jarvis/gateway/`

| Modul | Aufgabe |
|---|---|
| `modes.py` | `ChatMode` AUTO/FREE/SMART/DEEP/BUILD und was jeder Modus erlaubt (`ModePolicy`). |
| `config.py` | Rollen (`reasoning.free`, `reasoning.deep`, `engineer.standard`, `engineer.frontier`, `speech.stt`, `speech.tts`, `local.fast`, `local.build`) → Provider/Modell aus `config/providers.json`. Preise pro 1M Tokens in EUR, Budget-Caps. Kein Modellname außerhalb dieser Datei (Test erzwingt das). |
| `secrets.py` | `CredentialStore`: API-Schlüssel per Windows DPAPI verschlüsselt unter `data/jarvis/owner/provider_credentials.json`. Nur Präsenz + vier Endzeichen werden je berichtet. `redact()` schrubbt Schlüssel aus jedem Fehlertext. |
| `estimate.py` | Token- und Kostenschätzung vor dem Aufruf; `actual_cost()` aus der Provider-Usage. |
| `budget.py` | `BudgetGovernor`: `reserve(estimate × 1.5)` → Aufruf → `settle(actual)`. Harte Caps: Monat (€40), Tag, pro Aufgabe, Provider, Denken, Engineering. Append-only Ledger `data/jarvis/gateway/spend.jsonl`, restart-fest. Verweigerte Reservierung = kein Aufruf. |
| `privacy.py` | Privacy-Router vor dem Modellaufruf: Chunks einzeln als PUBLIC/PRIVATE/SECRET klassifiziert (Quelle + Inhalt). Provider mit `may_train_on_requests` bekommen nur PUBLIC. SECRET verlässt die Maschine nie. Sensible Owner-Nachricht schließt die Free-Lane für die ganze Anfrage. |
| `task.py` | Typisierter Task-Vektor (13 Features aus dem Brief). Objektive Features aus ZEUS-Fakten (`TaskFacts`), weiche Features darf ein Modell nur *anheben*: `D_final = max(rule, semantic)`. `required_reliability` = τ(x). |
| `learning.py` | Performance-Ledger (kein Prompt, kein Chain-of-Thought) und `ReliabilityModel`: Beta-Posterior je (Rolle, Aufgabenklasse), Prior aus Config, `q = p̂ − β·σ`. Nur `GOAL_VERIFIED`-Beobachtungen bewegen die Schätzung, Provider-Ausfälle nie. |
| `router.py` | Harte Regeln zuerst (deterministische Fähigkeit → lokal; BROKEN/missing → Engineering; destruktiv → Owner-Freigabe; Ambiguität → Rückfrage; ≥2 Subsysteme → Komposition; frische Info → Research). Dann `argmin Cost(m,x)` unter `q(m,x) ≥ τ(x)`. Kein Kaskadieren: die Wahl fällt vor dem Aufruf. |
| `health.py` | Sechs Zustände: provider_unavailable, rate_limit, quota_exhausted, authentication_error, model_unavailable, task_failure. Cooldowns pro Provider. Ein Ausfall ist kein Beleg für „braucht stärkeres Modell“. |
| `transport.py` | Der einzige HTTP-Weg. `Transport.issue()` stellt ein Ticket aus — und verweigert es bei FREE + metered (`ZeroCostViolation`) oder metered ohne Reservierung (`ReservationRequired`), bevor ein Byte gesendet wird. URL muss unter der Provider-Base-URL liegen. |
| `providers.py` | Adapter Gemini (`generateContent`, thinkingBudget, responseSchema-Sanitizer), OpenAI (`chat/completions`, `reasoning_effort`, cached tokens), Anthropic (`/v1/messages`, cache_read), OpenAI-kompatibel. Liefern Text + normalisierte Usage. |
| `persona.py` | System-Prompt jeder Cloud-Rolle = Owner-Persona (`config.system_prompt()`) + Identitätsklausel. Ausgabe-Guard schreibt „Ich bin Gemini …“ / „as an AI model from OpenAI“ zu ZEUS um. Provider-Identität bleibt in Diagnostics sichtbar. |
| `gateway.py` | `ModelGateway.complete()` führt alles in Reihenfolge aus. `GatewayBrainProvider` spricht das bestehende `BrainProvider`-Protokoll (generate/generate_structured/generate_stream), sodass alle bestehenden Aufrufstellen ohne Umbau über das Gateway laufen. Chat-Modus und Task-Fakten kommen aus einer `contextvars`-Variable, die der Antwort-Thread setzt. |

### Verdrahtung ins Produkt

- `core/kernel.py`: `kernel.provider(FAST_LOCAL)` liefert jetzt den `GatewayBrainProvider` mit dem lokalen Ollama-Modell als **Offline-Fallback**. Alle 15 FAST_LOCAL-Aufrufstellen (semantische Interpretation, Komposition, Aktionsplanung, Konversation, Coach, Zusammenfassung, Research …) laufen damit über das Gateway. `kernel.local_provider(tier)` ist der Rohzugriff für GPU-Housekeeping (Warmup, Eviction). `ZEUS_GATEWAY=0` schaltet zurück auf rein lokal.
- `service/core.py`: `chat_mode` pro Konversation, `set_chat_mode()`, Modus im `/api/message`-Body, `RequestContext` im Antwort-Thread, objektive Fakten nach der Klassifikation (`_gateway_facts`), API-Methoden (`gateway_status`, `gateway_estimate`, `providers_status`, `provider_set_credential`, `provider_clear_credential`, `provider_enable`, `gateway_ledger`), Gateway in `diagnostics()`. Modus wird in den Preferences gemerkt und beim Start wiederhergestellt.
- `service/http.py`: `/api/gateway/status|mode|estimate|ledger`, `/api/providers`, `/api/providers/credential`, `/api/providers/credential/clear`, `/api/providers/enable`. Credential-Endpunkte verlangen die Owner-Freigabe `CREDENTIALS`.
- UI: `ui/core/gateway.js` rendert die Modus-Leiste über dem Composer (AUTO/FREE/SMART/DEEP/BUILD, Schätzung beim Tippen, `Monat: €x / €40.00`, „nur lokal“ wenn kein Cloud-Denkmodell konfiguriert). `chat.js` sendet den Modus mit. `owner.js` hat die Sektion „Provider / Modell-Gateway“: Schlüssel eingeben (write-only), Provider aktivieren, Preise, Rollen, Budgetstand.

### Prüfen

```bash
cd Jarvis
python -m pytest tests/test_model_gateway.py tests/test_gateway_integration.py -q   # 50 Tests, offline
python -m jarvis.verify_ui                                                            # UI-Gate
```

Live geprüft am 2026-09-09 auf einer zweiten Instanz (`python -m jarvis.serve --port 8477 --no-warm`): Modus-Klick ändert Server-Wahrheit, Schätzung erscheint, FREE-Frage „Was bedeutet Netzwerkadressübersetzung?“ wurde in 3 s über `local.fast` (Offline-Fallback, 568/65 Tokens, €0.00) beantwortet und im Gateway verbucht.

### Was das Gateway heute konkret garantiert

1. **FREE ist eine Mauer.** Bei `ChatMode.FREE` verweigert `Transport.issue()` jeden metered Provider, bevor ein Request gebaut wird. Kein Adapter hat einen anderen Netzwerkweg.
2. **Keine Reservierung, kein Aufruf.** Metered Rollen ohne Preis in der Config sind nicht aufrufbar.
3. **Keine Kaskade.** Ein Ausfall (429/5xx/401) wird klassifiziert, der Provider bekommt einen Cooldown, und die Anfrage wird *nicht* automatisch bei einem teureren Modell wiederholt.
4. **Private Inhalte gehen nie an trainierende Provider.** Owner-Nachricht mit IBAN/Befund/Passwort schließt die Free-Lane komplett.
5. **ZEUS bleibt ZEUS.** Identitätsklausel im System-Prompt plus Ausgabe-Guard; Provider nur in Diagnostics.

## Sprint 2 (gleicher Tag): Outcomes, Engineering-Router, Owner-Kostenpolitik

- **Goal-Verification als Lernsignal.** Jede Gateway-Antwort hängt an ihrer Request-ID, bis die Welt geprüft wurde: Receipt einer Projektoperation, `GOAL_SATISFIED` einer Komposition, Verifikation einer Fähigkeit/Aktion, oder der Daumen des Owners auf eine Konversationsantwort (`/api/feedback`). Erst dann bewegt sich die Zuverlässigkeitsschätzung. Nie beurteilte Antworten werden vergessen, nicht als Erfolg gezählt.
- **Owner-Spending-Dokument ist die Kostenpolitik.** `CostPolicy.load()` liest `config/owner/spending.json` (`paid_api`, `usage_credits`, `cloud_gpu` …) über `cost_policy.json`. Der Router lässt metered Routen nur zu, wenn `paid_api` freigegeben ist. Heute ist es aus: SMART/DEEP/BUILD nutzen bis zur Owner-Transaktion nur kostenlose Wege, und die UI sagt das.
- **Engineering-Router auf Gateway-Rollen.** `engineer.codex` (Subscription, €0), `engineer.standard`, `engineer.frontier` werden auf gelernter Zuverlässigkeit je Engineering-Klasse und Kosten gerankt, *vor* der Ausführung. Kleine Änderung → Codex; großes neues Subsystem → direkt Frontier (nur in BUILD, nur mit freigegebener bezahlter API, nur im Budget). Sonst Codex als „best available“ mit ehrlicher Begründung, oder die Warteschlange.
- **API-Engineer** (`experts/api_engineer.py`): Brief plus relevante Dateien an die gewählte Rolle, Unified Diff zurück, `git apply` im Kandidaten-Worktree, eine Reparaturrunde, dann die unabhängige Verifikation des Expert-Gateways. Als `explicit_only` markiert: das Expert-Gateway iteriert nie von einem nicht verfügbaren Codex zu einem bezahlten Engineer.
- **SelfDev** trägt den Chat-Modus der Anfrage (`mission.chat_mode`), baut den Engineering-Task-Vektor aus den INVESTIGATE-Fakten und reicht den Job an den benannten Provider.

## Sprint 3: ZEUS Catalog (P0.6), Build-Freigabe, weiche Features, Intent-Fix

- **Katalog** (`Jarvis/catalog/`): `python -m catalog.build` erzeugt deterministisch `ZEUS_MAP.yaml` (Pakete, Module, Zweck, Tests je Modul, API-Routen, UI-Module, Entry Points), `INTERFACES.yaml` (öffentliche Klassen/Methoden/Funktionen mit Signaturen), `DEPENDENCY_GRAPH.json` (Imports und Dependents), `ARCHITECTURE.md` (Überblick) und `CAPABILITIES.yaml` (Laufzeitfakten, nicht Teil der Aktualitätsprüfung). `python -m catalog.check` meldet veraltete Module, Verträge, Abhängigkeiten, Tests, Zwecke, Routen; `tests/test_catalog.py` macht einen veralteten Katalog zum Testfehler. Die SelfDev-Verifikation regeneriert den Katalog im Kandidaten, damit jede Änderung ihren Katalog mitbringt.
- **Engineering Context Builder** (`catalog/context.py`): aus den betroffenen Dateien die Manifeste, Verträge, direkten Dependents, zugehörigen Tests und Vertrags-Ausschnitte der direkten Abhängigkeiten, in einem Zeichenbudget und nach Relevanz geordnet. Codex bekommt ihn als `ExpertJob.context`, der API-Engineer vor den Dateiinhalten. Gemessen: zwei Module → ~15 KB Kontext statt Repository-Exploration.
- **Build-Freigabe** (§15): ein bezahlter Engineer parkt die Mission bei AWAITING_BUILD mit Schätzbereich und hartem Maximum; ausgegeben wird erst nach „Build starten“ unter Missions.
- **Weiche Features** (§7): der semantische Planer schätzt im selben Aufruf reasoning_depth, context_dependency, long_horizon, novelty; sie heben nur an (`max(rule, semantic)`) und steuern Denkstufe und Modellwahl der folgenden Aufrufe.
- **Intent-Fix**: „Was ist NAT? Antworte in einem Satz.“ ist keine Ordnersuche mehr.

## Was ausdrücklich noch fehlt (ehrlich)

Aus P0:

- **§9 Semantic World Model, §10 Capability Semantic Contract, §11 Capability Graph/Komposition** — nicht begonnen. Der Router hat einen `COMPOSE`-Override, aber keine Effekt/Vorbedingungs-Suche über existierende Fähigkeiten.
- **§12/§13 Engineering Router** — umgestellt (Sprint 2). Offen: eine strukturierte `EngineeringSpec` und ein „Start build“-Dialog mit Schätzung *vor* dem Spending; heute begrenzen BUILD-Modus, Owner-Spending-Freigabe und die Caps das Ausgeben. Der API-Engineer arbeitet als Diff-Generator mit einer Reparaturrunde, nicht als mehrstufiger Agent mit Werkzeugen.
- **§7 weiche Features** — verdrahtet (Sprint 3) über den semantischen Planer; Ambiguität bleibt dessen `clarify`-Entscheidung.
- **§20 Goal Verification als Lernsignal** — verdrahtet (Sprint 2) für Projektoperationen, Kompositionen, Fähigkeiten, Aktionen und Owner-Feedback. Konversationsantworten ohne Feedback bleiben unbeurteilt.
- **Preise** — die Werte in `config/providers.json` sind unbestätigte Platzhalter (`confirmed: false`); die UI sagt das dazu. Der Owner muss sie gegen die Preislisten prüfen.
- **Kein echter Provider-Aufruf** ist bisher erfolgt (keine Schlüssel hinterlegt). Die Adapter sind gegen die dokumentierten Antwortformen getestet, nicht gegen den Live-Dienst. Gemini-`thinkingBudget` und OpenAI-`max_completion_tokens`/`reasoning_effort` sind nach Doku, nicht live verifiziert.
- **Threadsicherheit der Reservierungen** ist per RLock gegeben; zwei ZEUS-Prozesse über *ein* Ledger sind nicht abgesichert (bekannte Falle, siehe Registry).

P0.5 (Voice: TEXT/LISTEN/TALK, `speech.stt/tts`) — nur die Rollen sind in der Config deklariert, keine Adapter, keine UI.
P0.6 (ZEUS Catalog, Engineering Context Builder) — gebaut (Sprint 3). Offen: Modul-Manifeste mit Permissions je Capability und ein Impact-Test-Lauf statt der vollen Suite bei isolierten Änderungen.
UI-Komplettumbau — nicht begonnen; die Modus-Leiste und die Provider-Sektion folgen dem bestehenden Design.

## Beobachtungen aus dem Live-Test

- „Was ist NAT? Antworte in einem Satz.“ wurde vom deterministischen Dateisystem-Router als Suche nach einem Ordner „Satz“ verstanden und nie ans Gateway gereicht. Das ist der bestehende Regex-Router, nicht das Gateway — und genau die Art Fehlinterpretation, die der Brief beschreibt. Der Router-Umbau auf semantische Deutung ist der nächste sinnvolle Schritt.
- `tests/test_action_receipts.py::test_a_new_conversation_clears_the_transcript_but_never_the_record` scheiterte bereits vor diesem Sprint auf dem Baseline-Stand (42fd946 + Revert).
