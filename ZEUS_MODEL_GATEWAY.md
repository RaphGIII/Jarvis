# ZEUS Model Gateway — Stand nach Sprint 4 (2026-09-14)

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
- **Fähigkeit nach Bedeutung** (§10, Teil): der semantische Planer sieht die installierten, gesunden Fähigkeiten (id, Zweck, zwei Beispiele) und darf `capability.run <id>` aus dieser geschlossenen Liste wählen. Eine Anfrage mit konkretem Objekt (Pfad, Datei, Ordner), die der lexikalische Resolver nicht kennt, bekommt diese Lesung vor der Prosa-Antwort; reiner Chat zahlt den Aufruf nicht. Kommt die Wahl vom Offline-Modell oder mit niedriger Konfidenz, fragt ZEUS („Meinst du: …?“) statt zu handeln — das 4B-Modell hat schon einmal Entropie mit Prüfsumme verwechselt.

## Sprint 4 (2026-09-14): Capability Semantic Contracts und Kompositionsplaner

**Manifest V2** (`capabilities/contracts.py`, `capabilities/models.py`): jedes Manifest trägt ein kompaktes `contract`-Objekt mit `goals`, `events`, `consumes`, `produces`, `preconditions`, `effects`, `permissions`, `dependencies`, `tests`, `implementation_entrypoint` und optional `related_projects`, `domain`, `risk_class` (harmless/reversible/irreversible), `latency_class` (fast/medium/slow), `cost_class` (free/cheap/metered). Tokens sind `snake_case`, leere Felder werden nicht gespeichert, die Registry validiert. Alt-Manifeste bekommen einen abgeleiteten Vertrag (`<id>.result` als Effekt, `inferred: true`) und bleiben planbar als Endpunkt. Ein Engineer deklariert den Vertrag als `contract.json` neben `main.py`; die Installation übernimmt ihn. `SkillSpecification.metadata["contract"]` fließt ebenfalls ein. Der Katalog (`CAPABILITIES.yaml`) listet jeden Vertrag plus einen Effekt-Index (produced_by / consumed_by / goals_served_by / events_relevant_to).

**Weltmodell** (`service/world.py`): Fakten mit Herkunft und Ablauf. Ereignisse (`note_event`, `POST /api/world/event`), produzierte Fakten aus verifizierten Ausführungen (Vertrag → `provides`), Owner-Projekte als `project.<slug>`. Ereignisdetails werden zu Fakten (`chess_game_finished.result_loss`).

**Planer** (`capabilities/planner.py`): Zustandsraumsuche von S0 über anwendbare Fähigkeiten (`requires ⊆ S`, `S' = S ∪ provides`) bis das Ziel gilt, Tiefe ≤ 6, mit Relevanzfilter (nur Fähigkeiten, deren Effekte das Ziel transitiv braucht) und Branch-and-Bound. Ranking: `PlanCost = execution_cost + latency_penalty + risk_penalty + uncertainty_penalty + steps_penalty` (free/cheap/metered = 0/1/3; fast/medium/slow = 0/1/2; harmless/reversible/irreversible = 0/1/4; Unsicherheit = 5 × (1 − Zuverlässigkeit); 0,5 je Schritt). Zuverlässigkeit: gelernte `success_rate` ab 3 Aufrufen, sonst aus Health (HEALTHY 0,95, AT_RISK 0,6, abzüglich Fehlversuche); BROKEN wird nie verwendet. Der kürzeste Plan verliert gegen einen zuverlässigeren. Zieltokens, die ein `goals`-Eintrag sind, werden auf die Effekte der dienenden Fähigkeiten abgebildet.

**Fehlende Fähigkeit** nur nach Beweis: Abschluss aller erreichbaren Fakten; `missing_effects` = Zielfakten, die nichts produziert; `closest_partial_plan` = die längste laufbare Kette im Themengebiet; `unmet_inputs` = Fakten, die die Welt liefern muss (als `events` deklariert). `kind = unmet_input` (nichts bauen, Voraussetzung nennen) vs. `missing_capability` (Engineering mit `engineering_brief`: Ziel, Zustand, erreichte Kette, fehlender Effekt, geforderter `contract.json`).

**Semantische Zielableitung** (`capabilities/semantic_goals.py`): geschlossenes Vokabular aus allen `goals`/`produces`/`effects`, als JSON-Enum an den konfigurierten Provider (Gateway). Erfundene Tokens werden verworfen. Grounding: ein Ziel gilt nur, wenn ein `events`-Token im Weltzustand steht, ein `related_projects`-Projekt aktiv ist oder die Anfrage Wörter aus dem Vokabular der Fähigkeit enthält; sonst CLARIFY. Antwortet nur der Offline-Fallback (4B), ist das Ergebnis `UNAVAILABLE` — typisiert, nie eine Entscheidung. Reiner Chat ohne Kontext, konkretes Objekt oder Vokabelüberschneidung zahlt keinen Aufruf.

**§7-Isolation**: `JarvisCore.semantic_authority()` prüft ohne Netz, ob die Route ein Cloud-Denkmodell wäre. Modellbasierte Komposition (`_answer_by_composition` ohne fertigen Plan) und der `capability.missing`-Zweig laufen nur mit Autorität; sonst typisierte Meldung. Der Vertragsplaner übergibt fertige Pläne an die bestehende Ausführungsmaschine (Receipts, GOAL_SATISFIED, Outcome-Lernen).

**Metriken** (`catalog/metrics.py`): `manifest_tokens`, `catalog_context_tokens`, `source_tokens`, `other_tokens`, `total_tokens` je Kompositions-Ergebnis (`/api/compose/plan`) und je Engineering-Kontext. Gemessen im Held-out-Szenario: drei Verträge plus Satz < 2000 Tokens.

**Held-out-Test** (`tests/test_semantic_composition.py`): Projekt „Schach Training“, Ereignis `chess_game_finished result=loss`, vier Paraphrasen, die in keinem Manifest stehen → Plan capture → analyze → classify ausgeführt, GOAL_SATISFIED, Weltmodell aktualisiert. Negativkontrollen: ohne Kontext keine semantische Anfrage; fremdes Ereignis → Rückfrage; abgelaufenes Ereignis → keine Aktion; Fachwort ohne Ereignis → „Voraussetzung fehlt“, nichts gebaut; fehlender Effekt → Engineering mit Beweis; nur Offline-Modell → typisiert nicht verfügbar, nichts ausgeführt. Ehrlich: der Fake-Provider ist absichtlich „trigger-happy“; die Negativkontrollen beweisen die Systemgrenzen, nicht das Sprachverständnis des Modells. Ein Live-Test (`ZEUS_LIVE_SEMANTIC=1`) prüft den echten Provider.

## Sprint 5 (2026-09-14): Modellgetriebene Komposition — das Denkmodell ist die Intelligenz

**Architekturklärung.** Der Kompositionsplaner ist kein zweites Gehirn. Er erdet Tokens, prüft Machbarkeit, findet triviale Pfade deterministisch und beweist, dass ein Effekt unerreichbar ist. Verstehen und Planen tut das gewählte Denkmodell hinter dem Gateway. Der alte Ableitungs- und Kompositionspfad (`capabilities/semantic_goals.py`, `capabilities/composition.py`, `Composer.plan/replan`) ist entfernt; es gibt genau einen Kompositionspfad.

**Der autoritative Fluss** (`capabilities/intelligence.py`, `JarvisCore._answer_by_intelligence`): Owner-Äußerung → Retrieval des relevanten Welt-/Kontextausschnitts (deterministisch) → Denkmodell → `GoalSpec` → kompakte Fähigkeitskarten (Stufe 1) → deterministischer Kurzweg, sonst Denkmodell mit vollständigen Verträgen (Stufe 2) → `PlanSpec` → deterministische Validierung (requires/provides-Simulation) → bei Ungültigkeit geht genau das Problem (fehlender Effekt, unbekannte id, nicht erreichtes Ziel) einmal an das Modell zurück → bleibt es ungültig, beweist der Planer: Pfad (nur wenn keine Eingaben fehlen) oder `MISSING_CAPABILITY` mit Evidenz → Ausführung über die bestehende Missionsmaschine (Receipts) → unabhängige Verifikation → `GOAL_SATISFIED`. Rohfehler erreichen den Owner nicht: Validierungsprobleme gehen an das Modell, der Owner bekommt eine Frage oder das Ergebnis.

**GoalSpec**: `primary_goal`, `secondary_goals`, `referenced_entities`, `relevant_project`, `relevant_recent_events`, `constraints`, `privacy_class` (public/private/secret), `ambiguity`, `confidence`, `reason`, `clarification_question`; das Vokabular ist ein JSON-Enum aus den `goals`/`produces` der abgerufenen Karten, Projekttitel und Ereignistokens aus dem abgerufenen Weltausschnitt. Erfundene Tokens werden verworfen und unter `rejected` gelistet (ein erfundenes Primärziel wird nicht durch ein Sekundärziel ersetzt). **PlanSpec**: geordnete `steps` (`capability_id` aus der Enum, `intended_effect`, `why`, `arguments_json`, `role` required/optional), `expected_final_goal`, `required_permissions`, `reason`, `missing`, `source` (provider / reproposal / deterministic).

**Zweistufiges Retrieval**: eine Karte ist relevant, wenn eines ihrer `events` im Weltzustand steht, ein `related_projects`-Projekt aktiv ist oder die Anfrage ein Wort ihres Vokabulars enthält; Verbinder (konsumiert, was eine relevante Karte produziert, oder produziert, was eine braucht) folgen bis zum Fixpunkt. Nur deren Ereignisse und Projekte gehen mit. Stufe 1 = eine Zeile je Karte; Stufe 2 = Vertrag + Eingabeschema nur der abgerufenen Karten. Gemessen (Held-out, drei Verträge): `capability_summary_tokens` ≈ 60–250, `detailed_contract_tokens` > Stufe 1, `world_context_tokens` < 200, `total_model_context_tokens` < 3000 bei zwei Provider-Aufrufen. Bei einer Wetterfrage ohne Kontext wird nichts abgerufen und kein Aufruf bezahlt.

**Modellautorität**: liefert das Gateway nur den Offline-Fallback oder keine Route, ist das Ergebnis der typisierte Status `INTELLIGENCE_UNAVAILABLE`; das lokale Modell erzeugt weder GoalSpec noch PlanSpec. Mit Weltkontext (ein Spiel ist gerade zu Ende) wird das dem Owner gesagt; ohne Kontext übernehmen die typisierten Einzelaktionspfade (Datei schreiben, registrierte Fähigkeit) oder Prosa.

**EngineeringSpec** (`capabilities/engineering_spec.py`): `owner_goal`, `goal_spec`, `existing_coverage`, `closest_partial_plan`, `missing_effects`, `reusable_capabilities`, `relevant_interfaces`, `required_permissions`, `acceptance_criteria`, `privacy_constraints`, `catalog_context` (minimal, über den Context Builder), `suggested_contract`, `task_class`, `metrics`; gespeichert unter `data/jarvis/engineering_specs/<id>.json`, als Brief an den Engineer. Der Engineering-Router entscheidet **vor** der Ausführung (`_engineer_for_capability`: Task-Vektor aus der Spec, Rollen engineer.codex/engineer.standard/engineer.frontier); ohne Route wird die Spec behalten und der Owner informiert — kein Engineer, kein lokaler Coder, kein Standard→Frontier-Fallback. Der gewählte API-Engineer wird dem Expert-Gateway namentlich übergeben; der API-Engineer arbeitet jetzt auch in einem Fähigkeits-Workspace ohne Git.

**Echte Ereignisquellen**: jedes *verifizierte* Receipt wird zum Weltereignis (`project_created`, `media_playing`/`media_paused`, `file_written`/`file_read`, `action.<kind>.verified`, plus die deklarierten Effekte der Built-ins, z. B. `knowledge_node`); unverifizierte Receipts verändern das Weltmodell nicht. Es gibt keinen abgeleiteten Weltzustand.

**Held-out** (`tests/test_semantic_composition.py`, `tests/test_intelligence_flow.py`): „fuck, schon wieder verloren“, „was mache ich eigentlich dauernd falsch?“, „nimm die letzte partie mal auseinander“, „mach daraus was für mein training“ → mit Ereignis und Projekt: GoalSpec, vom Modell vorgeschlagener Plan capture → analyze → classify, validiert, ausgeführt, GOAL_SATISFIED, Weltmodell aktualisiert, kein Aufruf ans lokale Modell. Negativkontrollen ohne Kontext: die ersten drei Sätze lösen keinen semantischen Aufruf aus (Konversation bzw. Rückfrage ohne Planer-Diagnostik); „training“ trifft ein Vertragswort, das Modell antwortet „none“ und fragt; fremdes Ereignis → keine Planung; abgelaufenes Ereignis → keine Aktion; ein vom Modell behauptetes, aber ungeerdetes Ziel → Rückfrage ohne PlanSpec; Wortgrund ohne Spiel → „Voraussetzung fehlt“, nichts gebaut; fehlender Effekt → EngineeringSpec mit Beweis, kein Engineer ohne Route. Ehrlich: der Provider ist im Test geskriptet; er antwortet nur mit Ereignis oder Projekt im abgerufenen Kontext mit dem Schachziel. Der Live-Test (`ZEUS_LIVE_SEMANTIC=1`) prüft den echten Provider.

## Was ausdrücklich noch fehlt (ehrlich)

Aus P0:

- **§9/§10/§11** — gebaut (Sprint 4/5). Ereignisquellen: `/api/world/event` und jedes verifizierte Receipt (Projekt, Medien, Datei, Fähigkeit, Built-ins); kein Bildschirm-/Sitzungsbeobachter, keine Schach-Quelle im Produkt. Offen: der `COMPOSE`-Override des Gateway-Routers ist eine Routing-Klasse; der eine Kompositionspfad ist `_answer_by_intelligence`.
- **§12/§13 Engineering Router** — umgestellt (Sprint 2), `EngineeringSpec` und Entscheidung vor der Ausführung (Sprint 5). Der API-Engineer arbeitet als Diff-Generator mit einer Reparaturrunde, nicht als mehrstufiger Agent mit Werkzeugen; ein echter API-Engineer-Lauf ist noch nicht erfolgt (keine Schlüssel).
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
