/* Capability Center: every acquired capability with its LIFECYCLE
   (DISCOVERED … ACTIVE … SUPERSEDED) and its RUNTIME HEALTH (HEALTHY /
   AT_RISK / BROKEN / UNKNOWN, written by every real execution) side by side —
   ACTIVE is not HEALTHY. Opening one shows the typed contract, how it
   resolves (goal/target types, examples, aliases), where it came from and
   whether it needs an engineer at runtime (it must not: codex_required=false
   is the whole point). Actions go through the normal router (Test / Repair /
   Improve / Disable); nothing here runs arbitrary code.

   The queue at the top is requests that arrived while Codex was unavailable.
   They are not failures and not silently dropped work — they are what ZEUS
   still owes the owner, visible until an acquisition serves them. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const LIFECYCLE_TONE = { ACTIVE: "ok", VERIFIED: "ok", TESTING: "blue", DRAFT: "blue", DISCOVERED: "dim",
  DEGRADED: "warn", BROKEN: "bad", DISABLED: "bad", SUPERSEDED: "dim" };
const HEALTH_TONE = { HEALTHY: "ok", AT_RISK: "warn", BROKEN: "bad", UNKNOWN: "dim" };
const LIFE_WORD = { ACTIVE: "aktiv", VERIFIED: "geprüft", TESTING: "in Prüfung", DRAFT: "Entwurf", DISCOVERED: "entdeckt", DEGRADED: "eingeschränkt", BROKEN: "gestört", DISABLED: "aus", SUPERSEDED: "ersetzt" };
const HEALTH_WORD = { HEALTHY: "gesund", AT_RISK: "gefährdet", BROKEN: "gestört", UNKNOWN: "ungeprüft" };
const readable = (id) => String(id || "").replace(/[._-]+/g, " ");

export const view = {
  id: "capabilities",
  title: "Fähigkeiten",
  async mount(pane, params) {
    const [data, queue] = await Promise.all([api("/api/capabilities"), api("/api/capabilities/requests").catch(() => ({ requests: [] }))]);
    const caps = data.capabilities || [];
    const pending = (queue && queue.requests) || [];
    const search = el("input", { placeholder: "Fähigkeiten filtern …", value: params.q || "" });
    const grid = el("div", { class: "grid" });
    const render = () => {
      clear(grid);
      const q = search.value.toLowerCase();
      const shown = caps.filter((c) => !q || `${c.capability_id} ${c.description} ${(c.aliases || []).join(" ")}`.toLowerCase().includes(q));
      if (!shown.length) grid.append(el("div", { class: "empty", text: caps.length ? "Nichts passt zu diesem Filter." : "Noch keine erworbenen Fähigkeiten. Bitte ZEUS um etwas, das es noch nicht kann, und es lernt es." }));
      for (const c of shown) {
        const h = health(c);
        const life = lifecycle(c);
        // what ZEUS can do, in words -- the technical id is secondary
        const card = el("div", { class: "card click" }, el("div", { class: "title", text: purpose(c) || readable(c.capability_id) }),
          el("div", { class: "meta" }, badge(LIFE_WORD[life] || life, LIFECYCLE_TONE[life] || "dim"), badge(HEALTH_WORD[h.health] || h.health, HEALTH_TONE[h.health] || "dim"),
            c.codex_required ? badge("braucht Verstärkung", "warn") : null,
            h.consecutive_failures ? el("span", { text: `${h.consecutive_failures}× zuletzt gescheitert` }) : null,
            el("span", { class: "mono-id", text: readable(c.capability_id) })));
        card.onclick = () => inspect(c);
        grid.append(card);
      }
    };
    search.oninput = render;
    const broken = caps.filter((c) => health(c).health === "BROKEN").length;
    const remote = caps.filter((c) => c.codex_required).length;
    pane.append(el("div", { class: "toolbar" }, search,
      el("span", { class: "empty", style: { padding: 0 },
        text: `${caps.length} Fähigkeiten${broken ? ` · ${broken} gestört` : ""}${remote ? ` · ${remote} unvollendet` : ""}` })));
    if (pending.length) pane.append(queueSection(pending));
    pane.append(grid);
    render();
    if (params.id) { const c = caps.find((x) => x.capability_id === params.id); if (c) inspect(c); }
  },
};

/* Requests parked because Codex was unavailable. Shown as owed work, with the
   state that parked them, so "I will do it later" is checkable rather than a
   promise made in a chat bubble that nobody can find again. */
function queueSection(pending) {
  const firstLine = (t) => String(t || "").split("\n").map((l) => l.trim()).filter(Boolean)[0] || "";
  const rows = pending.map((r) => el("div", { class: "kv" },
    el("span", { class: "k", text: new Date((r.requested_at || 0) * 1000).toLocaleDateString("de-DE") }),
    el("span", { class: "v", text: firstLine(r.goal).slice(0, 120) })));
  return el("div", { class: "card" }, el("div", { class: "title", text: `${pending.length} vorgemerkte Wünsche` }),
    el("div", { class: "meta" }, badge("wartet auf Verstärkung", "warn")), ...rows);
}

function parse(v) {
  if (typeof v !== "string") return v;
  try { return JSON.parse(v.split(String.fromCharCode(39)).join(String.fromCharCode(34)).replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false").replace(/\bNone\b/g, "null")); } catch { return v; }
}

function lifecycle(c) {
  return String(c.lifecycle || (c.status === "active" ? "ACTIVE" : "DISABLED")).toUpperCase();
}

function health(c) {
  const h = parse(c.health) || {};
  return { state: h.state || "unverified", health: String(c.health_state || h.health || "UNKNOWN").toUpperCase(),
    consecutive_failures: h.consecutive_failures || 0, calls: h.calls || 0, last_ok_at: h.last_ok_at || "", last_error_at: h.last_error_at || "",
    last_error: h.last_error || "", last_used: h.last_used || "", repairs: Array.isArray(h.repairs) ? h.repairs : [] };
}

/* The stable purpose line: the first sentence of the description that is
   about the capability itself, never the acquisition prose ("A previous task
   of this kind was solved as follows"). */
function purpose(c) {
  const lines = String(c.description || "").split("\n").map((l) => l.trim()).filter(Boolean);
  const own = lines.find((l) => !/previous task|solved as follows|was solved|of this kind/i.test(l)) || lines[0] || "";
  return own.slice(0, 90);
}

function inspect(c) {
  const schemaKeys = (s) => { const p = parse(s); return p && p.properties ? Object.entries(p.properties).map(([k, v]) => `${k}${(p.required || []).includes(k) ? "*" : ""}: ${v.type || "?"}`).join(", ") : String(s || "").slice(0, 120); };
  const validation = parse(c.validation_status) || {};
  const checks = Array.isArray(validation.checks) ? validation.checks : [];
  const meta = parse(c.creation_metadata) || {};
  const h = health(c);
  const list = (x) => (Array.isArray(x) ? x : []).join(" · ");
  views.inspect(c.capability_id,
    section("Was sie kann", kv("Kennung", c.capability_id), kv("Familie", c.family || "—"), kv("Version", c.version), kv("Zweck", purpose(c)),
      kv("Eingaben", schemaKeys(c.input_schema)), kv("Ausgaben", schemaKeys(c.output_schema)),
      kv("Voraussetzungen", list(c.preconditions) || ((c.permissions_required || []).length ? `Berechtigungen: ${c.permissions_required.join(", ")}` : "keine angegeben")),
      kv("Wirkung", list(c.side_effects) || meta.effects || meta.side_effects || "nicht angegeben"),
      kv("Abhängigkeiten", (c.runtime_dependencies || c.dependencies || []).join(", ") || "keine"),
      kv("Quelle", c.provider || meta.provider || meta.author || "lokal gebaut"),
      kv("Prüfung", checks.length ? checks.map((x) => x.name || x.check).filter(Boolean).join(", ") : "keine hinterlegt"),
      kv("Einstieg", c.entrypoint), kv("Umsetzung", c.implementation_path || c.source_location, "mono")),
    section("Wann sie greift", kv("Zieltypen", list(c.goal_types) || "—"), kv("Objekte", list(c.target_types) || "—"),
      kv("Beispiele", list(c.examples) || "—"), kv("Gegenbeispiele", list(c.anti_examples) || "—"),
      kv("Gelernte Formulierungen", list(c.aliases) || "—"),
      kv("Bedeutungssignatur", c.semantic_signature || "—", "mono"),
      kv("Ersetzt", list(c.supersedes) || "—")),
    section("Betrieb", kv("Status", LIFE_WORD[lifecycle(c)] || lifecycle(c)), kv("Installation", c.status), kv("Gesundheit", HEALTH_WORD[h.health] || h.health),
      kv("Läuft", c.codex_required ? "noch nicht fertig – braucht Verstärkung" : "eingebaut, ohne Engineer"),
      kv("Denkt mit", c.runtime_brain || "nein"), kv("Antwortzeit", c.latency_class || "unbekannt"),
      kv("Sicherheitsstufe", c.security_level ?? 0), kv("Gebaut von", `${c.created_by || "?"} (${c.source || "?"})`),
      kv("Aufrufe", h.calls), kv("Erfolgsquote", h.calls ? `${Math.round((c.success_rate || 0) * 100)}%` : "—"),
      kv("Fehlschläge", c.failure_count ?? h.consecutive_failures),
      kv("Zuletzt genutzt", h.last_used || meta.last_used || "nie"),
      kv("Zuletzt geprüft", c.last_verified || validation.verified_at || meta.verified_at || (validation.verified ? "beim Erwerb" : "")),
      kv("Zuletzt erfolgreich", h.last_ok_at || "—"),
      kv("Letzter Fehler", h.last_error_at ? `${h.last_error_at}: ${h.last_error}` : "—"), kv("Fehlschläge in Folge", h.consecutive_failures)),
    h.repairs.length ? section("Reparaturen", ...h.repairs.map((r) => kv(r.at, `${r.ok ? "ok" : "failed"} — ${r.detail}`))) : null,
    checks.length ? section("Prüfung beim Erwerb", ...checks.map((x) => el("div", { class: "kv" }, el("span", { class: "k", text: (x.ok ?? x.passed) === false ? "✗" : "✓" }), el("span", { class: "v", text: x.name || x.check || x.detail || JSON.stringify(x).slice(0, 120) })))) : null,
    section("Beschreibung", el("div", { class: "kv" }, el("span", { class: "v", text: c.description || "" }))),
    el("div", { class: "toolbar" },
      button("Test", () => chat.send(`Zeus, teste deine Fähigkeit ${c.capability_id} mit einem echten Aufruf und berichte das Ergebnis.`), "primary"),
      button("Repair", () => chat.send(`Zeus, repariere deine Fähigkeit ${c.capability_id}, sie funktioniert nicht richtig.`)),
      button("Improve", () => chat.send(`Zeus, verbessere deine Fähigkeit ${c.capability_id}.`)),
      button("Verlauf", () => views.open("activity", { q: c.capability_id })),
      button("Belege", () => views.open("activity", { q: `capability.${c.capability_id}` })),
      button(c.status === "disabled" ? "Enable" : "Disable", () => chat.send(c.status === "disabled" ? `Zeus, aktiviere deine Fähigkeit ${c.capability_id} wieder.` : `Zeus, deaktiviere deine Fähigkeit ${c.capability_id}.`), "ghost danger")),
  );
}
