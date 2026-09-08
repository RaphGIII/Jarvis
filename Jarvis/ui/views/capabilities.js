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

export const view = {
  id: "capabilities",
  title: "Capabilities",
  async mount(pane, params) {
    const [data, queue] = await Promise.all([api("/api/capabilities"), api("/api/capabilities/requests").catch(() => ({ requests: [] }))]);
    const caps = data.capabilities || [];
    const pending = (queue && queue.requests) || [];
    const search = el("input", { placeholder: "Filter capabilities…", value: params.q || "" });
    const grid = el("div", { class: "grid" });
    const render = () => {
      clear(grid);
      const q = search.value.toLowerCase();
      const shown = caps.filter((c) => !q || `${c.capability_id} ${c.description} ${(c.aliases || []).join(" ")}`.toLowerCase().includes(q));
      if (!shown.length) grid.append(el("div", { class: "empty", text: caps.length ? "Nothing matches." : "No capabilities acquired yet. Ask for something ZEUS cannot do and it learns it." }));
      for (const c of shown) {
        const h = health(c);
        const life = lifecycle(c);
        const card = el("div", { class: "card click" }, el("div", { class: "title", text: c.capability_id }),
          el("div", { class: "meta" }, badge(life, LIFECYCLE_TONE[life] || "dim"), badge(h.health, HEALTH_TONE[h.health] || "dim"),
            el("span", { text: `v${c.version || "?"}` }),
            c.codex_required ? badge("needs Codex", "bad") : badge("local", "ok"),
            h.consecutive_failures ? el("span", { text: `${h.consecutive_failures} failure(s) in a row` }) : null,
            el("span", { text: purpose(c) })));
        card.onclick = () => inspect(c);
        grid.append(card);
      }
    };
    search.oninput = render;
    const broken = caps.filter((c) => health(c).health === "BROKEN").length;
    const remote = caps.filter((c) => c.codex_required).length;
    pane.append(el("div", { class: "toolbar" }, search,
      el("span", { class: "empty", style: { padding: 0 },
        text: `${caps.length} registered${broken ? ` · ${broken} BROKEN` : ""}${remote ? ` · ${remote} still need Codex` : ""}` })));
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
  const rows = pending.map((r) => el("div", { class: "kv" },
    el("span", { class: "k", text: new Date((r.requested_at || 0) * 1000).toLocaleString() }),
    el("span", { class: "v", text: `${r.goal} — ${r.reason || "queued"}` })));
  return el("div", { class: "card" }, el("div", { class: "title", text: `${pending.length} queued capability request(s)` }),
    el("div", { class: "meta" }, badge("waiting for Codex", "warn")), ...rows);
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
    section("Contract", kv("id", c.capability_id), kv("family", c.family || "—"), kv("version", c.version), kv("purpose", purpose(c)),
      kv("inputs", schemaKeys(c.input_schema)), kv("outputs", schemaKeys(c.output_schema)),
      kv("preconditions", list(c.preconditions) || ((c.permissions_required || []).length ? `permissions: ${c.permissions_required.join(", ")}` : "none declared")),
      kv("effects / side effects", list(c.side_effects) || meta.effects || meta.side_effects || "not declared"),
      kv("dependencies", (c.runtime_dependencies || c.dependencies || []).join(", ") || "none"),
      kv("provider", c.provider || meta.provider || meta.author || "local build"),
      kv("verification strategy", checks.length ? checks.map((x) => x.name || x.check).filter(Boolean).join(", ") : "none recorded"),
      kv("entrypoint", c.entrypoint), kv("implementation", c.implementation_path || c.source_location, "mono")),
    section("Resolution", kv("goal types", list(c.goal_types) || "—"), kv("target types", list(c.target_types) || "—"),
      kv("examples", list(c.examples) || "—"), kv("anti-examples", list(c.anti_examples) || "—"),
      kv("learned phrasings (aliases)", list(c.aliases) || "—"),
      kv("semantic signature", c.semantic_signature || "—", "mono"),
      kv("supersedes", list(c.supersedes) || "—")),
    section("Runtime", kv("lifecycle", lifecycle(c)), kv("installation status", c.status), kv("runtime health", h.health),
      kv("runs on", c.codex_required ? "CODEX REQUIRED — this is not finished work" : "locally, no engineer"),
      kv("runtime brain", c.runtime_brain || "none"), kv("latency class", c.latency_class || "unknown"),
      kv("security level", c.security_level ?? 0), kv("built by", `${c.created_by || "?"} (${c.source || "?"})`),
      kv("calls", h.calls), kv("success rate", h.calls ? `${Math.round((c.success_rate || 0) * 100)}%` : "—"),
      kv("failures", c.failure_count ?? h.consecutive_failures),
      kv("last used", h.last_used || meta.last_used || "never"),
      kv("last verified", c.last_verified || validation.verified_at || meta.verified_at || (validation.verified ? "at acquisition" : "")),
      kv("last ok", h.last_ok_at || "—"),
      kv("last error", h.last_error_at ? `${h.last_error_at}: ${h.last_error}` : "—"), kv("consecutive failures", h.consecutive_failures)),
    h.repairs.length ? section("Repair history", ...h.repairs.map((r) => kv(r.at, `${r.ok ? "ok" : "failed"} — ${r.detail}`))) : null,
    checks.length ? section("Acquisition verification", ...checks.map((x) => el("div", { class: "kv" }, el("span", { class: "k", text: (x.ok ?? x.passed) === false ? "✗" : "✓" }), el("span", { class: "v", text: x.name || x.check || x.detail || JSON.stringify(x).slice(0, 120) })))) : null,
    section("Description (as registered)", el("div", { class: "kv" }, el("span", { class: "v", text: c.description || "" }))),
    el("div", { class: "toolbar" },
      button("Test", () => chat.send(`Zeus, teste deine Fähigkeit ${c.capability_id} mit einem echten Aufruf und berichte das Ergebnis.`), "primary"),
      button("Repair", () => chat.send(`Zeus, repariere deine Fähigkeit ${c.capability_id}, sie funktioniert nicht richtig.`)),
      button("Improve", () => chat.send(`Zeus, verbessere deine Fähigkeit ${c.capability_id}.`)),
      button("History", () => views.open("activity", { q: c.capability_id })),
      button("Evidence (receipts)", () => views.open("activity", { q: `capability.${c.capability_id}` })),
      button(c.status === "disabled" ? "Enable" : "Disable", () => chat.send(c.status === "disabled" ? `Zeus, aktiviere deine Fähigkeit ${c.capability_id} wieder.` : `Zeus, deaktiviere deine Fähigkeit ${c.capability_id}.`), "ghost danger")),
  );
}
