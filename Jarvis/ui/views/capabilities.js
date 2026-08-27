/* Capability Center: every acquired capability with its INSTALLATION STATE
   (status) and its RUNTIME HEALTH (healthy / degraded / failing / unverified,
   written by every real execution) side by side — ACTIVE is not HEALTHY.
   Opening one shows the typed contract, version, provider, dependencies,
   last verification, latest failures, receipts and repair history. Actions
   go through the normal router (Test / Repair / Improve / Disable); nothing
   here runs arbitrary code. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const TONE = { active: "ok", degraded: "warn", acquiring: "blue", repairing: "blue", disabled: "bad", unverified: "dim", deprecated: "dim" };
const HEALTH_TONE = { healthy: "ok", degraded: "warn", failing: "bad", unverified: "dim" };

export const view = {
  id: "capabilities",
  title: "Capabilities",
  async mount(pane, params) {
    const data = await api("/api/capabilities");
    const caps = data.capabilities || [];
    const search = el("input", { placeholder: "Filter capabilities…", value: params.q || "" });
    const grid = el("div", { class: "grid" });
    const render = () => {
      clear(grid);
      const q = search.value.toLowerCase();
      const shown = caps.filter((c) => !q || `${c.capability_id} ${c.description}`.toLowerCase().includes(q));
      if (!shown.length) grid.append(el("div", { class: "empty", text: caps.length ? "Nothing matches." : "No capabilities acquired yet. Ask for something ZEUS cannot do and it learns it." }));
      for (const c of shown) {
        const h = health(c);
        const card = el("div", { class: "card click" }, el("div", { class: "title", text: c.capability_id }),
          el("div", { class: "meta" }, badge(c.status || "?", TONE[c.status] || "dim"), badge(h.state, HEALTH_TONE[h.state] || "dim"), el("span", { text: `v${c.version || "?"}` }),
            h.consecutive_failures ? el("span", { text: `${h.consecutive_failures} failure(s) in a row` }) : null,
            el("span", { text: purpose(c) })));
        card.onclick = () => inspect(c);
        grid.append(card);
      }
    };
    search.oninput = render;
    const failing = caps.filter((c) => health(c).state === "failing").length;
    pane.append(el("div", { class: "toolbar" }, search, el("span", { class: "empty", style: { padding: 0 }, text: `${caps.length} registered${failing ? ` · ${failing} FAILING` : ""}` })), grid);
    render();
    if (params.id) { const c = caps.find((x) => x.capability_id === params.id); if (c) inspect(c); }
  },
};

function parse(v) {
  if (typeof v !== "string") return v;
  try { return JSON.parse(v.split(String.fromCharCode(39)).join(String.fromCharCode(34)).replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false").replace(/\bNone\b/g, "null")); } catch { return v; }
}

function health(c) {
  const h = parse(c.health) || {};
  return { state: h.state || "unverified", consecutive_failures: h.consecutive_failures || 0, calls: h.calls || 0, last_ok_at: h.last_ok_at || "", last_error_at: h.last_error_at || "",
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
  views.inspect(c.capability_id,
    section("Contract", kv("id", c.capability_id), kv("version", c.version), kv("purpose", purpose(c)),
      kv("inputs", schemaKeys(c.input_schema)), kv("outputs", schemaKeys(c.output_schema)),
      kv("preconditions", (c.permissions_required || []).length ? `permissions: ${c.permissions_required.join(", ")}` : "none declared"),
      kv("effects / side effects", meta.effects || meta.side_effects || "not declared"),
      kv("dependencies", (c.dependencies || []).join(", ") || "none"), kv("provider", c.provider || meta.provider || meta.author || "local build"),
      kv("verification strategy", checks.length ? checks.map((x) => x.name || x.check).filter(Boolean).join(", ") : "none recorded"),
      kv("entrypoint", c.entrypoint), kv("source", c.source_location, "mono")),
    section("State", kv("installation status", c.status), kv("runtime health", h.state.toUpperCase()), kv("calls", h.calls), kv("last used", h.last_used || meta.last_used || "never"),
      kv("last verified", validation.verified_at || meta.verified_at || (validation.verified ? "at acquisition" : "")), kv("last ok", h.last_ok_at || "—"),
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
