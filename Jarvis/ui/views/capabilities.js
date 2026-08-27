/* Capability Center: every acquired capability with status, version,
   verification, inputs/outputs, dependencies, repair history; "Improve" and
   "Repair" start missions through the normal router. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

const TONE = { active: "ok", degraded: "warn", acquiring: "blue", repairing: "blue", disabled: "bad", unverified: "dim", deprecated: "dim" };

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
        const card = el("div", { class: "card click" }, el("div", { class: "title", text: c.capability_id }),
          el("div", { class: "meta" }, badge(c.status || "?", TONE[c.status] || "dim"), el("span", { text: `v${c.version || "?"}` }),
            el("span", { text: (c.description || "").split("\n")[0].slice(0, 80) })));
        card.onclick = () => inspect(c);
        grid.append(card);
      }
    };
    search.oninput = render;
    pane.append(el("div", { class: "toolbar" }, search, el("span", { class: "empty", style: { padding: 0 }, text: `${caps.length} registered` })), grid);
    render();
    if (params.id) { const c = caps.find((x) => x.capability_id === params.id); if (c) inspect(c); }
  },
};

function parse(v) {
  if (typeof v !== "string") return v;
  try { return JSON.parse(v.split(String.fromCharCode(39)).join(String.fromCharCode(34)).replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false").replace(/\bNone\b/g, "null")); } catch { return v; }
}

function inspect(c) {
  const schema = (s) => { const p = parse(s); return p && p.properties ? Object.keys(p.properties).join(", ") : String(s || "").slice(0, 120); };
  const validation = parse(c.validation_status) || {};
  const checks = Array.isArray(validation.checks) ? validation.checks : [];
  const meta = parse(c.creation_metadata) || {};
  views.inspect(c.capability_id,
    section("Status", kv("status", c.status), kv("version", c.version), kv("verified", checks.length ? `${checks.filter((x) => x.ok !== false && x.passed !== false).length}/${checks.length} checks` : "no record"),
      kv("provider", c.provider || meta.provider || ""), kv("created", meta.created_at || ""), kv("last used", c.last_used || meta.last_used || "")),
    section("Contract", kv("inputs", schema(c.input_schema)), kv("outputs", schema(c.output_schema)), kv("dependencies", (c.dependencies || []).join(", ") || "none"),
      kv("permissions", (c.permissions_required || []).join(", ") || "none"), kv("entrypoint", c.entrypoint), kv("source", c.source_location, "mono")),
    section("Description", el("div", { class: "kv" }, el("span", { class: "v", text: c.description || "" }))),
    checks.length ? section("Verification strategy", ...checks.map((x) => el("div", { class: "kv" }, el("span", { class: "k", text: (x.ok ?? x.passed) === false ? "✗" : "✓" }), el("span", { class: "v", text: x.name || x.check || x.detail || JSON.stringify(x).slice(0, 120) })))) : null,
    el("div", { class: "toolbar" },
      button("Improve this capability", () => chat.send(`Zeus, verbessere deine Fähigkeit ${c.capability_id}.`), "primary"),
      button("Repair", () => chat.send(`Zeus, repariere deine Fähigkeit ${c.capability_id}, sie funktioniert nicht richtig.`)),
      button("Activity", () => views.open("activity", { q: c.capability_id }))),
  );
}
