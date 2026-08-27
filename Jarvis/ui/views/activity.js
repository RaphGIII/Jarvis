/* Activity: the owner's transparent operation log. Human-readable rows from
   the durable log; every row expands to the recorded facts (executor,
   target, receipt, verifier, evidence). Nothing is recomputed in the browser. */

import { $, el, clear, clockOf, dateOf, kv, section, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import { state, setPref } from "../core/state.js";
import * as views from "../core/views.js";

const KINDS = {
  "request": { label: "REQUEST", cls: "req", icon: "›" },
  "answer": { label: "ANSWER", cls: "ans", icon: "‹" },
  "action.verified": { label: "VERIFIED", cls: "ok", icon: "✓" },
  "action.ran": { label: "UNVERIFIED", cls: "warn", icon: "○" },
  "action.failed": { label: "FAILED", cls: "bad", icon: "✗" },
  "tool": { label: "TOOL", cls: "tool", icon: "⌁" },
  "state.working": { label: "WORKING", cls: "work", icon: "⚙" },
  "state.verifying": { label: "VERIFYING", cls: "work", icon: "⚙" },
  "state.researching": { label: "RESEARCH", cls: "tool", icon: "⌕" },
  "state.coding": { label: "CODING", cls: "work", icon: "⌁" },
  "state.waiting": { label: "WAITING", cls: "warn", icon: "…" },
  "state.error": { label: "ERROR", cls: "bad", icon: "✗" },
  "error": { label: "ERROR", cls: "bad", icon: "✗" },
  "progress": { label: "PROGRESS", cls: "tool", icon: "⚙" },
  "notification": { label: "NOTE", cls: "tool", icon: "•" },
};

let live = null;

export const view = {
  id: "activity",
  title: "Activity",
  async mount(pane, params) {
    const toolbar = el("div", { class: "toolbar" });
    const search = el("input", { placeholder: "Filter…", value: params.q || "" });
    const kind = el("select", {}, el("option", { value: "", text: "all kinds" }),
      ...Object.entries(KINDS).map(([k, m]) => el("option", { value: k, text: m.label.toLowerCase() })));
    const echo = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } },
      el("input", { type: "checkbox", checked: state.ui.echoActions, onChange: (e) => setPref("echoActions", e.target.checked) }),
      " echo actions into the conversation");
    toolbar.append(search, kind, echo);
    const list = el("div", { class: "activity" });
    pane.append(toolbar, list);

    const data = await api("/api/activity", { limit: 400 });
    let entries = (data.activity || []).slice().reverse();
    const render = () => {
      clear(list);
      const q = search.value.toLowerCase();
      const shown = entries.filter((e) => (!kind.value || e.kind === kind.value) &&
        (!q || `${e.summary} ${e.kind} ${e.receipt_id}`.toLowerCase().includes(q)));
      if (data.error) { list.append(el("div", { class: "empty", text: `The activity log could not be read: ${data.error}` })); return; }
      if (!shown.length) { list.append(el("div", { class: "empty", text: "Nothing recorded yet. Every request, action and verification appears here." })); return; }
      let day = "";
      for (const entry of shown.slice(0, 400)) {
        const d = dateOf(entry.at);
        if (d !== day) { day = d; list.append(el("h4", { class: "day", text: d, style: { margin: "14px 0 6px", fontSize: "10px", letterSpacing: ".2em", color: "var(--faint)" } })); }
        list.append(row(entry, params.receipt && entry.receipt_id === params.receipt));
      }
    };
    search.oninput = render;
    kind.onchange = render;
    render();
    live = bus.on("*", () => {});
    view._refresh = async () => {
      const fresh = await api("/api/activity", { limit: 400 });
      entries = (fresh.activity || []).slice().reverse();
      render();
    };
    view._sub = bus.on("tool", () => view._refresh());
  },
  unmount() {
    view._sub?.();
    live?.();
  },
};

function row(entry, highlight) {
  const meta = KINDS[entry.kind] || { label: entry.kind.toUpperCase(), cls: "tool", icon: "•" };
  const node = el("div", { class: `act ${meta.cls}` });
  const head = el("div", { class: "act-head" },
    el("span", { class: "act-tag", text: `${meta.icon} ${meta.label}` }),
    el("span", { class: "act-sum", text: entry.summary || "(no detail recorded)" }),
    el("span", { class: "act-when", text: clockOf(entry.at) }));
  node.append(head);
  const detail = technical(entry);
  if (detail) {
    head.classList.add("expandable");
    detail.hidden = !highlight;
    head.onclick = () => { detail.hidden = !detail.hidden; inspect(entry); };
    node.append(detail);
  }
  if (highlight) node.style.outline = "1px solid var(--blue-deep)";
  return node;
}

function technical(entry) {
  const receipt = entry.receipt_id ? entry.detail : null;
  const d = entry.detail || {};
  if (!receipt && !Object.keys(d).length) return null;
  const body = el("div", { class: "act-body" });
  const line = (k, v) => { if (v !== undefined && v !== null && v !== "") body.append(el("div", { class: "act-kv" }, el("span", { class: "act-k", text: k }), el("span", { class: "act-v", text: String(v) }))); };
  line("event", entry.kind);
  line("timestamp", entry.at);
  line("scope", entry.scope);
  if (receipt) {
    const ev = receipt.evidence || {};
    line("action", receipt.kind);
    line("executor", receipt.executor);
    line("target", ev.path || ev.project_id || ev.title || ev.track || ev.query || "");
    line("result", receipt.verified ? "verified" : receipt.ok ? "ran, not verified" : "failed");
    line("duration", receipt.duration_seconds !== undefined ? `${receipt.duration_seconds}s` : "");
    if (!receipt.ok) line("failure", receipt.detail);
    line("receipt", receipt.id);
    const checks = receipt.verifications || [];
    if (checks.length) {
      body.append(el("div", { class: "act-k", text: "verification evidence", style: { marginTop: "6px" } }));
      for (const c of checks) body.append(el("div", { class: "act-check" + (c.passed ? "" : " bad"), text: `${c.passed ? "✓" : "✗"} ${c.check}` + (c.observed ? ` — ${c.observed}` : "") }));
    } else {
      body.append(el("div", { class: "act-check bad", text: "✗ no verification was recorded for this action" }));
    }
    return body;
  }
  if (d.mission_id) line("mission", d.mission_id);
  if (d.phase) line("phase", d.phase);
  if (d.routing) {
    line("route", `${d.routing.top_level} (${d.routing.confidence})`);
    line("reason", d.routing.reason);
    if ((d.routing.conflicts || []).length) line("overruled", d.routing.conflicts.join(" | "));
    if ((d.routing.corrections || []).length) line("corrections", d.routing.corrections.join(", "));
  }
  for (const [k, v] of Object.entries(d)) {
    if (["summary", "routing", "mission_id", "phase", "receipt", "text"].includes(k)) continue;
    if (v === null || v === "" || typeof v === "object") continue;
    line(k, v);
  }
  return body.children.length ? body : null;
}

function inspect(entry) {
  const d = entry.detail || {};
  const children = [section("Event", kv("kind", entry.kind), kv("at", entry.at), kv("seq", entry.seq), kv("scope", entry.scope), kv("receipt", entry.receipt_id))];
  if (d.routing) {
    children.push(section("Routing", kv("top level", `${d.routing.top_level} · ${d.routing.confidence}`), kv("reason", d.routing.reason),
      kv("operation", d.routing.reading?.operation), kv("object", d.routing.reading?.object),
      kv("self / world", `${d.routing.reading?.self_score} / ${d.routing.reading?.world_score}`),
      kv("overruled", (d.routing.conflicts || []).join("\n")), kv("corrections", (d.routing.corrections || []).join(", "))));
  }
  if (entry.receipt_id && d.evidence) children.push(section("Evidence", kv("json", JSON.stringify(d.evidence, null, 1), "mono")));
  if (d.mission_id) children.push(el("div", { class: "toolbar" }, button("Open mission", () => views.open("missions", { mission: d.mission_id }))));
  views.inspect(entry.summary ? entry.summary.slice(0, 60) : entry.kind, ...children);
}
