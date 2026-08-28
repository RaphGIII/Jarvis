/* Thoughts: what ZEUS noticed on its own — typed observations with the
   evidence they rest on, the context they concern, confidence and
   importance. Nothing here is an event ZEUS claims happened; every thought
   lists the records it was derived from. Owner actions: save to Knowledge,
   attach to project, create mission, dismiss, mute type, tell me more. */

import { el, clear, kv, section, badge, button, ago } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";

const FILTERS = [["", "All"], ["NEW", "New"], ["IMPORTANT", "Important"], ["SAVED", "Saved"], ["DISMISSED", "Dismissed"], ["ACTED_ON", "Acted on"]];
const TONE = { URGENT: "bad", HIGH: "warn", MEDIUM: "blue", LOW: "dim" };
const TYPE_TONE = { WARNING: "bad", PROJECT_RISK: "warn", INSIGHT: "blue", CONNECTION: "blue", OPTIMIZATION: "ok", REMINDER: "dim", OPPORTUNITY: "ok", FOLLOW_UP: "dim", QUESTION: "dim", IDEA: "blue" };

export const view = {
  id: "thoughts",
  title: "Thoughts",
  async mount(pane, params) {
    let filter = params.status || "";
    const tabs = el("div", { class: "toolbar" });
    const list = el("div");
    const status = el("span", { class: "empty", style: { padding: 0 } });
    const load = async () => {
      const data = await api("/api/thoughts", { status: filter });
      clear(list);
      const rows = data.thoughts || [];
      const c = data.counts || {};
      status.textContent = `${c.NEW || 0} new · ${c.IMPORTANT || 0} important · ${c.SAVED || 0} saved · ${c.ACTED_ON || 0} acted on · ${c.DISMISSED || 0} dismissed` + ((data.muted_types || []).length ? ` · muted: ${data.muted_types.join(", ")}` : "");
      if (!rows.length) { list.append(el("div", { class: "empty", text: "Nothing noticed yet. ZEUS looks at missions, corrections, capability health and project activity after each finished mission and every 30 minutes." })); return; }
      for (const t of rows) list.append(card(t, load));
    };
    for (const [key, label] of FILTERS) {
      tabs.append(el("button", { class: "ghost", "aria-pressed": filter === key ? "true" : "false", text: label, onClick: () => { filter = key; for (const b of tabs.querySelectorAll("button")) b.setAttribute("aria-pressed", b.textContent === label ? "true" : "false"); load(); } }));
    }
    tabs.append(button("Think now", async () => { await api("/api/thoughts/think", { trigger: "manual" }); load(); }), status);
    pane.append(tabs, list);
    await load();
  },
};

function card(t, reload) {
  const node = el("div", { class: "card" + (t.status === "DISMISSED" ? " dim" : "") },
    el("div", { class: "title" }, badge(t.type, TYPE_TONE[t.type] || "dim"), " ", badge(t.importance, TONE[t.importance] || "dim"), " ", t.title,
      t.count > 1 ? el("span", { class: "empty", style: { padding: "0 8px" }, text: `×${t.count}` }) : null),
    el("div", { class: "meta" }, el("span", { text: t.status }), el("span", { text: `confidence ${Math.round((t.confidence || 0) * 100)}%` }),
      el("span", { text: ago(t.updated_at || t.generated_at) }), t.delivered_how ? el("span", { text: `told: ${t.delivered_how}` }) : null),
    el("div", { class: "kv" }, el("span", { class: "v", text: t.text })),
    el("div", { class: "kv" }, el("span", { class: "k", text: "why" }), el("span", { class: "v", text: t.why_it_matters })),
    t.suggested_action ? el("div", { class: "kv" }, el("span", { class: "k", text: "suggested" }), el("span", { class: "v", text: t.suggested_action })) : null,
    el("details", {}, el("summary", { text: `evidence (${(t.evidence || []).length})` }),
      ...(t.evidence || []).map((e) => el("div", { class: "kv" }, el("span", { class: "k", text: e.kind }), el("span", { class: "v mono", text: `${e.ref} — ${e.summary}` })))),
    el("div", { class: "toolbar" },
      button("Save to Knowledge", () => act(t, "save_knowledge", reload)),
      (t.context && (t.context.project_id || (t.context.project_ids || []).length)) ? button("Attach to project", () => act(t, "attach_project", reload)) : null,
      button("Create mission", () => act(t, "create_mission", reload), "primary"),
      button("Tell me more", () => act(t, "tell_me_more", reload)),
      t.status !== "DISMISSED" ? button("Dismiss", () => act(t, "dismiss", reload), "ghost danger") : null,
      button("Mute this type", () => act(t, "mute_type", reload), "ghost")));
  return node;
}

async function act(t, action, reload) {
  const r = await api("/api/thoughts/act", { id: t.thought_id, action });
  if (r.ok === false) alert(r.error || "failed");
  if (action === "create_mission" && r.mission_id) views.open("missions", { mission: r.mission_id });
  reload();
}
