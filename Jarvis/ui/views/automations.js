/* Automationen: what ZEUS does without being asked.

   Timers that are running, the observations ZEUS made on its own (with the
   evidence behind them), and the opt-in habit observer that suggests routines.
   Each control acts on the store behind it; nothing here is decoration. */

import { el, clear } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";

export const view = {
  id: "automations",
  title: "Automationen",
  async mount(pane) {
    const root = el("div", { class: "memory" });
    pane.append(root);
    await render(root);
  },
};

async function render(root) {
  const [timers, thoughts, observer, patterns] = await Promise.all([
    api("/api/timers"), api("/api/thoughts", { status: "" }), api("/api/observer/status"), api("/api/observer/patterns", { since_hours: 168 }),
  ]);
  clear(root);
  root.append(el("header", { class: "mm-head" }, el("h1", { class: "mm-title", text: "Automationen" }),
    el("p", { class: "mm-lead", text: "Was ZEUS von selbst tut: Timer, eigene Beobachtungen und – wenn du es erlaubst – Vorschläge aus deinen Gewohnheiten." })));

  const running = (timers && timers.timers) || [];
  root.append(section("Timer", running.map((t) => row(t.label || `${t.minutes} Minuten`, `noch ${Math.max(1, Math.round(t.remaining_seconds / 60))} min`)),
    "Kein Timer läuft. Sag: „Stell einen Timer auf 25 Minuten.“"));

  const open = ((thoughts && thoughts.thoughts) || []).filter((t) => ["NEW", "IMPORTANT"].includes(t.status)).slice(0, 6);
  root.append(section("Von selbst bemerkt", open.map((t) => {
    const r = row(t.title, t.suggested_action || "");
    r.append(el("button", { class: "mm-link", text: "Erledigt", "aria-label": `„${t.title}“ verwerfen`, onClick: async () => {
      await api("/api/thoughts/act", { id: t.id || t.thought_id, action: "dismiss" }); render(root);
    } }));
    return r;
  }), "Nichts Offenes.", el("button", { class: "mm-link", text: "Alle Beobachtungen", onClick: () => views.open("thoughts", {}) })));

  const enabled = Boolean(observer && observer.enabled);
  const toggle = el("button", { class: "switch" + (enabled ? " on" : ""), role: "switch", "aria-checked": String(enabled), "aria-label": "Gewohnheiten erkennen",
    onClick: async () => { await api("/api/observer/enable", { enabled: !enabled }); render(root); } });
  const suggestions = ((patterns && patterns.suggestions) || []).map((s) => row(typeof s === "string" ? s : (s.text || s.title || JSON.stringify(s))));
  root.append(section("Gewohnheiten erkennen",
    enabled ? suggestions : [], enabled ? "Noch zu wenig beobachtet, um etwas vorzuschlagen." : `Aus. ${(observer && observer.records) || ""}`, toggle));
}

function row(text, meta = "") {
  return el("div", { class: "mm-row", role: "listitem" }, el("span", { class: "mm-text", text }), meta ? el("span", { class: "mm-meta", text: meta }) : null);
}

function section(title, rows, empty, action = null) {
  const list = el("div", { class: "mm-list", role: "list" });
  if (rows.length) list.append(...rows);
  else list.append(el("div", { class: "mm-empty", text: empty }));
  return el("section", { class: "mm-section" }, el("div", { class: "mm-section-head" }, el("h2", { text: title }), action), list);
}
