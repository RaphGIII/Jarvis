/* Home and Chat: the same place in two moods.

   Home is the calm entry: the water, ZEUS's ribbon, one line of greeting, the
   composer and at most four actions that come from what the owner is actually
   doing -- the material last studied, the project last touched, today's
   calendar.  Nothing is shown that does not lead somewhere.  Chat is the
   conversation on the same ground; its history is one click away. */

import { $, el, clear } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";
import * as views from "./views.js";
import * as history from "./history.js";

let chatMod = null;

export function init({ chat }) {
  chatMod = chat;
  bus.on("mode:home", () => setMode("home"));
  bus.on("mode:chat", (p) => setMode("chat", p));
  bus.on("user_message", (p) => { if (!p._replay) setMode("chat"); });
  bus.on("view:close", () => bus.emit("mode:changed", {}));
  greet();
  setInterval(greet, 10 * 60 * 1000);
  refreshActions();
  bus.on("study:changed", refreshActions);
  setInterval(refreshActions, 5 * 60 * 1000);
}

export function setMode(mode, payload = {}) {
  const app = $("app");
  const chat = mode === "chat";
  app.classList.toggle("conversing", chat);
  app.dataset.mode = chat ? "chat" : "home";
  $("topTitle").textContent = chat ? (payload.title || "Chat") : "";
  const toggle = $("btnHistory");
  if (toggle) toggle.hidden = !chat;
  const empty = $("chatEmpty");
  if (empty) empty.hidden = !chat || $("log").querySelector(".turn") !== null;
  if (chat) renderChatEmpty();
  bus.emit("mode:changed", { mode });
}

export function greet() {
  const node = $("homeGreeting");
  if (!node) return;
  const hour = new Date().getHours();
  const part = hour < 5 ? "Gute Nacht" : hour < 11 ? "Guten Morgen" : hour < 18 ? "Guten Tag" : "Guten Abend";
  const who = window.OWNER_NAME ? `, ${window.OWNER_NAME}` : "";
  node.textContent = `${part}${who}.`;
}

function action(label, detail, run, icon) {
  return el("button", { class: "home-action", onClick: run, title: detail ? `${label} – ${detail}` : label },
    el("span", { class: "ha-label", text: label }), detail ? el("span", { class: "ha-detail", text: detail }) : null, icon ? el("span", { class: "ha-arrow", "aria-hidden": "true", text: "→" }) : null);
}

export async function refreshActions() {
  const box = $("homeActions");
  if (!box) return;
  const [docs, overview, calendar] = await Promise.all([
    api("/api/study/documents", { limit: 1, order: "recent" }),
    api("/api/projects/overview"),
    api("/api/calendar/list", { start: startOfDay().toISOString(), end: endOfDay().toISOString() }),
  ]);
  const actions = [];
  const last = (docs && docs.documents && docs.documents[0]) || null;
  if (last) {
    const remembered = readLastStudy();
    const unit = remembered && remembered.doc === last.id ? remembered.unit : 1;
    actions.push(action("Weiter lernen", last.title || last.filename, () => views.open("study", { doc: last.id, unit: String(unit) }), true));
  } else {
    actions.push(action("Studium", "Unterlagen hinzufügen", () => views.open("study", {}), true));
  }
  const projects = ((overview && overview.projects) || []).filter((p) => (p.origin || "owner") === "owner");
  const recent = projects.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")))[0];
  if (recent) actions.push(action("Projekt fortsetzen", recent.title || recent.id, () => views.open("projects", { focus: recent.title || recent.id }), true));
  actions.push(action("Dateien durchsuchen", "", () => { setMode("chat"); chatMod?.focusComposer("Suche in meinen Dateien nach "); }, false));
  const events = (calendar && (calendar.events || calendar.items)) || [];
  actions.push(events.length ? action("Heute", events.length === 1 ? "1 Termin" : `${events.length} Termine`, () => views.open("calendar", {}), true)
                             : action("Tag planen", "Kalender", () => views.open("calendar", {}), true));
  clear(box);
  box.append(...actions.slice(0, 4));
}

function startOfDay() { const d = new Date(); d.setHours(0, 0, 0, 0); return d; }
function endOfDay() { const d = new Date(); d.setHours(23, 59, 59, 999); return d; }

export function readLastStudy() {
  try { return JSON.parse(localStorage.getItem("zeus.study.last") || "null"); } catch { return null; }
}

/* The chat without a conversation yet: the last few conversations, to pick one up again. */
function renderChatEmpty() {
  const box = $("chatEmpty");
  if (!box || box.hidden) return;
  const items = history.recent(4);
  clear(box);
  box.append(el("div", { class: "ce-title", text: "Neues Gespräch" }),
    el("div", { class: "ce-sub", text: "Frag etwas oder sag ZEUS, was zu tun ist." }));
  if (items.length) {
    const list = el("div", { class: "ce-list" }, el("div", { class: "ce-label", text: "Zuletzt" }));
    for (const c of items) list.append(el("button", { class: "ce-item", onClick: () => history.openChat(c.id) }, el("span", { text: c.title || "Chat" })));
    box.append(list);
  }
}
