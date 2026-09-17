/* The navigation: one thin column of destinations, and ZEUS's status at the foot.

   Home is the calm entry, Chat the conversation (its archive lives in Chat's
   own history panel), every other entry opens the view that owns that part
   of ZEUS.  No descriptions under the items, no counters that do not ask for
   anything.  The active destination is marked by a quiet surface. */

import { $, el, clear } from "./dom.js";
import * as bus from "./bus.js";
import * as views from "./views.js";
import * as history from "./history.js";

/* thin monochrome icons, one stroke weight, drawn inline */
const ICON = {
  home: '<path d="M4.5 11.2 12 5l7.5 6.2V19a1 1 0 0 1-1 1h-4.2v-5h-4.6v5H5.5a1 1 0 0 1-1-1z"/>',
  chat: '<path d="M5 6.5h14M5 11h10M5 15.5h7"/>',
  study: '<path d="M4.5 5.5h6.2c.8 0 1.3.5 1.3 1.3V19c0-.8-.6-1.3-1.3-1.3H4.5z"/><path d="M19.5 5.5h-6.2c-.8 0-1.3.5-1.3 1.3V19c0-.8.6-1.3 1.3-1.3h6.2z"/>',
  missions: '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M21.5 12h-3"/>',
  knowledge: '<circle cx="7" cy="8" r="2"/><circle cx="17" cy="7" r="2"/><circle cx="12" cy="17" r="2"/><path d="M8.7 9.2 11 15.4M15.6 8.5 12.8 15.2M9 7.8l6-.6"/>',
  memory: '<path d="M12 4.5c-3.2 0-5.5 2.2-5.5 5 0 1.9 1 3.2 2.2 4.1.6.5.8 1 .8 1.7v1.2h5v-1.2c0-.7.2-1.2.8-1.7 1.2-.9 2.2-2.2 2.2-4.1 0-2.8-2.3-5-5.5-5z"/><path d="M10 19.5h4"/>',
  files: '<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4"/>',
  calendar: '<rect x="4" y="5.5" width="16" height="14.5" rx="2"/><path d="M4 10h16M8.5 3.5v4M15.5 3.5v4"/>',
  agents: '<circle cx="9" cy="9" r="3"/><circle cx="16.5" cy="10.5" r="2.3"/><path d="M3.8 19c.8-3 2.8-4.5 5.2-4.5s4.4 1.5 5.2 4.5M14.5 15.2c.6-.3 1.3-.5 2-.5 2 0 3.3 1.2 3.9 3.3"/>',
  voice: '<path d="M4 12h1.5M7.5 8.5v7M11 5.5v13M14.5 8v8M18 10.5v3M20 12h.5"/>',
  automations: '<path d="M5.5 8h9.5l-2.5-2.5M18.5 16H9l2.5 2.5"/>',
  settings: '<circle cx="12" cy="12" r="2.8"/><path d="M12 3.5v2.3M12 18.2v2.3M3.5 12h2.3M18.2 12h2.3M6 6l1.6 1.6M16.4 16.4 18 18M6 18l1.6-1.6M16.4 7.6 18 6"/>',
};
export const icon = (name) => el("span", { class: "i", "aria-hidden": "true", html: `<svg viewBox="0 0 24 24">${ICON[name] || ""}</svg>` });

/* [destination (a view id, or the modes "home" / "chat"), label, icon] */
export const NAV = [
  ["home", "Home", "home"],
  ["chat", "Chat", "chat"],
  ["study", "Studium", "study"],
  ["missions", "Missionen", "missions"],
  ["knowledge", "Wissen", "knowledge"],
  ["memory", "Gedächtnis", "memory"],
  ["files", "Dateien", "files"],
  ["calendar", "Kalender", "calendar"],
  ["capabilities", "Agenten", "agents"],
  ["voice", "Stimme", "voice"],
  ["automations", "Automationen", "automations"],
  ["settings", "Einstellungen", "settings"],
];
const STATE_WORDS = {
  idle: "Online", listening: "Hört zu", transcribing: "Versteht", thinking: "Denkt", speaking: "Spricht", waiting: "Wartet auf dich",
  working: "Arbeitet", verifying: "Prüft", coding: "Entwickelt", researching: "Sucht", error: "Unterbrochen", offline: "Offline",
};

let lastState = "offline";

export function init({ chat, toast }) {
  history.init({ chat, toast });
  build();
  bus.on("state", (p) => renderFoot(p && p.state));
  bus.on("view:open", renderNav);
  bus.on("view:close", renderNav);
  bus.on("mode:changed", renderNav);
}

export function refresh() {
  return history.refresh();
}

export function newChat() {
  return history.newChat();
}

function build() {
  const side = $("sidebar");
  clear(side);
  side.append(
    el("button", { class: "sb-head", "aria-label": "ZEUS – Home", onClick: () => go("home") },
      el("span", { class: "sb-mark", "aria-hidden": "true" }), el("span", { class: "sb-brand", text: window.PRODUCT_NAME || "ZEUS" })),
    el("nav", { class: "sb-nav", id: "sbNav", "aria-label": "Bereiche" }),
    el("div", { class: "sb-foot", id: "sbFoot", role: "status", "aria-live": "polite" }),
  );
  renderNav();
  renderFoot();
}

export function currentDestination() {
  const view = views.currentView();
  if (view) return view.id;
  return $("app").classList.contains("conversing") ? "chat" : "home";
}

export function go(destination) {
  if (destination === "home") {
    views.close();
    history.close();
    bus.emit("mode:home", {});
    return;
  }
  if (destination === "chat") {
    views.close();
    bus.emit("mode:chat", {});
    return;
  }
  views.open(destination, {});
}

function renderNav() {
  const nav = $("sbNav");
  if (!nav) return;
  clear(nav);
  const here = currentDestination();
  for (const [id, label, ico] of NAV) {
    const on = here === id || (id === "settings" && here === "owner");
    nav.append(el("button", { class: "sb-item" + (on ? " on" : ""), "aria-current": on ? "page" : "false", title: label, onClick: () => go(id) },
      icon(ico), el("span", { class: "lbl", text: label })));
  }
}

function renderFoot(stateName) {
  const foot = $("sbFoot");
  if (!foot) return;
  if (stateName) lastState = stateName;
  clear(foot);
  const tone = lastState === "error" ? "bad" : lastState === "offline" ? "off" : lastState === "waiting" ? "warn" : "ok";
  foot.append(
    el("span", { class: "sb-foot-name", text: window.PRODUCT_NAME || "ZEUS" }),
    el("span", { class: "sb-foot-state" }, el("span", { class: "dot " + tone, "aria-hidden": "true" }), el("span", { text: STATE_WORDS[lastState] || STATE_WORDS.idle })));
}
