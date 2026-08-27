/* The view registry: Presence (the eye and the conversation), Workspace (a
   named view in the central pane with an optional inspector) and Mission
   Control are the three modes; a view is {id, title, mount, unmount}. Opening
   one writes the hash, so the browser's back button and a reload land where
   the owner was. The inspector is a shared right pane any view can fill. */

import { $, el, clear } from "./dom.js";
import * as bus from "./bus.js";
import { state, setPref } from "./state.js";

const registry = new Map();
let current = null;         // {id, params, view}
let inspectorOpen = false;

export function register(view) {
  registry.set(view.id, view);
  return view;
}

export function all() {
  return [...registry.values()];
}

export function get(id) {
  return registry.get(id) || null;
}

export function currentView() {
  return current;
}

export function isWorkspace() {
  return current !== null;
}

/* Open a view in the workspace. `params` is anything the view understands
   (a project id, a search query). Presence mode is `close()`. */
export async function open(id, params = {}, { push = true } = {}) {
  const view = registry.get(id);
  if (!view) return false;
  const app = $("app");
  const pane = $("workspacePane");
  if (current && current.view.unmount) {
    try { current.view.unmount(); } catch (err) { console.error(err); }
  }
  clear(pane);
  current = { id, params, view };
  app.classList.add("workspace");
  app.dataset.view = id;
  $("workspaceTitle").textContent = view.title;
  $("breadcrumb").textContent = "";
  for (const b of document.querySelectorAll("[data-view-button]")) {
    b.setAttribute("aria-pressed", b.dataset.viewButton === id ? "true" : "false");
  }
  if (push) {
    const hash = "#" + id + (Object.keys(params).length ? "?" + new URLSearchParams(params).toString() : "");
    if (location.hash !== hash) history.pushState({ view: id, params }, "", hash);
  }
  setPref("lastView", id);
  closeInspector();
  try {
    await view.mount(pane, params);
  } catch (err) {
    console.error(`[views] ${id} failed to mount`, err);
    pane.append(el("div", { class: "empty", text: `${view.title} could not be opened: ${err.message || err}` }));
  }
  bus.emit("view:open", { id, params });
  return true;
}

export function close({ push = true } = {}) {
  if (current && current.view.unmount) {
    try { current.view.unmount(); } catch (err) { console.error(err); }
  }
  current = null;
  const app = $("app");
  app.classList.remove("workspace");
  delete app.dataset.view;
  clear($("workspacePane"));
  closeInspector();
  for (const b of document.querySelectorAll("[data-view-button]")) b.setAttribute("aria-pressed", "false");
  if (push && location.hash) history.pushState({}, "", location.pathname + location.search);
  setPref("lastView", "");
  bus.emit("view:close", {});
}

export function breadcrumb(parts) {
  const node = $("breadcrumb");
  clear(node);
  parts.forEach((part, i) => {
    if (i) node.append(el("span", { class: "sep", text: "›" }));
    if (part.onClick) node.append(el("a", { href: "#", text: part.label, onClick: (e) => { e.preventDefault(); part.onClick(); } }));
    else node.append(el("span", { text: part.label }));
  });
}

/* The inspector: the right pane for whatever is selected. */
export function inspect(title, ...children) {
  const pane = $("inspector");
  clear($("inspectorBody"));
  $("inspectorTitle").textContent = title;
  $("inspectorBody").append(...children.flat().filter(Boolean));
  pane.classList.add("open");
  $("app").classList.add("inspecting");
  inspectorOpen = true;
}

export function closeInspector() {
  if (!inspectorOpen) return;
  $("inspector").classList.remove("open");
  $("app").classList.remove("inspecting");
  inspectorOpen = false;
}

export function inspectorIsOpen() {
  return inspectorOpen;
}

/* Restore from the URL hash on load / back button. */
export function fromHash() {
  const hash = location.hash.replace(/^#/, "");
  if (!hash) return null;
  const [id, query] = hash.split("?");
  const params = Object.fromEntries(new URLSearchParams(query || ""));
  return registry.has(id) ? { id, params } : null;
}

window.addEventListener("popstate", () => {
  const target = fromHash();
  if (target) open(target.id, target.params, { push: false });
  else if (current) close({ push: false });
});
