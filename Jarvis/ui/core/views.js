/* The view registry: Presence (the eye and the conversation), Workspace (a
   named view in the central pane with an optional inspector) and Mission
   Control are the three modes; a view is {id, title, mount, unmount}. Opening
   one writes the hash, so the browser's back button and a reload land where
   the owner was. The inspector is a shared right pane any view can fill.

   Keep-alive: a view that declares suspend()/resume(params) is PARKED when
   the owner leaves it — its DOM (one .view-root wrapper, display:contents so
   layout is identical) is detached, not destroyed, and its loops paused.
   Coming back reattaches the same scene: camera, layout, canvas — no refetch,
   no relayout. resume(params) returns false to demand a fresh mount (deep
   links, changed params, stale data); views without the hooks keep the old
   unmount-and-rebuild lifecycle. */

import { $, el, clear } from "./dom.js";
import * as bus from "./bus.js";
import { state, setPref } from "./state.js";

const registry = new Map();
// Spatial views are an environment, not a page: no rail, no pane header,
// the surface takes everything below the thin top bar.
const SPATIAL = new Set(["projects", "files", "knowledge"]);
let current = null;         // {id, params, view, root}
let inspectorOpen = false;
let openSeq = 0;            // stale async mounts must not touch the pane
const parked = new Map();   // id -> {root, params} of suspended keep-alive views

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

/* Leaving a view: park it (keep-alive) or tear it down. Either way the pane
   ends up empty and `current` is null. */
function parkCurrent() {
  if (!current) return;
  const leaving = current;
  current = null;
  if (leaving.view.suspend && leaving.root && leaving.ready) {
    try { leaving.view.suspend(); } catch (err) { console.error(err); }
    parked.set(leaving.id, { root: leaving.root, params: leaving.params });
    leaving.root.remove();
  } else {
    if (leaving.view.unmount) {
      try { leaving.view.unmount(); } catch (err) { console.error(err); }
    }
    if (leaving.root) leaving.root.remove();
  }
  clear($("workspacePane"));
}

/* A lightweight shell so heavy views never show a black void: the pane gets
   an immediate, animated state while data loads asynchronously. */
function loadingShell(id) {
  return el("div", { class: "view-loading" },
    el("div", { class: "view-loading-orb" }),
    el("span", { text: SPATIAL.has(id) ? "Synchronisiere Universum…" : "Synchronisiere…" }));
}

/* Open a view in the workspace. `params` is anything the view understands
   (a project id, a search query). Presence mode is `close()`.
   `force` bypasses a parked scene and rebuilds from fresh data. */
export async function open(id, params = {}, { push = true, force = false } = {}) {
  const view = registry.get(id);
  if (!view) return false;
  const seq = ++openSeq;
  const app = $("app");
  const pane = $("workspacePane");
  parkCurrent();
  app.classList.add("workspace");
  app.classList.toggle("spatial", SPATIAL.has(id));
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

  // a parked scene comes back exactly as it was — same DOM, same camera
  const cached = parked.get(id);
  parked.delete(id);
  if (cached && !force && view.resume) {
    pane.append(cached.root);
    let resumed = false;
    try { resumed = Boolean(view.resume(params)); } catch (err) { console.error(err); }
    if (resumed) {
      current = { id, params, view, root: cached.root, ready: true };
      bus.emit("view:open", { id, params });
      return true;
    }
    cached.root.remove();
    if (view.unmount) { try { view.unmount(); } catch (err) { console.error(err); } }
  } else if (cached) {
    // parked but bypassed (force, or the view lost its resume hook): tear down
    if (view.unmount) { try { view.unmount(); } catch (err) { console.error(err); } }
  }

  // fresh mount into a display:contents wrapper: layout-identical to mounting
  // into the pane, but detachable as ONE node and immune to stale async
  // mounts appending into someone else's view
  const root = el("div", { class: "view-root" });
  const shell = loadingShell(view.id);
  root.append(shell);
  pane.append(root);
  const cur = { id, params, view, root, ready: false };
  current = cur;
  try {
    await view.mount(root, params);
  } catch (err) {
    console.error(`[views] ${id} failed to mount`, err);
    root.append(el("div", { class: "empty", text: `${view.title} could not be opened: ${err.message || err}` }));
  }
  shell.remove();
  if (seq !== openSeq) {
    // superseded while loading: discard what this stale mount built, and
    // release its module state unless a same-view open now owns it
    root.remove();
    if (current?.view !== view && view.unmount) {
      try { view.unmount(); } catch (err) { console.error(err); }
    }
    return false;
  }
  cur.ready = true;
  bus.emit("view:open", { id, params });
  return true;
}

export function close({ push = true } = {}) {
  openSeq += 1; // any in-flight mount is now stale
  parkCurrent();
  const app = $("app");
  app.classList.remove("workspace");
  app.classList.remove("spatial");
  delete app.dataset.view;
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
