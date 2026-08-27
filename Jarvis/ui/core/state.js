/* One observable holding what the server last said: the state event, the
   status poll, the GPU reading, the health readiness, the active mission.
   Everything that renders truth reads it from here, so the eye, the pills,
   the HUD and a view opened later all show the same thing. Nothing here is
   ever guessed on the client: every field is a copy of a server payload. */

import * as bus from "./bus.js";

export const state = {
  eye: { state: "offline", detail: "", since: "", progress: null, busy: false },
  status: {},          // /api/status
  gpu: null,           // /api/gpu
  health: {},          // /api/health
  mission: null,       // the most recent active selfdev/capability mission (from events)
  connection: "connecting",
  ui: load(),          // persisted per-viewer preferences
};

function load() {
  const defaults = { workspace: "", split: null, pinned: [], reducedMotion: false, echoActions: false, lastView: "" };
  try {
    const raw = localStorage.getItem("zeus.ui");
    return raw ? { ...defaults, ...JSON.parse(raw) } : defaults;
  } catch {
    return defaults;
  }
}

export function persist() {
  try { localStorage.setItem("zeus.ui", JSON.stringify(state.ui)); } catch { /* private window, or blocked */ }
}

export function set(key, value) {
  state[key] = value;
  bus.emit(`state:${key}`, value);
}

export function setPref(key, value) {
  state.ui[key] = value;
  persist();
  bus.emit("state:ui", state.ui);
}

/* Blue / green / red: the owner's colour law. Every state maps to exactly one
   category; motion may differ wildly, the category never does. */
export const CATEGORY = {
  idle: "blue", listening: "blue", transcribing: "blue", thinking: "blue", speaking: "blue", waiting: "blue",
  working: "green", verifying: "green", coding: "green", researching: "green", expert: "green",
  error: "red", offline: "red",
};

export function category(name) {
  return CATEGORY[name] || "blue";
}
