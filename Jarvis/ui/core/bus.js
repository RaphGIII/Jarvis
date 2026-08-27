/* The client-side event bus. Server events (over SSE) and UI events share it,
   so a view subscribes to what it cares about and nothing else has to know
   the view exists. Handlers never throw past the bus: one broken view must
   not take the eye down with it. */

const handlers = new Map();

export function on(type, fn) {
  if (!handlers.has(type)) handlers.set(type, new Set());
  handlers.get(type).add(fn);
  return () => off(type, fn);
}

export function off(type, fn) {
  handlers.get(type)?.delete(fn);
}

export function emit(type, payload = {}) {
  for (const fn of handlers.get(type) || []) {
    try { fn(payload, type); } catch (err) { console.error(`[bus] ${type} handler failed`, err); }
  }
  for (const fn of handlers.get("*") || []) {
    try { fn(payload, type); } catch (err) { console.error("[bus] wildcard handler failed", err); }
  }
}

export function types() {
  return [...handlers.keys()].filter((t) => t !== "*");
}
