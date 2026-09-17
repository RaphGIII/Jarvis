/* ZEUS presence: server states map onto the ribbon's states, receipts pulse, reduced motion holds still.
   Run: node ui/tests/presence.test.mjs  (the Python test suite runs it too). */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { presenceFor, pulseForReceipt, PRESENCE_STATES, attach } from "../core/presence.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}

const SERVER_STATES = ["idle", "listening", "transcribing", "thinking", "speaking", "working", "verifying", "researching", "coding", "waiting", "error", "offline"];

// 1. every server state has one presence state, and only the documented ones exist
const mapped = Object.fromEntries(SERVER_STATES.map((s) => [s, presenceFor(s)]));
check("every server state maps to a presence state", SERVER_STATES.every((s) => PRESENCE_STATES.includes(mapped[s])), JSON.stringify(mapped));
check("working, coding, researching and verifying are executing", ["working", "coding", "researching", "verifying"].every((s) => mapped[s] === "executing"));
check("listening and transcribing are listening", mapped.listening === "listening" && mapped.transcribing === "listening");
check("idle and waiting are calm", mapped.idle === "idle" && mapped.waiting === "idle");
check("an unknown state is idle, never drama", presenceFor("something_new") === "idle" && presenceFor("") === "idle");

// 2. receipts pulse only when they say something
check("a verified receipt pulses success", pulseForReceipt({ verified: true, ok: true }) === "success");
check("a failed receipt pulses error", pulseForReceipt({ verified: false, ok: false }) === "error");
check("an unverified success does not pulse", pulseForReceipt({ verified: false, ok: true }) === "" && pulseForReceipt(null) === "");

// 3. the ribbon itself, in a minimal browser stand-in
const calls = [];
const ctx = new Proxy({}, { get: (target, key) => (key in target ? target[key] : (...args) => { calls.push(String(key)); return { addColorStop() {} }; }), set: (target, key, value) => { target[key] = value; return true; } });
let reducedMotion = false;
globalThis.window = { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: false }) };
globalThis.document = { hasFocus: () => true, hidden: false, body: { classList: { contains: (c) => c === "reduced-motion" && reducedMotion } } };
const here = dirname(fileURLToPath(import.meta.url));
new Function(readFileSync(join(here, "..", "orb.js"), "utf8"))();
const canvas = { getContext: () => ctx, getBoundingClientRect: () => ({ width: 720, height: 128 }), width: 0, height: 0 };
const Presence = globalThis.window.ZeusPresence;
check("orb.js registers ZeusPresence (and the older names)", typeof Presence === "function" && window.ZeusOrb === Presence && window.JarvisEye === Presence);
const tableKeys = Object.keys(globalThis.window.JARVIS_STATES).sort();
check("the ribbon's state table has exactly the server's states", JSON.stringify(tableKeys) === JSON.stringify([...SERVER_STATES].sort()), tableKeys.join(","));

const ribbon = new Presence(canvas);
ribbon.setState("idle");
ribbon.setState("working");
check("entering an action sends a directional pulse", ribbon.pulses.length === 1 && ribbon.mode === "executing");
ribbon.pulses = [];
ribbon.setState("idle");
check("finishing an action sends one success pulse", ribbon.pulses.length === 1 && !ribbon.pulses[0].warm && ribbon.pulses[0].strength === 1);
ribbon.pulses = [];
ribbon.setState("error");
check("an error sends one muted warm pulse", ribbon.pulses.length === 1 && ribbon.pulses[0].warm && ribbon.pulses[0].strength < 1);
ribbon.setState("error");
check("staying in error does not pulse again", ribbon.pulses.length === 1);

ribbon.setState("idle");
const quiet = ribbon.now.amp;
ribbon.setState("speaking");
for (let i = 0; i < 60; i++) { ribbon.setEnergy(0.9); ribbon.step(1 / 60, false); }   // playback feeds the level every frame
check("speaking with audio energy opens the ribbon", ribbon.energy > 0.5 && ribbon.now.amp > quiet, `energy ${ribbon.energy} amp ${ribbon.now.amp}`);
calls.length = 0;
ribbon.draw(false);
check("the ribbon draws strokes, not a sphere", calls.includes("stroke") && !calls.includes("arc"), [...new Set(calls)].join(","));

reducedMotion = true;
const still = new Presence(canvas);
const before = still.t;
still.setState("thinking");
still.step(0.25, still.reduced());
check("reduced motion: no drift and the state is taken at once", still.t === before && still.now.amp === globalThis.window.JARVIS_STATES.thinking.amp);
reducedMotion = false;

// 4. the controller drives the ribbon from bus events
const handlers = {};
const bus = { on: (type, fn) => { (handlers[type] ||= []).push(fn); } };
const seen = [];
const fake = { setState: (s) => seen.push(["state", s]), pulseOnce: (k) => seen.push(["pulse", k]), setEnergy: (e) => seen.push(["energy", e]), setBackgroundWork() {} };
globalThis.document.documentElement = { dataset: {} };
const controller = attach(fake, bus);
handlers.state[0]({ state: "coding" });
handlers.tool[0]({ receipt: { verified: true, ok: true } });
handlers.tool[0]({ receipt: { verified: true, ok: true }, _replay: true });
handlers.speech[0]({ energy: 0.4 });
check("the controller passes the state, pulses on live receipts only, and forwards speech energy",
      controller.state === "executing" && JSON.stringify(seen) === JSON.stringify([["state", "coding"], ["pulse", "success"], ["energy", 0.4]]), JSON.stringify(seen));

console.log(failures ? `${failures} FAILED` : "ALL OK");
process.exit(failures ? 1 : 0);
