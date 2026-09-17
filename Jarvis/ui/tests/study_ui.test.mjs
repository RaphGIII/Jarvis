/* Studium polish, the parts of the interface that are logic: the narrow-window drawer, marking phrases in text
   (umlauts, hyphenation, every occurrence, one primary), and which missions may call ZEUS busy.
   Run: node ui/tests/study_ui.test.mjs  (the Python test suite runs it too). */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { NARROW_QUERY, toggleOutcome, trapTarget, createDrawer } from "../core/drawer.js";
import { findPhrases, primarySpan } from "../core/textmarks.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, "..", "zeus.css"), "utf-8");

/* ------------------------------------------------------------ a small fake DOM for the drawer */

class ClassList {
  constructor() { this.set = new Set(); }
  toggle(name, force) { const on = force === undefined ? !this.set.has(name) : Boolean(force); if (on) this.set.add(name); else this.set.delete(name); return on; }
  contains(name) { return this.set.has(name); }
  add(name) { this.set.add(name); }
  remove(name) { this.set.delete(name); }
}
class Node {
  constructor(name, children = []) { this.name = name; this.children = children; this.attrs = {}; this.classList = new ClassList(); this.listeners = {}; this.hidden = false; this.inert = false; this.disabled = false; for (const c of children) c.parent = this; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  click() { for (const fn of this.listeners.click || []) fn({}); }
  focus() { globalThis.document.activeElement = this; }
  contains(node) { for (let n = node; n; n = n.parent) if (n === this) return true; return false; }
  querySelectorAll() { return this.children; }
  querySelector(selector) { return selector === ".sb-item.on" ? this.children.find((c) => c.classList.contains("on")) || null : this.children[0] || null; }
}

const body = new Node("body");
globalThis.document = { activeElement: body, body, contains: () => true };
globalThis.requestAnimationFrame = (fn) => fn();

function world(narrow) {
  const listeners = [];
  const media = { matches: narrow, addEventListener: (_t, fn) => listeners.push(fn) };
  const items = [new Node("home"), new Node("chat"), new Node("study")];
  items[2].classList.add("on");
  const sidebar = new Node("sidebar", items);
  const toggle = new Node("toggle");
  const main = new Node("main", [toggle]);
  const app = new Node("app", [sidebar, main]);
  const scrim = new Node("scrim");
  const collapses = [];
  const drawer = createDrawer({ app, sidebar, scrim, toggle, main, matchMedia: (q) => { media.query = q; return media; }, onCollapse: (c) => collapses.push(c) });
  return { media, listeners, items, sidebar, toggle, main, app, scrim, drawer, collapses };
}

// 1. the pure decisions
check("a narrow toggle opens and closes the drawer", toggleOutcome({ narrow: true, open: false }).open === true && toggleOutcome({ narrow: true, open: true }).open === false);
check("a wide toggle collapses the column instead", toggleOutcome({ narrow: false, collapsed: false }).drawer === false && toggleOutcome({ narrow: false, collapsed: false }).collapsed === true);
const [a, b, c] = ["a", "b", "c"];
check("Tab from the last item wraps to the first", trapTarget([a, b, c], c, false) === a);
check("Shift+Tab from the first item wraps to the last", trapTarget([a, b, c], a, true) === c);
check("Tab inside the drawer moves normally", trapTarget([a, b, c], b, false) === null);
check("focus outside the drawer is pulled in", trapTarget([a, b, c], "elsewhere", false) === a);

// 2. narrow: overlay, inert content, focus in and back, Escape
{
  const w = world(true);
  check("the drawer uses the shared narrow query", w.media.query === NARROW_QUERY && NARROW_QUERY === "(max-width: 860px)");
  check("closed at start", !w.app.classList.contains("sidebar-open") && w.scrim.hidden === true && w.main.inert === false);
  w.toggle.focus();
  w.toggle.click();
  check("pressing the toggle opens the drawer over the content", w.app.classList.contains("sidebar-open") && w.scrim.hidden === false);
  check("the content behind is inert while the drawer is open", w.main.inert === true);
  check("the toggle says it is expanded", w.toggle.getAttribute("aria-expanded") === "true");
  check("focus moves to the current destination inside the drawer", globalThis.document.activeElement === w.items[2]);
  check("the drawer is announced as a modal dialog", w.sidebar.getAttribute("role") === "dialog" && w.sidebar.getAttribute("aria-modal") === "true");
  const trapped = { key: "Tab", shiftKey: false, prevented: false, preventDefault() { this.prevented = true; } };
  globalThis.document.activeElement = w.items[2];
  w.drawer.onKeydown(trapped);
  check("Tab from the last destination stays inside the drawer", trapped.prevented && globalThis.document.activeElement === w.items[0]);
  const esc = { key: "Escape", prevented: false, preventDefault() { this.prevented = true; } };
  check("Escape is handled by the drawer", w.drawer.onKeydown(esc) === true && esc.prevented);
  check("Escape closes the drawer and releases the content", !w.app.classList.contains("sidebar-open") && w.scrim.hidden === true && w.main.inert === false);
  check("focus returns to the toggle", globalThis.document.activeElement === w.toggle);
  check("a closed drawer ignores Escape (the app handles it)", w.drawer.onKeydown({ key: "Escape", preventDefault() {} }) === false);
  w.toggle.click();
  w.scrim.click();
  check("the veil closes the drawer", !w.app.classList.contains("sidebar-open"));
  w.toggle.click();
  w.media.matches = false;
  for (const fn of w.listeners) fn();
  check("growing past the breakpoint closes the drawer", !w.app.classList.contains("sidebar-open") && w.main.inert === false);
  check("a narrow drawer never collapses the wide column", w.collapses.length === 0);
}

// 3. wide: a column that collapses, never a drawer
{
  const w = world(false);
  w.toggle.click();
  check("a wide toggle collapses the sidebar column", w.app.classList.contains("sidebar-collapsed") && !w.app.classList.contains("sidebar-open"));
  check("the collapse is remembered", w.collapses.length === 1 && w.collapses[0] === true);
  check("nothing is inert on a wide window", w.main.inert === false && w.scrim.hidden === true);
  w.drawer.setOpen(true);
  check("a wide window refuses to open a drawer", !w.app.classList.contains("sidebar-open"));
}

// 4. the stylesheet keeps the content's width
const narrow = css.slice(css.indexOf("@media (max-width: 860px)"));
check("at 860 px and below the grid is one column", /#app, #app\.sidebar-collapsed, #app\.sidebar-open \{ grid-template-columns: minmax\(0, 1fr\); \}/.test(narrow));
check("the narrow sidebar is fixed over the content", /#sidebar \{ position: fixed;/.test(narrow) && /transform: translateX\(-104%\)/.test(narrow));
check("the old push layout is gone", !/#app\.sidebar-open \{ grid-template-columns: min\(80vw/.test(css));

// 5. marking phrases: every occurrence, umlauts, hyphenation, one primary
const page = "2.1 Vorlast und der Frank-Starling-Mechanismus\nDer Frank-Starling-Mechanismus beschreibt ... Klinisch erklärt der Frank-Starling-\nMechanismus, warum.";
const spans = findPhrases(page, ["Frank-Starling-Mechanismus", "Vorlast"]);
const focus = spans.filter((s) => s[2] === 0);
check("every occurrence of the focus is found, even across a line-end hyphen", focus.length === 3, JSON.stringify(spans));
check("other phrases are found too", spans.some((s) => s[2] === 1 && page.slice(s[0], s[1]) === "Vorlast"));
check("the primary is the occurrence the result pointed at", primarySpan(spans, page.lastIndexOf("Frank"), 0) === focus[2]);
check("without an offset the first occurrence is primary", primarySpan(spans, null) === focus[0]);
check("case and umlauts do not matter", findPhrases("Die Vorlast FÖRDERT das Schlagvolumen", ["fördert"]).length === 1
  && findPhrases("Der Weg fördert Bewegung", ["FORDERT"]).length === 1);
check("ß matches ss", findPhrases("Die Größe des Herzens", ["Grösse"]).length === 1);
check("spans never overlap", findPhrases("Basalganglienschleife", ["Basalganglienschleife", "schleife"]).length === 1);
check("regex characters in a phrase are literal", findPhrases("Na+/K+-ATPase (Pumpe)", ["Na+/K+-ATPase (Pumpe)"]).length === 1);

// 6. missions that sit for days do not keep ZEUS "busy"
const worksurface = readFileSync(join(here, "..", "core", "worksurface.js"), "utf-8");
const liveSource = worksurface.slice(worksurface.indexOf("const LIVE_WINDOW_MS"), worksurface.indexOf("function activeMissions"));
const missionIsLive = new Function(`${liveSource.replace("export function", "function")}; return missionIsLive;`)();
const now = Date.parse("2026-09-17T12:00:00Z");
check("a mission started two weeks ago is not live", !missionIsLive({ state: "active", started: "2026-09-02T19:23:33+00:00" }, now));
check("a mission started an hour ago is live", missionIsLive({ state: "active", started: "2026-09-17T11:00:00+00:00" }, now));
check("an old mission with fresh progress is live", missionIsLive({ state: "active", started: "2026-09-02T19:23:33+00:00" }, now, now - 60000));
check("epoch seconds are understood", !missionIsLive({ state: "active", started: 1788461069.68 }, now));
check("finished missions are never live", !missionIsLive({ state: "active", finished: true, started: "2026-09-17T11:00:00+00:00" }, now));

console.log(failures ? `${failures} FAILED` : "ALL OK");
process.exit(failures ? 1 : 0);
