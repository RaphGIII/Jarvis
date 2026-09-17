/* Studium "Sterne": the pure geometry of the sky (ui/core/starfield.js) -- camera transforms, grid hit-testing,
   emphasis, label level of detail, viewport culling at 5 000 documents, keyboard movement -- and the view's
   wiring as far as it can be read without a browser.
   Run: node ui/tests/study_galaxy.test.mjs  (the Python test suite runs it too). */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as sf from "../core/starfield.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

/* a deterministic library: n points in clusters on roughly [-1, 1] */
function library(n, seed = 7) {
  let s = seed;
  const rnd = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
  const nodes = [];
  for (let i = 0; i < n; i += 1) {
    const c = i % 24;
    const cx = Math.cos(c * 2.4) * 0.7 * Math.sqrt(c / 24), cy = Math.sin(c * 2.4) * 0.7 * Math.sqrt(c / 24);
    const a = rnd() * Math.PI * 2, r = Math.sqrt(rnd()) * 0.12;
    nodes.push({ id: `d${i}`, x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r });
  }
  return nodes;
}

/* ------------------------------------------------------------ transforms */

const view = { w: 1200, h: 700 };
const camera = { x: 0.13, y: -0.4, zoom: 412.5 };
for (const [wx, wy] of [[0, 0], [0.5, -0.25], [-1, 1], [0.13, -0.4]]) {
  const s = sf.worldToScreen(camera, view, wx, wy);
  const back = sf.screenToWorld(camera, view, s.x, s.y);
  check(`world -> screen -> world round trip (${wx}, ${wy})`, near(back.x, wx, 1e-12) && near(back.y, wy, 1e-12));
}
const centre = sf.worldToScreen(camera, view, camera.x, camera.y);
check("the camera point sits in the middle of the view", near(centre.x, 600) && near(centre.y, 350));
const zoomed = sf.zoomAt(camera, view, 900, 200, 2);
const before = sf.screenToWorld(camera, view, 900, 200), after = sf.screenToWorld(zoomed, view, 900, 200);
check("zooming keeps the point under the cursor", near(before.x, after.x, 1e-12) && near(before.y, after.y, 1e-12) && near(zoomed.zoom, 825));
check("zoom is clamped", sf.zoomAt(camera, view, 0, 0, 1e9).zoom === sf.MAX_ZOOM && sf.zoomAt(camera, view, 0, 0, 1e-9).zoom === sf.MIN_ZOOM);
const fit = sf.fitCamera([{ x: -1, y: -0.5 }, { x: 1, y: 0.5 }], view, { margin: 50 });
const corner = sf.worldToScreen(fit, view, -1, -0.5), other = sf.worldToScreen(fit, view, 1, 0.5);
check("fit-to-all shows every point inside the margin", corner.x >= 49.999 && other.x <= 1150.001 && corner.y >= 49.999 && other.y <= 650.001, JSON.stringify({ corner, other }));
const mid = sf.lerpCamera({ x: 0, y: 0, zoom: 100 }, { x: 2, y: 2, zoom: 400 }, 0.5);
check("an eased camera move passes the midpoint (zoom in log space)", near(mid.x, 1) && near(mid.zoom, 200, 1e-9));
check("an eased camera move ends exactly at the target", JSON.stringify(sf.lerpCamera({ x: 0, y: 0, zoom: 100 }, { x: 2, y: 3, zoom: 400 }, 1)) === JSON.stringify({ x: 2, y: 3, zoom: 400 }));

/* ------------------------------------------------------------ hit-testing */

const nodes = library(5000);
const grid = sf.buildGrid(nodes, sf.gridCellFor(nodes));
let agree = 0;
let scanned = 0;
for (let k = 0; k < 400; k += 1) {
  const wx = Math.cos(k) * 0.8, wy = Math.sin(k * 1.7) * 0.8, radius = 0.03;
  const fast = sf.nearestInGrid(grid, wx, wy, radius);
  let best = null, bestD = radius * radius;
  for (const n of nodes) { const d = (n.x - wx) ** 2 + (n.y - wy) ** 2; if (d <= bestD) { bestD = d; best = n; } }
  if (fast === best) agree += 1;
  if (best) scanned += 1;
}
check("grid hit-testing returns the nearest node within the radius (400 probes vs brute force)", agree === 400, `${agree}/400`);
check("hit-testing probes actually hit something", scanned > 100, String(scanned));
check("nothing within the radius -> no node", sf.nearestInGrid(grid, 5, 5, 0.01) === null);
const target = nodes[1234];
check("a click right on a point finds that point", sf.nearestInGrid(grid, target.x + 1e-7, target.y, 0.001) === target);
const far = sf.nearestInGrid(grid, 0, 0, 10);
check("a huge radius (far zoomed out) still answers without scanning cell by cell", far !== null);

/* ------------------------------------------------------------ emphasis + level of detail */

const emphasis = new Set(["d1", "d2"]);
check("without emphasis a node keeps its brightness", sf.emphasisAlpha("d9", null, 0.6) === 0.6 && sf.emphasisAlpha("d9", new Set(), 0.6) === 0.6);
check("emphasised nodes brighten", sf.emphasisAlpha("d1", emphasis, 0.6) > 0.6);
const dimmed = sf.emphasisAlpha("d9", emphasis, 0.6);
check("the others dim but do not vanish", dimmed < 0.6 && dimmed > 0);
check("a subject label waits until its cluster is a place on screen", !sf.clusterLabelVisible({ kind: "subject", r: 0.05 }, 300) && sf.clusterLabelVisible({ kind: "subject", r: 0.2 }, 300));
check("topic labels need more room than subject labels", !sf.clusterLabelVisible({ kind: "topic", label: "Herz", r: 0.2 }, 300) && sf.clusterLabelVisible({ kind: "topic", label: "Herz", r: 0.2 }, 400));
check("the rest of a subject has no label", !sf.clusterLabelVisible({ kind: "rest", label: "", r: 5 }, 900));
check("a hovered or selected document is always named", sf.nodeLabelVisible({ hovered: true }) && sf.nodeLabelVisible({ selected: true }));
check("a crowded cluster far away names no document", !sf.nodeLabelVisible({ clusterRadius: 0.2, clusterSize: 400, zoom: 400 }));
check("zoomed in on the same cluster, documents are named", sf.nodeLabelVisible({ clusterRadius: 0.2, clusterSize: 400, zoom: 8000 }));

/* ------------------------------------------------------------ culling at 5 000 */

const wide = sf.fitCamera(nodes, view, { margin: 40 });
check("fit to all: all 5 000 documents are in view", sf.visibleNodes(nodes, wide, view).length === 5000);
const close = { ...wide, zoom: wide.zoom * 8 };
const inView = sf.visibleNodes(nodes, close, view, 0);
const bounds = sf.viewBounds(close, view, 0);
const brute = nodes.filter((n) => n.x >= bounds.minX && n.x <= bounds.maxX && n.y >= bounds.minY && n.y <= bounds.maxY).length;
check("zoomed in, only what the view shows is drawn", inView.length === brute && inView.length < 5000 && inView.length > 0, `${inView.length}`);
let started = performance.now();
for (let i = 0; i < 60; i += 1) sf.visibleNodes(nodes, close, view);
const cullMs = (performance.now() - started) / 60;
check("culling 5 000 documents costs well under a frame", cullMs < 4, `${cullMs.toFixed(3)} ms`);
started = performance.now();
for (let i = 0; i < 1000; i += 1) sf.nearestInGrid(grid, Math.cos(i) * 0.7, Math.sin(i) * 0.7, 12 / wide.zoom);
const hitMs = (performance.now() - started) / 1000;
check("one hover hit-test at 5 000 documents is far under a millisecond", hitMs < 0.5, `${hitMs.toFixed(4)} ms`);
console.log(`     cull ${cullMs.toFixed(3)} ms/frame, hit-test ${(hitMs * 1000).toFixed(1)} µs`);

/* ------------------------------------------------------------ keyboard */

const plus = [
  { id: "c", x: 0, y: 0 }, { id: "right", x: 1, y: 0.1 }, { id: "farRight", x: 3, y: 0 }, { id: "left", x: -0.8, y: 0 },
  { id: "up", x: 0.05, y: -1 }, { id: "down", x: 0, y: 0.9 }, { id: "diagonal", x: 0.5, y: 0.9 },
];
const c = plus[0];
check("ArrowRight picks the nearest node to the right", sf.nearestInDirection(plus, c, "ArrowRight").id === "right");
check("ArrowLeft picks the node to the left", sf.nearestInDirection(plus, c, "ArrowLeft").id === "left");
check("ArrowUp picks the node above (screen y grows downward)", sf.nearestInDirection(plus, c, "ArrowUp").id === "up");
check("ArrowDown prefers straight below over a diagonal", sf.nearestInDirection(plus, c, "ArrowDown").id === "down");
check("nothing further in a direction -> no move", sf.nearestInDirection(plus, plus[2], "ArrowRight") === null);
check("an unknown key moves nothing", sf.nearestInDirection(plus, c, "Enter") === null);
check("without a selection, the arrows start at the node nearest the view centre", sf.nearestTo(plus, 0.9, 0).id === "right");

/* ------------------------------------------------------------ words */

check("units in words", sf.unitsWord({ source_type: "pdf", units: 12 }) === "12 Seiten" && sf.unitsWord({ source_type: "pdf", units: 1 }) === "1 Seite"
  && sf.unitsWord({ source_type: "pptx", units: 8 }) === "8 Folien");
check("types in words", sf.typeWord({ source_type: "docx" }) === "Word" && sf.typeWord({ source_type: "pdf", goodnotes: true }) === "GoodNotes" && sf.typeWord({ source_type: "image" }) === "Bild");
check("index states in words", sf.stateWord({ state: "processing" }) === "wird indexiert" && sf.stateWord({ state: "failed" }).includes("fehlgeschlagen") && sf.stateWord({}) === "durchsuchbar");

/* ------------------------------------------------------------ the view's wiring (read, not run) */

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "views", "study_galaxy.js"), "utf-8");
const study = readFileSync(join(here, "..", "views", "study.js"), "utf-8");
const css = readFileSync(join(here, "..", "zeus.css"), "utf-8");
check("no permanent animation loop: requestAnimationFrame only for invalidate and a finite camera ease",
  (source.match(/requestAnimationFrame\(/g) || []).length === 3 && source.includes("t < 1 ? requestAnimationFrame(step) : 0"));
check("reduced motion jumps instead of easing", source.includes("reduced-motion") && source.includes("prefers-reduced-motion"));
check("the canvas is reachable and labelled", source.includes('tabindex: "0"') && source.includes("aria-label"));
for (const label of ["Öffnen", "Im Studium suchen", "Im Chat verwenden", "Ähnliche Unterlagen", "Zu Fach zuordnen"]) check(`menu offers "${label}"`, source.includes(`"${label}"`));
check("the list stays the default: a fresh profile starts in Liste", study.includes('root.dataset.studyView = "list"') && study.includes('stored === "sky"'));
check("the choice is remembered under zeus.study.view", study.includes('"zeus.study.view"'));
check("the Sterne CSS lives in its own block right after the Studium block", css.indexOf("/* ---- Studium: Sterne ---- */") > css.indexOf("/* ---- Studium ---- */")
  && css.indexOf("/* ---- Studium: Sterne ---- */") < css.indexOf("/* ---- the viewer ---- */"));
check("no neon, no glow in the sky", !/text-shadow|drop-shadow|createRadialGradient|shadowBlur/.test(source));

if (failures) { console.log(`\n${failures} FAILED`); process.exit(1); }
console.log("\nALL OK");
