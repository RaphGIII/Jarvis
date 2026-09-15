/* The ZEUS performance control: one compact selector in the composer instead
   of a row of mode chips.  The owner chooses how much performance a request
   may use -- Automatisch, Ohne Kosten, Mehr Leistung, Maximale Leistung,
   Entwicklung -- and ZEUS maps that onto its internal intelligence classes.
   No provider and no model is ever named here; the monthly spend is shown
   quietly next to it, and a cost estimate appears only when a request would
   cost money. */

import { $, el, clear, debounce } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";
import { state } from "./state.js";

/* Off means off: with the setting disabled the spend line is empty and is never written again. */
export function spendVisible() {
  return state.ui.showSpend !== false;
}

export const LEVELS = [
  ["AUTO", "Automatisch", "ZEUS wählt die passende Leistung selbst"],
  ["FREE", "Ohne Kosten", "nur kostenlose Intelligenz, nie bezahlte"],
  ["SMART", "Mehr Leistung", "stärkeres Denken für anspruchsvolle Fragen"],
  ["DEEP", "Maximale Leistung", "das stärkste Denken, wenn es darauf ankommt"],
  ["BUILD", "Entwicklung", "ZEUS baut oder repariert Fähigkeiten"],
];

let status = null;
let mode = "AUTO";
let root = null;
let btn = null;
let spendEl = null;
let estimateEl = null;
let menu = null;

export function currentMode() {
  return mode;
}

export function labelFor(name) {
  return (LEVELS.find(([id]) => id === name) || LEVELS[0])[1];
}

function eur(value) {
  const n = Number(value || 0);
  return "€" + (n < 0.01 && n > 0 ? n.toFixed(3) : n.toFixed(2));
}

function render() {
  if (!btn) return;
  clear(btn);
  btn.dataset.level = mode;
  btn.title = "Leistung: " + labelFor(mode);
  btn.append(el("span", { class: "brand", text: "ZEUS" }), el("span", { class: "lvl", text: labelFor(mode) }), el("span", { class: "chev", text: "▾" }));
  clear(spendEl);
  if (!spendVisible()) { spendEl.hidden = true; return; }
  if (status && status.spend) {
    const s = status.spend;
    spendEl.append(el("b", { text: eur(s.month) }), el("span", { text: ` / ${eur(s.monthly_hard_cap)}` }));
    spendEl.hidden = false;
    spendEl.title = "Ausgaben diesen Monat / dein monatliches Limit";
  }
}

export async function refresh() {
  const data = await api("/api/gateway/status");
  if (data && data.ok !== false && data.spend) {
    status = data;
    if (data.mode) mode = data.mode;
  }
  render();
}

async function setMode(next) {
  const r = await api("/api/gateway/mode", { mode: next });
  if (r && r.ok) {
    mode = r.mode;
    render();
    bus.emit("gateway:mode", mode);
    refresh();
  }
  closeMenu();
}

function openMenu() {
  if (menu) { closeMenu(); return; }
  menu = el("div", { class: "perf-menu glass" }, el("h6", { text: "Leistung für diesen Chat" }));
  for (const [id, label, desc] of LEVELS) {
    menu.append(el("button", { class: "perf-opt" + (id === mode ? " on" : ""), onClick: () => setMode(id) },
      el("span", { class: "mark", text: id === mode ? "✓" : "" }),
      el("span", {}, el("span", { class: "t", text: label }), el("span", { class: "d", text: desc }))));
  }
  root.append(menu);
  setTimeout(() => document.addEventListener("click", onOutside), 0);
}

function onOutside(e) {
  if (menu && !menu.contains(e.target) && e.target !== btn && !btn.contains(e.target)) closeMenu();
}

function closeMenu() {
  menu?.remove();
  menu = null;
  document.removeEventListener("click", onOutside);
}

function showEstimate(text, tone) {
  if (!estimateEl) return;
  estimateEl.textContent = text || "";
  estimateEl.dataset.tone = tone || "";
  estimateEl.hidden = !text;
}

const estimateFor = debounce(async (text) => {
  const clean = (text || "").trim();
  if (clean.length < 12) { showEstimate("", ""); return; }
  const r = await api("/api/gateway/estimate", { text: clean, mode });
  if (!r || r.ok === false || !r.decision) { showEstimate("", ""); return; }
  const d = r.decision;
  if (d.kind === "refused") { showEstimate("in dieser Leistungsstufe nicht möglich", "warn"); return; }
  if (d.kind !== "model") { showEstimate("", ""); return; }
  const cost = Number((d.estimate || {}).estimated_eur || 0);
  if (cost <= 0 || !spendVisible()) { showEstimate("", ""); return; }
  const [lo, hi] = (d.estimate || {}).range_eur || [cost, cost];
  showEstimate(`≈ ${eur(lo)} – ${eur(hi)}`, "cost");
}, 500);

export function init() {
  root = $("perf");
  if (!root) return;
  clear(root);
  btn = el("button", { class: "perf-btn", title: "Leistungsstufe wählen", onClick: (e) => { e.stopPropagation(); openMenu(); } });
  spendEl = $("perfSpend") || el("span", { class: "perf-spend" });
  estimateEl = $("perfEstimate") || el("span", { class: "perf-est", hidden: true });
  root.append(btn);
  if (!spendEl.isConnected) root.append(spendEl);
  if (!estimateEl.isConnected) root.append(estimateEl);
  render();
  refresh();
  const input = $("input");
  if (input) input.addEventListener("input", () => estimateFor(input.value));
  bus.on("message", () => { showEstimate("", ""); refresh(); });
  bus.on("pref:showSpend", () => render());
  bus.on("user_message", () => showEstimate("", ""));
  setInterval(refresh, 60_000);
}
