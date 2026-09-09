/* The mode bar above the composer: AUTO / FREE / SMART / DEEP / BUILD, the
   estimated cost of what is being typed when it is not trivially zero, and
   the month's spend against the hard cap.

   The mode is server truth (POST /api/gateway/mode), not a browser
   preference: it is what the gateway enforces, so it has to live where the
   gateway can read it. The bar re-reads /api/gateway/status after every
   answer, because that is when the ledger changes. */

import { $, el, clear, debounce } from "./dom.js";
import { api } from "./api.js";
import * as bus from "./bus.js";

const MODES = [
  ["AUTO", "günstigster verlässlicher Weg"],
  ["FREE", "null Kosten, bezahlte Anbieter gesperrt"],
  ["SMART", "günstiges Cloud-Denken im Budget"],
  ["DEEP", "stärkstes Denkmodell"],
  ["BUILD", "Engineering: Fähigkeiten bauen/reparieren"],
];

let status = null;
let mode = "AUTO";
let chips = new Map();
let spendEl = null;
let estimateEl = null;

export function currentMode() {
  return mode;
}

function eur(value) {
  const n = Number(value || 0);
  return "€" + (n < 0.01 && n > 0 ? n.toFixed(3) : n.toFixed(2));
}

function render() {
  for (const [name, chip] of chips) chip.classList.toggle("on", name === mode);
  if (!spendEl) return;
  clear(spendEl);
  if (!status) { spendEl.append(el("span", { class: "dim", text: "Gateway…" })); return; }
  const s = status.spend || {};
  spendEl.append(
    el("span", { class: "mode-now", text: mode }),
    el("span", { class: "sep", text: "·" }),
    el("span", { title: "Ausgaben diesen Monat / hartes Monatslimit",
                 text: `Monat: ${eur(s.month)} / ${eur(s.monthly_hard_cap)}` }),
  );
  if (status.cloud_reasoning_available === false) {
    spendEl.append(el("span", { class: "sep", text: "·" }),
                   el("span", { class: "warn", title: "Kein Cloud-Denkmodell konfiguriert: Owner Settings › Provider",
                                text: "nur lokal" }));
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
    if (r.hint) showEstimate(r.hint, "");
    refresh();
  }
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
  if (d.kind === "refused") {
    showEstimate(d.suggestion || "in diesem Modus nicht möglich", "warn");
    return;
  }
  if (d.kind !== "model") { showEstimate("", ""); return; }
  const est = d.estimate || {};
  const cost = Number(est.estimated_eur || 0);
  if (cost <= 0) {
    showEstimate(d.offline_fallback ? "lokal · €0.00" : "€0.00", "");
    return;
  }
  const [lo, hi] = est.range_eur || [cost, cost];
  const confirmed = est.pricing_confirmed ? "" : " (Preise unbestätigt)";
  showEstimate(`≈ ${eur(lo)} – ${eur(hi)} · ${d.role}${confirmed}`, "cost");
}, 500);

export function init() {
  const bar = $("modebar");
  if (!bar) return;
  clear(bar);
  chips = new Map();
  const group = el("div", { class: "modes", role: "radiogroup", "aria-label": "Chat-Modus" });
  for (const [name, hint] of MODES) {
    const chip = el("button", { class: "chip", text: name, title: hint, role: "radio", onClick: () => setMode(name) });
    chips.set(name, chip);
    group.append(chip);
  }
  spendEl = el("div", { class: "spend", id: "gatewaySpend" });
  estimateEl = el("div", { class: "estimate", id: "gatewayEstimate", hidden: true });
  bar.append(group, estimateEl, spendEl);
  render();
  refresh();
  const input = $("input");
  if (input) input.addEventListener("input", () => estimateFor(input.value));
  bus.on("message", () => { showEstimate("", ""); refresh(); });
  bus.on("user_message", () => showEstimate("", ""));
  setInterval(refresh, 60_000);
}
