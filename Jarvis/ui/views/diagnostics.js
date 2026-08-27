/* Diagnostics: "is ZEUS healthy?" from the doctor (deterministic, never wakes
   a model), shown per SUBSYSTEM with a HEALTHY / DEGRADED / FAILING verdict —
   a process that exists is not a function that works — with the checks
   nested under each subsystem. Resources, the router inspector and process
   counts on demand. Measured tier probes are an explicit button that says
   its cost. */

import { el, clear, kv, section, badge, button, ago } from "../core/dom.js";
import { api } from "../core/api.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";

const TONE = { HEALTHY: "ok", DEGRADED: "warn", FAILING: "bad" };
const ORDER = ["core", "voice", "wakeword", "knowledge", "capabilities", "projects", "expert", "infrastructure"];

export const view = {
  id: "diagnostics",
  title: "Diagnostics",
  async mount(pane) {
    const head = el("div", { class: "card" }, el("div", { class: "title", text: "Checking…" }));
    const systems = el("div", { class: "check-list" });
    pane.append(head, section("Subsystems", systems));
    const doctor = await api("/api/doctor");
    clear(head);
    const overall = doctor.overall || (doctor.healthy ? "HEALTHY" : "FAILING");
    head.append(
      el("div", { class: "title" }, badge(overall, TONE[overall] || "bad"), " ", doctor.summary || ""),
      el("div", { class: "meta" }, el("span", { text: `${(doctor.checks || []).length} checks in ${doctor.seconds}s` }), el("span", { text: doctor.at })),
    );
    const checksBySub = {};
    for (const c of doctor.checks || []) (checksBySub[c.subsystem || "infrastructure"] ||= []).push(c);
    const subs = (doctor.subsystems || []).slice().sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
    for (const s of subs) {
      const box = el("div", { class: "card" });
      box.append(el("div", { class: "title" }, badge(s.health, TONE[s.health] || "bad"), " ", s.name.toUpperCase(), s.important ? "" : el("span", { class: "empty", style: { padding: "0 8px" }, text: "(infrastructure)" })));
      if (s.detail) box.append(el("div", { class: "empty", style: { padding: "2px 0 6px" }, text: s.detail }));
      for (const c of checksBySub[s.name] || []) {
        const row = el("div", { class: "row" }, el("span", { class: "dot " + c.level }), el("span", { class: "name", text: c.name }),
          el("span", { class: "detail" }, c.detail, c.remedy ? el("div", { class: "remedy", text: "→ " + c.remedy }) : null));
        row.onclick = () => views.inspect(c.name, section("Finding", kv("subsystem", c.subsystem), kv("health", c.health), kv("level", c.level), kv("detail", c.detail), kv("remedy", c.remedy)),
          section("Data", kv("json", JSON.stringify(c.data || {}, null, 1), "mono")));
        row.style.cursor = "pointer";
        box.append(row);
      }
      systems.append(box);
    }
    // resources: the readings the page already has
    const gpu = state.gpu || {};
    const health = state.health || {};
    pane.append(section("Resources",
      kv("GPU", gpu.available ? `${gpu.name || "GPU"} · ${gpu.utilization_percent}% · ${gpu.memory_used_mib}/${gpu.memory_total_mib} MiB` : "no reading"),
      kv("readiness", Object.entries(health.readiness || {}).map(([k, v]) => `${k}${v ? " ✓" : " ·"}`).join("  ")),
      kv("revision", (health.revision || "").slice(0, 12)), kv("uptime", `${health.uptime_seconds || 0}s`), kv("window", health.window ? (health.window.visible ? "visible" : health.window.exists ? "hidden" : "closed") : "—")));
    const procs = el("div");
    pane.append(section("Processes", procs, el("div", { class: "toolbar" }, button("Count from the process table", async () => {
      const p = await api("/api/processes");
      clear(procs);
      for (const [role, n] of Object.entries(p.counts || {})) procs.append(kv(role, n));
    }))));
    const voice = el("div");
    pane.append(section("Voice", voice, el("div", { class: "toolbar" }, button("Read the voice stack", async () => {
      const [w, v] = await Promise.all([api("/api/voice/wake"), api("/api/voice")]);
      clear(voice);
      const ev = w.evaluation || {};
      voice.append(kv("listener process", health.voice ? "ready" : "not ready"), kv("STT (recogniser)", health.recogniser ? "ready" : "not ready"),
        kv("wake model", `${w.model_kind} ${w.model_fingerprint || ""}`), kv("effective threshold", `${w.effective_threshold} (${w.threshold_source})`),
        kv("wake evaluation", ev.at ? `recall ${ev.positive_recall} · rejection ${ev.negative_rejection} · ${ev.in_sample === false ? "held-out" : "in-sample"} · ${ev.at}` : "none"),
        kv("listener = test config", w.listener ? (w.listener_match ? "yes" : `NO: listener ${w.listener.fingerprint} @ ${w.listener.threshold}`) : "no report"),
        kv("TTS", (v.engine || {}).available ? `available · volume ${(v.settings || {}).volume}` : "unavailable"),
        kv("barge-in", "listener interrupts speech on wake (POST /api/stop)"),
        kv("ignored utterances (last)", (v.rejected_utterances || []).map((r) => `${r.reason}: „${r.text}“`).join("\n") || "none"));
    }), button("Voice Studio", () => views.open("voice")))));
    pane.append(section("Which brain handled this?", el("div", { class: "empty", style: { padding: 0 }, text: "Every request's route is in Activity (routed: …) with top level, confidence, executor and the signals that were overruled." }),
      el("div", { class: "toolbar" }, button("Open routing evidence", () => views.open("activity", { q: "routed:" })))));
    pane.append(section("Measured tiers", el("div", { class: "empty", style: { padding: 0 }, text: "Probing a tier is a real generation and evicts the chat model for ~28 s. Only on purpose." }),
      el("div", { class: "toolbar" }, button("Probe now (costs a generation)", async () => {
        const d = await api("/api/diagnostics", { refresh: true });
        views.inspect("Measured", el("pre", { class: "code", text: JSON.stringify(d.kernel || d, null, 1) }));
      }, "ghost danger"))));
  },
};
