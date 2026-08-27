/* Diagnostics: "is ZEUS healthy?" from the doctor (deterministic, never wakes
   a model), with resources, the router inspector and process counts on
   demand. Measured tier probes are an explicit button that says its cost. */

import { el, clear, kv, section, badge, button, ago } from "../core/dom.js";
import { api } from "../core/api.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";

export const view = {
  id: "diagnostics",
  title: "Diagnostics",
  async mount(pane) {
    const head = el("div", { class: "card" }, el("div", { class: "title", text: "Checking…" }));
    const checks = el("div", { class: "check-list" });
    pane.append(head, section("Doctor", checks));
    const doctor = await api("/api/doctor");
    clear(head);
    head.append(
      el("div", { class: "title" }, badge(doctor.healthy ? "HEALTHY" : "ATTENTION", doctor.healthy ? "ok" : "bad"), " ", doctor.summary || ""),
      el("div", { class: "meta" }, el("span", { text: `${(doctor.checks || []).length} checks in ${doctor.seconds}s` }), el("span", { text: doctor.at })),
    );
    for (const c of doctor.checks || []) {
      const row = el("div", { class: "row" }, el("span", { class: "dot " + c.level }), el("span", { class: "name", text: c.name }),
        el("span", { class: "detail" }, c.detail, c.remedy ? el("div", { class: "remedy", text: "→ " + c.remedy }) : null));
      row.onclick = () => views.inspect(c.name, section("Finding", kv("level", c.level), kv("detail", c.detail), kv("remedy", c.remedy)),
        section("Data", kv("json", JSON.stringify(c.data || {}, null, 1), "mono")));
      row.style.cursor = "pointer";
      checks.append(row);
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
    pane.append(section("Which brain handled this?", el("div", { class: "empty", style: { padding: 0 }, text: "Every request's route is in Activity (routed: …) with top level, confidence, executor and the signals that were overruled." }),
      el("div", { class: "toolbar" }, button("Open routing evidence", () => views.open("activity", { q: "routed:" })))));
    pane.append(section("Measured tiers", el("div", { class: "empty", style: { padding: 0 }, text: "Probing a tier is a real generation and evicts the chat model for ~28 s. Only on purpose." }),
      el("div", { class: "toolbar" }, button("Probe now (costs a generation)", async () => {
        const d = await api("/api/diagnostics", { refresh: true });
        views.inspect("Measured", el("pre", { class: "code", text: JSON.stringify(d.kernel || d, null, 1) }));
      }, "ghost danger"))));
  },
};
