/* Mission Control: every long-running autonomous job — self-development
   missions and capability acquisitions — with phase, evidence, isolation
   reports, the candidate diff, cancel/resume. Real mission files only. */

import { el, clear, kv, section, badge, button, ago, seconds, clockOf } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";

const PHASES = ["UNDERSTAND", "INVESTIGATE", "BUILD", "VERIFY", "ESCALATE", "PROMOTE", "RESTARTING", "DONE"];
const isFinished = (m) => ["DONE", "FAILED", "CANCELLED"].includes(m.phase);

export const view = {
  id: "missions",
  title: "Mission Control",
  async mount(pane, params) {
    const tabs = el("div", { class: "toolbar" });
    const list = el("div");
    let filter = params.filter || "all";
    const data = await api("/api/selfdev");
    const missions = (data.missions || []).slice().reverse();
    const render = () => {
      clear(list);
      const shown = missions.filter((m) => filter === "all" || (filter === "active" ? !isFinished(m) : filter === "failed" ? m.outcome === "failed" || m.outcome === "rolled_back" : filter === "done" ? m.outcome === "promoted" : true));
      if (!shown.length) list.append(el("div", { class: "empty", text: missions.length ? "Nothing in this filter." : "No missions yet. Say „Zeus, ändere …“ and one appears here." }));
      for (const m of shown) list.append(card(m, () => views.open("missions", { ...params, mission: m.mission_id })));
    };
    for (const [key, label] of [["all", "All"], ["active", "Active"], ["done", "Promoted"], ["failed", "Failed"]]) {
      tabs.append(el("button", { class: "ghost", "aria-pressed": filter === key ? "true" : "false", text: label, onClick: () => { filter = key; for (const b of tabs.querySelectorAll("button")) b.setAttribute("aria-pressed", b.textContent === label ? "true" : "false"); render(); } }));
    }
    tabs.append(el("span", { class: "empty", style: { padding: 0 }, text: `${missions.filter((m) => !isFinished(m)).length} active · ${missions.length} total` }));
    pane.append(tabs, list);
    render();
    if (params.mission) {
      const m = missions.find((x) => x.mission_id === params.mission);
      if (m) inspect(m);
    }
    view._sub = bus.on("progress", async (p) => {
      if (p.kind !== "selfdev") return;
      const fresh = await api("/api/selfdev");
      missions.length = 0;
      missions.push(...(fresh.missions || []).slice().reverse());
      render();
    });
  },
  unmount() { view._sub?.(); },
};

function tone(m) {
  if (m.outcome === "promoted") return "ok";
  if (m.outcome === "cancelled") return "dim";
  if (m.outcome === "failed" || m.outcome === "rolled_back") return "bad";
  return "active";
}

function card(m, open) {
  const total = Object.values(m.timings || {}).reduce((a, b) => a + b, 0);
  const idx = PHASES.indexOf(m.phase);
  const node = el("div", { class: "card click" },
    el("div", { class: "title", text: m.request || "(no request)" }),
    el("div", { class: "meta" }, badge("selfdev", "blue"), badge(m.phase || "?", tone(m)), badge(m.outcome || "running", tone(m)),
      el("span", { text: `${m.changed_files?.length || 0} files` }), el("span", { text: m.escalated ? "expert used" : "local" }),
      el("span", { text: seconds(total) }), el("span", { text: ago(m.updated_at) })),
    el("div", { class: "bar " + (isFinished(m) ? "" : "green"), style: { marginTop: "8px" } }, el("i", { style: { width: `${idx >= 0 ? ((idx + 1) / PHASES.length) * 100 : 5}%` } })));
  node.onclick = open;
  return node;
}

function inspect(m) {
  const acceptance = (m.acceptance || []).map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: "criterion" }), el("span", { class: "v", text: a.criterion })));
  const checks = (m.verification?.checks || []).map((c) => el("div", { class: "kv" }, el("span", { class: "k", text: c.ok ? "✓" : "✗" }), el("span", { class: "v", text: c.criterion })));
  const isolation = (m.isolation || []).map((r) => el("div", { class: "kv" }, el("span", { class: "k", text: r.phase }),
    el("span", { class: "v", text: r.clean ? "live tree unchanged" : `BREACH: ${r.contamination.join(", ")} — restored ${r.restored.join(", ")}` })));
  const events = (m.events || []).slice(-30).map((e) => el("div", { class: "tl " + (e.phase === "FAILED" ? "bad" : e.phase === "DONE" ? "ok" : "work") },
    el("span", { class: "when", text: clockOf(e.at) }), el("span", { class: "text", text: `${e.phase}: ${e.detail || ""}` }), e.error ? el("span", { class: "sub", text: e.error }) : null));
  const actions = el("div", { class: "toolbar" });
  if (!isFinished(m) && m.phase !== "RESTARTING") actions.append(button("Cancel", async () => { await api("/api/selfdev/cancel", { mission_id: m.mission_id }); }, "ghost danger"));
  if (m.outcome === "failed" && m.verification?.ok) actions.append(button("Resume (verified candidate)", async () => { await api("/api/selfdev/resume", { mission_id: m.mission_id }); }, "primary"));
  if (m.evidence_patch) actions.append(button("Diff (kept)", () => showDiff(m)));
  if (m.expected_revision) actions.append(button("Version", () => views.open("release")));
  views.inspect(`Mission ${m.mission_id}`,
    section("Requested modification", el("div", { class: "kv" }, el("span", { class: "v", text: m.request }))),
    section("State", kv("phase", m.phase), kv("outcome", m.outcome || "running"), kv("reason", m.reason), kv("started", m.started_at), kv("updated", m.updated_at),
      kv("baseline → candidate", m.expected_revision ? `→ ${m.expected_revision.slice(0, 12)}` : ""), kv("area", m.area)),
    m.routing ? section("Routing", kv("top level", `${m.routing.top_level} · ${m.routing.confidence}`), kv("reason", m.routing.reason)) : null,
    section("Executors", kv("BUILD_LOCAL attempts", m.local_attempts), kv("model calls", m.model_calls), kv("expert", m.escalated ? `${m.expert?.provider || "expert"} · ${m.expert?.status || ""} · ${m.expert?.seconds || ""}s` : "not used"),
      kv("timings", Object.entries(m.timings || {}).map(([k, v]) => `${k} ${v}s`).join(" · "))),
    section("Files changed", kv("files", (m.changed_files || []).join("\n") || "none")),
    acceptance.length ? section("Acceptance", ...acceptance) : null,
    checks.length ? section("Verifier", ...checks, kv("tests", (m.verification?.tests || []).join(", "))) : null,
    isolation.length ? section("Isolation", ...isolation) : null,
    m.promotion?.promotion_id ? section("Promotion", kv("id", m.promotion.promotion_id), kv("outcome", m.promotion.outcome), kv("revision", m.promotion.promoted_revision)) : null,
    section("Workstream", el("div", { class: "timeline" }, ...events)),
    actions,
  );
}

async function showDiff(m) {
  const r = await api("/api/selfdev/diff", { mission_id: m.mission_id });
  const node = el("div", { class: "diff" });
  for (const line of (r.patch || r.error || "").split("\n")) {
    node.append(el("div", { class: line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "", text: line }));
  }
  views.inspect(`Diff ${m.mission_id}`, node, el("div", { class: "toolbar" }, button("Back", () => inspect(m))));
}
