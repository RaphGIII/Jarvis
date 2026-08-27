/* Mission Control: every long-running autonomous job — self-development
   missions, composed engine missions and capability acquisitions — from the
   unified /api/missions. Attempts at the same request are one family with an
   attempt strip; PHASE, RESULT and DEPLOYMENT are three concepts rendered
   apart (no more "CANCELLED CANCELLED"); titles are concise and durable,
   the owner's full prompt stays in the deep view. Real mission files only. */

import { el, clear, kv, section, badge, button, ago, seconds, clockOf } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";

const PHASES = {
  selfdev: ["UNDERSTAND", "INVESTIGATE", "BUILD", "VERIFY", "ESCALATE", "PROMOTE", "RESTARTING", "DONE"],
  engine: ["CREATED", "UNDERSTAND", "PLAN", "EXECUTE", "VERIFY", "DIAGNOSE", "COMPLETE"],
  acquisition: ["UNDERSTAND", "SPECIFY", "BUILD", "VERIFY", "PROMOTE", "DONE"],
};
const FILTERS = [["all", "All"], ["active", "Active"], ["waiting", "Waiting"], ["blocked", "Blocked"], ["paused", "Paused"], ["failed", "Failed"], ["cancelled", "Cancelled"], ["completed", "Completed"]];
const STATE_TONE = { active: "active", waiting: "warn", blocked: "bad", paused: "dim", failed: "bad", cancelled: "dim", completed: "ok" };
const isFinished = (m) => m.finished || ["completed", "failed", "cancelled"].includes(m.state);

export const view = {
  id: "missions",
  title: "Mission Control",
  async mount(pane, params) {
    const tabs = el("div", { class: "toolbar" });
    const list = el("div");
    let filter = params.filter || "all";
    let missions = [];
    const load = async () => { const data = await api("/api/missions", {}); missions = data.missions || []; };
    await load();
    const render = () => {
      clear(list);
      const shown = missions.filter((m) => filter === "all" || m.state === filter);
      if (!shown.length) list.append(el("div", { class: "empty", text: missions.length ? "Nothing in this filter." : "No missions yet. Say „Zeus, ändere …“ and one appears here." }));
      for (const fam of families(shown)) list.append(card(fam, (m) => views.open("missions", { ...params, mission: m.id })));
      const counts = FILTERS.slice(1).map(([k]) => `${missions.filter((m) => m.state === k).length} ${k}`).filter((t) => !t.startsWith("0 ")).join(" · ");
      status.textContent = `${counts || "nothing"} · ${missions.length} total`;
    };
    const status = el("span", { class: "empty", style: { padding: 0 } });
    for (const [key, label] of FILTERS) {
      tabs.append(el("button", { class: "ghost", "aria-pressed": filter === key ? "true" : "false", text: label, onClick: () => { filter = key; for (const b of tabs.querySelectorAll("button")) b.setAttribute("aria-pressed", b.textContent === label ? "true" : "false"); render(); } }));
    }
    tabs.append(status);
    pane.append(tabs, list);
    render();
    if (params.mission) {
      const m = missions.find((x) => x.id === params.mission);
      if (m) inspect(m);
    }
    view._sub = bus.on("progress", async () => { await load(); render(); });
  },
  unmount() { view._sub?.(); },
};

/* Attempts at the same request (same normalised sentence) become one family:
   the newest attempt is the card, the older ones an attempt strip. */
function families(missions) {
  const byFamily = new Map();
  for (const m of missions) {
    const key = m.family || m.id;
    if (!byFamily.has(key)) byFamily.set(key, []);
    byFamily.get(key).push(m);
  }
  return [...byFamily.values()].map((attempts) => attempts.sort((a, b) => String(b.updated || "").localeCompare(String(a.updated || ""))));
}

function phaseBar(m) {
  const phases = PHASES[m.system] || PHASES.engine;
  const idx = phases.indexOf(String(m.phase || "").toUpperCase());
  const pct = m.state === "completed" ? 100 : idx >= 0 ? ((idx + 1) / phases.length) * 100 : 8;
  return el("div", { class: "bar " + (isFinished(m) ? "" : "green"), style: { marginTop: "8px" } }, el("i", { style: { width: `${pct}%` } }));
}

function card(attempts, open) {
  const m = attempts[0];
  const node = el("div", { class: "card click" },
    el("div", { class: "title", text: m.title || m.goal || "(no request)" }),
    el("div", { class: "meta" }, badge(m.system || m.kind, "blue"), el("span", { text: "phase" }), badge(m.phase || "?", isFinished(m) ? "dim" : "active"),
      el("span", { text: "result" }), badge(m.state, STATE_TONE[m.state] || "dim"),
      m.deployment ? [el("span", { text: "deployment" }), badge(m.deployment, m.deployment === "promoted" ? "ok" : "bad")] : null,
      m.tasks?.total ? el("span", { text: `${m.tasks.done}/${m.tasks.total} tasks` }) : null,
      el("span", { text: `${m.evidence || 0} evidence` }), el("span", { text: ago(m.updated) })),
    m.next_action ? el("div", { class: "empty", style: { padding: "4px 0 0" }, text: `next: ${m.next_action}` }) : null,
    (m.blockers || []).length ? el("div", { class: "empty", style: { padding: "4px 0 0", color: "var(--red)" }, text: `blocked: ${m.blockers.join("; ")}` }) : null,
    phaseBar(m));
  if (attempts.length > 1) {
    const strip = el("div", { class: "meta", style: { marginTop: "6px" } }, el("span", { text: `${attempts.length} attempts:` }));
    attempts.slice().reverse().forEach((a, i) => {
      const b = el("button", { class: "ghost", style: { padding: "0 6px", fontSize: "10px" }, text: `#${i + 1} ${a.state}` });
      b.onclick = (e) => { e.stopPropagation(); open(a); };
      strip.append(b);
    });
    node.append(strip);
  }
  node.onclick = () => open(m);
  return node;
}

async function inspect(row) {
  const detail = await api("/api/mission", { id: row.id });
  if (!detail.ok) { views.inspect(`Mission ${row.id}`, el("div", { class: "empty", text: detail.error || "no detail" })); return; }
  if (detail.system === "selfdev") return inspectSelfdev(detail.mission, row);
  const m = detail.mission || {};
  const tasks = (m.tasks || []).map((t) => el("div", { class: "kv" }, el("span", { class: "k", text: t.status || "?" }), el("span", { class: "v", text: `${t.title || t.task_id || ""}${t.result ? " — " + String(t.result).slice(0, 100) : ""}` })));
  const evidence = (m.evidence || []).slice(-20).map((e) => el("div", { class: "kv" }, el("span", { class: "k", text: e.kind || e.source || "evidence" }), el("span", { class: "v", text: (e.summary || e.detail || e.text || JSON.stringify(e)).slice(0, 160) })));
  const history = (m.history || m.transitions || m.events || []).slice(-30).map((e) => el("div", { class: "tl " + (String(e.phase).includes("FAIL") ? "bad" : String(e.phase).includes("COMPLETE") ? "ok" : "work") },
    el("span", { class: "when", text: clockOf(e.at) }), el("span", { class: "text", text: `${e.phase || ""}: ${e.detail || ""}` })));
  const actions = el("div", { class: "toolbar" });
  if (!isFinished(row)) {
    actions.append(button("Pause", () => api("/api/mission/pause", { mission_id: row.id })));
    actions.append(button("Cancel", () => api("/api/mission/cancel", { mission_id: row.id }), "ghost danger"));
  }
  if (row.state === "paused" || row.state === "blocked") actions.append(button("Resume", () => api("/api/mission/resume", { mission_id: row.id }), "primary"));
  views.inspect(row.title || `Mission ${row.id}`,
    section("Goal", el("div", { class: "kv" }, el("span", { class: "v", text: m.goal || row.goal || "" })), kv("interpretation", m.interpretation)),
    (m.constraints || []).length ? section("Constraints", ...m.constraints.map((c) => kv("must", c))) : null,
    (m.acceptance_criteria || []).length ? section("Acceptance criteria", ...m.acceptance_criteria.map((c) => kv("criterion", c))) : null,
    section("State", kv("phase", m.phase), kv("result", row.state), kv("outcome", m.outcome || "running"), kv("next action", m.next_action),
      kv("blocker", (m.blockers || []).join("; ")), kv("owner input", m.owner_input_required), kv("attempts of this request", row.attempts),
      kv("started", m.created_at), kv("updated", m.updated_at)),
    tasks.length ? section("Tasks", ...tasks) : null,
    evidence.length ? section("Evidence", ...evidence) : null,
    detail.brief ? section("Brief", el("pre", { class: "code", text: typeof detail.brief === "string" ? detail.brief : JSON.stringify(detail.brief, null, 1) })) : null,
    history.length ? section("Workstream", el("div", { class: "timeline" }, ...history)) : null,
    actions,
  );
}

function inspectSelfdev(m, row) {
  const acceptance = (m.acceptance || []).map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: "criterion" }), el("span", { class: "v", text: a.criterion })));
  const checks = (m.verification?.checks || []).map((c) => el("div", { class: "kv" }, el("span", { class: "k", text: c.ok ? "✓" : "✗" }), el("span", { class: "v", text: c.criterion })));
  const isolation = (m.isolation || []).map((r) => el("div", { class: "kv" }, el("span", { class: "k", text: r.phase }),
    el("span", { class: "v", text: r.clean ? "live tree unchanged" : `BREACH: ${r.contamination.join(", ")} — restored ${r.restored.join(", ")}` })));
  const events = (m.events || []).slice(-30).map((e) => el("div", { class: "tl " + (e.phase === "FAILED" ? "bad" : e.phase === "DONE" ? "ok" : "work") },
    el("span", { class: "when", text: clockOf(e.at) }), el("span", { class: "text", text: `${e.phase}: ${e.detail || ""}` }), e.error ? el("span", { class: "sub", text: e.error }) : null));
  const finished = ["DONE", "FAILED", "CANCELLED"].includes(m.phase);
  const actions = el("div", { class: "toolbar" });
  if (!finished && m.phase !== "RESTARTING") actions.append(button("Cancel", async () => { await api("/api/selfdev/cancel", { mission_id: m.mission_id }); }, "ghost danger"));
  if (m.outcome === "failed" && m.verification?.ok) actions.append(button("Resume (verified candidate)", async () => { await api("/api/selfdev/resume", { mission_id: m.mission_id }); }, "primary"));
  if (m.evidence_patch) actions.append(button("Diff (kept)", () => showDiff(m, row)));
  if (m.expected_revision) actions.append(button("Version", () => views.open("release")));
  views.inspect(row.title || `Mission ${m.mission_id}`,
    section("Requested modification (the owner's words)", el("div", { class: "kv" }, el("span", { class: "v", text: m.request }))),
    section("State", kv("phase", m.phase), kv("result", row.state), kv("deployment", row.deployment || "not deployed"), kv("reason", m.reason),
      kv("attempts of this request", row.attempts), kv("started", m.started_at), kv("updated", m.updated_at),
      kv("baseline → candidate", m.expected_revision ? `→ ${m.expected_revision.slice(0, 12)}` : ""), kv("area", m.area)),
    m.routing ? section("Routing", kv("top level", `${m.routing.top_level} · ${m.routing.confidence}`), kv("reason", m.routing.reason)) : null,
    section("Executors", kv("BUILD_LOCAL attempts", m.local_attempts), kv("model calls", m.model_calls), kv("expert", m.escalated ? `${m.expert?.provider || "expert"} · ${m.expert?.status || ""} · ${m.expert?.seconds || ""}s` : "not used"),
      kv("timings", Object.entries(m.timings || {}).map(([k, v]) => `${k} ${v}s`).join(" · "))),
    section("Files changed", kv("files", (m.changed_files || []).join("\n") || "none")),
    acceptance.length ? section("Acceptance", ...acceptance) : null,
    checks.length ? section("Verifier", ...checks, kv("tests", (m.verification?.tests || []).join(", "))) : null,
    m.verification_goal ? section("Goal check", kv("verdict", typeof m.verification_goal === "string" ? m.verification_goal : JSON.stringify(m.verification_goal).slice(0, 200))) : null,
    isolation.length ? section("Isolation", ...isolation) : null,
    m.promotion?.promotion_id ? section("Promotion", kv("id", m.promotion.promotion_id), kv("outcome", m.promotion.outcome), kv("revision", m.promotion.promoted_revision)) : null,
    section("Workstream", el("div", { class: "timeline" }, ...events)),
    actions,
  );
}

async function showDiff(m, row) {
  const r = await api("/api/selfdev/diff", { mission_id: m.mission_id });
  const node = el("div", { class: "diff" });
  for (const line of (r.patch || r.error || "").split("\n")) {
    node.append(el("div", { class: line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "", text: line }));
  }
  views.inspect(`Diff ${m.mission_id}`, node, el("div", { class: "toolbar" }, button("Back", () => inspectSelfdev(m, row))));
}
