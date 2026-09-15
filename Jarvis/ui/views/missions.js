/* Mission Control: every long-running autonomous job — self-development
   missions, composed engine missions and capability acquisitions — from the
   unified /api/missions. Attempts at the same request are one family with an
   attempt strip; PHASE, RESULT and DEPLOYMENT are three concepts rendered
   apart (no more "CANCELLED CANCELLED"); titles are concise and durable,
   the owner's full prompt stays in the deep view. Real mission files only. */

import { el, clear, kv, section, badge, button, ago, seconds, clockOf } from "../core/dom.js";
import { api } from "../core/api.js";
import { withAuth } from "../core/authgate.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";

const PHASES = {
  selfdev: ["UNDERSTAND", "INVESTIGATE", "SURVEY", "ENGINEER", "BUILD", "ESCALATE", "VERIFY", "PROMOTE", "RESTARTING", "DONE"],
  engine: ["CREATED", "UNDERSTAND", "PLAN", "EXECUTE", "VERIFY", "DIAGNOSE", "COMPLETE"],
  acquisition: ["UNDERSTAND", "SPECIFY", "BUILD", "VERIFY", "PROMOTE", "DONE"],
};
const FILTERS = [["all", "Alle"], ["active", "Aktiv"], ["waiting", "Warten"], ["blocked", "Blockiert"], ["paused", "Pausiert"], ["failed", "Gescheitert"], ["cancelled", "Abgebrochen"], ["completed", "Abgeschlossen"]];
const STATE_TONE = { active: "active", waiting: "warn", blocked: "bad", paused: "dim", failed: "bad", cancelled: "dim", completed: "ok" };
/* Owner-facing words for the work.  Internal phase names never reach the card. */
const STATE_WORD = { active: "Aktiv", waiting: "Warten", blocked: "Freigabe erforderlich", paused: "Pausiert", failed: "Gescheitert", cancelled: "Abgebrochen", completed: "Abgeschlossen" };
const PHASE_WORD = {
  UNDERSTAND: "Analyse", INVESTIGATE: "Research", SURVEY: "Research", SPECIFY: "Analyse", PLAN: "Analyse", DIAGNOSE: "Analyse", CREATED: "Warten",
  ENGINEER: "Entwicklung", BUILD: "Entwicklung", EXECUTE: "Entwicklung", ESCALATE: "Entwicklung", VERIFY: "Verifikation", PROMOTE: "Abschluss",
  RESTARTING: "Abschluss", DONE: "Abgeschlossen", COMPLETE: "Abgeschlossen", AWAITING_AUTHORIZATION: "Freigabe erforderlich", AWAITING_BUILD: "Freigabe erforderlich",
  WAITING: "Warten", FAILED: "Gescheitert", CANCELLED: "Abgebrochen",
};
const phaseWord = (m) => m.state === "completed" ? "Abgeschlossen" : m.owner_input_required ? "Freigabe erforderlich" : PHASE_WORD[String(m.phase || "").toUpperCase()] || STATE_WORD[m.state] || "Arbeit";
const SYSTEM_WORD = { selfdev: "Entwicklung", engine: "Auftrag", acquisition: "Neue Fähigkeit" };
const isFinished = (m) => m.finished || ["completed", "failed", "cancelled"].includes(m.state);

export const view = {
  id: "missions",
  title: "Missionen",
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
      if (!shown.length) list.append(el("div", { class: "empty", text: missions.length ? "Nichts in dieser Auswahl." : "Noch keine Missionen. Gib ZEUS eine längere Aufgabe, und sie erscheint hier." }));
      for (const fam of families(shown)) list.append(card(fam, (m) => views.open("missions", { ...params, mission: m.id })));
      const counts = FILTERS.slice(1).map(([k, label]) => `${missions.filter((m) => m.state === k).length} ${label.toLowerCase()}`).filter((t) => !t.startsWith("0 ")).join(" · ");
      status.textContent = counts || "";
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
    el("div", { class: "title", text: m.title || m.goal || "Auftrag" }),
    el("div", { class: "meta" }, badge(phaseWord(m), isFinished(m) ? (m.state === "completed" ? "ok" : STATE_TONE[m.state] || "dim") : "active"),
      el("span", { text: SYSTEM_WORD[m.system] || "" }),
      m.tasks?.total ? el("span", { text: `${m.tasks.done}/${m.tasks.total} Schritte` }) : null,
      el("span", { text: ago(m.updated) })),
    m.next_action && !isFinished(m) ? el("div", { class: "empty", style: { padding: "4px 0 0" }, text: m.next_action }) : null,
    (m.blockers || []).length && !isFinished(m) ? el("div", { class: "empty", style: { padding: "4px 0 0" }, text: "Wartet auf dich – Details in der Mission." }) : null,
    phaseBar(m));
  if (attempts.length > 1) {
    const strip = el("div", { class: "meta", style: { marginTop: "6px" } }, el("span", { text: `${attempts.length} Anläufe:` }));
    attempts.slice().reverse().forEach((a, i) => {
      const b = el("button", { class: "ghost", style: { padding: "0 6px", fontSize: "10px" }, text: `#${i + 1} ${STATE_WORD[a.state] || a.state}` });
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
  if (!detail.ok) { views.inspect(row.title || "Mission", el("div", { class: "empty", text: "Zu dieser Mission liegen gerade keine Details vor." })); return; }
  if (detail.system === "selfdev") return inspectSelfdev(detail.mission, row);
  const m = detail.mission || {};
  const tasks = (m.tasks || []).map((t) => el("div", { class: "kv" }, el("span", { class: "k", text: t.status || "?" }), el("span", { class: "v", text: `${t.title || t.task_id || ""}${t.result ? " — " + String(t.result).slice(0, 100) : ""}` })));
  const evidence = (m.evidence || []).slice(-20).map((e) => el("div", { class: "kv" }, el("span", { class: "k", text: e.kind || e.source || "evidence" }), el("span", { class: "v", text: (e.summary || e.detail || e.text || JSON.stringify(e)).slice(0, 160) })));
  const history = (m.history || m.transitions || m.events || []).slice(-30).map((e) => el("div", { class: "tl " + (String(e.phase).includes("FAIL") ? "bad" : String(e.phase).includes("COMPLETE") ? "ok" : "work") },
    el("span", { class: "when", text: clockOf(e.at) }), el("span", { class: "text", text: `${e.phase || ""}: ${e.detail || ""}` })));
  const actions = el("div", { class: "toolbar" });
  if (!isFinished(row)) {
    actions.append(button("Pausieren", () => api("/api/mission/pause", { mission_id: row.id })));
    actions.append(button("Abbrechen", () => api("/api/mission/cancel", { mission_id: row.id }), "ghost danger"));
  }
  if (row.state === "paused" || row.state === "blocked") actions.append(button("Fortsetzen", () => api("/api/mission/resume", { mission_id: row.id }), "primary"));
  views.inspect(row.title || "Mission",
    section("Ziel", el("div", { class: "kv" }, el("span", { class: "v", text: m.goal || row.goal || "" })), kv("interpretation", m.interpretation)),
    (m.constraints || []).length ? section("Constraints", ...m.constraints.map((c) => kv("must", c))) : null,
    (m.acceptance_criteria || []).length ? section("Acceptance criteria", ...m.acceptance_criteria.map((c) => kv("criterion", c))) : null,
    section("State", kv("phase", m.phase), kv("result", row.state), kv("outcome", m.outcome || "running"), kv("next action", m.next_action),
      kv("blocker", (m.blockers || []).join("; ")), kv("owner input", m.owner_input_required), kv("attempts of this request", row.attempts),
      kv("started", m.created_at), kv("updated", m.updated_at)),
    tasks.length ? section("Tasks", ...tasks) : null,
    evidence.length ? section("Evidence", ...evidence) : null,
    detail.brief ? section("Brief", el("pre", { class: "code", text: typeof detail.brief === "string" ? detail.brief : JSON.stringify(detail.brief, null, 1) })) : null,
    history.length ? section("Verlauf", el("div", { class: "timeline" }, ...history)) : null,
    actions,
    el("details", { class: "padv" }, el("summary", { text: "Technische Details" }), kv("mission", row.id), kv("phase", m.phase), kv("outcome", m.outcome || "running")),
  );
}

function inspectSelfdev(m, row) {
  const acceptance = (m.acceptance || []).map((a) => el("div", { class: "kv" }, el("span", { class: "k", text: "criterion" }), el("span", { class: "v", text: a.criterion })));
  const checks = (m.verification?.checks || []).map((c) => el("div", { class: "kv" }, el("span", { class: "k", text: c.ok ? "✓" : "✗" }),
    el("span", { class: "v", text: c.ok ? c.criterion : `${c.criterion} — ${String(c.output || "").trim().slice(0, 240)}` })));
  const isolation = (m.isolation || []).map((r) => el("div", { class: "kv" }, el("span", { class: "k", text: r.phase }),
    el("span", { class: "v", text: r.clean ? "live tree unchanged" : `BREACH: ${r.contamination.join(", ")} — restored ${r.restored.join(", ")}` })));
  const events = (m.events || []).slice(-30).map((e) => el("div", { class: "tl " + (e.phase === "FAILED" ? "bad" : e.phase === "DONE" ? "ok" : "work") },
    el("span", { class: "when", text: clockOf(e.at) }), el("span", { class: "text", text: `${e.phase}: ${e.detail || ""}` }), e.error ? el("span", { class: "sub", text: e.error }) : null));
  const finished = ["DONE", "FAILED", "CANCELLED", "AWAITING_AUTHORIZATION", "AWAITING_BUILD", "WAITING"].includes(m.phase);
  const actions = el("div", { class: "toolbar" });
  /* A metered engineer costs money. The mission shows the estimate and the
     hard maximum and spends nothing until the owner starts the build here. */
  if (m.phase === "AWAITING_BUILD") {
    const eng = m.engineering || {};
    const rng = eng.estimate_range_eur || [0, 0];
    const eur = (v) => "€" + Number(v || 0).toFixed(2);
    actions.append(el("span", { class: "meta", text: `${eng.role || "?"} · geschätzt ${eur(rng[0])}–${eur(rng[1])} · hartes Maximum ${eur(eng.hard_max_eur)}` }));
    actions.append(button(`Build starten (max. ${eur(eng.hard_max_eur)})`, async () => {
      if (!confirm(`Diesen Build mit ${eng.role || "dem Engineer"} starten?\n\nGeschätzt ${eur(rng[0])}–${eur(rng[1])}, hartes Maximum ${eur(eng.hard_max_eur)}.\nDas Ergebnis wird verifiziert und erst nach deinem Passwort übernommen.`)) return;
      const r = await api("/api/selfdev/build", { mission_id: m.mission_id });
      alert(r.ok ? "Build gestartet." : (r.error || "?"));
      views.open("missions");
    }, "primary"));
  }
  /* The owner's half of the promotion gate, and the only way through it. The
     change is engineered, tested and verified in its isolated worktree and
     goes no further until a password typed HERE mints a short-lived, scoped,
     single-use SELFDEV_PROMOTE token. Nothing said in the chat, and nothing
     any model reports, opens this. */
  if (m.phase === "AWAITING_AUTHORIZATION") {
    actions.append(button("Freigeben und übernehmen (Passwort)", async () => {
      if (!confirm(`Diese Änderung in das laufende Produkt übernehmen und neu starten?\n\n${(m.changed_files || []).join("\n")}`)) return;
      const r = await withAuth("SELFDEV_PROMOTE", (authorization) =>
        api("/api/selfdev/authorize", { mission_id: m.mission_id, authorization }));
      alert(r.needs_setup ? r.error
            : r.needs_auth ? "Abgebrochen – ohne Passwort keine Freigabe."
            : r.ok ? "Freigegeben. Übernahme läuft, danach startet ZEUS neu."
            : (r.error || "?"));
      views.open("missions");
    }, "primary"));
  }
  if (!finished && m.phase !== "RESTARTING") actions.append(button("Abbrechen", async () => { await api("/api/selfdev/cancel", { mission_id: m.mission_id }); }, "ghost danger"));
  if (m.outcome === "failed" && m.verification?.ok) actions.append(button("Fortsetzen (geprüfter Kandidat)", async () => { await api("/api/selfdev/resume", { mission_id: m.mission_id }); }, "primary"));
  if (m.evidence_patch) actions.append(button("Änderungen ansehen", () => showDiff(m, row)));
  if (m.expected_revision) actions.append(button("Version", () => views.open("release")));
  const c = m.control || {};
  const already = (c.ALREADY_IMPLEMENTED || []).map((line) => el("div", { class: "kv" },
    el("span", { class: "k", text: "exists" }), el("span", { class: "v", text: String(line).slice(0, 200) })));
  views.inspect(row.title || "Mission",
    section("Requested modification (the owner's words)", el("div", { class: "kv" }, el("span", { class: "v", text: m.request }))),
    section("Auftrag",
      kv("goal", c.GOAL || m.request),
      kv("route", c.ROUTE),
      kv("engineer", c.ENGINEER || "not chosen yet"),
      kv("lokale Bauversuche", String(c.BUILD_LOCAL_INVOCATIONS ?? m.local_attempts ?? 0)),
      kv("current phase", c.PHASE || m.phase),
      kv("files changed", (c.FILES_CHANGED || []).join(", ") || "none"),
      kv("tests", (c.TESTS || []).join(", ") || "none"),
      kv("failure reason", c.FAILURE_REASON || ""),
      kv("candidate", c.CANDIDATE || "none"),
      kv("awaiting owner", c.AWAITING_OWNER ? "yes — the password promotes it, nothing else does" : "no"),
      kv("result", c.RESULT || m.outcome || "running")),
    already.length ? section("Already implemented before this mission started", ...already) : null,
    section("State", kv("phase", m.phase), kv("result", row.state), kv("deployment", row.deployment || "not deployed"), kv("reason", m.reason),
      kv("attempts of this request", row.attempts), kv("started", m.started_at), kv("updated", m.updated_at),
      kv("baseline → candidate", m.expected_revision ? `→ ${m.expected_revision.slice(0, 12)}` : ""), kv("area", m.area)),
    m.routing ? section("Routing", kv("top level", `${m.routing.top_level} · ${m.routing.confidence}`), kv("reason", m.routing.reason)) : null,
    section("Ausführung", kv("lokale Bauversuche", m.local_attempts), kv("Modellaufrufe", m.model_calls), kv("Verstärkung", m.escalated ? `${m.expert?.status || "eingesetzt"} · ${m.expert?.seconds || ""}s` : "nicht eingesetzt"),
      kv("timings", Object.entries(m.timings || {}).map(([k, v]) => `${k} ${v}s`).join(" · "))),
    section("Files changed", kv("files", (m.changed_files || []).join("\n") || "none")),
    acceptance.length ? section("Acceptance", ...acceptance) : null,
    checks.length ? section("Verifier", ...checks, kv("tests", (m.verification?.tests || []).join(", "))) : null,
    m.verification_goal ? section("Goal check", kv("verdict", typeof m.verification_goal === "string" ? m.verification_goal : JSON.stringify(m.verification_goal).slice(0, 200))) : null,
    isolation.length ? section("Isolation", ...isolation) : null,
    m.promotion?.promotion_id ? section("Promotion", kv("id", m.promotion.promotion_id), kv("outcome", m.promotion.outcome), kv("revision", m.promotion.promoted_revision)) : null,
    section("Verlauf", el("div", { class: "timeline" }, ...events)),
    actions,
    el("details", { class: "padv" }, el("summary", { text: "Technische Details" }), kv("mission", m.mission_id), kv("row", row.id)),
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
