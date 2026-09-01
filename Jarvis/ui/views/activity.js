/* Activity: the owner's transparent operation log, request-centric. Rows
   come from the durable log unchanged; the browser only *groups* them: every
   `request` row opens a group that holds what ZEUS did for it (routing, plan,
   actions, failures, replan, the final goal verdict) until the next request.
   Identical consecutive requests (a mis-heard "Toys." five times) collapse
   into one group with a repeat count. Expand a group for the technical view:
   timestamps, receipt ids, executor, duration, targets, evidence. */

import { $, el, clear, clockOf, dateOf, kv, section, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import { state, setPref } from "../core/state.js";
import * as views from "../core/views.js";

const KINDS = {
  "request": { label: "REQUEST", cls: "req", icon: "›" },
  "wake": { label: "WAKE", cls: "tool", icon: "◉" },
  "voice.accepted": { label: "VOICE", cls: "ok", icon: "🎙" },
  "voice.rejected": { label: "VOICE REJECTED", cls: "warn", icon: "🎙" },
  "answer": { label: "ANSWER", cls: "ans", icon: "‹" },
  "action.verified": { label: "VERIFIED", cls: "ok", icon: "✓" },
  "action.ran": { label: "UNVERIFIED", cls: "warn", icon: "○" },
  "action.failed": { label: "FAILED", cls: "bad", icon: "✗" },
  "tool": { label: "TOOL", cls: "tool", icon: "⌁" },
  "state.working": { label: "WORKING", cls: "work", icon: "⚙" },
  "state.verifying": { label: "VERIFYING", cls: "work", icon: "⚙" },
  "state.researching": { label: "RESEARCH", cls: "tool", icon: "⌕" },
  "state.coding": { label: "CODING", cls: "work", icon: "⌁" },
  "state.waiting": { label: "WAITING", cls: "warn", icon: "…" },
  "state.error": { label: "ERROR", cls: "bad", icon: "✗" },
  "error": { label: "ERROR", cls: "bad", icon: "✗" },
  "progress": { label: "PROGRESS", cls: "tool", icon: "⚙" },
  "notification": { label: "NOTE", cls: "tool", icon: "•" },
  "diagnostic": { label: "DIAG", cls: "tool", icon: "•" },
  "knowledge": { label: "KNOWLEDGE", cls: "tool", icon: "◆" },
};

let live = null;

export const view = {
  id: "activity",
  title: "Activity",
  async mount(pane, params) {
    const toolbar = el("div", { class: "toolbar" });
    const search = el("input", { placeholder: "Filter…", value: params.q || "" });
    const kind = el("select", {}, el("option", { value: "", text: "all kinds" }),
      ...Object.entries(KINDS).map(([k, m]) => el("option", { value: k, text: m.label.toLowerCase() })));
    const grouped = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } },
      el("input", { type: "checkbox", checked: state.ui.groupActivity !== false, onChange: (e) => { setPref("groupActivity", e.target.checked); render(); } }),
      " group by request");
    const echo = el("label", { class: "empty", style: { padding: 0, cursor: "pointer" } },
      el("input", { type: "checkbox", checked: state.ui.echoActions, onChange: (e) => setPref("echoActions", e.target.checked) }),
      " echo actions into the conversation");
    toolbar.append(search, kind, grouped, echo);
    const list = el("div", { class: "activity" });
    pane.append(toolbar, list);

    const data = await api("/api/activity", { limit: 600 });
    let entries = (data.activity || []).slice();
    const render = () => {
      clear(list);
      const q = search.value.toLowerCase();
      if (data.error) { list.append(el("div", { class: "empty", text: `The activity log could not be read: ${data.error}` })); return; }
      const matches = (e) => (!kind.value || e.kind === kind.value) && (!q || `${e.summary} ${e.kind} ${e.receipt_id}`.toLowerCase().includes(q));
      const useGroups = state.ui.groupActivity !== false && !kind.value;
      const items = useGroups ? groups(entries).filter((g) => g.rows.some(matches)) : entries.filter(matches).map((e) => ({ head: e, rows: [e], repeats: 1 }));
      items.reverse();
      if (!items.length) { list.append(el("div", { class: "empty", text: "Nothing recorded yet. Every request, action and verification appears here." })); return; }
      let day = "";
      for (const g of items.slice(0, 400)) {
        const d = dateOf(g.head.at);
        if (d !== day) { day = d; list.append(el("h4", { class: "day", text: d, style: { margin: "14px 0 6px", fontSize: "10px", letterSpacing: ".2em", color: "var(--faint)" } })); }
        list.append(useGroups && g.rows.length > 1 ? group(g, params) : row(g.head, params.receipt && g.head.receipt_id === params.receipt));
      }
    };
    search.oninput = render;
    kind.onchange = render;
    render();
    live = bus.on("*", () => {});
    view._refresh = async () => {
      const fresh = await api("/api/activity", { limit: 600 });
      entries = (fresh.activity || []).slice();
      render();
    };
    view._sub = bus.on("tool", () => view._refresh());
  },
  unmount() {
    view._sub?.();
    live?.();
  },
};

/* Chronological rows -> groups. A `request` row starts a group; everything
   until the next request belongs to it. Consecutive groups with the same
   request text and the same answer shape collapse into one with `repeats`. */
function groups(entries) {
  const out = [];
  let current = null;
  for (const e of entries) {
    // A rejected utterance is its own group: it produced no request, and it
    // must not be filed under the previous one.  An accepted utterance's
    // trace arrives just *before* its request row: it opens the group, and
    // the request row that follows becomes the group's head.
    if (e.kind === "request" && current && current.pendingVoice) {
      current.head = e; current.rows.push(e); current.key = norm(e.summary); current.pendingVoice = false;
      continue;
    }
    if (e.kind === "request" || e.kind === "voice.rejected" || e.kind === "voice.accepted" || !current) {
      current = { head: e, rows: [e], repeats: 1, key: e.kind === "request" ? norm(e.summary) : e.kind === "voice.rejected" ? "rejected:" + norm(e.summary) : "",
                  pendingVoice: e.kind === "voice.accepted" };
      out.push(current);
    } else {
      current.rows.push(e);
    }
  }
  const merged = [];
  for (const g of out) {
    const last = merged[merged.length - 1];
    if (last && g.key && g.key === last.key && shape(g) === shape(last)) { last.repeats += 1; last.last = g.head.at; continue; }
    merged.push(g);
  }
  return merged;
}
const norm = (t) => String(t || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
const shape = (g) => g.rows.map((r) => r.kind).join(",");

function verdict(g) {
  const goal = g.rows.find((r) => r.kind === "tool" && /^goal:/.test(r.summary || ""));
  if (goal) return /SATISFIED/.test(goal.summary) && !/NOT satisfied/.test(goal.summary) ? ["GOAL SATISFIED", "ok"] : ["GOAL NOT MET", "bad"];
  if (g.rows.some((r) => r.kind === "action.failed" || r.kind === "state.error" || r.kind === "error")) return ["FAILED", "bad"];
  if (g.rows.some((r) => r.kind === "action.verified")) return ["VERIFIED", "ok"];
  if (g.rows.some((r) => r.kind === "answer")) return ["ANSWERED", "ans"];
  return ["", ""];
}

/* Append-only owner corrections: the original stays; the edit is a new
   record; the STT lexicon learns from transcript edits. */
function correctionTools(entry) {
  const meta = (entry.detail || {}).meta || {};
  const wrap = el("span", { class: "act-tools" });
  const link = (text, title) => el("a", { href: "#", class: "act-tool-link", title, text });

  const editor = (type, placeholder, value) => (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    wrap.parentElement.querySelector(".act-edit")?.remove();
    const input = el("input", { value, placeholder, class: "act-edit-input" });
    const save = async (rerun) => {
      const out = await api("/api/activity/correct", { request_id: meta.request_id || "", seq: entry.seq || 0,
        type, original: entry.summary || "", corrected: input.value, rerun });
      box.replaceWith(el("span", { class: "act-corrected", text: out.ok ? ` → ${input.value}` : ` (${out.error || "nicht gespeichert"})` }));
    };
    const box = el("span", { class: "act-edit" }, input,
      el("button", { class: "chip", text: "speichern", onClick: () => save(false) }),
      el("button", { class: "chip", text: "speichern + erneut ausführen", onClick: () => save(true) }));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") save(false); if (e.key === "Escape") box.remove(); });
    wrap.after(box);
    input.focus();
  };

  const edit = link("Transkript korrigieren", "Was wurde wirklich gesagt? Das Original bleibt als Beweis stehen.");
  edit.onclick = editor("TRANSCRIPT", "so hätte es heißen sollen", entry.summary || "");
  const intent = link("Intent korrigieren", "Was war gemeint? Zeus lernt die Absicht, nicht nur die Wörter.");
  intent.onclick = editor("INTENT", "was gemeint war, z. B. „Spiel Rammstein“", "");

  const rate = link("Antwort bewerten", "Feedback zu dieser Anfrage");
  rate.onclick = (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    wrap.parentElement.querySelector(".act-edit")?.remove();
    const box = el("span", { class: "act-edit" });
    for (const [key, label] of [["TOO_SHORT", "zu kurz"], ["TOO_LONG", "zu lang"], ["MISUNDERSTOOD", "missverstanden"],
                                ["WRONG_ACTION", "falsche Aktion"], ["WRONG_FACT", "falscher Fakt"], ["OTHER", "anderes"]]) {
      box.append(el("button", { class: "chip", text: label, onClick: async () => {
        await api("/api/feedback", { kind: "response", rating: "down", category: key, request_id: meta.request_id || "" });
        box.replaceWith(el("span", { class: "act-corrected", text: " Präferenz gespeichert" }));
      } }));
    }
    wrap.after(box);
  };

  wrap.append(edit, intent, rate);
  return wrap;
}

function group(g, params) {
  const [label, cls] = verdict(g);
  const route = g.rows.find((r) => r.kind === "tool" && /^routed:/.test(r.summary || ""));
  const headMeta = KINDS[g.head.kind] || { label: "REQUEST", cls: "req", icon: "›" };
  const node = el("div", { class: `act ${g.head.kind === "request" ? "req" : headMeta.cls}` });
  const body = el("div", { class: "act-body" });
  body.hidden = true;
  const head = el("div", { class: "act-head expandable" },
    el("span", { class: "act-tag", text: `${headMeta.icon} ${headMeta.label}` + (g.repeats > 1 ? ` ×${g.repeats}` : "") }),
    el("span", { class: "act-sum", text: g.head.summary || "(no detail recorded)" }),
    g.head.kind === "request" ? correctionTools(g.head) : null,
    label ? el("span", { class: `act-tag ${cls}`, text: label }) : null,
    el("span", { class: "act-when", text: clockOf(g.head.at) }));
  head.onclick = () => { body.hidden = !body.hidden; };
  const summary = el("div", { class: "empty", style: { padding: "2px 0 4px 18px" } });
  const steps = g.rows.filter((r) => r !== g.head).filter((r) => r.kind !== "state.working" && r.kind !== "state.verifying" && r.kind !== "state.coding");
  const short = steps.map((r) => (KINDS[r.kind] || { icon: "•" }).icon + " " + String(r.summary || r.kind).slice(0, 70));
  summary.textContent = (route ? route.summary.replace(/^routed:\s*/, "→ ") + " · " : "") + short.slice(0, 4).join("  ·  ") + (short.length > 4 ? `  ·  +${short.length - 4} more` : "");
  for (const r of g.rows.filter((x) => x !== g.head)) body.append(row(r, params.receipt && r.receipt_id === params.receipt));
  node.append(head, summary, body);
  return node;
}

function row(entry, highlight) {
  const meta = KINDS[entry.kind] || { label: entry.kind.toUpperCase(), cls: "tool", icon: "•" };
  const node = el("div", { class: `act ${meta.cls}` });
  const head = el("div", { class: "act-head" },
    el("span", { class: "act-tag", text: `${meta.icon} ${meta.label}` }),
    el("span", { class: "act-sum", text: entry.summary || "(no detail recorded)" }),
    el("span", { class: "act-when", text: clockOf(entry.at) }));
  node.append(head);
  const detail = technical(entry);
  if (detail) {
    head.classList.add("expandable");
    detail.hidden = !highlight;
    head.onclick = (e) => { e.stopPropagation(); detail.hidden = !detail.hidden; inspect(entry); };
    node.append(detail);
  }
  if (highlight) node.style.outline = "1px solid var(--blue-deep)";
  return node;
}

function technical(entry) {
  const receipt = entry.receipt_id ? entry.detail : null;
  const d = entry.detail || {};
  if (!receipt && !Object.keys(d).length) return null;
  const body = el("div", { class: "act-body" });
  const line = (k, v) => { if (v !== undefined && v !== null && v !== "") body.append(el("div", { class: "act-kv" }, el("span", { class: "act-k", text: k }), el("span", { class: "act-v", text: String(v) }))); };
  line("event", entry.kind);
  line("timestamp", entry.at);
  line("scope", entry.scope);
  if (receipt) {
    const ev = receipt.evidence || {};
    line("action", receipt.kind);
    line("executor", receipt.executor);
    line("target", ev.path || ev.project_id || ev.title || ev.track || ev.query || ev.node_id || "");
    line("result", receipt.verified ? "verified" : receipt.ok ? "ran, not verified" : "failed");
    line("duration", receipt.duration_seconds !== undefined ? `${receipt.duration_seconds}s` : "");
    if (!receipt.ok) line("failure", receipt.detail);
    line("receipt", receipt.id);
    const checks = receipt.verifications || [];
    if (checks.length) {
      body.append(el("div", { class: "act-k", text: "verification evidence", style: { marginTop: "6px" } }));
      for (const c of checks) body.append(el("div", { class: "act-check" + (c.passed ? "" : " bad"), text: `${c.passed ? "✓" : "✗"} ${c.check}` + (c.observed ? ` — ${c.observed}` : "") }));
    } else {
      body.append(el("div", { class: "act-check bad", text: "✗ no verification was recorded for this action" }));
    }
    return body;
  }
  if (d.voice_trace) { voiceTrace(body, d); return body; }
  if (d.understanding) {
    line("top-level intent", d.understanding.top);
    line("why", d.understanding.reason);
    if (d.understanding.action) {
      line("operation", d.understanding.action.operation);
      line("target", d.understanding.action.target);
      line("arguments", JSON.stringify(d.understanding.action.arguments));
      line("confidence", d.understanding.action.confidence);
      if ((d.understanding.action.missing || []).length) line("missing", d.understanding.action.missing.join(", "));
    }
  }
  if (d.mission_id) line("mission", d.mission_id);
  if (d.phase) line("phase", d.phase);
  if (d.goal) { line("ACTION_EXECUTED", d.goal.ACTION_EXECUTED); line("EXECUTION_VERIFIED", d.goal.EXECUTION_VERIFIED); line("GOAL_SATISFIED", d.goal.GOAL_SATISFIED); line("reasons", (d.goal.reasons || []).join("; ")); }
  if (d.forbidden) line("forbidden steps", d.forbidden.join(", "));
  if (d.routing) {
    line("route", `${d.routing.top_level} (${d.routing.confidence})`);
    line("reason", d.routing.reason);
    if ((d.routing.conflicts || []).length) line("overruled", d.routing.conflicts.join(" | "));
    if ((d.routing.corrections || []).length) line("corrections", d.routing.corrections.join(", "));
  }
  for (const [k, v] of Object.entries(d)) {
    if (["summary", "routing", "mission_id", "phase", "receipt", "text", "goal", "forbidden", "plan"].includes(k)) continue;
    if (v === null || v === "" || typeof v === "object") continue;
    line(k, v);
  }
  return body.children.length ? body : null;
}

/* The voice chain for one utterance: WAKE → AUDIO → STT → VERDICT, each with
   the numbers the gate actually saw. Bug reports become "utterance vs…-u1
   rejected: self-echo 0.71" instead of "it said something weird". */
function voiceTrace(body, d) {
  const u = d.utterance || {}, a = u.audio || {}, s = u.stt || {}, v = d.verdict || {};
  const block = (title, rows) => {
    body.append(el("div", { class: "act-k", text: title, style: { marginTop: "6px" } }));
    for (const [k, val] of rows) if (val !== undefined && val !== null && val !== "") body.append(el("div", { class: "act-kv" }, el("span", { class: "act-k", text: k }), el("span", { class: "act-v", text: String(val) })));
  };
  block("WAKE", [["session", u.wake_session_id || u.session_id], ["score", u.wake_score], ["utterance", u.utterance_id], ["source", u.source]]);
  block("AUDIO", [["duration", a.duration_seconds !== undefined ? `${a.duration_seconds}s` : ""], ["speech", a.speech_seconds !== undefined ? `${a.speech_seconds}s (${Math.round((a.speech_fraction || 0) * 100)}%)` : ""],
    ["rms / peak", a.rms !== undefined ? `${a.rms} / ${a.peak}` : ""], ["noise floor", a.noise_floor], ["device speech", (u.device || {}).speech_seconds ? `${u.device.speech_seconds}s` : ""],
    ["ZEUS speaking", u.speaking_overlap ? "yes" : "no"]]);
  block("STT", [["raw", u.raw_transcript], ["normalized", u.normalized_transcript], ["language", `${s.language || ""} ${s.language_probability !== undefined ? "(" + s.language_probability + ")" : ""}`],
    ["no-speech prob.", s.no_speech_probability], ["avg logprob", s.avg_logprob], ["compression", s.compression_ratio], ["model / elapsed", s.model ? `${s.model} / ${s.elapsed}s` : ""],
    ["replacements", ((d.normalization || {}).replacements || []).map((r) => `${r.heard} → ${r.meant}`).join(", ")], ["wake word removed", (d.segmentation || {}).removed]]);
  block(v.accepted ? "VERDICT · ACCEPTED" : "VERDICT · REJECTED", [["reason", v.reason], ["confidence", v.confidence !== undefined ? `${v.confidence} (${v.level})` : ""]]);
  for (const c of v.checks || []) body.append(el("div", { class: "act-check" + (c.passed ? "" : " bad"), text: `${c.passed ? "✓" : "✗"} ${c.name}` + (c.observed ? ` — ${c.observed}` : "") }));
}

function inspect(entry) {
  const d = entry.detail || {};
  const children = [section("Event", kv("kind", entry.kind), kv("at", entry.at), kv("seq", entry.seq), kv("scope", entry.scope), kv("receipt", entry.receipt_id))];
  if (d.routing) {
    children.push(section("Routing", kv("top level", `${d.routing.top_level} · ${d.routing.confidence}`), kv("reason", d.routing.reason),
      kv("operation", d.routing.reading?.operation), kv("object", d.routing.reading?.object),
      kv("self / world", `${d.routing.reading?.self_score} / ${d.routing.reading?.world_score}`),
      kv("overruled", (d.routing.conflicts || []).join("\n")), kv("corrections", (d.routing.corrections || []).join(", "))));
  }
  if (d.plan) children.push(section("Plan", kv("steps", (d.plan.steps || []).map((s) => `${s.status === "forbidden" ? "⛔ " : ""}${s.step} [${s.role || "required"}]`).join("\n")),
    kv("constraints", JSON.stringify(d.plan.constraints || {}, null, 1), "mono")));
  if (entry.receipt_id && d.evidence) children.push(section("Evidence", kv("json", JSON.stringify(d.evidence, null, 1), "mono")));
  if (d.mission_id) children.push(el("div", { class: "toolbar" }, button("Open mission", () => views.open("missions", { mission: d.mission_id }))));
  views.inspect(entry.summary ? entry.summary.slice(0, 60) : entry.kind, ...children);
}
