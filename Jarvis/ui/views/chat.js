/* The conversation: turns, streaming tokens, receipts as evidence, the
   composer, and the Korrigieren dialog on every receipt.

   Provenance rules, because the live product showed sentences nobody said:
   - a USER turn is rendered only for a user_message event, which the core
     publishes only for a message with provenance (typed, a wake session, a
     press, an owner action); its meta says which, and a spoken turn shows
     its wake tag, confidence and the raw transcript when it was normalised;
   - a thought is ZEUS's (meta.source zeus_thought / thought_inbox) and is
     rendered as INSIGHT, never as "You";
   - replayed history (a refresh) is rendered dimmed as the past, never
     spoken, and never re-triggers anything;
   - nothing is shown for a transcript before the gate accepted it. */

import { $, el, clear } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";
import * as corrections from "./corrections.js";
import * as playback from "../voice/playback.js";
import * as performance from "../core/performance.js";
import { render as renderMarkdown, renderPartial } from "../core/markdown.js";

let streaming = null;
let streamText = "";
let renderQueued = false;
let eye = null;
let historyDivider = false;

export function init(deps) {
  eye = deps.eye;
  wireComposer();
  performance.init();
  bus.on("user_message", (p) => {
    document.querySelector(".turn.interim")?.remove();
    const meta = p.meta || {};
    if (meta.source === "thought_inbox") {
      addTurn("insight" + (p._replay ? " history" : ""), "Thought", p.text, p);
      return;
    }
    const who = meta.source === "microphone" || meta.source === "ui_mic" ? "Du · 🎙" : meta.source === "correction_rerun" ? "Du · korrigiert" : "Du";
    const what = addTurn("user" + (p._replay ? " history" : ""), who, p.text, p);
    if (pending && !p._replay) { $("log").append(pending); scrollDown(); }
    if (what && meta.wake_word) what.append(el("span", { class: "wake-tag", text: ` ${meta.wake_word} · ${Number(meta.wake_score).toFixed(2)}` }));
    if (what && meta.speech_level) {
      what.append(el("span", { class: "wake-tag", title: `speech confidence ${meta.speech_confidence}`, text: ` ${meta.speech_level}` }));
    }
    if (what && meta.raw_transcript && meta.normalized && meta.raw_transcript !== meta.normalized && (meta.replacements || []).length) {
      what.append(el("div", { class: "heard", text: `gehört: „${meta.raw_transcript}“` }));
    }
    $("app").classList.add("conversing");
  });
  bus.on("token", (p) => { if (!p._replay) appendToken(p.text || ""); });
  bus.on("message", (p) => finishStreaming(p.text || "", p));
  bus.on("transcript", () => { /* the verdict follows as a user_message, or not at all */ });
  bus.on("error", (p) => {
    if (p._replay) return;
    clearPending();
    endStreaming();
    // §UX: raw ProviderError(...) lines belong in Activity/Diagnostics, not
    // in the conversation.  The chat gets one honest human sentence.
    console.warn("[zeus error]", p.error);
    // The chat shows one ZEUS sentence; the technical cause lives under Fortschritt / Erweitert.
    if (p.silent) return;
    addTurn("error", "!", "Das hat gerade nicht geklappt. Details findest du unter Fortschritt.");
  });
  bus.on("notification", (p) => {
    if (p._replay || !p.text) return;
    if (p.kind === "thought") { addTurn("insight", "Insight", p.text, p); return; }
    if (p.kind === "open_view") return;
    addTurn("note", "", p.text);
  });
  bus.on("state", (p) => { if (p.state !== "thinking" && p.state !== "speaking") endStreaming(); if (p.state === "idle" || p.state === "error" || p.state === "waiting") clearPending(); });
  bus.on("tool", onToolOrProgress);
  bus.on("progress", onToolOrProgress);
}

function onToolOrProgress(payload) {
  if (payload._replay) return;
  if (payload.receipt) { addReceipt(payload.receipt); return; }
  if (state.ui.echoActions && payload.summary) addTurn("note", "", payload.summary);
}

export function reset() {
  clearPending();
  clear($("log"));
  streaming = null;
  historyDivider = false;
  $("app").classList.remove("conversing");
}

export function addTurn(kind, who, text, payload) {
  if (!text) return null;
  const replay = Boolean(payload && payload._replay);
  if (replay && !historyDivider) {
    historyDivider = true;
    $("log").append(el("div", { class: "turn divider" }, el("div", { class: "who", text: "" }), el("div", { class: "what", text: "— zuvor —" })));
  }
  if (!replay && historyDivider) {
    historyDivider = false;
    $("log").append(el("div", { class: "turn divider" }, el("div", { class: "who", text: "" }), el("div", { class: "what", text: "— jetzt —" })));
  }
  const what = el("div", { class: "what", text });
  const turn = el("div", { class: `turn ${kind}${replay ? " history" : ""}` }, el("div", { class: "who", text: who }), what);
  if (payload && payload.meta && payload.meta.utterance_id) turn.dataset.utterance = payload.meta.utterance_id;
  if (payload && payload.meta && payload.meta.request_id) turn.dataset.request = payload.meta.request_id;
  $("log").append(turn);
  scrollDown();
  return what;
}

/* Streaming: every token is appended to the text so far and the turn is
   re-rendered as stream-safe Markdown on the next animation frame -- content
   appears as it arrives, never as a typewriter over a finished answer.  The
   message event then renders the completed text once, deterministically. */
function appendToken(text) {
  if (!streaming) {
    clearPending();
    streaming = addTurn("jarvis", window.ASSISTANT_NAME || "ZEUS", "​");
    if (streaming) { streaming.classList.add("md"); streaming.textContent = ""; streaming.append(el("span", { class: "cursor" })); }
    streamText = "";
  }
  if (!streaming) return;
  streamText += text;
  eye?.noteOutput?.(text.length);   // the wave answers to output, smoothed by the sphere itself
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    if (!streaming) return;
    streaming.innerHTML = renderPartial(streamText);
    streaming.append(el("span", { class: "cursor" }));
    scrollDown();
  });
}

function finishStreaming(finalText, payload) {
  clearPending();
  const meta = (payload && payload.meta) || {};
  if (meta.source === "zeus_thought") {
    if (streaming) { streaming.parentElement?.remove(); streaming = null; }
    addTurn("insight", "Insight", finalText, payload);
    return;
  }
  let what = null;
  if (streaming && !(payload && payload._replay)) { what = streaming; streaming = null; streamText = ""; }
  else if (finalText) what = addTurn("jarvis", window.ASSISTANT_NAME || "ZEUS", finalText, payload);
  if (what) {
    what.classList.add("md");
    what.innerHTML = renderMarkdown(finalText);   // the exact completed answer, rendered once
  }
  if (what && meta.completion && (meta.completion.truncated || meta.completion.complete === false) && !meta.completion.aborted) attachContinue(what, meta.completion);
  if (what && !(payload && payload._replay)) attachFeedback(what, payload || {});
  $("app").classList.add("conversing");
  scrollDown();
}

/* A ceiling-truncated answer says so and offers to go on in the same conversation.
   Continuing is a normal request: the cost policy decides, nothing is spent by itself. */
function attachContinue(what, completion) {
  const why = completion.truncated
    ? "Die Antwort wurde am Längenlimit abgeschnitten. "
    : "Die Antwort ist möglicherweise unvollständig. ";
  const note = el("div", { class: "truncated" },
    el("span", { text: why }),
    el("a", { href: "#", class: "fb-btn", text: "Weiter", onClick: (ev) => {
      ev.preventDefault();
      send("Bitte setze deine letzte Antwort genau an der Stelle fort, an der sie abgebrochen wurde, ohne den bisherigen Text zu wiederholen.");
    } }));
  what.append(note);
}

/* ------------------------------------------------------------------ */
/* response feedback: 👍 👎 Korrigieren under every ZEUS answer         */
/* ------------------------------------------------------------------ */

const FB_CATEGORIES = [
  ["TOO_SHORT", "zu kurz"], ["TOO_LONG", "zu lang"], ["WRONG_FACT", "falscher Fakt"], ["MISUNDERSTOOD", "missverstanden"],
  ["WRONG_ACTION", "falsche Aktion"], ["INCOMPLETE", "unvollständig"], ["TOO_TECHNICAL", "zu technisch"],
  ["TOO_SIMPLE", "zu einfach"], ["BAD_STYLE", "Stil"], ["BAD_PRONUNCIATION", "Aussprache"], ["OTHER", "anderes"],
];

function attachFeedback(what, payload) {
  const meta = payload.meta || {};
  const requestId = meta.request_id || "";
  const row = el("div", { class: "fb" });
  const flash = (text) => { const n = el("span", { class: "fb-flash", text }); row.append(n); setTimeout(() => n.remove(), 2500); };
  const up = el("a", { href: "#", class: "fb-btn", title: "Gute Antwort", text: "👍" });
  const down = el("a", { href: "#", class: "fb-btn", title: "Antwort bewerten", text: "👎" });
  const korr = el("a", { href: "#", class: "fb-btn fb-korr", text: "Korrigieren" });
  up.onclick = async (ev) => {
    ev.preventDefault();
    await api("/api/feedback", { kind: "response", rating: "up", request_id: requestId });
    row.querySelector(".fb-pop")?.remove();
    flash("Danke, gemerkt");
  };
  const openPop = (ev, focusText) => {
    ev.preventDefault();
    row.querySelector(".fb-pop")?.remove();
    const note = el("input", { placeholder: "optional: was genau?", class: "fb-note" });
    const pop = el("div", { class: "fb-pop" });
    for (const [key, label] of FB_CATEGORIES) {
      pop.append(el("button", { class: "chip", text: label, onClick: async () => {
        await api("/api/feedback", { kind: "response", rating: "down", category: key, text: note.value, request_id: requestId });
        pop.remove(); flash("Präferenz gespeichert");
      } }));
    }
    const send = el("button", { class: "chip on", text: "senden", onClick: async () => {
      await api("/api/feedback", { kind: "response", rating: "down", category: "OTHER", text: note.value, request_id: requestId });
      pop.remove(); flash("Präferenz gespeichert");
    } });
    pop.append(note, send);
    row.append(pop);
    if (focusText) note.focus();
  };
  down.onclick = (ev) => openPop(ev, false);
  korr.onclick = (ev) => openPop(ev, true);
  row.append(up, down, korr);
  what.append(row);
}

export function endStreaming() {
  if (!streaming) return;
  streaming.querySelector(".cursor")?.remove();
  streaming = null;
  streamText = "";
}

function scrollDown() {
  const log = $("log");
  log.scrollTop = log.scrollHeight;
}

/* A receipt, rendered as evidence -- compact: one line and the links, the
   checks behind a click. The verdict word comes from the receipt's own
   `verified` flag, never from any text the model wrote. */
export function addReceipt(receipt) {
  const verdict = receipt.verified ? "geprüft" : receipt.ok ? "ausgeführt, ungeprüft" : "gescheitert";
  const what = addTurn(`receipt ${receipt.verified ? "good" : "bad"}`, `${receipt.kind} · ${verdict}`,
                       receipt.detail || receipt.kind || "action");
  if (!what) return;
  const checks = el("div", { class: "checks" });
  checks.hidden = receipt.verified;
  for (const check of receipt.verifications || []) {
    checks.append(el("div", { class: "check" + (check.passed ? "" : " bad"),
                              text: `${check.passed ? "✓" : "✗"} ${check.check}` + (check.observed ? ` — ${check.observed}` : "") }));
  }
  checks.append(el("div", { class: "check", text: receipt.id }));
  what.append(checks);
  const links = el("div", { class: "links" });
  links.append(el("a", { class: "korrigieren", href: "#", text: "Korrigieren",
                         onClick: (ev) => { ev.preventDefault(); corrections.openDialog(receipt.id); } }));
  links.append(el("a", { href: "#", text: "Beleg", onClick: (ev) => { ev.preventDefault(); views.open("activity", { receipt: receipt.id }); } }));
  const n = (receipt.verifications || []).length;
  links.append(el("a", { href: "#", text: checks.hidden ? `${n} Prüfungen` : "weniger",
                         onClick: (ev) => { ev.preventDefault(); checks.hidden = !checks.hidden; ev.target.textContent = checks.hidden ? `${n} Prüfungen` : "weniger"; } }));
  what.append(links);
  scrollDown();
}

/* ------------------------------------------------------------------ */
/* composer                                                            */
/* ------------------------------------------------------------------ */

function requestId() {
  return (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, "").slice(0, 12);
}

export function send(text, source = "text", extra = {}) {
  const clean = (text || "").trim();
  if (!clean) return;
  document.querySelector(".turn.interim")?.remove();
  if (views.isWorkspace()) views.close();
  showPending();
  // One id per press: a retried POST cannot become a second request.
  const body = { text: clean, source, request_id: requestId(), mode: extra.mode || performance.currentMode() };
  if (extra.authorization) body.authorization = extra.authorization;
  return api("/api/message", body);
}

/* Between the owner's message and the first token ZEUS is visibly at work: one quiet line,
   replaced by the answer the moment it starts, never left behind. */
let pending = null;
function showPending() {
  clearPending();
  pending = el("div", { class: "turn jarvis pending" }, el("div", { class: "who", text: window.ASSISTANT_NAME || "ZEUS" }),
    el("div", { class: "what" }, el("span", { class: "thinking" }, el("i"), el("i"), el("i")), el("span", { class: "thinking-word", text: "denkt" })));
  $("log").append(pending);
  $("app").classList.add("conversing");
  scrollDown();
}
export function clearPending() {
  pending?.remove();
  pending = null;
}

/* The sidebar and the palette hand a beginning to the composer; the owner finishes the sentence. */
export function focusComposer(prefill) {
  const input = $("input");
  if (!input) return;
  if (prefill != null) { input.value = prefill; input.dispatchEvent(new Event("input")); }
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

function wireComposer() {
  const input = $("input");
  const submit = () => {
    const text = input.value;
    input.value = "";
    input.style.height = "auto";
    send(text);
  };
  $("btnSend").onclick = submit;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(200, input.scrollHeight) + "px";
  });
}
