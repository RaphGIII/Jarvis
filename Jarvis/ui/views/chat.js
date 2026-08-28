/* The conversation: turns, streaming tokens, receipts as evidence, the
   composer, and the Korrigieren dialog on every receipt. */

import { $, el, clear } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";
import * as corrections from "./corrections.js";
import * as playback from "../voice/playback.js";

let streaming = null;
let eye = null;

export function init(deps) {
  eye = deps.eye;
  wireComposer();
  bus.on("user_message", (p) => {
    const what = addTurn("user", "You", p.text);
    // a spoken request: the wake word is session metadata, not content
    if (what && p.meta && p.meta.wake_word) what.append(el("span", { class: "wake-tag", text: ` ${p.meta.wake_word} · ${Number(p.meta.wake_score).toFixed(2)}` }));
    $("app").classList.add("conversing");
  });
  bus.on("token", (p) => appendToken(p.text || ""));
  bus.on("message", (p) => finishStreaming(p.text || ""));
  bus.on("transcript", (p) => showInterim(p.text || ""));
  bus.on("error", (p) => { endStreaming(); addTurn("error", "Error", p.error || "something went wrong"); });
  bus.on("notification", (p) => { if (p.text) addTurn("note", "", p.text); });
  bus.on("state", (p) => { if (p.state !== "thinking" && p.state !== "speaking") endStreaming(); });
  bus.on("tool", onToolOrProgress);
  bus.on("progress", onToolOrProgress);
}

function onToolOrProgress(payload) {
  if (payload.receipt) { addReceipt(payload.receipt); return; }
  if (state.ui.echoActions && payload.summary) addTurn("note", "", payload.summary);
}

export function reset() {
  clear($("log"));
  streaming = null;
  $("app").classList.remove("conversing");
}

export function addTurn(kind, who, text) {
  if (!text) return null;
  const what = el("div", { class: "what", text });
  const turn = el("div", { class: `turn ${kind}` }, el("div", { class: "who", text: who }), what);
  $("log").append(turn);
  scrollDown();
  return what;
}

function appendToken(text) {
  if (!streaming) {
    streaming = addTurn("jarvis", window.ASSISTANT_NAME || "ZEUS", "​");
    if (streaming) { streaming.textContent = ""; streaming.append(el("span", { class: "cursor" })); }
  }
  if (!streaming) return;
  streaming.insertBefore(document.createTextNode(text), streaming.querySelector(".cursor"));
  scrollDown();
}

function finishStreaming(finalText) {
  if (streaming) { streaming.textContent = finalText; streaming = null; }
  else if (finalText) addTurn("jarvis", window.ASSISTANT_NAME || "ZEUS", finalText);
  $("app").classList.add("conversing");
  scrollDown();
}

export function endStreaming() {
  if (!streaming) return;
  streaming.querySelector(".cursor")?.remove();
  streaming = null;
}

function showInterim(text) {
  let node = $("log").querySelector(".turn.interim .what");
  if (!node) node = addTurn("note interim", "", text);
  if (node) node.textContent = text;
}

function scrollDown() {
  const log = $("log");
  log.scrollTop = log.scrollHeight;
}

/* A receipt, rendered as evidence. The verdict word comes from the receipt's
   own `verified` flag, never from any text the model wrote. */
export function addReceipt(receipt) {
  const verdict = receipt.verified ? "verified" : receipt.ok ? "ran, unverified" : "failed";
  const what = addTurn(`receipt ${receipt.verified ? "good" : "bad"}`, `${receipt.kind} · ${verdict}`,
                       receipt.detail || receipt.kind || "action");
  if (!what) return;
  for (const check of receipt.verifications || []) {
    what.append(el("div", { class: "check" + (check.passed ? "" : " bad"),
                            text: `${check.passed ? "✓" : "✗"} ${check.check}` + (check.observed ? ` — ${check.observed}` : "") }));
  }
  what.append(el("div", { class: "check", text: receipt.id }));
  const links = el("div", { class: "links" });
  links.append(el("a", { class: "korrigieren", href: "#", text: "Korrigieren",
                         onClick: (ev) => { ev.preventDefault(); corrections.openDialog(receipt.id); } }));
  links.append(el("a", { href: "#", text: "Beleg", onClick: (ev) => { ev.preventDefault(); views.open("activity", { receipt: receipt.id }); } }));
  what.append(links);
  scrollDown();
}

/* ------------------------------------------------------------------ */
/* composer                                                            */
/* ------------------------------------------------------------------ */

export function send(text) {
  const clean = (text || "").trim();
  if (!clean) return;
  document.querySelector(".turn.interim")?.remove();
  if (views.isWorkspace()) views.close();
  return api("/api/message", { text: clean });
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
    input.style.height = Math.min(150, input.scrollHeight) + "px";
  });
}
