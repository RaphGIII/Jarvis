/*
 * The client. Subscribes to the event stream, renders it, and posts input back.
 *
 * All state comes from the server. The UI never guesses what Jarvis is doing --
 * it renders the state event and nothing else. That is what keeps the browser,
 * a future TV view and a portable device showing the same thing, and it is why
 * "start thinking" is not something the send button does locally.
 */

const $ = (id) => document.getElementById(id);

const ui = {
  app: null, log: null, input: null, stateLabel: null, detail: null,
  connPill: null, costPill: null, panel: null, panelBody: null, panelTitle: null,
};

let eye = null;
let stream = null;
let lastSeq = 0;
let streaming = null;   // the live assistant turn being appended to
let activityOn = false;
let reconnectDelay = 500;

function startJarvis() {
  for (const key of Object.keys(ui)) ui[key] = $(key === "app" ? "app" : key);
  ui.app = $("app");
  ui.log = $("log");
  ui.input = $("input");
  ui.stateLabel = $("stateLabel");
  ui.detail = $("detail");
  ui.connPill = $("connPill");
  ui.costPill = $("costPill");
  ui.panel = $("panel");
  ui.panelBody = $("panelBody");
  ui.panelTitle = $("panelTitle");

  eye = new JarvisEye($("eye"));
  eye.start();

  wireInput();
  connect();
  refreshStatus();
  setInterval(refreshStatus, 15000);
}

/* ------------------------------------------------------------------ */
/* transport                                                           */
/* ------------------------------------------------------------------ */

function api(path, body) {
  return fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-Jarvis-Token": window.JARVIS_TOKEN },
    body: body === undefined ? undefined : JSON.stringify(body || {}),
  }).then((r) => r.json());
}

function connect() {
  if (stream) stream.close();
  stream = new EventSource(`/events?token=${encodeURIComponent(window.JARVIS_TOKEN)}&since=${lastSeq}`);

  stream.onopen = () => {
    reconnectDelay = 500;
    setPill(ui.connPill, "connected", "live");
    refreshStatus();
  };

  stream.onerror = () => {
    setPill(ui.connPill, "reconnecting", "warn");
    // EventSource retries on its own, but only while the server is reachable.
    // An explicit backoff covers the case where the core has actually stopped.
    stream.close();
    reconnectDelay = Math.min(10000, reconnectDelay * 2);
    setTimeout(connect, reconnectDelay);
  };

  for (const type of ["state", "token", "message", "user_message", "transcript",
                      "tool", "progress", "notification", "error", "speech", "diagnostic"]) {
    stream.addEventListener(type, (e) => {
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      if (event.seq) lastSeq = Math.max(lastSeq, event.seq);
      handle(type, event.payload || {});
    });
  }
}

/* ------------------------------------------------------------------ */
/* rendering                                                           */
/* ------------------------------------------------------------------ */

function handle(type, payload) {
  switch (type) {
    case "state":
      eye.setState(payload.state || "idle");
      ui.stateLabel.textContent = payload.state || "idle";
      ui.detail.textContent = payload.detail || "";
      // A finished answer closes the streaming turn, so a later token cannot
      // append to a message the user has already seen completed.
      if (payload.state !== "thinking" && payload.state !== "speaking") endStreaming();
      break;

    case "user_message":
      addTurn("user", "You", payload.text);
      ui.app.classList.add("conversing");
      break;

    case "token":
      appendToken(payload.text || "");
      break;

    case "message":
      finishStreaming(payload.text || "");
      break;

    case "transcript":
      showInterim(payload.text || "");
      break;

    case "error":
      endStreaming();
      addTurn("error", "Error", payload.error || "something went wrong");
      break;

    case "notification":
      if (payload.text) addTurn("note", "", payload.text);
      break;

    case "speech":
      if (typeof payload.energy === "number") eye.setEnergy(payload.energy);
      break;

    case "tool":
    case "progress":
      if (activityOn && payload.summary) addTurn("note", "", payload.summary);
      break;
  }
}

function addTurn(kind, who, text) {
  if (!text) return null;
  const el = document.createElement("div");
  el.className = `turn ${kind}`;
  const w = document.createElement("div");
  w.className = "who";
  w.textContent = who;
  const c = document.createElement("div");
  c.className = "what";
  c.textContent = text;
  el.append(w, c);
  ui.log.appendChild(el);
  scrollDown();
  return c;
}

function appendToken(text) {
  if (!streaming) {
    streaming = addTurn("jarvis", "Jarvis", "​");
    if (streaming) streaming.textContent = "";
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    streaming?.appendChild(cursor);
  }
  if (!streaming) return;
  const cursor = streaming.querySelector(".cursor");
  const node = document.createTextNode(text);
  streaming.insertBefore(node, cursor);
  scrollDown();
}

function finishStreaming(finalText) {
  if (streaming) {
    // Trust the final message over the accumulated tokens: they can differ if
    // the client reconnected mid-answer and missed a chunk.
    streaming.textContent = finalText;
    streaming = null;
  } else if (finalText) {
    addTurn("jarvis", "Jarvis", finalText);
  }
  ui.app.classList.add("conversing");
  scrollDown();
}

function endStreaming() {
  if (!streaming) return;
  streaming.querySelector(".cursor")?.remove();
  streaming = null;
}

function showInterim(text) {
  let el = ui.log.querySelector(".turn.interim .what");
  if (!el) el = addTurn("note interim", "", text);
  if (el) el.textContent = text;
}

function scrollDown() {
  ui.log.scrollTop = ui.log.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* input                                                               */
/* ------------------------------------------------------------------ */

function wireInput() {
  const send = () => {
    const text = ui.input.value.trim();
    if (!text) return;
    ui.input.value = "";
    ui.input.style.height = "auto";
    document.querySelector(".turn.interim")?.remove();
    api("/api/message", { text });
  };

  $("btnSend").onclick = send;

  ui.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });

  ui.input.addEventListener("input", () => {
    ui.input.style.height = "auto";
    ui.input.style.height = Math.min(150, ui.input.scrollHeight) + "px";
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { api("/api/stop", {}); endStreaming(); }
  });

  $("btnMic").onclick = () => toggleMic();

  $("btnActivity").onclick = (e) => {
    activityOn = !activityOn;
    e.currentTarget.setAttribute("aria-pressed", String(activityOn));
  };

  $("btnDiag").onclick = async () => {
    openPanel("Diagnostics", "loading…");
    const data = await api("/api/diagnostics");
    ui.panelBody.innerHTML = "";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(data, null, 2);
    ui.panelBody.appendChild(pre);
  };

  $("btnClose").onclick = () => ui.panel.classList.remove("open");
}

function openPanel(title, text) {
  ui.panelTitle.textContent = title;
  ui.panelBody.textContent = text;
  ui.panel.classList.add("open");
}

/* ------------------------------------------------------------------ */
/* microphone (browser-side capture; recognition happens server-side)   */
/* ------------------------------------------------------------------ */

let micStream = null;
let micAnalyser = null;
let micRaf = 0;

async function toggleMic() {
  const btn = $("btnMic");
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
    cancelAnimationFrame(micRaf);
    btn.classList.remove("recording");
    eye.setEnergy(0);
    api("/api/state");
    return;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    addTurn("error", "Error", "microphone unavailable: " + err.message);
    return;
  }
  btn.classList.add("recording");

  // Local level metering only -- it drives the eye so the user can see they
  // are being heard. Nothing is uploaded from here; the server owns capture.
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const source = ctx.createMediaStreamSource(micStream);
  micAnalyser = ctx.createAnalyser();
  micAnalyser.fftSize = 512;
  source.connect(micAnalyser);
  const buffer = new Uint8Array(micAnalyser.frequencyBinCount);

  const meter = () => {
    micAnalyser.getByteTimeDomainData(buffer);
    let peak = 0;
    for (const v of buffer) peak = Math.max(peak, Math.abs(v - 128) / 128);
    eye.setEnergy(Math.min(1, peak * 2.2));
    micRaf = requestAnimationFrame(meter);
  };
  meter();
}

/* ------------------------------------------------------------------ */
/* status                                                              */
/* ------------------------------------------------------------------ */

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    const conn = status.connection || "OFFLINE";
    const tone = conn === "OFFLINE" ? "bad" : conn === "EXPERT QUOTA EXHAUSTED" ? "warn" : "live";
    setPill(ui.connPill, conn.toLowerCase(), tone);
  } catch {
    setPill(ui.connPill, "offline", "bad");
  }
}

function setPill(el, text, tone) {
  if (!el) return;
  el.textContent = text;
  el.className = "pill" + (tone ? " " + tone : "");
}

window.startJarvis = startJarvis;
