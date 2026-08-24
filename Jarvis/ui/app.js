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

  // TV/kiosk mode is a URL flag rather than a build: the same page, scaled up
  // with the pointer affordances removed. ?tv=1 on the Jarvis URL.
  if (new URLSearchParams(location.search).has("tv")) {
    document.body.classList.add("tv");
    document.documentElement.requestFullscreen?.().catch(() => {});
  }

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
      if (payload.stop) { stopSpeech(); break; }
      if (payload.url) enqueueSpeech(payload);
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

  $("btnProjects").onclick = () => showProjects();
  $("btnGraph").onclick = () => openGraph();
  $("btnGraphClose").onclick = () => closeGraph();
  $("graphSearch").addEventListener("input", (e) => graph?.setFilter(e.target.value));
}

/* ------------------------------------------------------------------ */
/* projects                                                            */
/* ------------------------------------------------------------------ */

const STATE_CLASS = {
  active: "active", running: "active", working: "active",
  blocked: "blocked", failed: "blocked",
  accepted: "done", complete: "done", completed: "done",
};

async function showProjects() {
  openPanel("Projects", "loading…");
  const data = await api("/api/projects");
  const projects = data.projects || [];
  ui.panelBody.innerHTML = "";

  if (!projects.length) {
    ui.panelBody.textContent = "No projects yet. Describe something to build and Jarvis will open one.";
    return;
  }

  for (const project of projects) {
    const card = document.createElement("div");
    card.className = "project";

    const goal = document.createElement("div");
    goal.className = "goal";
    goal.textContent = project.goal || "(no goal recorded)";

    const meta = document.createElement("div");
    meta.className = "meta";
    const badge = document.createElement("span");
    badge.className = "badge " + (STATE_CLASS[project.state] || "idle");
    badge.textContent = project.state || "unknown";
    meta.appendChild(badge);
    for (const [label, value] of [["tasks", project.tasks], ["steps", project.steps]]) {
      const span = document.createElement("span");
      span.textContent = `${value} ${label}`;
      meta.appendChild(span);
    }

    card.append(goal, meta);
    // The id is what "continue the chess project" has to resolve to, so it is
    // what the card carries rather than the title.
    card.onclick = () => showProject(project.id);
    ui.panelBody.appendChild(card);
  }
}

async function showProject(id) {
  openPanel("Project", "loading…");
  const detail = await api("/api/project", { id });
  ui.panelBody.innerHTML = "";
  if (detail.error) { ui.panelBody.textContent = detail.error; return; }

  const goal = document.createElement("div");
  goal.className = "goal";
  goal.style.marginBottom = "14px";
  goal.textContent = detail.goal || "";
  ui.panelBody.appendChild(goal);

  addSection("Acceptance", (detail.acceptance || []).map((item) =>
    `${item.satisfied ? "✓" : "·"}  ${item.text}`));
  addSection("Tasks", (detail.tasks || []).map((task) =>
    `${task.status}  ${task.title}${task.attempts ? ` (${task.attempts} attempts)` : ""}`));

  const steps = document.createElement("div");
  steps.className = "steps";
  const heading = document.createElement("div");
  heading.className = "node-type";
  heading.textContent = "Recent activity";
  steps.appendChild(heading);
  for (const step of (detail.steps || []).slice(-25).reverse()) {
    const row = document.createElement("div");
    row.className = "step" + (step.success ? "" : " bad");
    const phase = document.createElement("div");
    phase.className = "phase";
    phase.textContent = step.phase;
    const summary = document.createElement("div");
    summary.className = "sum";
    summary.textContent = step.summary;
    row.append(phase, summary);
    steps.appendChild(row);
  }
  ui.panelBody.appendChild(steps);

  const button = document.createElement("button");
  button.className = "ghost";
  button.style.marginTop = "14px";
  button.textContent = "Continue this project";
  button.onclick = () => {
    ui.panel.classList.remove("open");
    api("/api/message", { text: `Continue the project: ${detail.goal}` });
  };
  ui.panelBody.appendChild(button);
}

function addSection(title, lines) {
  if (!lines.length) return;
  const heading = document.createElement("div");
  heading.className = "node-type";
  heading.style.marginTop = "12px";
  heading.textContent = title;
  ui.panelBody.appendChild(heading);
  for (const line of lines) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = "";
    const k = document.createElement("span");
    k.className = "v";
    k.style.textAlign = "left";
    k.textContent = line;
    row.appendChild(k);
    ui.panelBody.appendChild(row);
  }
}

/* ------------------------------------------------------------------ */
/* knowledge graph                                                     */
/* ------------------------------------------------------------------ */

let graph = null;
let graphNode = null;

async function openGraph() {
  const view = $("graphView");
  view.classList.add("open");

  if (!graph) {
    graph = new KnowledgeStarfield($("graphCanvas"), { onSelect: showNode });
    window.addEventListener("resize", () => graph.resize());
    wireNodeCard();
  }
  graph.resize();
  graph.start();

  const data = await api("/api/knowledge/graph", { limit: 400 });
  graph.load(data);
  $("graphCount").textContent =
    `${(data.nodes || []).length} nodes · ${(data.edges || []).length} links` +
    (data.truncated ? " (truncated)" : "");
}

function closeGraph() {
  $("graphView").classList.remove("open");
  // Stop the layout when it is not visible: it is the only thing in the UI
  // that burns CPU continuously, and a hidden animation is pure waste on a
  // machine that is also running a local model.
  graph?.stop();
}

async function showNode(node) {
  const card = $("nodeCard");
  if (!node) { card.hidden = true; graphNode = null; return; }

  graphNode = node;
  card.hidden = false;
  $("nodeType").textContent = node.type;
  $("nodeTitle").textContent = node.title;
  $("nodeBody").textContent = node.body ? node.body.slice(0, 600) : "(no content)";
  $("nodeLinks").innerHTML = "";

  const detail = await api("/api/knowledge/node", { id: node.id });
  const related = [...(detail.outgoing || []), ...(detail.incoming || [])];
  for (const item of related.slice(0, 12)) {
    const button = document.createElement("button");
    button.textContent = `${item.edge.type} → ${item.node.title}`.slice(0, 42);
    button.onclick = () => {
      graph.focusOn(item.node.id);
      const target = graph.byId.get(item.node.id);
      if (target) showNode(target);
    };
    $("nodeLinks").appendChild(button);
  }
}

function wireNodeCard() {
  $("btnAskAbout").onclick = () => {
    if (!graphNode) return;
    closeGraph();
    api("/api/message", { text: `Tell me about "${graphNode.title}" from my knowledge graph.` });
  };
  $("btnReadAloud").onclick = () => {
    if (!graphNode) return;
    api("/api/voice", { enabled: true, speak_replies: true });
    api("/api/message", { text: `Read this note aloud: "${graphNode.title}". ${graphNode.body || ""}`.slice(0, 1500) });
  };
  $("btnExpand").onclick = async () => {
    if (!graphNode) return;
    const data = await api("/api/knowledge/graph", { query: graphNode.title, limit: 400 });
    graph.load(data);
    graph.focusOn(graphNode.id);
  };
}

function openPanel(title, text) {
  ui.panelTitle.textContent = title;
  ui.panelBody.textContent = text;
  ui.panel.classList.add("open");
}

/* ------------------------------------------------------------------ */
/* microphone: capture here, recognise on the server                   */
/* ------------------------------------------------------------------ */

/*
 * The browser records and uploads; the core transcribes. That split is
 * deliberate -- it is the same shape the future HDMI box and phone client will
 * use, so the browser is the first device client rather than a special case.
 */

let micStream = null;
let recorder = null;
let micRaf = 0;
let audioCtx = null;

async function toggleMic() {
  const btn = $("btnMic");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    btn.classList.remove("recording");
    return;
  }

  try {
    micStream = micStream || (await navigator.mediaDevices.getUserMedia({ audio: true }));
  } catch (err) {
    addTurn("error", "Error", "microphone unavailable: " + err.message);
    return;
  }

  // Barge-in: speaking into a talking Jarvis stops it.
  stopSpeech();
  api("/api/stop", {});

  const chunks = [];
  recorder = new MediaRecorder(micStream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = async () => {
    cancelAnimationFrame(micRaf);
    eye.setEnergy(0);
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    const wav = await toWav(blob);
    const result = await fetch("/api/voice/utterance", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Jarvis-Token": window.JARVIS_TOKEN },
      body: wav,
    }).then((r) => r.json()).catch(() => ({ ok: false }));
    if (!result.ok) addTurn("note", "", result.reason || "nothing was heard");
  };

  recorder.start();
  btn.classList.add("recording");
  meterInto(micStream);
}

function meterInto(stream) {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  const tick = () => {
    analyser.getByteTimeDomainData(buffer);
    let peak = 0;
    for (const v of buffer) peak = Math.max(peak, Math.abs(v - 128) / 128);
    eye.setEnergy(Math.min(1, peak * 2.2));
    micRaf = requestAnimationFrame(tick);
  };
  tick();
}

/*
 * MediaRecorder gives webm/opus; whisper wants PCM. Decoding through
 * AudioContext and writing a WAV header here avoids putting a transcoder on
 * the server, and the browser's decoder handles whatever container it chose.
 */
async function toWav(blob) {
  const ctx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;                       // what whisper works in
  const length = Math.ceil(decoded.duration * rate);
  const offline = new OfflineAudioContext(1, length, rate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  const samples = rendered.getChannelData(0);

  const out = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const ascii = (off, text) => { for (let i = 0; i < text.length; i++) out.setUint8(off + i, text.charCodeAt(i)); };
  ascii(0, "RIFF"); out.setUint32(4, 36 + samples.length * 2, true); ascii(8, "WAVE");
  ascii(12, "fmt "); out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, rate, true); out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  ascii(36, "data"); out.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    out.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return out.buffer;
}

/* ------------------------------------------------------------------ */
/* speech playback                                                     */
/* ------------------------------------------------------------------ */

/*
 * Phrases arrive as they are synthesised, which is the whole point, but they
 * must be PLAYED strictly in order or the answer comes out scrambled. So audio
 * goes into a queue and one element drains it; arrival order is generation
 * order, and generation order is the order Jarvis meant.
 */
const speechQueue = [];
let speechPlaying = false;
let currentAudio = null;

function enqueueSpeech(payload) {
  speechQueue.push(payload);
  if (!speechPlaying) drainSpeech();
}

async function drainSpeech() {
  speechPlaying = true;
  while (speechQueue.length) {
    const item = speechQueue.shift();
    try {
      await playOne(item.url);
    } catch {
      /* a phrase that will not play must not stop the rest of the answer */
    }
  }
  speechPlaying = false;
  currentAudio = null;
  eye.setEnergy(0);
}

function playOne(url) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(`${url}?token=${encodeURIComponent(window.JARVIS_TOKEN)}`);
    currentAudio = audio;
    audio.onended = resolve;
    audio.onerror = reject;
    // A crude level so the eye pulses while speaking; the real envelope would
    // need an AnalyserNode per clip, which is not worth the allocation churn.
    audio.onplay = () => eye.setEnergy(0.55);
    audio.play().catch(reject);
  });
}

function stopSpeech() {
  speechQueue.length = 0;
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  speechPlaying = false;
  eye.setEnergy(0);
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
