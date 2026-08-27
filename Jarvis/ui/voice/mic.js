/* Microphone: capture here, recognise on the server. The browser records and
   uploads 16 kHz WAV; the core transcribes. The same split the HDMI box and
   the phone will use, so the browser is the first device client. */

import { $ } from "../core/dom.js";
import { api, postBytes } from "../core/api.js";
import * as playback from "./playback.js";
import { addTurn } from "../views/chat.js";

let micStream = null;
let recorder = null;
let micRaf = 0;
let audioCtx = null;
let eye = null;

export function init(deps) {
  eye = deps.eye;
  $("btnMic").onclick = () => toggle();
}

export function isRecording() {
  return Boolean(recorder && recorder.state === "recording");
}

export async function toggle() {
  const btn = $("btnMic");
  if (isRecording()) { recorder.stop(); btn.classList.remove("recording"); return; }
  try {
    micStream = micStream || (await navigator.mediaDevices.getUserMedia({ audio: true }));
  } catch (err) {
    addTurn("error", "Error", "microphone unavailable: " + err.message);
    return;
  }
  playback.stop();
  api("/api/stop", {});
  const chunks = [];
  recorder = new MediaRecorder(micStream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = async () => {
    cancelAnimationFrame(micRaf);
    eye.setEnergy(0);
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    const wav = await toWav(blob);
    // The press is the authorisation: the core acts on this utterance
    // because a person asked for it, not because a detector fired.
    const result = await postBytes("/api/voice/utterance?origin=ui", wav);
    if (!result.ok) addTurn("note", "", result.reason || result.error || "nothing was heard");
  };
  recorder.start();
  btn.classList.add("recording");
  meterInto(micStream);
}

/* Record a fixed-length clip and return it as 16 kHz WAV bytes (the wake-word wizard). */
export async function recordClip(seconds, onLevel) {
  micStream = micStream || (await navigator.mediaDevices.getUserMedia({ audio: true }));
  const chunks = [];
  const rec = new MediaRecorder(micStream);
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const done = new Promise((resolve) => { rec.onstop = resolve; });
  rec.start();
  const stopMeter = meterInto(micStream, onLevel);
  await new Promise((r) => setTimeout(r, seconds * 1000));
  rec.stop();
  await done;
  stopMeter();
  return toWav(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
}

function meterInto(stream, onLevel) {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  const source = audioCtx.createMediaStreamSource(stream);
  source.connect(analyser);
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  let raf = 0;
  const tick = () => {
    analyser.getByteTimeDomainData(buffer);
    let peak = 0;
    for (const v of buffer) peak = Math.max(peak, Math.abs(v - 128) / 128);
    const level = Math.min(1, peak * 2.2);
    eye?.setEnergy(level);
    if (onLevel) onLevel(level);
    raf = requestAnimationFrame(tick);
  };
  tick();
  micRaf = raf;
  return () => { cancelAnimationFrame(raf); cancelAnimationFrame(micRaf); source.disconnect(); eye?.setEnergy(0); };
}

async function toWav(blob) {
  const ctx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;
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
    const c = Math.max(-1, Math.min(1, samples[i]));
    out.setInt16(44 + i * 2, c < 0 ? c * 0x8000 : c * 0x7fff, true);
  }
  return out.buffer;
}
