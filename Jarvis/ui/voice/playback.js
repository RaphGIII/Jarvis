/* Speech playback: a queue of audio the server synthesised, played in order,
   stoppable at once (barge-in from the microphone or Esc). */

import { api, audioUrl } from "../core/api.js";
import * as bus from "../core/bus.js";

const queue = [];
let playing = false;
let current = null;
let eye = null;
let audioCtx = null;
let meterRaf = 0;
/* TTS playback volume (0..1). This is the ONLY place the owner's "voice
   volume" setting acts: on the <audio> element that plays ZEUS's speech. */
let volume = 1;

export function setVolume(v) {
  const n = Number(v);
  volume = Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : 1;
  if (current) current.volume = volume;
}

export function getVolume() {
  return volume;
}

export function init(deps) {
  eye = deps.eye;
  api("/api/voice").then((r) => { const s = r.settings || r; if (s && s.volume !== undefined) setVolume(s.volume); }).catch(() => {});
  bus.on("speech", (payload) => {
    // Audio from replayed history is never played: a page refresh used to
    // make ZEUS say its last answers again, which read as ghost speech.
    if (payload._replay) return;
    if (payload.stop) { stop(); return; }
    if (payload.url) enqueue(payload);
  });
}

export function enqueue(payload) {
  queue.push(payload);
  if (!playing) drain();
}

async function drain() {
  playing = true;
  while (queue.length) {
    const item = queue.shift();
    try { await playOne(item.url); } catch { /* interrupted or unplayable */ }
  }
  playing = false;
  current = null;
  eye?.setEnergy(0);
}

/* The ribbon follows the voice: the level of the audio actually playing, frame by frame. */
function meter(audio) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaElementSource(audio);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    const buffer = new Uint8Array(analyser.fftSize);
    const tick = () => {
      if (current !== audio) return;
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (const v of buffer) { const d = (v - 128) / 128; sum += d * d; }
      eye?.setEnergy(Math.min(1, Math.sqrt(sum / buffer.length) * 3.2));
      meterRaf = requestAnimationFrame(tick);
    };
    tick();
  } catch {
    eye?.setEnergy(0.5);   // no analyser (an old engine): a steady level instead
  }
}

function playOne(url) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl(url));
    audio.volume = volume;
    current = audio;
    audio.onended = () => { cancelAnimationFrame(meterRaf); resolve(); };
    audio.onerror = (err) => { cancelAnimationFrame(meterRaf); reject(err); };
    audio.onplay = () => meter(audio);
    audio.play().catch(reject);
  });
}

export function stop() {
  queue.length = 0;
  if (current) { current.pause(); current.currentTime = 0; current = null; }
  playing = false;
  eye?.setEnergy(0);
}

export function isPlaying() {
  return playing;
}
