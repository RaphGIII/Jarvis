/* Speech playback: a queue of audio the server synthesised, played in order,
   stoppable at once (barge-in from the microphone or Esc). */

import { api, audioUrl } from "../core/api.js";
import * as bus from "../core/bus.js";

const queue = [];
let playing = false;
let current = null;
let eye = null;
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

function playOne(url) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl(url));
    audio.volume = volume;
    current = audio;
    audio.onended = resolve;
    audio.onerror = reject;
    audio.onplay = () => eye?.setEnergy(0.55);
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
