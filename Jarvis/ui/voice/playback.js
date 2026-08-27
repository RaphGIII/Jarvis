/* Speech playback: a queue of audio the server synthesised, played in order,
   stoppable at once (barge-in from the microphone or Esc). */

import { audioUrl } from "../core/api.js";
import * as bus from "../core/bus.js";

const queue = [];
let playing = false;
let current = null;
let eye = null;

export function init(deps) {
  eye = deps.eye;
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
