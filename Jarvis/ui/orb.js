/* ZEUS presence: a ribbon of light, not an object.

   A handful of fine, translucent strands move together like one piece of silk
   drifting in slow air.  No sphere, no eye, no symbol.  The ribbon is quiet
   when nothing happens and says what ZEUS is doing through how it moves:

     idle        almost still: a low, slow drift
     listening   the strands open with the voice (microphone energy)
     thinking    a more complex fold, slow pulses travelling along it
     speaking    the amplitude follows the audio that is playing
     executing   a directional pulse passes through, again and again
     success     one soft bright pulse travels once
     error       one muted warm pulse -- never a flash

   Every JarvisState maps onto one of these (STATES below; the table's keys
   are the server's states, exactly).  Parameters are eased every frame, so
   one state flows into the next.  The loop runs at display rate while the
   window is focused, slower when it is not, not at all when hidden; with
   reduced motion the ribbon is drawn still and changes only when the state
   does.

   API kept for the shell: setState / setEnergy / setActivity / noteOutput /
   setAudioFrequencyData / setBackgroundWork / setThemeShift / setIntensity /
   pulseOnce / start / stop / resize.  Registered as window.ZeusPresence,
   window.ZeusOrb and window.JarvisEye. */

const STATES = {
  idle:         { mode: "idle",      amp: 0.16, speed: 0.10, fold: 0.00, light: 0.62, travel: 0.0 },
  listening:    { mode: "listening", amp: 0.22, speed: 0.18, fold: 0.10, light: 0.78, travel: 0.0 },
  transcribing: { mode: "listening", amp: 0.20, speed: 0.20, fold: 0.18, light: 0.76, travel: 0.0 },
  thinking:     { mode: "thinking",  amp: 0.26, speed: 0.22, fold: 0.55, light: 0.82, travel: 0.35 },
  speaking:     { mode: "speaking",  amp: 0.30, speed: 0.24, fold: 0.25, light: 0.90, travel: 0.0 },
  waiting:      { mode: "idle",      amp: 0.15, speed: 0.09, fold: 0.00, light: 0.60, travel: 0.0 },
  working:      { mode: "executing", amp: 0.24, speed: 0.20, fold: 0.30, light: 0.84, travel: 1.0 },
  verifying:    { mode: "executing", amp: 0.22, speed: 0.18, fold: 0.25, light: 0.80, travel: 0.7 },
  coding:       { mode: "executing", amp: 0.24, speed: 0.20, fold: 0.30, light: 0.84, travel: 1.0 },
  researching:  { mode: "executing", amp: 0.24, speed: 0.20, fold: 0.35, light: 0.84, travel: 0.9 },
  error:        { mode: "error",     amp: 0.14, speed: 0.08, fold: 0.05, light: 0.52, travel: 0.0 },
  offline:      { mode: "offline",   amp: 0.06, speed: 0.03, fold: 0.00, light: 0.26, travel: 0.0 },
};
const EXECUTING = new Set(["working", "verifying", "coding", "researching"]);
const SETTLED = new Set(["idle", "waiting"]);

const GOLD = [206, 184, 138];
const IVORY = [236, 230, 216];
const OLIVE = [150, 160, 128];
const EMBER = [196, 132, 112];

const TAU = Math.PI * 2;
const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${Math.max(0, Math.min(1, a)).toFixed(3)})`;

class ZeusPresence {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = "offline";
    this.target = { ...STATES.offline };
    this.now = { ...STATES.offline };
    this.energy = 0;          // eased external energy (mic, speech)
    this.external = 0;
    this.inflow = 0;
    this.rate = 0;
    this.audio = null;
    this.work = 0;
    this.intensity = 1;
    this.t = 0;
    this.pulses = [];         // {x, speed, strength, warm}
    this.nextTravel = 0;
    this.running = false;
    this.last = 0;
    this.lastDraw = 0;
    this.focused = document.hasFocus ? document.hasFocus() : true;
    this.dirty = true;
    this.w = 0; this.h = 0;
    window.addEventListener("focus", () => { this.focused = true; });
    window.addEventListener("blur", () => { this.focused = false; });
    this.resize();
  }

  // ---- api ---------------------------------------------------------------
  setState(name) {
    const prev = this.state;
    this.state = STATES[name] ? name : "idle";
    this.target = { ...STATES[this.state] };
    this.dirty = true;
    if (EXECUTING.has(prev) && SETTLED.has(this.state)) this.pulseOnce("success");
    if (this.state === "error" && prev !== "error") this.pulseOnce("error");
    if (EXECUTING.has(this.state) && !EXECUTING.has(prev)) this.pulseOnce("executing");
  }
  setEnergy(value) { this.external = Math.max(this.external * 0.6, clamp01(value)); this.dirty = true; }
  setActivity(level) { this.external = clamp01(level); this.dirty = true; }
  noteOutput(chars) { this.inflow += Math.max(0, Number(chars) || 0); }
  setAudioFrequencyData(bins) { this.audio = bins && bins.length ? bins : null; }
  setBackgroundWork(count) { this.work = Math.max(0, Number(count) || 0); }
  setThemeShift() { /* the ribbon keeps one palette */ }
  setIntensity(value) { this.intensity = Math.max(0, Math.min(1.6, Number(value) || 0)); this.dirty = true; }
  /* kind: success | error | executing */
  pulseOnce(kind = "success") {
    // reduced motion: nothing travels along the ribbon; the state itself (colour, stillness) carries the outcome
    if (this.reduced()) { this.dirty = true; return; }
    const strength = kind === "success" ? 1 : kind === "error" ? 0.8 : 0.65;
    this.pulses.push({ x: -0.15, speed: kind === "executing" ? 0.55 : 0.42, strength, warm: kind === "error" });
    if (this.pulses.length > 4) this.pulses.shift();
    this.dirty = true;
  }
  get mode() { return this.target.mode; }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(80, Math.round(rect.width || this.canvas.width || 640));
    const h = Math.max(40, Math.round(rect.height || w / 6));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = w; this.h = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }

  reduced() {
    return document.body.classList.contains("reduced-motion") || this.intensity === 0
      || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  start() {
    if (this.running) return;
    this.running = true;
    const loop = (time) => {
      if (!this.running) return;
      requestAnimationFrame(loop);
      if (document.hidden) { this.last = time; return; }
      const reduced = this.reduced();
      // frame budget: display rate when focused, ~20 fps behind other windows, redraw-on-change when motion is reduced
      const interval = reduced ? 250 : this.focused ? 0 : 50;
      if (time - this.lastDraw < interval) return;
      const dt = Math.min(0.05, (time - (this.last || time)) / 1000) || 0.016;
      this.last = time;
      this.step(dt, reduced);
      if (reduced && !this.dirty && !this.pulses.length) return;
      this.lastDraw = time;
      this.draw(reduced);
      this.dirty = false;
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }

  // ---- motion --------------------------------------------------------------
  step(dt, reduced) {
    // with reduced motion there is no easing to watch: the ribbon takes the new state at once
    const k = reduced ? 1 : 1 - Math.exp(-dt * 2.2);
    for (const key of ["amp", "speed", "fold", "light", "travel"]) this.now[key] += (this.target[key] - this.now[key]) * k;
    this.now.mode = this.target.mode;
    const perSecond = this.inflow / Math.max(dt, 0.001);
    this.inflow = 0;
    this.rate += (perSecond - this.rate) * (1 - Math.exp(-dt * (perSecond > this.rate ? 6 : 1.5)));
    let audio = 0;
    if (this.audio) {
      let sum = 0;
      const n = Math.min(this.audio.length, 48);
      for (let i = 0; i < n; i++) sum += this.audio[i];
      audio = clamp01(sum / n / 150);
    }
    const target = Math.max(this.external, audio, clamp01(this.rate / 140) * 0.6);
    this.energy += (target - this.energy) * (1 - Math.exp(-dt * (target > this.energy ? 9 : 2.4)));
    this.external *= Math.exp(-dt * 1.6);
    if (!reduced) this.t += dt * (0.25 + this.now.speed * 2.2 + this.energy * 0.8) * this.intensity;
    // executing: a pulse passes through at a steady interval
    if (this.now.travel > 0.5 && !reduced) {
      this.nextTravel -= dt;
      if (this.nextTravel <= 0) { this.pulseOnce("executing"); this.nextTravel = 2.4; }
    } else if (this.now.mode === "thinking" && !reduced) {
      this.nextTravel -= dt;
      if (this.nextTravel <= 0) { this.pulses.push({ x: -0.1, speed: 0.22, strength: 0.35, warm: false }); this.nextTravel = 3.6; }
    }
    for (const p of this.pulses) p.x += dt * p.speed * (reduced ? 4 : 1);
    this.pulses = this.pulses.filter((p) => p.x < 1.2);
  }

  /* the displacement of one strand at x in [0,1] */
  strand(x, index, count, amp, fold) {
    const t = this.t;
    const s = index / Math.max(1, count - 1) - 0.5;          // -0.5 .. 0.5 across the ribbon
    const envelope = Math.pow(Math.sin(Math.PI * x), 1.35);    // tapered ends
    let y = Math.sin(x * 5.2 + t * 0.9 + s * 0.9) * 0.55
          + Math.sin(x * 2.3 - t * 0.55 + s * 2.1) * 0.35
          + Math.sin(x * 9.1 + t * 1.3 + s * 3.4) * 0.18 * fold
          + Math.sin(x * 14.0 - t * 1.9 + s * 5.2) * 0.10 * fold;
    y += s * (0.55 + 0.35 * Math.sin(x * 3.1 + t * 0.4));     // the ribbon's width, breathing
    for (const p of this.pulses) {
      const d = (x - p.x) / 0.07;
      y += Math.exp(-d * d) * 0.45 * p.strength * Math.sin(s * 3 + 1.2);
    }
    return y * envelope * amp;
  }

  draw(reduced) {
    const ctx = this.ctx;
    const { w, h } = this;
    const p = this.now;
    ctx.clearRect(0, 0, w, h);
    const energy = reduced ? this.energy * 0.5 : this.energy;
    const amp = (p.amp + energy * 0.34 + Math.min(0.06, this.work * 0.02)) * h * 0.95;
    const fold = Math.min(1, p.fold + energy * 0.4);
    const light = clamp01(p.light + energy * 0.25);
    const warmPulse = this.pulses.find((q) => q.warm);
    const count = 9;
    const steps = Math.max(60, Math.min(180, Math.round(w / 5)));
    const cy = h / 2;
    const base = p.mode === "error" || warmPulse ? mix(GOLD, EMBER, warmPulse ? 0.5 : 0.35) : GOLD;
    for (let i = 0; i < count; i++) {
      const s = i / (count - 1);
      const edge = 1 - Math.abs(s - 0.5) * 1.6;                 // inner strands brighter
      const colour = mix(mix(base, IVORY, 0.35 + 0.35 * edge), OLIVE, 0.18 * (1 - edge));
      // gradient along the ribbon: fades in and out at the ends, brighter where a pulse is
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      const stops = [0, 0.18, 0.5, 0.82, 1];
      for (const at of stops) {
        let alpha = Math.sin(Math.PI * at) * (0.10 + 0.22 * edge) * light;
        for (const q of this.pulses) {
          const d = (at - q.x) / 0.16;
          alpha += Math.exp(-d * d) * 0.35 * q.strength * edge;
        }
        grad.addColorStop(at, rgba(q_colour(colour, at, this.pulses), alpha));
      }
      ctx.beginPath();
      for (let k = 0; k <= steps; k++) {
        const x = k / steps;
        const y = cy + this.strand(x, i, count, amp, fold);
        if (k === 0) ctx.moveTo(x * w, y); else ctx.lineTo(x * w, y);
      }
      ctx.strokeStyle = grad;
      ctx.lineWidth = 0.8 + 0.7 * edge;
      ctx.stroke();
    }
    // a faint veil of light between the strands: the silk itself
    ctx.beginPath();
    for (let k = 0; k <= steps; k++) {
      const x = k / steps;
      const y = cy + this.strand(x, 0, count, amp, fold);
      if (k === 0) ctx.moveTo(x * w, y); else ctx.lineTo(x * w, y);
    }
    for (let k = steps; k >= 0; k--) {
      const x = k / steps;
      ctx.lineTo(x * w, cy + this.strand(x, count - 1, count, amp, fold));
    }
    ctx.closePath();
    const veil = ctx.createLinearGradient(0, 0, w, 0);
    veil.addColorStop(0, rgba(base, 0));
    veil.addColorStop(0.5, rgba(base, 0.05 * light));
    veil.addColorStop(1, rgba(base, 0));
    ctx.fillStyle = veil;
    ctx.fill();
  }
}

/* A pulse warms or brightens the colour where it passes. */
function q_colour(colour, at, pulses) {
  let out = colour;
  for (const p of pulses) {
    const d = Math.abs(at - p.x);
    if (d < 0.2) out = mix(out, p.warm ? EMBER : IVORY, (1 - d / 0.2) * 0.6 * p.strength);
  }
  return out;
}

/* the older names the shell and saved themes still use */
const ZeusOrb = ZeusPresence;

window.ZeusPresence = ZeusPresence;
window.JarvisEye = ZeusOrb;
window.ZeusOrb = ZeusOrb;
window.ZeusSphere = ZeusOrb;
window.JARVIS_STATES = STATES;
