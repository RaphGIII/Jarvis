/*
 * The Jarvis eye.
 *
 * An abstract luminous aperture: concentric arcs, a rotating iris lattice and a
 * pupil that breathes. Drawn from scratch on a canvas -- no images, no external
 * assets, nothing derived from anyone else's artwork.
 *
 * The central idea is that the eye has no per-state animations. It has a small
 * set of continuous PARAMETERS -- pupil size, spin rate, glow, ring tension,
 * jitter, hue -- and each state is a target for those parameters. Every frame
 * eases the live values towards the target, so a state change is a physical
 * settling rather than one animation being swapped for another. That is what
 * makes THINKING -> SPEAKING look like the same entity changing what it is
 * doing, instead of two clips cutting.
 *
 * It also means a new state costs one entry in STATES, which matters because
 * Jarvis is expected to modify this file through its own development pipeline.
 */

const STATES = {
  idle:         { pupil: 0.30, spin:  0.10, glow: 0.35, tension: 1.00, jitter: 0.00, hue: 198, arcs: 0.55, breathe: 0.030 },
  listening:    { pupil: 0.45, spin:  0.22, glow: 0.85, tension: 1.10, jitter: 0.00, hue: 190, arcs: 1.00, breathe: 0.090 },
  transcribing: { pupil: 0.38, spin:  0.75, glow: 0.70, tension: 0.94, jitter: 0.30, hue: 186, arcs: 0.85, breathe: 0.050 },
  thinking:     { pupil: 0.22, spin:  1.35, glow: 0.72, tension: 0.86, jitter: 0.10, hue: 205, arcs: 0.80, breathe: 0.022 },
  speaking:     { pupil: 0.40, spin:  0.30, glow: 1.00, tension: 1.06, jitter: 0.00, hue: 194, arcs: 0.95, breathe: 0.140 },
  working:      { pupil: 0.26, spin:  1.00, glow: 0.62, tension: 0.90, jitter: 0.06, hue: 168, arcs: 0.75, breathe: 0.035 },
  // Narrow pupil, almost no spin: checking, not doing. Deliberately distinct
  // from working, because "I did it" and "I confirmed it" are different claims.
  verifying:    { pupil: 0.18, spin:  0.18, glow: 0.70, tension: 1.04, jitter: 0.02, hue: 128, arcs: 0.90, breathe: 0.028 },
  researching:  { pupil: 0.33, spin:  0.62, glow: 0.66, tension: 0.97, jitter: 0.14, hue: 268, arcs: 0.85, breathe: 0.040 },
  coding:       { pupil: 0.24, spin:  1.15, glow: 0.68, tension: 0.88, jitter: 0.05, hue: 148, arcs: 0.78, breathe: 0.030 },
  waiting:      { pupil: 0.34, spin:  0.06, glow: 0.42, tension: 1.02, jitter: 0.00, hue: 42,  arcs: 0.60, breathe: 0.055 },
  error:        { pupil: 0.30, spin:  0.05, glow: 0.80, tension: 0.80, jitter: 0.55, hue: 356, arcs: 0.70, breathe: 0.020 },
  offline:      { pupil: 0.14, spin:  0.02, glow: 0.14, tension: 1.00, jitter: 0.00, hue: 220, arcs: 0.25, breathe: 0.010 },
};

class JarvisEye {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = "idle";
    this.live = { ...STATES.idle };
    this.phase = 0;
    this.spinAngle = 0;
    // Loudness from the microphone or the speech pipeline, 0..1. Lets the eye
    // pulse with the actual voice rather than a decorative sine wave.
    this.energy = 0;
    this.blink = 0;
    this.nextBlink = 2 + Math.random() * 5;
    this.lastFrame = performance.now();
    this.running = false;
  }

  setState(name) {
    if (STATES[name]) this.state = name;
  }

  setEnergy(value) {
    this.energy = Math.max(0, Math.min(1, value || 0));
  }

  start() {
    if (this.running) return;
    this.running = true;
    const frame = (now) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.lastFrame) / 1000);
      this.lastFrame = now;
      this.step(dt);
      this.draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  stop() { this.running = false; }

  step(dt) {
    const target = STATES[this.state] || STATES.idle;
    // Critically-damped-ish easing: fast enough to feel responsive, slow
    // enough that a burst of rapid state changes reads as one movement.
    const k = 1 - Math.exp(-dt * 4.5);
    for (const key of Object.keys(target)) {
      if (key === "hue") {
        // Hue travels the short way round the colour wheel, otherwise a jump
        // from red (356) to teal (168) sweeps through every hue in between.
        let delta = target.hue - this.live.hue;
        if (delta > 180) delta -= 360;
        if (delta < -180) delta += 360;
        this.live.hue = (this.live.hue + delta * k + 360) % 360;
      } else {
        this.live[key] += (target[key] - this.live[key]) * k;
      }
    }

    this.phase += dt;
    this.spinAngle += dt * this.live.spin;

    this.nextBlink -= dt;
    if (this.nextBlink <= 0) {
      this.blink = 1;
      this.nextBlink = 3 + Math.random() * 7;
    }
    this.blink = Math.max(0, this.blink - dt * 6.5);
  }

  draw() {
    const ctx = this.ctx;
    const size = this.canvas.width;
    const mid = size / 2;
    const unit = size / 2;

    ctx.clearRect(0, 0, size, size);

    const hue = this.live.hue;
    const glow = this.live.glow;
    const stroke = (l, a) => `hsla(${hue.toFixed(0)}, 85%, ${l}%, ${a})`;

    const breath = 1 + Math.sin(this.phase * 2.0) * this.live.breathe + this.energy * 0.10;
    const jitter = () => (Math.random() - 0.5) * this.live.jitter * unit * 0.035;
    const lidClose = this.blink * 0.92;

    ctx.save();
    ctx.translate(mid, mid);

    // --- ambient halo -------------------------------------------------
    const halo = ctx.createRadialGradient(0, 0, unit * 0.05, 0, 0, unit * 0.98);
    halo.addColorStop(0, `hsla(${hue}, 90%, 60%, ${0.20 * glow})`);
    halo.addColorStop(0.45, `hsla(${hue}, 90%, 50%, ${0.07 * glow})`);
    halo.addColorStop(1, "hsla(0, 0%, 0%, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(0, 0, unit * 0.98, 0, Math.PI * 2);
    ctx.fill();

    // --- outer segmented ring ----------------------------------------
    const outer = unit * 0.90 * this.live.tension;
    const segments = 48;
    ctx.lineWidth = Math.max(1, unit * 0.012);
    for (let i = 0; i < segments; i++) {
      const a0 = (i / segments) * Math.PI * 2 - this.spinAngle * 0.35;
      const span = (Math.PI * 2 / segments) * 0.55;
      const wave = 0.5 + 0.5 * Math.sin(this.phase * 1.6 + i * 0.5);
      ctx.strokeStyle = stroke(52 + wave * 22, (0.16 + 0.5 * wave * this.live.arcs) * glow);
      ctx.beginPath();
      ctx.arc(jitter(), jitter(), outer, a0, a0 + span);
      ctx.stroke();
    }

    // --- sweeping arcs ------------------------------------------------
    const arcs = [
      { r: 0.76, from: 0.00, len: 1.5, dir: 1, w: 0.020 },
      { r: 0.68, from: 2.10, len: 1.0, dir: -1, w: 0.014 },
      { r: 0.60, from: 4.00, len: 2.1, dir: 1, w: 0.010 },
    ];
    for (const arc of arcs) {
      const a0 = arc.from + this.spinAngle * arc.dir * 1.4;
      ctx.lineWidth = Math.max(1, unit * arc.w);
      ctx.strokeStyle = stroke(72, 0.5 * this.live.arcs * glow);
      ctx.beginPath();
      ctx.arc(0, 0, unit * arc.r * this.live.tension, a0, a0 + arc.len);
      ctx.stroke();
    }

    // --- iris lattice --------------------------------------------------
    const irisR = unit * 0.52 * this.live.tension * breath;
    const blades = 32;
    ctx.lineWidth = Math.max(1, unit * 0.008);
    for (let i = 0; i < blades; i++) {
      const a = (i / blades) * Math.PI * 2 + this.spinAngle;
      const inner = irisR * (0.42 + 0.10 * Math.sin(this.phase * 2.4 + i));
      ctx.strokeStyle = stroke(60, (0.10 + 0.22 * this.live.arcs) * glow);
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * inner, Math.sin(a) * inner);
      ctx.lineTo(Math.cos(a) * irisR, Math.sin(a) * irisR);
      ctx.stroke();
    }

    // --- iris body -----------------------------------------------------
    const iris = ctx.createRadialGradient(0, 0, irisR * 0.15, 0, 0, irisR);
    iris.addColorStop(0, `hsla(${hue}, 95%, 62%, ${0.30 * glow})`);
    iris.addColorStop(0.72, `hsla(${hue}, 90%, 42%, ${0.16 * glow})`);
    iris.addColorStop(1, `hsla(${hue}, 80%, 30%, 0.02)`);
    ctx.fillStyle = iris;
    ctx.beginPath();
    ctx.arc(0, 0, irisR, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = stroke(78, 0.75 * glow);
    ctx.lineWidth = Math.max(1, unit * 0.011);
    ctx.beginPath();
    ctx.arc(0, 0, irisR, 0, Math.PI * 2);
    ctx.stroke();

    // --- pupil ---------------------------------------------------------
    const pupilR = irisR * this.live.pupil * (1 + this.energy * 0.28);
    const pupil = ctx.createRadialGradient(0, 0, 0, 0, 0, Math.max(1, pupilR));
    pupil.addColorStop(0, `hsla(${hue}, 100%, 88%, ${0.95 * Math.max(0.25, glow)})`);
    pupil.addColorStop(0.55, `hsla(${hue}, 95%, 62%, ${0.55 * glow})`);
    pupil.addColorStop(1, `hsla(${hue}, 90%, 40%, 0)`);
    ctx.fillStyle = pupil;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(1, pupilR), 0, Math.PI * 2);
    ctx.fill();

    // --- eyelids (blink) ------------------------------------------------
    if (lidClose > 0.001) {
      ctx.fillStyle = "#05070c";
      const h = unit * lidClose;
      ctx.fillRect(-unit, -unit, unit * 2, h);
      ctx.fillRect(-unit, unit - h, unit * 2, h);
    }

    ctx.restore();
  }
}

window.JarvisEye = JarvisEye;
window.JARVIS_STATES = STATES;
