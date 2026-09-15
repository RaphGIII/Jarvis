/* The ZEUS sphere.

   One dark optical object: a dense near-black sphere with a thin luminous
   circumference that catches warm light from the upper left, and one
   continuous wave passing horizontally through its centre and a little
   beyond it on both sides.  Nothing else -- no rings, no eye, no particles.

   Canvas 2D, one requestAnimationFrame loop, delta-time based, every
   parameter eased towards its target so states flow into each other.  The
   wave is a smoothed activity envelope translated into amplitude and
   harmonic content:

     output activity (tokens, speech energy, later real audio)
       -> attack/release envelope
         -> amplitude, frequency, complexity of the wave

   API (kept from the previous object so the shell does not change):
     setState(name)            one of the server's states (see STATES)
     setEnergy(0..1)           speech energy from playback / microphone
     setActivity(0..1)         an external activity floor (Voice later)
     noteOutput(chars)         streamed output arrived: raises the envelope
     setAudioFrequencyData(a)  Uint8Array of frequency bins (Voice later)
     setBackgroundWork(n)      running jobs: a little more life in the rim
     setThemeShift / setIntensity / pulseOnce / start / stop / resize
   Registered as window.JarvisEye and window.ZeusOrb. */

const STATES = {
  idle:         { amp: 0.032, freq: 1.00, complexity: 0.22, speed: 0.50, warmth: 0, rim: 0.60, energy: 0.10 },
  listening:    { amp: 0.055, freq: 1.10, complexity: 0.32, speed: 0.70, warmth: 0, rim: 0.70, energy: 0.20 },
  transcribing: { amp: 0.065, freq: 1.30, complexity: 0.40, speed: 0.80, warmth: 0, rim: 0.70, energy: 0.25 },
  thinking:     { amp: 0.080, freq: 1.70, complexity: 0.70, speed: 1.15, warmth: 0, rim: 0.85, energy: 0.35 },
  speaking:     { amp: 0.110, freq: 1.40, complexity: 0.55, speed: 1.00, warmth: 0, rim: 0.85, energy: 0.45 },
  waiting:      { amp: 0.038, freq: 1.00, complexity: 0.28, speed: 0.45, warmth: 0, rim: 0.62, energy: 0.12 },
  working:      { amp: 0.085, freq: 1.50, complexity: 0.60, speed: 0.95, warmth: 0, rim: 0.88, energy: 0.45 },
  verifying:    { amp: 0.075, freq: 1.45, complexity: 0.55, speed: 0.90, warmth: 0, rim: 0.85, energy: 0.40 },
  coding:       { amp: 0.085, freq: 1.60, complexity: 0.62, speed: 0.95, warmth: 0, rim: 0.88, energy: 0.45 },
  researching:  { amp: 0.080, freq: 1.50, complexity: 0.58, speed: 0.92, warmth: 0, rim: 0.86, energy: 0.42 },
  error:        { amp: 0.048, freq: 0.90, complexity: 0.30, speed: 0.40, warmth: 1, rim: 0.72, energy: 0.15 },
  offline:      { amp: 0.014, freq: 0.80, complexity: 0.12, speed: 0.20, warmth: 0, rim: 0.32, energy: 0.03 },
};
const SUCCESS_STATES = new Set(["idle", "waiting"]);
const WORK_STATES = new Set(["working", "verifying", "coding", "researching"]);

// the light: champagne, and the warm shift it takes on failure
const LIGHT = [241, 230, 209];
const CHAMPAGNE = [226, 206, 170];
const DEEP = [205, 183, 143];
const WARM = [214, 148, 118];

const clamp01 = (v) => Math.max(0, Math.min(1, Number(v) || 0));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${Math.max(0, Math.min(1, a)).toFixed(3)})`;

class ZeusOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = "offline";
    this.target = { ...STATES.offline };
    this.now = { ...STATES.offline };
    // activity envelope
    this.inflow = 0;          // characters that arrived since the last step
    this.rate = 0;            // smoothed characters per second
    this.activity = 0;        // displayed 0..1
    this.external = 0;        // setActivity / setEnergy floor
    this.audio = null;        // Uint8Array frequency bins, when Voice drives the wave
    this.audioLevel = 0;
    this.work = 0;
    this.hueShift = 0;
    this.intensity = 1;
    this.t = 0;
    this.pulse = 0;
    this.running = false;
    this.last = 0;
    this.quality = 1;
    this.frameCost = 0;
    this.w = 0; this.h = 0; this.r = 0;
    this.resize();
  }

  // ---- api -----------------------------------------------------------
  setState(name) {
    const prev = this.state;
    this.state = STATES[name] ? name : "idle";
    this.target = { ...STATES[this.state] };
    if (WORK_STATES.has(prev) && SUCCESS_STATES.has(this.state)) this.pulse = 1;   // work finished: one quiet pulse
  }
  setEnergy(value) { this.external = Math.max(this.external * 0.5, clamp01(value)); }
  setActivity(level) { this.external = clamp01(level); }
  noteOutput(chars) { this.inflow += Math.max(0, Number(chars) || 0); }
  setAudioFrequencyData(bins) { this.audio = bins && bins.length ? bins : null; }
  setBackgroundWork(count) { this.work = Math.max(0, Number(count) || 0); }
  setThemeShift(degrees) { this.hueShift = Number(degrees) || 0; }
  setIntensity(value) { this.intensity = Math.max(0, Math.min(1.6, Number(value) || 0)); }
  pulseOnce() { this.pulse = 1; }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(48, Math.round(rect.width || this.canvas.width || 420));
    const h = Math.max(34, Math.round(rect.height || w / 1.4));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = w; this.h = h;
    this.r = Math.min(h * 0.44, w / 2.62);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cache = null;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const loop = (t) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (t - (this.last || t)) / 1000) || 0.016;
      this.last = t;
      const started = performance.now();
      this.step(dt);
      if (!document.hidden) this.draw();
      this.frameCost = this.frameCost * 0.9 + (performance.now() - started) * 0.1;
      // a slow machine gets a simpler object, never a stutter
      if (this.frameCost > 8 && this.quality > 0.5) this.quality -= 0.05;
      else if (this.frameCost < 3.5 && this.quality < 1) this.quality += 0.02;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }

  // ---- motion --------------------------------------------------------
  reduced() { return document.body.classList.contains("reduced-motion") || this.intensity === 0; }

  step(dt) {
    // every state parameter eases towards its target: no state change is a cut
    const k = 1 - Math.exp(-dt * 2.8);
    for (const key of Object.keys(this.target)) this.now[key] += (this.target[key] - this.now[key]) * k;

    // the activity envelope: output arriving raises it quickly, silence lets it fall slowly
    const perSecond = this.inflow / Math.max(dt, 0.001);
    this.inflow = 0;
    const rateK = perSecond > this.rate ? 1 - Math.exp(-dt * 8) : 1 - Math.exp(-dt * 1.6);
    this.rate += (perSecond - this.rate) * rateK;
    let audio = 0;
    if (this.audio) {
      let sum = 0;
      const n = Math.min(this.audio.length, 48);
      for (let i = 0; i < n; i++) sum += this.audio[i];
      audio = clamp01((sum / n) / 160);
    }
    this.audioLevel += (audio - this.audioLevel) * (1 - Math.exp(-dt * 10));
    const target = Math.max(clamp01(this.rate / 110), this.external, this.audioLevel);
    const actK = target > this.activity ? 1 - Math.exp(-dt * 6) : 1 - Math.exp(-dt * 2.2);
    this.activity += (target - this.activity) * actK;
    this.external *= Math.exp(-dt * 1.2);   // a speech-energy sample is a moment, not a level

    const motion = this.reduced() ? 0.15 : this.intensity;
    this.t += dt * (0.55 + this.now.speed * 1.4 + this.activity * 1.6) * motion;
    if (this.pulse > 0) this.pulse = Math.max(0, this.pulse - dt / 0.75);
  }

  // ---- drawing -------------------------------------------------------
  wave(u, t, amp, freq, complexity) {
    // u in [-1.25, 1.25] (sphere radii from the centre); a Gaussian window keeps the line
    // alive across the sphere and lets it settle just beyond the limb
    const env = Math.exp(-(u * u) * 1.35);
    const base = Math.sin(u * Math.PI * freq * 2.0 + t);
    const h2 = 0.45 * complexity * Math.sin(u * Math.PI * freq * 3.7 - t * 1.6 + 1.3);
    const h3 = 0.22 * complexity * Math.sin(u * Math.PI * freq * 6.1 + t * 2.3 + 0.4);
    const drift = 0.30 * Math.sin(u * Math.PI * 0.8 - t * 0.31);
    let a = 0;
    if (this.audio) {
      const i = Math.min(this.audio.length - 1, Math.floor(Math.abs(u) / 1.25 * Math.min(this.audio.length, 40)));
      a = (this.audio[i] / 255) * 0.9 * Math.sin(u * Math.PI * 5 + t * 3);
    }
    return amp * env * (base + h2 + h3 + drift + a) / (1 + 0.7 * complexity);
  }

  drawWave(ctx, cx, cy, r, colour, alpha, width, amp, freq, complexity, phase) {
    const x0 = cx - r * 1.25, x1 = cx + r * 1.25;
    const steps = Math.round((x1 - x0) / (this.quality > 0.7 ? 2 : 4));
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const x = x0 + (x1 - x0) * (i / steps);
      const u = (x - cx) / r;
      const y = cy + this.wave(u, this.t + phase, amp, freq, complexity) * r;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    const g = ctx.createLinearGradient(x0, 0, x1, 0);
    g.addColorStop(0, rgba(colour, 0));
    g.addColorStop(0.09, rgba(colour, alpha * 0.55));
    g.addColorStop(0.5, rgba(colour, alpha));
    g.addColorStop(0.91, rgba(colour, alpha * 0.55));
    g.addColorStop(1, rgba(colour, 0));
    ctx.strokeStyle = g;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  draw() {
    const ctx = this.ctx;
    const { w, h, r } = this;
    const cx = w / 2, cy = h / 2;
    const p = this.now;
    const reduced = this.reduced();
    const activity = reduced ? this.activity * 0.4 : this.activity;
    const energy = clamp01(p.energy + activity * 0.5 + Math.min(0.2, this.work * 0.07));
    const warm = clamp01(p.warmth);
    const light = mix(LIGHT, WARM, warm * 0.75);
    const champagne = mix(CHAMPAGNE, WARM, warm * 0.75);
    const deep = mix(DEEP, WARM, warm * 0.6);
    const rim = clamp01(p.rim + energy * 0.15);
    ctx.clearRect(0, 0, w, h);

    // beneath: an extremely soft warm presence, the only glow the object has
    const under = ctx.createRadialGradient(cx, cy + r * 0.25, r * 0.4, cx, cy + r * 0.25, r * 1.9);
    under.addColorStop(0, rgba(champagne, 0.045 + energy * 0.03));
    under.addColorStop(1, rgba(champagne, 0));
    ctx.fillStyle = under;
    ctx.fillRect(0, 0, w, h);

    // the body: dense, near black, lit from the upper left, a hint of depth inside
    const body = ctx.createRadialGradient(cx - r * 0.36, cy - r * 0.40, r * 0.08, cx, cy, r);
    body.addColorStop(0, "#232019");
    body.addColorStop(0.22, "#15130f");
    body.addColorStop(0.55, "#0b0a09");
    body.addColorStop(0.85, "#060606");
    body.addColorStop(1, "#050505");
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = body; ctx.fill();
    // weight: the lower hemisphere falls away into shadow
    const mass = ctx.createLinearGradient(0, cy - r * 0.2, 0, cy + r);
    mass.addColorStop(0, "rgba(0,0,0,0)");
    mass.addColorStop(1, "rgba(0,0,0,.42)");
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = mass; ctx.fill();
    // a faint bounced warmth low right: the object sits in light
    const bounce = ctx.createRadialGradient(cx + r * 0.45, cy + r * 0.55, 0, cx + r * 0.45, cy + r * 0.55, r * 0.7);
    bounce.addColorStop(0, rgba(champagne, 0.035));
    bounce.addColorStop(1, rgba(champagne, 0));
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = bounce; ctx.fill();

    // the wave: one thin luminous line through the exact middle, a soft trace under it,
    // fine secondary traces for depth when the machine can afford them
    const amp = p.amp + activity * 0.09;
    const freq = p.freq + activity * 0.5;
    const complexity = Math.min(1, p.complexity + activity * 0.45);
    this.drawWave(ctx, cx, cy, r, champagne, 0.10 + energy * 0.08, 7, amp, freq, complexity, 0);
    if (this.quality > 0.65 && !reduced) {
      this.drawWave(ctx, cx, cy, r, deep, 0.22, 0.7, amp * 0.72, freq * 1.08, complexity, 0.9);
      this.drawWave(ctx, cx, cy, r, deep, 0.16, 0.6, amp * 0.55, freq * 0.93, complexity, -1.4);
    }
    this.drawWave(ctx, cx, cy, r, light, 0.88, 1.3, amp, freq, complexity, 0);

    // the limb from inside: the glass thickens towards the edge
    const limb = ctx.createRadialGradient(cx, cy, r * 0.84, cx, cy, r);
    limb.addColorStop(0, rgba(champagne, 0));
    limb.addColorStop(0.7, rgba(champagne, 0.05 * rim));
    limb.addColorStop(0.96, rgba(champagne, 0.16 * rim));
    limb.addColorStop(1, rgba(light, 0.30 * rim));
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = limb; ctx.fill();

    // the circumference: thin, catching light strongest at the upper left
    const width = Math.max(0.9, r * 0.011);
    let stroke;
    if (ctx.createConicGradient) {
      const g = ctx.createConicGradient(-Math.PI * 0.75, cx, cy);   // starts at the upper left
      g.addColorStop(0, rgba(light, 0.95 * rim));
      g.addColorStop(0.16, rgba(champagne, 0.55 * rim));
      g.addColorStop(0.42, rgba(deep, 0.18 * rim));
      g.addColorStop(0.58, rgba(deep, 0.14 * rim));
      g.addColorStop(0.82, rgba(champagne, 0.40 * rim));
      g.addColorStop(1, rgba(light, 0.95 * rim));
      stroke = g;
    } else {
      stroke = rgba(champagne, 0.45 * rim);
    }
    ctx.beginPath(); ctx.arc(cx, cy, r - width / 2, 0, Math.PI * 2);
    ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke();
    // a wider, fainter halo just outside the edge
    ctx.beginPath(); ctx.arc(cx, cy, r + 1, 0, Math.PI * 2);
    ctx.strokeStyle = rgba(champagne, 0.07 * rim + energy * 0.04); ctx.lineWidth = 3; ctx.stroke();

    // specular: one soft reflection and one small sharp one -- polished glass, not a bubble
    const hi = ctx.createRadialGradient(cx - r * 0.44, cy - r * 0.50, 0, cx - r * 0.44, cy - r * 0.50, r * 0.34);
    hi.addColorStop(0, "rgba(255,250,242,.20)");
    hi.addColorStop(0.5, "rgba(255,250,242,.05)");
    hi.addColorStop(1, "rgba(255,250,242,0)");
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = hi; ctx.fill();
    const spec = ctx.createRadialGradient(cx - r * 0.40, cy - r * 0.56, 0, cx - r * 0.40, cy - r * 0.56, r * 0.085);
    spec.addColorStop(0, "rgba(255,250,242,.34)");
    spec.addColorStop(0.55, "rgba(255,250,242,.10)");
    spec.addColorStop(1, "rgba(255,250,242,0)");
    ctx.beginPath(); ctx.ellipse(cx - r * 0.40, cy - r * 0.56, r * 0.11, r * 0.05, -0.62, 0, Math.PI * 2);
    ctx.fillStyle = spec; ctx.fill();
    // a cooler, barely visible reflection low right for physical truth
    ctx.beginPath(); ctx.ellipse(cx + r * 0.46, cy + r * 0.54, r * 0.10, r * 0.026, 0.8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(205,210,220,.025)"; ctx.fill();

    // success: one quiet outward pulse of the circumference, under 800 ms
    if (this.pulse > 0) {
      const k = 1 - this.pulse;
      ctx.beginPath(); ctx.arc(cx, cy, r * (1.0 + k * 0.16), 0, Math.PI * 2);
      ctx.strokeStyle = rgba(light, 0.45 * this.pulse);
      ctx.lineWidth = 1 + 1.5 * this.pulse;
      ctx.stroke();
    }
  }
}

window.JarvisEye = ZeusOrb;
window.ZeusOrb = ZeusOrb;
window.ZeusSphere = ZeusOrb;
window.JARVIS_STATES = STATES;
