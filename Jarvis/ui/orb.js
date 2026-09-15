/* The ZEUS core: a procedural black hole.

   A perfectly black event horizon, a thin photon ring around it, a luminous
   accretion band of ivory light passing in front of the horizon and, lensed
   over the top and under the bottom, the far side of that same band -- the
   look of a massive object bending light, drawn from scratch as geometry and
   gradients.  No asset, no texture, no particles.

   Everything is driven by a small parameter set eased every frame (delta
   time), so states flow into each other:

     idle          slow flow along the band, low light
     listening / thinking      faster flow, brighter halo
     speaking / working        stronger, wider band
     streaming answer          the band's edge breathes with the output
                               (an activity envelope: attack, release)
     error                     a warm shift, slower
     offline                   almost still, very dim

   API kept for the shell: setState / setEnergy / setActivity / noteOutput /
   setAudioFrequencyData / setBackgroundWork / setThemeShift / setIntensity /
   pulseOnce / start / stop / resize.  Registered as window.JarvisEye,
   window.ZeusOrb and window.ZeusSphere. */

const STATES = {
  idle:         { flow: 0.55, light: 0.62, band: 1.00, halo: 0.55, warmth: 0, energy: 0.10, wave: 0.020 },
  listening:    { flow: 0.85, light: 0.74, band: 1.04, halo: 0.68, warmth: 0, energy: 0.22, wave: 0.030 },
  transcribing: { flow: 0.95, light: 0.76, band: 1.05, halo: 0.70, warmth: 0, energy: 0.26, wave: 0.034 },
  thinking:     { flow: 1.35, light: 0.86, band: 1.08, halo: 0.86, warmth: 0, energy: 0.40, wave: 0.048 },
  speaking:     { flow: 1.15, light: 0.92, band: 1.14, halo: 0.88, warmth: 0, energy: 0.50, wave: 0.060 },
  waiting:      { flow: 0.50, light: 0.64, band: 1.00, halo: 0.58, warmth: 0, energy: 0.12, wave: 0.022 },
  working:      { flow: 1.20, light: 0.90, band: 1.12, halo: 0.90, warmth: 0, energy: 0.48, wave: 0.050 },
  verifying:    { flow: 1.05, light: 0.86, band: 1.08, halo: 0.86, warmth: 0, energy: 0.42, wave: 0.044 },
  coding:       { flow: 1.20, light: 0.90, band: 1.12, halo: 0.90, warmth: 0, energy: 0.48, wave: 0.050 },
  researching:  { flow: 1.10, light: 0.88, band: 1.10, halo: 0.88, warmth: 0, energy: 0.45, wave: 0.046 },
  error:        { flow: 0.35, light: 0.55, band: 0.98, halo: 0.50, warmth: 1, energy: 0.15, wave: 0.016 },
  offline:      { flow: 0.12, light: 0.30, band: 0.94, halo: 0.25, warmth: 0, energy: 0.03, wave: 0.006 },
};
const SUCCESS_STATES = new Set(["idle", "waiting"]);
const WORK_STATES = new Set(["working", "verifying", "coding", "researching"]);

// the light: ivory and champagne, graphite for what lies behind, a warm shift on failure
const IVORY = [247, 240, 226];
const CHAMPAGNE = [226, 206, 170];
const DEEP = [196, 176, 138];
const GRAPHITE = [120, 116, 108];
const WARM = [222, 156, 118];

const TAU = Math.PI * 2;
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
    this.inflow = 0;
    this.rate = 0;
    this.activity = 0;
    this.external = 0;
    this.audio = null;
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
    this.w = 0; this.h = 0; this.u = 0;
    this.resize();
  }

  // ---- api -----------------------------------------------------------
  setState(name) {
    const prev = this.state;
    this.state = STATES[name] ? name : "idle";
    this.target = { ...STATES[this.state] };
    if (WORK_STATES.has(prev) && SUCCESS_STATES.has(this.state)) this.pulse = 1;
  }
  setEnergy(value) { this.external = Math.max(this.external * 0.5, clamp01(value)); }
  setActivity(level) { this.external = clamp01(level); }
  noteOutput(chars) { this.inflow += Math.max(0, Number(chars) || 0); }
  setAudioFrequencyData(bins) { this.audio = bins && bins.length ? bins : null; }
  setBackgroundWork(count) { this.work = Math.max(0, Number(count) || 0); }
  setThemeShift(degrees) { this.hueShift = Number(degrees) || 0; }
  setIntensity(value) { this.intensity = Math.max(0, Math.min(1.6, Number(value) || 0)); }
  pulseOnce() { this.pulse = 1; }

  /* Sharp at every scale: the CSS box is rounded to whole pixels and the backing store matches it
     times the real device pixel ratio, so nothing is resampled. */
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(48, Math.round(rect.width || this.canvas.width || 420));
    const h = Math.max(34, Math.round(rect.height || w / 1.4));
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    this.w = w; this.h = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // the unit: the black hole fits the height, the band fits the width
    this.u = Math.min(h / 2.15, w / 3.05);
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
    const k = 1 - Math.exp(-dt * 2.6);
    for (const key of Object.keys(this.target)) this.now[key] += (this.target[key] - this.now[key]) * k;
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
    this.external *= Math.exp(-dt * 1.2);
    const motion = this.reduced() ? 0.12 : this.intensity;
    this.t += dt * (0.35 + this.now.flow * 0.9 + this.activity * 1.4) * motion;
    if (this.pulse > 0) this.pulse = Math.max(0, this.pulse - dt / 0.75);
  }

  /* the band's edge: a slow drift, and with activity an equalizer-like breath along the ring */
  edge(theta, amp, complexity) {
    const t = this.t;
    const a = Math.sin(theta * 3 + t * 1.3) * 0.5 + Math.sin(theta * 5 - t * 2.1 + 0.7) * 0.32 * complexity + Math.sin(theta * 9 + t * 3.4) * 0.18 * complexity;
    let audio = 0;
    if (this.audio) {
      const i = Math.min(this.audio.length - 1, Math.floor(((theta / TAU) % 1) * Math.min(this.audio.length, 40)));
      audio = (this.audio[i] / 255) * 0.9;
    }
    return amp * (a + audio);
  }

  /* one half of the accretion band (front: below the centre, back: above), as a closed path */
  bandPath(ctx, cx, cy, ai, bi, ao, bo, front, amp, complexity, tilt) {
    const steps = this.quality > 0.7 ? 96 : 56;
    const from = front ? 0 : Math.PI, to = front ? Math.PI : TAU;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const pt = (a, b, th, lift) => {
      const x0 = Math.cos(th) * a, y0 = Math.sin(th) * b - lift;
      return [cx + x0 * ct - y0 * st, cy + x0 * st + y0 * ct];
    };
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const th = from + (to - from) * (i / steps);
      const e = 1 + this.edge(th, amp, complexity);
      const [x, y] = pt(ao * e, bo * e, th, 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    for (let i = steps; i >= 0; i--) {
      const th = from + (to - from) * (i / steps);
      const [x, y] = pt(ai, bi, th, 0);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
  }

  draw() {
    const ctx = this.ctx;
    const { w, h, u } = this;
    const cx = Math.round(w / 2), cy = Math.round(h / 2) + Math.round(u * 0.04);
    const p = this.now;
    const reduced = this.reduced();
    const activity = reduced ? this.activity * 0.4 : this.activity;
    const energy = clamp01(p.energy + activity * 0.5 + Math.min(0.2, this.work * 0.07));
    const warm = clamp01(p.warmth);
    const ivory = mix(IVORY, WARM, warm * 0.55);
    const champagne = mix(CHAMPAGNE, WARM, warm * 0.7);
    const deep = mix(DEEP, WARM, warm * 0.6);
    const light = clamp01(p.light + activity * 0.25);
    const halo = clamp01(p.halo + activity * 0.2);
    const amp = p.wave + activity * 0.075;
    const complexity = Math.min(1, 0.35 + activity * 0.65);
    const tilt = -0.075;
    // geometry
    const rh = u * 0.60;                       // the horizon
    const ai = u * 0.70, bi = u * 0.155;       // inner edge of the band
    const ao = u * 1.44 * p.band, bo = u * 0.36 * p.band;
    ctx.clearRect(0, 0, w, h);

    // ambient: the object warms the space around it, barely
    const amb = ctx.createRadialGradient(cx, cy, rh * 0.8, cx, cy, u * 1.7);
    amb.addColorStop(0, rgba(champagne, 0.07 * light + energy * 0.03));
    amb.addColorStop(0.55, rgba(champagne, 0.02 * light));
    amb.addColorStop(1, rgba(champagne, 0));
    ctx.fillStyle = amb;
    ctx.fillRect(0, 0, w, h);

    // ---- behind the horizon: the far side of the band, and the lensed halo over the top
    ctx.save();
    // back half of the band (fainter, graphite into champagne), flowing
    this.bandPath(ctx, cx, cy, ai, bi, ao * 0.96, bo * 0.92, false, amp * 0.7, complexity, tilt);
    let g = ctx.createLinearGradient(cx - ao, 0, cx + ao, 0);
    g.addColorStop(0, rgba(deep, 0.50 * light));
    g.addColorStop(0.5, rgba(GRAPHITE, 0.26 * light));
    g.addColorStop(1, rgba(deep, 0.18 * light));
    ctx.fillStyle = g;
    ctx.fill();
    // the lensed far side: a thin bright arc over the top of the horizon and a fainter one below
    // drawn as short segments so the light fades out towards the ends and pulses along the arc
    const lens = rh * 1.30;
    const lensArc = (r, from, to, width, peak) => {
      const n = 28;
      for (let i = 0; i < n; i++) {
        const a0 = from + (to - from) * (i / n), a1 = from + (to - from) * ((i + 1.15) / n);
        const f = i / (n - 1);
        const fade = Math.sin(f * Math.PI);
        const travel = 0.75 + 0.25 * Math.sin(f * 9 - this.t * 1.7);
        ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1);
        ctx.lineWidth = width * (0.6 + 0.4 * fade); ctx.strokeStyle = rgba(ivory, peak * fade * travel); ctx.lineCap = "butt"; ctx.stroke();
      }
    };
    lensArc(lens, Math.PI * 1.06, Math.PI * 1.94, u * 0.052, halo * 0.62);
    g = ctx.createRadialGradient(cx, cy, lens - u * 0.05, cx, cy, lens + u * 0.16);
    g.addColorStop(0, rgba(champagne, 0.16 * halo)); g.addColorStop(1, rgba(champagne, 0));
    ctx.beginPath(); ctx.arc(cx, cy, lens + u * 0.16, Math.PI, TAU); ctx.lineTo(cx - lens, cy); ctx.fillStyle = g; ctx.fill();
    lensArc(lens * 0.97, Math.PI * 0.14, Math.PI * 0.86, u * 0.030, halo * 0.24);
    ctx.restore();

    // ---- the photon ring: the thin, brightest circle, alive with light travelling along it
    const ring = rh * 1.045;
    const segs = this.quality > 0.7 ? 96 : 48;
    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * TAU, a1 = ((i + 1.2) / segs) * TAU;
      const th = (a0 + a1) / 2;
      const travel = 0.5 + 0.5 * Math.sin(th * 2 - this.t * 2.2) * Math.sin(th * 5 + this.t * 1.1);
      const doppler = 0.66 + 0.34 * Math.cos(th - Math.PI * 0.92);   // brightest where the light comes toward us
      const alpha = light * (0.62 + 0.36 * travel) * doppler;
      ctx.beginPath(); ctx.arc(cx, cy, ring, a0, a1 + 0.02);
      ctx.lineWidth = u * 0.020 + travel * u * 0.008 + energy * u * 0.006;
      ctx.strokeStyle = rgba(ivory, alpha);
      ctx.lineCap = "butt";
      ctx.stroke();
    }
    // its soft glow outward
    g = ctx.createRadialGradient(cx, cy, ring, cx, cy, ring + u * 0.22);
    g.addColorStop(0, rgba(champagne, 0.28 * light));
    g.addColorStop(1, rgba(champagne, 0));
    ctx.beginPath(); ctx.arc(cx, cy, ring + u * 0.22, 0, TAU);
    ctx.fillStyle = g; ctx.fill();

    // ---- the horizon: perfectly black, razor edged
    ctx.beginPath(); ctx.arc(cx, cy, rh, 0, TAU);
    ctx.fillStyle = "#000000"; ctx.fill();
    // the inner edge of the horizon catches nothing; a hair of graphite marks where light stops
    ctx.beginPath(); ctx.arc(cx, cy, rh, 0, TAU);
    ctx.lineWidth = 1; ctx.strokeStyle = rgba(GRAPHITE, 0.35 * light); ctx.stroke();

    // ---- in front of the horizon: the near side of the band, the brightest matter
    ctx.save();
    this.bandPath(ctx, cx, cy, ai, bi, ao, bo, true, amp, complexity, tilt);
    ctx.clip();
    // brightness: strongest at the inner edge, decaying outward (radial), beamed to the left (linear)
    g = ctx.createRadialGradient(cx, cy, ai * 0.9, cx, cy, ao);
    g.addColorStop(0, rgba(ivory, 0.98 * light));
    g.addColorStop(0.22, rgba(champagne, 0.82 * light));
    g.addColorStop(0.62, rgba(deep, 0.42 * light));
    g.addColorStop(1, rgba(deep, 0.06 * light));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    g = ctx.createLinearGradient(cx - ao, 0, cx + ao, 0);
    g.addColorStop(0, rgba(ivory, 0.35 * light));
    g.addColorStop(0.45, rgba(ivory, 0));
    g.addColorStop(1, "rgba(0,0,0,.45)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    // flowing structure: fine elliptical traces moving along the band
    if (!reduced || true) {
      const traces = this.quality > 0.7 ? 9 : 5;
      ctx.lineCap = "round";
      for (let i = 0; i < traces; i++) {
        const f = (i + 0.5) / traces;
        const a = ai + (ao - ai) * f, b = bi + (bo - bi) * f;
        const phase = this.t * (1.6 + f * 0.9) + i * 1.9;
        const parts = 14;
        for (let s = 0; s < parts; s++) {
          const th0 = (s / parts) * Math.PI, th1 = ((s + 0.7) / parts) * Math.PI;
          const shimmer = 0.5 + 0.5 * Math.sin(th0 * 4 + phase);
          if (shimmer < 0.35) continue;
          ctx.beginPath();
          ctx.ellipse(cx, cy, a, b, tilt, th0, th1);
          ctx.lineWidth = 0.9 + (1 - f) * 0.9;
          ctx.strokeStyle = rgba(ivory, (0.10 + 0.32 * shimmer) * light * (1 - f * 0.55));
          ctx.stroke();
        }
      }
    }
    ctx.restore();
    // a crisp light edge where the band meets the horizon
    ctx.save();
    ctx.beginPath(); ctx.arc(cx, cy, rh, 0, TAU); ctx.clip();
    ctx.beginPath(); ctx.ellipse(cx, cy, ai, bi, tilt, 0, Math.PI);
    ctx.lineWidth = 1.2; ctx.strokeStyle = rgba(ivory, 0.55 * light); ctx.stroke();
    ctx.restore();

    // ---- success: one quiet outward pulse of the photon ring
    if (this.pulse > 0) {
      const k = 1 - this.pulse;
      ctx.beginPath(); ctx.arc(cx, cy, ring * (1 + k * 0.22), 0, TAU);
      ctx.strokeStyle = rgba(ivory, 0.5 * this.pulse);
      ctx.lineWidth = 1 + 1.6 * this.pulse;
      ctx.stroke();
    }
  }
}

window.JarvisEye = ZeusOrb;
window.ZeusOrb = ZeusOrb;
window.ZeusSphere = ZeusOrb;
window.JARVIS_STATES = STATES;
