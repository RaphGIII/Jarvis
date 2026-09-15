/* The ZEUS Orb: a refractive glass sphere with a spectral band circulating
   around its limb -- one object, rebuilt from the old radar eye.

   Canvas 2D only: layered radial gradients for the glass body and its
   refraction, a conic spectral ring drawn as short arcs (so the colour moves
   along the circumference instead of spinning as a texture), a chromatic
   halo offset per channel, and a soft volumetric glow.  No rings that blink,
   no crosshairs, no particle storms.

   States drive one parameter set, eased every frame: circulation speed,
   spectral saturation, energy (directional flow), warmth (the error shift)
   and a pulse (success).  The colour law stays the owner's: blue attending,
   green acting, warm amber on failure -- never a red alarm.

   API kept from the previous eye so the shell does not change:
   setState / setEnergy / setBackgroundWork / setThemeShift / setIntensity /
   start / stop / resize / pulse.  Registered as window.JarvisEye. */

const STATES = {
  idle:         { speed: 0.10, sat: 0.55, energy: 0.12, warmth: 0, hue: 205, breathe: 0.030 },
  listening:    { speed: 0.16, sat: 0.65, energy: 0.20, warmth: 0, hue: 200, breathe: 0.040 },
  transcribing: { speed: 0.22, sat: 0.70, energy: 0.25, warmth: 0, hue: 198, breathe: 0.040 },
  thinking:     { speed: 0.42, sat: 0.85, energy: 0.32, warmth: 0, hue: 210, breathe: 0.055 },
  speaking:     { speed: 0.26, sat: 0.75, energy: 0.35, warmth: 0, hue: 204, breathe: 0.060 },
  waiting:      { speed: 0.12, sat: 0.55, energy: 0.15, warmth: 0, hue: 214, breathe: 0.030 },
  working:      { speed: 0.36, sat: 0.85, energy: 0.55, warmth: 0, hue: 158, breathe: 0.050 },
  verifying:    { speed: 0.30, sat: 0.80, energy: 0.45, warmth: 0, hue: 160, breathe: 0.045 },
  coding:       { speed: 0.36, sat: 0.85, energy: 0.55, warmth: 0, hue: 155, breathe: 0.050 },
  researching:  { speed: 0.34, sat: 0.85, energy: 0.50, warmth: 0, hue: 165, breathe: 0.050 },
  error:        { speed: 0.14, sat: 0.70, energy: 0.20, warmth: 1, hue: 32,  breathe: 0.025 },
  offline:      { speed: 0.04, sat: 0.20, energy: 0.05, warmth: 0, hue: 215, breathe: 0.015 },
};
const SUCCESS_STATES = new Set(["idle", "waiting"]);
const WORK_STATES = new Set(["working", "verifying", "coding", "researching"]);

class ZeusOrb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = "offline";
    this.target = { ...STATES.offline };
    this.now = { ...STATES.offline };
    this.energyBoost = 0;
    this.work = 0;
    this.hueShift = 0;
    this.intensity = 1;
    this.phase = 0;
    this.breath = 0;
    this.pulse = 0;
    this.pulseAt = 0;
    this.running = false;
    this.last = 0;
    this.quality = 1;
    this.frameCost = 0;
    this.size = 0;
    this.resize();
  }

  // ---- api -----------------------------------------------------------
  setState(name) {
    const prev = this.state;
    this.state = STATES[name] ? name : "idle";
    this.target = { ...STATES[this.state] };
    if (WORK_STATES.has(prev) && SUCCESS_STATES.has(this.state)) this.pulse = 1; // work finished: one outward pulse
  }
  setEnergy(value) { this.energyBoost = Math.max(0, Math.min(1, Number(value) || 0)); }
  setBackgroundWork(count) { this.work = Math.max(0, Number(count) || 0); }
  setThemeShift(degrees) { this.hueShift = Number(degrees) || 0; }
  setIntensity(value) { this.intensity = Math.max(0, Math.min(1.6, Number(value) || 0)); }
  pulseOnce() { this.pulse = 1; }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const css = Math.max(48, Math.round(rect.width || this.canvas.width || 320));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.size = css;
    this.canvas.width = Math.round(css * dpr);
    this.canvas.height = Math.round(css * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
      // quality throttling: a slow machine gets a simpler orb, never a stutter
      if (this.frameCost > 9 && this.quality > 0.5) this.quality -= 0.05;
      else if (this.frameCost < 4 && this.quality < 1) this.quality += 0.02;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }

  // ---- motion --------------------------------------------------------
  step(dt) {
    const k = 1 - Math.exp(-dt * 3.2);
    for (const key of Object.keys(this.target)) {
      const to = this.target[key];
      if (key === "hue") {
        let d = ((to - this.now.hue + 540) % 360) - 180;
        this.now.hue += d * k;
      } else {
        this.now[key] += (to - this.now[key]) * k;
      }
    }
    const reduced = document.body.classList.contains("reduced-motion") || this.intensity === 0;
    const motion = reduced ? 0.15 : this.intensity;
    const energy = Math.min(1, this.now.energy + this.energyBoost * 0.6 + Math.min(0.3, this.work * 0.08));
    this.phase += dt * (0.25 + this.now.speed * 2.2 + energy * 0.8) * motion;
    this.breath += dt * (0.9 + this.now.breathe * 8) * motion;
    if (this.pulse > 0) this.pulse = Math.max(0, this.pulse - dt * 1.4);
  }

  // ---- drawing -------------------------------------------------------
  draw() {
    const ctx = this.ctx;
    const s = this.size;
    const c = s / 2;
    const q = this.quality;
    const p = this.now;
    const hue = (p.hue + this.hueShift + 360) % 360;
    const energy = Math.min(1, p.energy + this.energyBoost * 0.6 + Math.min(0.3, this.work * 0.08));
    const breath = 1 + Math.sin(this.breath) * p.breathe * 0.35;
    const r = s * 0.31 * breath;
    ctx.clearRect(0, 0, s, s);

    // volumetric glow behind the sphere
    const glow = ctx.createRadialGradient(c, c, r * 0.6, c, c, r * 2.4);
    glow.addColorStop(0, `hsla(${hue}, 90%, 62%, ${0.26 + energy * 0.2})`);
    glow.addColorStop(0.45, `hsla(${hue + 20}, 85%, 55%, ${0.10 + energy * 0.08})`);
    glow.addColorStop(1, "hsla(0,0%,0%,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, s, s);

    // the success pulse: one outward ring, fading
    if (this.pulse > 0) {
      const k = 1 - this.pulse;
      ctx.beginPath();
      ctx.arc(c, c, r * (1.05 + k * 1.1), 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(${hue}, 90%, 70%, ${0.5 * this.pulse})`;
      ctx.lineWidth = 2 + 6 * this.pulse;
      ctx.stroke();
    }

    // spectral band along the circumference: many short arcs, colour moving with the phase
    const segments = Math.round(72 * q) + 24;
    const bandWidth = r * (0.10 + energy * 0.05);
    for (let pass = 0; pass < 2; pass++) {
      const width = pass === 0 ? bandWidth * 1.9 : bandWidth;
      const alpha = pass === 0 ? 0.16 : 0.72;
      for (let i = 0; i < segments; i++) {
        const a0 = (i / segments) * Math.PI * 2;
        const a1 = ((i + 1.15) / segments) * Math.PI * 2;
        const t = ((i / segments) + this.phase * 0.25) % 1;
        // the spectrum: a controlled sweep around the base hue, wider with saturation
        const spread = 70 + p.sat * 90 + (p.warmth ? 30 : 0);
        const h = (hue - spread / 2 + spread * (0.5 + 0.5 * Math.sin(t * Math.PI * 2 + this.phase * 0.6)) + 360) % 360;
        const l = 55 + 18 * Math.sin(t * Math.PI * 4 - this.phase * 1.3);
        const sat = 60 + p.sat * 35;
        ctx.beginPath();
        ctx.arc(c, c, r * 1.02, a0, a1);
        ctx.strokeStyle = `hsla(${h}, ${sat}%, ${l}%, ${alpha * (0.6 + 0.4 * Math.sin(t * Math.PI * 2 * 3 + this.phase * 2))})`;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    }

    // chromatic dispersion: the band's fringes offset per channel
    if (q > 0.6) {
      for (const [dx, dy, col] of [[1.6, 0.4, `hsla(${(hue + 150) % 360}, 90%, 60%, .12)`], [-1.4, -0.6, `hsla(${(hue + 320) % 360}, 90%, 62%, .10)`]]) {
        ctx.beginPath();
        ctx.arc(c + dx, c + dy, r * 1.02, 0, Math.PI * 2);
        ctx.strokeStyle = col;
        ctx.lineWidth = bandWidth * 0.8;
        ctx.stroke();
      }
    }

    // the glass body: deep interior, refracted light, a bright rim
    const body = ctx.createRadialGradient(c - r * 0.32, c - r * 0.36, r * 0.05, c, c, r);
    body.addColorStop(0, `hsla(${hue}, 55%, ${28 + energy * 10}%, .96)`);
    body.addColorStop(0.55, `hsla(${hue + 10}, 60%, 10%, .97)`);
    body.addColorStop(1, `hsla(${hue}, 50%, 6%, 1)`);
    ctx.beginPath();
    ctx.arc(c, c, r, 0, Math.PI * 2);
    ctx.fillStyle = body;
    ctx.fill();

    // directional energy: a soft internal current whose angle follows the phase
    const flowA = this.phase * 0.7;
    const flow = ctx.createLinearGradient(c + Math.cos(flowA) * r, c + Math.sin(flowA) * r, c - Math.cos(flowA) * r, c - Math.sin(flowA) * r);
    flow.addColorStop(0, `hsla(${hue + 30}, 90%, 65%, ${0.05 + energy * 0.22})`);
    flow.addColorStop(0.5, "hsla(0,0%,0%,0)");
    flow.addColorStop(1, `hsla(${hue - 30}, 90%, 60%, ${0.03 + energy * 0.12})`);
    ctx.beginPath();
    ctx.arc(c, c, r * 0.98, 0, Math.PI * 2);
    ctx.fillStyle = flow;
    ctx.fill();

    // refraction: the limb bends the light -- a thin bright inner ring and a darker band inside it
    const limb = ctx.createRadialGradient(c, c, r * 0.78, c, c, r);
    limb.addColorStop(0, "hsla(0,0%,100%,0)");
    limb.addColorStop(0.72, `hsla(${hue}, 60%, 20%, .35)`);
    limb.addColorStop(0.92, `hsla(${hue}, 80%, 75%, ${0.35 + energy * 0.2})`);
    limb.addColorStop(1, `hsla(${hue}, 90%, 85%, .85)`);
    ctx.beginPath();
    ctx.arc(c, c, r, 0, Math.PI * 2);
    ctx.fillStyle = limb;
    ctx.fill();

    // specular highlights: one soft, one sharp -- the object is glass
    const hi = ctx.createRadialGradient(c - r * 0.42, c - r * 0.48, 0, c - r * 0.42, c - r * 0.48, r * 0.6);
    hi.addColorStop(0, "hsla(0,0%,100%,.55)");
    hi.addColorStop(0.35, "hsla(0,0%,100%,.10)");
    hi.addColorStop(1, "hsla(0,0%,100%,0)");
    ctx.beginPath();
    ctx.arc(c, c, r, 0, Math.PI * 2);
    ctx.fillStyle = hi;
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(c - r * 0.36, c - r * 0.52, r * 0.16, r * 0.07, -0.6, 0, Math.PI * 2);
    ctx.fillStyle = "hsla(0,0%,100%,.55)";
    ctx.fill();

    // the core: a small bright centre that breathes with the state
    const core = ctx.createRadialGradient(c, c, 0, c, c, r * (0.28 + energy * 0.12));
    core.addColorStop(0, `hsla(${hue}, 95%, ${78 + energy * 12}%, ${0.55 + energy * 0.35})`);
    core.addColorStop(0.6, `hsla(${hue + 15}, 90%, 60%, ${0.10 + energy * 0.15})`);
    core.addColorStop(1, "hsla(0,0%,0%,0)");
    ctx.beginPath();
    ctx.arc(c, c, r * 0.5, 0, Math.PI * 2);
    ctx.fillStyle = core;
    ctx.fill();

    // background work: a faint satellite on a slow orbit, one per job up to three
    for (let i = 0; i < Math.min(3, this.work); i++) {
      const a = this.phase * (0.35 + i * 0.07) + i * 2.1;
      const x = c + Math.cos(a) * r * 1.36;
      const y = c + Math.sin(a) * r * 1.36 * 0.55;
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue}, 90%, 78%, .85)`;
      ctx.fill();
    }
  }
}

window.JarvisEye = ZeusOrb;
window.ZeusOrb = ZeusOrb;
window.JARVIS_STATES = STATES;
