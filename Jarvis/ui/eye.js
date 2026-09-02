/*
 * The ZEUS eye.
 *
 * An abstract luminous aperture: concentric arcs, orbital rings, a rotating
 * iris lattice and a pupil that breathes. Drawn from scratch on a canvas -- no
 * images, no external assets, nothing derived from anyone else's artwork.
 *
 * The central idea is unchanged and is why this file is worth reading: the eye
 * has no per-state animations. It has a small set of continuous PARAMETERS --
 * pupil size, spin rate, glow, ring tension, jitter, hue -- and each state is a
 * target for those parameters. Every frame eases the live values towards the
 * target, so a state change is a physical settling rather than one animation
 * being swapped for another. That is what makes THINKING -> SPEAKING look like
 * the same entity changing what it is doing, instead of two clips cutting.
 *
 * COLOUR IS A CLAIM. Blue is the resting family: idle, listening, thinking,
 * speaking -- everything where ZEUS is attending or talking. Green means it is
 * acting or checking that its action worked. Red means something failed. There
 * are no other colours, because a colour nobody can name the meaning of is
 * decoration, and this one is supposed to be readable across a room.
 *
 * COST IS A CONSTRAINT. This runs on the same GPU that holds the local models,
 * so the eye must never be the reason a generation is slow. Three things keep
 * it cheap: static layers (technical markings, the noise tile) are baked once
 * into offscreen canvases and only redrawn on resize; the particle count and
 * ring count scale with the rendered size, so the 108px compact eye does a
 * fraction of the work of the hero eye; and the whole loop stops when the tab
 * is hidden and throttles itself if frames start costing too much.
 */

const BLUE = 202;
const GREEN = 148;
const RED = 356;

const STATES = {
  //                    pupil  spin  glow  tension jitter hue    arcs  breathe rings sweep
  idle:         { pupil: 0.30, spin: 0.10, glow: 0.38, tension: 1.00, jitter: 0.00, hue: BLUE,      arcs: 0.55, breathe: 0.030, rings: 0.55, sweep: 0.20 },
  listening:    { pupil: 0.45, spin: 0.22, glow: 0.88, tension: 1.10, jitter: 0.00, hue: BLUE - 10, arcs: 1.00, breathe: 0.090, rings: 0.85, sweep: 0.55 },
  transcribing: { pupil: 0.38, spin: 0.75, glow: 0.72, tension: 0.94, jitter: 0.28, hue: BLUE - 14, arcs: 0.85, breathe: 0.050, rings: 0.80, sweep: 0.70 },
  // Blue, but visibly busier: faster rings, a quicker sweep, a tighter pupil.
  thinking:     { pupil: 0.22, spin: 1.45, glow: 0.76, tension: 0.86, jitter: 0.10, hue: BLUE + 4,  arcs: 0.85, breathe: 0.022, rings: 1.00, sweep: 1.00 },
  speaking:     { pupil: 0.40, spin: 0.30, glow: 1.00, tension: 1.06, jitter: 0.00, hue: BLUE - 8,  arcs: 0.95, breathe: 0.140, rings: 0.70, sweep: 0.30 },
  // Green: acting on the world.
  working:      { pupil: 0.26, spin: 1.05, glow: 0.70, tension: 0.90, jitter: 0.06, hue: GREEN,     arcs: 0.85, breathe: 0.035, rings: 0.95, sweep: 0.80 },
  // Green, but narrow pupil and almost no spin: checking, not doing. "I did it"
  // and "I confirmed it" are different claims and should not look the same.
  verifying:    { pupil: 0.16, spin: 0.16, glow: 0.78, tension: 1.04, jitter: 0.02, hue: GREEN - 14, arcs: 0.95, breathe: 0.026, rings: 0.60, sweep: 1.00 },
  coding:       { pupil: 0.24, spin: 1.15, glow: 0.70, tension: 0.88, jitter: 0.05, hue: GREEN + 6, arcs: 0.80, breathe: 0.030, rings: 0.90, sweep: 0.70 },
  researching:  { pupil: 0.33, spin: 0.62, glow: 0.68, tension: 0.97, jitter: 0.12, hue: BLUE + 10, arcs: 0.85, breathe: 0.040, rings: 0.85, sweep: 0.60 },
  waiting:      { pupil: 0.34, spin: 0.06, glow: 0.44, tension: 1.02, jitter: 0.00, hue: BLUE + 6,  arcs: 0.60, breathe: 0.055, rings: 0.35, sweep: 0.10 },
  error:        { pupil: 0.30, spin: 0.05, glow: 0.85, tension: 0.80, jitter: 0.55, hue: RED,       arcs: 0.70, breathe: 0.020, rings: 0.45, sweep: 0.15 },
  offline:      { pupil: 0.14, spin: 0.02, glow: 0.14, tension: 1.00, jitter: 0.00, hue: BLUE + 18, arcs: 0.25, breathe: 0.010, rings: 0.15, sweep: 0.00 },
};

/* Orbital rings. Each is an ellipse with its own radius, speed, direction and
   vertical squash -- the squash is what reads as depth, because a circle seen
   at an angle is an ellipse and the brain does the rest. `depth` scales both
   the parallax offset and the brightness, so the far rings sit behind. */
const ORBITS = [
  { r: 0.94, squash: 0.30, speed: -0.22, depth: 0.30, width: 0.006, dash: 0 },
  { r: 0.86, squash: 0.86, speed: 0.34, depth: 0.55, width: 0.005, dash: 14 },
  { r: 0.80, squash: 0.16, speed: 0.48, depth: 0.85, width: 0.007, dash: 0 },
  { r: 0.71, squash: 0.62, speed: -0.62, depth: 1.00, width: 0.005, dash: 22 },
];

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

    /* Transition effects. `ripple` is a one-shot distortion fired on every
       state change; `glitch` is the harder, briefer version reserved for
       entering an error state. Both decay to zero and cost nothing at rest. */
    this.ripple = 0;
    this.glitch = 0;
    this.pulses = [];
    this.nextPulse = 3;

    this.particles = [];
    this.baked = { size: 0, markings: null, noise: null };

    // Adaptive quality. Starts full and drops if frames get expensive, so a
    // busy machine loses eye detail rather than model throughput.
    this.quality = 1;
    this._frameCost = 8;
  }

  setState(name) {
    if (!STATES[name] || name === this.state) return;
    const wasError = this.state === "error";
    this.state = name;
    this.ripple = 1;
    if (name === "error" && !wasError) this.glitch = 1;
  }

  setEnergy(value) {
    this.energy = Math.max(0, Math.min(1, value || 0));
  }

  /* Match the backing store to the element's rendered size, so the hero eye is
     crisp rather than a 210px canvas stretched to 440. Capped at 2x: beyond
     that the pixel count doubles again for a difference nobody can see. */
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const size = Math.max(64, Math.round(Math.min(rect.width, rect.height) * dpr));
    if (!size || size === this.canvas.width) return;
    this.canvas.width = size;
    this.canvas.height = size;
    this.bake(size);
    this.seedParticles(size);
  }

  /* ---- static layers, drawn once per size ------------------------------ */

  bake(size) {
    this.baked.size = size;
    this.baked.markings = this.bakeMarkings(size);
    this.baked.noise = this.bakeNoise(Math.min(128, Math.round(size / 3)));
  }

  /* Fine technical markings: major ticks, minor ticks, and short dashes that
     read as instrument lettering without being text in any language. */
  bakeMarkings(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const unit = size / 2;
    ctx.translate(unit, unit);
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    const majors = 24;
    const minors = 144;
    ctx.lineWidth = Math.max(1, unit * 0.004);
    for (let i = 0; i < minors; i++) {
      const a = (i / minors) * Math.PI * 2;
      const long = i % (minors / majors) === 0;
      const r0 = unit * (long ? 0.955 : 0.972);
      const r1 = unit * 0.99;
      ctx.globalAlpha = long ? 0.85 : 0.35;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
      ctx.lineTo(Math.cos(a) * r1, Math.sin(a) * r1);
      ctx.stroke();
    }
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = Math.max(1, unit * 0.006);
    for (let i = 0; i < majors; i++) {
      const a = (i / majors) * Math.PI * 2 + 0.04;
      const r = unit * 0.935;
      const len = ((i * 7) % 3) + 1;
      for (let d = 0; d < len; d++) {
        const aa = a + d * 0.012;
        ctx.beginPath();
        ctx.moveTo(Math.cos(aa) * r, Math.sin(aa) * r);
        ctx.lineTo(Math.cos(aa) * (r - unit * 0.018), Math.sin(aa) * (r - unit * 0.018));
        ctx.stroke();
      }
    }
    return c;
  }

  /* A small noise tile, tiled over the eye at very low alpha. Baked because
     generating noise per frame is the single most expensive thing this file
     could do, and it looks identical when it stands still. */
  bakeNoise(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const image = ctx.createImageData(size, size);
    for (let i = 0; i < image.data.length; i += 4) {
      const v = (Math.random() * 255) | 0;
      image.data[i] = image.data[i + 1] = image.data[i + 2] = v;
      image.data[i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return c;
  }

  seedParticles(size) {
    // Scaled to area, so the compact eye carries a handful and the hero eye a
    // field. Capped so a very large canvas cannot run away.
    const count = Math.max(10, Math.min(90, Math.round(size / 7)));
    this.particles = Array.from({ length: count }, () => ({
      a: Math.random() * Math.PI * 2,
      r: 0.30 + Math.random() * 0.68,
      speed: (0.05 + Math.random() * 0.28) * (Math.random() < 0.5 ? -1 : 1),
      size: 0.4 + Math.random() * 1.5,
      depth: 0.25 + Math.random() * 0.75,
      twinkle: Math.random() * Math.PI * 2,
    }));
  }

  /* ---- loop ------------------------------------------------------------- */

  start() {
    if (this.running) return;
    this.running = true;
    this.resize();
    const frame = (now) => {
      if (!this.running) return;
      requestAnimationFrame(frame);
      // A hidden tab still gets rAF in some browsers; skipping the work is
      // free and matters on a machine that is also holding two models.
      if (document.hidden) { this.lastFrame = now; return; }
      const dt = Math.min(0.05, (now - this.lastFrame) / 1000);
      this.lastFrame = now;
      const started = performance.now();
      this.step(dt);
      this.draw();
      this.governor(performance.now() - started);
    };
    requestAnimationFrame(frame);
  }

  stop() { this.running = false; }

  /* Drop detail if drawing starts costing real time. The eye degrades; the
     models do not. */
  governor(cost) {
    this._frameCost += (cost - this._frameCost) * 0.05;
    if (this._frameCost > 9 && this.quality > 0.35) this.quality -= 0.02;
    else if (this._frameCost < 4 && this.quality < 1) this.quality += 0.01;
  }

  step(dt) {
    const target = STATES[this.state] || STATES.idle;
    // Critically-damped-ish easing: fast enough to feel responsive, slow
    // enough that a burst of rapid state changes reads as one movement.
    const k = 1 - Math.exp(-dt * 4.5);
    for (const key of Object.keys(target)) {
      if (key === "hue") {
        // Hue travels the short way round the colour wheel, otherwise a jump
        // from red (356) to green (148) sweeps through every hue in between.
        let delta = target.hue - this.live.hue;
        if (delta > 180) delta -= 360;
        if (delta < -180) delta += 360;
        this.live.hue = (this.live.hue + delta * k + 360) % 360;
      } else {
        this.live[key] = (this.live[key] ?? target[key]) + (target[key] - (this.live[key] ?? target[key])) * k;
      }
    }

    this.phase += dt;
    this.spinAngle += dt * this.live.spin;
    this.ripple = Math.max(0, this.ripple - dt * 1.7);
    this.glitch = Math.max(0, this.glitch - dt * 2.6);

    for (const p of this.particles) {
      p.a += dt * p.speed * (0.35 + this.live.spin * 0.5);
      p.twinkle += dt * 2.2;
    }

    // Energy pulses: an occasional ring travelling outward, or inward when the
    // eye is verifying -- gathering rather than emitting.
    this.nextPulse -= dt * (0.4 + this.live.rings);
    if (this.nextPulse <= 0) {
      this.nextPulse = 2.2 + Math.random() * 3.4;
      this.pulses.push({ t: 0, inward: this.state === "verifying" || this.state === "listening" });
    }
    for (const pulse of this.pulses) pulse.t += dt * 0.85;
    this.pulses = this.pulses.filter((pulse) => pulse.t < 1);

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
    if (!size) return;
    if (this.baked.size !== size) this.bake(size);
    const mid = size / 2;
    const unit = size / 2;
    const detail = this.quality * Math.min(1, size / 260); // compact eye draws less

    ctx.clearRect(0, 0, size, size);

    const hue = this.live.hue;
    const glow = this.live.glow;
    const stroke = (l, a) => `hsla(${hue.toFixed(0)}, 85%, ${l}%, ${a})`;

    const breath = 1 + Math.sin(this.phase * 2.0) * this.live.breathe + this.energy * 0.10;
    const jitter = () => (Math.random() - 0.5) * this.live.jitter * unit * 0.035;
    const lidClose = this.blink * 0.92;

    // The ripple briefly swells and settles the whole assembly, so a state
    // change is felt as one movement rather than seen as a colour swap.
    const ripple = Math.sin(this.ripple * Math.PI) * 0.035;

    ctx.save();
    ctx.translate(mid, mid);

    // A glitch displaces the whole eye by a pixel or two for a few frames.
    if (this.glitch > 0.01) {
      ctx.translate((Math.random() - 0.5) * unit * 0.05 * this.glitch,
                    (Math.random() - 0.5) * unit * 0.02 * this.glitch);
    }
    ctx.scale(1 + ripple, 1 + ripple);

    this.drawHalo(ctx, unit, hue, glow);
    if (detail > 0.35) this.drawParticles(ctx, unit, hue, glow, detail);
    if (detail > 0.5) this.drawMarkings(ctx, unit, size, glow);
    this.drawOrbits(ctx, unit, stroke, glow, detail);
    if (detail > 0.45) this.drawSweep(ctx, unit, hue, glow);
    this.drawSegmentedRing(ctx, unit, stroke, glow, jitter);
    this.drawArcs(ctx, unit, stroke, glow);
    this.drawPulses(ctx, unit, hue, glow);
    const irisR = this.drawIris(ctx, unit, hue, stroke, glow, breath, detail);
    this.drawCore(ctx, irisR, hue, glow);
    if (detail > 0.6) this.drawNoise(ctx, unit, size);

    if (this.glitch > 0.01) this.drawGlitchBands(ctx, unit);

    if (lidClose > 0.001) {
      ctx.fillStyle = "#05070c";
      const h = unit * lidClose;
      ctx.fillRect(-unit, -unit, unit * 2, h);
      ctx.fillRect(-unit, unit - h, unit * 2, h);
    }

    ctx.restore();
  }

  /* ---- layers ----------------------------------------------------------- */

  drawHalo(ctx, unit, hue, glow) {
    const halo = ctx.createRadialGradient(0, 0, unit * 0.05, 0, 0, unit * 0.98);
    halo.addColorStop(0, `hsla(${hue}, 90%, 60%, ${0.20 * glow})`);
    halo.addColorStop(0.45, `hsla(${hue}, 90%, 50%, ${0.07 * glow})`);
    halo.addColorStop(1, "hsla(0, 0%, 0%, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(0, 0, unit * 0.98, 0, Math.PI * 2);
    ctx.fill();
  }

  drawParticles(ctx, unit, hue, glow, detail) {
    const count = Math.round(this.particles.length * detail);
    for (let i = 0; i < count; i++) {
      const p = this.particles[i];
      const twinkle = 0.45 + 0.55 * Math.sin(p.twinkle);
      const r = unit * p.r * (1 + Math.sin(this.phase * 0.6 + p.a) * 0.01);
      ctx.globalAlpha = twinkle * p.depth * 0.55 * glow;
      ctx.fillStyle = `hsl(${hue}, 90%, ${58 + p.depth * 18}%)`;
      ctx.beginPath();
      ctx.arc(Math.cos(p.a) * r, Math.sin(p.a) * r, p.size * p.depth * (unit / 210), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  drawMarkings(ctx, unit, size, glow) {
    const markings = this.baked.markings;
    if (!markings) return;
    ctx.save();
    ctx.rotate(this.spinAngle * 0.12);
    ctx.globalAlpha = 0.20 * glow;
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(markings, -unit, -unit, size, size);
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  drawOrbits(ctx, unit, stroke, glow, detail) {
    const shown = Math.max(1, Math.round(ORBITS.length * Math.min(1, detail + 0.35)));
    for (let i = 0; i < shown; i++) {
      const orbit = ORBITS[i];
      const angle = this.spinAngle * orbit.speed + i * 1.3;
      // Parallax: the nearer rings drift further from centre as the assembly
      // turns, which is what stops this reading as a flat diagram.
      const drift = unit * 0.012 * orbit.depth;
      const ox = Math.cos(angle * 0.7) * drift;
      const oy = Math.sin(angle * 0.9) * drift;
      ctx.save();
      ctx.translate(ox, oy);
      ctx.rotate(angle);
      ctx.scale(1, orbit.squash);
      ctx.lineWidth = Math.max(1, unit * orbit.width);
      ctx.strokeStyle = stroke(58 + orbit.depth * 20, (0.10 + 0.34 * orbit.depth) * this.live.rings * glow);
      if (orbit.dash) ctx.setLineDash([orbit.dash * (unit / 210), orbit.dash * 0.8 * (unit / 210)]);
      ctx.beginPath();
      ctx.arc(0, 0, unit * orbit.r * this.live.tension, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      // Electrons: one or two quanta riding each shell, Bohr-style. Drawn
      // outside the squashed transform so they stay round; their position is
      // the ellipse point rotated by the ring's own angle.
      if (detail > 0.4) {
        const R = unit * orbit.r * this.live.tension;
        const ca = Math.cos(angle), sa = Math.sin(angle);
        const count = 1 + (i % 2);
        for (let e = 0; e < count; e++) {
          const th = this.spinAngle * (0.9 + Math.abs(orbit.speed) * 2.4) * (orbit.speed < 0 ? -1 : 1) + e * Math.PI + i * 2.1;
          const lx = Math.cos(th) * R, ly = Math.sin(th) * R * orbit.squash;
          const ex = ox + lx * ca - ly * sa, ey = oy + lx * sa + ly * ca;
          const a = (0.30 + 0.45 * orbit.depth) * this.live.rings * glow;
          const er = Math.max(1, unit * 0.014 * (0.7 + orbit.depth * 0.5));
          ctx.fillStyle = stroke(70, a * 0.35);
          ctx.beginPath(); ctx.arc(ex, ey, er * 2.6, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = stroke(86, a);
          ctx.beginPath(); ctx.arc(ex, ey, er, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
  }

  /* A radial scanning sweep: a soft wedge of light rotating over the face. */
  drawSweep(ctx, unit, hue, glow) {
    const strength = this.live.sweep;
    if (strength < 0.05) return;
    const a = this.spinAngle * 1.1;
    const span = 0.55;
    const wedge = ctx.createLinearGradient(0, 0, Math.cos(a) * unit, Math.sin(a) * unit);
    wedge.addColorStop(0, `hsla(${hue}, 95%, 70%, 0)`);
    wedge.addColorStop(1, `hsla(${hue}, 95%, 72%, ${0.13 * strength * glow})`);
    ctx.fillStyle = wedge;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, unit * 0.92, a - span, a + span);
    ctx.closePath();
    ctx.fill();
  }

  drawSegmentedRing(ctx, unit, stroke, glow, jitter) {
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
  }

  drawArcs(ctx, unit, stroke, glow) {
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
  }

  drawPulses(ctx, unit, hue, glow) {
    for (const pulse of this.pulses) {
      const t = pulse.inward ? 1 - pulse.t : pulse.t;
      const fade = Math.sin(pulse.t * Math.PI);
      ctx.strokeStyle = `hsla(${hue}, 95%, 72%, ${0.30 * fade * glow})`;
      ctx.lineWidth = Math.max(1, unit * 0.006 * (1 + fade));
      ctx.beginPath();
      ctx.arc(0, 0, unit * (0.28 + t * 0.64), 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  drawIris(ctx, unit, hue, stroke, glow, breath, detail) {
    const irisR = unit * 0.52 * this.live.tension * breath;
    const blades = detail > 0.6 ? 32 : 16;
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
    return irisR;
  }

  /* The core: a pupil with its own slower breathing, so the centre feels alive
     independently of the rings turning around it. */
  drawCore(ctx, irisR, hue, glow) {
    const coreBreath = 1 + Math.sin(this.phase * 1.15) * 0.10;
    const pupilR = Math.max(1, irisR * this.live.pupil * coreBreath * (1 + this.energy * 0.28));
    const pupil = ctx.createRadialGradient(0, 0, 0, 0, 0, pupilR);
    pupil.addColorStop(0, `hsla(${hue}, 100%, 90%, ${0.95 * Math.max(0.25, glow)})`);
    pupil.addColorStop(0.5, `hsla(${hue}, 98%, 66%, ${0.60 * glow})`);
    pupil.addColorStop(1, `hsla(${hue}, 90%, 40%, 0)`);
    ctx.fillStyle = pupil;
    ctx.beginPath();
    ctx.arc(0, 0, pupilR, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = `hsla(${hue}, 100%, 96%, ${0.55 * glow})`;
    ctx.beginPath();
    ctx.arc(0, 0, pupilR * 0.30, 0, Math.PI * 2);
    ctx.fill();
  }

  drawNoise(ctx, unit, size) {
    const noise = this.baked.noise;
    if (!noise) return;
    ctx.save();
    // Clipped to the eye, not the canvas. `overlay` against transparent pixels
    // paints visible grey, which drew a faint square around the eye where the
    // page should have shown through -- the interference read as a bounding box.
    ctx.beginPath();
    ctx.arc(0, 0, unit * 0.97, 0, Math.PI * 2);
    ctx.clip();
    ctx.globalAlpha = 0.022;
    ctx.globalCompositeOperation = "overlay";
    // Nudged each frame so it shimmers instead of sitting there as a texture.
    const ox = (this.phase * 37) % noise.width;
    const oy = (this.phase * 23) % noise.height;
    ctx.drawImage(noise, -unit - ox, -unit - oy, size + noise.width, size + noise.height);
    ctx.restore();
  }

  /* Error glitch: a couple of displaced horizontal slices. Brief and
     controlled -- it should read as interference, not as a broken renderer. */
  drawGlitchBands(ctx, unit) {
    const bands = 3;
    ctx.save();
    ctx.beginPath();
    ctx.arc(0, 0, unit * 0.97, 0, Math.PI * 2);
    ctx.clip();
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < bands; i++) {
      const y = (Math.random() - 0.5) * unit * 1.6;
      const h = unit * (0.02 + Math.random() * 0.05);
      ctx.fillStyle = `hsla(${RED}, 95%, 62%, ${0.10 * this.glitch})`;
      ctx.fillRect(-unit + (Math.random() - 0.5) * unit * 0.2, y, unit * 2, h);
    }
    ctx.restore();
  }
}

window.JarvisEye = JarvisEye;
window.JARVIS_STATES = STATES;
