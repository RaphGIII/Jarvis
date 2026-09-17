/* The environment: a dark body of water at night, drawn once.

   Almost black water, a calm surface, distant dark mountains, a little mist,
   a warm trace of light on the horizon and its reflection.  Painted
   procedurally into a canvas from a fixed seed -- the same scene every time,
   no asset, no stock photograph -- and redrawn only when the window size
   changes.  It never moves: the only motion on the home screen is ZEUS. */

const SEED = 20260917;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* A distant mountain ridge: a few long, soft swells with a little detail on top, never a sawtooth. */
function ridge(width, random, { base, height, roughness, peaks }) {
  const points = [];
  const octaves = [];
  for (let o = 0; o < 4; o++) octaves.push({ f: (peaks * (o + 1) ** 1.55) / width, p: random() * 1000, a: 1 / (o + 1) ** (2.3 - roughness) });
  const norm = octaves.reduce((sum, o) => sum + o.a, 0);
  for (let x = 0; x <= width; x += 2) {
    let y = 0;
    for (const o of octaves) y += Math.sin(x * o.f * Math.PI * 2 + o.p) * o.a;
    // the range rises towards one side and settles towards the other, like a real valley
    const lean = 0.8 + 0.2 * Math.sin((x / width) * Math.PI * 1.1 + peaks);
    points.push([x, base - height * lean * (0.5 + 0.5 * y / norm)]);
  }
  return points;
}

export function paint(canvas, { width, height, dpr = 1 } = {}) {
  const w = Math.max(320, Math.round(width || canvas.clientWidth || 1440));
  const h = Math.max(240, Math.round(height || canvas.clientHeight || 900));
  const scale = Math.min(1.5, dpr || 1);
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const random = rng(SEED);
  const horizon = Math.round(h * 0.58);
  const glowX = w * 0.64;

  // sky: charcoal falling into a warm graphite at the horizon
  let g = ctx.createLinearGradient(0, 0, 0, horizon);
  g.addColorStop(0, "#08090a");
  g.addColorStop(0.55, "#0c0d0d");
  g.addColorStop(0.9, "#171512");
  g.addColorStop(1, "#1d1a15");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, horizon);

  // the last warm light, low and wide behind the mountains
  g = ctx.createRadialGradient(glowX, horizon, 0, glowX, horizon, w * 0.42);
  g.addColorStop(0, "rgba(201, 164, 112, 0.16)");
  g.addColorStop(0.35, "rgba(160, 128, 88, 0.06)");
  g.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, horizon + 2);

  // mountains: far (misted) to near (dark)
  const layers = [
    { base: horizon, height: h * 0.11, roughness: 0.35, peaks: 1.4, colour: "rgba(34, 35, 33, 0.8)" },
    { base: horizon, height: h * 0.07, roughness: 0.5, peaks: 2.3, colour: "rgba(22, 24, 22, 0.94)" },
    { base: horizon + 1, height: h * 0.035, roughness: 0.65, peaks: 3.6, colour: "rgba(13, 15, 14, 1)" },
  ];
  const ridges = [];
  for (const layer of layers) {
    const points = ridge(w, random, layer);
    ridges.push({ points, colour: layer.colour });
    ctx.beginPath();
    ctx.moveTo(0, horizon + 2);
    for (const [x, y] of points) ctx.lineTo(x, y);
    ctx.lineTo(w, horizon + 2);
    ctx.closePath();
    ctx.fillStyle = layer.colour;
    ctx.fill();
    // mist settling in front of each layer
    g = ctx.createLinearGradient(0, horizon - layer.height * 0.9, 0, horizon);
    g.addColorStop(0, "rgba(120, 118, 108, 0)");
    g.addColorStop(1, "rgba(120, 118, 108, 0.045)");
    ctx.fillStyle = g;
    ctx.fillRect(0, horizon - layer.height, w, layer.height);
  }

  // water: almost black, slightly lighter at the far edge
  g = ctx.createLinearGradient(0, horizon, 0, h);
  g.addColorStop(0, "#121210");
  g.addColorStop(0.08, "#0b0c0b");
  g.addColorStop(1, "#050606");
  ctx.fillStyle = g;
  ctx.fillRect(0, horizon, w, h - horizon);

  // the mountains, reflected, faint and compressed
  ctx.save();
  ctx.globalAlpha = 0.28;
  for (const { points, colour } of ridges) {
    ctx.beginPath();
    ctx.moveTo(0, horizon);
    for (const [x, y] of points) ctx.lineTo(x, horizon + (horizon - y) * 0.62);
    ctx.lineTo(w, horizon);
    ctx.closePath();
    ctx.fillStyle = colour;
    ctx.fill();
  }
  ctx.restore();

  // the warm light's path on the water: short, broken strokes that thin out towards the viewer
  for (let i = 0; i < 260; i++) {
    const depth = random() ** 1.7;
    const y = horizon + 2 + depth * (h - horizon) * 0.75;
    const spread = 14 + depth * w * 0.09;
    const x = glowX + (random() - 0.5) * spread * 2;
    const len = 6 + random() * (18 + depth * 60);
    const alpha = (1 - depth) ** 2.2 * (0.05 + random() * 0.16);
    ctx.fillStyle = `rgba(214, 180, 128, ${alpha.toFixed(3)})`;
    ctx.fillRect(x - len / 2, y, len, 1);
  }

  // the surface: a few long, barely visible lines, closer together near the horizon
  for (let i = 0; i < 70; i++) {
    const depth = (i / 70) ** 1.8;
    const y = horizon + 4 + depth * (h - horizon);
    const x = random() * w;
    const len = w * (0.05 + random() * 0.25);
    ctx.fillStyle = `rgba(200, 196, 184, ${(0.012 + random() * 0.018).toFixed(3)})`;
    ctx.fillRect(x - len / 2, y, len, 1);
  }

  // mist on the water line
  g = ctx.createLinearGradient(0, horizon - h * 0.03, 0, horizon + h * 0.05);
  g.addColorStop(0, "rgba(150, 146, 134, 0)");
  g.addColorStop(0.5, "rgba(150, 146, 134, 0.05)");
  g.addColorStop(1, "rgba(150, 146, 134, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, horizon - h * 0.03, w, h * 0.08);

  // fine grain
  const grain = document.createElement("canvas");
  grain.width = grain.height = 128;
  const gctx = grain.getContext("2d");
  const image = gctx.createImageData(128, 128);
  for (let i = 0; i < image.data.length; i += 4) {
    const v = Math.floor(random() * 255);
    image.data[i] = image.data[i + 1] = image.data[i + 2] = v;
    image.data[i + 3] = 10;
  }
  gctx.putImageData(image, 0, 0);
  ctx.fillStyle = ctx.createPattern(grain, "repeat");
  ctx.fillRect(0, 0, w, h);
  return { width: w, height: h, horizon };
}

/* Keep the scene sized to its box, repainting (debounced) only when the size changes. */
export function mount(canvas) {
  let timer = 0;
  let last = "";
  const repaint = () => {
    const rect = canvas.getBoundingClientRect();
    const key = `${Math.round(rect.width)}x${Math.round(rect.height)}`;
    if (!rect.width || key === last) return;
    last = key;
    paint(canvas, { width: rect.width, height: rect.height, dpr: window.devicePixelRatio || 1 });
    canvas.classList.add("painted");
  };
  const schedule = () => { clearTimeout(timer); timer = setTimeout(repaint, 180); };
  if (window.ResizeObserver) new ResizeObserver(schedule).observe(canvas);
  else window.addEventListener("resize", schedule);
  repaint();
  return { repaint };
}
