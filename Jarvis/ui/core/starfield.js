/* Starfield geometry: the pure parts of a pan/zoom point map, with no DOM.

   Used by the Studium "Sterne" view (views/study_galaxy.js) and tested in node
   (ui/tests/study_galaxy.test.mjs).  World coordinates are the server's layout
   (roughly [-1, 1]); a camera is { x, y, zoom } where zoom is screen pixels per
   world unit and (x, y) is the world point at the centre of the view. */

export const MIN_ZOOM = 40;
export const MAX_ZOOM = 60000;

export function worldToScreen(camera, view, wx, wy) {
  return { x: (wx - camera.x) * camera.zoom + view.w / 2, y: (wy - camera.y) * camera.zoom + view.h / 2 };
}

export function screenToWorld(camera, view, sx, sy) {
  return { x: (sx - view.w / 2) / camera.zoom + camera.x, y: (sy - view.h / 2) / camera.zoom + camera.y };
}

export function clampZoom(zoom) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

/* Zoom by a factor while the world point under (sx, sy) stays under it. */
export function zoomAt(camera, view, sx, sy, factor) {
  const before = screenToWorld(camera, view, sx, sy);
  const zoom = clampZoom(camera.zoom * factor);
  const next = { x: camera.x, y: camera.y, zoom };
  const after = screenToWorld(next, view, sx, sy);
  next.x += before.x - after.x;
  next.y += before.y - after.y;
  return next;
}

/* The camera that shows every point (with a margin in pixels); a single point gets a calm, not an absurd, zoom. */
export function fitCamera(points, view, { margin = 48, maxZoom = 900 } = {}) {
  if (!points.length || view.w <= 0 || view.h <= 0) return { x: 0, y: 0, zoom: clampZoom(Math.min(view.w || 400, view.h || 400) / 2.4) };
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    const r = p.r || 0;
    if (p.x - r < minX) minX = p.x - r;
    if (p.x + r > maxX) maxX = p.x + r;
    if (p.y - r < minY) minY = p.y - r;
    if (p.y + r > maxY) maxY = p.y + r;
  }
  const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6);
  const zoom = clampZoom(Math.min((view.w - 2 * margin) / spanX, (view.h - 2 * margin) / spanY, maxZoom));
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, zoom };
}

export function easeInOut(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/* A camera between two cameras; zoom eases in log space so a big zoom change does not rush at the end. */
export function lerpCamera(a, b, t) {
  const k = easeInOut(Math.min(1, Math.max(0, t)));
  if (k >= 1) return { x: b.x, y: b.y, zoom: b.zoom };
  return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, zoom: Math.exp(Math.log(a.zoom) + (Math.log(b.zoom) - Math.log(a.zoom)) * k) };
}

/* ------------------------------------------------------------------ spatial grid (world coordinates) */

export function buildGrid(nodes, cell) {
  const size = cell > 0 ? cell : 0.05;
  const cells = new Map();
  for (let i = 0; i < nodes.length; i += 1) {
    const n = nodes[i];
    const key = `${Math.floor(n.x / size)},${Math.floor(n.y / size)}`;
    let bucket = cells.get(key);
    if (!bucket) { bucket = []; cells.set(key, bucket); }
    bucket.push(i);
  }
  return { cell: size, cells, nodes };
}

/* A cell size that keeps a few nodes per cell for this layout. */
export function gridCellFor(nodes) {
  if (nodes.length < 2) return 0.25;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) { minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x); minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y); }
  const area = Math.max((maxX - minX) * (maxY - minY), 1e-6);
  return Math.max(Math.sqrt(area * 4 / nodes.length), 1e-4);
}

/* The node nearest to (wx, wy) within radius (world units), or null.  Scans only the cells the radius touches;
   when the radius covers very many cells (far zoomed out) the scan is bounded by the node count instead. */
export function nearestInGrid(grid, wx, wy, radius) {
  const { cell, cells, nodes } = grid;
  const x0 = Math.floor((wx - radius) / cell), x1 = Math.floor((wx + radius) / cell);
  const y0 = Math.floor((wy - radius) / cell), y1 = Math.floor((wy + radius) / cell);
  let best = null, bestD = radius * radius;
  const visit = (i) => {
    const n = nodes[i];
    const d = (n.x - wx) ** 2 + (n.y - wy) ** 2;
    if (d <= bestD) { bestD = d; best = n; }
  };
  if ((x1 - x0 + 1) * (y1 - y0 + 1) > cells.size) {
    for (const bucket of cells.values()) for (const i of bucket) visit(i);
  } else {
    for (let gx = x0; gx <= x1; gx += 1) {
      for (let gy = y0; gy <= y1; gy += 1) {
        const bucket = cells.get(`${gx},${gy}`);
        if (bucket) for (const i of bucket) visit(i);
      }
    }
  }
  return best;
}

/* ------------------------------------------------------------------ what is drawn, and how bright */

/* The world rectangle the view shows, widened by a margin in pixels. */
export function viewBounds(camera, view, margin = 24) {
  const a = screenToWorld(camera, view, -margin, -margin);
  const b = screenToWorld(camera, view, view.w + margin, view.h + margin);
  return { minX: a.x, minY: a.y, maxX: b.x, maxY: b.y };
}

export function visibleNodes(nodes, camera, view, margin = 24) {
  const b = viewBounds(camera, view, margin);
  const out = [];
  for (const n of nodes) if (n.x >= b.minX && n.x <= b.maxX && n.y >= b.minY && n.y <= b.maxY) out.push(n);
  return out;
}

/* Emphasis: with a set, matching nodes brighten and the rest dim (never vanish); without one, everything is itself. */
export function emphasisAlpha(nodeId, emphasis, base = 1) {
  if (!emphasis || !emphasis.size) return base;
  return emphasis.has(nodeId) ? Math.min(1, base * 1.35 + 0.15) : base * 0.22;
}

/* A cluster label once the cluster is big enough on screen to be a place; topic labels need more room than subjects. */
export function clusterLabelVisible(cluster, zoom) {
  const px = (cluster.r || 0) * zoom;
  if (cluster.kind === "subject") return px >= 28;
  if (cluster.kind === "topic") return px >= 70 && Boolean(cluster.label);
  return false;
}

/* A document label: the hovered or selected one always; others only when their neighbours are far enough apart
   on screen that the words do not collide (cluster radius on screen per member). */
export function nodeLabelVisible({ hovered = false, selected = false, clusterRadius = 0, clusterSize = 1, zoom = 1 }) {
  if (hovered || selected) return true;
  const spacing = (clusterRadius * zoom) / Math.sqrt(Math.max(1, clusterSize));
  return spacing >= 64;
}

/* ------------------------------------------------------------------ keyboard */

const DIRECTIONS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

/* The node nearest in a direction: within a 45° cone either side, preferring what lies straight ahead. */
export function nearestInDirection(nodes, from, key) {
  const dir = DIRECTIONS[key];
  if (!dir || !from) return null;
  let best = null, bestScore = Infinity;
  for (const n of nodes) {
    if (n === from || n.id === from.id) continue;
    const dx = n.x - from.x, dy = n.y - from.y;
    const along = dx * dir[0] + dy * dir[1];
    if (along <= 1e-9) continue;
    const across = Math.abs(dx * dir[1] - dy * dir[0]);
    if (across > along) continue;
    const score = along + across * 2;
    if (score < bestScore) { bestScore = score; best = n; }
  }
  return best;
}

export function nearestTo(nodes, wx, wy) {
  let best = null, bestD = Infinity;
  for (const n of nodes) {
    const d = (n.x - wx) ** 2 + (n.y - wy) ** 2;
    if (d < bestD) { bestD = d; best = n; }
  }
  return best;
}

/* ------------------------------------------------------------------ words */

const TYPE_WORD = { pdf: "PDF", docx: "Word", pptx: "Folien", markdown: "Notiz", text: "Text", image: "Bild" };
const STATE_WORD = { ready: "durchsuchbar", processing: "wird indexiert", queued: "wartet auf Indexierung", failed: "Indexierung fehlgeschlagen" };

export function typeWord(node) {
  return node.goodnotes ? "GoodNotes" : TYPE_WORD[node.source_type] || "Dokument";
}

export function unitsWord(node) {
  const n = Number(node.units) || 0;
  if (node.source_type === "pptx") return n === 1 ? "1 Folie" : `${n} Folien`;
  if (["pdf", "image"].includes(node.source_type)) return n === 1 ? "1 Seite" : `${n} Seiten`;
  return n === 1 ? "1 Abschnitt" : `${n} Abschnitte`;
}

export function stateWord(node) {
  return STATE_WORD[node.state] || STATE_WORD.ready;
}
