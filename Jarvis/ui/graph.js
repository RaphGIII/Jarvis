/*
 * The knowledge graph as a starfield.
 *
 * On not using a graph library: the brief suggests one, and it is right that
 * writing a physics engine to draw a diagram is a poor trade. But the service
 * is stdlib-only and the UI has no build step -- adding a bundler and an npm
 * tree so Jarvis can draw dots would put its own interface outside its own
 * development loop, which the brief also requires it to be inside. What is left
 * is about a hundred lines of Barnes-Hut-free force layout, which is enough for
 * the few hundred nodes a personal knowledge base actually holds. If it ever
 * needs tens of thousands, the layout is one object and can be replaced.
 *
 * Two things make it feel like space rather than a diagram: nodes are drawn as
 * glows whose radius follows their degree, so structure is visible before any
 * label is read; and the layout keeps running quietly after it settles, so the
 * constellation drifts instead of freezing into a picture.
 */

const NODE_COLOURS = {
  note: 198, concept: 268, project: 168, person: 42, task: 12,
  capability: 148, document: 210, conversation: 300, source: 60, fact: 190,
};

class KnowledgeStarfield {
  constructor(canvas, { onSelect } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelect = onSelect || (() => {});
    this.nodes = [];
    this.edges = [];
    this.byId = new Map();
    this.camera = { x: 0, y: 0, zoom: 1 };
    this.selected = null;
    this.hovered = null;
    this.running = false;
    this.alpha = 1;          // layout "temperature"; decays as it settles
    this.filter = "";
    this._wire();
  }

  /* ---- data ------------------------------------------------------- */

  load(payload) {
    const previous = new Map(this.nodes.map((n) => [n.id, n]));
    this.nodes = (payload.nodes || []).map((raw, index) => {
      // Keep positions across reloads so adding one note does not reshuffle
      // the whole sky and lose the user's mental map of it.
      const old = previous.get(raw.id);
      const angle = (index / Math.max(1, (payload.nodes || []).length)) * Math.PI * 2;
      const radius = 120 + (index % 7) * 40;
      return {
        id: raw.id,
        title: raw.title || "(untitled)",
        type: raw.type || "note",
        tags: raw.tags || [],
        body: raw.body || "",
        x: old ? old.x : Math.cos(angle) * radius,
        y: old ? old.y : Math.sin(angle) * radius,
        vx: 0, vy: 0,
        degree: 0,
      };
    });
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));

    this.edges = (payload.edges || [])
      .map((raw) => ({ source: this.byId.get(raw.source), target: this.byId.get(raw.target), type: raw.type }))
      .filter((e) => e.source && e.target);

    for (const edge of this.edges) { edge.source.degree++; edge.target.degree++; }
    this.alpha = 1;
    this.draw();
  }

  /* ---- layout ------------------------------------------------------ */

  step() {
    const nodes = this.nodes;
    if (!nodes.length) return;

    // Repulsion. O(n^2), which is fine to a few hundred nodes and honest about
    // where the ceiling is.
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let distance2 = dx * dx + dy * dy;
        if (distance2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; distance2 = 0.01; }
        if (distance2 > 640000) continue;             // far apart: ignore
        const force = 900 / distance2;
        const distance = Math.sqrt(distance2);
        a.vx += (dx / distance) * force; a.vy += (dy / distance) * force;
        b.vx -= (dx / distance) * force; b.vy -= (dy / distance) * force;
      }
    }

    // Springs along edges.
    for (const edge of this.edges) {
      const dx = edge.target.x - edge.source.x;
      const dy = edge.target.y - edge.source.y;
      const distance = Math.hypot(dx, dy) || 1;
      const force = (distance - 110) * 0.008;
      const fx = (dx / distance) * force, fy = (dy / distance) * force;
      edge.source.vx += fx; edge.source.vy += fy;
      edge.target.vx -= fx; edge.target.vy -= fy;
    }

    // Weak pull to the origin so disconnected nodes do not drift to infinity.
    for (const node of this.nodes) {
      node.vx -= node.x * 0.0012;
      node.vy -= node.y * 0.0012;
      node.x += node.vx * this.alpha;
      node.y += node.vy * this.alpha;
      node.vx *= 0.86;
      node.vy *= 0.86;
    }

    // Never quite reaches zero: a sky that still breathes reads as alive,
    // and it costs nothing once the forces have balanced.
    this.alpha = Math.max(0.05, this.alpha * 0.985);
  }

  /* ---- rendering --------------------------------------------------- */

  start() {
    if (this.running) return;
    this.running = true;
    const frame = () => {
      if (!this.running) return;
      this.step();
      this.draw();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  stop() { this.running = false; }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, rect.width * scale);
    this.canvas.height = Math.max(1, rect.height * scale);
    this.ctx.setTransform(scale, 0, 0, scale, 0, 0);
    this.draw();
  }

  toScreen(node) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (node.x - this.camera.x) * this.camera.zoom + rect.width / 2,
      y: (node.y - this.camera.y) * this.camera.zoom + rect.height / 2,
    };
  }

  matches(node) {
    if (!this.filter) return true;
    const needle = this.filter.toLowerCase();
    return (
      node.title.toLowerCase().includes(needle) ||
      node.type.toLowerCase().includes(needle) ||
      (node.tags || []).some((t) => String(t).toLowerCase().includes(needle))
    );
  }

  draw() {
    const ctx = this.ctx;
    const rect = this.canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!this.nodes.length) {
      ctx.fillStyle = "#3c4a60";
      ctx.font = "14px 'Segoe UI', system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Nothing in the knowledge graph yet.", rect.width / 2, rect.height / 2);
      return;
    }

    const neighbours = new Set();
    if (this.selected) {
      for (const edge of this.edges) {
        if (edge.source === this.selected) neighbours.add(edge.target);
        if (edge.target === this.selected) neighbours.add(edge.source);
      }
    }

    // Edges first, so nodes sit on top of their own connections.
    for (const edge of this.edges) {
      const a = this.toScreen(edge.source), b = this.toScreen(edge.target);
      const involved = this.selected && (edge.source === this.selected || edge.target === this.selected);
      ctx.strokeStyle = involved ? "rgba(120,200,255,0.55)" : "rgba(90,120,160,0.16)";
      ctx.lineWidth = involved ? 1.4 : 0.7;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (const node of this.nodes) {
      const p = this.toScreen(node);
      if (p.x < -60 || p.y < -60 || p.x > rect.width + 60 || p.y > rect.height + 60) continue;

      const hue = NODE_COLOURS[node.type] ?? 200;
      const dim = this.filter && !this.matches(node);
      const focused = node === this.selected || node === this.hovered;
      // Radius follows degree so structure is legible before any label is.
      const radius = (4 + Math.min(9, Math.sqrt(node.degree) * 3)) * this.camera.zoom;
      const strength = dim ? 0.12 : node === this.selected ? 1 : neighbours.has(node) ? 0.8 : 0.5;

      const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 3.2);
      glow.addColorStop(0, `hsla(${hue}, 90%, 70%, ${0.55 * strength})`);
      glow.addColorStop(1, "hsla(0,0%,0%,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 3.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `hsla(${hue}, 95%, ${focused ? 88 : 74}%, ${Math.max(0.2, strength)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Labels only where they can be read: zoomed in, focused, or important.
      const labelled = this.camera.zoom > 0.85 || focused || node.degree >= 4;
      if (labelled && !dim) {
        ctx.fillStyle = focused ? "#e8f1fb" : "rgba(200,216,235,0.72)";
        ctx.font = `${focused ? 13 : 11}px 'Segoe UI', system-ui, sans-serif`;
        ctx.textAlign = "center";
        const label = node.title.length > 30 ? node.title.slice(0, 29) + "…" : node.title;
        ctx.fillText(label, p.x, p.y + radius + 14);
      }
    }
  }

  /* ---- interaction -------------------------------------------------- */

  nodeAt(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    let best = null, bestDistance = 26;
    for (const node of this.nodes) {
      const p = this.toScreen(node);
      const distance = Math.hypot(p.x - x, p.y - y);
      if (distance < bestDistance) { best = node; bestDistance = distance; }
    }
    return best;
  }

  _wire() {
    let dragging = false, lastX = 0, lastY = 0, moved = 0;

    this.canvas.addEventListener("pointerdown", (e) => {
      dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY;
      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener("pointermove", (e) => {
      if (dragging) {
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);
        this.camera.x -= dx / this.camera.zoom;
        this.camera.y -= dy / this.camera.zoom;
        lastX = e.clientX; lastY = e.clientY;
        this.draw();
      } else {
        const found = this.nodeAt(e.clientX, e.clientY);
        if (found !== this.hovered) {
          this.hovered = found;
          this.canvas.style.cursor = found ? "pointer" : "grab";
          this.draw();
        }
      }
    });

    this.canvas.addEventListener("pointerup", (e) => {
      dragging = false;
      // A drag is not a click. Without this, panning the sky selects whatever
      // happened to be under the finger when it stopped.
      if (moved < 5) {
        const found = this.nodeAt(e.clientX, e.clientY);
        this.selected = found;
        this.draw();
        this.onSelect(found);
      }
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const before = {
        x: (e.clientX - rect.left - rect.width / 2) / this.camera.zoom + this.camera.x,
        y: (e.clientY - rect.top - rect.height / 2) / this.camera.zoom + this.camera.y,
      };
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      this.camera.zoom = Math.min(4, Math.max(0.15, this.camera.zoom * factor));
      // Zoom toward the cursor rather than the centre, so the thing being
      // examined stays under the pointer.
      const after = {
        x: (e.clientX - rect.left - rect.width / 2) / this.camera.zoom + this.camera.x,
        y: (e.clientY - rect.top - rect.height / 2) / this.camera.zoom + this.camera.y,
      };
      this.camera.x += before.x - after.x;
      this.camera.y += before.y - after.y;
      this.draw();
    }, { passive: false });
  }

  focusOn(nodeId) {
    const node = this.byId.get(nodeId);
    if (!node) return;
    this.selected = node;
    this.camera.x = node.x;
    this.camera.y = node.y;
    this.camera.zoom = Math.max(this.camera.zoom, 1.2);
    this.draw();
  }

  setFilter(text) {
    this.filter = (text || "").trim();
    this.draw();
  }
}

window.KnowledgeStarfield = KnowledgeStarfield;
