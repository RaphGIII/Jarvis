/* Wissen: the knowledge universe.

   The DEFAULT view is the galaxy — the same spatial language as Projects and
   Files (one cosmos, three regions): every knowledge node is a real node
   from /api/knowledge/graph, drawn as a small planet, clustered into DOMAIN
   sectors (Studium, Medizin, Technik, ZEUS intern, Fähigkeiten, Projekte,
   Dateien, Erkenntnisse, Notizen) with soft nebulae underneath; real graph
   edges stay visible as faint relations.  The owner's LIBRARY (real folders
   and files under D:\ZEUS_Wissen, via /api/library) appears as its own
   BIBLIOTHEK sector — those bodies are actual files that Explorer shows.

   Domain assignment is a documented view-only heuristic: node TYPE first
   (capability/experiment → Fähigkeiten, project → Projekte, file/document →
   Dateien, findings/lessons/decisions/ideas → Erkenntnisse), then keyword
   hints on title+body for Studium/Medizin/Technik/ZEUS intern; the graph
   itself is never rewritten by the view.

   Editing is not lost: the ✎ drawer holds the typed composer (node, link,
   ingest) and the real library actions (folder, note, import, PDF summary);
   the inspector on every node keeps read/link/delete.  The former strata
   board remains reachable via the "Ebenen" chip. */

import { $, el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";
import { Galaxy, warp } from "./projects.js";

const DOMAIN_HUE = {
  STUDIUM: [240, 200, 110], MEDIZIN: [255, 140, 150], TECHNIK: [127, 224, 180],
  "ZEUS INTERN": [102, 201, 255], "FÄHIGKEITEN": [120, 210, 220], PROJEKTE: [160, 190, 255],
  DATEIEN: [200, 180, 140], ERKENNTNISSE: [240, 170, 90], NOTIZEN: [150, 170, 200],
  BIBLIOTHEK: [190, 150, 250],
};
const DOMAIN_COLOUR = Object.fromEntries(Object.entries(DOMAIN_HUE).map(([k, [r, g, b]]) => [k, `rgb(${r},${g},${b})`]));

const STUDY_HINT = /\b(studium|uni|klausur|pruefung|prüfung|vorlesung|semester|lernen|kapitel|skript|lecture|exam)\b/i;
const MED_HINT = /\b(medizin|anatomie|physiologie|biochemie|pharma|klinik|patient|diagnose\b|therapie|arzt)\b/i;
const ZEUS_HINT = /\b(zeus|jarvis|wakeword|wake.word|voice|selfdev|mission|capability|ollama|whisper|supervisor|stt|tts)\b/i;
const TECH_HINT = /\b(python|code|test_|server|gpu|api|windows|datei|repo|git|json|schema|function|klasse|bug)\b/i;

function domainOf(node) {
  const type = node.type || "note";
  if (type === "capability" || type === "experiment") return "FÄHIGKEITEN";
  if (type === "project") return "PROJEKTE";
  if (type === "file" || type === "document") return "DATEIEN";
  if (["technical_finding", "verified_lesson", "decision", "idea"].includes(type)) return "ERKENNTNISSE";
  const probe = `${node.title || ""} ${(node.body || "").slice(0, 300)}`;
  if (MED_HINT.test(probe)) return "MEDIZIN";
  if (STUDY_HINT.test(probe)) return "STUDIUM";
  if (ZEUS_HINT.test(probe)) return "ZEUS INTERN";
  if (TECH_HINT.test(probe)) return "TECHNIK";
  return type === "concept" ? "TECHNIK" : "NOTIZEN";
}

let galaxy = null;
let graphOverlay = null;
let graphNode = null;
let active = false;

export const view = {
  id: "knowledge",
  title: "Wissen",
  async mount(pane, params) {
    active = true;
    if (params.mode === "list") return mountList(pane, params);
    await mountGalaxy(pane, params);
  },
  unmount() { active = false; galaxy?.destroy(); galaxy = null; },
};

/* ---- the Wissen galaxy (default) ------------------------------------- */

async function mountGalaxy(pane, params) {
  const search = el("input", { placeholder: "Fokus… (Titel, Typ)", value: params.q || "", style: { maxWidth: "220px" } });
  const counts = el("span", { class: "empty", style: { padding: 0 } });
  const listChip = el("button", { class: "chip", text: "☰ Ebenen", onClick: () => views.open("knowledge", { mode: "list" }) });
  const editChip = el("button", { class: "chip", text: "✎ Bearbeiten", onClick: () => { drawer.dataset.open = drawer.dataset.open === "true" ? "false" : "true"; } });
  pane.append(el("div", { class: "toolbar galaxy-overlay" }, search, listChip, editChip, counts));

  const wrap = el("div", { class: "galaxy-wrap" });
  const canvas = el("canvas", { id: "constellation", class: "galaxy files" });
  const legend = el("div", { class: "galaxy-legend" },
    ...Object.entries(DOMAIN_COLOUR).map(([k, c]) => el("span", { style: { "--c": c }, text: k.toLowerCase() })));
  const hint = el("div", { class: "galaxy-hint", text: "klick öffnen · rechtsklick menü · ✎ bearbeiten · zoom = nähe" });
  wrap.append(canvas, legend, hint);
  pane.append(wrap);

  const [data, lib] = await Promise.all([
    api("/api/knowledge/graph", { query: "", limit: 400 }),
    api("/api/library/tree").catch(() => ({ entries: [] })),
  ]);
  const kNodes = data.nodes || [], kEdges = data.edges || [];
  const degree = {};
  for (const e of kEdges) { degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1; }

  const nodes = [];
  const edges = [];
  for (const raw of kNodes) {
    const domain = domainOf(raw);
    const d = degree[raw.id] || 0;
    nodes.push({
      id: raw.id, kind: "project", label: (raw.title || "(ohne Titel)").slice(0, 30), hue: DOMAIN_HUE[domain],
      importance: d >= 6 ? "ACTIVE" : "NORMAL", tasks: 0,
      baseR: 5 + Math.min(11, Math.sqrt(d) * 2.6), ringed: d >= 8,
      health: { state: "HEALTHY", reason: "" },
      data: { isKnowledge: true, node: raw, domain },
    });
  }
  for (const e of kEdges) edges.push({ source: e.source, target: e.target, type: "relates" });
  // the library: real files on the shelf, their own sector
  for (const entry of (lib.entries || []).slice(0, 120)) {
    nodes.push({
      id: "lib:" + entry.path, kind: "project", label: entry.name.slice(0, 28), hue: DOMAIN_HUE.BIBLIOTHEK,
      importance: entry.type === "dir" ? "ACTIVE" : "NORMAL", tasks: 0,
      baseR: entry.type === "dir" ? 10 : 5.5, ringed: entry.type === "dir" && entry.depth === 0,
      health: { state: "HEALTHY", reason: "" },
      data: { isLibrary: true, entry, domain: "BIBLIOTHEK" },
    });
  }

  galaxy = new Galaxy(canvas, wrap, { nodes, edges }, {
    mode: "GALAXY", persistDrag: false, minZoom: 0.3,
    onZoomOutBeyond: () => { warp(canvas, "rise"); views.open("files", { path: "" }); },
    onSelect: (n) => select(n),
    onDoubleClick: (n) => select(n),
    onContext: (n, x, y) => contextMenu(n, x, y),
    drawUnder: (ctx, g) => nebulae(ctx, g),
    drawExtras: (ctx, g) => sectorLabels(ctx, g),
    // progressive disclosure: from afar only the sectors and the largest
    // bodies carry names; approach a cluster and its titles fade in
    labelFor: (n, z, emphasized) => emphasized || n.r >= 11 || z >= 1.6 || (z >= 1.15 && n.r >= 7.5),
  });
  clusterDomains(galaxy);
  window.zeusGalaxy = galaxy;
  counts.textContent = `${kNodes.length} Wissensknoten · ${kEdges.length} Verknüpfungen · ${(lib.entries || []).length} Bibliothek` + (data.truncated ? " · gekürzt" : "");
  search.oninput = () => galaxy?.focusText(search.value);
  if (search.value) galaxy.focusText(search.value);

  const drawer = editDrawer(lib);
  pane.append(drawer);
}

/* Sector targets: domains on an ellipse, members on a golden spiral inside
   their sector, then relaxed apart — the same rules the File universe uses,
   so the two feel like one cosmos. */
function clusterDomains(g) {
  const members = g.nodes.filter((n) => n.data?.domain);
  const domains = [...new Set(members.map((n) => n.data.domain))];
  const W = g.W(), H = g.H();
  domains.forEach((domain, di) => {
    const a = (di / Math.max(1, domains.length)) * Math.PI * 2 - Math.PI / 2;
    const spread = domains.length > 1 ? 1 : 0;
    const cx = W / 2 + Math.cos(a) * W * 0.33 * spread;
    const cy = H / 2 + Math.sin(a) * H * 0.31 * spread;
    const mine = members.filter((n) => n.data.domain === domain);
    mine.forEach((n, i) => {
      const ga = i * 2.399963 + di;
      const rr = 26 + 24 * Math.sqrt(i);
      n.tx = cx + Math.cos(ga) * rr; n.ty = cy + Math.sin(ga) * rr * 0.75;
    });
  });
  for (let it = 0; it < 80; it++) {
    let moved = false;
    for (const a of members) for (const b of members) {
      if (a === b) continue;
      const dx = a.tx - b.tx, dy = a.ty - b.ty, dd = Math.max(1, Math.hypot(dx, dy));
      const min = a.r + b.r + 22;
      if (dd < min) {
        const push = (min - dd) / 2, ux = dx / dd, uy = dy / dd;
        a.tx += ux * push; a.ty += uy * push; b.tx -= ux * push; b.ty -= uy * push; moved = true;
      }
    }
    for (const n of members) { const R = n.r + 12; n.tx = Math.max(R, Math.min(W - R, n.tx)); n.ty = Math.max(R + 28, Math.min(H - R - 22, n.ty)); }
    if (!moved) break;
  }
}

function groupsOf(g) {
  const groups = new Map();
  for (const n of g.nodes) {
    if (!n.data?.domain || !n.visible) continue;
    if (!groups.has(n.data.domain)) groups.set(n.data.domain, []);
    groups.get(n.data.domain).push(n);
  }
  return groups;
}

function nebulae(ctx, g) {
  for (const [domain, ns] of groupsOf(g)) {
    if (ns.length < 2) continue;
    let sx = 0, sy = 0;
    for (const n of ns) { const [x, y] = g.toScreen(n); sx += x; sy += y; }
    const cx = sx / ns.length, cy = sy / ns.length;
    let spread = 50;
    for (const n of ns) { const [x, y] = g.toScreen(n); spread = Math.max(spread, Math.hypot(x - cx, y - cy) + n.r * g.cam.z + 26); }
    const [r, gg, b] = DOMAIN_HUE[domain] || DOMAIN_HUE.NOTIZEN;
    const neb = ctx.createRadialGradient(cx, cy, spread * 0.15, cx, cy, spread);
    neb.addColorStop(0, `rgba(${r},${gg},${b},.07)`); neb.addColorStop(0.7, `rgba(${r},${gg},${b},.03)`); neb.addColorStop(1, "transparent");
    ctx.fillStyle = neb; ctx.beginPath(); ctx.arc(cx, cy, spread, 0, Math.PI * 2); ctx.fill();
  }
}

function sectorLabels(ctx, g) {
  ctx.textAlign = "center"; ctx.font = "600 10px Segoe UI, sans-serif";
  for (const [domain, ns] of groupsOf(g)) {
    if (ns.length < 2) continue;
    let sx = 0, top = Infinity;
    for (const n of ns) { const [x, y] = g.toScreen(n); sx += x; top = Math.min(top, y - n.r * g.cam.z); }
    const [r, gg, b] = DOMAIN_HUE[domain] || DOMAIN_HUE.NOTIZEN;
    ctx.fillStyle = `rgba(${r},${gg},${b},.42)`;
    ctx.fillText(domain, sx / ns.length, top - 22);
  }
}

function select(n) {
  const d = n.data || {};
  if (d.isKnowledge) return inspectNode(d.node);
  if (d.isLibrary) return inspectLibrary(d.entry);
}

function contextMenu(n, sx, sy) {
  if (!galaxy) return;
  galaxy.closeMenu();
  const d = n.data || {};
  const item = (label, fn, cls = "") => el("button", { class: cls, text: label, onClick: async () => { galaxy?.closeMenu(); await fn(); } });
  const entries = [];
  if (d.isKnowledge) {
    entries.push(
      item("Öffnen / Inspizieren", () => inspectNode(d.node)),
      item("Fokus", () => galaxy.focusText(n.label)),
      item("Zeus fragen", () => chat.send(`Erzähl mir aus meinem Wissensgraphen über „${d.node.title}“.`)),
      item("Löschen…", async () => {
        if (!confirm(`„${d.node.title}“ endgültig aus dem Wissensgraphen löschen?`)) return;
        const r = await api("/api/knowledge/delete", { id: d.node.id, confirm: true });
        if (r.ok === false) alert(r.error || "nicht gelöscht");
        views.open("knowledge", {}, { push: false });
      }, "danger"));
  } else if (d.isLibrary) {
    entries.push(
      item("Im Explorer öffnen", () => api("/api/fs/open", { path: libAbs(d.entry) })),
      d.entry.type === "file" && ["md", "txt"].includes(d.entry.ext) ? item("Lesen", () => inspectLibrary(d.entry)) : null,
      d.entry.type === "file" && d.entry.ext === "pdf" ? item("PDF zusammenfassen", () => summarizePdf(libAbs(d.entry))) : null,
      item("In Wissen ingestieren", async () => { const r = await api("/api/knowledge/ingest", { path: libAbs(d.entry) }); alert(r.ok ? "ingestiert" : (r.error || "fehlgeschlagen")); }));
  }
  const menu = el("div", { class: "galaxy-menu choice" }, el("h6", { text: n.label }), ...entries.filter(Boolean));
  const off = (n.r || 8) * galaxy.cam.z + 20;
  menu.style.left = Math.max(8, Math.min(sx + off, galaxy.W() - 240)) + "px";
  menu.style.top = Math.max(8, Math.min(sy - 16, galaxy.H() - 220)) + "px";
  galaxy.wrap.append(menu); galaxy.menu = menu;
}

let libRoot = "";
function libAbs(entry) { return (libRoot ? libRoot + "\\" : "") + entry.path.replaceAll("/", "\\"); }

async function inspectLibrary(entry) {
  const isNote = entry.type === "file" && ["md", "txt"].includes(entry.ext);
  const note = isNote ? await api("/api/library/read", { path: entry.path }) : null;
  views.inspect(entry.name,
    el("div", { class: "meta" }, badge(entry.type === "dir" ? "ORDNER" : (entry.ext || "DATEI").toUpperCase(), "blue"), " ", badge("BIBLIOTHEK", "dim")),
    section("Ablage", kv("pfad", entry.path), kv("liegt real unter", libRoot || "D:\\ZEUS_Wissen")),
    note && note.ok ? section("Inhalt", el("pre", { class: "code", text: (note.text || "").slice(0, 3000) })) : null,
    el("div", { class: "toolbar" },
      button("Im Explorer öffnen", () => api("/api/fs/open", { path: libAbs(entry) }), "primary"),
      entry.type === "file" && entry.ext === "pdf" ? button("Zusammenfassen", () => summarizePdf(libAbs(entry))) : null,
      button("Ingestieren", async () => { const r = await api("/api/knowledge/ingest", { path: libAbs(entry) }); alert(r.ok ? "in den Wissensgraphen aufgenommen" : (r.error || "fehlgeschlagen")); })));
}

async function summarizePdf(absPath) {
  const status = el("div", { class: "empty", text: "Lese und fasse zusammen… (lokales Modell, dauert einen Moment)" });
  views.inspect("PDF-Zusammenfassung", status);
  const r = await api("/api/pdf/summarize", { path: absPath });
  if (r.ok === false) { status.textContent = r.error || "fehlgeschlagen"; return; }
  views.inspect("PDF-Zusammenfassung",
    el("div", { class: "meta" }, badge("ECHT GESPEICHERT", "ok"), el("span", { text: ` ${r.pages} Seiten · ${r.chars} Zeichen gelesen` })),
    section("Zusammenfassung", el("pre", { class: "code", text: (r.summary || "").slice(0, 4000) })),
    section("Abgelegt", kv("datei", r.summary_file || "(nicht gespeichert)"), kv("wissensknoten", r.knowledge_node || "(keiner)")),
    el("div", { class: "toolbar" }, button("Wissen neu laden", () => views.open("knowledge"), "primary")));
}

/* The ✎ drawer: the typed composer plus REAL library actions. */
function editDrawer(lib) {
  libRoot = lib.root || "";
  const drawer = el("div", { class: "focus-drawer kedit", dataset: { open: "false" } });
  const status = el("div", { class: "empty" });
  const folder = el("input", { placeholder: "Neuer Ordner, z. B. Studium/Anatomie" });
  const noteFolder = el("input", { placeholder: "Ordner (leer = Notizen)", style: { maxWidth: "180px" } });
  const noteTitle = el("input", { placeholder: "Titel der Notiz" });
  const noteText = el("textarea", { placeholder: "Inhalt (Markdown)", rows: 3, style: { width: "100%" } });
  const importPath = el("input", { placeholder: "Datei importieren: absoluter Pfad (auch PDF)", style: { minWidth: "300px" } });
  const pdfPath = el("input", { placeholder: "PDF zusammenfassen: absoluter Pfad", style: { minWidth: "300px" } });
  const panel = el("div", { class: "focus-panel" },
    el("div", { class: "focus-col", style: { minWidth: "300px" } },
      el("h5", { text: `Bibliothek (echte Dateien · ${lib.root || "D:\\ZEUS_Wissen"})` }),
      el("div", { class: "toolbar" }, folder, button("Ordner anlegen", async () => {
        const r = await api("/api/library/folder", { path: folder.value });
        status.textContent = r.ok ? `Ordner existiert real: ${r.path}` : (r.error || "fehlgeschlagen");
        if (r.ok) { folder.value = ""; views.open("knowledge", {}, { push: false }); }
      }, "primary")),
      el("div", { class: "toolbar" }, noteFolder, noteTitle),
      noteText,
      el("div", { class: "toolbar" }, button("Notiz speichern (.md)", async () => {
        const r = await api("/api/library/note", { folder: noteFolder.value, title: noteTitle.value, text: noteText.value });
        status.textContent = r.ok ? `gespeichert: ${r.relative} (zurückgelesen: ja)` : (r.error || "fehlgeschlagen");
        if (r.ok) { noteTitle.value = ""; noteText.value = ""; views.open("knowledge", {}, { push: false }); }
      }, "primary"),
        button("Bibliothek im Explorer", () => api("/api/fs/open", { path: lib.root || "D:\\ZEUS_Wissen" }))),
      el("div", { class: "toolbar" }, importPath, button("Importieren", async () => {
        const r = await api("/api/library/import", { source: importPath.value });
        status.textContent = r.ok ? `importiert: ${r.relative}` : (r.error || "fehlgeschlagen");
        if (r.ok) { importPath.value = ""; views.open("knowledge", {}, { push: false }); }
      })),
      el("div", { class: "toolbar" }, pdfPath, button("PDF lesen + zusammenfassen", async () => {
        if (!pdfPath.value.trim()) return;
        await summarizePdf(pdfPath.value.trim());
      }))),
    el("div", { class: "focus-col", style: { minWidth: "320px" } },
      el("h5", { text: "Wissensgraph (Knoten, Verknüpfung, Ingest)" }),
      composer(() => views.open("knowledge", {}, { push: false }))),
  );
  panel.firstChild.append(status);
  const toggle = el("div", { class: "focus-toggle" }, el("b", { text: "✎ Wissen bearbeiten" }),
    el("span", { text: "Ordner · Notizen · Import · PDF · Knoten" }));
  toggle.onclick = () => { drawer.dataset.open = drawer.dataset.open === "true" ? "false" : "true"; };
  drawer.append(panel, toggle);
  return drawer;
}

/* ---- the former strata board, kept as the "Ebenen" mode --------------- */

async function mountList(pane, params) {
  // the strata board is a reading surface, not a spatial one
  $("app").classList.remove("spatial");
  const search = el("input", { placeholder: "Suchen… (springt zur Fundstelle)", value: params.q || "" });
  const summary = el("span", { class: "empty", style: { padding: 0 } });
  const crumbs = el("div", { class: "fs-crumbs" });
  const board = el("div", { class: "kboard" });
  const data = await api("/api/knowledge/graph", { query: "", limit: 400 });
  const nodes = data.nodes || [], edges = data.edges || [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const near = new Map(); const degree = {};
  for (const e of edges) {
    degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1;
    if (!near.has(e.source)) near.set(e.source, new Set());
    if (!near.has(e.target)) near.set(e.target, new Set());
    near.get(e.source).add(e.target); near.get(e.target).add(e.source);
  }
  const FINDING = new Set(["technical_finding", "verified_lesson", "decision", "note", "document", "idea"]);
  const maxDeg = Math.max(1, ...nodes.map((n) => degree[n.id] || 0));
  const neighbours = (id) => [...(near.get(id) || [])].map((x) => byId.get(x)).filter(Boolean);
  let path = [];
  const STRATUM = ["DOMÄNEN", "THEMEN", "KONZEPTE & BEFUNDE", "BEFUNDE"];
  const plate = (n) => {
    const d = degree[n.id] || 0;
    const leaf = FINDING.has(n.type) || d <= 1 || path.includes(n.id);
    const box = el("div", { class: "kplate" + (leaf ? " leaf" : "") + (FINDING.has(n.type) ? " finding" : ""), dataset: { type: n.type } },
      el("div", { class: "kplate-title", text: n.title }),
      el("div", { class: "kplate-meta" }, badge(n.type, FINDING.has(n.type) ? "amber" : "blue"),
        el("span", { text: `${d} Verknüpfungen` }), n.updated_at ? el("span", { text: n.updated_at.slice(0, 10) }) : null),
      el("div", { class: "kplate-deg" }, el("i", { style: { width: `${Math.max(6, Math.round((d / maxDeg) * 100))}%` } })));
    box.onclick = () => { if (leaf) { inspectNode(n); } else { path.push(n.id); render(); } };
    return box;
  };
  const render = () => {
    clear(crumbs); clear(board);
    crumbs.append(el("button", { class: "chip" + (path.length ? "" : " on"), text: "◈ Wissen", onClick: () => { path = []; search.value = ""; render(); } }));
    path.forEach((id, i) => crumbs.append(el("span", { class: "sep", text: "›" }),
      el("button", { class: "chip" + (i === path.length - 1 ? " on" : ""), text: (byId.get(id)?.title || id).slice(0, 30), onClick: () => { path = path.slice(0, i + 1); render(); } })));
    const q = search.value.trim().toLowerCase();
    let level, label;
    if (q) {
      level = nodes.filter((n) => (n.title || "").toLowerCase().includes(q) || String(n.body || "").toLowerCase().includes(q)).slice(0, 40);
      label = `SUCHE · ${level.length} Treffer`;
    } else if (!path.length) {
      level = nodes.filter((n) => !FINDING.has(n.type)).sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0)).slice(0, 9);
      label = STRATUM[0];
    } else {
      const cur = path.at(-1);
      level = neighbours(cur).filter((n) => !path.includes(n.id))
        .sort((a, b) => (FINDING.has(a.type) ? 1 : 0) - (FINDING.has(b.type) ? 1 : 0) || (degree[b.id] || 0) - (degree[a.id] || 0)).slice(0, 30);
      label = STRATUM[Math.min(path.length, STRATUM.length - 1)];
    }
    board.append(el("div", { class: "kdepth", text: `${label} · Ebene ${q ? "—" : path.length}` }));
    const grid = el("div", { class: "kgrid depth-" + Math.min(path.length, 2) });
    if (!level.length) grid.append(el("div", { class: "empty", text: q ? "Keine Treffer." : "Noch nichts hier." }));
    for (const n of level) grid.append(plate(n));
    board.append(grid);
    summary.textContent = `${nodes.length} Knoten · ${edges.length} Kanten${data.truncated ? " (gekürzt)" : ""}`;
  };
  let timer = null;
  search.oninput = () => { clearTimeout(timer); timer = setTimeout(render, 250); };
  pane.append(el("div", { class: "toolbar" }, crumbs, search,
    button("◈ Galaxy", () => views.open("knowledge"), "primary"), summary));
  pane.append(board);
  pane.append(el("details", {}, el("summary", { text: "Wissen erfassen (Knoten, Verknüpfung, Dokument-Ingest)" }), composer(() => views.open("knowledge", { mode: "list" }))));
  render();
}

/* Typed writes into the one graph — unchanged, reachable from both modes. */
function composer(reload) {
  const box = el("div");
  const title = el("input", { placeholder: "Title", style: { minWidth: "220px" } });
  const type = el("select", {}, ...["technical_finding", "note", "decision", "concept", "document", "verified_lesson", "device", "project", "idea"].map((t) => el("option", { value: t, text: t })));
  const text = el("textarea", { placeholder: "Content", rows: 3, style: { width: "100%" } });
  const links = el("input", { placeholder: "Concerns (comma-separated): ZEUS, Voice, Wakeword", style: { minWidth: "280px" } });
  const relation = el("select", {}, ...["concerns", "applies_to", "part_of", "relates_to", "supports", "contradicts", "depends_on", "about"].map((t) => el("option", { value: t, text: t })));
  const result = el("div", { class: "empty" });
  const create = button("Create node", async () => {
    const r = await api("/api/knowledge/create", { title: title.value, text: text.value, type: type.value, links: links.value.split(",").map((l) => l.trim()).filter(Boolean).map((t) => ({ target: t, relation: relation.value })) });
    result.textContent = r.ok ? `stored ${r.type} „${r.title}“ (${r.node_id}) · read back: ${r.read_back ? "yes" : "NO"} · searchable: ${r.searchable ? "yes" : "NO"} · relations: ${(r.relations || []).map((x) => `${x.relation}→${x.target}`).join(", ") || "none"}` : (r.error || "failed");
    if (r.ok) { title.value = ""; text.value = ""; reload(); }
  }, "primary");
  const src = el("input", { placeholder: "Link: source title", style: { minWidth: "180px" } });
  const dst = el("input", { placeholder: "target title", style: { minWidth: "180px" } });
  const rel2 = el("select", {}, ...["concerns", "applies_to", "part_of", "relates_to", "supports", "contradicts", "depends_on"].map((t) => el("option", { value: t, text: t })));
  const link = button("Link", async () => {
    const r = await api("/api/knowledge/link", { source: src.value, target: dst.value, relation: rel2.value });
    result.textContent = r.ok ? `${r.source} —${r.relation}→ ${r.target} (${r.edge_id})` : (r.error || "failed");
    if (r.ok) reload();
  });
  const path = el("input", { placeholder: "Ingest: file or folder path", style: { minWidth: "280px" } });
  const ingest = button("Ingest document", async () => {
    const r = await api("/api/knowledge/ingest", { path: path.value });
    result.textContent = r.ok ? `ingested: ${r.title || r.files_ingested + " file(s)"}` : (r.error || "failed");
    if (r.ok) reload();
  });
  box.append(el("div", { class: "toolbar" }, title, type, relation), text, el("div", { class: "toolbar" }, links, create),
    el("div", { class: "toolbar" }, src, rel2, dst, link), el("div", { class: "toolbar" }, path, ingest), result);
  return box;
}

/* The inspector for a node: backlinks and forward links from the real graph. */
export async function inspectNode(node) {
  const detail = await api("/api/knowledge/node", { id: node.id });
  const out = (detail.outgoing || []).map((it) => link(`${it.edge.type} →`, it.node));
  const inc = (detail.incoming || []).map((it) => link(`← ${it.edge.type}`, it.node));
  views.inspect(node.title,
    section("Node", kv("type", node.type), kv("id", node.id), kv("source", node.source || node.provenance || ""), kv("updated", node.updated_at || ""), kv("confidence", node.confidence ?? "")),
    node.body ? section("Content", el("div", { class: "kv" }, el("span", { class: "v", text: String(node.body).slice(0, 1200) }))) : null,
    section(`Forward links (${out.length})`, ...(out.length ? out : [el("div", { class: "empty", text: "none" })])),
    section(`Backlinks (${inc.length})`, ...(inc.length ? inc : [el("div", { class: "empty", text: "none" })])),
    el("div", { class: "toolbar" },
      button("Ask ZEUS", () => chat.send(`Tell me about "${node.title}" from my knowledge graph.`), "primary"),
      button("Fokus in der Galaxy", () => { galaxy ? galaxy.focusText(node.title) : views.open("knowledge", { q: node.title }); }),
      button("Löschen", async () => {
        if (!confirm(`„${node.title}“ endgültig löschen?`)) return;
        const r = await api("/api/knowledge/delete", { id: node.id, confirm: true });
        if (r.ok === false) { alert(r.error || "nicht gelöscht"); return; }
        views.closeInspector(); views.open("knowledge", {}, { push: false });
      }, "ghost danger")),
  );
}

function link(label, node) {
  return el("div", { class: "kv" }, el("span", { class: "k", text: label }),
    el("a", { class: "v", href: "#", text: node.title, onClick: (e) => { e.preventDefault(); inspectNode(node); } }));
}

/* ---- the starfield overlay (still available via palette) -------------- */

export async function openGraph(query = "", focusId = "") {
  const overlay = $("graphView");
  overlay.classList.add("open");
  if (!graphOverlay) {
    graphOverlay = new KnowledgeStarfield($("graphCanvas"), { onSelect: showNode });
    window.addEventListener("resize", () => graphOverlay.resize());
    $("btnGraphClose").onclick = closeGraph;
    $("graphSearch").addEventListener("input", (e) => graphOverlay?.setFilter(e.target.value));
    $("btnAskAbout").onclick = () => { if (graphNode) { closeGraph(); chat.send(`Tell me about "${graphNode.title}" from my knowledge graph.`); } };
    $("btnReadAloud").onclick = () => { if (graphNode) { api("/api/voice", { enabled: true, speak_replies: true }); chat.send(`Read this note aloud: "${graphNode.title}". ${graphNode.body || ""}`.slice(0, 1500)); } };
    $("btnExpand").onclick = async () => { if (!graphNode) return; const data = await api("/api/knowledge/graph", { query: graphNode.title, limit: 400 }); graphOverlay.load(data); graphOverlay.focusOn(graphNode.id); };
  }
  graphOverlay.resize();
  graphOverlay.start();
  const data = await api("/api/knowledge/graph", { query, limit: 400 });
  graphOverlay.load(data);
  $("graphCount").textContent = `${(data.nodes || []).length} nodes · ${(data.edges || []).length} links` + (data.truncated ? " (truncated)" : "");
  if (focusId) { graphOverlay.focusOn(focusId); const target = graphOverlay.byId.get(focusId); if (target) showNode(target); }
}

export function closeGraph() {
  $("graphView").classList.remove("open");
  graphOverlay?.stop();
}

async function showNode(node) {
  const card = $("nodeCard");
  if (!node) { card.hidden = true; graphNode = null; return; }
  graphNode = node;
  card.hidden = false;
  $("nodeType").textContent = node.type;
  $("nodeTitle").textContent = node.title;
  $("nodeBody").textContent = node.body ? node.body.slice(0, 600) : "(no content)";
  const links = clear($("nodeLinks"));
  const detail = await api("/api/knowledge/node", { id: node.id });
  const add = (title, items, dir) => {
    if (!items.length) return;
    links.append(el("h6", { text: title }));
    for (const item of items.slice(0, 10)) {
      links.append(el("button", { text: `${dir === "out" ? item.edge.type + " → " : "← " + item.edge.type + " "}${item.node.title}`.slice(0, 44),
        onClick: () => { graphOverlay.focusOn(item.node.id); const t = graphOverlay.byId.get(item.node.id); if (t) showNode(t); } }));
    }
  };
  add("Forward links", detail.outgoing || [], "out");
  add("Backlinks", detail.incoming || [], "in");
}
