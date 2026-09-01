/* Knowledge: the starfield graph (real nodes, real edges), a local graph and
   backlinks for the selected node, search, and a list view for scale. */

import { $, el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as chat from "./chat.js";

let graph = null;
let graphNode = null;

export const view = {
  id: "knowledge",
  title: "Knowledge",
  /* Not a galaxy: knowledge is DEPTH. The view descends through strata —
     DOMÄNEN (the most connected hubs) → THEMEN (their neighbours) →
     BEFUNDE (findings, lessons, decisions, notes at the bottom). Only the
     current stratum is fully lit; you never see all nodes at once. Every
     plate is a real node; sizes come from real link counts. */
  async mount(pane, params) {
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
    let path = []; // ids of the descended strata
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
      board.append(el("div", { class: "kdepth", text: `${label} · Ebene ${q ? "—" : path.length}` + (path.length || q ? "" : ` · ${nodes.length} Knoten, ${edges.length} Kanten insgesamt — nie alle auf einmal`) }));
      const grid = el("div", { class: "kgrid depth-" + Math.min(path.length, 2) });
      if (!level.length) grid.append(el("div", { class: "empty", text: q ? "Keine Treffer." : "Noch nichts hier. Ein Dokument ingestieren, eine Mission abschließen oder eine Entscheidung festhalten — dann erscheint es." }));
      for (const n of level) grid.append(plate(n));
      board.append(grid);
      summary.textContent = `${nodes.length} Knoten · ${edges.length} Kanten${data.truncated ? " (gekürzt)" : ""}`;
    };
    let timer = null;
    search.oninput = () => { clearTimeout(timer); timer = setTimeout(render, 250); };
    pane.append(el("div", { class: "toolbar" }, crumbs, search, button("Graph-Overlay", () => openGraph(search.value), "primary"), summary));
    pane.append(board);
    pane.append(section("Store knowledge", composer(() => views.open("knowledge"))));
    render();
  },
};

/* Typed writes into the one graph: a node with its type and the concepts it
   concerns (created as CONCEPT nodes when missing), a relation between two
   existing nodes, and document/text ingestion with provenance. Every write
   answers with what was read back, so the panel never claims what the graph
   does not hold. */
function composer(reload) {
  const box = el("div");
  const title = el("input", { placeholder: "Title", style: { minWidth: "220px" } });
  const type = el("select", {}, ...["technical_finding", "note", "decision", "concept", "document", "verified_lesson", "device", "project", "idea"].map((t) => el("option", { value: t, text: t })));
  const text = el("textarea", { placeholder: "Content", rows: 3, style: { width: "100%" } });
  const links = el("input", { placeholder: "Concerns (comma-separated): ZEUS, Voice, Wakeword", style: { minWidth: "320px" } });
  const relation = el("select", {}, ...["concerns", "applies_to", "part_of", "relates_to", "supports", "contradicts", "depends_on", "about"].map((t) => el("option", { value: t, text: t })));
  const result = el("div", { class: "empty" });
  const create = button("Create node", async () => {
    const r = await api("/api/knowledge/create", { title: title.value, text: text.value, type: type.value, links: links.value.split(",").map((l) => l.trim()).filter(Boolean).map((t) => ({ target: t, relation: relation.value })) });
    result.textContent = r.ok ? `stored ${r.type} „${r.title}“ (${r.node_id}) · read back: ${r.read_back ? "yes" : "NO"} · searchable: ${r.searchable ? "yes" : "NO"} · relations: ${(r.relations || []).map((x) => `${x.relation}→${x.target}`).join(", ") || "none"}` : (r.error || "failed");
    if (r.ok) { title.value = ""; text.value = ""; reload(); }
  }, "primary");
  const src = el("input", { placeholder: "Link: source title", style: { minWidth: "200px" } });
  const dst = el("input", { placeholder: "target title", style: { minWidth: "200px" } });
  const rel2 = el("select", {}, ...["concerns", "applies_to", "part_of", "relates_to", "supports", "contradicts", "depends_on"].map((t) => el("option", { value: t, text: t })));
  const link = button("Link", async () => {
    const r = await api("/api/knowledge/link", { source: src.value, target: dst.value, relation: rel2.value });
    result.textContent = r.ok ? `${r.source} —${r.relation}→ ${r.target} (${r.edge_id})` : (r.error || "failed");
    if (r.ok) reload();
  });
  const path = el("input", { placeholder: "Ingest: file or folder path", style: { minWidth: "320px" } });
  const ingest = button("Ingest document", async () => {
    const r = await api("/api/knowledge/ingest", { path: path.value });
    result.textContent = r.ok ? `ingested: ${r.title || r.files_ingested + " file(s)"} (provenance: the file path; sections become nodes, wikilinks become edges)` : (r.error || "failed");
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
      button("Focus in graph", () => openGraph(node.title, node.id))),
  );
}

function link(label, node) {
  return el("div", { class: "kv" }, el("span", { class: "k", text: label }),
    el("a", { class: "v", href: "#", text: node.title, onClick: (e) => { e.preventDefault(); inspectNode(node); } }));
}

/* ---- the starfield overlay ------------------------------------------ */

export async function openGraph(query = "", focusId = "") {
  const overlay = $("graphView");
  overlay.classList.add("open");
  if (!graph) {
    graph = new KnowledgeStarfield($("graphCanvas"), { onSelect: showNode });
    window.addEventListener("resize", () => graph.resize());
    $("btnGraphClose").onclick = closeGraph;
    $("graphSearch").addEventListener("input", (e) => graph?.setFilter(e.target.value));
    $("btnAskAbout").onclick = () => { if (graphNode) { closeGraph(); chat.send(`Tell me about "${graphNode.title}" from my knowledge graph.`); } };
    $("btnReadAloud").onclick = () => { if (graphNode) { api("/api/voice", { enabled: true, speak_replies: true }); chat.send(`Read this note aloud: "${graphNode.title}". ${graphNode.body || ""}`.slice(0, 1500)); } };
    $("btnExpand").onclick = async () => { if (!graphNode) return; const data = await api("/api/knowledge/graph", { query: graphNode.title, limit: 400 }); graph.load(data); graph.focusOn(graphNode.id); };
  }
  graph.resize();
  graph.start();
  const data = await api("/api/knowledge/graph", { query, limit: 400 });
  graph.load(data);
  $("graphCount").textContent = `${(data.nodes || []).length} nodes · ${(data.edges || []).length} links` + (data.truncated ? " (truncated)" : "");
  if (focusId) { graph.focusOn(focusId); const target = graph.byId.get(focusId); if (target) showNode(target); }
}

export function closeGraph() {
  $("graphView").classList.remove("open");
  graph?.stop();
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
        onClick: () => { graph.focusOn(item.node.id); const t = graph.byId.get(item.node.id); if (t) showNode(t); } }));
    }
  };
  add("Forward links", detail.outgoing || [], "out");
  add("Backlinks", detail.incoming || [], "in");
}
