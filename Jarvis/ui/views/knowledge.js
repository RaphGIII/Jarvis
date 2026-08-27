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
  async mount(pane, params) {
    const search = el("input", { placeholder: "Search knowledge…", value: params.q || "" });
    const list = el("div", { class: "grid" });
    const summary = el("span", { class: "empty", style: { padding: 0 } });
    const load = async () => {
      const data = await api("/api/knowledge/graph", { query: search.value, limit: 200 });
      clear(list);
      const nodes = data.nodes || [];
      summary.textContent = `${nodes.length} nodes · ${(data.edges || []).length} links${data.truncated ? " (truncated)" : ""}`;
      if (!nodes.length) list.append(el("div", { class: "empty", text: "Nothing here yet. Ingest a document, finish a mission or record a decision and it appears." }));
      const degree = {};
      for (const e of data.edges || []) { degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1; }
      for (const n of nodes.sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0)).slice(0, 120)) {
        const card = el("div", { class: "card click" }, el("div", { class: "title", text: n.title }),
          el("div", { class: "meta" }, badge(n.type, "blue"), el("span", { text: `${degree[n.id] || 0} links` }), n.updated_at ? el("span", { text: n.updated_at.slice(0, 10) }) : null));
        card.onclick = () => inspectNode(n);
        list.append(card);
      }
    };
    let timer = null;
    search.oninput = () => { clearTimeout(timer); timer = setTimeout(load, 300); };
    pane.append(el("div", { class: "toolbar" }, search, button("Open graph", () => openGraph(search.value), "primary"), summary));
    pane.append(section("Store knowledge", composer(load)));
    pane.append(list);
    await load();
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
