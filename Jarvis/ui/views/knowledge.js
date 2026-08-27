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
    pane.append(el("div", { class: "toolbar" }, search, button("Open graph", () => openGraph(search.value), "primary"), summary), list);
    await load();
  },
};

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
