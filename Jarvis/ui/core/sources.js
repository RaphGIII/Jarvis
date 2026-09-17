/* A source reference: where something is in the owner's material, one click from the place itself.

   Document name, location ("Seite 42", "Folie 7", "Herzmechanik › Vorlast"),
   the excerpt with the matching words marked, and "Öffnen" -- which opens the
   Studium viewer at exactly that location, every match marked, the one this
   result points at first.  A place found by meaning rather than by the words
   says so.  Drawn as a thin rule and type, not as a card. */

import { el } from "./dom.js";
import * as views from "./views.js";

const TYPE_WORD = { pdf: "PDF", docx: "Word", pptx: "Folien", markdown: "Notiz", text: "Text", image: "Bild" };

export function excerptNode(text, highlights) {
  const node = el("span", { class: "src-excerpt" });
  let at = 0;
  for (const [start, end] of highlights || []) {
    if (start > at) node.append(document.createTextNode(text.slice(at, start)));
    node.append(el("mark", { text: text.slice(Math.max(start, at), end) }));
    at = Math.max(at, end);
  }
  node.append(document.createTextNode(text.slice(at)));
  return node;
}

/* The viewer's address for a place: the same parameters the server gives the chat (service/study_actions.view_params). */
export function viewParams(source) {
  const location = source.location || {};
  const params = { doc: source.document_id, unit: String(location.unit || 1) };
  if (source.focus) params.focus = String(source.focus).slice(0, 200);
  const others = (source.phrases || []).filter((p) => p && p !== source.focus).slice(0, 5);
  if (others.length) params.hl = others.map((p) => String(p).slice(0, 80)).join("|");
  if (source.at !== undefined && source.at !== null) params.at = String(source.at);
  if (location.paragraph != null && location.page == null && location.slide == null) params.para = String(location.paragraph);
  if (source.figure && source.figure.number) params.fig = String(source.figure.number);
  return params;
}

export function openSource(source) {
  views.open("study", viewParams(source));
}

export function sourceRef(source, { compact = false } = {}) {
  const location = source.location || {};
  const kind = source.goodnotes ? "GoodNotes" : TYPE_WORD[source.source_type] || "";
  const ref = el("div", { class: "src" + (compact ? " compact" : "") + (source.match === "semantic" ? " semantic" : ""), role: "listitem" });
  ref.append(
    el("div", { class: "src-head" },
      el("span", { class: "src-name", text: source.filename || source.title || "Dokument" }),
      el("span", { class: "src-where", text: location.label || "" }),
      kind ? el("span", { class: "src-kind", text: kind }) : null,
      source.match === "semantic" ? el("span", { class: "src-kind src-meaning", title: "Gefunden über die Bedeutung, nicht über die Wörter", text: "inhaltlich" }) : null,
      // recognised text (a scan, handwriting) can be misread: an uncertain place says so, the page itself decides
      source.certainty === "possible"
        ? el("span", { class: "src-kind src-possible", title: "Unsicher erkannt – sieh auf der Seite nach", text: source.handwriting ? "möglicher Treffer · Handschrift" : "möglicher Treffer" })
        : source.handwriting ? el("span", { class: "src-kind", title: "Aus der Handschrifterkennung", text: "Handschrift" }) : null),
    source.excerpt ? el("div", { class: "src-body" }, excerptNode(source.excerpt, source.highlights)) : null,
    el("button", { class: "src-open", text: "Öffnen", "aria-label": `${source.filename || source.title}, ${location.label || ""} öffnen`,
                   onClick: () => openSource(source) }));
  return ref;
}

export function sourceList(sources, options = {}) {
  const list = el("div", { class: "src-list", role: "list", "aria-label": "Quellen" });
  for (const source of sources || []) list.append(sourceRef(source, options));
  return list;
}
