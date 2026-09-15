/* Stream-safe Markdown for the conversation.

   render(text)               -- the completed answer, deterministic
   render(text, {partial})    -- whatever has arrived so far: an unclosed code
                                 fence renders as an open block, an unfinished
                                 equation or emphasis marker renders as plain
                                 text, a half table is a table.  Nothing here
                                 ever shows a raw "$", a "\frac", "&amp;" or a
                                 replacement glyph.

   Pure functions over strings -- no DOM -- so the same module runs under
   node in the tests. */

import { escapeHtml, splitMath, renderMath, chemistryToUnicode, proseSymbols } from "./mathtex.js";

const FENCE = /^(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;

/* Repair the encoding of provider text without changing what it says:
   doubled backslashes before commands, entities shown literally, the
   replacement glyph, unsupported chemistry markup. Fenced code untouched. */
export function sanitizeProse(text) {
  let t = String(text ?? "").replace(/�/g, "");
  t = t.replace(/\\\\(?=[a-zA-Z(\[\]){}])/g, "\\");
  for (let round = 0; round < 2 && /&(#\d{1,6}|#x[0-9a-fA-F]{1,6}|[a-zA-Z]{2,10});/.test(t); round += 1) {
    t = t.replace(/&(#\d{1,6}|#x[0-9a-fA-F]{1,6}|[a-zA-Z]{2,10});/g, (m) => decodeEntity(m));
  }
  t = t.replace(/\\(?:ce|pu)\{((?:[^{}]|\{[^{}]*\})*)\}/g, (_, f) => chemistryToUnicode(f));
  return t;
}

function decodeEntity(entity) {
  const named = { amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: " ", deg: "°", times: "×", middot: "·", rarr: "→", larr: "←",
                  harr: "↔", alpha: "α", beta: "β", gamma: "γ", delta: "δ", Delta: "Δ", micro: "µ", plusmn: "±", ndash: "–", mdash: "—",
                  hellip: "…", sup2: "²", sup3: "³", frac12: "½", le: "≤", ge: "≥", ne: "≠", infin: "∞" };
  const body = entity.slice(1, -1);
  if (body.startsWith("#x") || body.startsWith("#X")) return String.fromCodePoint(parseInt(body.slice(2), 16));
  if (body.startsWith("#")) return String.fromCodePoint(parseInt(body.slice(1), 10));
  return Object.prototype.hasOwnProperty.call(named, body) ? named[body] : entity;
}

/* ---------------------------------------------------------------- inline */

function inline(text, partial) {
  // Math first: its body must not be touched by emphasis rules.
  return splitMath(text).map((part) => {
    if (part.kind !== "text") return renderMath(part.value, part.kind);
    return inlineText(proseSymbols(part.value), partial);
  }).join("");
}

function inlineText(text, partial) {
  let out = "";
  let i = 0;
  const s = text;
  while (i < s.length) {
    // inline code
    if (s[i] === "`") {
      const run = /^`+/.exec(s.slice(i))[0];
      const end = s.indexOf(run, i + run.length);
      if (end !== -1) { out += `<code>${escapeHtml(s.slice(i + run.length, end))}</code>`; i = end + run.length; continue; }
      out += escapeHtml(run); i += run.length; continue;   // unmatched: literal
    }
    // strong / emphasis
    const em = /^(\*\*|__|\*|_)(?=\S)/.exec(s.slice(i));
    if (em && (i === 0 || !/[\w]/.test(s[i - 1]) || em[1].startsWith("*"))) {
      const mark = em[1];
      const end = findClose(s, i + mark.length, mark);
      if (end !== -1) {
        const tag = mark.length === 2 ? "b" : "i";
        out += `<${tag}>${inlineText(s.slice(i + mark.length, end), partial)}</${tag}>`;
        i = end + mark.length;
        continue;
      }
      out += escapeHtml(mark); i += mark.length; continue;  // unmatched: literal
    }
    // links [text](url)
    if (s[i] === "[") {
      const m = /^\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/.exec(s.slice(i));
      if (m) { out += `<a href="${escapeHtml(safeUrl(m[2]))}" target="_blank" rel="noopener">${inlineText(m[1], partial)}</a>`; i += m[0].length; continue; }
    }
    // bare URLs
    if (/^https?:\/\//.test(s.slice(i))) {
      const m = /^https?:\/\/[^\s<>)]+/.exec(s.slice(i));
      out += `<a href="${escapeHtml(safeUrl(m[0]))}" target="_blank" rel="noopener">${escapeHtml(m[0])}</a>`; i += m[0].length; continue;
    }
    out += escapeHtml(s[i]);
    i += 1;
  }
  return out;
}

function findClose(s, from, mark) {
  let j = from;
  while (j < s.length) {
    const k = s.indexOf(mark, j);
    if (k === -1) return -1;
    if (k > 0 && !/\s/.test(s[k - 1]) && (mark.length === 2 || !/[\w]/.test(s[k + mark.length] ?? " "))) return k;
    j = k + 1;
  }
  return -1;
}

function safeUrl(url) {
  return /^(https?:|mailto:|#|\/)/i.test(url) ? url : "#";
}

/* ---------------------------------------------------------------- blocks */

export function render(text, options = {}) {
  const partial = Boolean(options.partial);
  const lines = sanitizeProse(text).replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1][0];
      const lang = fence[2];
      const body = [];
      i += 1;
      let closed = false;
      while (i < lines.length) {
        if (lines[i].startsWith(marker.repeat(3)) && lines[i].trim().length >= 3 && new RegExp(`^${marker === "`" ? "`" : "~"}{3,}\\s*$`).test(lines[i])) { closed = true; i += 1; break; }
        body.push(lines[i]); i += 1;
      }
      html.push(`<pre class="code${closed ? "" : " open"}"${lang ? ` data-lang="${escapeHtml(lang)}"` : ""}><code>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (/^\s*$/.test(line)) { i += 1; continue; }
    const heading = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) { html.push(`<h${heading[1].length + 1}>${inline(heading[2], partial)}</h${heading[1].length + 1}>`); i += 1; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { html.push("<hr>"); i += 1; continue; }
    if (/^\$\$\s*$/.test(line) || /^\\\[\s*$/.test(line)) {
      const close = line.startsWith("$$") ? /^\$\$\s*$/ : /^\\\]\s*$/;
      const body = [];
      let j = i + 1;
      let closed = false;
      while (j < lines.length) { if (close.test(lines[j])) { closed = true; break; } body.push(lines[j]); j += 1; }
      if (closed) { html.push(renderMath(body.join("\n"), "display")); i = j + 1; continue; }
      // unclosed display math: shown as text until it closes
      html.push(`<p>${escapeHtml(line)}</p>`); i += 1; continue;
    }
    if (line.startsWith(">")) {
      const body = [];
      while (i < lines.length && lines[i].startsWith(">")) { body.push(lines[i].replace(/^>\s?/, "")); i += 1; }
      html.push(`<blockquote>${render(body.join("\n"), options)}</blockquote>`);
      continue;
    }
    if (isTableStart(lines, i)) {
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { rows.push(lines[i]); i += 1; }
      html.push(renderTable(rows, partial));
      continue;
    }
    const item = listItem(line);
    if (item) {
      html.push(renderList(lines, i, item.ordered, partial, (n) => { i = n; }));
      continue;
    }
    // paragraph: consecutive plain lines
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !FENCE.test(lines[i]) && !/^#{1,6}\s/.test(lines[i]) && !lines[i].startsWith(">")
           && !listItem(lines[i]) && !isTableStart(lines, i) && !/^\$\$\s*$/.test(lines[i]) && !/^\\\[\s*$/.test(lines[i])) {
      para.push(lines[i]); i += 1;
    }
    html.push(`<p>${inline(para.join("\n"), partial).replace(/\n/g, "<br>")}</p>`);
  }
  return html.join("\n");
}

function listItem(line) {
  const m = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line);
  if (!m) return null;
  return { indent: m[1].length, ordered: /\d/.test(m[2]), text: m[3] };
}

function renderList(lines, start, ordered, partial, setIndex) {
  const tag = ordered ? "ol" : "ul";
  const items = [];
  let i = start;
  const base = listItem(lines[start]).indent;
  while (i < lines.length) {
    const item = listItem(lines[i]);
    if (!item || item.indent < base) break;
    if (item.indent > base) {
      // nested list: collect its lines and render recursively
      const sub = [];
      while (i < lines.length && (listItem(lines[i])?.indent > base || (/^\s+\S/.test(lines[i]) && !listItem(lines[i])))) { sub.push(lines[i].slice(base + 2)); i += 1; }
      items[items.length - 1] = items[items.length - 1].replace(/<\/li>$/, "") + render(sub.join("\n"), { partial }) + "</li>";
      continue;
    }
    if (item.ordered !== ordered) break;
    let text = item.text;
    i += 1;
    // continuation lines (indented, not new items)
    while (i < lines.length && /^\s+\S/.test(lines[i]) && !listItem(lines[i])) { text += "\n" + lines[i].trim(); i += 1; }
    const task = /^\[([ xX])\]\s+/.exec(text);
    if (task) text = (task[1] === " " ? "☐ " : "☑ ") + text.slice(task[0].length);
    items.push(`<li>${inline(text, partial)}</li>`);
  }
  setIndex(i);
  return `<${tag}>${items.join("")}</${tag}>`;
}

function isTableStart(lines, i) {
  return /^\s*\|.*\|\s*$/.test(lines[i] ?? "") && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1] ?? "");
}

function cells(row) {
  return row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function renderTable(rows, partial) {
  const header = cells(rows[0]);
  const aligns = cells(rows[1] || "").map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "left"));
  const body = rows.slice(2).map((r) => `<tr>${cells(r).map((c, k) => `<td style="text-align:${aligns[k] || "left"}">${inline(c, partial)}</td>`).join("")}</tr>`);
  return `<table><thead><tr>${header.map((c, k) => `<th style="text-align:${aligns[k] || "left"}">${inline(c, partial)}</th>`).join("")}</tr></thead><tbody>${body.join("")}</tbody></table>`;
}

/* The unfinished tail of a partial stream that must not be rendered as markup yet:
   an open fence renders open; an odd "$$" or a trailing single "$" is text. */
export function renderPartial(text) {
  return render(text, { partial: true });
}
