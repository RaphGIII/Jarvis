/* Finding phrases in text the way the owner reads it: case, accents and hyphenation do not matter.

   Pure functions (no DOM), shared by the Studium viewer and its tests. */

function foldChar(ch) {
  return ch.toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace("ß", "ss");
}

/* Every occurrence of every phrase in `text`: [start, end, rank] in text offsets, rank = the phrase's position
   (0 = the focus).  Case- and diacritics-insensitive, hyphens / spaces / soft hyphens loose, no overlaps. */
export function findPhrases(text, phrases) {
  const folded = [];
  const index = [];
  const source = String(text || "");
  for (let i = 0; i < source.length; i++) {
    for (const c of foldChar(source[i])) { folded.push(c); index.push(i); }
  }
  const hay = folded.join("");
  const spans = [];
  (phrases || []).forEach((phrase, rank) => {
    const needle = [...String(phrase || "")].map(foldChar).join("").trim();
    if (needle.length < 2) return;
    const pattern = new RegExp(needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/[-\s]+/g, "[-\\s\u00ad]*"), "g");
    for (let m = pattern.exec(hay); m; m = pattern.exec(hay)) {
      if (!m[0].length) { pattern.lastIndex++; continue; }
      spans.push([index[m.index], index[m.index + m[0].length - 1] + 1, rank]);
    }
  });
  spans.sort((a, b) => a[0] - b[0] || a[2] - b[2]);
  const kept = [];
  for (const span of spans) if (!kept.length || span[0] >= kept[kept.length - 1][1]) kept.push(span);
  return kept;
}

/* The span of the focus phrase (rank 0) nearest to `at`, the offset the search result pointed at; null without one. */
export function primarySpan(spans, at, offset = 0) {
  let best = null;
  let distance = Infinity;
  for (const span of spans) {
    if (span[2] !== 0) continue;
    const d = at == null ? span[0] : Math.abs(offset + span[0] - at);
    if (d < distance) { distance = d; best = span; }
  }
  return best;
}
