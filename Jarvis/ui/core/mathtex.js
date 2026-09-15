/* A LaTeX subset, rendered deterministically to HTML -- no fonts, no library,
   no network.  What medicine, chemistry and physiology answers actually use:
   sub- and superscripts, fractions, roots, Greek letters, arrows, operators,
   \text and \mathrm, and mhchem's \ce{...}.  Anything unknown degrades to
   readable plain text: a command loses its backslash, braces disappear.  No
   raw "$", no "\frac", no doubled backslash ever reaches the screen. */

const SYMBOLS = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", iota: "ι",
  kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ", upsilon: "υ", phi: "φ",
  varphi: "φ", chi: "χ", psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ",
  Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  rightarrow: "→", to: "→", leftarrow: "←", leftrightarrow: "↔", Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔",
  rightleftharpoons: "⇌", longrightarrow: "⟶", uparrow: "↑", downarrow: "↓", mapsto: "↦",
  cdot: "·", times: "×", div: "÷", pm: "±", mp: "∓", le: "≤", leq: "≤", ge: "≥", geq: "≥", ne: "≠", neq: "≠", approx: "≈",
  equiv: "≡", sim: "∼", simeq: "≃", propto: "∝", infty: "∞", partial: "∂", nabla: "∇", sum: "∑", prod: "∏", int: "∫",
  sqrt: "√", degree: "°", circ: "°", bullet: "•", ldots: "…", cdots: "⋯", dots: "…", quad: "  ", qquad: "    ", ";": " ", ",": " ",
  " ": " ", "{": "{", "}": "}", "%": "%", "&": "&", "#": "#", "_": "_", "^": "^", "$": "$", "\\": "\n",
  ln: "ln", log: "log", exp: "exp", sin: "sin", cos: "cos", tan: "tan", lim: "lim", max: "max", min: "min", det: "det",
  langle: "⟨", rangle: "⟩", lvert: "|", rvert: "|", vert: "|", mid: "|", forall: "∀", exists: "∃", in: "∈", notin: "∉",
  subset: "⊂", cup: "∪", cap: "∩", emptyset: "∅", neg: "¬", land: "∧", lor: "∨", hbar: "ℏ", ell: "ℓ", Re: "ℜ", Im: "ℑ",
  left: "", right: "", displaystyle: "", textstyle: "", mathrm: "", mathbf: "", mathit: "", text: "", operatorname: "", mathcal: "",
  bigl: "", bigr: "", Bigl: "", Bigr: "", big: "", Big: "", nonumber: "", label: "",
};

/* Operators written as words keep a space before an operand: "ln Q", not "lnQ". */
const FUNCTIONS = new Set(["ln", "log", "exp", "sin", "cos", "tan", "lim", "max", "min", "det"]);

const SUB = { "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "(": "₍", ")": "₎" };
const SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "(": "⁽", ")": "⁾" };

export function escapeHtml(text) {
  return String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const mapChars = (text, table) => String(text).split("").map((c) => table[c] ?? c).join("");

/* mhchem: H2O -> H₂O, Ca^2+ -> Ca²⁺, HCO3- -> HCO₃⁻, -> -> →, <=> -> ⇌. */
export function chemistryToUnicode(formula) {
  let t = String(formula).trim();
  for (const [from, to] of [["<=>>", "⇌"], ["<<=>", "⇌"], ["<=>", "⇌"], ["<->", "↔"], ["->", "→"], ["<-", "←"]]) t = t.split(from).join(to);
  t = t.replace(/\^(\{[^}]*\}|[0-9]*[+\-])/g, (_, body) => mapChars(body.replace(/[{}]/g, ""), SUP));
  t = t.replace(/(?<=[A-Za-z0-9)\]])([+\-])(?=$|[\s,;.)\]→⇌↔←])/g, (_, sign) => SUP[sign]);
  t = t.replace(/(?<=[A-Za-z)\]])(\d+)/g, (_, digits) => mapChars(digits, SUB));
  t = t.replace(/_\{?(\d+)\}?/g, (_, digits) => mapChars(digits, SUB));
  return t;
}

/* Read a balanced {...} group starting at `i` (which must be "{"); returns [content, nextIndex]. */
function group(src, i) {
  if (src[i] !== "{") return [src[i] ?? "", i + 1];
  let depth = 0;
  for (let j = i; j < src.length; j += 1) {
    if (src[j] === "{") depth += 1;
    else if (src[j] === "}") { depth -= 1; if (depth === 0) return [src.slice(i + 1, j), j + 1]; }
  }
  return [src.slice(i + 1), src.length];
}

function argument(src, i) {
  // The argument of ^ or _: a group, a command, or one character.
  if (src[i] === "{") return group(src, i);
  if (src[i] === "\\") { const m = /^\\[a-zA-Z]+/.exec(src.slice(i)); if (m) return [m[0], i + m[0].length]; return [src[i + 1] ?? "", i + 2]; }
  return [src[i] ?? "", i + 1];
}

/* Plain-text sub/superscripts where Unicode has the glyphs; HTML otherwise. */
function script(content, kind) {
  const inner = texToHtml(content);
  const plain = inner.replace(/<[^>]+>/g, "");
  if (kind === "sup" && (plain === "°" || plain === "∘")) return "°";
  const table = kind === "sub" ? SUB : SUP;
  if (plain.length > 0 && plain.length <= 4 && plain.split("").every((c) => table[c])) return mapChars(plain, table);
  return `<${kind}>${inner}</${kind}>`;
}

export function texToHtml(src) {
  let out = "";
  let i = 0;
  const s = String(src ?? "");
  while (i < s.length) {
    const c = s[i];
    if (c === "\\") {
      const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i));
      const name = m ? m[1] : "";
      i += m ? m[0].length : 1;
      // TeX swallows the whitespace after a command name.
      if (/^[a-zA-Z]+$/.test(name)) while (s[i] === " ") i += 1;
      if (FUNCTIONS.has(name)) {
        out += escapeHtml(SYMBOLS[name]) + (/[A-Za-z0-9(\\]/.test(s[i] ?? "") ? " " : "");
        continue;
      }
      if (name === "frac" || name === "dfrac" || name === "tfrac") {
        const [num, after] = group(s, skipSpace(s, i));
        const [den, after2] = group(s, skipSpace(s, after));
        i = after2;
        out += `<span class="frac"><span class="num">${texToHtml(num)}</span><span class="den">${texToHtml(den)}</span></span>`;
      } else if (name === "sqrt") {
        let index = "";
        let j = skipSpace(s, i);
        if (s[j] === "[") { const end = s.indexOf("]", j); index = s.slice(j + 1, end === -1 ? s.length : end); j = end === -1 ? s.length : end + 1; }
        const [body, after] = group(s, skipSpace(s, j));
        i = after;
        out += (index ? `<sup>${texToHtml(index)}</sup>` : "") + `√<span class="sqrt">${texToHtml(body)}</span>`;
      } else if (name === "ce" || name === "pu") {
        const [body, after] = group(s, skipSpace(s, i));
        i = after;
        out += escapeHtml(chemistryToUnicode(body));
      } else if (name === "text" || name === "mathrm" || name === "mathbf" || name === "mathit" || name === "operatorname" || name === "textbf" || name === "textit" || name === "mathcal") {
        const [body, after] = group(s, skipSpace(s, i));
        i = after;
        const inner = escapeHtml(body);
        out += name === "mathbf" || name === "textbf" ? `<b>${inner}</b>` : name === "mathit" || name === "textit" ? `<i>${inner}</i>` : inner;
      } else if (name === "overline" || name === "bar" || name === "hat" || name === "vec") {
        const [body, after] = group(s, skipSpace(s, i));
        i = after;
        out += `<span class="${name === "vec" ? "vec" : "over"}">${texToHtml(body)}</span>`;
      } else if (Object.prototype.hasOwnProperty.call(SYMBOLS, name)) {
        out += escapeHtml(SYMBOLS[name]);
      } else {
        out += escapeHtml(name); // unknown command: its name, never a backslash
      }
      continue;
    }
    if (c === "^" || c === "_") {
      const [body, after] = argument(s, i + 1);
      i = after;
      out += script(body, c === "^" ? "sup" : "sub");
      continue;
    }
    if (c === "{" || c === "}") { i += 1; continue; }
    if (c === "~") { out += " "; i += 1; continue; }
    out += escapeHtml(c);
    i += 1;
  }
  return out;
}

function skipSpace(s, i) { while (i < s.length && s[i] === " ") i += 1; return i; }

/* Every math delimiter form models emit: $$...$$, \[...\], $...$, \(...\).
   Returns [{kind: "text"|"inline"|"display", value}] -- an unmatched
   delimiter is text, so a partial stream never renders half an equation. */
export function splitMath(text) {
  const parts = [];
  const s = String(text ?? "");
  let i = 0;
  let textStart = 0;
  const push = (kind, value) => { if (value) parts.push({ kind, value }); };
  while (i < s.length) {
    let opener = null;
    if (s.startsWith("$$", i)) opener = ["$$", "$$", "display"];
    else if (s.startsWith("\\[", i)) opener = ["\\[", "\\]", "display"];
    else if (s.startsWith("\\(", i)) opener = ["\\(", "\\)", "inline"];
    else if (s[i] === "$" && !/\s/.test(s[i + 1] ?? " ") && s[i - 1] !== "\\") opener = ["$", "$", "inline"];
    if (!opener) { i += 1; continue; }
    const [open, close, kind] = opener;
    const end = s.indexOf(close, i + open.length);
    if (end === -1) { i += open.length; continue; }
    const body = s.slice(i + open.length, end);
    if (kind === "inline" && open === "$" && (/\n\n/.test(body) || body.length > 400 || /\s$/.test(body) || /^\s/.test(body) || /^\d+([.,]\d+)?(\s|$)/.test(body))) { i += 1; continue; }
    push("text", s.slice(textStart, i));
    push(kind, body);
    i = end + close.length;
    textStart = i;
  }
  push("text", s.slice(textStart));
  return parts;
}

/* Bare symbol commands in prose ("\Delta G", "\rightarrow"): the glyph, never the command. */
export function proseSymbols(text) {
  return String(text ?? "").replace(/\\([a-zA-Z]+)(?![a-zA-Z{])( ?)/g, (m, name, space) => {
    if (!Object.prototype.hasOwnProperty.call(SYMBOLS, name) || SYMBOLS[name] === "") return name + space;
    const glyph = SYMBOLS[name];
    // A Greek letter binds to what follows (ΔG, αβ); an arrow or operator keeps its spacing (A → B, ln Q).
    const letter = /^\p{L}$/u.test(glyph) && !FUNCTIONS.has(name);
    return glyph + (letter ? "" : space || (FUNCTIONS.has(name) ? " " : ""));
  });
}

export function renderMath(body, kind) {
  const html = texToHtml(body);
  return kind === "display" ? `<div class="math display">${html}</div>` : `<span class="math">${html}</span>`;
}
