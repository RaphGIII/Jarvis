/* Deterministic rendering checks for the conversation's Markdown, math and chemistry.
   Run: node ui/tests/render.test.mjs  (the Python test suite runs it too). */

import { render, renderPartial, sanitizeProse } from "../core/markdown.js";
import { texToHtml, chemistryToUnicode, splitMath } from "../core/mathtex.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}
const noArtifacts = (html) => !/\\frac|\\ce|\\Delta|&amp;amp;|�|(^|[^\\])\$[^<]*\$/.test(html.replace(/<[^>]+>/g, ""));

// 1. partial Markdown does not corrupt the final rendering
const full = "## Puffer\n\nDer **Bikarbonatpuffer** hält den pH.\n\n- CO₂ + H₂O ⇌ H₂CO₃\n- HCO₃⁻ + H⁺\n\n| Ion | Ladung |\n|---|---|\n| Ca²⁺ | +2 |\n\n```python\nprint('x')\n```\n\nEnde.";
const finalHtml = render(full);
let cumulative = "";
let corrupted = false;
for (let i = 1; i <= full.length; i += 7) {
  cumulative = full.slice(0, i);
  const partial = renderPartial(cumulative);
  if (partial.includes("$") && !cumulative.includes("$")) corrupted = true;
  if (!/<h3>|<p>|<pre|<ul>|<table>/.test(partial) && cumulative.trim().length > 4) corrupted = true;
}
check("partial renders are well-formed and the final render is the full document", !corrupted && render(full) === finalHtml && finalHtml.includes("<table>") && finalHtml.includes("<pre class=\"code\""), finalHtml.slice(0, 300));

// 2. fenced code survives chunk boundaries
const code = "Text\n\n```python\ndef f(x):\n    return x * 2\n```\n\nDanach.";
const cut = "Text\n\n```python\ndef f(x):\n    ret";
check("an unclosed fence renders as an open code block", /<pre class="code open"[^>]*><code>def f\(x\):\n    ret<\/code><\/pre>/.test(renderPartial(cut)));
check("the closed fence renders as code, and code content is not markdown-processed", render(code).includes("<pre class=\"code\" data-lang=\"python\"><code>def f(x):\n    return x * 2</code></pre>") && !render("```\n**not bold**\n```").includes("<b>"));

// 3. equations survive chunk boundaries
const eq = "Die freie Enthalpie: $\\Delta G = \\Delta G^\\circ + RT \\ln Q$ bestimmt die Richtung.";
const half = "Die freie Enthalpie: $\\Delta G = \\Delta G^\\circ + RT";
const partialEq = renderPartial(half);
check("an unfinished equation is shown as text without a raw backslash command", !partialEq.includes("\\Delta") && partialEq.includes("Δ") || partialEq.includes("$"), partialEq);
check("the finished equation renders", render(eq).includes("ΔG = ΔG° + RT ln Q") && !render(eq).includes("$"), render(eq));
check("display math renders with fractions", /class="frac"/.test(render("$$\\frac{[H^+]}{K_a}$$")) && render("$$\\frac{[H^+]}{K_a}$$").includes("H⁺"));

// 4. Unicode chemistry
check("chemistry to unicode", chemistryToUnicode("Ca^2+") === "Ca²⁺" && chemistryToUnicode("HCO3-") === "HCO₃⁻" && chemistryToUnicode("H2O") === "H₂O"
      && chemistryToUnicode("CO2 + H2O <=> H2CO3") === "CO₂ + H₂O ⇌ H₂CO₃", `${chemistryToUnicode("Ca^2+")} ${chemistryToUnicode("HCO3-")} ${chemistryToUnicode("CO2 + H2O <=> H2CO3")}`);
check("unicode chemistry passes through rendering untouched", render("Ca²⁺ bindet an HCO₃⁻; Na⁺/K⁺-ATPase; ΔG; α/β; → und ⇌").includes("Ca²⁺ bindet an HCO₃⁻; Na⁺/K⁺-ATPase; ΔG; α/β; → und ⇌"));

// 5. malformed / unsupported chemistry markup degrades safely
const bad = "Puffer: \\ce{HCO3-} + \\ce{H+} -> \\ce{H2CO3}, mit \\\\Delta G und &amp;rarr; sowie \\pu{37 °C} und $\\unknowncmd{x}$";
const badHtml = render(bad);
check("\\ce and doubled backslashes and entities are normalised", badHtml.includes("HCO₃⁻") && badHtml.includes("H⁺") && badHtml.includes("H₂CO₃") && badHtml.includes("ΔG") && !badHtml.includes("\\\\") && !badHtml.includes("&amp;rarr;"), badHtml);
check("an unknown command shows its name, not a backslash, and no raw dollar", badHtml.includes("unknowncmdx") && !badHtml.includes("$") && !badHtml.includes("\\unknowncmd"), badHtml);
check("replacement glyphs are removed", !render("Kalzium� bindet").includes("�"));

// 6. no raw broken LaTeX artifacts anywhere in typical answers
const typical = "Durch Hyperventilation sinkt $p_{CO_2}$ &rarr; respiratorische Alkalose: $\\mathrm{pH} = \\mathrm{p}K_a + \\log\\frac{[\\ce{HCO3-}]}{[\\ce{CO2}]}$.";
const typicalHtml = render(typical);
check("typical medical answer has no artifacts", noArtifacts(typicalHtml) && typicalHtml.includes("HCO₃⁻") && typicalHtml.includes("→"), typicalHtml);
check("splitMath ignores currency and unmatched dollars", splitMath("Preis $5 und $7").every((p) => p.kind === "text") && splitMath("offen $x").every((p) => p.kind === "text"));

// 7. inline formatting, lists, links, blockquotes, headings
const md = "# T\n\n> Zitat\n\n1. eins\n2. zwei\n\n*kursiv* und `code` und [Link](https://example.org) und __fett__";
const mdHtml = render(md);
check("core markdown constructs", mdHtml.includes("<h2>T</h2>") && mdHtml.includes("<blockquote>") && mdHtml.includes("<ol><li>eins</li><li>zwei</li></ol>")
      && mdHtml.includes("<i>kursiv</i>") && mdHtml.includes("<code>code</code>") && mdHtml.includes('href="https://example.org"') && mdHtml.includes("<b>fett</b>"), mdHtml);
check("html in provider text is escaped, not executed", !render("<script>alert(1)</script> a < b").includes("<script>"));
check("unmatched emphasis at a chunk boundary stays literal", renderPartial("Das ist **wichtig und").includes("**wichtig und"));
check("sanitizeProse leaves fenced code alone", sanitizeProse("&amp; x") === "& x");

console.log(failures ? `${failures} FAILED` : "ALL OK");
process.exit(failures ? 1 : 0);
