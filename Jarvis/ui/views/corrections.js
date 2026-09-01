/* Korrekturen: what the owner has corrected, with scope, provenance and use
   count; and the Korrigieren dialog opened from any receipt. Four kinds are
   kept apart on screen: owner core, owner preference, owner correction,
   technical learned trajectory. */

import { $, el, clear, kv, section, badge, button, debounce, ago } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import { addTurn } from "./chat.js";

const KIND = {
  OWNER_PREFERENCE: ["Owner preference", "blue"],
  INTENT_ERROR: ["Owner correction", "amber"],
  ENTITY_RESOLUTION_ERROR: ["Owner correction", "amber"],
  PARAMETER_ERROR: ["Owner correction", "amber"],
  EXECUTION_FAILURE: ["Technical", "dim"],
  VERIFICATION_DEFECT: ["Technical", "dim"],
  CAPABILITY_DEFECT: ["Technical", "dim"],
};

export const view = {
  id: "corrections",
  title: "Korrekturen",
  async mount(pane, params) {
    const data = await api("/api/corrections");
    const rows = (data.corrections || []).slice().reverse();
    const filter = el("select", {}, el("option", { value: "", text: "all kinds" }),
      ...Object.keys(KIND).map((k) => el("option", { value: k, text: k.toLowerCase().replace(/_/g, " ") })));
    const search = el("input", { placeholder: "Search corrections…" });
    const list = el("div");
    const render = () => {
      clear(list);
      const q = search.value.toLowerCase();
      const shown = rows.filter((c) => (!filter.value || c.classification === filter.value) &&
        (!q || `${c.what_was_wrong} ${c.original_request}`.toLowerCase().includes(q)));
      if (!shown.length) { list.append(el("div", { class: "empty", text: rows.length ? "Nothing matches." : "Noch keine Korrekturen. Every receipt carries „Korrigieren“." })); return; }
      for (const c of shown) list.append(card(c, () => views.open("corrections", params)));
    };
    filter.onchange = render;
    search.oninput = render;
    pane.append(el("div", { class: "toolbar" }, search, filter, el("span", { class: "empty", style: { padding: 0 }, text: `${rows.length} learned` })));
    pane.append(list);
    render();
  },
};

function card(c, reload) {
  const [label, tone] = KIND[c.classification] || ["Owner correction", "amber"];
  const node = el("div", { class: "card click" + (c.active ? "" : " off") });
  node.append(
    el("div", { class: "title", text: c.what_was_wrong }),
    el("div", { class: "meta" }, badge(label, tone), badge(c.scope.toLowerCase().replace(/_/g, " "), "dim"),
      el("span", { text: `${c.applied_count}× applied` }), el("span", { text: ago(c.at) }), c.active ? null : badge("disabled", "bad")),
  );
  node.onclick = () => inspect(c, reload);
  return node;
}

function inspect(c, reload) {
  const rule = c.then && c.then.overrides ? JSON.stringify(c.then.overrides) : "";
  const scopeSel = el("select", {}, ...["THIS_REQUEST", "INTENT_SPECIFIC", "ENTITY_SPECIFIC", "DOMAIN_SPECIFIC", "GLOBAL_OWNER_PREFERENCE"]
    .map((s) => el("option", { value: s, text: s.toLowerCase().replace(/_/g, " "), selected: s === c.scope })));
  const note = el("textarea", { value: c.what_was_wrong });
  views.inspect("Korrektur",
    section("What was wrong", el("div", { class: "field" }, note)),
    section("Why it was learned", kv("request", c.original_request), kv("read as", c.parsed_intent || ""), kv("receipt", c.receipt_id || ""),
      kv("classification", c.classification), kv("reason", c.reason || ""), kv("learned", c.at), kv("applied", `${c.applied_count}×`)),
    section("Scope", el("div", { class: "field" }, scopeSel), kv("when", JSON.stringify(c.when || {}), "mono"), rule ? kv("rule", rule, "mono") : null),
    el("div", { class: "toolbar" },
      button("Save", async () => {
        const r = await api("/api/correction/update", { correction_id: c.correction_id, changes: { what_was_wrong: note.value, scope: scopeSel.value } });
        if (r.ok) { reload(); views.closeInspector(); }
      }, "primary"),
      button(c.active ? "Disable" : "Enable", async () => { await api("/api/correction/update", { correction_id: c.correction_id, changes: { active: !c.active } }); reload(); views.closeInspector(); }),
      button("Delete", async () => { await api("/api/correction/delete", { correction_id: c.correction_id }); reload(); views.closeInspector(); }, "ghost danger"),
    ),
  );
}

/* ---- the Korrigieren dialog (from a receipt) ---------------------- */

export async function openDialog(receiptId) {
  const ctx = await api("/api/correction/context", { receipt_id: receiptId });
  if (!ctx.ok) { addTurn("error", "Error", ctx.error || "no context"); return; }
  const body = el("div", { class: "correction" });
  const rows = [
    ["Anfrage", ctx.original_request],
    ["Gelesen als", `${ctx.parsed_intent} — ${ctx.intent_reason}`],
    ["Entitäten", Object.keys(ctx.entities || {}).length ? JSON.stringify(ctx.entities) : "—"],
    ["Ausgeführt", ctx.executed_action],
    ["Beobachtet", ctx.observed_result],
    ["Beleg", `${receiptId} · ${ctx.verified ? "verifiziert" : ctx.ok_flag ? "gelaufen, unverifiziert" : "fehlgeschlagen"}`],
  ];
  for (const [k, v] of rows) body.append(el("div", { class: "row" }, el("span", { class: "k", text: k }), el("span", { class: "v", text: v || "—" })));
  // What went wrong, in the owner's words. The category decides which
  // system learns: MISHEARD -> the recogniser's vocabulary, WRONG_INTENT ->
  // the router, WRONG_TARGET -> the resolver, WRONG_RESULT -> the verifier,
  // PRONUNCIATION -> the lexicon. The protected personality never changes here.
  const CATEGORIES = [
    ["MISHEARD", "Falsch gehört", "z. B. „Starkfisch → Stockfish“ oder „ich meinte Stockfish“"],
    ["WRONG_INTENT", "Falsch verstanden", "z. B. „Das war eine Frage, keine Aktion“"],
    ["WRONG_TARGET", "Falsches Ziel", "z. B. „Ich meinte das Projekt Biochemie, nicht Bio“"],
    ["WRONG_RESULT", "Falsches Ergebnis", "z. B. „Es lief nicht Rammstein, sondern etwas anderes“"],
    ["INCOMPLETE", "Unvollständig", "z. B. „Die dritte Aufgabe fehlt“"],
    ["PRONUNCIATION", "Aussprache", "z. B. „Sprich ‚Spotify‘ wie ‚Spottifai‘ aus“"],
    ["OTHER", "Anderes", "ein Satz, was anders sein soll"],
  ];
  let category = "";
  const chips = el("div", { class: "chips" });
  const input = el("textarea", { rows: 3, placeholder: "Was war falsch? Ein Satz genügt." });
  for (const [key, label, hint] of CATEGORIES) {
    chips.append(el("button", { class: "chip", type: "button", text: label, title: hint, onClick: (ev) => {
      category = key;
      for (const c of chips.querySelectorAll(".chip")) c.classList.toggle("on", c === ev.currentTarget);
      input.placeholder = hint;
      input.focus();
    } }));
  }
  const guess = el("div", { class: "guess" });
  const classSel = el("select");
  const scopeSel = el("select");
  input.oninput = debounce(async () => {
    const g = await api("/api/correction/classify", { what_was_wrong: input.value, receipt_id: receiptId });
    if (!g.ok) return;
    fill(classSel, g.classes, g.classification);
    fill(scopeSel, g.scopes, g.scope);
    guess.textContent = g.reason + (g.then && g.then.overrides ? ` · Regel: ${JSON.stringify(g.then.overrides)}` : "");
  }, 350);
  const actions = el("div", { class: "row" });
  const save = async (rerun) => {
    const result = await api("/api/correction/save", {
      what_was_wrong: input.value, receipt_id: receiptId, classification: category ? "" : classSel.value, scope: scopeSel.value, rerun, category,
    });
    if (result.ok) {
      $("panel").classList.remove("open");
      addTurn("note", "", `Gelernt: ${result.correction ? result.correction.classification : result.classification} · ${(result.correction ? result.correction.scope : result.scope || "").toLowerCase().replace(/_/g, " ")}` +
        (result.vocabulary ? ` · „${result.vocabulary.heard}“ → „${result.vocabulary.meant}“` : "") +
        (result.rerun ? ` · erneut ausgeführt` : ""));
    } else {
      guess.textContent = result.error || "nicht gespeichert";
    }
  };
  actions.append(
    button("Korrigieren & lernen", () => save(false), "primary"),
    button("Jetzt korrigiert erneut ausführen", () => save(true)),
    button("Nur diesmal", () => { scopeSel.value = "THIS_REQUEST"; save(true); }),
  );
  body.append(el("div", { class: "k", text: "Was war falsch?" }), chips, input, guess, el("details", {}, el("summary", { text: "Klasse und Reichweite (automatisch)" }), el("div", { class: "row" }, classSel, scopeSel)), actions);
  $("panelTitle").textContent = "Korrigieren";
  clear($("panelBody")).append(body);
  $("panel").classList.add("open");
  input.focus();
}

function fill(select, options, chosen) {
  const current = select.value;
  clear(select);
  for (const o of options || []) select.append(el("option", { value: o, text: o.toLowerCase().replace(/_/g, " ") }));
  select.value = (options || []).includes(current) && current !== "" ? current : chosen;
}
