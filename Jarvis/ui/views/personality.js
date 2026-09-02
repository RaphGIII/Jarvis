/* Persönlichkeit: the owner's behavioural control centre.

   One coherent surface, top to bottom exactly the way behaviour is built:

                     ZEUS CORE
                IDENTITY | BEHAVIOR
                        ↓
                   OWNER RULES
                        ↓
                LEARNED PREFERENCES
                        ↓
               EFFECTIVE PERSONALITY

   Every box is real data from the server; the source inspector names where
   each layer lives and who may write it; owner rules always outrank learned
   ones; protected edits go through the password gate.  Secrets are never
   rendered.  (Security and system ownership live in the Owner view.) */

import { el, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import { personalityPanel } from "./owner.js";

export const view = {
  id: "personality",
  title: "Persönlichkeit",
  async mount(pane, params) {
    const [p, adapt, auth] = await Promise.all([
      api("/api/owner/personality"), api("/api/adaptation"), api("/api/auth/status")]);
    if (p.ok === false) { pane.append(el("div", { class: "empty", text: p.error || "unavailable" })); return; }
    const reload = () => views.open("personality");
    const core = p.core || {};
    const prefs = p.preferences || {};
    const rules = adapt.rules || [];
    const owned = rules.filter((r) => r.source === "OWNER_RULE");
    const learned = rules.filter((r) => r.source !== "OWNER_RULE");

    // ---- the hierarchy, as a visual flow --------------------------------
    const node = (title, lines, tone = "") => el("div", { class: "pnode " + tone },
      el("div", { class: "pnode-title", text: title }),
      ...lines.filter(Boolean).map((l) => el("div", { class: "pnode-line", text: l })));
    const flow = el("div", { class: "pflow" },
      el("div", { class: "prow one" }, node("ZEUS CORE", [
        `Version ${p.version ?? "?"} · geschützt` , auth.configured ? "Passwort-Gate aktiv" : "noch kein Passwort gesetzt"], "root")),
      el("div", { class: "pjoin split" }),
      el("div", { class: "prow two" },
        node("IDENTITÄT", [
          (core.character || []).slice(0, 2).join(" · ") || "—",
          `Sprache: Deutsch · Anrede: ${prefs.address || "du"}`,
        ]),
        node("VERHALTEN", [
          `Initiative ${prefs.initiative ?? 50} · Humor ${prefs.humour ?? 50}`,
          `Detailtiefe ${prefs.technical_depth ?? 50} · Wärme ${prefs.warmth ?? 50}`,
          `Direktheit ${prefs.directness ?? 60} · Nüchternheit ${prefs.sobriety ?? 50}`,
          "Ehrlichkeit: fest verdrahtet",
        ])),
      el("div", { class: "pjoin" }),
      el("div", { class: "prow one" }, node("OWNER-REGELN", [
        owned.length ? `${owned.length} Regel${owned.length === 1 ? "" : "n"} — gelten immer zuerst` : "noch keine — unten anlegen"], "owner")),
      el("div", { class: "pjoin" }),
      el("div", { class: "prow one" }, node("GELERNTE PRÄFERENZEN", [
        learned.length ? `${learned.length} aus deinem Feedback · begrenzt & abklingend` : "noch nichts gelernt — 👍/👎 unter Antworten füttert dies"], "learned")),
      el("div", { class: "pjoin" }),
      el("div", { class: "prow one" }, node("EFFEKTIVE PERSÖNLICHKEIT", [
        `${(p.blocks || []).length} Prompt-Blöcke in fester Reihenfolge — unten wörtlich einsehbar`], "eff")));
    pane.append(el("div", { class: "pflow-wrap" }, flow));

    // ---- rules: owner vs learned, with evidence and full control --------
    pane.append(section("Regeln", rulesBoard(owned, learned, reload)));

    // ---- configuration sources with a real inspector --------------------
    pane.append(section("Konfigurations-Quellen", await sourcesBoard(p, adapt, reload)));

    // ---- the working controls (dials, protected core, prompt) -----------
    pane.append(el("details", { class: "padv" },
      el("summary", { text: "Feinabstimmung: Regler, geschützter Kern, effektiver Prompt" }),
      await personalityPanel(reload)));
  },
};

/* ---- rules ------------------------------------------------------------ */
function ruleCard(r, { learned }, reload) {
  const conf = Math.round((r.effective_confidence ?? r.confidence ?? 0) * 100);
  const dir = r.weight > 0 ? "→ mehr davon" : r.weight < 0 ? "→ weniger davon" : "";
  const evidence = (r.evidence || []).length;
  const box = el("div", { class: "prule" + (r.enabled === false ? " off" : "") },
    el("div", { class: "prule-head" },
      badge(learned ? "GELERNT" : "OWNER", learned ? "blue" : "amber"),
      el("span", { class: "prule-text", text: r.text || r.domain })),
    el("div", { class: "prule-meta" },
      el("span", { text: scopeLabel(r.scope) }),
      el("span", { text: `Gewicht ${Number(r.weight).toFixed(2)} ${dir}` }),
      el("span", { text: `Konfidenz ${conf}%` }),
      evidence ? el("span", { text: `Belege: ${evidence}× Feedback` }) : null,
      r.enabled === false ? badge("AUS", "dim") : null),
    el("div", { class: "toolbar" },
      button(r.enabled === false ? "Aktivieren" : "Deaktivieren", async () => {
        await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "update", changes: { enabled: r.enabled === false } }); reload();
      }, "ghost"),
      button("Bearbeiten", async () => {
        const text = prompt("Regeltext:", r.text || "");
        if (text === null || !text.trim()) return;
        await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "update", changes: { text: text.trim() } }); reload();
      }, "ghost"),
      learned ? button("Zur Owner-Regel machen", async () => {
        await api("/api/adaptation/rule", { action: "add", text: r.text || "", domain: r.domain || "STYLE", scope: r.scope || {} });
        await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "delete" });
        reload();
      }, "ghost") : null,
      button("Löschen", async () => { await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "delete" }); reload(); }, "ghost danger")));
  return box;
}

const SCOPE_LABELS = {
  technical_explanation: "bei technischen Erklärungen", action_confirmation: "bei Bestätigungen",
  small_talk: "im Smalltalk", conversation: "im Gespräch",
};
function scopeLabel(scope) {
  const kind = (scope || {}).kind || "";
  return SCOPE_LABELS[kind] || (kind ? `Kontext: ${kind}` : "überall");
}

function rulesBoard(owned, learned, reload) {
  const box = el("div");
  const cols = el("div", { class: "prule-cols" },
    el("div", {}, el("h5", { class: "prule-colhead", text: `OWNER-REGELN (${owned.length}) — gelten immer zuerst` }),
      owned.length ? el("div", {}, ...owned.map((r) => ruleCard(r, { learned: false }, reload)))
        : el("div", { class: "empty", text: "Noch keine eigene Regel." })),
    el("div", {}, el("h5", { class: "prule-colhead", text: `GELERNT AUS FEEDBACK (${learned.length}) — begrenzt, abklingend, löschbar` }),
      learned.length ? el("div", {}, ...learned.map((r) => ruleCard(r, { learned: true }, reload)))
        : el("div", { class: "empty", text: "Noch nichts gelernt. 👍/👎 und Korrekturen unter Antworten landen hier." })));
  const text = el("input", { placeholder: "Neue Regel, z. B. „Bei medizinischen Erklärungen ausführlicher antworten.“", style: { minWidth: "360px" } });
  const kind = el("select", {}, ...[["technical_explanation", "bei technischen Erklärungen"], ["action_confirmation", "bei Bestätigungen"], ["small_talk", "im Smalltalk"], ["conversation", "im Gespräch"], ["", "überall"]]
    .map(([v, t]) => el("option", { value: v, text: t })));
  box.append(cols, el("div", { class: "toolbar" }, text, kind,
    button("Owner-Regel anlegen", async () => {
      if (!text.value.trim()) return;
      await api("/api/adaptation/rule", { action: "add", text: text.value.trim(), domain: "STYLE", scope: kind.value ? { kind: kind.value } : {} });
      reload();
    }, "primary")));
  return box;
}

/* ---- configuration sources ------------------------------------------- */
async function sourcesBoard(p, adapt, reload) {
  const owner = await api("/api/owner").catch(() => ({}));
  const docs = owner.documents || {};
  const mask = (obj) => Object.fromEntries(Object.entries(obj || {}).map(([k, v]) =>
    [k, /secret|token|password|key$/i.test(k) ? "••••••" : v]));
  const lastChange = (name) => {
    const h = (owner.history || []).filter((x) => (x.documents || []).includes(name)).at(-1);
    return h ? (h.at || h.applied_at || "").slice(0, 19) : "";
  };
  const SOURCES = [
    ["Identität", "wer Zeus ist", () => inspectValues("Identität", mask(docs.identity), "data/jarvis/owner · Dokument identity", lastChange("identity"))],
    ["Persönlichkeits-Kern", "Charakter, Epistemik — geschützt", () => inspectValues("Persönlichkeits-Kern", p.core || {}, `data/jarvis/owner · Version ${p.version ?? "?"} · PERSONALITY_EDIT nötig`, (p.history || []).at(-1)?.at?.slice(0, 19) || "")],
    ["Gesprächsregeln", "wie Zeus spricht", () => inspectValues("Gesprächsregeln", { conversation: (p.core || {}).conversation || [] }, "Teil des geschützten Kerns", "")],
    ["Präferenzen (Regler)", "Knappheit, Humor, Wärme…", () => inspectValues("Präferenzen", p.preferences || {}, "data/jarvis/owner · Vorschlag + Bestätigung", (p.history || []).at(-1)?.at?.slice(0, 19) || "")],
    ["Adaptive Regeln", "aus deinem Feedback", () => inspectValues("Adaptive Regeln", Object.fromEntries((adapt.rules || []).map((r) => [r.rule_id, `${r.text} (${Number(r.weight).toFixed(2)})`])), "data/jarvis/owner/adaptive.json · dein Feedback + deterministischer Code", "")],
    ["Aussprache & Sprach-Lexikon", "heard→meant, gesprochene Formen", () => views.open("voice")],
    ["Sicherheitsrichtlinie", "Passwort-Gate, Scopes", () => inspectValues("Sicherheitsrichtlinie", mask(docs.security), "data/jarvis/owner · Dokument security · SECURITY_CONFIG", lastChange("security"))],
    ["Berechtigungen (Policy)", "was Zeus darf", () => inspectValues("Policy", mask(docs.policy), "data/jarvis/owner · Dokument policy", lastChange("policy"))],
    ["Kosten (Spending)", "Expert-/Kosten-Grenzen", () => inspectValues("Spending", mask(docs.spending), "data/jarvis/owner · Dokument spending", lastChange("spending"))],
  ];
  const grid = el("div", { class: "psources" });
  for (const [name, what, openIt] of SOURCES) {
    const row = el("div", { class: "psource" },
      el("div", { class: "psource-name", text: name }),
      el("div", { class: "psource-what", text: what }));
    row.onclick = openIt;
    grid.append(row);
  }
  return grid;
}

function inspectValues(title, values, source, changed) {
  const rows = Object.entries(values || {});
  views.inspect(title,
    el("div", { class: "meta" }, badge("QUELLE", "blue"), el("span", { text: " " + source }),
      changed ? el("span", { text: ` · zuletzt geändert ${changed}` }) : null),
    rows.length ? el("div", {}, ...rows.map(([k, v]) => kv(k, Array.isArray(v) ? v.join("\n") : typeof v === "object" && v !== null ? JSON.stringify(v, null, 1) : String(v))))
      : el("div", { class: "empty", text: "(leer)" }),
    el("div", { class: "toolbar" },
      button("Im Owner-Bereich ändern (geschützt)", () => views.open("owner"), "ghost")));
}
