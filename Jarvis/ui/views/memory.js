/* Gedächtnis: what ZEUS keeps about the owner -- and only that.

   Personal memory is kept apart from study knowledge (Studium) and project
   knowledge (Projekte).  Here: the facts the owner asked ZEUS to remember
   with the password, the rules of the relationship and the way ZEUS speaks,
   the preferences learned from feedback, and the corrections that change how
   ZEUS acts.  Every row is a record from the store that holds it; editing
   stays where the rules for editing live. */

import { el } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";

export const view = {
  id: "memory",
  title: "Gedächtnis",
  async mount(pane) {
    const root = el("div", { class: "memory" },
      el("header", { class: "mm-head" }, el("h1", { class: "mm-title", text: "Gedächtnis" }),
        el("p", { class: "mm-lead", text: "Was ZEUS über dich weiß und wie es mit dir umgeht. Lernmaterial liegt im Studium, Projektwissen in den Projekten." })));
    pane.append(root);
    const [facts, personality, adaptation, corrections] = await Promise.all([
      api("/api/knowledge/graph", { query: "geschützt", limit: 60 }), api("/api/personality"), api("/api/adaptation"), api("/api/corrections"),
    ]);
    root.append(section("Über dich", "Mit deinem Passwort gespeichert.",
      ((facts && facts.nodes) || []).filter((n) => (n.tags || []).includes("geschützt")).map((n) =>
        row(String(n.title || "").replace(/^Geschützt:\s*/, ""), dateOf(n.updated_at || n.created_at))),
      "Noch nichts. Sag zum Beispiel: „Merke dir, dass …“ – ZEUS fragt nach deinem Passwort."));
    const labels = Object.fromEntries(((personality && personality.categories) || []).map((c) => [c.id, c.label]));
    root.append(section("Beziehung und Regeln", "Gilt in jeder Antwort.",
      ((personality && personality.rules) || []).filter((r) => r.enabled !== false).map((r) => row(r.text, labels[r.category] || "", r.protected ? "geschützt" : "")),
      "Noch keine eigenen Regeln.", el("button", { class: "mm-link", text: "Persönlichkeit bearbeiten", onClick: () => views.open("settings", { tab: "personality" }) })));
    root.append(section("Gelernte Vorlieben", "Aus deinem Feedback.",
      ((adaptation && adaptation.rules) || []).map((r) => row(r.text, r.enabled ? "aktiv" : "aus")),
      "Noch nichts gelernt. Ein 👎 unter einer Antwort bringt ZEUS etwas bei."));
    root.append(section("Korrekturen", "So handelt ZEUS seit deiner Korrektur anders.",
      ((corrections && corrections.corrections) || []).filter((c) => c.active !== false).slice(0, 12).map((c) => row(c.what_was_wrong || c.original_request, dateOf(c.at))),
      "Keine Korrekturen.", el("button", { class: "mm-link", text: "Alle Korrekturen", onClick: () => views.open("corrections", {}) })));
  },
};

function dateOf(iso) {
  const d = iso ? new Date(iso) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleDateString("de-DE", { day: "numeric", month: "short", year: "numeric" }) : "";
}

function row(text, meta = "", mark = "") {
  return el("div", { class: "mm-row", role: "listitem" }, el("span", { class: "mm-text", text }),
    mark ? el("span", { class: "mm-mark", text: mark }) : null, meta ? el("span", { class: "mm-meta", text: meta }) : null);
}

function section(title, lead, rows, empty, action = null) {
  const list = el("div", { class: "mm-list", role: "list" });
  if (rows.length) list.append(...rows);
  else list.append(el("div", { class: "mm-empty", text: empty }));
  return el("section", { class: "mm-section" },
    el("div", { class: "mm-section-head" }, el("h2", { text: title }), el("span", { class: "mm-section-lead", text: lead }), action),
    list);
}
