/* Settings: Allgemein · Persönlichkeit · Erscheinungsbild · Leistung & Kosten
   · Datenschutz · Voice · Erweitert.

   The ordinary owner never meets a provider, a model or an internal role
   here.  "Erweitert" holds the technical truth -- the former Owner panel with
   its providers, credentials, protected operations and diagnostics -- and is
   the only surface that may show it.  The personality editor writes the
   owner's editable layer through /api/personality/save (versioned, source
   OWNER_UI); the protected identity is shown, never edited from here. */

import { $, el, clear, kv } from "../core/dom.js";
import { api } from "../core/api.js";
import { state, setPref } from "../core/state.js";
import * as views from "../core/views.js";
import * as authgate from "../core/authgate.js";
import { LEVELS, labelFor } from "../core/performance.js";

const TABS = [
  ["general", "Allgemein", "◎"],
  ["personality", "Persönlichkeit", "✦"],
  ["appearance", "Erscheinungsbild", "◑"],
  ["performance", "Leistung & Kosten", "◈"],
  ["privacy", "Datenschutz", "◇"],
  ["voice", "Voice", "◍"],
  ["advanced", "Erweitert", "⚙"],
];

const DIALS = [
  ["conciseness", "Knapp ↔ ausführlich", "knapp", "ausführlich"],
  ["formality", "Locker ↔ formell", "locker", "formell"],
  ["warmth", "Sachlich ↔ warm", "sachlich", "warm"],
  ["directness", "Diplomatisch ↔ direkt", "diplomatisch", "direkt"],
  ["technical_depth", "Zugänglich ↔ technisch", "zugänglich", "technisch"],
  ["humour", "Ernst ↔ humorvoll", "ernst", "humorvoll"],
  ["proactivity", "Zurückhaltend ↔ proaktiv", "zurückhaltend", "proaktiv"],
];
const RESPONSE = [
  ["answer_length", "Standard-Antwortlänge", [["auto", "passend"], ["brief", "kurz"], ["normal", "normal"], ["detailed", "ausführlich"]]],
  ["structure", "Struktur", [["auto", "passend"], ["prose", "Fließtext"], ["structured", "strukturiert"]]],
  ["headings", "Überschriften", [["auto", "bei Bedarf"], ["yes", "ja"], ["no", "nein"]]],
  ["bullets", "Aufzählungen", [["auto", "bei Bedarf"], ["yes", "ja"], ["no", "nein"]]],
  ["examples", "Beispiele", [["rarely", "selten"], ["sometimes", "manchmal"], ["often", "oft"]]],
  ["equations", "Gleichungen", [["exact", "exakt"], ["simplified", "vereinfacht"], ["avoid", "vermeiden"]]],
  ["clinical_relevance", "Klinische Relevanz", [["when_relevant", "wenn medizinisch"], ["always", "immer"], ["never", "nie"]]],
  ["code_explanation", "Code-Erklärung", [["brief", "kurz"], ["thorough", "gründlich"], ["none", "keine"]]],
  ["follow_up", "Anschlussvorschläge", [["when_useful", "wenn nützlich"], ["always", "immer"], ["never", "nie"]]],
];

export const view = {
  id: "settings",
  title: "Einstellungen",
  async mount(pane, params) {
    const tab = TABS.some(([id]) => id === params.tab) ? params.tab : "general";
    const nav = el("nav", { class: "settings-nav" });
    const body = el("div", { class: "settings-body" });
    for (const [id, label, ico] of TABS) {
      nav.append(el("button", { class: id === tab ? "on" : "", onClick: () => views.open("settings", { tab: id }) },
        el("span", { class: "ico", text: ico }), label));
    }
    pane.append(el("div", { class: "settings" }, nav, body));
    const renderers = { general, personality, appearance, performance, privacy, voice, advanced };
    try {
      await renderers[tab](body);
    } catch (err) {
      body.append(el("div", { class: "note warn", text: `Dieser Bereich konnte nicht geladen werden: ${err.message || err}` }));
    }
  },
};

const h = (text, lead) => [el("h3", { text }), lead ? el("div", { class: "lead", text: lead }) : null];
const card = (title, ...children) => el("div", { class: "card-glass glass" }, title ? el("h4", { text: title }) : null, ...children);
const row = (label, control, small) => el("div", { class: "row" }, el("div", { class: "lbl" }, label, small ? el("small", { text: small }) : null), control);
const toggle = (on, onChange) => el("button", { class: "switch" + (on ? " on" : ""), role: "switch", "aria-checked": String(Boolean(on)),
  onClick: (e) => { const next = !e.currentTarget.classList.contains("on"); e.currentTarget.classList.toggle("on", next); e.currentTarget.setAttribute("aria-checked", String(next)); onChange(next); } });
const seg = (options, value, onChange) => {
  const box = el("div", { class: "seg" });
  const buttons = options.map(([v, label]) => el("button", { class: v === value ? "on" : "", text: label, onClick: () => {
    for (const b of buttons) b.classList.toggle("on", b === btn); onChange(v); }, }));
  let btn = null;
  buttons.forEach((b, i) => { b.onclick = () => { buttons.forEach((x) => x.classList.remove("on")); b.classList.add("on"); onChange(options[i][0]); }; });
  box.append(...buttons);
  return box;
};

/* ---------------------------------------------------------------- general */
async function general(body) {
  body.append(...h("Allgemein", "Wie ZEUS heißt, wie es dich anspricht und in welcher Sprache es antwortet."));
  const p = await api("/api/personality");
  const owner = (p && p.owner) || {};
  const prefs = (p && p.preferences) || {};
  const changes = { owner: {}, preferences: {} };
  const identity = (p && p.protected && p.protected.core_identity) || {};
  body.append(card("Identität",
    row("Name", el("input", { type: "text", value: identity.assistant_name || "ZEUS", disabled: true }), "Der Produktname ZEUS und der Erbauer sind geschützt und ändern sich nur über Erweitert › Systembesitz."),
    row("Erbauer", el("input", { type: "text", value: identity.creator || "Raphael", disabled: true })),
  ));
  body.append(card("Ansprache & Sprache",
    row("Wie ZEUS dich anspricht", el("input", { type: "text", value: owner.owner_name || "", placeholder: "z. B. Raphael", onInput: (e) => { changes.owner.owner_name = e.target.value; } }), "Dein Name, so wie ZEUS ihn verwenden soll."),
    row("Anrede", seg([["du", "du"], ["Sie", "Sie"]], owner.address || prefs.address || "du", (v) => { changes.owner.address = v; changes.preferences.address = v; })),
    row("Bevorzugte Sprache", seg([["auto", "wie du schreibst"], ["de", "Deutsch"], ["en", "Englisch"]], owner.language || prefs.language || "auto", (v) => { changes.owner.language = v; changes.preferences.language = v; })),
    row("Sprachverhalten", seg([["follow", "folgt dir"], ["fixed", "bleibt fest"]], owner.default_language_behaviour || "follow", (v) => { changes.owner.default_language_behaviour = v; })),
  ));
  body.append(saveBar(() => api("/api/personality/save", { changes, reason: "Allgemein bearbeitet" }), "Allgemein"));
  const defaults = await api("/api/defaults");
  if (defaults && defaults.defaults) {
    const d = defaults.defaults;
    body.append(card("Ablageorte",
      ...Object.entries(d).map(([key, value]) => row(key.replace(/_/g, " "), el("input", { type: "text", value: String(value || ""), onChange: async (e) => {
        await api("/api/defaults/set", { key, value: e.target.value }); } })))));
  }
}

/* ------------------------------------------------------------ personality */
async function personality(body) {
  body.append(...h("Persönlichkeit", "Wie ZEUS spricht und sich verhält – unabhängig davon, welche Intelligenz intern gerade rechnet."));
  const p = await api("/api/personality");
  if (!p || p.ok === false) { body.append(el("div", { class: "note warn", text: p?.error || "Persönlichkeit nicht verfügbar." })); return; }
  const changes = { preferences: { ...(p.preferences || {}) }, response: { ...(p.response || {}) }, owner: { ...(p.owner || {}) }, rules: (p.rules || []).map((r) => ({ ...r })) };
  const preview = el("div", { class: "contract-preview", text: p.contract?.text || "" });
  const previewMeta = el("div", { class: "note", text: `≈ ${p.contract?.estimated_tokens || 0} Tokens pro Anfrage · Version ${p.revision || 0}` });
  const refreshPreview = async () => {
    const r = await api("/api/personality/preview", { changes });
    if (r && r.contract) { preview.textContent = r.contract.text; previewMeta.textContent = `≈ ${r.contract.estimated_tokens} Tokens pro Anfrage · Version ${p.revision || 0} (Vorschau)`; }
  };

  // communication style
  const style = card("Kommunikationsstil");
  for (const [key, label, lo, hi] of DIALS) {
    const value = Number(changes.preferences[key] ?? 50);
    const val = el("span", { class: "val", text: String(value) });
    const range = el("input", { type: "range", min: 0, max: 100, value, onInput: (e) => { changes.preferences[key] = Number(e.target.value); val.textContent = e.target.value; },
                               onChange: refreshPreview });
    style.append(el("div", { class: "slider-row" }, el("span", { text: label }), el("div", {}, range, el("div", { class: "ends" }, el("span", { text: lo }), el("span", { text: hi }))), val));
  }
  body.append(style);

  // response preferences
  const resp = card("Antworten");
  for (const [key, label, options] of RESPONSE) {
    resp.append(row(label, seg(options, changes.response[key] || options[0][0], (v) => { changes.response[key] = v; refreshPreview(); })));
  }
  body.append(resp);

  // custom rules
  const rules = card("Eigene Regeln", el("div", { class: "note", text: "Dauerhafte Regeln in deinen Worten. Sie gelten bei jeder Antwort, egal welche Intelligenz rechnet." }));
  const list = el("div", { class: "rules" });
  const renderRules = () => {
    clear(list);
    changes.rules.forEach((r, i) => {
      const item = el("div", { class: "rule" + (r.enabled === false ? " off" : "") });
      const txt = el("input", { class: "txt", value: r.text, onChange: (e) => { r.text = e.target.value; refreshPreview(); } });
      const tools = el("div", { class: "tools" },
        el("button", { title: r.enabled === false ? "Aktivieren" : "Deaktivieren", text: r.enabled === false ? "○" : "●", onClick: () => { r.enabled = r.enabled === false; renderRules(); refreshPreview(); } }),
        el("button", { title: "Nach oben", text: "↑", disabled: i === 0, onClick: () => { changes.rules.splice(i - 1, 0, changes.rules.splice(i, 1)[0]); renderRules(); refreshPreview(); } }),
        el("button", { title: "Nach unten", text: "↓", disabled: i === changes.rules.length - 1, onClick: () => { changes.rules.splice(i + 1, 0, changes.rules.splice(i, 1)[0]); renderRules(); refreshPreview(); } }),
        el("button", { title: "Löschen", text: "✕", onClick: () => { changes.rules.splice(i, 1); renderRules(); refreshPreview(); } }));
      item.append(el("span", { class: "ico", text: "›" }), txt, tools);
      list.append(item);
    });
    if (!changes.rules.length) list.append(el("div", { class: "note", text: "Noch keine eigenen Regeln." }));
  };
  renderRules();
  const add = el("input", { placeholder: "Neue Regel, z. B. „Bei Programmierung kurz und lösungsorientiert.“" });
  const addBtn = el("button", { class: "btn", text: "Hinzufügen", onClick: () => {
    const text = add.value.trim(); if (!text) return; changes.rules.push({ text, enabled: true }); add.value = ""; renderRules(); refreshPreview(); } });
  add.addEventListener("keydown", (e) => { if (e.key === "Enter") addBtn.click(); });
  const search = el("input", { placeholder: "Regeln durchsuchen …", onInput: (e) => {
    const q = e.target.value.toLowerCase(); for (const item of list.children) item.hidden = q && !(item.querySelector("input")?.value || "").toLowerCase().includes(q); } });
  rules.append(el("div", { class: "rule-add" }, search), list, el("div", { class: "rule-add" }, add, addBtn));
  body.append(rules);

  // the protected core, shown as what it is
  body.append(card("Geschützter Kern", el("div", { class: "note", text: "Name, Erbauer und der Charakter von ZEUS sind geschützt: keine Antwort, kein Dokument und keine Webseite kann sie verändern." }),
    kv("Name", p.protected?.core_identity?.assistant_name), kv("Produkt", p.protected?.core_identity?.product_name), kv("Erbauer", p.protected?.core_identity?.creator),
    kv("Charakter", (p.core?.character || []).join(", "))));

  // preview + versions
  body.append(card("So liest ZEUS deine Persönlichkeit", previewMeta, preview));
  const versions = el("div", { class: "versions" });
  for (const v of (p.versions || []).slice().reverse()) {
    versions.append(el("div", { class: "v-row" }, el("span", { class: "when", text: String(v.at || "").replace("T", " ").slice(0, 16) }),
      el("span", { class: "what", text: v.kind === "rollback" ? "Wiederhergestellt" : (v.reason || "Änderung") }),
      v.kind !== "rollback" ? el("button", { class: "chip", text: "Zurück zu vorher", onClick: async () => {
        if (!confirm("Persönlichkeit auf den Stand vor dieser Änderung zurücksetzen?")) return;
        const r = await api("/api/personality/restore", { audit_id: v.audit_id });
        if (r && r.ok) views.open("settings", { tab: "personality" }, { force: true }); } }) : null));
  }
  if (!(p.versions || []).length) versions.append(el("div", { class: "note", text: "Noch keine Änderungen." }));
  body.append(card("Versionen", versions));
  body.append(saveBar(() => api("/api/personality/save", { changes, reason: "Persönlichkeit bearbeitet" }), "personality"));
}

function saveBar(save, tab) {
  const msg = el("span", { class: "msg", text: "" });
  const bar = el("div", { class: "save-bar glass-strong" }, msg, el("button", { class: "btn primary", text: "Speichern", onClick: async (e) => {
    e.currentTarget.disabled = true;
    const r = await save();
    e.currentTarget.disabled = false;
    if (r && r.ok) { msg.textContent = "Gespeichert."; setTimeout(() => views.open("settings", { tab }, { force: true }), 400); }
    else msg.textContent = (r && r.error) || "Konnte nicht speichern.";
  } }));
  return bar;
}

/* ------------------------------------------------------------- appearance */
async function appearance(body) {
  body.append(...h("Erscheinungsbild", "Dunkel, hell oder wie das System. Glas und Bewegung nach deinem Geschmack."));
  const ui = state.ui;
  const applyAppearance = (v) => { setPref("appearance", v); window.zeus?.applyAppearance?.(v); };
  body.append(card("Darstellung",
    row("Erscheinung", seg([["system", "System"], ["dark", "Dunkel"], ["light", "Hell"]], ui.appearance || "system", applyAppearance)),
    row("Glasintensität", seg([["low", "gering"], ["normal", "normal"], ["high", "stark"]], ui.glass || "normal", (v) => { setPref("glass", v); window.zeus?.applyAppearance?.(); })),
    row("Animationen", seg([["OFF", "aus"], ["LOW", "wenig"], ["NORMAL", "normal"], ["HIGH", "viel"]], ui.animIntensity || "NORMAL", (v) => { setPref("animIntensity", v); window.zeus?.applyAppearance?.(); })),
    row("Bewegung reduzieren", toggle(Boolean(ui.reducedMotion), (on) => { setPref("reducedMotion", on); document.body.classList.toggle("reduced-motion", on); }), "Für Barrierefreiheit: keine Übergänge, ruhiger Orb."),
  ));
}

/* ------------------------------------------------------------ performance */
async function performance(body) {
  body.append(...h("Leistung & Kosten", "ZEUS entscheidet selbst, wie viel Intelligenz eine Aufgabe braucht. Geld gibt es nur aus, wenn du es erlaubst."));
  const status = await api("/api/gateway/status");
  const spend = (status && status.spend) || {};
  const emergency = (status && status.emergency) || {};
  const owner = await api("/api/owner");
  const spending = (owner && owner.documents && owner.documents.spending) || {};
  body.append(card("Diesen Monat",
    row("Ausgaben", el("span", { text: `€${Number(spend.month || 0).toFixed(2)} von €${Number(spend.monthly_hard_cap || 0).toFixed(2)}` }), "Hartes Monatslimit; darüber hinaus wird nichts ausgegeben."),
    row("Ausgaben im Chat anzeigen", toggle(state.ui.showSpend !== false, (on) => { setPref("showSpend", on); }), "Aus: unter dem Eingabefeld erscheint keine Zahl mehr – und sie ändert sich auch nicht."),
    row("Standard-Leistung", seg(LEVELS.map(([id, label]) => [id, label]), status?.mode || "AUTO", async (v) => { await api("/api/gateway/mode", { mode: v }); })),
  ));
  const changes = {};
  body.append(card("Bezahlte Intelligenz",
    row("Bezahlte Intelligenz erlauben", toggle(Boolean(spending.paid_api), (on) => { changes.paid_api = on; }), "Ohne diese Freigabe bleibt ZEUS vollständig kostenlos."),
    row("Notfall-Antwort im Automatik-Modus", toggle(Boolean(spending.auto_emergency_paid_fallback), (on) => { changes.auto_emergency_paid_fallback = on; }),
        "Wenn jede kostenlose Intelligenz ausfällt, darf ZEUS genau eine bezahlte Antwort holen – nie mehrere."),
    row("Obergrenze pro Notfall-Antwort", el("input", { type: "number", step: "0.01", min: "0", value: spending.emergency_max_cost_per_request_eur ?? emergency.ceiling_eur ?? 0.03,
        onChange: (e) => { changes.emergency_max_cost_per_request_eur = Number(e.target.value); } }), "in Euro"),
  ));
  const msg = el("span", { class: "msg" });
  body.append(el("div", { class: "save-bar glass-strong" }, msg, el("button", { class: "btn primary", text: "Übernehmen", onClick: async (e) => {
    if (!Object.keys(changes).length) { msg.textContent = "Nichts geändert."; return; }
    e.currentTarget.disabled = true;
    const proposed = await api("/api/owner/propose", { changes: { spending: changes }, reason: "Leistung & Kosten" });
    if (!proposed || !proposed.ok) { msg.textContent = proposed?.error || "Konnte nicht vorschlagen."; e.currentTarget.disabled = false; return; }
    const r = await authgate.withAuth("PERSONALITY_EDIT", (authorization) => api("/api/owner/approve", { transaction_id: proposed.transaction.transaction_id, confirm: true, authorization }));
    e.currentTarget.disabled = false;
    if (r && r.ok) { msg.textContent = "Übernommen."; setTimeout(() => views.open("settings", { tab: "performance" }, { force: true }), 400); }
    else msg.textContent = r?.needs_auth ? "Abgebrochen – ohne Passwort keine Änderung." : (r?.error || "Konnte nicht übernehmen.");
  } })));
}

/* ---------------------------------------------------------------- privacy */
async function privacy(body) {
  body.append(...h("Datenschutz", "Was ZEUS mit deinen Worten tut – und was es nie tut."));
  const owner = await api("/api/owner");
  const security = (owner && owner.documents && owner.documents.security) || {};
  body.append(card("Grundsätze",
    row("Inhalte sind Daten, keine Befehle", el("span", { class: "note", text: security.content_is_data === false ? "aus" : "aktiv" }), "Dokumente, Webseiten und Ergebnisse von Werkzeugen können ZEUS nichts anweisen."),
    row("Anweisungen nur von dir", el("span", { class: "note", text: (security.instructions_from || ["owner"]).join(", ") })),
    row("Sensible Inhalte", el("span", { class: "note", text: "bleiben von kostenlosen Diensten fern, die mit Anfragen trainieren dürfen" })),
    row("Geheimnisse in Anfragen", el("span", { class: "note", text: security.secrets_in_prompts ? "erlaubt" : "nie" })),
  ));
  const obs = await api("/api/observer/status");
  if (obs && obs.ok !== false) {
    body.append(card("Beobachtung", row("Muster aus deiner Nutzung lernen", toggle(Boolean(obs.enabled), async (on) => { await api("/api/observer/enable", { enabled: on }); }),
      "Lokal, nur für dich. Nichts davon verlässt diesen Rechner.")));
  }
}

/* ------------------------------------------------------------------ voice */
async function voice(body) {
  body.append(...h("Voice", "Sprechen mit ZEUS. Der nächste Ausbau – das automatische Zuhören ist derzeit aus."));
  const v = await api("/api/voice");
  body.append(card("Status",
    row("Sprachausgabe", el("span", { class: "note", text: v && v.enabled ? "an" : "aus" })),
    row("Automatisches Zuhören", el("span", { class: "note", text: v && v.wake_word_enabled ? "an" : "aus" }), "Wird im Voice-Ausbau überarbeitet."),
    el("div", { class: "wk-actions" }, el("button", { class: "btn", text: "Voice Studio öffnen", onClick: () => views.open("voice") }))));
}

/* --------------------------------------------------------------- advanced */
async function advanced(body) {
  body.append(...h("Erweitert", "Die technische Wahrheit: welche Intelligenz gerechnet hat, was sie gekostet hat, wie es dem System geht. Nur hier stehen Anbieter und Modelle."));
  const status = await api("/api/gateway/status");
  if (status && status.ok !== false) {
    const diag = card("Diagnose · Intelligenz");
    diag.append(kv("Modus", status.mode), kv("Ausgaben Monat", `€${Number(status.spend?.month || 0).toFixed(4)}`), kv("Notfall-Antwort", status.emergency?.enabled ? `an · Obergrenze €${status.emergency.ceiling_eur}` : "aus"));
    for (const [role, r] of Object.entries(status.roles || {})) {
      if (!r.configured && !r.enabled) continue;
      diag.append(kv(role, `${r.provider} / ${r.model} · ${r.configured ? "konfiguriert" : "nicht konfiguriert"} · ${r.health}`, "mono"));
    }
    const routes = el("div", { class: "note" });
    for (const r of status.zero_cost_routes || []) {
      routes.append(el("div", { text: `${r.provider_id}/${r.model_id} · ${r.eligible ? "einsatzbereit" : r.reason} · ${r.verified_zero_cost ? `verifiziert ${r.verified_at}` : "Kosten unverifiziert"}` }));
    }
    diag.append(el("h4", { text: "Kostenlose Routen" }), routes);
    const recent = el("div", { class: "note" });
    for (const r of (status.recent || []).slice(-8).reverse()) {
      recent.append(el("div", { text: `${new Date(r.at * 1000).toLocaleTimeString()} · ${r.role} · ${r.provider}/${r.model} · €${Number(r.actual_eur || 0).toFixed(4)} · ${Number(r.latency_seconds || 0).toFixed(1)}s` }));
    }
    diag.append(el("h4", { text: "Letzte Generationen" }), recent);
    body.append(diag);
  }
  const host = el("div");
  body.append(card("Systembesitz, Anbieter, Schlüssel, Freigaben", host));
  const owner = await import("./owner.js");
  await owner.view.mount(host, {});
  const diagHost = el("div");
  body.append(card("Systemdiagnose", diagHost));
  const diagnostics = await import("./diagnostics.js");
  await diagnostics.view.mount(diagHost, {});
  const relHost = el("div");
  body.append(card("Versionen", relHost));
  const release = await import("./release.js");
  await release.view.mount(relHost, {});
}
