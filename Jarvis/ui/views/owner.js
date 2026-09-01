/* Owner Settings: the five owner-core documents (identity, personality,
   policy, spending, security), read-only until a change is proposed; a
   proposal shows its diff and needs explicit confirmation; every change is
   audited and can be rolled back. Ordinary SelfDev cannot edit these.

   Personality has its own panel: the protected core (who Zeus is) shown as
   what it is and locked; the owner's dials (how Zeus talks) adjustable and
   proposed like any owner change; the effective prompt in the order a model
   receives it; reset to defaults and the change history. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import * as authgate from "../core/authgate.js";

const DOCS = ["identity", "policy", "spending", "security"];
const DIALS = [
  ["conciseness", "Conciseness", "long-form", "very short"],
  ["formality", "Formality", "informal", "formal"],
  ["humour", "Humour", "none", "dry, often"],
  ["proactivity", "Proactivity", "only when asked", "points things out"],
  ["technical_depth", "Technical depth", "minimal", "detailed"],
  ["warmth", "Conversational warmth", "matter-of-fact", "warm"],
  ["initiative", "Initiative", "waits", "takes the next step"],
];

export const view = {
  id: "owner",
  title: "Owner",
  async mount(pane, params) {
    const data = await api("/api/owner");
    if (data.ok === false) { pane.append(el("div", { class: "empty", text: data.error || "owner core unavailable" })); return; }
    pane.append(el("div", { class: "card" }, el("div", { class: "title" }, badge("OWNER CORE", "amber"), " Protected configuration"),
      el("div", { class: "meta", text: "Changes here never happen from a chat sentence or a self-development mission: propose → diff → confirm → snapshot → apply → audit." })));
    pane.append(section("Personality", await personalityPanel(() => views.open("owner"))));
    const docs = data.documents || {};
    const grid = el("div", { class: "grid" });
    for (const name of DOCS) {
      const doc = docs[name] || {};
      const card = el("div", { class: "card click" }, el("div", { class: "title", text: name }),
        el("div", { class: "meta", text: Object.keys(doc).slice(0, 6).join(" · ") || "(empty)" }));
      card.onclick = () => editDocument(name, doc, () => views.open("owner"));
      grid.append(card);
    }
    pane.append(section("Documents", grid));
    const pending = data.pending || [];
    if (pending.length) {
      pane.append(section("Pending proposals", ...pending.map((t) => proposal(t, () => views.open("owner")))));
    }
    const history = data.history || [];
    pane.append(section("Audit", ...(history.length ? history.slice().reverse().slice(0, 20).map((h) => el("div", { class: "kv" },
      el("span", { class: "k", text: (h.at || h.applied_at || "").slice(0, 19) }),
      el("span", { class: "v" }, `${h.action || h.kind || "change"} · ${h.reason || ""} · ${(h.documents || []).join(", ")}`,
        h.audit_id ? button("Roll back", async () => { if (confirm(`Roll back ${h.audit_id}?`)) { await api("/api/owner/rollback", { audit_id: h.audit_id, confirm: true }); views.open("owner"); } }, "ghost danger") : null))) : [el("div", { class: "empty", text: "No owner changes recorded." })])));
    pane.append(section("Security", await securityPanel(() => views.open("owner"))));
    pane.append(section("Gelernt aus deinem Feedback", await adaptationPanel(() => views.open("owner"))));
    pane.append(section("Protected paths", el("div", { class: "kv" }, el("span", { class: "v mono", text: (data.protected_paths || []).join("\n") }))));
  },
};

/* ---- the Owner Security Center ------------------------------------- */
export async function securityPanel(reload) {
  const s = await api("/api/auth/status");
  const box = el("div");
  const state = s.configured ? (s.locked ? "LOCKED" : "UNLOCKED") : "NO PASSWORD SET";
  box.append(el("div", { class: "meta" },
    badge(state, s.configured ? (s.locked ? "ok" : "warn") : "bad"),
    el("span", { text: ` KDF ${s.kdf || "scrypt"} · Speicher ${s.storage || "file"} · Passwort wird nur manuell eingegeben, nie von Modellen gesehen` })));
  if ((s.sessions || []).length) {
    for (const t of s.sessions) box.append(kv("freigegeben", `${t.scope} · noch ${Math.round(t.expires_in)}s`));
  }
  box.append(el("div", { class: "kv" }, el("span", { class: "k", text: "geschützt (Passwort nötig)" }),
    el("span", { class: "v", text: "geschützte Persönlichkeit · Sicherheitsrichtlinie · SelfDev-Promotion · Software-Installation · endgültiges Löschen · Massen-Dateioperationen · Zugangsdaten · Autostart/System" })));
  box.append(el("div", { class: "toolbar" },
    button(s.configured ? "Passwort ändern" : "Passwort festlegen", async () => {
      const current = s.configured ? (prompt("Aktuelles Passwort:") || "") : "";
      const next = prompt("Neues Passwort (mind. 8 Zeichen):") || "";
      if (!next) return;
      const r = await api("/api/auth/setup", { password: next, current });
      alert(r.ok ? "Gespeichert." : (r.error || "Nicht gespeichert."));
      reload();
    }, "primary"),
    button("Alles sperren", async () => { await api("/api/auth/lock", {}); authgate.dropCache(); reload(); }, "ghost")));
  return box;
}

/* ---- learned adaptive rules: inspect, disable, delete ---------------- */
export async function adaptationPanel(reload) {
  const data = await api("/api/adaptation");
  const box = el("div");
  const rules = data.rules || [];
  if (!rules.length) box.append(el("div", { class: "empty", text: "Noch nichts gelernt. 👍/👎 unter Antworten und Korrekturen landen hier – begrenzt, abklingend, löschbar." }));
  for (const r of rules) {
    const conf = Math.round((r.effective_confidence ?? r.confidence ?? 0) * 100);
    box.append(el("div", { class: "kv" },
      el("span", { class: "k" }, badge(r.source === "OWNER_RULE" ? "REGEL" : "GELERNT", r.source === "OWNER_RULE" ? "amber" : "blue")),
      el("span", { class: "v" },
        `${r.text || r.domain} · ${JSON.stringify(r.scope)} · Gewicht ${Number(r.weight).toFixed(2)} · Konfidenz ${conf}%` + (r.enabled ? "" : " · AUS"),
        el("div", { class: "toolbar" },
          button(r.enabled ? "Deaktivieren" : "Aktivieren", async () => { await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "update", changes: { enabled: !r.enabled } }); reload(); }, "ghost"),
          button("Löschen", async () => { await api("/api/adaptation/rule", { rule_id: r.rule_id, action: "delete" }); reload(); }, "ghost danger")))));
  }
  const text = el("input", { placeholder: "Eigene Regel, z. B. „Bei medizinischen Erklärungen ausführlicher antworten.“", style: { minWidth: "340px" } });
  const kind = el("select", {}, ...[["technical_explanation", "bei technischen Erklärungen"], ["action_confirmation", "bei Bestätigungen"], ["conversation", "im Gespräch"], ["", "überall"]]
    .map(([v, t]) => el("option", { value: v, text: t })));
  box.append(el("div", { class: "toolbar" }, text, kind,
    button("Regel hinzufügen", async () => {
      if (!text.value.trim()) return;
      await api("/api/adaptation/rule", { action: "add", text: text.value.trim(), domain: "STYLE", scope: kind.value ? { kind: kind.value } : {} });
      reload();
    }, "primary")));
  return box;
}

/* The personality panel: dials → proposal; core locked; prompt preview; history. */
async function personalityPanel(reload) {
  const p = await api("/api/owner/personality");
  const box = el("div");
  if (p.ok === false) { box.append(el("div", { class: "empty", text: p.error || "unavailable" })); return box; }
  const prefs = { ...(p.preferences || {}) };
  const changed = {};
  const dials = el("div", { class: "dials" });
  for (const [key, label, low, high] of DIALS) {
    const value = el("span", { class: "dial-value", text: String(prefs[key] ?? 50) });
    const input = el("input", { type: "range", min: "0", max: "100", step: "5", value: String(prefs[key] ?? 50) });
    input.oninput = () => { value.textContent = input.value; changed[key] = Number(input.value); };
    dials.append(el("div", { class: "dial" }, el("div", { class: "dial-head" }, el("span", { class: "dial-label", text: label }), value),
      input, el("div", { class: "dial-ends" }, el("span", { text: low }), el("span", { text: high }))));
  }
  const spokenLength = el("select", {}, ...["short", "medium"].map((v) => el("option", { value: v, text: v === "short" ? "spoken answers: one or two sentences" : "spoken answers: a few sentences", selected: (prefs.spoken_answer_length || "short") === v })));
  spokenLength.onchange = () => { changed.spoken_answer_length = spokenLength.value; };
  const address = el("select", {}, ...["du", "Sie", "you"].map((v) => el("option", { value: v, text: `address: ${v}`, selected: (prefs.address || "du") === v })));
  address.onchange = () => { changed.address = address.value; };
  const reason = el("input", { placeholder: "Why? (recorded in the audit)" });
  const status = el("div", { class: "empty" });
  const propose = button("Propose these preferences", async () => {
    if (!Object.keys(changed).length) { status.textContent = "nothing changed"; return; }
    const r = await api("/api/owner/propose", { changes: { personality: { preferences: changed } }, reason: reason.value || "personality preferences" });
    status.textContent = r.ok === false ? (r.error || "refused") : `proposed ${r.transaction?.transaction_id} — confirm below`;
    if (r.ok !== false) reload();
  }, "primary");
  const reset = button("Reset to defaults", async () => {
    const r = await api("/api/owner/propose", { changes: { personality: { preferences: p.defaults?.preferences || {} } }, reason: "reset personality preferences to defaults" });
    status.textContent = r.ok === false ? (r.error || "refused") : `reset proposed ${r.transaction?.transaction_id} — confirm below`;
    if (r.ok !== false) reload();
  });
  const core = p.core || {};
  const coreBox = el("div", { class: "core-lock" },
    el("div", { class: "meta" }, badge("PROTECTED CORE", "amber"), el("span", { text: "who Zeus is — models, SelfDev, corrections and imported prompts can read it, never write it" })),
    kv("character", (core.character || []).join(" · ")),
    kv("conversation", (core.conversation || []).join("\n")),
    kv("behaviour", (core.behaviour || []).join("\n")),
    kv("emotional language", (core.emotional_language || []).join("\n")),
    kv("epistemics", core.epistemics || ""));
  const unlock = button("Unlock core (owner only)", () => {
    if (!confirm("Editing the core changes who Zeus is. Continue?")) return;
    if (!confirm("Second confirmation: the change is proposed as a protected transaction and audited. Proceed?")) return;
    editCore(core, reload);
  }, "ghost danger");
  const blocks = (p.blocks || []).map(([name, text], i) => el("div", { class: "prompt-block" }, el("div", { class: "meta" }, badge(`${i + 1} · ${name}`, name === "core" ? "amber" : "blue")), el("pre", { class: "code", text })));
  const history = (p.history || []).slice().reverse().slice(0, 10).map((h) => kv((h.at || h.applied_at || "").slice(0, 19), `${h.action || "change"} · ${h.reason || ""}` + (h.audit_id ? ` · ${h.audit_id}` : "")));
  box.append(
    el("div", { class: "meta" }, badge("EFFECTIVE", "ok"), el("span", { text: `version ${p.version} · order: identity → core → honesty → preferences → conversation context → task style` })),
    el("div", { class: "toolbar" }, spokenLength, address),
    dials,
    el("div", { class: "toolbar" }, reason, propose, reset, status),
    coreBox, el("div", { class: "toolbar" }, unlock),
    el("details", {}, el("summary", { text: "Effective prompt (as a model receives it)" }), ...blocks));
  if (history.length) box.append(el("details", {}, el("summary", { text: `History (${history.length})` }), ...history));
  return box;
}

function editCore(core, reload) {
  const fields = [];
  const form = el("div");
  for (const [key, value] of Object.entries(core)) {
    const input = el("textarea", { value: Array.isArray(value) ? value.join("\n") : String(value), rows: Array.isArray(value) ? Math.min(8, value.length + 1) : 2, style: { width: "100%" } });
    fields.push([key, value, input]);
    form.append(el("div", { class: "field" }, el("label", { text: key }), input));
  }
  const reason = el("input", { placeholder: "Why? (recorded in the audit)" });
  views.inspect("Owner · personality core (protected)", section("Core", form), el("div", { class: "field" }, el("label", { text: "reason" }), reason),
    el("div", { class: "toolbar" }, button("Propose core change", async () => {
      const next = {};
      for (const [key, value, input] of fields) {
        const v = Array.isArray(value) ? input.value.split("\n").map((l) => l.trim()).filter(Boolean) : input.value.trim();
        if (JSON.stringify(v) !== JSON.stringify(value)) next[key] = v;
      }
      if (!Object.keys(next).length) { alert("nothing changed"); return; }
      // the protected core: the security gate asks for the manually typed
      // password and mints a scoped PERSONALITY_EDIT token for this call
      const r = await authgate.withAuth("PERSONALITY_EDIT", (token) =>
        api("/api/owner/propose", { changes: { personality: { core: next } }, reason: reason.value || "personality core",
                                    unlock_core: true, authorization: token }));
      if (r.ok === false || r.error) { alert(r.error || "refused"); return; }
      reload();
    }, "primary")));
}

/* Edit = propose. Values are edited as JSON per key so no secret-shaped
   field is ever rendered raw: keys that look like secrets are masked. */
function editDocument(name, doc, reload) {
  const fields = [];
  const form = el("div");
  for (const [key, value] of Object.entries(doc)) {
    const secret = /secret|token|password|key$/i.test(key);
    const input = el(typeof value === "string" && value.length > 60 ? "textarea" : "input", { value: secret ? "••••••" : (typeof value === "string" ? value : JSON.stringify(value)), disabled: secret });
    fields.push([key, value, input, secret]);
    form.append(el("div", { class: "field" }, el("label", { text: key }), input));
  }
  const reason = el("input", { placeholder: "Why? (recorded in the audit)" });
  views.inspect(`Owner · ${name}`, section("Values", form), el("div", { class: "field" }, el("label", { text: "reason" }), reason),
    el("div", { class: "toolbar" }, button("Propose change", async () => {
      const changes = {};
      for (const [key, value, input, secret] of fields) {
        if (secret) continue;
        let next = input.value;
        if (typeof value !== "string") { try { next = JSON.parse(input.value); } catch { continue; } }
        if (JSON.stringify(next) !== JSON.stringify(value)) changes[key] = next;
      }
      if (!Object.keys(changes).length) { alert("nothing changed"); return; }
      const r = await api("/api/owner/propose", { changes: { [name]: changes }, reason: reason.value || "owner settings" });
      if (r.ok === false || r.error) { alert(r.error || "refused"); return; }
      reload();
    }, "primary")));
}

function proposal(t, reload) {
  const diff = el("div", { class: "diff" });
  const lines = typeof t.diff === "string" ? t.diff.split("\n") : JSON.stringify(t.diff || {}, null, 1).split("\n");
  for (const line of lines) diff.append(el("div", { class: line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "", text: line }));
  return el("div", { class: "card" },
    el("div", { class: "title" }, badge("PROPOSED", "amber"), ` ${t.transaction_id} · ${t.reason || ""}`),
    el("div", { class: "meta", text: `${(t.documents || []).join(", ")} · ${t.origin || ""} · ${t.proposed_at || ""}` }),
    diff,
    el("div", { class: "toolbar" },
      button("Confirm & apply", async () => {
        const r = await authgate.withAuth("PERSONALITY_EDIT", (token) =>
          api("/api/owner/approve", { transaction_id: t.transaction_id, confirm: true, authorization: token }));
        if (r.error) alert(r.error);
        reload();
      }, "primary"),
      button("Reject", async () => { await api("/api/owner/reject", { transaction_id: t.transaction_id }); reload(); }, "ghost danger")));
}
