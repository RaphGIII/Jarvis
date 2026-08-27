/* Owner Settings: the five owner-core documents (identity, personality,
   policy, spending, security), read-only until a change is proposed; a
   proposal shows its diff and needs explicit confirmation; every change is
   audited and can be rolled back. Ordinary SelfDev cannot edit these. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";

const DOCS = ["identity", "personality", "policy", "spending", "security"];

export const view = {
  id: "owner",
  title: "Owner",
  async mount(pane, params) {
    const data = await api("/api/owner");
    if (data.ok === false) { pane.append(el("div", { class: "empty", text: data.error || "owner core unavailable" })); return; }
    pane.append(el("div", { class: "card" }, el("div", { class: "title" }, badge("OWNER CORE", "amber"), " Protected configuration"),
      el("div", { class: "meta", text: "Changes here never happen from a chat sentence or a self-development mission: propose → diff → confirm → snapshot → apply → audit." })));
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
    pane.append(section("Protected paths", el("div", { class: "kv" }, el("span", { class: "v mono", text: (data.protected_paths || []).join("\n") }))));
  },
};

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
      button("Confirm & apply", async () => { const r = await api("/api/owner/approve", { transaction_id: t.transaction_id, confirm: true }); if (r.error) alert(r.error); reload(); }, "primary"),
      button("Reject", async () => { await api("/api/owner/reject", { transaction_id: t.transaction_id }); reload(); }, "ghost danger")));
}
