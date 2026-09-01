/* The Owner Security Gate, client side: a compact modal where the owner
   MANUALLY types the password. The value goes in one POST to /api/auth/unlock
   (or /setup) and nowhere else — never into chat, never into a prompt, never
   into localStorage. Tokens are scoped and short-lived; we cache them only in
   memory for their lifetime. No model output can drive this dialog: it is
   created here, filled by the keyboard, and read by trusted server code. */

import { el, clear } from "./dom.js";
import { api } from "./api.js";

const cache = new Map(); // scope -> { token, until }

export async function status() {
  return api("/api/auth/status");
}

/* Resolve to an authorization token for `scope`, or null if the owner cancelled. */
export function ensureAuth(scope, { title = "", reason = "" } = {}) {
  const hit = cache.get(scope);
  if (hit && hit.until > Date.now() + 2000) return Promise.resolve(hit.token);
  return new Promise((resolve) => openDialog(scope, { title, reason, resolve }));
}

/* Wrap an API response: if it says needs_auth, ask, then retry with the token. */
export async function withAuth(scope, call) {
  let out = await call("");
  if (out && out.needs_auth) {
    const token = await ensureAuth(out.needs_auth, { reason: out.error });
    if (!token) return out;
    out = await call(token);
  }
  return out;
}

function openDialog(scope, { title, reason, resolve }) {
  document.getElementById("authModal")?.remove();
  let setupMode = false;
  const heading = el("h3", { text: title || "Geschützte Änderung" });
  const sub = el("p", { class: "auth-sub", text: reason || `Freigabe für: ${scope}` });
  const msg = el("div", { class: "auth-msg" });
  const pw = el("input", { type: "password", placeholder: "Passwort", autocomplete: "new-password", spellcheck: false });
  const pw2 = el("input", { type: "password", placeholder: "Passwort wiederholen", autocomplete: "new-password", style: { display: "none" } });
  const ok = el("button", { class: "primary", text: "Freigeben" });
  const cancel = el("button", { class: "ghost", text: "Abbrechen" });
  const card = el("div", { class: "auth-card" }, heading, sub, msg, pw, pw2, el("div", { class: "auth-row" }, cancel, ok));
  const modal = el("div", { id: "authModal", class: "auth-modal" }, card);
  const close = (token) => { modal.remove(); resolve(token || null); };
  cancel.onclick = () => close(null);
  modal.onclick = (e) => { if (e.target === modal) close(null); };

  const submit = async () => {
    ok.disabled = true;
    try {
      if (setupMode) {
        if (pw.value !== pw2.value) { msg.textContent = "Die Passwörter stimmen nicht überein."; ok.disabled = false; return; }
        const made = await api("/api/auth/setup", { password: pw.value });
        if (!made.ok) { msg.textContent = made.error || "Nicht gespeichert."; ok.disabled = false; return; }
        setupMode = false; pw2.style.display = "none"; heading.textContent = title || "Geschützte Änderung";
        msg.textContent = "Passwort gesetzt. Jetzt freigeben:";
        ok.textContent = "Freigeben"; ok.disabled = false; pw.value = ""; pw.focus();
        return;
      }
      const out = await api("/api/auth/unlock", { scope, password: pw.value });
      pw.value = "";
      if (out.needs_setup) {
        setupMode = true; pw2.style.display = "";
        heading.textContent = "Owner-Passwort festlegen";
        msg.textContent = "Es ist noch kein Passwort gesetzt (mindestens 8 Zeichen).";
        ok.textContent = "Passwort setzen"; ok.disabled = false; pw.focus();
        return;
      }
      if (!out.ok) { msg.textContent = out.error || "Falsches Passwort."; ok.disabled = false; pw.focus(); return; }
      cache.set(scope, { token: out.authorization, until: Date.now() + (out.expires_in || 300) * 1000 - 5000 });
      close(out.authorization);
    } catch (err) {
      msg.textContent = String(err && err.message || err); ok.disabled = false;
    }
  };
  ok.onclick = submit;
  pw.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); if (e.key === "Escape") close(null); });
  pw2.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
  document.body.append(modal);
  pw.focus();
}

export function dropCache(scope = "") {
  if (scope) cache.delete(scope); else cache.clear();
}
