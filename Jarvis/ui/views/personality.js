/* Persönlichkeit: the control centre for who ZEUS is and how he behaves.

   One view that makes the whole behaviour pipeline visible in the order it
   is actually applied — IDENTITY → CORE → HONESTY → PREFERENCES → ADAPTIVE
   RULES → TASK RULES → EFFECTIVE BEHAVIOUR — plus a configuration inspector
   that says where every layer lives and who may change it.  Owner rules
   always outrank inferred ones; protected layers need the owner password.
   Nothing here is invented: every block shown is the block a model
   receives, fetched from the server. */

import { el, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as views from "../core/views.js";
import { securityPanel, adaptationPanel } from "./owner.js";

const STAGES = [
  ["identity", "IDENTITÄT", "wer Zeus ist (Name, Besitzer, Zuhause)", "owner-Dokument · Passwort-geschützt"],
  ["core", "KERN", "Charakter, Gesprächsregeln, Epistemik — geschützt", "PERSONALITY_EDIT nötig"],
  ["honesty", "EHRLICHKEIT", "keine erfundenen Fakten, Unsicherheit benennen", "fest verdrahtet"],
  ["preferences", "PRÄFERENZEN", "deine Regler: Knappheit, Humor, Wärme, Du/Sie", "Owner-Vorschlag + Bestätigung"],
  ["adaptive", "ADAPTIVE REGELN", "gelernt aus 👍/👎 und Korrekturen, begrenzt + abklingend", "jederzeit einsehbar/löschbar"],
  ["task", "AUFGABENREGELN", "kontextabhängig: technisch ≠ Smalltalk ≠ Bestätigung", "automatisch pro Anfrage"],
  ["effective", "EFFEKTIV", "der Prompt, wie ihn ein Modell wirklich erhält", "unten wörtlich einsehbar"],
];

export const view = {
  id: "personality",
  title: "Persönlichkeit",
  async mount(pane, params) {
    const [p, adapt, auth] = await Promise.all([
      api("/api/owner/personality"), api("/api/adaptation"), api("/api/auth/status")]);
    if (p.ok === false) { pane.append(el("div", { class: "empty", text: p.error || "unavailable" })); return; }
    const blocks = p.blocks || [];
    const blockByName = new Map(blocks.map(([name, text]) => [name, text]));
    const rules = adapt.rules || [];

    // the pipeline, in the order it is applied
    const flow = el("div", { class: "pline" });
    for (const [key, label, what, who] of STAGES) {
      const live = key === "adaptive" ? `${rules.filter((r) => r.enabled !== false).length} aktiv`
        : blockByName.has(key) ? "aktiv" : ["honesty", "task", "effective"].includes(key) ? "aktiv" : blocks.length ? "aktiv" : "—";
      const card = el("div", { class: "pstage", onClick: () => {
        document.getElementById("pstage-" + key)?.scrollIntoView({ behavior: "smooth", block: "start" });
      } },
        el("div", { class: "pstage-name", text: label }),
        el("div", { class: "pstage-what", text: what }),
        el("div", { class: "pstage-who" }, badge(live, live === "—" ? "dim" : "ok"), el("span", { text: " " + who })));
      flow.append(card, el("span", { class: "parrow", text: "→" }));
    }
    flow.lastChild.remove();
    pane.append(el("div", { class: "card" },
      el("div", { class: "title" }, badge("VERHALTENS-PIPELINE", "amber"), ` Version ${p.version ?? "?"} · Reihenfolge ist verbindlich`),
      el("div", { class: "meta", text: "Owner-Regeln schlagen gelernte Regeln. Geschützte Ebenen brauchen dein Passwort. Kein Modell kann diese Reihenfolge ändern." }),
      flow));

    // identity + core: read-only here, edited (password-gated) in Owner
    const core = p.core || {};
    pane.append(anchor("identity"), section("Identität & Kern (geschützt)",
      el("div", { class: "core-lock" },
        kv("charakter", (core.character || []).join(" · ") || "—"),
        kv("gespräch", (core.conversation || []).join("\n") || "—"),
        kv("epistemik", core.epistemics || "—")),
      el("div", { class: "toolbar" },
        button("Im Owner-Bereich bearbeiten (Passwort)", () => views.open("owner"), "primary"),
        badge(auth.configured ? (auth.locked ? "GESPERRT" : "FREIGEGEBEN") : "KEIN PASSWORT", auth.configured ? "ok" : "bad"))));

    pane.append(anchor("preferences"), section("Präferenzen (deine Regler)",
      el("div", { class: "kv" }, el("span", { class: "v", text: Object.entries(p.preferences || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "Standardwerte" })),
      el("div", { class: "toolbar" }, button("Regler anpassen", () => views.open("owner")))));

    pane.append(anchor("adaptive"), section("Adaptive Regeln (aus deinem Feedback)", await adaptationPanel(() => views.open("personality"))));

    pane.append(anchor("task"), section("Aufgabenregeln (Kontext)", el("div", { class: "kv" }, el("span", { class: "v", text:
      "Jede Anfrage wird klassifiziert (technische Erklärung · Bestätigung · Smalltalk · Gespräch) und nur die Regeln dieses Kontexts wirken. Eine Regel „technisch ausführlicher“ macht Bestätigungen nicht länger." }))));

    pane.append(anchor("effective"), section("Effektives Verhalten (wörtlich)",
      el("details", { open: false }, el("summary", { text: `Der Prompt in Modell-Reihenfolge (${blocks.length} Blöcke)` }),
        ...blocks.map(([name, text], i) => el("div", { class: "prompt-block" },
          el("div", { class: "meta" }, badge(`${i + 1} · ${name}`, name === "core" ? "amber" : "blue")), el("pre", { class: "code", text }))))));

    // configuration inspector: where each layer lives, who writes it
    const row = (name, where, writer, open) => el("div", { class: "kv" },
      el("span", { class: "k", text: name }),
      el("span", { class: "v" }, `${where} · schreibt: ${writer} `, open ? button("öffnen", open, "ghost") : null));
    pane.append(section("Konfigurations-Inspektor",
      row("Identität / Policy / Spending / Security", "data/jarvis/owner (Dokumente, auditiert)", "nur Owner-Vorschlag + Bestätigung", () => views.open("owner")),
      row("Persönlichkeits-Kern + Regler", "data/jarvis/owner (Versionen + Historie)", "Owner, Passwort-Scope PERSONALITY_EDIT", () => views.open("owner")),
      row("Adaptive Regeln", "data/jarvis/owner/adaptive.json (begrenzt, abklingend)", "dein Feedback · deterministischer Code", null),
      row("Sprach-Lexikon & Aussprache", "data/jarvis/owner (heard→meant, kontext-begrenzt)", "deine Korrekturen", () => views.open("voice")),
      row("Sicherheits-Gate", "data/jarvis/owner/auth.json (scrypt + DPAPI, nie Klartext)", "nur du, manuell getippt", null)));

    pane.append(section("Sicherheit", await securityPanel(() => views.open("personality"))));
  },
};

const anchor = (key) => el("div", { id: "pstage-" + key });
