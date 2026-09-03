/* Version / Rollback center: the running revision, the known-good pointer,
   deployment receipts, and the ZEUS.exe release pipeline (candidates,
   verification, promotion, the previous release kept for rollback). */

import { el, clear, kv, section, badge, button, ago } from "../core/dom.js";
import { api } from "../core/api.js";
import { withAuth } from "../core/authgate.js";
import * as views from "../core/views.js";

export const view = {
  id: "release",
  title: "Versions",
  async mount(pane) {
    const [sup, rel, health] = await Promise.all([api("/api/supervisor"), api("/api/release"), api("/api/health")]);
    const kg = sup.known_good || {};
    pane.append(el("div", { class: "card" },
      el("div", { class: "title" }, "Running ", el("code", { text: (health.revision || "").slice(0, 12) }), " ",
        kg.revision === health.revision ? badge("KNOWN GOOD", "ok") : badge("NOT YET KNOWN GOOD", "warn")),
      el("div", { class: "meta" }, el("span", { text: `known-good ${(kg.revision || "").slice(0, 12)} verified ${kg.verified_at || "?"}` }),
        el("span", { text: `previous ${(kg.previous || "").slice(0, 12)}` }), el("span", { text: sup.supervised ? "supervised" : "not supervised" }))));

    const receipts = (sup.deployments || []).slice().reverse();
    pane.append(section("Deployments (supervisor receipts)", el("div", { class: "timeline" }, ...(receipts.length ? receipts.slice(0, 25).map((r) =>
      el("div", { class: "tl " + (["healthy", "promoted", "handed_over"].includes(r.outcome) ? "ok" : ["rolled_back", "crashed", "held", "failed"].includes(r.outcome) ? "bad" : "work") },
        el("span", { class: "when", text: (r.at || "").slice(0, 19) }), el("span", { class: "text", text: `${r.kind} · ${r.outcome} · ${(r.revision || "").slice(0, 10)} ${r.reason ? "— " + r.reason : ""}` }),
        r.duration_seconds ? el("span", { class: "sub", text: `${r.duration_seconds}s` }) : null)) : [el("div", { class: "empty", text: "no receipts" })]))));

    const kgv = (rel.known_good || {}).version || {};
    pane.append(section("ZEUS.exe",
      el("div", { class: "card" }, el("div", { class: "title" }, "Known-good launcher ", rel.needs_rebuild ? badge("STALE", "warn") : badge("CURRENT", "ok")),
        el("div", { class: "meta" }, el("span", { text: `built from ${(kgv.revision || "").slice(0, 12)} · launcher ${kgv.launcher_fingerprint || "?"} · ${kgv.built_at || ""}` }),
          el("span", { text: rel.needs_rebuild_reason || "" }), el("span", { text: rel.previous?.exists ? "previous release kept" : "no previous release" })),
        el("div", { class: "toolbar" },
          button("Build & verify candidate", async () => { const r = await api("/api/release/build", { verify: true }); alert(r.ok ? "building in the background (about a minute); Activity shows the outcome" : r.error); }, "primary"),
          rel.previous?.exists ? button("Roll back to previous release", async () => { if (confirm("Rename the previous ZEUS.exe release back into place?")) { const r = await api("/api/release/rollback", { confirm: true }); alert(r.outcome || r.error); views.open("release"); } }, "ghost danger") : null)),
      ...(rel.candidates || []).slice().reverse().map((c) => el("div", { class: "card" },
        el("div", { class: "title" }, c.verified ? badge("VERIFIED", "ok") : badge("UNVERIFIED", "dim"), " ", c.id),
        el("div", { class: "meta", text: `${(c.version?.revision || "").slice(0, 12)} · launcher ${c.version?.launcher_fingerprint || "?"} · ${c.version?.built_at || ""}` }),
        el("div", { class: "toolbar" },
          c.verified ? button("Promote & relaunch", async () => {
            if (!confirm("Promote this candidate and relaunch ZEUS into it?")) return;
            // promotion is a SELFDEV_PROMOTE change: withAuth shows the password
            // modal on needs_auth and retries with the minted token (the earlier
            // button swallowed needs_auth and silently did nothing)
            const r = await withAuth("SELFDEV_PROMOTE", (authorization) =>
              api("/api/release/promote", { candidate: c.path, relaunch: true, authorization }));
            alert(r.needs_auth ? "Abgebrochen – ohne Passwort keine Freigabe."
                               : `${r.outcome || r.error || "?"}: ${r.reason || ""}`);
            views.open("release");
          }, "primary")
                     : button("Verify", async () => { const r = await api("/api/release/verify", { candidate: c.path }); alert(`${r.outcome}: ${r.reason}`); views.open("release"); })))),
    ));
    pane.append(section("Release history", el("div", { class: "timeline" }, ...((rel.history || []).slice().reverse().map((h) =>
      el("div", { class: "tl " + (["built", "verified", "promoted", "healthy"].includes(h.outcome) ? "ok" : ["failed", "rejected", "rolled_back", "refused"].includes(h.outcome) ? "bad" : "work") },
        el("span", { class: "when", text: (h.at || "").slice(0, 19) }), el("span", { class: "text", text: `${h.kind} · ${h.outcome} · ${h.reason || ""}`.slice(0, 200) })))))));
  },
};
