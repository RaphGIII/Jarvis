/* Schach Analyse: one button starts a separate process that watches the
   screen for a chessboard, recognises the position (the owner's trained
   detector, CPU only), infers the side to move, asks the local Stockfish for
   the five best moves and shows them in an always-on-top panel in the
   top-left corner. This view starts/stops it and mirrors its status file. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";

export const view = {
  id: "chess",
  title: "Schach Analyse",
  async mount(pane) {
    const head = el("div", { class: "card" });
    const detail = el("div");
    let timer = null;
    const render = async () => {
      const r = await api("/api/tools/chess");
      const s = r.status || {};
      clear(head); clear(detail);
      head.append(
        el("div", { class: "title" }, "Schach Analyse ", badge(r.running ? "RUNNING" : "STOPPED", r.running ? "ok" : "dim"), " ",
          s.board ? badge(`board ${s.board.size}px`, "blue") : badge("no board", "warn")),
        el("div", { class: "meta" }, el("span", { text: `Stockfish: ${r.stockfish || "not found"}` }), el("span", { text: `model: ${(r.model || "not found").split("\\").slice(-3).join("\\")}` }),
          s.pid ? el("span", { text: `pid ${s.pid}` }) : null, s.frames !== undefined ? el("span", { text: `${s.frames} frames · ${s.analyses || 0} analyses · ${s.frame_ms || "?"} ms/frame` }) : null),
        el("div", { class: "toolbar" },
          button(r.running ? "Stop" : "Start Schach Analyse", async () => { await api(r.running ? "/api/tools/chess/stop" : "/api/tools/chess/start", {}); setTimeout(render, 800); }, r.running ? "ghost danger" : "primary"),
          button("Refresh", render)));
      if (s.fen || s.last_error) {
        detail.append(section("Position",
          kv("FEN", s.fen || "—", "mono"), kv("side to move", s.side ? `${s.side === "w" ? "white" : "black"} — ${s.side_source}` : "—"),
          kv("you play", s.orientation || "—"), kv("detections", s.detections ?? "—"), kv("engine", s.engine_ms ? `${s.engine_ms} ms` : "—"), kv("error", s.last_error || "")));
        if ((s.lines || []).length) detail.append(section("Best moves", ...s.lines.map((l) => kv(`${l.rank}. ${l.san}`, `${l.score}  ·  depth ${l.depth}  ·  ${l.pv}`))));
      } else {
        detail.append(el("div", { class: "empty", text: r.running ? "Watching the screen for an 8×8 board…" : "Press Start; the panel appears in the top-left corner of the screen and follows the game while you keep using the PC." }));
      }
    };
    pane.append(head, detail);
    await render();
    timer = setInterval(render, 2000);
    view._stop = () => clearInterval(timer);
  },
  unmount() { view._stop?.(); },
};
