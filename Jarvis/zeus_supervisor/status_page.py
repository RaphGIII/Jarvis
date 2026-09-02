"""What the browser sees while ZEUS itself is not answering.

The UI's address is the core's port.  When the core is down -- booting, being
rolled back, held after a boot loop -- a browser pointed there would get a
connection error and nothing else, which tells the owner nothing.  So the
supervisor holds the port itself in those moments and serves one page: what it
is doing, what failed, and what would fix it.  The page refreshes itself and
disappears the moment the real interface is back.

It is released *before* the core is launched, so the two never compete for the
address.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

# The boot experience.  This page IS the first second of the product: it is
# what the window shows from T0 until the core answers.  Three rules learned
# live:
#   * NEVER location.reload() -- a reload fired into the port-handover gap
#     (supervisor released :8420, core not yet bound) lands on the browser's
#     ERR_CONNECTION_REFUSED page, which has no script and never recovers.
#     The DOM updates in place from the polled JSON instead.
#   * Navigation to the real UI must keep location.search: the token rides in
#     the query, and '/' without it cannot open the event stream.
#   * lang="de" + notranslate, or Edge offers to translate the boot screen.
PAGE = """<!doctype html><html lang="de" translate="no"><head><meta charset="utf-8">
<meta name="google" content="notranslate"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZEUS</title>
<style>
html,body{margin:0;height:100%;overflow:hidden}
body{background:radial-gradient(120% 100% at 50% 38%,#0b1424 0%,#060a14 55%,#02040a 100%);
     color:#c8d6e5;font:15px/1.5 system-ui,'Segoe UI',sans-serif;
     display:flex;align-items:center;justify-content:center}
#stars{position:fixed;inset:0;z-index:0}
.scene{position:relative;z-index:1;text-align:center;max-width:680px;padding:0 32px}
.rig{position:relative;width:220px;height:220px;margin:0 auto 26px}
.orb{position:absolute;left:50%;top:50%;width:74px;height:74px;margin:-37px 0 0 -37px;border-radius:50%;
     background:radial-gradient(circle at 42% 38%,#bfe6ff 0%,#3fa9ff 34%,#0b2c52 75%,#05080f 100%);
     box-shadow:0 0 34px rgba(80,170,255,.55),0 0 110px rgba(60,140,255,.25);
     animation:pulse 2.8s ease-in-out infinite}
.held .orb{background:radial-gradient(circle at 42% 38%,#ffd2c2 0%,#ff7a4d 36%,#5a180b 78%,#0f0503 100%);
     box-shadow:0 0 34px rgba(255,120,70,.5),0 0 110px rgba(255,90,60,.22)}
@keyframes pulse{0%,100%{transform:scale(.96);filter:brightness(.92)}50%{transform:scale(1);filter:brightness(1.08)}}
.ring{position:absolute;left:50%;top:50%;border:1px solid rgba(110,180,255,.22);border-radius:50%}
.r1{width:130px;height:130px;margin:-65px 0 0 -65px;animation:spin 7s linear infinite}
.r2{width:176px;height:176px;margin:-88px 0 0 -88px;animation:spin 12s linear infinite reverse}
.r3{width:216px;height:216px;margin:-108px 0 0 -108px;animation:spin 19s linear infinite}
.ring i{position:absolute;top:-3px;left:50%;width:6px;height:6px;margin-left:-3px;border-radius:50%;
     background:#9fdcff;box-shadow:0 0 10px rgba(140,210,255,.9)}
@keyframes spin{to{transform:rotate(360deg)}}
h1{margin:0 0 2px;font-size:24px;letter-spacing:.5em;color:#cfe8ff;font-weight:600;text-indent:.5em}
.phase{color:#6f8aa5;font-size:11px;text-transform:uppercase;letter-spacing:.34em;margin:6px 0 14px;min-height:16px}
.detail{white-space:pre-wrap;color:#9fb4cc;font-size:13px;min-height:20px}
.systems{list-style:none;display:flex;gap:26px;justify-content:center;padding:0;margin:26px 0 0;
     font-size:10px;letter-spacing:.24em;color:#33465e;text-transform:uppercase}
.systems li{transition:color .8s ease,text-shadow .8s ease}
.systems li.on{color:#8fd3ff;text-shadow:0 0 12px rgba(120,200,255,.5)}
.remedy{margin:16px auto 0;max-width:520px;padding:10px 14px;background:rgba(255,150,70,.07);
     border-left:3px solid #ffb454;color:#ffd9a0;white-space:pre-wrap;font-size:13px;text-align:left}
.log{margin:18px auto 0;max-width:560px;font:11px/1.5 ui-monospace,Consolas,monospace;color:#4b617c;
     white-space:pre-wrap;max-height:150px;overflow:auto;text-align:left}
</style></head><body class="__STATE__">
<canvas id="stars"></canvas>
<div class="scene">
  <div class="rig">
    <div class="ring r3"><i></i></div><div class="ring r2"><i></i></div><div class="ring r1"><i></i></div>
    <div class="orb"></div>
  </div>
  <h1>ZEUS</h1>
  <div class="phase" id="phase">__PHASE__</div>
  <div class="detail" id="detail">__DETAIL__</div>
  <ul class="systems" id="systems">
    <li data-i="0">Kern</li><li data-i="1">Lokale Intelligenz</li>
    <li data-i="2">Stimme</li><li data-i="3">Universum</li>
  </ul>
  __REMEDY__
  <div class="log" id="log">__LOG__</div>
</div>
<script>
(function(){
  // a static, cheap starfield: painted once, never animated
  var c=document.getElementById('stars'),x=c.getContext('2d');
  function paint(){
    c.width=innerWidth;c.height=innerHeight;x.clearRect(0,0,c.width,c.height);
    for(var i=0;i<170;i++){var s=Math.random()*1.4+.3;
      x.globalAlpha=Math.random()*.55+.1;x.fillStyle=Math.random()<.85?'#cfe4ff':'#ffe9c9';
      x.fillRect(Math.random()*c.width,Math.random()*c.height,s,s);}
    x.globalAlpha=1;
  }
  paint();addEventListener('resize',paint);

  var lit={preflight:1,ollama:2,starting:3,launching:3,waiting:3,ready:4};
  function illuminate(phase){
    var n=0,p=String(phase||'').toLowerCase();
    for(var k in lit){if(p.indexOf(k)>=0){n=lit[k];break;}}
    var items=document.querySelectorAll('#systems li');
    for(var i=0;i<items.length;i++)items[i].classList.toggle('on',i<n);
  }
  illuminate(document.getElementById('phase').textContent);

  var failures=0;
  function tick(){
    fetch('/api/health',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
      if(d&&d.supervisor){
        // still booting: update THIS page in place -- never reload into the gap
        document.getElementById('phase').textContent=d.phase||'starting';
        document.getElementById('detail').textContent=d.detail||'';
        document.body.className=/held|error/i.test(String(d.phase||''))?'held':'';
        if(d.log)document.getElementById('log').textContent=d.log.slice(-12).join('\\n');
        illuminate(d.phase);
        failures=0;setTimeout(tick,700);return;
      }
      // the core answered: hand over, KEEPING the token in the query
      location.replace('/'+location.search);
    }).catch(function(){failures++;setTimeout(tick,failures<40?500:2000);});
  }
  setTimeout(tick,700);
})();
</script></body></html>"""


class StatusPage:
    def __init__(self, host: str, port: int, snapshot: Callable[[], dict[str, Any]], *,
                 on_stop: Callable[[], Any] | None = None, token: str = "") -> None:
        self.host = host
        self.port = port
        self.snapshot = snapshot
        #: A held supervisor must remain stoppable from outside: ``POST
        #: /api/quit`` (with the shared token) on this page ends it.
        self.on_stop = on_stop
        self.token = token
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> bool:
        if self._server is not None:
            return True
        page = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                state = page.snapshot()
                if self.path.startswith("/api/") or self.path.startswith("/events"):
                    body = json.dumps({"supervisor": True, "ready": False, **state}, default=str).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                else:
                    phase = str(state.get("phase", "starting"))
                    remedy = state.get("remedy", "")
                    body = (PAGE
                            .replace("__STATE__", "held" if phase.lower() in {"error", "held"} else "")
                            .replace("__PHASE__", html.escape(phase))
                            .replace("__DETAIL__", html.escape(str(state.get("detail", ""))))
                            .replace("__REMEDY__", f'<div class="remedy">{html.escape(str(remedy))}</div>' if remedy else "")
                            .replace("__LOG__", html.escape("\n".join(str(l) for l in state.get("log", [])[-12:])))
                            ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in {"/api/quit", "/api/shutdown", "/api/stop"} and page.on_stop is not None:
                    supplied = self.headers.get("X-Jarvis-Token", "")
                    if not supplied and "token=" in self.path:
                        supplied = self.path.split("token=", 1)[1].split("&", 1)[0]
                    if page.token and supplied != page.token:
                        body = b'{"error": "unauthorised"}'
                        self.send_response(401)
                    else:
                        try:
                            page.on_stop()
                        except Exception:  # noqa: BLE001
                            pass
                        body = b'{"ok": true, "supervisor": true, "stopping": true}'
                        self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.do_GET()

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self._server = None
            return False
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="zeus-status-page")
        self._thread.start()
        return True

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
