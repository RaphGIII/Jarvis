/* The one way to talk to the core. GET when there is no body, POST otherwise;
   the token rides in a header; errors become {ok:false, error} instead of
   exceptions, so a view never has to guess whether a failure was transport or
   an answer. */

export function token() {
  return window.JARVIS_TOKEN || "";
}

export async function api(path, body) {
  try {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json", "X-Jarvis-Token": token() },
      body: body === undefined ? undefined : JSON.stringify(body || {}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok && data.ok === undefined) data.ok = false;
    if (!response.ok && !data.error) data.error = `HTTP ${response.status}`;
    return data;
  } catch (err) {
    return { ok: false, error: err && err.message ? err.message : String(err), transport: true };
  }
}

export async function postBytes(path, bytes, contentType = "application/octet-stream") {
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": contentType, "X-Jarvis-Token": token() },
      body: bytes,
    });
    return await response.json().catch(() => ({ ok: false }));
  } catch (err) {
    return { ok: false, error: err && err.message ? err.message : String(err), transport: true };
  }
}

export function audioUrl(url) {
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token())}`;
}
