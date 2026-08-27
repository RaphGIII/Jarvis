/* DOM helpers. Every view builds nodes with these, so there is one place that
   knows how an element is made and one way to clear a container. */

export const $ = (id) => document.getElementById(id);

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null) continue;
    if (key === "class" || key === "className") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key in node && key !== "list") { try { node[key] = value; } catch { node.setAttribute(key, value); } }
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function clockOf(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? String(iso).slice(11, 19) : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function dateOf(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? String(iso).slice(0, 10) : d.toLocaleDateString();
}

export function ago(iso) {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  if (isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function seconds(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "";
  if (v < 60) return `${v.toFixed(v < 10 ? 1 : 0)}s`;
  if (v < 3600) return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
  return `${Math.floor(v / 3600)}h ${Math.floor((v % 3600) / 60)}m`;
}

/* A key/value line, the unit of every inspector. */
export function kv(key, value, cls = "") {
  if (value === undefined || value === null || value === "") return null;
  return el("div", { class: "kv " + cls }, el("span", { class: "k", text: key }), el("span", { class: "v", text: String(value) }));
}

export function section(title, ...children) {
  return el("section", { class: "sec" }, el("h4", { text: title }), ...children);
}

export function badge(text, tone = "") {
  return el("span", { class: "badge " + tone, text });
}

export function button(text, onClick, cls = "ghost") {
  return el("button", { class: cls, onClick, text });
}

/* Debounce, for inputs that ask the server on every keystroke. */
export function debounce(fn, ms) {
  let timer = null;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}
