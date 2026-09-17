/* The navigation on a narrow window: a drawer over the content, never beside it.

   Wide (> 860 px)   the sidebar is a column; the toggle collapses it and the choice is remembered.
   Narrow (<= 860 px) the sidebar is a drawer: it opens over the content with a veil behind it,
                     the content keeps its full width and becomes inert, focus moves into the
                     drawer and is kept there (Tab cycles), Escape / the veil / choosing a
                     destination closes it, and focus returns to the toggle.

   The width decision is one media query shared with zeus.css. */

export const NARROW_QUERY = "(max-width: 860px)";

/* What a toggle press means at this width: {drawer: bool, open?: bool, collapsed?: bool}. */
export function toggleOutcome({ narrow, open, collapsed }) {
  return narrow ? { drawer: true, open: !open } : { drawer: false, collapsed: !collapsed };
}

/* The element Tab should move to, keeping focus inside `items` (a list of focusable elements); null when no wrap is needed. */
export function trapTarget(items, active, backwards) {
  if (!items.length) return null;
  const first = items[0];
  const last = items[items.length - 1];
  const inside = items.includes(active);
  if (!inside) return backwards ? last : first;
  if (backwards && active === first) return last;
  if (!backwards && active === last) return first;
  return null;
}

export function createDrawer({ app, sidebar, scrim, toggle, main, matchMedia = window.matchMedia.bind(window), onCollapse = () => {} }) {
  const query = matchMedia(NARROW_QUERY);
  let returnFocus = null;

  const focusables = () => [...sidebar.querySelectorAll("button, [href], input, [tabindex]:not([tabindex='-1'])")].filter((node) => !node.disabled);

  function isOpen() {
    return app.classList.contains("sidebar-open");
  }

  function setOpen(open, { restoreFocus = true } = {}) {
    const narrow = query.matches;
    const next = Boolean(open) && narrow;
    const was = isOpen();
    app.classList.toggle("sidebar-open", next);
    scrim.hidden = !next;
    if (main) main.inert = next;
    toggle.setAttribute("aria-expanded", String(narrow ? next : !app.classList.contains("sidebar-collapsed")));
    sidebar.setAttribute("aria-modal", next ? "true" : "false");
    if (next) sidebar.setAttribute("role", "dialog");
    else sidebar.removeAttribute("role");
    if (next && !was) {
      returnFocus = document.activeElement;
      const current = sidebar.querySelector(".sb-item.on") || focusables()[0];
      requestAnimationFrame(() => current?.focus({ preventScroll: true }));
    }
    if (!next && was && restoreFocus) {
      const target = returnFocus && document.contains(returnFocus) && returnFocus !== document.body ? returnFocus : toggle;
      if (sidebar.contains(document.activeElement) || document.activeElement === document.body) target.focus({ preventScroll: true });
      returnFocus = null;
    }
  }

  function press() {
    const outcome = toggleOutcome({ narrow: query.matches, open: isOpen(), collapsed: app.classList.contains("sidebar-collapsed") });
    if (outcome.drawer) {
      if (!outcome.open) returnFocus = toggle;
      setOpen(outcome.open);
      return outcome;
    }
    app.classList.toggle("sidebar-collapsed", outcome.collapsed);
    toggle.setAttribute("aria-expanded", String(!outcome.collapsed));
    onCollapse(outcome.collapsed);
    return outcome;
  }

  function onKeydown(event) {
    if (!isOpen()) return false;
    if (event.key === "Escape") {
      event.preventDefault();
      returnFocus = toggle;
      setOpen(false);
      return true;
    }
    if (event.key === "Tab") {
      const target = trapTarget(focusables(), document.activeElement, event.shiftKey);
      if (target) { event.preventDefault(); target.focus(); }
      return true;
    }
    return false;
  }

  toggle.addEventListener("click", press);
  scrim.addEventListener("click", () => { returnFocus = toggle; setOpen(false); });
  const onChange = () => setOpen(false, { restoreFocus: false });
  query.addEventListener?.("change", onChange);
  setOpen(false, { restoreFocus: false });

  return { isOpen, setOpen, press, onKeydown, close: () => setOpen(false), query };
}
