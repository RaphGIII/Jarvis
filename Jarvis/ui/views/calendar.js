/* Kalender: the owner's local-first calendar.

   A month grid with real persisted events (/api/calendar/*), a day agenda,
   and an editor in the inspector.  Voice/chat entries ("Trag morgen um 14
   Uhr Lernen ein") land here through the same store; a `calendar`
   notification refreshes the view live. */

import { $, el, clear, kv, section, badge, button } from "../core/dom.js";
import { api } from "../core/api.js";
import * as bus from "../core/bus.js";
import * as views from "../core/views.js";

const WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"];
const MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
                "August", "September", "Oktober", "November", "Dezember"];

let active = false;
let shown = null;      // first day of the displayed month
let selected = null;   // selected day (Date)
let events = [];       // events of the displayed range
let paneRef = null;

export const view = {
  id: "calendar",
  title: "Kalender",
  async mount(pane, params) {
    active = true;
    paneRef = pane;
    const now = new Date();
    shown = shown || new Date(now.getFullYear(), now.getMonth(), 1);
    selected = selected || now;
    if (!view._bused) {
      view._bused = true;
      bus.on("notification", (p) => {
        if (active && (p?.kind === "calendar" || p?.kind === "reminder")) render();
      });
    }
    await render();
  },
  unmount() { active = false; paneRef = null; },
};

function pad(n) { return String(n).padStart(2, "0"); }
function isoDay(d) { return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`; }
function sameDay(a, b) { return a && b && isoDay(a) === isoDay(b); }

async function fetchRange(from, to) {
  const out = await api("/api/calendar/list", { start: from.toISOString(), end: to.toISOString() });
  events = out.events || [];
}

async function render() {
  if (!paneRef) return;
  const pane = paneRef;
  const gridStart = new Date(shown);
  gridStart.setDate(1 - ((shown.getDay() + 6) % 7)); // back to Monday
  const gridEnd = new Date(gridStart);
  gridEnd.setDate(gridStart.getDate() + 42);
  await fetchRange(gridStart, gridEnd);
  if (!active) return;
  clear(pane);

  const title = el("h3", { class: "cal-title", text: `${MONTHS[shown.getMonth()]} ${shown.getFullYear()}` });
  const nav = (delta, label) => el("button", { class: "chip", text: label, onClick: () => {
    shown = new Date(shown.getFullYear(), shown.getMonth() + delta, 1); render(); } });
  const today = el("button", { class: "chip", text: "Heute", onClick: () => {
    const n = new Date(); shown = new Date(n.getFullYear(), n.getMonth(), 1); selected = n; render(); } });
  const search = el("input", { placeholder: "Suchen…", style: { maxWidth: "180px" } });
  search.onchange = async () => {
    if (!search.value.trim()) return render();
    const out = await api("/api/calendar/list", { query: search.value.trim() });
    showSearchResults(out.events || [], search.value.trim());
  };
  const add = button("+ Termin", () => editEvent(null), "primary");
  const exportBtn = el("button", { class: "chip", text: "⭳ .ics", title: "Als .ics exportieren", onClick: async () => {
    const out = await api("/api/calendar/export", {});
    alertLine(out.ok ? `Exportiert: ${out.path}` : `Export fehlgeschlagen: ${out.error || ""}`);
  } });
  pane.append(el("div", { class: "toolbar" }, nav(-1, "‹"), title, nav(1, "›"), today, search, add, exportBtn));
  const note = el("div", { class: "empty cal-note", style: { padding: "2px 0" } });
  pane.append(note);

  const grid = el("div", { class: "cal-grid" });
  for (const wd of WEEKDAYS) grid.append(el("div", { class: "cal-head", text: wd }));
  const now = new Date();
  for (let i = 0; i < 42; i += 1) {
    const day = new Date(gridStart);
    day.setDate(gridStart.getDate() + i);
    const inMonth = day.getMonth() === shown.getMonth();
    const dayEvents = events.filter((e) => e.start.slice(0, 10) === isoDay(day));
    const cell = el("div", {
      class: "cal-cell" + (inMonth ? "" : " dim") + (sameDay(day, now) ? " today" : "") + (sameDay(day, selected) ? " sel" : ""),
      onClick: () => { selected = new Date(day); render(); },
      onDblClick: () => { selected = new Date(day); editEvent(null); },
    }, el("span", { class: "cal-day", text: String(day.getDate()) }));
    for (const e of dayEvents.slice(0, 3)) {
      cell.append(el("div", { class: "cal-chip", text: `${e.start.slice(11, 16)} ${e.title}`.slice(0, 26),
                              onClick: (ev) => { ev.stopPropagation(); inspectEvent(e); } }));
    }
    if (dayEvents.length > 3) cell.append(el("div", { class: "cal-chip more", text: `+${dayEvents.length - 3} weitere` }));
    grid.append(cell);
  }
  pane.append(grid);

  // the selected day's agenda under the grid
  const dayEvents = events.filter((e) => e.start.slice(0, 10) === isoDay(selected));
  const agenda = el("div", { class: "cal-agenda" },
    el("h4", { text: selected.toLocaleDateString("de-DE", { weekday: "long", day: "numeric", month: "long" }) }));
  if (!dayEvents.length) agenda.append(el("div", { class: "empty", text: "Keine Termine. Doppelklick auf einen Tag legt einen an." }));
  for (const e of dayEvents) {
    agenda.append(el("div", { class: "cal-row", onClick: () => inspectEvent(e) },
      el("b", { text: `${e.start.slice(11, 16)}–${e.end.slice(11, 16)}` }),
      el("span", { text: e.title }),
      e.location ? el("span", { class: "empty", style: { padding: 0 }, text: e.location }) : null,
      e.reminder_minutes != null ? badge(`⏰ ${e.reminder_minutes}m`, "dim") : null));
  }
  pane.append(agenda);

  function alertLine(text) { note.textContent = text; }
}

function showSearchResults(rows, query) {
  const pane = paneRef;
  if (!pane) return;
  clear(pane);
  pane.append(el("div", { class: "toolbar" },
    el("button", { class: "chip", text: "‹ Kalender", onClick: () => render() }),
    el("h3", { class: "cal-title", text: `Suche: „${query}“ (${rows.length})` })));
  const wrap = el("div", { class: "cal-agenda" });
  if (!rows.length) wrap.append(el("div", { class: "empty", text: "Nichts gefunden." }));
  for (const e of rows) {
    wrap.append(el("div", { class: "cal-row", onClick: () => inspectEvent(e) },
      el("b", { text: e.start.slice(0, 16).replace("T", " ") }), el("span", { text: e.title })));
  }
  pane.append(wrap);
}

function inspectEvent(e) {
  views.inspect(e.title,
    el("div", { class: "meta" }, badge("TERMIN", "blue"), " ", badge(e.source || "owner", "dim")),
    section("Zeit", kv("beginn", e.start.slice(0, 16).replace("T", " ")), kv("ende", e.end.slice(0, 16).replace("T", " ")),
      e.location ? kv("ort", e.location) : null,
      e.reminder_minutes != null ? kv("erinnerung", `${e.reminder_minutes} Minuten vorher`) : null),
    e.notes ? section("Notizen", el("div", { text: e.notes })) : null,
    el("div", { class: "toolbar" },
      button("Bearbeiten", () => editEvent(e), "primary"),
      button("Löschen", async (ev) => {
        // two clicks instead of a native confirm(): modal dialogs freeze the
        // embedded window and every automated driver alike
        const b = ev.currentTarget;
        if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = "Wirklich löschen?"; return; }
        await api("/api/calendar/delete", { id: e.id });
        views.closeInspector();
        render();
      })));
}

function editEvent(existing) {
  const day = existing ? existing.start.slice(0, 10) : isoDay(selected || new Date());
  const f = {
    title: el("input", { value: existing?.title || "", placeholder: "Titel" }),
    date: el("input", { type: "date", value: day }),
    start: el("input", { type: "time", value: existing ? existing.start.slice(11, 16) : "14:00" }),
    end: el("input", { type: "time", value: existing ? existing.end.slice(11, 16) : "15:00" }),
    location: el("input", { value: existing?.location || "", placeholder: "Ort (optional)" }),
    notes: el("textarea", { value: existing?.notes || "", placeholder: "Notizen (optional)", rows: 3 }),
    reminder: el("input", { type: "number", value: existing?.reminder_minutes ?? "", placeholder: "Erinnerung (Min. vorher)", min: 0 }),
  };
  const row = (label, node) => el("div", { class: "kv" }, el("span", { class: "k", text: label }), el("span", { class: "v" }, node));
  views.inspect(existing ? "Termin bearbeiten" : "Neuer Termin",
    row("titel", f.title), row("datum", f.date), row("von", f.start), row("bis", f.end),
    row("ort", f.location), row("notizen", f.notes), row("erinnerung", f.reminder),
    el("div", { class: "toolbar" }, button("Speichern", async () => {
      const tz = -new Date().getTimezoneOffset();
      const off = `${tz >= 0 ? "+" : "-"}${pad(Math.floor(Math.abs(tz) / 60))}:${pad(Math.abs(tz) % 60)}`;
      const startIso = `${f.date.value}T${f.start.value}:00${off}`;
      const endIso = `${f.date.value}T${f.end.value}:00${off}`;
      const payload = { title: f.title.value.trim(), start: startIso, end: endIso,
                        location: f.location.value.trim(), notes: f.notes.value.trim(),
                        reminder_minutes: f.reminder.value === "" ? "" : Number(f.reminder.value) };
      if (!payload.title) { f.title.focus(); return; }
      if (existing) await api("/api/calendar/update", { id: existing.id, changes: payload });
      else await api("/api/calendar/create", payload);
      views.closeInspector();
      render();
    }, "primary")));
}
