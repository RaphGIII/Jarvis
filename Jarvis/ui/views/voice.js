/* Voice Studio: wake word, microphone/speaker/voice settings, live input
   meter, and the wake-word training wizard ("Say Zeus 15 times", then 10
   non-Zeus phrases; train; measured recall / rejection on the owner's own
   recordings). Recording happens here, in the product, never in PowerShell.

   What the wake-word card shows is evidence, not decoration: the model kind
   the listener runs (OWNER / SYNTHETIC / NONE), the one effective threshold
   both the listener and the test below use, where that number came from,
   the last test score, recall and rejection from the last evaluation of the
   owner's recordings (labelled in-sample or held-out), and whether the
   listener process reported the same model fingerprint and threshold. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api, postBytes } from "../core/api.js";
import { state } from "../core/state.js";
import * as mic from "../voice/mic.js";
import * as playback from "../voice/playback.js";

const pct = (v) => (v === null || v === undefined ? "?" : `${Math.round(v * 100)}%`);
const num = (v, d = 2) => (v === null || v === undefined ? "?" : Number(v).toFixed(d));

export const view = {
  id: "voice",
  title: "Voice Studio",
  async mount(pane) {
    const [settings, wake] = await Promise.all([api("/api/voice"), api("/api/voice/wake")]);
    const health = state.health || {};
    const s = settings.settings || settings;

    const evidence = el("div");
    const renderWake = (w) => {
      clear(evidence);
      const ev = w.evaluation;
      const hold = w.owner_holdout;
      const kind = w.model_kind || (w.model_trained ? "SYNTHETIC" : "NONE");
      evidence.append(
        el("div", { class: "title" }, "Wake word ", badge((w.wake_word || "zeus").toUpperCase(), "blue"), " ",
          badge(`MODEL: ${kind}`, kind === "OWNER" ? "ok" : kind === "SYNTHETIC" ? "warn" : "bad"), " ",
          w.listener ? badge(w.listener_match ? "LISTENER = TEST" : "LISTENER DIFFERS", w.listener_match ? "ok" : "bad") : badge("LISTENER SILENT", "warn")),
        el("div", { class: "meta" },
          el("span", { text: `listener ${health.voice ? "ready" : "not ready"}` }),
          el("span", { text: `recogniser ${health.recogniser ? "ready" : "not ready"}` }),
          el("span", { text: `effective threshold ${num(w.effective_threshold)} (${w.threshold_source || "?"})` }),
          el("span", { text: w.last_score ? `last test score ${num(w.last_score.score, 3)} ${w.last_score.detected ? "DETECTED" : "not detected"} at ${w.last_score.at.slice(11, 19)}` : "no test yet" })),
        el("div", { class: "meta" },
          ev ? el("span", { text: `positive recall ${pct(ev.positive_recall)} (${ev.positives_detected ?? "?"}/${ev.counts?.positive_usable ?? "?"}) · negative rejection ${pct(ev.negative_rejection)} (${ev.false_activations ?? "?"} false of ${ev.counts?.negative ?? "?"}) · ${ev.in_sample === false ? "held-out" : "in-sample"}${ev.stale ? " · STALE (model changed since)" : ""} · evaluated ${ev.at ? ev.at.replace("T", " ") : "?"}` })
             : el("span", { text: "no evaluation yet — press Calibrate" }),
          hold ? el("span", { text: `held-out check: recall ${pct(hold.at_effective_threshold?.recall)} · rejection ${pct(hold.at_effective_threshold?.rejection)} on ${hold.counts?.positive_usable ?? "?"}+/${hold.counts?.negative ?? "?"}− recordings the model never saw` }) : null,
          ev ? el("span", { text: `scores: positives min ${num(ev.positive_scores?.min, 3)} / median ${num(ev.positive_scores?.median, 3)} / max ${num(ev.positive_scores?.max, 3)}; negatives max ${num(ev.negative_scores?.max, 3)} · recommended threshold ${ev.recommended_threshold ?? "?"}${ev.separates === false ? " (samples do NOT separate)" : ""}` }) : null,
          ev && ev.silent_positives?.length ? el("span", { text: `silent recordings skipped: ${ev.silent_positives.join(", ")}` }) : null,
          ev && !ev.hard_negatives_evaluated ? el("span", { text: "hard negatives (Jesus, Servus): none recorded — record them under data/wake/hard_negative to measure" }) : null,
          w.listener ? el("span", { text: `listener pid ${w.listener.pid}: threshold ${num(w.listener.threshold)}, model ${w.listener.fingerprint || "?"}${w.listener_match ? "" : ` ≠ ${w.model_fingerprint || "?"}`}` }) : null));
    };
    renderWake(wake);
    // wake feedback: the owner grades real detections, ZEUS learns thresholds
    const wakeFb = el("div", { class: "toolbar" });
    const fbNote = el("span", { class: "empty", style: { padding: 0 } });
    const sendWake = async (rating, category, label) => {
      const last = (await api("/api/voice/wake")).last_score;
      const r = await api("/api/feedback", { kind: "wake", rating, category,
        text: last ? `score ${Number(last.score).toFixed(3)} at ${last.at}` : "no last detection", session: last ? last.at : "" });
      fbNote.textContent = r.ok === false ? (r.error || "nicht gespeichert") : label + (r.insight ? " · Muster erkannt → Vorschlag erstellt" : "");
    };
    wakeFb.append(
      button("✓ Letzte Erkennung war Zeus", () => sendWake("up", "WAKE_CORRECT", "gemerkt: korrekt")),
      button("✗ Das war nicht Zeus", () => sendWake("down", "WAKE_WRONG", "gemerkt: Fehlauslösung"), "ghost danger"),
      button("Ich sagte Zeus — nicht gehört", () => sendWake("down", "WAKE_MISSED", "gemerkt: verpasst"), "ghost"),
      fbNote);
    // outside `evidence`: renderWake clears that node on every refresh
    pane.append(el("div", { class: "card" }, evidence, wakeFb));

    const form = el("div");
    const field = (label, key, type = "text", hint = "") => {
      const input = el("input", type === "number" ? { type, value: s[key] ?? "", step: "0.01", min: "0", max: "1" } : { type, value: s[key] ?? "" });
      form.append(el("div", { class: "field" }, el("label", { text: label }), input, hint ? el("div", { class: "empty", text: hint }) : null));
      return () => [key, type === "number" ? (input.value === "" ? null : Number(input.value)) : input.value];
    };
    const readers = [
      field("microphone (device name or index)", "microphone"), field("speaker / output", "output"), field("voice (piper model)", "voice"),
      field("wake threshold 0–1 (sensitivity)", "wake_sensitivity", "number", "the score „Zeus“ must reach on two consecutive frames; lower = more sensitive; empty = the model's recommendation"),
      field("voice volume 0–1", "volume", "number", "playback of ZEUS's speech only — never touches the microphone or wake detection"),
      field("language", "language"),
    ];
    const saved = el("div", { class: "empty" });
    pane.append(section("Settings", form, el("div", { class: "toolbar" }, button("Save", async () => {
      const payload = Object.fromEntries(readers.map((r) => r()).filter(([, v]) => v !== "" && !Number.isNaN(v)));
      const r = await api("/api/voice", payload);
      if (r.ok === false) { saved.textContent = r.error || "not saved"; return; }
      if (typeof payload.volume === "number") playback.setVolume(payload.volume);
      const w = await api("/api/voice/wake");
      renderWake(w);
      saved.textContent = `saved · effective threshold ${num(w.effective_threshold)} (${w.threshold_source})`;
    }, "primary"), saved)));

    const meter = el("div", { class: "bar", style: { width: "260px" } }, el("i", { style: { width: "0%" } }));
    pane.append(section("Live input", meter, el("div", { class: "toolbar" }, button("Test microphone (2 s)", async () => {
      await mic.recordClip(2, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
      meter.firstChild.style.width = "0%";
    }))));

    pane.append(section("Train the wake word", wizard(wake, renderWake)));
    pane.append(section("Pronunciation (spoken form only — the written text never changes)", await pronunciation()));
  },
};

async function pronunciation() {
  const box = el("div");
  const data = await api("/api/voice/pronunciation");
  const preview = el("input", { placeholder: "Preview a sentence: ZEUS verwendet die GPU über GitHub.", style: { minWidth: "360px" } });
  const out = el("div", { class: "empty" });
  const surface = el("input", { placeholder: "word as written (e.g. Spotify)" });
  const spoken = el("input", { placeholder: "how to say it (e.g. Spottifai)" });
  const status = el("div", { class: "empty" });
  const list = el("div");
  const render = (d) => {
    clear(list);
    const own = d.owner_entries || [];
    list.append(el("div", { class: "meta" }, el("span", { text: `provider ${d.provider} · ${(d.entries || []).length} entries (${own.length} yours) · ${d.path}` })));
    for (const e of own) list.append(el("div", { class: "kv" }, el("span", { class: "k", text: e.surface }), el("span", { class: "v" }, `${e.spoken_as[d.provider] || e.spoken_as.generic} (${e.language})`,
      button("Remove", async () => { await api("/api/voice/pronunciation/remove", { surface: e.surface, language: e.language }); render(await api("/api/voice/pronunciation")); }, "ghost danger"))));
    for (const r of (d.recent || []).slice(-5)) list.append(el("div", { class: "kv" }, el("span", { class: "k", text: "recently spoken" }), el("span", { class: "v", text: `„${r.displayed}“ → „${r.spoken}“` })));
  };
  render(data);
  box.append(el("div", { class: "toolbar" }, preview, button("Preview", async () => { const r = await api("/api/voice/pronunciation", { text: preview.value }); out.textContent = r.preview ? `spoken as: ${r.preview.spoken}` : ""; })), out,
    el("div", { class: "toolbar" }, surface, spoken, button("Learn & test", async () => {
      const r = await api("/api/voice/pronunciation/set", { surface: surface.value, spoken: spoken.value });
      status.textContent = r.ok ? `learned ${r.entry.surface} → ${r.entry.spoken_as[Object.keys(r.entry.spoken_as)[0]]}; synthesis ${r.test.tried ? (r.test.ok ? `ok (${r.test.seconds}s)` : "failed: " + (r.test.error || "")) : "not tried"}` : (r.error || "failed");
      if (r.test && r.test.url) new Audio(r.test.url + (location.search.includes("token") ? "" : "")).play().catch(() => {});
      render(await api("/api/voice/pronunciation"));
    }, "primary"), status), list);
  return box;
}

function wizard(wake, renderWake) {
  const box = el("div");
  const status = el("div", { class: "empty", style: { padding: "6px 0" }, text: `${wake.positive || 0} owner recordings of „Zeus“, ${wake.negative || 0} other phrases${wake.hard_negative ? `, ${wake.hard_negative} hard negatives` : ""} on disk.` });
  const meter = el("div", { class: "bar", style: { width: "260px", margin: "6px 0" } }, el("i", { style: { width: "0%" } }));
  const prompt = el("div", { class: "card", style: { fontSize: "18px", textAlign: "center" }, text: "Ready." });
  const NEGATIVES = ["Wie spät ist es?", "Mach das Licht aus.", "Ich gehe jetzt einkaufen.", "Zeig mir die Nachrichten.", "Das Wetter ist heute schön.",
                     "Spiel etwas Musik.", "Ruf meine Mutter an.", "Was steht heute an?", "Erinnere mich an den Termin.", "Guten Morgen."];
  const HARD = ["Jesus", "Servus", "Zeit", "Deus", "Zeug"];
  let running = false;
  async function record(kind, count, prompts) {
    if (running) return;
    running = true;
    for (let i = 0; i < count; i++) {
      prompt.textContent = kind === "positive" ? `Sag „Zeus“ (${i + 1}/${count})` : `Sag: „${prompts[i % prompts.length]}“ (${i + 1}/${count})`;
      await new Promise((r) => setTimeout(r, 700));
      // 2.0 s for one word: the recorder needs a moment to start, and a
      // 1.6 s clip lost the word entirely once (owner_016 was silent).
      const wav = await mic.recordClip(kind === "positive" ? 2.0 : 2.6, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
      meter.firstChild.style.width = "0%";
      const r = await postBytes(`/api/voice/wake/record?kind=${kind}`, wav, "audio/wav");
      status.textContent = r.ok ? `${r.positive} owner recordings of „Zeus“, ${r.negative} other phrases on disk.` : (r.error || "recording failed");
    }
    prompt.textContent = "Done.";
    running = false;
  }
  box.append(status, prompt, meter, el("div", { class: "toolbar" },
    button("1 · Say „Zeus“ 15 times", () => record("positive", 15, []), "primary"),
    button("2 · Say 10 other phrases", () => record("negative", 10, NEGATIVES)),
    button("2b · Hard negatives (Jesus, Servus…)", () => record("hard_negative", 5, HARD)),
    button("3 · Train", async () => {
      prompt.textContent = "Training… (several minutes)";
      const r = await api("/api/voice/wake/train", {});
      if (r.ok && r.status) renderWake(r.status);
      const ev = r.status?.evaluation;
      prompt.textContent = r.ok ? `Trained. Owner recall ${pct(ev?.positive_recall)}, rejection ${pct(ev?.negative_rejection)} at threshold ${num(r.status?.effective_threshold)} — the listener reloads it by itself.` : (r.error || "training failed");
    }),
    button("4 · Test detection (say Zeus)", async () => {
      prompt.textContent = "Say „Zeus“ now…";
      const wav = await mic.recordClip(2.0, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
      const r = await postBytes("/api/voice/wake/test", wav, "audio/wav");
      prompt.textContent = r.ok ? `Score ${Number(r.score).toFixed(3)} — ${r.detected ? "DETECTED" : "not detected"} (threshold ${num(r.threshold)}, ${r.threshold_source})${r.silent ? " — the recording was silent" : ""}` : (r.error || "test failed");
      if (r.ok) renderWake(await api("/api/voice/wake"));
    }),
    button("Calibrate", async () => {
      prompt.textContent = "Evaluating every owner recording through the detector…";
      const r = await api("/api/voice/wake/evaluate", {});
      if (!r.ok) { prompt.textContent = r.error || "evaluation failed"; return; }
      renderWake(r.status);
      const rep = r.report;
      prompt.textContent = `Recall ${pct(rep.at_effective_threshold?.recall)} / rejection ${pct(rep.at_effective_threshold?.rejection)} at ${num(rep.effective_threshold)}; recommended ${rep.recommended_threshold} — set it under Settings if you want it.`;
    })));
  return box;
}
