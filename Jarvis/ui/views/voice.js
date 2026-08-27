/* Voice Studio: wake word, microphone/speaker/voice settings, live input
   meter, and the wake-word training wizard ("Say Zeus 15 times", then 10
   non-Zeus phrases; train; measured recall / false activations). Recording
   happens here, in the product, never in PowerShell. */

import { el, clear, kv, section, badge, button } from "../core/dom.js";
import { api, postBytes } from "../core/api.js";
import { state } from "../core/state.js";
import * as views from "../core/views.js";
import * as mic from "../voice/mic.js";

export const view = {
  id: "voice",
  title: "Voice Studio",
  async mount(pane) {
    const [settings, wake] = await Promise.all([api("/api/voice"), api("/api/voice/wake")]);
    const health = state.health || {};
    pane.append(el("div", { class: "card" },
      el("div", { class: "title" }, "Wake word ", badge((wake.wake_word || "zeus").toUpperCase(), "blue"), " ",
        wake.model_trained ? badge(wake.owner_samples ? "TRAINED ON OWNER" : "SYNTHETIC ONLY", wake.owner_samples ? "ok" : "warn") : badge("NOT TRAINED", "bad")),
      el("div", { class: "meta" }, el("span", { text: `listener ${health.voice ? "ready" : "not ready"}` }), el("span", { text: `recogniser ${health.recogniser ? "ready" : "not ready"}` }),
        el("span", { text: wake.metrics ? `recall ${wake.metrics.recall ?? "?"} · false activations/h ${wake.metrics.false_per_hour ?? "?"}` : "no measurement" }))));

    const s = settings.settings || settings;
    const form = el("div");
    const field = (label, key, type = "text") => {
      const input = el("input", { type, value: s[key] ?? "" });
      form.append(el("div", { class: "field" }, el("label", { text: label }), input));
      return () => [key, type === "number" ? Number(input.value) : input.value];
    };
    const readers = [
      field("microphone (device name or index)", "microphone"), field("speaker / output", "output"), field("voice (piper model)", "voice"),
      field("wake sensitivity 0–1", "wake_sensitivity", "number"), field("voice volume 0–1", "volume", "number"), field("language", "language"),
    ];
    pane.append(section("Settings", form, el("div", { class: "toolbar" }, button("Save", async () => {
      const payload = Object.fromEntries(readers.map((r) => r()).filter(([, v]) => v !== "" && !Number.isNaN(v)));
      const r = await api("/api/voice", payload);
      alert(r.ok === false ? r.error : "saved");
    }, "primary"))));

    const meter = el("div", { class: "bar", style: { width: "260px" } }, el("i", { style: { width: "0%" } }));
    pane.append(section("Live input", meter, el("div", { class: "toolbar" }, button("Test microphone (2 s)", async () => {
      await mic.recordClip(2, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
      meter.firstChild.style.width = "0%";
    }))));

    pane.append(section("Train the wake word", wizard(wake)));
  },
};

function wizard(wake) {
  const box = el("div");
  const status = el("div", { class: "empty", style: { padding: "6px 0" }, text: `${wake.positive || 0} owner recordings of „Zeus“, ${wake.negative || 0} other phrases on disk.` });
  const meter = el("div", { class: "bar", style: { width: "260px", margin: "6px 0" } }, el("i", { style: { width: "0%" } }));
  const prompt = el("div", { class: "card", style: { fontSize: "18px", textAlign: "center" }, text: "Ready." });
  const NEGATIVES = ["Wie spät ist es?", "Mach das Licht aus.", "Ich gehe jetzt einkaufen.", "Zeig mir die Nachrichten.", "Das Wetter ist heute schön.",
                     "Spiel etwas Musik.", "Ruf meine Mutter an.", "Was steht heute an?", "Erinnere mich an den Termin.", "Guten Morgen."];
  let running = false;
  async function record(kind, count, prompts) {
    if (running) return;
    running = true;
    for (let i = 0; i < count; i++) {
      prompt.textContent = kind === "positive" ? `Sag „Zeus“ (${i + 1}/${count})` : `Sag: „${prompts[i % prompts.length]}“ (${i + 1}/${count})`;
      await new Promise((r) => setTimeout(r, 700));
      const wav = await mic.recordClip(kind === "positive" ? 1.6 : 2.6, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
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
    button("3 · Train", async () => { prompt.textContent = "Training… (a minute or two)"; const r = await api("/api/voice/wake/train", {}); prompt.textContent = r.ok ? `Trained: recall ${r.metrics?.recall ?? "?"}, false activations/h ${r.metrics?.false_per_hour ?? "?"} (restart ZEUS to load it)` : (r.error || "training failed"); }),
    button("4 · Test detection (say Zeus)", async () => {
      prompt.textContent = "Say „Zeus“ now…";
      const wav = await mic.recordClip(2.0, (level) => { meter.firstChild.style.width = `${Math.round(level * 100)}%`; });
      const r = await postBytes("/api/voice/wake/test", wav, "audio/wav");
      prompt.textContent = r.ok ? `Score ${Number(r.score).toFixed(3)} — ${r.detected ? "DETECTED" : "not detected"} (threshold ${r.threshold})` : (r.error || "test failed");
    })));
  return box;
}
