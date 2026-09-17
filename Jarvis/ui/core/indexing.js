/* The Studium indexing line: what the background indexer is doing, in words.

   The server keeps the truth (persisted jobs, /api/study/indexing); this module only turns its
   summary into the one quiet line under the search and the rows of the compact job list.
   Pure functions, no DOM: ui/tests/study_indexing.test.mjs covers them. */

export const PHASE_WORDS = {
  queued: "wartet", parsing: "wird gelesen", rendering: "Seiten werden gezeichnet", ocr: "Texterkennung", chunking: "wird gegliedert",
  indexing: "wird indexiert", embedding: "semantische Suche wird vorbereitet", ready: "durchsuchbar", failed: "nicht gelesen",
};

export const FAILURE_WORDS = {
  unsupported: "Dieses Format kann Studium nicht lesen.",
  missing: "Die Datei ist nicht mehr da.",
  moved: "Die Datei wurde verschoben.",
  too_large: "Die Datei ist zu groß.",
  parse_failed: "Die Datei ließ sich nicht lesen – sie ist vielleicht beschädigt.",
  encrypted: "Diese PDF ist mit einem Passwort geschützt.",
  goodnotes_native: "GoodNotes-Dateien kann ZEUS nicht direkt lesen. Exportiere das Notizbuch als PDF.",
  ocr_failed: "Die Texterkennung hat hier nichts lesen können.",
  handwriting_unavailable: "Handschrifterkennung ist gerade nicht verfügbar – gedruckter Text ist durchsuchbar.",
  empty: "Die Datei ist leer.",
  embedding: "Semantische Suche nicht verfügbar – die Stichwortsuche funktioniert.",
  cancelled: "abgebrochen",
};

const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const percent = (fraction) => `${Math.max(0, Math.min(100, Math.round(100 * (Number(fraction) || 0))))} %`;

/* What one job is doing, precisely: the page of the OCR, the passages of the embedding. */
export function jobWords(job) {
  if (!job) return "";
  if (job.state === "failed") return FAILURE_WORDS[job.reason] || job.error || "nicht gelesen";
  if (job.state === "cancelled") return "abgebrochen";
  if (job.state === "skipped") return job.reason === "duplicate" ? "war schon da" : "übersprungen";
  if (job.state === "ready" || job.state === "done") return job.kind === "embed" ? "semantisch durchsuchbar" : "durchsuchbar";
  if (job.state === "queued") return "wartet";
  const phase = job.phase || "parsing";
  if (phase === "ocr" && Number(job.ocr_pages_total)) return `Texterkennung · Seite ${Math.min(Number(job.ocr_pages_completed || 0) + 1, Number(job.ocr_pages_total))} von ${job.ocr_pages_total}`;
  if (phase === "embedding" && Number(job.embedding_chunks_total)) return `semantische Suche · ${job.embedding_chunks_completed || 0} von ${job.embedding_chunks_total} Abschnitten`;
  if (phase === "rendering" && Number(job.slides_total)) return `Folien werden gezeichnet · ${job.slides_completed || 0} von ${job.slides_total}`;
  if (phase === "parsing" && Number(job.pages_total)) return `wird gelesen · ${count(Number(job.pages_total), "Seite", "Seiten")}`;
  return PHASE_WORDS[phase] || phase;
}

/* The line under the search: null when there is nothing to say (the line collapses). */
export function lineFor(summary, { uploading = 0 } = {}) {
  if (!summary || summary.ok === false) return uploading ? { busy: true, text: `${count(uploading, "Datei wird", "Dateien werden")} übergeben …`, fraction: null } : null;
  const total = Number(summary.documents_total) || 0;
  const done = Number(summary.documents_completed) || 0;
  const fraction = (Number(summary.percent) || 0) / 100;
  if (summary.active) {
    const current = summary.current;
    const detail = current ? `${current.name || "Material"} · ${jobWords(current)}` : "";
    const head = total > 1 ? `Indexiere ${Math.min(done + 1, total)} von ${total} Dokumenten · ${percent(fraction)}` : `Indexiert · ${percent(fraction)}`;
    return { busy: true, text: head, detail, fraction };
  }
  if (uploading) return { busy: true, text: `${count(uploading, "Datei wird", "Dateien werden")} übergeben …`, fraction: null };
  const failed = (summary.jobs || []).filter((j) => j.state === "failed").length;
  if (total) {
    const ready = (summary.jobs || []).filter((j) => (j.state === "ready" || j.state === "done") && j.kind !== "embed").length;
    const text = failed ? `Fertig · ${count(failed, "Datei", "Dateien")} nicht gelesen` : `Fertig · ${count(ready || done, "Dokument", "Dokumente")} durchsuchbar`;
    return { busy: false, done: true, text, fraction: 1, failed };
  }
  return null;
}

/* The compact list behind the line: running first, then waiting, then what just finished. */
export function jobRows(summary, dismissed = new Set()) {
  const order = { running: 0, queued: 1, failed: 2, ready: 3, done: 3, skipped: 4, cancelled: 5 };
  const seen = new Set();
  const jobs = [...((summary && summary.jobs) || []), ...((summary && summary.failed) || [])].filter((j) => {
    // one line per job -- and a file that failed the same way twice is one line, not two
    const same = j && j.state === "failed" ? `failed:${j.name}:${j.reason}` : "";
    if (!j || seen.has(j.job_id) || (same && seen.has(same)) || dismissed.has(j.job_id)) return false;
    seen.add(j.job_id);
    if (same) seen.add(same);
    return true;
  });
  return jobs.sort((a, b) => (order[a.state] ?? 9) - (order[b.state] ?? 9) || (a.created_at || 0) - (b.created_at || 0))
    .map((j) => ({ id: j.job_id, name: j.name || (j.kind === "embed" ? "Semantische Suche" : "Material"), words: jobWords(j), state: j.state,
                   fraction: Number(j.fraction) || 0, cancellable: j.state === "queued" || j.state === "running" }));
}
