/* The Studium indexing line in words: real percentages, the page being read, calm failures, a line that folds away.
   Run: node ui/tests/study_indexing.test.mjs  (the Python test suite runs it too). */

import { lineFor, jobRows, jobWords } from "../core/indexing.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}

const running = { job_id: "a", kind: "import", name: "Physiologie.pdf", state: "running", phase: "ocr", ocr_pages_total: 64, ocr_pages_completed: 17, fraction: 0.41, created_at: 2 };
const waiting = { job_id: "b", kind: "import", name: "Anatomie.pptx", state: "queued", phase: "queued", fraction: 0, created_at: 3 };
const done = { job_id: "c", kind: "import", name: "Niere.md", state: "ready", phase: "ready", fraction: 1, created_at: 1 };
const broken = { job_id: "d", kind: "import", name: "Geheim.pdf", state: "failed", reason: "encrypted", fraction: 1, created_at: 4 };

const active = { ok: true, active: true, documents_total: 3, documents_completed: 1, percent: 47, current: running, jobs: [done, running, waiting], failed: [] };
const line = lineFor(active);
check("active: which document of how many, the real percentage", line.busy && line.text === "Indexiere 2 von 3 Dokumenten · 47 %", line.text);
check("active: the page the OCR is on", line.detail === "Physiologie.pdf · Texterkennung · Seite 18 von 64", line.detail);
check("one document: no 'von 1'", lineFor({ ...active, documents_total: 1, documents_completed: 0 }).text === "Indexiert · 47 %");
check("embedding progress in passages", jobWords({ state: "running", phase: "embedding", embedding_chunks_total: 120, embedding_chunks_completed: 30 }) === "semantische Suche · 30 von 120 Abschnitten");

const finished = lineFor({ ok: true, active: false, documents_total: 2, documents_completed: 2, percent: 100, jobs: [done, { ...done, job_id: "e" }], failed: [] });
check("finished: says so, then may fold", finished.done && finished.text === "Fertig · 2 Dokumente durchsuchbar", finished.text);
check("nothing happening: no line", lineFor({ ok: true, active: false, documents_total: 0, documents_completed: 0, percent: 100, jobs: [], failed: [] }) === null);
check("handing files over before the server knows them", lineFor(null, { uploading: 2 }).text === "2 Dateien werden übergeben …");

check("encrypted: calm and actionable", jobWords(broken) === "Diese PDF ist mit einem Passwort geschützt.");
check("GoodNotes native: honest", jobWords({ state: "failed", reason: "goodnotes_native" }).includes("als PDF"));
check("unknown reason: the server's own words", jobWords({ state: "failed", reason: "x", error: "Kaputt." }) === "Kaputt.");

const rows = jobRows({ ...active, failed: [broken] });
check("list order: running, waiting, failed, done", rows.map((r) => r.id).join() === "a,b,d,c", rows.map((r) => r.id).join());
check("running and waiting can be cancelled, finished cannot", rows.filter((r) => r.cancellable).map((r) => r.id).join() === "a,b");
check("dismissed failures stay away", !jobRows({ ...active, failed: [broken] }, new Set(["d"])).some((r) => r.id === "d"));
check("a job listed twice appears once", jobRows({ jobs: [broken], failed: [broken] }).length === 1);
check("the same file failing the same way twice is one row", jobRows({ jobs: [], failed: [broken, { ...broken, job_id: "d2" }] }).length === 1);

console.log(failures ? `${failures} FAILED` : "ALL OK");
process.exit(failures ? 1 : 0);
