/* The Studium PDF viewer's rendering plan: preview then sharp, tiles only beyond the cap, the bounded bitmap cache,
   one view generation per page change, 204 = superseded, aborted fetches are quiet.
   Run: node ui/tests/study_pdf.test.mjs  (the Python test suite runs it too). */

import { renderPlan, visibleTiles, visibleFraction, BitmapCache, PageLoader, SHARP_CAP, TILE_PX } from "../core/pagerender.js";

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) { console.log(`ok   ${name}`); return; }
  failures += 1;
  console.log(`FAIL ${name}${detail ? " -- " + detail : ""}`);
}

// -- plan ---------------------------------------------------------------------------------------------
const normal = renderPlan(880, 1);
check("100 % on a normal screen: preview == sharp, no tiles", normal.preview === 900 && normal.sharp === 900 && !normal.tiles, JSON.stringify(normal));
const retina = renderPlan(880, 2);
check("HiDPI: sharp at device pixels, preview smaller", retina.sharp === 1800 && retina.preview === 900 && !retina.tiles, JSON.stringify(retina));
const zoomed = renderPlan(2640, 2);
check("zoomed far in: the full page is capped, tiles carry the detail",
  zoomed.sharp === SHARP_CAP && zoomed.tiles && zoomed.tileScale === 5200, JSON.stringify(zoomed));
check("never more than the tile cap", renderPlan(9000, 3).tileScale === 12000);

// -- tiles --------------------------------------------------------------------------------------------
const top = visibleTiles({ x: 0, y: 0, w: 0.3, h: 0.2 }, 5200, 595 / 842, 0);
const all = visibleTiles({ x: 0, y: 0, w: 1, h: 1 }, 5200, 595 / 842, 0);
check("only the visible tiles are asked for", top.length > 0 && top.length < all.length / 3, `${top.length} of ${all.length}`);
check("every tile is at most TILE_PX at the render scale", all.every((t) => t.clip[2] * 5200 <= TILE_PX + 1));
check("tiles cover the page", Math.abs(all.filter((t) => t.clip[1] === 0).reduce((sum, t) => sum + t.clip[2], 0) - 1) < 0.01);
const frac = visibleFraction({ left: 0, top: -500, right: 1000, bottom: 1000, width: 1000, height: 1500 }, { left: 0, top: 0, right: 800, bottom: 500 });
check("visible fraction of a scrolled sheet", Math.abs(frac.y - 1 / 3) < 1e-6 && Math.abs(frac.h - 1 / 3) < 1e-6 && Math.abs(frac.w - 0.8) < 1e-6, JSON.stringify(frac));

// -- cache --------------------------------------------------------------------------------------------
const revoked = [];
const cache = new BitmapCache(3, (url) => revoked.push(url));
cache.set("a", "blob:a"); cache.set("b", "blob:b"); cache.keep = new Set(["a"]); cache.set("c", "blob:c"); cache.set("d", "blob:d");
check("bounded, oldest unprotected bitmap revoked", cache.map.size === 3 && revoked.join() === "blob:b" && cache.get("a") === "blob:a", revoked.join());

// -- loader -------------------------------------------------------------------------------------------
globalThis.URL.createObjectURL = (blob) => `blob:${blob.name}`;
const asked = [];
const fetchImpl = async (url, options) => {
  asked.push(url);
  if (url.includes("page=9")) return { status: 204, ok: true };
  if (url.includes("page=7")) {
    return new Promise((_, reject) => options.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }))));
  }
  return { status: 200, ok: true, blob: async () => ({ name: url }) };
};
const loader = new PageLoader({ urlFor: (page, width, clip, seq, prio, session) => `page=${page}&w=${width}&seq=${seq}&prio=${prio}&s=${session}`, fetchImpl,
                                cache: new BitmapCache(8, () => {}) });
loader.begin();
const first = await loader.load(1, 900);
await loader.load(1, 1800);
check("one view generation shares one seq (preview and sharp are not superseding each other)", asked[0].includes("seq=1") && asked[1].includes("seq=1"), asked.join(" "));
check("a cached bitmap is not fetched again", (await loader.load(1, 900)) === first && asked.length === 2);
const slow = loader.load(7, 900);
loader.begin();
check("a page change aborts the old fetch quietly", (await slow) === null);
check("the next page change has a newer generation", loader.seq === 2);
check("superseded on the server (204) shows nothing", (await loader.load(9, 900)) === null);

// seen live: the browser's fetch called as a method of the loader throws "Illegal invocation" and every page showed as broken
globalThis.fetch = function strictFetch() {
  if (this !== undefined && this !== globalThis) throw new TypeError("Illegal invocation");
  return Promise.resolve({ status: 200, ok: true, blob: async () => ({ name: "default" }) });
};
const plain = new PageLoader({ urlFor: () => "page=1", cache: new BitmapCache(4, () => {}) });
plain.begin();
check("the default fetch is called unbound", (await plain.load(1, 900).catch((e) => String(e))) === "blob:default");

if (failures) { console.log(`${failures} failed`); process.exit(1); }
console.log("all passed");
