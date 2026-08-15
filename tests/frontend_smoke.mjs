/* Headless smoke test for the frontend ES modules (web/js/app.js, web/js/fluid.js,
   web/js/steps.js). Loads the real modules under a minimal DOM + fetch stub,
   lets the async init chains run, and asserts the UI populates. Run with:
   node tests/frontend_smoke.mjs  */

import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const DATA_FOLDER = join(REPO, "Sturgeon from Elsa and Ted", "midlines");
const FILES = Array.from({ length: 15 }, (_, i) => `fish_${i}.xls`);

function makeCtx() {
  return new Proxy({}, {
    get(t, p) {
      if (p === "measureText") return () => ({ width: 10 });
      return t[p] !== undefined ? t[p] : (() => {});
    },
    set() { return true; },
  });
}

function makeEl(tag) {
  const el = {
    tagName: tag.toUpperCase(), children: [], value: "", textContent: "", innerHTML: "",
    checked: false, type: "", min: "", max: "", step: "", title: "", href: "", download: "",
    style: {}, dataset: {}, options: [], _listeners: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); },
    removeEventListener() {},
    appendChild(c) {
      this.children.push(c);
      if (this.tagName === "SELECT") {
        this.options.push(c);
        if (!this.value) this.value = this.options[0] ? this.options[0].value : "";
      }
    },
    setAttribute() {}, getAttribute() { return null; },
    querySelector() { return makeEl("div"); },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { width: 300, height: 200, left: 0, top: 0 }; },
    getContext() { return this._ctx || (this._ctx = makeCtx()); },
    click() {}, focus() {}, blur() {}, scrollIntoView() {}, remove() {},
    _setInnerHTML(v) { this.children.length = 0; this.options.length = 0; },
  };
  Object.defineProperty(el, "innerHTML", {
    get() { return el._html || ""; },
    set(v) { el._html = v; if (el.tagName === "SELECT") { el.children.length = 0; el.options.length = 0; } },
  });
  Object.defineProperty(el, "selectedIndex", { get: () => 0, set: () => {} });
  Object.defineProperty(el, "selectedOptions", {
    get: () => { const o = el.options.find(x => x.value === el.value); return [o || { textContent: el.value }]; },
  });
  return el;
}

const byId = {};
globalThis.document = {
  documentElement: { dataset: { theme: "light" } },
  getElementById(id) { return byId[id] || (byId[id] = makeEl("SELECT")); },
  createElement(tag) { return makeEl(tag); },
  querySelector() { return makeEl("div"); },
  querySelectorAll() { return []; },
};
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.matchMedia = () => ({ matches: false });
globalThis.window = { devicePixelRatio: 1, addEventListener() {}, open() {} };
globalThis.location = { search: "" };
globalThis.URL.createObjectURL = () => "blob:stub";
globalThis.URL.revokeObjectURL = () => {};

let fails = 0;
function check(name, cond) {
  console.log((cond ? "PASS " : "FAIL ") + name);
  if (!cond) fails++;
}

const METHODS = ["grow_area", "grow_bs_area", "grow_bs_mp_area", "grow_inflect_area",
  "grow_linear", "grow_bs_linear", "grow_bs_mp_linear", "grow_inflect_linear",
  "equal_segments", "diminishing_segments"].map((id, i) => ({
    id, label: id, param: i < 8 ? "error_threshold" : "segment_count",
    param_label: i < 8 ? "error threshold" : "segments", type: i < 8 ? "err" : "count" }));

const framesData = Array.from({ length: 5 }, () =>
  Array.from({ length: 20 }, (_, p) => [p * 2, Math.sin(p / 3) * 10, 0]));

async function fakeFetch(url) {
  const u = String(url);
  // background-job payloads: async=1 requests return {job}, the progress poll
  // delivers the finished result
  const jobPayloads = fakeFetch.jobPayloads;
  if (u.includes("/api/fluid/progress")) {
    const id = u.split("job=")[1] || "";
    const payload = jobPayloads[id] || {};
    delete jobPayloads[id];
    return { ok: true, json: async () => ({ status: "done", progress: 1, message: "done", result: payload }) };
  }
  if (u.includes("/api/meta")) return { ok: true, json: async () => ({ folder: DATA_FOLDER, port: 8123 }) };
  if (u.includes("/api/methods")) return { ok: true, json: async () => ({ methods: METHODS }) };
  if (u.includes("/api/files")) return { ok: true, json: async () => ({ folder: DATA_FOLDER, files: FILES, dirs: [] }) };
  if (u.includes("/api/sessions")) return { ok: true, json: async () => ({ sessions: [] }) };
  if (u.includes("/api/generate")) return { ok: true, json: async () => ({
    method: "grow_bs_area", param: 5, ptype: "err", time: 0.01,
    joints: [[0, 0, 0], [5, 4, 10], [9, 8, 19]], count: 3,
    avg_linear: 0.1, avg_area: 0.2, stride: 1,
    per_linear: [0.1, 0.1, 0.1, 0.1, 0.1], per_area: [0.2, 0.2, 0.2, 0.2, 0.2],
  }) };
  if (u.includes("/api/data")) return { ok: true, json: async () => ({
    name: "fish_0.xls", num_frames: 5, num_points: 20, stride: 1,
    frames: framesData,
    trace: framesData.map(() => framesData[0].map((_, i) => i)),
    actual: [20, 20, 20, 20, 20], has_z: false,
  }) };
  if (u.includes("/api/fluid/simulate")) {
    const payload = {
      method: "equal_segments", param: 6, ptype: "count", count: 7, total_cm: 55, steps: 10,
      sim_dt: 0.05, flow_target: 0.035, osc_amp: 2, osc_freq: 0.008, body_kind: "segments",
      mode: u.includes("mode=swim") ? "swim" : "tethered", cycles: 1.0,
      joint_pts: [[0, 0], [30, 5], [60, 10]], rest: [[0, 0], [30, 5], [60, 10]],
      nx: 100, ny: 80, xs: [0, 10], ys: [0, 10], stride: 4, frame_stride: 1,
      frames: Array.from({ length: 4 }, () => ({
        u: [[0, 0], [0, 0]], body: [[0, 0], [30, 5], [60, 10]],
        joints: [[0, 0], [30, 5], [60, 10]], flow: 0.03,
        ...(u.includes("mode=swim") ? { target: [[0, 0], [30, 5], [60, 10]] } : {}),
      })),
    };
    if (u.includes("async=1")) {
      const id = "job_" + (++fakeFetch.jobSeq);
      jobPayloads[id] = payload;
      return { ok: true, json: async () => ({ job: id, steps: 10 }) };
    }
    return { ok: true, json: async () => payload };
  }
  if (u.includes("/api/fluid/compare")) {
    const payload = {
      file: "fish_0.xls", nx: 100, ny: 80, flow_target: 0.035, osc_amp: 2.5,
      osc_freq: 0.012, cycles: 1.0, steps: 60, threshold: 0.5, segments: 10,
      runs: [
        { id: "grow_bs_area", label: "grow_bs_area", param: 0.5, ptype: "err",
          count: 5, total_cm: 55, body_kind: "segments",
          metrics: { steps: 60, warmup: 15, speed: 0.012, speed_bl: 0.004,
            thrust: 0.02, power: 0.004, efficiency: 0.06, cost: 0.33,
            strouhal: 2.1, tail_amp: 2.3, track_rms: 0.05, yaw_rms: 0.02,
            wake: 0.001, finite: true, body_len_cells: 55 },
          series: { t: [0, 1, 2], speed: [0, 0.01, 0.012], thrust: [0, 0.01, 0.02],
            power: [0.004, 0.004, 0.004], tail_y: [0, 1, 0], flow: [0, 0, 0] },
          joint_pts: [[0, 0], [30, 5], [60, 10]], rest: [[0, 0], [30, 5], [60, 10]],
          xs: [0, 10], ys: [0, 10], stride: 4, frame_stride: 1, sim_dt: 0.05,
          frames: Array.from({ length: 3 }, () => ({
            u: [[0, 0], [0, 0]], body: [[0, 0], [30, 5], [60, 10]],
            joints: [[0, 0], [30, 5], [60, 10]], flow: 0.03,
            target: [[0, 0], [30, 5], [60, 10]],
          })) },
        { id: "equal_segments", label: "equal_segments", param: 10, ptype: "count",
          count: 11, total_cm: 55, body_kind: "segments",
          metrics: { steps: 60, warmup: 15, speed: 0.008, speed_bl: 0.003,
            thrust: 0.01, power: 0.005, efficiency: 0.016, cost: 0.62,
            strouhal: 3.4, tail_amp: 2.0, track_rms: 0.08, yaw_rms: 0.03,
            wake: 0.001, finite: true, body_len_cells: 55 },
          series: { t: [0, 1, 2], speed: [0, 0.006, 0.008], thrust: [0, 0.005, 0.01],
            power: [0.005, 0.005, 0.005], tail_y: [0, 1, 0], flow: [0, 0, 0] },
          joint_pts: [[0, 0], [30, 5], [60, 10]], rest: [[0, 0], [30, 5], [60, 10]],
          xs: [0, 10], ys: [0, 10], stride: 4, frame_stride: 1, sim_dt: 0.05,
          frames: Array.from({ length: 3 }, () => ({
            u: [[0, 0], [0, 0]], body: [[0, 0], [30, 5], [60, 10]],
            joints: [[0, 0], [30, 5], [60, 10]], flow: 0.03,
            target: [[0, 0], [30, 5], [60, 10]],
          })) },
      ],
    };
    if (u.includes("async=1")) {
      const id = "job_" + (++fakeFetch.jobSeq);
      jobPayloads[id] = payload;
      return { ok: true, json: async () => ({ job: id, runs_total: 2, steps: 60 }) };
    }
    return { ok: true, json: async () => payload };
  }
  if (u.includes("/api/steps")) return { ok: true, json: async () => ({
    file: "fish_0.xls", max_steps: 400,
    methods: [
      { id: "grow_area", label: "grow_area", param: 0.5, ptype: "err", truncated: false,
        avg_linear: 0.1, avg_area: 0.2, count: 5,
        steps: [
          [[0, 0, 0], [2, 3, 5]],
          [[0, 0, 0], [2, 3, 5], [4, 6, 10]],
          [[0, 0, 0], [2, 3, 5], [4, 6, 10], [9, 8, 19]],
          [[0, 0, 0], [2, 3, 5], [4, 6, 10], [9, 8, 19], [12, 4, 15]],
        ] },
      { id: "equal_segments", label: "equal_segments", param: 10, ptype: "count",
        truncated: false, avg_linear: 0.05, avg_area: 0.1, count: 2,
        steps: [[[0, 0, 0]], [[0, 0, 0], [9, 8, 19]]] },
    ],
  }) };
  throw new Error("unexpected fetch: " + u);
}
fakeFetch.jobPayloads = {};
fakeFetch.jobSeq = 0;
globalThis.fetch = fakeFetch;

process.on("unhandledRejection", (e) => { console.log("UNHANDLED REJECTION:", e && e.stack || e); });

/* ---------------- main viewer (web/js/app.js) ---------------- */
await import(pathToFileURL(join(REPO, "web", "js", "app.js")).href);
await sleep(600);
const status = byId["status"];
check("viewer: status shows file count", (status.textContent || "").includes("file(s) in"));
check("viewer: file select populated (15)", byId["file"].options.length === 15);
check("viewer: method select populated (10)", byId["mMethod"] ? byId["mMethod"].options.length === 10 : true);
byId["themeToggle"].onclick();
check("viewer: theme toggles to dark", document.documentElement.dataset.theme === "dark");
check("viewer: theme button label flips", byId["themeToggle"].textContent === "Light");
byId["themeToggle"].onclick();
check("viewer: theme toggles back to light", document.documentElement.dataset.theme === "light");

/* ---------------- fluid page (web/js/fluid.js) ---------------- */
await import(pathToFileURL(join(REPO, "web", "js", "fluid.js")).href);
await sleep(600);
check("fluid: file select populated (15)", byId["file"].options.length === 15);
check("fluid: method select populated (10)", byId["method"].options.length === 10);
check("fluid: simulation ran", (status.textContent || "").includes("simulation ready"));
check("fluid: progress bar reported progress", (byId["progressMsg"].textContent || "").length > 0);
byId["resHigh"]._listeners["click"][0]();
check("fluid: resolution toggle switches to high", (status.textContent || "").includes("resolution: high"));
byId["resMax"]._listeners["click"][0]();
check("fluid: resolution toggle switches to max", (status.textContent || "").includes("resolution: max (800 × 480"));
byId["resLow"]._listeners["click"][0]();
check("fluid: resolution toggle back to low", (status.textContent || "").includes("resolution: low"));
byId["themeToggle"].onclick();
check("fluid: theme toggles to dark", document.documentElement.dataset.theme === "dark");
byId["themeToggle"].onclick();
document.getElementById("loop").checked = true;
byId["play"].onclick();
check("fluid: play toggles to Pause", byId["play"].textContent === "Pause");
byId["play"].onclick();
check("fluid: pause toggles back to Play", byId["play"].textContent === "Play");
byId["simMode"].value = "swim";
byId["simMode"]._listeners["change"][0]();
check("fluid: swim mode switches the amp label", byId["ampLabel"].textContent === "tail amp");
byId["freqVal"].value = "0.012";
byId["amp"]._listeners["input"][0]();
check("fluid: amp drag does not change the freq value", byId["freqVal"].value === "0.012");
byId["freq"].value = "1000";
byId["freq"]._listeners["input"][0]();
check("fluid: log freq slider maps to the max frequency", byId["freqVal"].value === "0.0500");
byId["freqVal"].value = "0.0005";
byId["freqVal"]._listeners["input"][0]();
check("fluid: typed freq syncs the slider to the minimum", Number(byId["freq"].value) === 0);
byId["btnCompare"].onclick();
await sleep(600);
check("fluid: compare table populated", (byId["cmpTable"].innerHTML || "").includes("speed"));
check("fluid: compare status shows runs", (byId["compareStatus"].textContent || "").includes("compare done"));
check("fluid: compare run cards populated (2)", byId["cmpRuns"].children.length === 2);

/* ---------------- playback helper (web/js/helpers.js advancePlay) ---------------- */
const helpersMod = await import(pathToFileURL(join(REPO, "web", "js", "helpers.js")).href);
check("playback: loop wraps end -> start", helpersMod.advancePlay(3, 4, true, 1).join(",") === "0,1");
check("playback: rewind bounces at the end", helpersMod.advancePlay(3, 4, false, 1).join(",") === "2,-1");
check("playback: rewind bounces at the start", helpersMod.advancePlay(0, 4, false, -1).join(",") === "1,1");
check("playback: loop ignores a stale reverse direction", helpersMod.advancePlay(2, 4, true, -1).join(",") === "3,1");
check("playback: single frame stays put", helpersMod.advancePlay(0, 1, false, 1).join(",") === "0,1");

/* ---------------- step-through page (web/js/steps.js) ---------------- */
await import(pathToFileURL(join(REPO, "web", "js", "steps.js")).href);
await sleep(600);
check("steps: method chips populated (10)", byId["methodList"].children.length === 10);
check("steps: chart grid populated (2)", byId["grid"].children.length === 2);
check("steps: status shows step summary", (status.textContent || "").includes("steps"));
const stepVal0 = byId["stepVal"].textContent;
byId["forward"]._listeners["click"][0]();
check("steps: forward advances the counter", byId["stepVal"].textContent !== stepVal0);
byId["skip5"]._listeners["click"][0]();
check("steps: skip 5 clamps at the end", byId["stepVal"].textContent === "step 3 / 3");

console.log(fails === 0 ? "ALL PASS" : fails + " FAILURES");
process.exit(fails === 0 ? 0 : 1);
