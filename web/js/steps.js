/* Step-through page (web/steps.html): runs every selected growth method with
   step recording (/api/steps) and shows each method's joint configuration on
   its own midline chart. Forward / Skip 5 replay the shared step counter; a
   step slider and a frame slider scrub through the recorded snapshots. */

"use strict";
import { bounds2d, fmt, makeMap, setupCanvas, themeColors } from "./helpers.js";
import { getMethods, loadMethods } from "./methods.js";
const $ = (id) => document.getElementById(id);

// method list comes from the backend /api/methods (see js/methods.js)
let METHODS = getMethods();

const gstate = { folder: "", data: null, runs: [], cvs: [], step: 0, frame: 0, maxSteps: 0 };
const MAX_STEPS = 400; // snapshot cap per method, also requested server-side

// GETs a JSON API endpoint and throws on non-2xx responses.
async function api(path) {
  const resp = await fetch(path);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

// Shows a message in the status bar.
function setStatus(msg) { $("status").textContent = msg; }

// Builds the method checkbox list; brute-force methods are off by default.
function buildMethodList() {
  const wrap = $("methodList");
  wrap.innerHTML = "";
  gstate.cbs = [];
  const EXCLUDE = new Set(["brute_quantity", "brute_max_area"]);
  for (const [id, label] of METHODS) {
    const lab = document.createElement("label");
    lab.className = "mChip";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = id;
    cb.checked = !EXCLUDE.has(id);
    const span = document.createElement("span");
    span.textContent = label;
    lab.appendChild(cb);
    lab.appendChild(span);
    wrap.appendChild(lab);
    gstate.cbs.push(cb);
  }
}

// Ids of the checked method checkboxes.
function selectedMethods() {
  return gstate.cbs.filter(cb => cb.checked).map(cb => cb.value);
}

// Loads the file list for the folder, then auto-runs the step comparison.
async function loadFiles() {
  const meta = await api("/api/meta");
  const folder = $("folder").value.trim() || gstate.folder || meta.folder;
  const res = await api("/api/files?folder=" + encodeURIComponent(folder));
  gstate.folder = res.folder;
  $("folder").value = res.folder;
  const sel = $("file");
  sel.innerHTML = "";
  for (const f of res.files) {
    const o = document.createElement("option");
    o.value = f; o.textContent = f;
    sel.appendChild(o);
  }
  setStatus(res.files.length + " file(s) in " + res.folder);
  if (res.files.length) run();
}

// Fetches the display data and the step snapshots, then draws the grid.
async function run() {
  const file = $("file").value;
  if (!file) { setStatus("select a file first"); return; }
  const methods = selectedMethods();
  if (!methods.length) { setStatus("select at least one method"); return; }
  setStatus("running methods step by step ...");
  const q = "folder=" + encodeURIComponent(gstate.folder) +
            "&file=" + encodeURIComponent(file) +
            "&methods=" + methods.join(",") +
            "&threshold=" + $("threshold").value +
            "&segments=" + $("segments").value +
            "&max_steps=" + MAX_STEPS;
  try {
    const [data, steps] = await Promise.all([
      api("/api/data?folder=" + encodeURIComponent(gstate.folder) + "&file=" + encodeURIComponent(file)),
      api("/api/steps?" + q),
    ]);
    gstate.data = data;
    gstate.runs = steps.methods;
    gstate.frame = 0;
    $("frame").max = data.frames.length - 1;
    $("frame").value = 0;
    $("frameVal").textContent = "0 / " + data.num_frames;
    gstate.maxSteps = Math.max(1, ...gstate.runs.filter(r => !r.error).map(r => r.steps.length));
    gstate.step = 0;
    buildGrid();
    drawAll();
    setStatus(gstate.runs.length + " methods | max " + (gstate.maxSteps - 1) + " steps | use Forward / Skip 5");
  } catch (e) { setStatus("error: " + e.message); }
}

// Recreates the chart grid from the current runs (one card per method).
function buildGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  gstate.cvs = [];
  for (const run of gstate.runs) {
    const sec = document.createElement("section");
    sec.className = "chartCard";
    const title = document.createElement("h2");
    title.className = "chartTitle";
    const cv = document.createElement("canvas");
    cv.className = "chartCv";
    const foot = document.createElement("div");
    foot.className = "chartFoot";
    sec.appendChild(title);
    sec.appendChild(cv);
    sec.appendChild(foot);
    grid.appendChild(sec);
    gstate.cvs.push({ title, cv, foot });
  }
}

// Advances the shared step counter (clamped to the longest run).
function stepForward(n) {
  gstate.step = Math.min(gstate.maxSteps - 1, gstate.step + n);
  drawAll();
}

// Rewinds the shared step counter (clamped to zero).
function stepBack(n) {
  gstate.step = Math.max(0, gstate.step - n);
  drawAll();
}

// Syncs the step slider and counter label with the current step.
function updateStepControls() {
  const s = $("step");
  s.max = gstate.maxSteps - 1;
  s.value = Math.min(gstate.step, gstate.maxSteps - 1);
  $("stepVal").textContent = "step " + gstate.step + " / " + (gstate.maxSteps - 1);
}

// Redraws every chart at the current step and updates the controls.
function drawAll() {
  for (let i = 0; i < gstate.runs.length; i++) drawChart(gstate.runs[i], i);
  updateStepControls();
}

// Draws one method's midline with its joints at the current step and frame.
function drawChart(run, i) {
  const { title, cv, foot } = gstate.cvs[i];
  if (run.error) {
    title.textContent = run.label;
    foot.textContent = run.error;
    foot.classList.add("chartErr");
    return;
  }
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  const T = themeColors();
  g.clearRect(0, 0, w, h);
  const d = gstate.data;
  if (!d) return;
  const map = makeMap(bounds2d(d), w, h);
  const snap = run.steps[Math.min(gstate.step, run.steps.length - 1)];
  const f = Math.min(gstate.frame, d.frames.length - 1);
  const pts = d.frames[f];
  const tr = (d.trace[f] || []);
  const actual = d.actual[f];

  // the fish midline for the current frame (traced order, no padding)
  g.strokeStyle = T.midline; g.lineWidth = 1.4;
  g.beginPath();
  tr.forEach((row, k) => {
    const sx = map.x(pts[row][0]), sy = map.y(pts[row][1]);
    if (k === 0) g.moveTo(sx, sy); else g.lineTo(sx, sy);
  });
  g.stroke();

  // the joints at this step, located on the displayed frame by row index
  const jp = snap.map(j => (j[2] < actual ? [pts[j[2]][0], pts[j[2]][1]] : [j[0], j[1]]));
  if (jp.length > 1) {
    g.strokeStyle = "#d62728"; g.lineWidth = 1.6;
    g.beginPath();
    jp.forEach((p, k) => {
      const sx = map.x(p[0]), sy = map.y(p[1]);
      if (k === 0) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    });
    g.stroke();
  }
  for (const p of jp) {
    g.beginPath(); g.arc(map.x(p[0]), map.y(p[1]), 3.5, 0, Math.PI * 2);
    g.fillStyle = "#2ca02c"; g.fill();
    g.lineWidth = 1; g.strokeStyle = T.markerEdge; g.stroke();
  }

  // head and tail markers
  if (tr.length) {
    g.beginPath(); g.arc(map.x(pts[tr[0]][0]), map.y(pts[tr[0]][1]), 4, 0, Math.PI * 2);
    g.fillStyle = "#d62728"; g.fill();
    g.beginPath(); g.arc(map.x(pts[tr[tr.length - 1]][0]), map.y(pts[tr[tr.length - 1]][1]), 4, 0, Math.PI * 2);
    g.fillStyle = "#ff7f0e"; g.fill();
  }

  const done = gstate.step >= run.steps.length - 1;
  title.textContent = run.label + " - " + jp.length + " joints";
  foot.textContent =
    (run.truncated && done ? "capped at " + run.steps.length + " recorded steps | " : "") +
    "avg linear " + fmt(run.avg_linear) + " cm | avg area " + fmt(run.avg_area) + " cm^2";
}

/* ---------------- wiring ---------------- */

$("btnFolder").onclick = loadFiles;
$("folder").addEventListener("keydown", (e) => { if (e.key === "Enter") loadFiles(); });
$("run").onclick = run;
$("forward").addEventListener("click", () => stepForward(1));
$("skip5").addEventListener("click", () => stepForward(5));
$("back").addEventListener("click", () => stepBack(1));
$("back5").addEventListener("click", () => stepBack(5));
$("step").addEventListener("input", (e) => { gstate.step = parseInt(e.target.value, 10); drawAll(); });
$("frame").addEventListener("input", (e) => {
  gstate.frame = parseInt(e.target.value, 10);
  $("frameVal").textContent = gstate.frame + " / " + (gstate.data ? gstate.data.num_frames : 0);
  drawAll();
});
$("threshold").addEventListener("input", (e) => { $("thresholdVal").textContent = e.target.value; });
$("segments").addEventListener("input", (e) => { $("segmentsVal").textContent = e.target.value; });

/* ---------------- dark mode ---------------- */

// Labels the toggle with the mode you will switch TO.
function updateThemeBtn() {
  const dark = (document.documentElement.dataset.theme || "") === "dark";
  $("themeToggle").textContent = dark ? "Light" : "Dark";
}

// Flips the theme, persists it and redraws the charts.
function toggleTheme() {
  const dark = (document.documentElement.dataset.theme || "") === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("fishseg-theme", document.documentElement.dataset.theme); } catch (e) {}
  updateThemeBtn();
  drawAll();
}

$("themeToggle").onclick = toggleTheme;
updateThemeBtn();

// Boot: load the method list, then the file list (which auto-runs).
(async function init() {
  await loadMethods();
  METHODS = getMethods();
  buildMethodList();
  loadFiles().catch(e => setStatus("error: " + e.message));
})();
