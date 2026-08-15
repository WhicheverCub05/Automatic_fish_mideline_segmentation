/* Fluid simulation page (web/fluid.html): runs the joint configuration of a
   data file through the backend LBM fluid simulation (/api/fluid/simulate)
   and animates the flow field, the fish body and the joint markers. Two
   movement modes: 'tethered' (head oscillated in a current) and 'swim'
   (free swimming - joints actuated with a traveling wave, metrics reported).
   The compare panel runs several generation algorithms through the same
   free-swim simulation (/api/fluid/compare) and ranks their performance. */

"use strict";
import { advancePlay, fmt, setupCanvas, themeColors } from "./helpers.js";
import { getMethods, loadMethods } from "./methods.js";
const $ = (id) => document.getElementById(id);

// method list comes from the backend /api/methods (see js/methods.js)
let METHODS = getMethods();

const gstate = {
  folder: "", sim: null, frame: 0, playing: false, playDir: 1, timer: null,
  res: "low", mode: "tethered", compare: null,
};
const qp = new URLSearchParams(location.search);

// Lattice sizes per resolution (lattice cells). Low matches the original
// defaults; higher resolutions give a finer flow field at a runtime cost.
const RESOLUTIONS = {
  low: { nx: 260, ny: 160 },
  medium: { nx: 320, ny: 200 },
  high: { nx: 400, ny: 240 },
  max: { nx: 800, ny: 480 },
};
const RES_BUTTONS = { low: "resLow", medium: "resMed", high: "resHigh", max: "resMax" };

// Color palette for the compare series / run cards (one color per run).
const PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#17becf"];

// GETs a JSON API endpoint and throws on non-2xx responses.
async function api(path) {
  const resp = await fetch(path);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

// Shows a message in the status bar.
function setStatus(msg) { $("status").textContent = msg; }

/* ---------------- progress bar ---------------- */

// Shows/hides the progress bar under the header.
function showProgress(on) { $("progressWrap").style.display = on ? "flex" : "none"; }

// Sets the progress bar fill and message text.
function setProgress(frac, msg) {
  showProgress(true);
  $("progressFill").style.width = Math.max(0, Math.min(100, Math.round(frac * 100))) + "%";
  $("progressMsg").textContent = msg + " (" + Math.round(frac * 100) + "%)";
}

// Starts a background simulation job (async=1) and polls it to completion,
// driving the progress bar; resolves with the finished result payload.
async function pollJob(url, initialStatus) {
  setStatus(initialStatus);
  const job = await api(url + "&async=1");
  setProgress(0.01, "starting simulation");
  while (true) {
    await new Promise(res => setTimeout(res, 400));
    const p = await api("/api/fluid/progress?job=" + encodeURIComponent(job.job));
    setProgress(p.progress || 0, p.message || "");
    if (p.status === "done") { showProgress(false); return p.result; }
    if (p.status === "error") { showProgress(false); throw new Error(p.error || "simulation failed"); }
  }
}

// Frequency range (cycles/step). The slider is LOG-SCALE (the range spans two
// decades) and stays in sync with the exact-value text box; dragging one never
// moves the other slider (amp is fully independent).
const FREQ_MIN = 0.0005, FREQ_MAX = 0.05, FREQ_SLIDER_MAX = 1000;

// Maps the log slider position (0..FREQ_SLIDER_MAX) to a frequency.
function sliderToFreq(v) {
  const t = parseFloat(v) / FREQ_SLIDER_MAX;
  return FREQ_MIN * Math.pow(FREQ_MAX / FREQ_MIN, t);
}

// Maps a frequency to the nearest log slider position.
function freqToSlider(f) {
  const t = Math.log(Math.min(FREQ_MAX, Math.max(FREQ_MIN, f)) / FREQ_MIN) /
            Math.log(FREQ_MAX / FREQ_MIN);
  return Math.round(t * FREQ_SLIDER_MAX);
}

// Reads the frequency from the text box (falling back to the slider).
function currentFreq() {
  let f = parseFloat($("freqVal").value);
  if (!isFinite(f)) f = sliderToFreq($("freq").value);
  return Math.min(FREQ_MAX, Math.max(FREQ_MIN, f));
}

// Sets both freq widgets to the same (clamped) frequency.
function setFreq(f) {
  f = Math.min(FREQ_MAX, Math.max(FREQ_MIN, f));
  $("freqVal").value = f.toFixed(4);
  $("freq").value = freqToSlider(f);
}

// Fills the method dropdown from the shared method list.
function fillMethodOptions() {
  const sel = $("method");
  sel.innerHTML = "";
  for (const [id, label] of METHODS) {
    const o = document.createElement("option");
    o.value = id; o.textContent = label;
    sel.appendChild(o);
  }
  sel.value = "grow_bs_area";
}

// Builds the compare-panel method checkboxes (brute-force methods off).
function fillCompareMethods() {
  const list = $("methodList");
  list.innerHTML = "";
  for (const [id, label] of METHODS) {
    const w = document.createElement("span");
    w.className = "cmpMethod";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = id;
    cb.checked = !id.startsWith("brute_");
    const t = document.createElement("span");
    t.textContent = label;
    w.appendChild(cb);
    w.appendChild(t);
    list.appendChild(w);
  }
}

// Adapts the param slider (range/step/label) to the selected method type.
function onMethodChange() {
  const m = METHODS.find(x => x[0] === $("method").value);
  const count = m[2] === "count";
  $("param").min = count ? "2" : "0.05";
  $("param").max = count ? "40" : "20";
  $("param").step = count ? "1" : "0.05";
  $("param").value = count ? "10" : "0.5";
  $("paramLabel").textContent = count ? "segments" : "threshold";
  $("paramVal").textContent = $("param").value;
}

// Switches the movement mode UI (tethered head wiggle vs free-swim wave).
function onModeChange() {
  const swim = $("simMode").value === "swim";
  gstate.mode = swim ? "swim" : "tethered";
  $("ampLabel").textContent = swim ? "tail amp" : "head amp";
  $("cyclesLabel").style.display = swim ? "" : "none";
  $("cycles").style.display = swim ? "" : "none";
  $("cyclesVal").style.display = swim ? "" : "none";
  if (swim) {
    // propulsion tests default to still water (the flow slider becomes a
    // background-current dial; a strong current can carry the fish backward)
    $("flow").value = "0";
    $("flowVal").textContent = "0";
    // sensible free-swim defaults (the backend clamps amp * freq anyway)
    if (parseFloat($("amp").value) > 4) { $("amp").value = "2.5"; $("ampVal").textContent = "2.5"; }
    if (currentFreq() > 0.012) setFreq(0.012);
  }
  setStatus(swim
    ? "swim mode: joints are actuated with a traveling wave, the fish propels itself"
    : "tethered mode: the head oscillates in a background current");
}

// Reads the current method parameter (int for count methods, float otherwise).
function currentParam() {
  const m = METHODS.find(x => x[0] === $("method").value);
  return m[2] === "count" ? parseInt($("param").value, 10) : parseFloat($("param").value);
}

// Shows the freq slider value in the text box. The amp and freq sliders have
// INDEPENDENT ranges and dragging one never moves the other; if the amp * freq
// combination exceeds the stability envelope the backend clamps freq down, and
// the real value is shown in the diagnostics panel.
function onFreqInput() {
  $("freqVal").value = sliderToFreq($("freq").value).toFixed(4);
}

// Switches the simulation resolution (low/medium/high lattice sizes).
function setResolution(res) {
  if (!RESOLUTIONS[res]) return;
  gstate.res = res;
  for (const [key, id] of Object.entries(RES_BUTTONS)) $(id).classList.toggle("active", key === res);
  const r = RESOLUTIONS[res];
  setStatus("resolution: " + res + " (" + r.nx + " × " + r.ny + " lattice)");
}

// Loads the file list (and query-param overrides), then starts the simulation.
async function loadFiles() {
  const meta = await api("/api/meta");
  let folder = $("folder").value.trim() || gstate.folder || meta.folder;
  if (qp.get("folder")) folder = qp.get("folder");
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
  const dirs = $("dirs");
  dirs.innerHTML = "";
  for (const d of ["..", ...res.dirs]) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = d;
    chip.onclick = () => {
      $("folder").value = d === ".." ? res.folder.replace(/[\\/][^\\/]*$/, "") : res.folder + "\\" + d;
      loadFiles();
    };
    dirs.appendChild(chip);
  }
  if (qp.get("file") && res.files.indexOf(qp.get("file")) >= 0) sel.value = qp.get("file");
  if (qp.get("method")) { $("method").value = qp.get("method"); onMethodChange(); }
  if (qp.get("param")) { $("param").value = qp.get("param"); $("paramVal").textContent = qp.get("param"); }
  if (qp.get("flow")) { $("flow").value = qp.get("flow"); $("flowVal").textContent = qp.get("flow"); }
  if (qp.get("steps")) { $("steps").value = qp.get("steps"); $("stepsVal").textContent = qp.get("steps"); }
  if (qp.get("amp")) { $("amp").value = qp.get("amp"); $("ampVal").textContent = qp.get("amp"); }
  if (qp.get("freq")) setFreq(parseFloat(qp.get("freq")));
  if (qp.get("cycles")) { $("cycles").value = qp.get("cycles"); $("cyclesVal").textContent = qp.get("cycles"); }
  if (qp.get("mode")) { $("simMode").value = qp.get("mode"); }
  if (qp.get("res")) setResolution(qp.get("res"));
  onModeChange();
  onFreqInput();
  setStatus(res.files.length + " file(s) in " + res.folder);
  if (res.files.length) run();
}

// Requests a simulation for the selected file/method and renders its frames.
async function run() {
  const file = $("file").value;
  if (!file) { setStatus("select a file first"); return; }
  const swim = gstate.mode === "swim";
  const res = RESOLUTIONS[gstate.res];
  const q = "folder=" + encodeURIComponent(gstate.folder) +
            "&file=" + encodeURIComponent(file) +
            "&method=" + encodeURIComponent($("method").value) +
            "&param=" + currentParam() +
            "&mode=" + encodeURIComponent(gstate.mode) +
            "&flow=" + parseFloat($("flow").value) +
            "&amp=" + parseFloat($("amp").value) +
            "&freq=" + currentFreq() +
            "&cycles=" + parseFloat($("cycles").value) +
            "&steps=" + parseInt($("steps").value, 10) +
            "&nx=" + res.nx +
            "&ny=" + res.ny;
  try {
    const s = await pollJob("/api/fluid/simulate?" + q, swim
      ? "generating joints and running free-swim simulation ..."
      : "generating joints and running fluid simulation ...");
    gstate.sim = s;
    gstate.frame = 0;
    gstate.playDir = 1;
    $("frame").max = s.frames.length - 1;
    $("frame").value = 0;
    $("frameVal").textContent = "0 / " + s.frames.length;
    setStatus("simulation ready: " + s.count + " joints, body " + s.total_cm + " cm, " +
              s.frames.length + " frames (" + s.steps + " sim steps)" +
              (s.metrics ? " - swim metrics in the diagnostics panel" : ""));
    drawFrame();
    drawFlowChart();
  } catch (e) { setStatus("error: " + e.message); }
}

/* ---------------- drawing ---------------- */

// Flow-speed colormap stops (dark blue -> teal -> yellow).
const V0 = [0.27, 0.00, 0.33], V1 = [0.13, 0.58, 0.55], V2 = [0.99, 0.85, 0.56];
// Maps a 0..1 speed fraction to an RGB color on the two-stop gradient.
function speedColor(t) {
  const a = t < 0.5 ? V0 : V1, b = t < 0.5 ? V1 : V2, f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return [0, 1, 2].map(i => Math.round((a[i] + (b[i] - a[i]) * f) * 255));
}

// Draws the current frame: flow field, rest/body chains, joints, head, stats.
function drawFrame() {
  const s = gstate.sim;
  const cv = $("sim");
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  g.fillStyle = "#10141a";
  g.fillRect(0, 0, w, h);
  if (!s) return;
  const fr = s.frames[gstate.frame];
  if (!fr) return;

  const ncols = s.xs.length, nrows = s.ys.length;
  const cw = w / ncols, ch = h / nrows;
  const maxSpeed = 0.05;
  for (let yi = 0; yi < nrows; yi++) {
    const row = fr.u[yi];
    const y = h - (yi + 1) * ch;
    for (let xi = 0; xi < ncols; xi++) {
      const t = Math.min(1, row[xi] / maxSpeed);
      const c = speedColor(t);
      g.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      g.fillRect(xi * cw, y, Math.ceil(cw), Math.ceil(ch));
    }
  }

  const PX = (x) => x / s.nx * w, PY = (y) => h - y / s.ny * h;

  g.strokeStyle = "rgba(255,255,255,0.5)"; g.lineWidth = 1.2; g.setLineDash([4, 4]);
  g.beginPath();
  s.rest.forEach((p, i) => { const x = PX(p[0]), y = PY(p[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
  g.stroke();
  g.setLineDash([]);

  // commanded wave ghost (free-swim mode): what the servos were asking for
  if (fr.target) {
    g.strokeStyle = "rgba(53,224,224,0.8)"; g.lineWidth = 1.4; g.setLineDash([2, 3]);
    g.beginPath();
    fr.target.forEach((p, i) => { const x = PX(p[0]), y = PY(p[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
    g.setLineDash([]);
  }

  g.strokeStyle = "#ff4d4d"; g.lineWidth = 2.5;
  g.beginPath();
  fr.body.forEach((p, i) => { const x = PX(p[0]), y = PY(p[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
  g.stroke();

  // joints at the actual articulation points of the deformed body this frame
  for (const p of (fr.joints || s.joint_pts)) {
    g.beginPath(); g.arc(PX(p[0]), PY(p[1]), 4, 0, Math.PI * 2);
    g.fillStyle = "#2ee62e"; g.fill();
    g.lineWidth = 1.2; g.strokeStyle = "#0b0f14"; g.stroke();
  }
  const hp = fr.body[0];
  g.beginPath(); g.arc(PX(hp[0]), PY(hp[1]), 5, 0, Math.PI * 2);
  g.fillStyle = "#ffb733"; g.fill();
  g.lineWidth = 1.2; g.strokeStyle = "#0b0f14"; g.stroke();

  const simTime = gstate.frame * s.frame_stride * s.sim_dt;
  let diag =
    `<div><span>lattice</span><b>${s.nx} × ${s.ny} cells</b></div>` +
    `<div><span>body</span><b>${s.body_kind === "traced" ? "traced (unordered data)" : "segments"}</b></div>` +
    `<div><span>joints</span><b>${s.count}</b></div>` +
    `<div><span>body length</span><b>${s.total_cm} cm</b></div>` +
    `<div><span>frame</span><b>${gstate.frame} / ${s.frames.length - 1}</b></div>` +
    `<div><span>sim time</span><b>${fmt(simTime)} s</b></div>` +
    `<div><span>flow speed</span><b>${fr.flow.toFixed(5)}</b></div>` +
    `<div><span>target flow</span><b>${s.flow_target !== undefined ? s.flow_target : "0.035"}</b></div>` +
    `<div><span>${s.mode === "swim" ? "tail" : "head"} amp</span><b>${s.osc_amp !== undefined ? s.osc_amp : 2}</b></div>` +
    `<div><span>wave freq</span><b>${s.osc_freq !== undefined ? s.osc_freq : 0.008}</b></div>`;
  if (s.metrics) {
    const m = s.metrics;
    diag +=
      `<div><span>swim speed</span><b>${fmt(m.speed)} cells/step</b></div>` +
      `<div><span>speed</span><b>${fmt(m.speed_bl)} body lengths/s</b></div>` +
      `<div><span>thrust</span><b>${fmt(m.thrust)}</b></div>` +
      `<div><span>efficiency</span><b>${fmt(m.efficiency)}</b></div>` +
      `<div><span>cost (power/speed)</span><b>${fmt(m.cost)}</b></div>` +
      `<div><span>Strouhal</span><b>${fmt(m.strouhal)}</b></div>` +
      `<div><span>tail amplitude</span><b>${fmt(m.tail_amp)} cells</b></div>` +
      `<div><span>tracking RMS</span><b>${fmt(m.track_rms)} rad</b></div>`;
  }
  $("diag").innerHTML = diag;
}

// Draws the flow-speed-over-time line chart with the current frame marked.
function drawFlowChart() {
  const s = gstate.sim;
  const cv = $("flowChart");
  if (!s) return;
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  const T = themeColors();
  g.clearRect(0, 0, w, h);
  const flows = s.frames.map(f => f.flow);
  const max = Math.max(...flows, 0.01);
  g.strokeStyle = T.grid; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, h - 2); g.lineTo(w, h - 2); g.stroke();
  g.strokeStyle = "#1f77b4"; g.lineWidth = 1.6;
  g.beginPath();
  flows.forEach((v, i) => {
    const x = i / (flows.length - 1) * w;
    const y = h - 3 - (v / max) * (h - 8);
    if (i) g.lineTo(x, y); else g.moveTo(x, y);
  });
  g.stroke();
  const i = gstate.frame;
  g.strokeStyle = T.frameLine; g.lineWidth = 1;
  g.beginPath(); g.moveTo(i / (flows.length - 1) * w, 0); g.lineTo(i / (flows.length - 1) * w, h); g.stroke();
  g.fillStyle = T.text; g.font = "10px system-ui";
  g.fillText("flow " + flows[i].toFixed(4), 4, 10);
  g.fillText(fmt(max), w - 34, 10);
}

/* ---------------- compare panel ---------------- */

// Runs the free-swim comparison of every ticked generation method.
async function runCompare() {
  const ids = [];
  for (const w of $("methodList").children) {
    const cb = w.children && w.children[0];
    if (cb && cb.checked && cb.value) ids.push(cb.value);
  }
  if (!ids.length) { $("compareStatus").textContent = "tick at least one method"; return; }
  const res = RESOLUTIONS[gstate.res];
  const q = "folder=" + encodeURIComponent(gstate.folder) +
            "&file=" + encodeURIComponent($("file").value) +
            "&methods=" + encodeURIComponent(ids.join(",")) +
            "&threshold=" + parseFloat($("cmpThreshold").value) +
            "&segments=" + parseInt($("cmpSegments").value, 10) +
            "&flow=" + parseFloat($("flow").value) +
            "&amp=" + parseFloat($("amp").value) +
            "&freq=" + currentFreq() +
            "&cycles=" + parseFloat($("cycles").value) +
            "&steps=" + Math.min(parseInt($("steps").value, 10), 1600) +
            "&nx=" + res.nx +
            "&ny=" + res.ny;
  $("btnCompare").disabled = true;
  try {
    const result = await pollJob("/api/fluid/compare?" + q,
      "running " + ids.length + " free-swim simulation(s) ...");
    gstate.compare = result;
    renderCompare(result);
    const ok = result.runs.filter(r => !r.error).length;
    $("compareStatus").textContent = "compare done: " + ok + " run(s) in " +
      result.steps + " steps at " + result.osc_amp + " cells tail amplitude, " +
      result.osc_freq + " cycles/step";
  } catch (e) {
    $("compareStatus").textContent = "compare failed: " + e.message;
  }
  $("btnCompare").disabled = false;
}

// Renders the compare results: table with best-value highlighting, two
// time-series charts, and one selectable run card per method.
function renderCompare(res) {
  const runs = res.runs.filter(r => !r.error);
  const colors = {};
  runs.forEach((r, i) => { colors[r.id] = PALETTE[i % PALETTE.length]; });

  // results table
  const cols = [
    { key: "speed", label: "speed<br>(cells/step)", dir: "up", digits: 5 },
    { key: "speed_bl", label: "speed<br>(BL/s)", dir: "up", digits: 5 },
    { key: "thrust", label: "thrust", dir: "up", digits: 5 },
    { key: "power", label: "power", dir: "down", digits: 5 },
    { key: "efficiency", label: "efficiency", dir: "up", digits: 4 },
    { key: "cost", label: "cost", dir: "down", digits: 4 },
    { key: "strouhal", label: "Strouhal", dir: "up", digits: 3 },
    { key: "tail_amp", label: "tail amp", dir: "up", digits: 2 },
    { key: "track_rms", label: "track RMS", dir: "down", digits: 4 },
  ];
  const best = {};
  for (const c of cols) {
    const vals = runs.map(r => r.metrics[c.key]).filter(v => isFinite(v));
    if (!vals.length) continue;
    best[c.key] = c.dir === "up" ? Math.max(...vals) : Math.min(...vals);
  }
  let html = "<table><thead><tr><th>algorithm</th><th>joints</th>" +
    cols.map(c => "<th>" + c.label + "</th>").join("") +
    "<th>finite</th></tr></thead><tbody>";
  for (const r of runs) {
    const m = r.metrics;
    html += "<tr><td><span style='color:" + colors[r.id] + ";font-weight:600;'>" + r.label + "</span></td>" +
      "<td>" + r.count + "</td>";
    for (const c of cols) {
      const v = m[c.key];
      const cls = (isFinite(v) && best[c.key] !== undefined && Math.abs(v - best[c.key]) < 1e-12) ? " class='best'" : "";
      html += "<td" + cls + ">" + (isFinite(v) ? v.toFixed(c.digits) : "-") + "</td>";
    }
    html += "<td>" + (m.finite ? "yes" : "<span class='err'>no</span>") + "</td></tr>";
  }
  html += "</tbody></table>";
  for (const r of res.runs.filter(r => r.error)) {
    html += "<div class='err' style='margin-top:6px;font-size:12px;'>" + r.label +
      ": <b>error</b> - " + r.error + "</div>";
  }
  $("cmpTable").innerHTML = html;

  drawCompareChart($("cmpSpeed"), runs, colors, "speed", "speed (cells/step)");
  drawCompareChart($("cmpThrust"), runs, colors, "thrust", "thrust");

  // legend
  let legend = "";
  runs.forEach((r) => {
    legend += "<span><span class='dot' style='background:" + colors[r.id] + "'></span>" +
      r.label + " (" + r.count + " joints)</span>";
  });
  $("runLegend").innerHTML = legend;

  // run cards
  const grid = $("cmpRuns");
  grid.innerHTML = "";
  runs.forEach((r) => {
    const card = document.createElement("div");
    card.className = "runCard";
    const cv = document.createElement("canvas");
    const lbl = document.createElement("div");
    lbl.className = "runLabel";
    lbl.textContent = r.label;
    const stats = document.createElement("div");
    stats.className = "runStats";
    const m = r.metrics;
    stats.textContent = "U " + fmt(m.speed) + " | \u03b7 " + fmt(m.efficiency) +
      " | " + r.count + " joints";
    card.appendChild(cv);
    card.appendChild(lbl);
    card.appendChild(stats);
    card.onclick = () => selectCompareRun(r, card);
    grid.appendChild(card);
    drawRunThumb(cv, r, colors[r.id]);
  });

  // auto-inspect the fastest swimmer in the main viewer
  const ordered = runs.slice().sort((a, b) => (b.metrics.speed || 0) - (a.metrics.speed || 0));
  if (ordered.length) {
    selectCompareRun(ordered[0], grid.children[runs.indexOf(ordered[0])]);
    setStatus("compare done - showing the fastest run in the main viewer (" +
      ordered[0].label + "); click a run card to switch");
  }
}

// Draws one multi-series time chart (speed / thrust per run).
function drawCompareChart(cv, runs, colors, key, label) {
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  const T = themeColors();
  g.clearRect(0, 0, w, h);
  const series = runs.map(r => r.series[key]).filter(a => a && a.length);
  if (!series.length) return;
  const all = series.flat();
  let lo = Math.min(...all, 0), hi = Math.max(...all, 0.001);
  if (lo === hi) { lo -= 1; hi += 1; }
  const x = (i, n) => i / Math.max(1, n - 1) * w;
  const y = (v) => h - 4 - (v - lo) / (hi - lo) * (h - 16);
  g.strokeStyle = T.grid; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, y(0)); g.lineTo(w, y(0)); g.stroke();
  let idx = 0;
  for (const r of runs) {
    const s = r.series[key];
    if (!s || !s.length) continue;
    g.strokeStyle = colors[r.id]; g.lineWidth = 1.6;
    g.beginPath();
    s.forEach((v, i) => { if (i) g.lineTo(x(i, s.length), y(v)); else g.moveTo(x(0, s.length), y(v)); });
    g.stroke();
    idx++;
  }
  g.fillStyle = T.text; g.font = "10px system-ui";
  g.fillText(label, 4, 10);
  g.fillText(fmt(hi), w - 40, 10);
  g.fillText(fmt(lo), w - 40, h - 4);
}

// Draws a small thumbnail of the run's last recorded frame.
function drawRunThumb(cv, run, color) {
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  g.fillStyle = "#10141a";
  g.fillRect(0, 0, w, h);
  const fr = run.frames[run.frames.length - 1];
  if (!fr) return;
  const PX = (x) => x / run.nx * w, PY = (y) => h - y / run.ny * h;
  g.strokeStyle = color; g.lineWidth = 2;
  g.beginPath();
  fr.body.forEach((p, i) => { const x = PX(p[0]), y = PY(p[1]); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
  g.stroke();
  g.beginPath(); g.arc(PX(fr.body[0][0]), PY(fr.body[0][1]), 3, 0, Math.PI * 2);
  g.fillStyle = "#ffb733"; g.fill();
}

// Loads a compare run into the main viewer (same shape as /api/fluid/simulate).
function selectCompareRun(run, card) {
  const res = gstate.compare;
  if (!res) return;
  run.flow_target = res.flow_target;
  run.osc_amp = res.osc_amp;
  run.osc_freq = res.osc_freq;
  run.mode = "swim";
  run.method = run.id;
  gstate.sim = run;
  gstate.frame = 0;
  gstate.playDir = 1;
  $("frame").max = run.frames.length - 1;
  $("frame").value = 0;
  $("frameVal").textContent = "0 / " + run.frames.length;
  for (const c of $("cmpRuns").children) c.classList.remove("selected");
  if (card) card.classList.add("selected");
  drawFrame();
  drawFlowChart();
}

/* ---------------- playback ---------------- */

// Starts/stops the frame animation (loops, or rewinds when unchecked).
function togglePlay() {
  gstate.playing = !gstate.playing;
  $("play").textContent = gstate.playing ? "Pause" : "Play";
  if (gstate.playing) {
    gstate.timer = setInterval(() => {
      const n = gstate.sim ? gstate.sim.frames.length : 0;
      if (!n) return;
      [gstate.frame, gstate.playDir] = advancePlay(gstate.frame, n, $("loop").checked, gstate.playDir);
      $("frame").value = gstate.frame;
      $("frameVal").textContent = gstate.frame + " / " + (n - 1);
      drawFrame();
    }, 40);
  } else if (gstate.timer) { clearInterval(gstate.timer); gstate.timer = null; }
}

/* ---------------- wiring ---------------- */

$("btnFolder").onclick = loadFiles;
$("folder").addEventListener("keydown", (e) => { if (e.key === "Enter") loadFiles(); });
$("run").onclick = run;
$("play").onclick = togglePlay;
$("btnCompare").onclick = runCompare;
$("method").addEventListener("change", onMethodChange);
$("simMode").addEventListener("change", onModeChange);
$("resLow").addEventListener("click", () => setResolution("low"));
$("resMed").addEventListener("click", () => setResolution("medium"));
$("resHigh").addEventListener("click", () => setResolution("high"));
$("resMax").addEventListener("click", () => setResolution("max"));
$("flow").addEventListener("input", () => { $("flowVal").textContent = $("flow").value; });
$("amp").addEventListener("input", () => { $("ampVal").textContent = $("amp").value; });
$("freq").addEventListener("input", onFreqInput);
$("freqVal").addEventListener("input", () => {
  const f = parseFloat($("freqVal").value);
  if (isFinite(f)) $("freq").value = freqToSlider(f);
});
$("steps").addEventListener("input", () => { $("stepsVal").textContent = $("steps").value; });
$("param").addEventListener("input", () => { $("paramVal").textContent = $("param").value; });
$("cycles").addEventListener("input", () => { $("cyclesVal").textContent = $("cycles").value; });
$("cmpThreshold").addEventListener("input", () => { $("cmpThresholdVal").textContent = $("cmpThreshold").value; });
$("cmpSegments").addEventListener("input", () => { $("cmpSegmentsVal").textContent = $("cmpSegments").value; });
$("frame").addEventListener("input", (e) => {
  gstate.frame = parseInt(e.target.value, 10);
  $("frameVal").textContent = gstate.frame + " / " + (gstate.sim ? gstate.sim.frames.length - 1 : 0);
  drawFrame();
});

/* ---------------- dark mode ---------------- */

// Redraws every canvas after a theme change (without resetting the selected
// compare run / frame).
function redrawAll() {
  drawFrame();
  drawFlowChart();
  if (gstate.compare) {
    const runs = gstate.compare.runs.filter(r => !r.error);
    const colors = {};
    runs.forEach((r, i) => { colors[r.id] = PALETTE[i % PALETTE.length]; });
    drawCompareChart($("cmpSpeed"), runs, colors, "speed", "speed (cells/step)");
    drawCompareChart($("cmpThrust"), runs, colors, "thrust", "thrust");
  }
}

// Labels the toggle with the mode you will switch TO.
function updateThemeBtn() {
  const dark = (document.documentElement.dataset.theme || "") === "dark";
  $("themeToggle").textContent = dark ? "Light" : "Dark";
}

// Flips the theme, persists it and redraws everything.
function toggleTheme() {
  const dark = (document.documentElement.dataset.theme || "") === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("fishseg-theme", document.documentElement.dataset.theme); } catch (e) {}
  updateThemeBtn();
  redrawAll();
}

$("themeToggle").onclick = toggleTheme;
updateThemeBtn();
window.addEventListener("resize", () => { drawFrame(); drawFlowChart(); });

// Boot: load the method list, then the file list (which auto-runs the sim).
(async function init() {
  await loadMethods();
  METHODS = getMethods();
  fillMethodOptions();
  fillCompareMethods();
  onMethodChange();
  loadFiles().catch(e => setStatus("error: " + e.message));
})();
