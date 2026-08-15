/* Main viewer page (web/index.html): an interactive canvas dashboard for fish
   midline joint generation. Each "card" shows one data file - midline plot
   (2D/3D), per-frame error chart, segment-length table/heatmap - with sliders
   for the generation method parameter, playback, linking between cards and a
   compare panel. Session save/load and PNG/CSV export are included. */

"use strict";
import { advancePlay, buildCloud, bounds2d, downloadBlob, downloadCsv, esc, fmt, isPad, makeMap, setupCanvas, themeColors } from "./helpers.js";
import { getMethods, loadMethods } from "./methods.js";
const $ = (id) => document.getElementById(id);

// method list comes from the backend /api/methods (see js/methods.js)
let METHODS = getMethods();

const gstate = { folder: "" };
let nextCardId = 1;
const cards = [];
const links = { frame: new Set(), threshold: new Set(), segments: new Set(), data: new Set(), play: new Set(), method: new Set() };

const CMP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"];
// Stable color for a card in the compare charts.
function cmpColor(c) { return CMP_COLORS[c.id % CMP_COLORS.length]; }

// GETs a JSON API endpoint and throws on non-2xx responses.
async function api(path) {
  const resp = await fetch(path);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

// Shows a message in the status bar.
function setStatus(msg) { $("status").textContent = msg; }

/* ---------------- card creation ---------------- */

// HTML skeleton of a card: header, plot canvas, controls, stats panels.
function cardTemplate() {
  return `
  <div class="cardHead">
    <span class="cardTitle"></span>
    <button class="linkBtn" data-var="data" title="link data source">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button>
    <button class="linkAll" title="select / clear all links">link all</button>
    <span class="cardMeta"></span>
    <button class="masterSwitch" title="master link control: ON">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
    </button>
    <button class="pngBtn" title="download plot as PNG">png</button>
    <button class="simBtn" title="send this container to the fluid simulation page">sim</button>
    <button class="remove" title="remove view">&#10005;</button>
  </div>
  <div class="plotWrap">
    <canvas class="cv"></canvas>
    <div class="tip"></div>
  </div>
  <div class="cardLog"></div>
  <div class="cardControls">
    <select class="fileSel" title="data source"></select>
    <label class="v"><button class="linkBtn" data-var="method" title="link algorithm">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button> algorithm</label>
    <select class="method" title="joint-generation algorithm for this view"></select>
    <label class="v"><button class="linkBtn" data-var="threshold" title="link threshold">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button> threshold</label>
    <input type="range" class="threshold" min="0.05" max="20" step="0.05" value="0.5"
           title="error threshold (cm) for the error-based generation methods">
    <span class="thresholdVal">0.5</span>
    <label class="v"><button class="linkBtn" data-var="segments" title="link segments">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button> segments</label>
    <input type="range" class="segments" min="2" max="40" step="1" value="10"
           title="number of joint segments (count-based methods)">
    <span class="segmentsVal">10</span>
    <span class="modeBar">
      <button class="modeBtn active" data-mode="2d" title="2D plot of the midline">2D</button>
      <button class="modeBtn" data-mode="3d" title="3D plot (drag to rotate, scroll to zoom)">3D</button>
    </span>
    <label><input type="checkbox" class="ghost" title="overlay all other frames faintly in the background"> ghost</label>
    <label><input type="checkbox" class="echo" checked title="fade recently visited frames behind the current one"> echo</label>
    <label><input type="checkbox" class="loop" title="checked: loop from the end to the start / unchecked: play forward then rewind"> loop</label>
    <button class="playBtn" title="play / pause the frame animation">Play</button>
    <label class="v"><button class="linkBtn" data-var="play" title="link play">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button></label>
    <label class="v"><button class="linkBtn" data-var="frame" title="link frame">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
    </button> frame</label>
    <input type="range" class="frame" min="0" max="0" value="0"
           title="current frame / time step shown on the plot">
    <span class="frameVal">0</span>
  </div>
  <div class="cardBody">
    <div class="row2">
      <div class="panel statsPanel">
        <h2>Generation result <button class="exportBtn" title="export as CSV">export</button></h2>
        <div class="stats"></div>
      </div>
      <div class="panel errPanel"><h2>Error per frame (click)</h2>
        <select class="errMetric" title="error metric">
          <option value="area">area</option>
          <option value="linear">linear</option>
        </select>
        <canvas class="errCv"></canvas>
      </div>
    </div>
    <div class="panel"><h2>Segment lengths
      <span class="seglenTabs">
        <button class="seglenMode active" data-mode="table" title="table of each segment's length per frame">table</button>
        <button class="seglenMode" data-mode="heatmap" title="heatmap of segment length over all frames">heatmap</button>
      </span>
    </h2>
      <div class="seglen"></div>
      <canvas class="segHmCv" style="display:none"></canvas>
    </div>
  </div>`;
}

// Fills a method <select> from the shared method list and picks the default.
function fillMethodOptions(sel) {
  sel.innerHTML = "";
  for (const [id, label] of METHODS) {
    const opt = document.createElement("option");
    opt.value = id; opt.textContent = label;
    sel.appendChild(opt);
  }
  sel.value = "grow_bs_area";
}

// Copies the global file <select> options into a card's file selector.
function fillFileOptions(sel) {
  sel.innerHTML = "";
  for (const opt of $("file").options) {
    const o = document.createElement("option");
    o.value = opt.value; o.textContent = opt.textContent;
    sel.appendChild(o);
  }
}

// Creates a card for a data file: builds the DOM, wires all events, loads data.
function createCard(file) {
  const id = nextCardId++;
  const host = document.createElement("section");
  host.className = "card";
  host.id = "card-" + id;
  host.innerHTML = cardTemplate();
  $("deck").appendChild(host);
  const empty = document.querySelector(".empty");
  if (empty) empty.remove();

  const q = (s) => host.querySelector(s);
  const card = {
    id,
    el: {
      root: host, title: q(".cardTitle"), meta: q(".cardMeta"), remove: q(".remove"),
      cv: q(".cv"), tip: q(".tip"), log: q(".cardLog"),
      fileSel: q(".fileSel"), method: q(".method"),
      link_data: q("[data-var='data']"), link_method: q("[data-var='method']"),
      link_play: q("[data-var='play']"), link_threshold: q("[data-var='threshold']"),
      link_segments: q("[data-var='segments']"), link_frame: q("[data-var='frame']"),
      linkAll: q(".linkAll"), masterSwitch: q(".masterSwitch"),
      threshold: q(".threshold"), thresholdVal: q(".thresholdVal"),
      segments: q(".segments"), segmentsVal: q(".segmentsVal"),
      modeBar: q(".modeBar"), ghost: q(".ghost"), echo: q(".echo"),
      loop: q(".loop"),
      play: q(".playBtn"), frame: q(".frame"), frameVal: q(".frameVal"),
      stats: q(".stats"), errCv: q(".errCv"), errMetric: q(".errMetric"),
      exportBtn: q(".exportBtn"), pngBtn: q(".pngBtn"), simBtn: q(".simBtn"),
      seglen: q(".seglen"), segHmCv: q(".segHmCv"), seglenMode: q(".seglenMode"),
    },
    data: null, gen: null, file, frame: 0, playing: false, playDir: 1, mode: "2d",
    yaw: 0.6, pitch: 0.55, zoom: 1, echoK: 8, history: [], cloud: [],
    hist: new Map(), errMetric: "area", segMode: "table", segHm: [],
    redrawPending: false, mousedown: null, animTimer: null, debounce: null, map2d: null,
    master: true,
  };
  card.ctx = card.el.cv.getContext("2d");
  card.ectx = card.el.errCv.getContext("2d");

  fillMethodOptions(card.el.method);
  fillFileOptions(card.el.fileSel);
  card.el.fileSel.value = file;

  card.el.remove.addEventListener("click", () => removeCard(card));
  card.el.link_data.addEventListener("click", () => tickLink(card, "data"));
  card.el.link_method.addEventListener("click", () => tickLink(card, "method"));
  card.el.link_play.addEventListener("click", () => tickLink(card, "play"));
  card.el.link_threshold.addEventListener("click", () => tickLink(card, "threshold"));
  card.el.link_segments.addEventListener("click", () => tickLink(card, "segments"));
  card.el.link_frame.addEventListener("click", () => tickLink(card, "frame"));
  card.el.linkAll.addEventListener("click", () => selectAllLinks(card));
  card.el.masterSwitch.addEventListener("click", () => {
    card.master = !card.master;
    card.el.masterSwitch.classList.toggle("off", !card.master);
    card.el.masterSwitch.title = "master link control: " + (card.master ? "ON" : "OFF");
  });
  card.el.fileSel.addEventListener("change", () => cardLoadData(card, card.el.fileSel.value));
  card.el.method.addEventListener("change", () => {
    card.hist.clear();
    refreshMaster();
    queueGenerate(card);
  });
  card.el.threshold.addEventListener("input", (e) => {
    card.el.thresholdVal.textContent = e.target.value;
    if (links.threshold.has(card)) refreshMaster();
    queueGenerate(card);
  });
  card.el.segments.addEventListener("input", (e) => {
    card.el.segmentsVal.textContent = e.target.value;
    if (links.segments.has(card)) refreshMaster();
    queueGenerate(card);
  });
  card.el.ghost.addEventListener("change", () => cardDraw(card));
  card.el.echo.addEventListener("change", () => cardDraw(card));
  card.el.loop.addEventListener("change", () => { card.playDir = 1; });
  card.el.modeBar.addEventListener("click", (e) => {
    const b = e.target.closest(".modeBtn");
    if (b && !b.disabled) setMode(card, b.dataset.mode);
  });
  card.el.play.addEventListener("click", () => togglePlay(card));
  card.el.frame.addEventListener("input", (e) => {
    card.frame = parseInt(e.target.value, 10);
    card.el.frameVal.textContent = card.frame + " / " + (card.data ? card.data.num_frames : 0);
    if (links.frame.has(card)) refreshMaster();
    cardDraw(card);
  });
  card.el.cv.addEventListener("mousemove", (e) => onMove(card, e));
  card.el.cv.addEventListener("mouseleave", () => { card.el.tip.style.display = "none"; });
  card.el.cv.addEventListener("mousedown", (e) => { if (card.mode === "3d") card.mousedown = { x: e.clientX, y: e.clientY }; });
  card.el.cv.addEventListener("wheel", (e) => {
    if (card.mode !== "3d") return;
    e.preventDefault();
    card.zoom *= e.deltaY < 0 ? 1.12 : 0.89;
    card.zoom = Math.max(0.2, Math.min(8, card.zoom));
    request3dRedraw(card);
  }, { passive: false });
  card.el.errCv.addEventListener("mousemove", (e) => onErrMove(card, e));
  card.el.errCv.addEventListener("click", (e) => onErrClick(card, e));
  card.el.errMetric.addEventListener("change", (e) => { card.errMetric = e.target.value; cardDrawErr(card); });
  card.el.exportBtn.addEventListener("click", () => exportCardCsv(card));
  card.el.pngBtn.addEventListener("click", () => exportCardPng(card));
  card.el.simBtn.addEventListener("click", () => {
    if (!card.data) return;
    const m = METHODS.find(x => x[0] === card.el.method.value);
    const param = m[2] === "count"
      ? (parseInt(card.el.segments.value, 10) || 10)
      : (parseFloat(card.el.threshold.value) || 0.5);
    const q = new URLSearchParams({
      folder: gstate.folder,
      file: card.data.file,
      method: card.el.method.value,
      param: String(param),
    });
    window.open("/fluid?" + q.toString(), "_blank");
  });
  card.el.seglenMode.addEventListener("click", () => {
    const showTable = card.segMode !== "table";
    card.segMode = showTable ? "table" : "heatmap";
    card.el.seglenMode.classList.toggle("active", showTable);
    card.el.seglen.style.display = showTable ? "" : "none";
    card.el.segHmCv.style.display = showTable ? "none" : "";
    if (!showTable) cardDrawSegHm(card); else cardDrawSegments(card);
  });

  cards.push(card);
  card.loadPromise = cardLoadData(card, file);
  return card;
}

// Removes a card, its timers, links and compare contribution.
function removeCard(card) {
  if (card.animTimer) clearInterval(card.animTimer);
  card.el.root.remove();
  const i = cards.indexOf(card);
  if (i >= 0) cards.splice(i, 1);
  for (const v in links) links[v].delete(card);
  refreshMaster();
  drawCompare();
  if (!cards.length)
    $("deck").innerHTML = '<div class="empty">Select a file above and click &quot;+ Add view&quot; to start exploring.</div>';
}

/* ---------------- linking / master window ---------------- */

// Toggles a card's membership in a link set and its button highlight.
function tickLink(card, v) {
  const set = links[v];
  if (set.has(card)) set.delete(card); else set.add(card);
  const btn = card.el["link_" + v];
  if (btn) btn.classList.toggle("linked", set.has(card));
  refreshMaster();
}

// Toggles all six link types for a card at once.
function selectAllLinks(card) {
  const vars = ["frame", "threshold", "segments", "data", "play", "method"];
  const all = vars.every(v => links[v].has(card));
  for (const v of vars) {
    if (all) links[v].delete(card); else links[v].add(card);
    const btn = card.el["link_" + v];
    if (btn) btn.classList.toggle("linked", !all);
  }
  refreshMaster();
}

// First card linked for a variable (used to drive the master controls).
function firstLinked(v) { return links[v].values().next().value; }

// Syncs the master window rows with the current link state and values.
function refreshMaster() {
  const any = Object.values(links).some(s => s.size > 0);
  $("master").classList.toggle("hidden", !any);
  for (const v of ["frame", "threshold", "segments", "data", "play", "method"]) {
    const row = $("mRow-" + v);
    const has = links[v].size > 0;
    row.classList.toggle("hiddenRow", !has);
    if (!has) continue;
    const c = firstLinked(v);
    const mSlider = row.querySelector(".mSlider");
    const mNum = row.querySelector(".mNum");
    if (v === "frame") {
      const max = Math.max(0, ...[...links.frame].filter(c2 => c2.master && c2.data).map(c2 => c2.data.frames.length - 1));
      mSlider.max = max;
      mSlider.value = Math.min(c.frame, max);
      mNum.value = mSlider.value;
    } else if (v === "threshold") {
      mSlider.value = c.el.threshold.value;
      mNum.value = c.el.threshold.value;
    } else if (v === "segments") {
      mSlider.value = c.el.segments.value;
      mNum.value = c.el.segments.value;
    } else if (v === "data") {
      const sel = row.querySelector(".mFile");
      sel.innerHTML = "";
      for (const opt of $("file").options) {
        const o = document.createElement("option");
        o.value = opt.value; o.textContent = opt.textContent;
        sel.appendChild(o);
      }
      sel.value = c.data ? c.data.file : c.file;
    } else if (v === "play") {
      row.querySelector(".mPlayBtn").textContent = c.playing ? "Pause all" : "Play all";
    } else if (v === "method") {
      const sel = row.querySelector(".mMethod");
      sel.innerHTML = "";
      for (const [id, label] of METHODS) {
        const o = document.createElement("option");
        o.value = id; o.textContent = label;
        sel.appendChild(o);
      }
      sel.value = c.el.method.value;
    }
  }
}

// Applies a master-window value to every linked (and enabled) card.
function masterApply(v, val) {
  if (v === "frame") {
    for (const c of links.frame) {
      if (!c.master || !c.data) continue;
      const m = c.data.frames.length - 1;
      c.frame = Math.max(0, Math.min(Math.round(val), m));
      c.el.frame.value = c.frame;
      c.el.frameVal.textContent = c.frame + " / " + c.data.num_frames;
      cardDraw(c);
    }
  } else if (v === "threshold") {
    for (const c of links.threshold) {
      if (!c.master) continue;
      c.el.threshold.value = val;
      c.el.thresholdVal.textContent = fmt(val);
      queueGenerate(c);
    }
  } else if (v === "segments") {
    for (const c of links.segments) {
      if (!c.master) continue;
      const sv = Math.round(val);
      c.el.segments.value = sv;
      c.el.segmentsVal.textContent = sv;
      queueGenerate(c);
    }
  } else if (v === "data") {
    for (const c of links.data) {
      if (c.master) cardLoadData(c, val);
    }
  } else if (v === "play") {
    for (const c of links.play) {
      if (c.master) setPlaying(c, !!val);
    }
    refreshMaster();
  } else if (v === "method") {
    for (const c of links.method) {
      if (!c.master) continue;
      c.el.method.value = val;
      c.hist.clear();
      queueGenerate(c);
    }
  }
}

// Wires the master window sliders/inputs and its drag-to-move header.
(function setupMasterEvents() {
  document.querySelectorAll("#master .mSlider").forEach(sl => {
    sl.addEventListener("input", () => {
      const row = sl.closest(".mRow");
      row.querySelector(".mNum").value = sl.value;
      masterApply(sl.dataset.var, parseFloat(sl.value));
    });
  });
  document.querySelectorAll("#master .mNum").forEach(n => {
    n.addEventListener("input", () => {
      const row = n.closest(".mRow");
      const val = parseFloat(n.value);
      row.querySelector(".mSlider").value = val;
      masterApply(n.dataset.var, val);
    });
  });
  const mfile = document.querySelector("#master .mFile");
  if (mfile) mfile.addEventListener("change", (e) => masterApply("data", e.target.value));
  const mplay = document.querySelector("#master .mPlayBtn");
  if (mplay) mplay.addEventListener("click", () => {
    const c = firstLinked("play");
    if (c) masterApply("play", !c.playing);
  });
  const mmethod = document.querySelector("#master .mMethod");
  if (mmethod) mmethod.addEventListener("change", (e) => masterApply("method", e.target.value));

  const m = $("master"), head = m.querySelector(".masterHead");
  let dragging = null;
  head.addEventListener("mousedown", (e) => {
    dragging = { dx: e.clientX - m.offsetLeft, dy: e.clientY - m.offsetTop };
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    m.style.left = (e.clientX - dragging.dx) + "px";
    m.style.top = (e.clientY - dragging.dy) + "px";
    m.style.right = "auto";
  });
  window.addEventListener("mouseup", () => { dragging = null; });
})();

/* ---------------- data loading / generation (per card) ---------------- */

// Loads a file's display data into a card, then generates joints for it.
async function cardLoadData(card, file) {
  card.el.log.textContent = "loading " + file + " ...";
  card.data = null; card.gen = null; card.frame = 0; card.playDir = 1; card.history = []; card.hist = new Map();
  try {
    const data = await api("/api/data?folder=" + encodeURIComponent(gstate.folder) +
                           "&file=" + encodeURIComponent(file));
    card.data = data;
    card.file = file;
    card.el.fileSel.value = file;
    card.cloud = buildCloud(data);
    card.el.title.textContent = data.name;
    card.el.meta.textContent = data.frames.length + "/" + data.num_frames + " frames · " + data.num_points + " pts";
    card.el.frame.max = data.frames.length - 1;
    card.el.frame.value = 0;
    card.el.frameVal.textContent = "0 / " + data.num_frames;
    const btn3d = card.el.modeBar.querySelector("[data-mode='3d']");
    btn3d.disabled = !data.has_z;
    if (!data.has_z && card.mode === "3d") setMode(card, "2d");
    card.el.log.textContent = "loaded " + data.name + " | " + data.num_frames + " frames (" +
                              data.frames.length + " shown), " + data.num_points + " points";
    await cardGenerate(card);
  } catch (e) { card.el.log.textContent = "error: " + e.message; }
}

// Runs the generation method for the card's current file/method/param.
async function cardGenerate(card) {
  if (!card.data) return;
  const method = card.el.method.value;
  const m = METHODS.find(x => x[0] === method);
  const param = m[2] === "count"
    ? (parseInt(card.el.segments.value, 10) || 10)
    : (parseFloat(card.el.threshold.value) || 0.5);
  card.el.log.textContent = "generating " + card.el.method.selectedOptions[0].textContent + " ...";
  try {
    const gen = await api("/api/generate?folder=" + encodeURIComponent(gstate.folder) +
                          "&file=" + encodeURIComponent(card.data.file) +
                          "&method=" + encodeURIComponent(method) +
                          "&param=" + param);
    card.gen = gen;
    const hkey = Math.round(gen.param * 10000) / 10000;
    card.hist.set(hkey, { param: gen.param, linear: gen.avg_linear, area: gen.avg_area, time: gen.time, joints: gen.count });
    cardComputeSegHm(card);
    card.frame = Math.min(card.frame, card.data.frames.length - 1);
    cardDraw(card); cardDrawErr(card); drawCompare();
    card.el.log.textContent = "generated in " + gen.time + "s | " + gen.count + " joints | avg linear " +
                              gen.avg_linear + " cm | avg area " + gen.avg_area + " cm^2";
  } catch (e) { card.el.log.textContent = "error: " + e.message; }
}

// Debounces a regeneration request (sliders fire many input events).
function queueGenerate(card) {
  clearTimeout(card.debounce);
  card.debounce = setTimeout(() => cardGenerate(card), 250);
}

// Switches a card between 2D and 3D rendering modes.
function setMode(card, m) {
  card.mode = m;
  card.el.modeBar.querySelectorAll(".modeBtn").forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  cardDraw(card);
}

// Starts or stops a card's frame animation (bounces at either end).
function setPlaying(card, playing) {
  card.playing = playing;
  card.el.play.textContent = playing ? "Pause" : "Play";
  if (playing) {
    card.animTimer = setInterval(() => {
      if (!card.data) return;
      [card.frame, card.playDir] = advancePlay(card.frame, card.data.frames.length,
                                               card.el.loop.checked, card.playDir);
      card.el.frame.value = card.frame;
      card.el.frameVal.textContent = card.frame + " / " + card.data.num_frames;
      pushHistory(card);
      cardDraw(card); cardDrawErr(card);
    }, 80);
  } else if (card.animTimer) { clearInterval(card.animTimer); card.animTimer = null; }
}

// Toggles playback for the card, or every play-linked card at once.
function togglePlay(card) {
  const next = !card.playing;
  const targets = links.play.has(card) ? [...links.play] : [card];
  for (const c of targets) setPlaying(c, next);
  refreshMaster();
}

// Records the current frame in the fading echo history.
function pushHistory(card) {
  card.history.push({ f: card.frame, age: card.echoK });
  for (const item of card.history) item.age -= 1;
  card.history = card.history.filter(item => item.age > 0);
  if (card.history.length > card.echoK) card.history.shift();
}

/* ---------------- 2D drawing ---------------- */

// Draws the current card state (2D or 3D) with the frame clamped in range.
function cardDraw(card) {
  if (!card.data) return;
  card.frame = Math.max(0, Math.min(card.frame, card.data.frames.length - 1));
  if (card.mode === "3d") cardDraw3d(card); else cardDraw2d(card);
}

// Strokes one frame's traced midline as a polyline with a given style.
function cardDrawFrameLine(card, f, color, alpha, width) {
  const d = card.data, tr = d.trace[f], pts = d.frames[f];
  const g = card.ctx;
  g.beginPath();
  let started = false;
  for (const row of tr) {
    const p = pts[row];
    const sx = card.map2d.x(p[0]), sy = card.map2d.y(p[1]);
    if (!started) { g.moveTo(sx, sy); started = true; }
    else g.lineTo(sx, sy);
  }
  g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width;
  g.stroke(); g.globalAlpha = 1;
}

// Joint positions (x, y) of the generation result at a given frame.
function cardJointPoints(card, f) {
  const d = card.data;
  if (!d || !card.gen) return [];
  const tr = d.trace[f], pts = d.frames[f];
  const pos = new Map(tr.map((r, i) => [r, i]));
  const out = [];
  for (const j of card.gen.joints) {
    const row = j[2];
    if (pos.has(row)) { const p = pts[row]; out.push([p[0], p[1]]); }
    else if (row < d.actual[f]) { const p = pts[row]; out.push([p[0], p[1]]); }
  }
  return out;
}

// Length of every joint segment at a frame (cm).
function cardSegLengths(card, f) {
  const jp = cardJointPoints(card, f);
  const out = [];
  for (let i = 0; i < jp.length - 1; i++) {
    const dx = jp[i + 1][0] - jp[i][0], dy = jp[i + 1][1] - jp[i][1];
    out.push(Math.sqrt(dx * dx + dy * dy));
  }
  return out;
}

// Precomputes the segment lengths of every frame for the heatmap.
function cardComputeSegHm(card) {
  card.segHm = [];
  if (!card.data || !card.gen) return;
  for (let f = 0; f < card.data.frames.length; f++) card.segHm.push(cardSegLengths(card, f));
}

// Draws the 2D plot: axes, ghosts/echoes, midline, joints, stats, segments.
function cardDraw2d(card) {
  const { w, h } = setupCanvas(card.el.cv);
  const g = card.ctx;
  g.clearRect(0, 0, w, h);
  const d = card.data;
  if (!d) return;
  card.map2d = makeMap(bounds2d(d), w, h);
  const T = themeColors();
  const { plotL, plotT, plotW, plotH } = card.map2d;
  const xAxisY = plotT + plotH;
  const yAxisX = plotL;

  g.strokeStyle = T.grid; g.lineWidth = 1; g.font = "11px system-ui";
  const ticks = 5;
  const b = bounds2d(d);
  g.save();
  g.beginPath(); g.rect(plotL, plotT, plotW, plotH); g.clip();
  for (let i = 0; i <= ticks; i++) {
    const fx = b.x0 + (b.x1 - b.x0) * i / ticks;
    const sx = Math.round(card.map2d.x(fx)) + 0.5;
    g.beginPath(); g.moveTo(sx, plotT); g.lineTo(sx, xAxisY); g.stroke();
    const fy = b.y0 + (b.y1 - b.y0) * i / ticks;
    const sy = Math.round(card.map2d.y(fy)) + 0.5;
    g.beginPath(); g.moveTo(plotL, sy); g.lineTo(plotL + plotW, sy); g.stroke();
  }
  g.restore();
  g.fillStyle = T.text;
  for (let i = 0; i <= ticks; i++) {
    const fx = b.x0 + (b.x1 - b.x0) * i / ticks;
    const sx = Math.round(card.map2d.x(fx)) + 0.5;
    g.fillText(fmt(fx), sx - 14, xAxisY + 15);
    const fy = b.y0 + (b.y1 - b.y0) * i / ticks;
    const sy = Math.round(card.map2d.y(fy)) + 0.5;
    g.fillText(fmt(fy), 6, sy + 4);
  }
  g.strokeStyle = T.faint2; g.lineWidth = 1.2;
  g.beginPath(); g.moveTo(plotL, plotT); g.lineTo(plotL, xAxisY); g.lineTo(plotL + plotW, xAxisY); g.stroke();
  g.fillStyle = T.textStrong; g.font = "11px system-ui";
  g.fillText("x / cm", xAxisY + plotW - 40, xAxisY + 15);
  g.save(); g.translate(9, plotT + 12); g.rotate(-Math.PI / 2);
  g.fillText("y / cm", 0, 0); g.restore();

  if (card.el.ghost.checked && d.frames.length <= 120) {
    for (let f = 0; f < d.frames.length; f++)
      if (f !== card.frame) cardDrawFrameLine(card, f, T.faint, 0.35, 1);
  }
  if (card.el.echo.checked && card.history.length > 1) {
    for (const item of card.history) {
      if (item.f === card.frame) continue;
      const alpha = item.age / card.echoK;
      if (alpha > 0) cardDrawFrameLine(card, item.f, T.faint2, 0.35 * alpha, 1);
    }
  }
  cardDrawFrameLine(card, card.frame, "#1f77b4", 1, 2.2);

  const tr = (d.trace[card.frame] || []), pts = (d.frames[card.frame] || []);
  if (tr.length) {
    cardMarker(card, pts[tr[0]], "#d62728");
    cardMarker(card, pts[tr[tr.length - 1]], "#ff7f0e");
  }
  const jp = cardJointPoints(card, card.frame);
  if (jp.length > 1) {
    g.strokeStyle = "#d62728"; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(card.map2d.x(jp[0][0]), card.map2d.y(jp[0][1]));
    for (let i = 1; i < jp.length; i++) g.lineTo(card.map2d.x(jp[i][0]), card.map2d.y(jp[i][1]));
    g.stroke();
  }
  for (const p of jp) cardMarker(card, p, "#2ca02c");

  cardDrawStats(card); cardDrawSegments(card);
}

// Draws a small ringed marker at a 2D data point.
function cardMarker(card, p, color) {
  const g = card.ctx;
  g.beginPath();
  g.arc(card.map2d.x(p[0]), card.map2d.y(p[1]), 4.5, 0, Math.PI * 2);
  g.fillStyle = color; g.fill();
  g.lineWidth = 1.2; g.strokeStyle = "#fff"; g.stroke();
}

/* ---------------- 3D drawing ---------------- */

// Coalesces 3D redraws to the next animation frame.
function request3dRedraw(card) {
  if (card.redrawPending) return;
  card.redrawPending = true;
  requestAnimationFrame(() => { card.redrawPending = false; cardDraw3d(card); });
}

// Rotates a 3D point by the card's yaw/pitch (rot-y then rot-x).
function cardRot3(card, p) {
  const cy = Math.cos(card.yaw), sy = Math.sin(card.yaw);
  const x1 = p[0] * cy + p[2] * sy;
  const z1 = -p[0] * sy + p[2] * cy;
  const cp = Math.cos(card.pitch), sp = Math.sin(card.pitch);
  const y2 = p[1] * cp - z1 * sp;
  const z2 = p[1] * sp + z1 * cp;
  return [x1, y2, z2];
}

// Bounding box of all non-padding points in 3D.
function cardBounds3d(card) {
  const d = card.data;
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const fr of d.frames)
    for (const p of fr) {
      if (isPad(p)) continue;
      for (let k = 0; k < 3; k++) {
        if (p[k] < mn[k]) mn[k] = p[k];
        if (p[k] > mx[k]) mx[k] = p[k];
      }
    }
  for (let k = 0; k < 3; k++) { if (!isFinite(mn[k])) { mn[k] = -1; mx[k] = 1; } }
  return { mn, mx };
}

// Draws the 3D plot: grid, cloud, midline, joints, projected with yaw/pitch/zoom.
function cardDraw3d(card) {
  const { w, h } = setupCanvas(card.el.cv);
  const g = card.ctx;
  g.clearRect(0, 0, w, h);
  const d = card.data;
  if (!d) return;
  const T = themeColors();

  const { mn, mx } = cardBounds3d(card);
  const cx = w / 2, cy = h / 2;
  const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) / 2 || 1;
  const center = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
  const s = card.zoom * Math.min(w, h) / (2 * span) * 0.9;

  const proj = (p) => {
    const r = cardRot3(card, [p[0] - center[0], p[1] - center[1], p[2] - center[2]]);
    return [cx + r[0] * s, cy - r[1] * s];
  };

  const gz = mn[2];
  g.strokeStyle = T.grid3d; g.lineWidth = 1;
  const gridN = 6;
  for (let i = 0; i <= gridN; i++) {
    const fx = mn[0] + (mx[0] - mn[0]) * i / gridN;
    let [ax, ay] = proj([fx, mn[1], gz]);
    g.beginPath(); g.moveTo(ax, ay);
    [ax, ay] = proj([fx, mx[1], gz]); g.lineTo(ax, ay); g.stroke();
    const fy = mn[1] + (mx[1] - mn[1]) * i / gridN;
    [ax, ay] = proj([mn[0], fy, gz]);
    g.beginPath(); g.moveTo(ax, ay);
    [ax, ay] = proj([mx[0], fy, gz]); g.lineTo(ax, ay); g.stroke();
  }

  g.fillStyle = T.cloud;
  for (const p of card.cloud) {
    const [sx, sy] = proj(p);
    g.fillRect(sx, sy, 1.5, 1.5);
  }

  const tr = (d.trace[card.frame] || []), pts = (d.frames[card.frame] || []);
  if (tr.length) {
    g.strokeStyle = "#1f77b4"; g.lineWidth = 2.2;
    g.beginPath();
    for (let i = 0; i < tr.length; i++) {
      const [sx, sy] = proj(pts[tr[i]]);
      if (i === 0) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    }
    g.stroke();
    const [hx, hy] = proj(pts[tr[0]]);  cardMarker3(card, [hx, hy], "#d62728");
    const [tx, ty] = proj(pts[tr[tr.length - 1]]); cardMarker3(card, [tx, ty], "#ff7f0e");
  }

  const jp = [];
  const pos = new Map(tr.map((r, i) => [r, i]));
  if (card.gen)
    for (const j of card.gen.joints) {
      const row = j[2];
      let p = null;
      if (pos.has(row)) p = pts[row];
      else if (row < d.actual[card.frame]) p = pts[row];
      if (p) jp.push(p);
    }
  if (jp.length > 1) {
    g.strokeStyle = "#d62728"; g.lineWidth = 1.6;
    g.beginPath();
    for (let i = 0; i < jp.length; i++) {
      const [sx, sy] = proj(jp[i]);
      if (i === 0) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    }
    g.stroke();
  }
  for (const p of jp) { const [sx, sy] = proj(p); cardMarker3(card, [sx, sy], "#2ca02c"); }

  g.fillStyle = T.text; g.font = "11px system-ui";
  g.fillText("drag to rotate · scroll to zoom  (x length, y lateral, z depth)", 12, h - 10);
}

// Draws a small ringed marker at a projected 3D screen position.
function cardMarker3(card, [sx, sy], color) {
  const g = card.ctx;
  g.beginPath(); g.arc(sx, sy, 4, 0, Math.PI * 2);
  g.fillStyle = color; g.fill();
  g.lineWidth = 1.2; g.strokeStyle = "#fff"; g.stroke();
}

/* ---------------- error chart ---------------- */

// Draws the per-frame error bar chart for the selected metric.
function cardDrawErr(card) {
  const { w, h } = setupCanvas(card.el.errCv);
  const g = card.ectx;
  g.clearRect(0, 0, w, h);
  const d = card.data, gen = card.gen;
  if (!d || !gen) return;
  const T = themeColors();
  const metric = card.errMetric;
  const values = metric === "area" ? gen.per_area : gen.per_linear;
  const n = values.length;
  if (!n) return;
  const axis = 20, top = 8, bot = 8;
  const maxA = Math.max(...values) || 1;
  const rowH = (h - top - bot) / n;
  const barW = w - axis - 2;
  for (let i = 0; i < n; i++) {
    const y = top + i * rowH;
    const bw = (values[i] / maxA) * barW;
    g.fillStyle = (i === card.frame) ? "#1f77b4" : "#6f9fc9";
    g.fillRect(axis, y + 0.5, Math.max(1, bw), Math.max(1, rowH - 1));
  }
  g.strokeStyle = T.grid; g.lineWidth = 1;
  g.beginPath(); g.moveTo(axis, top); g.lineTo(axis, h - bot); g.stroke();
  g.fillStyle = T.text; g.font = "10px system-ui";
  g.fillText("0", 3, h - bot + 8);
  g.fillText(fmt(maxA), 3, top + 8);
  g.fillStyle = T.textStrong; g.font = "10px system-ui";
  g.save(); g.translate(w - 9, h - 3); g.rotate(-Math.PI / 2);
  g.fillText(metric === "area" ? "area error cm^2" : "linear error cm", 0, 0); g.restore();
}

// Frame index under a y position on the error canvas (-1 when outside).
function errFrameAtY(card, y) {
  const gen = card.gen;
  if (!gen || !gen.per_area.length) return -1;
  const n = gen.per_area.length;
  const rect = card.el.errCv.getBoundingClientRect();
  const rowH = (rect.height - 16) / n;
  return Math.max(0, Math.min(n - 1, Math.floor((y - rect.top - 8) / rowH)));
}

// Shows the hovered frame's error value as a tooltip.
function onErrMove(card, e) {
  const i = errFrameAtY(card, e.clientY);
  if (i < 0) return;
  const gen = card.gen;
  const val = card.errMetric === "area" ? gen.per_area[i] : gen.per_linear[i];
  card.el.errCv.title = `frame ${i}: ${card.errMetric} ${val} (${card.errMetric === "area" ? "cm^2" : "cm"})`;
}

// Jumps the card to the frame that was clicked in the error chart.
function onErrClick(card, e) {
  const i = errFrameAtY(card, e.clientY);
  if (i < 0) return;
  card.frame = i;
  card.el.frame.value = i;
  card.el.frameVal.textContent = i + " / " + card.data.num_frames;
  cardDraw(card);
}

/* ---------------- stats / segment lengths ---------------- */

// Renders the generation result stats table of the card.
function cardDrawStats(card) {
  const gen = card.gen;
  const el = card.el.stats;
  if (!gen) { el.innerHTML = ""; return; }
  const lens = cardSegLengths(card, card.frame);
  let totalLen = 0;
  for (const len of lens) totalLen += len;
  const maxLin = gen.per_linear.length ? Math.max(...gen.per_linear) : 0;
  const maxArea = gen.per_area.length ? Math.max(...gen.per_area) : 0;
  el.innerHTML =
    `<div><span>method</span><b>${card.el.method.selectedOptions[0].textContent}</b></div>` +
    `<div><span>param</span><b>${fmt(gen.param)}</b></div>` +
    `<div><span>joints</span><b>${gen.count}</b></div>` +
    `<div><span>generation time</span><b>${gen.time}s</b></div>` +
    `<div><span>avg linear error</span><b>${gen.avg_linear} cm</b></div>` +
    `<div><span>avg area error</span><b>${gen.avg_area} cm^2</b></div>` +
    `<div><span>max linear / frame</span><b>${fmt(maxLin)} cm</b></div>` +
    `<div><span>max area / frame</span><b>${fmt(maxArea)} cm^2</b></div>` +
    `<div><span>current frame length</span><b>${fmt(totalLen)} cm</b></div>`;
}

// Renders the segment-length table (or delegates to the heatmap).
function cardDrawSegments(card) {
  const el = card.el.seglen;
  if (card.segMode === "heatmap") { el.innerHTML = ""; cardDrawSegHm(card); return; }
  const lens = cardSegLengths(card, card.frame);
  const gen = card.gen;
  if (!gen || lens.length < 1) { el.innerHTML = ""; return; }
  const n = lens.length;
  const total = lens.reduce((a, b) => a + b, 0);
  const avg = total / n;
  const mn = Math.min(...lens), mx = Math.max(...lens);
  let run = 0;
  const rows = lens.map((len, i) => {
    run += len;
    return `<tr><td>${i + 1}</td><td>${fmt(len)}</td><td>${fmt(run)}</td></tr>`;
  }).join("");
  el.innerHTML =
    `<div class="segSummary">${n} segments · total ${fmt(total)} cm · avg ${fmt(avg)} · ` +
    `min ${fmt(mn)} · max ${fmt(mx)}</div>` +
    `<div class="segTable"><table><tr><th>seg</th><th>length cm</th><th>cumulative</th></tr>` +
    `${rows}</table></div>`;
}

// Draws the segment-length heatmap over all frames with the current frame line.
function cardDrawSegHm(card) {
  const cv = card.el.segHmCv;
  const { w, h } = setupCanvas(cv);
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  const hm = card.segHm;
  if (!hm.length || !hm[0].length) return;
  const T = themeColors();
  const nFrames = hm.length, nSeg = hm[0].length;
  let mn = Infinity, mx = -Infinity;
  for (const row of hm) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
  if (mn === mx) { mn -= 1; mx += 1; }
  const cw = w / nFrames, ch = h / nSeg;
  for (let f = 0; f < nFrames; f++) {
    for (let s = 0; s < nSeg; s++) {
      const t = (hm[f][s] - mn) / (mx - mn);
      g.fillStyle = `rgb(${Math.round(120 + 135 * t)},${Math.round(20 + 30 * t)},${Math.round(20 + 235 * t)})`;
      g.fillRect(f * cw, (nSeg - 1 - s) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 1.2;
  g.beginPath();
  g.moveTo((card.frame + 0.5) * cw, 0);
  g.lineTo((card.frame + 0.5) * cw, h);
  g.stroke();
  g.fillStyle = T.textStrong; g.font = "10px system-ui";
  g.fillText("min " + fmt(mn) + " cm", 3, 10);
  g.fillText("max " + fmt(mx) + " cm", w - 62, 10);
  g.fillText("segments (head top) vs frames", 3, h - 3);
}

/* ---------------- export (CSV / PNG) ---------------- */

// Exports the card's per-frame errors and segment lengths as two CSVs.
function exportCardCsv(card) {
  const d = card.data, gen = card.gen;
  if (!d || !gen) return;
  const name = (d.name || "data").replace(/\.(xls|csv)$/i, "");
  const rows = [["frame", "linear_error_cm", "area_error_cm2"]];
  for (let i = 0; i < gen.per_linear.length; i++)
    rows.push([i * gen.stride, gen.per_linear[i], gen.per_area[i]]);
  downloadCsv(`${name}_${gen.method}_${gen.param}_errors.csv`, rows);

  if (card.segHm.length) {
    const segRows = [["frame", ...card.segHm[0].map((_, s) => `seg_${s + 1}_cm`)]];
    card.segHm.forEach((row, f) => segRows.push([f * d.stride, ...row.map(v => v.toFixed(4))]));
    downloadCsv(`${name}_${gen.method}_${gen.param}_segment_lengths.csv`, segRows);
  }
}

// Exports the current plot canvas as a PNG.
function exportCardPng(card) {
  if (!card.data) return;
  const cv = card.el.cv;
  const name = (card.data.name || "data").replace(/\.(xls|csv)$/i, "");
  const method = card.el.method.selectedOptions[0].textContent.replace(/[^a-z0-9]+/gi, "_");
  const frame = card.frame * (card.data.stride || 1);
  cv.toBlob((blob) => downloadBlob(`${name}_${method}_frame${frame}_${card.mode}.png`, blob), "image/png");
}

// Exports every card's generation history as one compare CSV.
function exportCompareCsv() {
  const rows = [["file", "method", "param", "joints", "avg_linear_cm", "avg_area_cm2", "time_s"]];
  for (const c of cards) {
    const file = c.data ? c.data.name : c.file;
    const method = c.el.method.selectedOptions[0].textContent;
    for (const h of c.hist.values())
      rows.push([file, method, h.param, h.joints, h.linear, h.area, h.time]);
  }
  if (rows.length > 1) {
    const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
    downloadCsv(`compare_results_${stamp}.csv`, rows);
    setStatus("comparison exported");
  } else setStatus("nothing to export yet");
}

/* ---------------- hover tooltip ---------------- */

// Nearest non-padding data point to a canvas position (within 40 px).
function cardNearestPoint(card, mx, my) {
  const d = card.data;
  if (!d || card.mode !== "2d") return null;
  const pts = d.frames[card.frame];
  let best = null, bd = Infinity;
  for (let row = 0; row < pts.length; row++) {
    const p = pts[row];
    if (isPad(p)) continue;
    const sx = card.map2d.x(p[0]), sy = card.map2d.y(p[1]);
    const dd = (sx - mx) ** 2 + (sy - my) ** 2;
    if (dd < bd) { bd = dd; best = [row, p]; }
  }
  return bd < 40 ? best : null;
}

// Handles mousemove: 3D drag-to-rotate, or 2D point tooltip.
function onMove(card, e) {
  const rect = card.el.cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (card.mode === "3d") {
    if (card.mousedown) {
      card.yaw -= (e.clientX - card.mousedown.x) * 0.008;
      card.pitch += (e.clientY - card.mousedown.y) * 0.008;
      card.pitch = Math.max(-1.45, Math.min(1.45, card.pitch));
      card.mousedown = { x: e.clientX, y: e.clientY };
      request3dRedraw(card);
    }
    return;
  }
  const hit = cardNearestPoint(card, mx, my);
  const tip = card.el.tip;
  if (hit) {
    const [row, p] = hit;
    tip.style.display = "block";
    tip.style.left = (mx + 12) + "px";
    tip.style.top = (my + 12) + "px";
    tip.textContent = `point ${row}  (${fmt(p[0])}, ${fmt(p[1])})\nframe ${card.frame}  z=${fmt(p[2])}`;
  } else tip.style.display = "none";
}

/* ---------------- compare panel ---------------- */

// Computes the layout and data series of a compare chart for a metric.
function cmpLayout(metric, w, h) {
  const series = [];
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const c of cards) {
    if (!c.hist.size) continue;
    const pts = [...c.hist.values()].sort((a, b) => a.param - b.param)
      .map(p => ({ x: p.param, y: p[metric] }));
    series.push({ card: c, pts });
    for (const p of pts) {
      if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
    }
  }
  if (xMin === Infinity) { xMin = xMax = 0; yMin = yMax = 0; }
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  if (yMin === yMax) { yMin -= Math.abs(yMin) * 0.1 || 1; yMax += Math.abs(yMax) * 0.1 || 1; }
  const px = (xMax - xMin) * 0.06 || 1, py = (yMax - yMin) * 0.1 || 1;
  xMin -= px; xMax += px; yMin -= py; yMax += py;
  const m = { l: 36, r: 8, t: 6, b: 15 };
  return { w, h, series, xMin, xMax, yMin, yMax,
    X: (x) => m.l + (x - xMin) / (xMax - xMin) * (w - m.l - m.r),
    Y: (y) => m.t + (1 - (y - yMin) / (yMax - yMin)) * (h - m.t - m.b) };
}

// Draws one compare chart (linear/area/time/joints) onto its canvas.
function drawCmpChart(metric) {
  const cv = document.querySelector(`.cmpCv[data-metric="${metric}"]`);
  const { w, h } = setupCanvas(cv);
  const L = cmpLayout(metric, w, h);
  const g = cv.getContext("2d");
  const T = themeColors();
  g.clearRect(0, 0, L.w, L.h);
  g.font = "11px system-ui";
  if (!L.series.length) {
    g.fillStyle = T.faint2;
    g.fillText("no results yet - move sliders", 8, L.h / 2);
    return;
  }
  const m = { l: 36, r: 8, t: 6, b: 15 };
  g.strokeStyle = T.gridSoft; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = m.l + (L.w - m.l - m.r) * i / 4;
    g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, L.h - m.b); g.stroke();
    const y = m.t + (L.h - m.t - m.b) * i / 4;
    g.beginPath(); g.moveTo(m.l, y); g.lineTo(L.w - m.r, y); g.stroke();
  }
  g.strokeStyle = T.faint2;
  g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, L.h - m.b); g.lineTo(L.w - m.r, L.h - m.b); g.stroke();
  g.fillStyle = T.text;
  const yf = (v) => (metric === "time" && v < 1 ? v.toFixed(3) : fmt(v));
  g.fillText(yf(L.yMax), 2, m.t + 8);
  g.fillText(fmt(L.xMin), m.l - 2, L.h - m.b + 11);
  g.fillText(fmt(L.xMax), L.w - m.r - 24, L.h - m.b + 11);
  for (const s of L.series) {
    const color = cmpColor(s.card);
    g.strokeStyle = color; g.lineWidth = 1.6;
    g.beginPath();
    s.pts.forEach((p, i) => {
      const sx = L.X(p.x), sy = L.Y(p.y);
      if (i === 0) g.moveTo(sx, sy); else g.lineTo(sx, sy);
    });
    g.stroke();
    g.fillStyle = color;
    for (const p of s.pts) { g.beginPath(); g.arc(L.X(p.x), L.Y(p.y), 2.5, 0, Math.PI * 2); g.fill(); }
  }
}

// Redraws all compare charts and the legend when cards change.
function drawCompare() {
  const panel = $("compare");
  $("btnCompare").disabled = cards.length === 0;
  if (panel.classList.contains("hidden") || !cards.length) return;
  for (const metric of ["linear", "area", "time", "joints"]) drawCmpChart(metric);
  panel.querySelector(".cmpLegend").innerHTML = cards
    .filter(c => c.hist.size)
    .map(c => `<span><span class="cmpDot" style="background:${cmpColor(c)}"></span>` +
      `<span title="${esc(c.el.method.selectedOptions[0].textContent)}">` +
      `${esc(c.data ? c.data.name : c.file)} &middot; ${esc(c.el.method.selectedOptions[0].textContent)}</span></span>`)
    .join("");
}

// Wires the compare panel buttons and chart hover tooltips.
function setupCompareEvents() {
  $("btnCompare").addEventListener("click", () => {
    const panel = $("compare");
    panel.classList.toggle("hidden");
    if (!panel.classList.contains("hidden")) drawCompare();
  });
  $("cmpReset").addEventListener("click", () => {
    for (const c of cards) c.hist.clear();
    drawCompare();
  });
  $("cmpExport").addEventListener("click", exportCompareCsv);
  document.querySelectorAll(".cmpCv").forEach(cv => {
    cv.addEventListener("mousemove", (e) => {
      const rect = cv.getBoundingClientRect();
      const L = cmpLayout(cv.dataset.metric, rect.width, rect.height);
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let best = null, bd = 36;
      for (const s of L.series)
        for (const p of s.pts) {
          const d = (L.X(p.x) - mx) ** 2 + (L.Y(p.y) - my) ** 2;
          if (d < bd) { bd = d; best = { card: s.card, p }; }
        }
      cv.title = best
        ? `${best.card.data ? best.card.data.name : best.card.file} · ${best.card.el.method.selectedOptions[0].textContent}\n` +
          `param ${fmt(best.p.x)} · ${cv.dataset.metric} ${fmt(best.p.y)}`
        : "";
    });
  });
}

/* ---------------- session save / load ---------------- */

// Saves all cards and links to a session folder via the API.
async function saveSession() {
  if (!cards.length) { setStatus("nothing to save - add a view first"); return; }
  const payload = {
    version: 1,
    saved_at: new Date().toISOString(),
    data_folder: gstate.folder,
    cards: cards.map(c => ({
      file: c.file,
      method: c.el.method.value,
      threshold: parseFloat(c.el.threshold.value) || 0.5,
      segments: parseInt(c.el.segments.value, 10) || 10,
      frame: c.frame,
      mode: c.mode,
      ghost: c.el.ghost.checked,
      echo: c.el.echo.checked,
      loop: c.el.loop.checked,
      master: c.master,
      hist: [...c.hist.values()],
    })),
    links: Object.fromEntries(Object.entries(links).map(([v, set]) => [v, [...set].map(c => cards.indexOf(c))])),
  };
  try {
    const resp = await fetch("/api/session/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.error || resp.statusText);
    const sel = $("sessionList");
    const opt = document.createElement("option");
    opt.value = res.name; opt.textContent = res.name;
    sel.appendChild(opt);
    sel.value = res.name;
    setStatus("session saved to " + res.path);
  } catch (e) { setStatus("save session error: " + e.message); }
}

// Refreshes the session dropdown from the server's saved sessions.
async function refreshSessionList() {
  const sel = $("sessionList");
  sel.innerHTML = "";
  try {
    const res = await api("/api/sessions");
    for (const name of res.sessions) {
      const o = document.createElement("option");
      o.value = name; o.textContent = name;
      sel.appendChild(o);
    }
  } catch (e) { /* server list unavailable - ignore */ }
}

// Restores a saved session: rebuilds cards, links, frames, modes and history.
async function loadSession() {
  const name = $("sessionList").value;
  if (!name) { setStatus("no saved session selected"); return; }
  let s;
  try {
    s = await api("/api/session/load?name=" + encodeURIComponent(name));
  } catch (e) { setStatus("load session error: " + e.message); return; }

  for (const c of [...cards]) removeCard(c);
  for (const v in links) links[v].clear();
  if (s.data_folder) $("folder").value = s.data_folder;
  try { await loadFiles(); } catch (e) { setStatus("load session error: " + e.message); return; }
  for (const c of [...cards]) removeCard(c);

  for (const spec of s.cards || []) {
    const card = createCard(spec.file);
    card.el.method.value = spec.method || card.el.method.value;
    if (spec.threshold != null) {
      card.el.threshold.value = spec.threshold;
      card.el.thresholdVal.textContent = fmt(spec.threshold);
    }
    if (spec.segments != null) {
      card.el.segments.value = spec.segments;
      card.el.segmentsVal.textContent = fmt(spec.segments);
    }
    card.el.ghost.checked = !!spec.ghost;
    card.el.echo.checked = spec.echo !== false;
    card.el.loop.checked = spec.loop === true;
    card.master = spec.master !== false;
    card.el.masterSwitch.classList.toggle("off", !card.master);
    card.el.masterSwitch.title = "master link control: " + (card.master ? "ON" : "OFF");
  }

  const lv = s.links || {};
  for (const v of ["frame", "threshold", "segments", "data", "play", "method"]) {
    for (const idx of lv[v] || []) {
      const c = cards[idx];
      if (c) {
        links[v].add(c);
        const btn = c.el["link_" + v];
        if (btn) btn.classList.add("linked");
      }
    }
  }

  await Promise.all(cards.map(c => c.loadPromise));
  (s.cards || []).forEach((spec, i) => {
    const c = cards[i];
    if (!c || !c.data) return;
    c.frame = Math.max(0, Math.min(spec.frame || 0, c.data.frames.length - 1));
    c.el.frame.value = c.frame;
    c.el.frameVal.textContent = c.frame + " / " + c.data.num_frames;
    if (spec.mode === "3d" && c.data.has_z) setMode(c, "3d");
    if (spec.hist) c.hist = new Map(spec.hist.map(h => [Math.round(h.param * 10000) / 10000, h]));
    cardDraw(c); cardDrawErr(c);
  });
  drawCompare();
  refreshMaster();
  setStatus("session loaded: " + name);
}

$("saveSession").onclick = saveSession;
$("loadSession").onclick = loadSession;

/* ---------------- global controls ---------------- */

// Loads the file list for a folder, populates selects and creates first card.
async function loadFiles() {
  const folder = $("folder").value.trim() || gstate.folder;
  const meta = await api("/api/meta");
  if (!folder) $("folder").value = meta.folder;
  gstate.folder = $("folder").value.trim() || meta.folder;
  const res = await api("/api/files?folder=" + encodeURIComponent(gstate.folder));
  gstate.folder = res.folder;
  $("folder").value = res.folder;

  const sel = $("file");
  sel.innerHTML = "";
  for (const f of res.files) {
    const opt = document.createElement("option");
    opt.value = f; opt.textContent = f;
    sel.appendChild(opt);
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
  setStatus(res.files.length + " file(s) in " + res.folder);
  refreshMaster();
  for (const c of cards) { fillFileOptions(c.el.fileSel); if (c.data) c.el.fileSel.value = c.data.file; }
  if (!cards.length && res.files.length) addCard(sel.value);
}

// Adds a new card for the given (or the currently selected) file.
function addCard(file) {
  const sel = $("file");
  const f = file || sel.value;
  if (!f) { setStatus("select a file first"); return; }
  createCard(f);
}

$("addCard").onclick = () => addCard();
$("btnFolder").onclick = loadFiles;
$("folder").addEventListener("keydown", (e) => { if (e.key === "Enter") loadFiles(); });

/* ---------------- dark mode ---------------- */

// Redraws every canvas on the page after a theme change.
function redrawAll() {
  for (const c of cards) {
    cardDraw(c);
    cardDrawErr(c);
    cardDrawStats(c);
    if (c.segMode === "heatmap") cardDrawSegHm(c); else cardDrawSegments(c);
  }
  drawCompare();
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

window.addEventListener("mouseup", () => { for (const c of cards) c.mousedown = null; });
window.addEventListener("resize", () => { for (const c of cards) { cardDraw(c); cardDrawErr(c); } drawCompare(); });

// Boot: load the method list, wire events, load files and sessions.
(async function init() {
  try {
    await loadMethods();
  } catch (e) {
    setStatus("error loading methods: " + e.message);
  }
  METHODS = getMethods();
  setupCompareEvents();
  loadFiles().catch(e => setStatus("error: " + e.message));
  refreshSessionList();
})();
