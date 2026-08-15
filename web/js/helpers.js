/* Pure DOM-free helpers shared by the main viewer and the fluid page.
   Nothing here touches `cards` / page state, so it stays testable. */

// Theme-aware canvas palette, mirroring the CSS :root[data-theme] variables.
// Reading the theme at draw time means a toggle just sets the attribute and
// triggers a redraw - no other state to keep in sync.
export function themeColors() {
  const dark = (document.documentElement.dataset.theme || "") === "dark";
  if (dark) return {
    dark: true, bg: "#0d1117", grid: "#2d333b", gridSoft: "#21262d",
    text: "#8b949e", textStrong: "#c9d1d9", faint: "#5a6572", faint2: "#6e7681",
    grid3d: "rgba(140,150,160,0.30)", cloud: "rgba(150,155,165,0.5)",
    midline: "#4a6b8a", markerEdge: "#0d1117", frameLine: "rgba(200,210,220,0.25)",
  };
  return {
    dark: false, bg: "#ffffff", grid: "#dfe3e8", gridSoft: "#eef1f4",
    text: "#667", textStrong: "#445", faint: "#c6ccd2", faint2: "#9aa3ab",
    grid3d: "rgba(140,150,160,0.25)", cloud: "rgba(150,155,165,0.4)",
    midline: "#b9cfe4", markerEdge: "#fff", frameLine: "rgba(40,45,55,0.35)",
  };
}


// Next playback state for a frame animation: when loop is true the animation
// wraps from the last frame back to the first (always moving forward); when
// false it rewinds - it bounces back and forth (ping-pong).
// Returns [nextFrame, nextPlayDir].
export function advancePlay(frame, n, loop, playDir) {
  if (n <= 1) return [frame, playDir];
  if (loop) return [(frame + 1) % n, 1];
  frame += playDir;
  if (frame >= n) return [n - 2, -1];
  if (frame < 0) return [1, 1];
  return [frame, playDir];
}

// HTML-escapes a string for safe insertion into innerHTML.
export function esc(s) {
  return String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
}

// Formats a number for display with a sensible number of decimals.
export function fmt(v) {
  if (!isFinite(v)) return "inf";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// True if a [x, y, z] point is zero padding (short frame filler).
export function isPad(p) { return Math.abs(p[0]) < 1e-9 && Math.abs(p[1]) < 1e-9 && Math.abs(p[2]) < 1e-9; }

// Sizes a canvas to its CSS box times the device pixel ratio (HiDPI-safe).
export function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.max(1, Math.round(rect.width * dpr));
  c.height = Math.max(1, Math.round(rect.height * dpr));
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w: rect.width, h: rect.height };
}

// Builds a downsampled 3D point cloud of all frames for the 3D background dots.
export function buildCloud(d) {
  const maxPts = 10000;
  const cloud = [];
  const fStep = Math.max(1, Math.floor(d.frames.length / 40));
  for (let f = 0; f < d.frames.length; f += fStep) {
    const fr = d.frames[f];
    const pStep = Math.max(1, Math.floor(fr.length / 100));
    for (let i = 0; i < fr.length; i += pStep) {
      const p = fr[i];
      if (isPad(p)) continue;
      cloud.push(p);
      if (cloud.length >= maxPts) return cloud;
    }
  }
  return cloud;
}

// Bounding box of all non-padding points, padded by 5% for the 2D axes.
export function bounds2d(d) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const fr of d.frames)
    for (const p of fr) {
      if (isPad(p)) continue;
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    }
  if (minX === Infinity) return { x0: 0, x1: 1, y0: 0, y1: 1 };
  const px = (maxX - minX) * 0.05 || 1, py = (maxY - minY) * 0.05 || 1;
  return { x0: minX - px, x1: maxX + px, y0: minY - py, y1: maxY + py };
}

// Data->canvas coordinate mapper with fixed margins and aspect-preserving scale.
export function makeMap(b, w, h, m = { l: 40, r: 10, t: 10, b: 26 }) {
  const availW = w - m.l - m.r;
  const availH = h - m.t - m.b;
  const scale = Math.min(availW / (b.x1 - b.x0), availH / (b.y1 - b.y0)) || 1;
  const plotW = (b.x1 - b.x0) * scale;
  const plotH = (b.y1 - b.y0) * scale;
  const plotL = m.l + (availW - plotW) / 2;
  const plotT = m.t + (availH - plotH) / 2;
  return {
    l: plotL, t: plotT, plotL, plotT, plotW, plotH, w, h, scale,
    x: (x) => plotL + (x - b.x0) * scale,
    y: (y) => plotT + (b.y1 - y) * scale,
  };
}

// Triggers a client-side download of a Blob under the given filename.
export function downloadBlob(filename, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// Downloads a 2D array of rows as a CSV file (quoting fields that need it).
export function downloadCsv(filename, rows) {
  const csv = rows.map(r => r.map(v => {
    if (typeof v === "string" && /[",\n]/.test(v)) return '"' + v.replace(/"/g, '""') + '"';
    return v;
  }).join(",")).join("\r\n");
  downloadBlob(filename, new Blob([csv], { type: "text/csv" }));
}
