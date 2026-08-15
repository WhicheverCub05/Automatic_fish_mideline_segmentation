/* Method list for the frontend, loaded from the backend /api/methods endpoint.
   The static FALLBACK below mirrors method_registry.py and keeps the pages
   usable (e.g. opened over file:// or while the server is starting); the live
   registry is the source of truth once fetched. */

// Static [id, label, param type] list used when the server is unreachable.
const FALLBACK = [
  ["grow_area", "grow_segments (area)", "err"],
  ["grow_bs_area", "grow_segments_binary_search (area)", "err"],
  ["grow_bs_mp_area", "grow_segments_binary_search_midpoint_only (area)", "err"],
  ["grow_inflect_area", "grow_segments_from_inflection (area)", "err"],
  ["grow_linear", "grow_segments (linear)", "err"],
  ["grow_bs_linear", "grow_segments_binary_search (linear)", "err"],
  ["grow_bs_mp_linear", "grow_segments_binary_search_midpoint_only (linear)", "err"],
  ["grow_inflect_linear", "grow_segments_from_inflection (linear)", "err"],
  ["equal_segments", "create_equal_segments", "count"],
  ["diminishing_segments", "create_diminishing_segments", "count"],
];

let cache = null;

// Fetches the live method list from /api/methods, falling back to FALLBACK.
export async function loadMethods() {
  if (cache) return cache;
  try {
    const resp = await fetch("/api/methods");
    if (!resp.ok) throw new Error("bad response");
    const data = await resp.json();
    cache = data.methods.map(m => [m.id, m.label, m.type]);
  } catch (e) {
    cache = FALLBACK; // server unreachable: keep the pages usable
  }
  return cache;
}

// Returns the cached method list (or the fallback before loadMethods runs).
export function getMethods() {
  return cache || FALLBACK;
}

// Populates a <select> with the methods and selects the default id.
export async function fillMethodSelect(sel, defaultId = "grow_bs_area") {
  const methods = await loadMethods();
  sel.innerHTML = "";
  for (const [id, label] of methods) {
    const o = document.createElement("option");
    o.value = id; o.textContent = label;
    sel.appendChild(o);
  }
  if (![...sel.options].some(o => o.value === defaultId)) defaultId = methods[0][0];
  sel.value = defaultId;
  return methods;
}
