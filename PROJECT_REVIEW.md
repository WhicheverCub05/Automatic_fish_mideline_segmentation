# Project Review — Automatic Fish Midline Segmentation

Date: 2026-08-11
Scope: full codebase review (Python backend, frontend, data pipeline, fluid simulation) for architectural and systematic improvements.

---

## 1. Executive summary

The project converts **fish midlines** (curved body shapes over time, from real tracking data or synthetic sine waves) into **joint configurations** (straight segments) for robotic fish, subject to an error threshold. It ships three front ends (CLI, Flask web viewer, and a standalone fluid simulation), several generation algorithms in two flavours (linear/area error), benchmarking scripts, and an excellent internal "AGENT_README" that documents the data model better than most production projects.

The core algorithms are sound and the web viewer is genuinely impressive (dependency-free Canvas rendering, live re-generation, multi-view linking, sessions). The codebase showed classic organic-growth symptoms — **a 1,448-line monolith**, **~800 lines of copy-pasted generation methods**, **buggy benchmarking helpers**, and **no test framework, packaging, or CI**. The two most consequential systematic problems were (a) the **data model is plain nested lists with two conflicting orientations** that are transposed by hand in several places, and (b) **the same algorithm was reimplemented once per error metric**. *(2026-08-11: all sixteen confirmed bugs in §5 were fixed and verified (§5.1 — #11 by the pytest suite, #12 by ruff, #13–14 while completing the remaining items, #15–16 by the new CLI menu tests); the three P1 items in §9 were completed — the growth methods are now implemented once and parametrised by metric (§6.2), the shared code moved into `core.py` breaking the import cycle (§6.5), and the error metrics + growth methods are vectorized (~40–275× faster, §7); the three P2 items are done — a method registry + `/api/methods` (§6.4), an 87-test pytest suite + frontend node smoke (§6.9), and packaging with console scripts (§6.9); the three P3 items are done — API-driven frontend method lists + JS moved into `web/js/` ES modules (§6.8), CI with ruff + pytest + frontend smoke (§6.9), and a `SimConfig` dataclass with deduped rasterisers (§6.7). The remaining items are also done: a `Midline` data model (§6.1), dead-code removal (§6.6), context-managed CSVs (§6.9), and repo hygiene (§6.10). Nothing on the roadmap is left open.)*

---

## 2. What the project does

- **Input**: `.xls` midlines (rows = body points, column pairs = frames), long-format `.csv` tracking data (`Midline_X/Y/Z` or `Relative_X/Y`, auto-detected), or synthetic sine-wave midlines.
- **Processing**: ~10 joint-generation methods — incremental growth, binary search, midpoint-only greedy, inflection-point, equal/diminishing segments, and brute-force removal — each existing in a **linear-error (cm)** and **area-error (cm²)** variant.
- **Evaluation**: per-segment linear (perpendicular distance) and area (trapezoid) error, averaged over all frames.
- **Outputs**: SVG plots, benchmark CSVs, an interactive web viewer with 2D/3D animation and per-frame error analysis, and a 2D Lattice-Boltzmann fluid-structure simulation of the resulting segmented fish.

---

## 3. Architecture overview

| Layer | Component | Lines | Role |
|---|---|---|---|
| Entry / CLI | `main.py` | 883 | Menu CLI, plotting/animations, step viewer. |
| Shared utilities | `core.py` | 420 | `Midline` data model (list-compatible wrapper: cached frame-major view, lazy render conversion, real-lengths padding info), loading (`.xls`/`.csv`), tracing, sine-wave synthesis, basic plotting. Leaf module — breaks the import cycle. |
| Algorithms | `generation_methods.py` | 561 | **All** generation methods, implemented once, parametrised by metric (`METRICS` strategy dict). |
| Algorithm wrappers | `generation_methods_linear_error.py` / `generation_methods_area_error.py` | 94 / 99 | Thin metric-fixed wrappers + brute-force methods. |
| Method registry | `method_registry.py` | 60 | **Single source of truth** for methods (id, label, func, param name/label/type) — CLI + web + tests. |
| Metrics | `calculate_error.py` | 251 | Linear + area error, scalar + vectorized (numpy); single total/per-frame implementation. |
| Benchmarking | `gather_data.py` | 144 | Only the two CLI-used functions remain (`compare_method_error`, `use_all_folder_data`); dead sine-wave/sandbox/brute-force helpers removed, CSVs written via context managers. |
| Simulation | `midline_fluid_sim.py` | ~590 | D2Q9 lattice-Boltzmann solver (fully periodic, no walls) + articulated rigid fish body (`RigidChainFish`, no springs/elasticity); `SimConfig` (tuned stability constants) + shared rasterisers. |
| Web server | `web_viewer.py` | 593 | Flask API; method registry + `/api/methods`, `/js/` module serving, caches, session save/load, fluid-sim endpoint (uses `SimConfig`); env-var/argv config that survives test-runner imports. |
| Web frontend | `web/index.html`, `web/fluid.html`, `web/js/` | 183 / 106 / ~1,700 | HTML pages are now just markup + one module import; all JS lives in `web/js/` ES modules (`app.js` viewer, `fluid.js` fluid page, `helpers.js` pure helpers, `methods.js` API-driven method list). |
| Tests | `tests/` | 87 pytest + 6 node checks | pytest: metrics (+edge cases), methods (incl. tiny/1-frame/straight midlines), `Midline`, core tracing/loaders, registry, **all CLI menus (scripted input)**, gather_data, Flask API (sessions, traversal, clamps), sim headless + unit stability. `frontend_smoke.mjs`: loads the real JS modules under a DOM stub (node). |
| CI | `.github/workflows/ci.yml` | — | ruff (`E9`+`F`) + pytest + frontend smoke on push (Python 3.11/3.12). |
| Data | `Fish_*.csv`, `Sturgeon from Elsa and Ted/` | — | Real tracking data + source clips (120 MB avi, untracked; git-lfs patterns in `.gitattributes`). |

**Data-flow**: file → loader → `[point][frame][x,y]` (generation) or `[frame][point][x,y,z]` (render) → generation method → `joints = [[x,y,point_index]…]` → error metrics / plotting / sim body.

**Data model** (documented in `AGENT_README.md` §"Core data model"):
- **Generation format** `midline[p][f][x,y]` — point-major. Used by all algorithms and metrics.
- **Render format** `midline[f][p][x,y,z]` — frame-major. Used by the viewer.
- **Joints** `[[x,y,point_index]…]` — x,y from frame 0; point_index maps onto every frame.
- **Padding** — CSV short frames are padded **per orientation**: generation format repeats each frame's tail point; render format zero-pads (`is_padding_point` detects it).

---

## 4. What works well

1. **The core segmentation idea is well thought out.** Head→tail joint configs spanning the whole body, tail always closed, error averaged over *all* frames — this is the right formulation for a robotic fish that must use one rigid config.
2. **Excellent institutional documentation.** `AGENT_README.md` documents orientations, mutation gotchas, stability tuning, and frontend internals — extremely valuable; keep it in sync as changes land.
3. **Robustness under real-world data.** Nearest-neighbour tracing (`midline_segments`, `trace_indices`) handles *unordered* CSV points and zero-padding; the web server feature-detects CSV columns instead of hardcoding; `is_padding_point` guards exist throughout the render path.
4. **The web viewer is a real engineering effort**: live re-generation on slider drag, ping-pong playback, per-frame error charts, segment-length heatmaps, cross-card linking, session save/load, and a full fluid-sim page — all with zero external JS.
5. **Safety-conscious web server**: path-traversal protection in `resolve_file` and session loading; input clamping on the fluid endpoint; method exceptions caught so the server stays alive; headless mode for the sim.
6. **Documented pitfalls** (e.g., the former `find_total_error` mutation — **fixed** in P1, now covered by a purity test — and the `amp·freq ≤ 0.05` LB stability envelope) show hard-won knowledge was preserved.

---

## 5. Confirmed bugs

Status legend: ✅ **FIXED 2026-08-11** — see §5.1 for details and verification.

| # | Location | Bug | Status |
|---|---|---|---|
| 1 | `gather_data.py:502` | `for f in range(all(all_files))` — `all()` of a list of strings is `True`, so `range(1)`: **only the first file is ever processed** by `compare_number_of_joints_brute_force_fish_data`. | ✅ Fixed |
| 2 | `gather_data.py:431`, `:194` | `for f in range(len(all_files) - 1)` — **skips the last file** in `use_all_folder_data` and `compare_method_error`. | ✅ Fixed |
| 3 | `gather_data.py:541` | Hardcoded `199` in `midline_reversed[abs(r - 199)]` — breaks on any midline without 200 points. | ✅ Fixed |
| 4 | `gather_data.py:569` | `compare_area_method_with_brute_force_joint_count` opens a file named `compare_starting_point_grow_segments_area_fish_data().csv` (copy-paste filename) — results go to the wrong file. | ✅ Fixed |
| 5 | **Generation on padded CSVs is wrong.** The CSV loaders pad short frames to the max point count (`main.py:91` etc.); the two big CSVs have **27–216 points/frame** (596/598 frames below max). Generation methods iterate the full padded range and force the tail joint to `len(midline)-1` — the tail lands on a `[0,0]` padding point in almost every frame, so area errors for the last segment are computed against the origin. Padding is only handled on the *render* path, not the *generation* path. | ✅ Fixed |
| 6 | `main.py:299` | `plot_3d_midline_with_joints` (dead code) indexes render-format `midline[int(joints[j][2])][0][2]` — frame and point indices swapped (latent bug if ever called). | ✅ Fixed |
| 7 | `main.py:1259+` | `visualise_generation_cli` — error thresholds for area methods are accepted in cm² but the prompt/`get_number` path doesn't distinguish metric types consistently across CLIs (menu 7 vs menu 1–4 use different prompts). Minor UX inconsistency. | ✅ Fixed |
| 8 | `calculate_error.py:99,126,157` | `find_total_area_error` / `find_total_linear_error` / `find_total_error` **mutate the caller's `joints` list** (append tail, then `pop`). If any code reads the list between calls (or an exception fires mid-function) the caller sees a corrupted config. `web_viewer.py:187` already works around this with `pts = list(joints)` — the fix is to make the metrics pure. | ✅ Fixed |
| 9 | `gather_data.py:454-459` | *(found while testing fix #2)* `use_all_folder_data` hardcodes frames `[0, 7]` — **IndexError on any midline with fewer than 8 frames** (e.g. the former `test/data1.xls`, 1 frame). | ✅ Fixed |
| 10 | `gather_data.py:463-468, 214-215` | *(found while testing fix #2)* plot filenames are sliced from the full file path, embedding a `\` separator → **`savefig` fails** for short/relative paths (`test\data1.xl`); same pattern in `compare_method_error`. | ✅ Fixed |

### 5.1 Bug fix log (2026-08-11)

| # | Fix applied | Files | Verification |
|---|---|---|---|
| 1 | `for f in range(all(all_files))` → `for f in range(len(all_files))` | `gather_data.py` | All files now iterated (verified by `use_all_folder_data`/`compare_method_error` smoke runs on `test/`, which was later removed per user decision). |
| 2 | Both `range(len(all_files) - 1)` → `range(len(all_files))` | `gather_data.py` | Last file no longer skipped (smoke run processed the only file in `test/`). |
| 3 | `abs(r - 199)` → `len(midline) - 1 - r` | `gather_data.py` | Reversal now correct for any point count. |
| 4 | CSV filename `compare_starting_point_grow_segments_area_fish_data()` → `compare_area_method_with_brute_force_joint_count()` | `gather_data.py` | Output lands in the right file. |
| 5 | Refactored the CSV loader into `load_3d_csv_data()` + `csv_data_to_midline()`; the **generation format now pads short frames by repeating each frame's last real point (tail extension)** instead of `[0,0]`, so the tail joint is a real point in every frame and no segment is scored against synthetic zeros. The **render format still zero-pads** (`is_padding_point` sentinel untouched). Also deleted the 176-line verbatim duplicate loader in `gather_data.py` (dead code). | `core.py`, `main.py`, `gather_data.py` | Real sturgeon CSV: tail row is `[0,0]` in **0/598** frames (was ~596); tail joint now at real coords; render path still pads 53,265 cells and tracing still skips them. |
| 6 | `joint_z = [midline[frame][int(joints[j][2])][2] …]` (was `midline[int(joints[j][2])][0][2]`) | `main.py` | Correct frame-major indexing; function still unused. |
| 7 | Method list in `visualise_generation_cli` now carries a param label; prompt shows the metric unit: `error threshold (cm^2)` / `error threshold (cm)` / `number of segments` / etc. | `main.py` | Piped CLI run: option 9 now prompts `error threshold (cm^2):`. |
| 8 | Added `joints_with_tail()` helper; `find_total_error` now evaluates against a local copy (computes both metrics); `find_total_area_error` / `find_total_linear_error` became thin wrappers over it. **Metrics are now pure** — no caller's list is ever mutated. | `calculate_error.py` | Purity assertion test (joints list unchanged after all three calls; wrapper values equal `find_total_error` components) passes; brute-force methods still work. |
| 9 | Frame selection clamped to available range: `plot_frames = [i for i in [0, 7] if i < len(midline[0])]` | `gather_data.py` | `use_all_folder_data` on the 1-frame `data1.xls` no longer raises. |
| 10 | Plot names/titles now use `os.path.basename()` / `os.path.splitext()` instead of raw path slicing | `gather_data.py` | SVGs save successfully for relative and short paths (verified). |
| 11 | CSV loader rejected 2D midlines: `has_midline_cols` required `Midline_Z`, so a `Midline_X`/`Midline_Y`-only CSV (no Z column) fell through to "columns not found" | `core.py` | 2D `Midline_X`/`Midline_Y` CSVs load now (z=0); caught by the pytest `synthetic_csv` fixture (found 2026-08-11). |
| 12 | `visualize_3d_animation_cli` (menu 10) still referenced the removed `gm_a`/`gm_l` imports — **NameError at runtime** if a method was applied to 3D datasets | `main.py` | Now uses the method registry; caught by `ruff check` (F821, found 2026-08-11 during the P3 lint pass) |
| 13 | `web_viewer.py` read `sys.argv[2]` as the port at **module import** — importing it from a test runner (e.g. `python -m pytest -q`) crashed with `ValueError` | `web_viewer.py` | argv is now parsed defensively (env vars still win); caught by `tests/test_web_api.py` import (found 2026-08-11) |
| 14 | After the frontend moved into `web/js/` ES modules, no route served them — the pages loaded blank with a 404 on `/js/app.js` | `web_viewer.py` | New `/js/<path>` route uses traversal-safe `send_from_directory`; covered by `test_web_js_modules` + the node frontend smoke (found 2026-08-11) |
| 15 | `pick_method_and_save_all` passed the `save_path` tuple unpacked into `get_user_save_path` — an explicit save dir crashed with `TypeError` (works only via `sys.argv[2]`) | `main.py` | Now unpacked (`get_user_save_path(data_path, *save_path)`); caught by the new CLI menu test (found 2026-08-11) |
| 16 | Menu 1 unpacked the registry entry in the wrong slot (`method` was the label, not the function) — `AttributeError: 'str' object has no attribute '__name__'` for every option 1–6 | `main.py` | Tuple unpacking fixed; caught by the new CLI menu test (found 2026-08-11) |

**End-to-end regression after all fixes**: `py_compile` clean on all modules; **`python -m pytest` green (87 tests, ~14 s)** — metric purity + vectorized≡scalar equivalence + edge cases (vertical/degenerate segments), all registry methods on synthetic + real data (incl. 2-point / single-frame / straight midlines), the `Midline` data model, `core` tracing + loaders, the method registry, **all five CLI menus with scripted input** (menus 1, 7, 8, 9, 10 — these caught bugs #15–16), `gather_data`, Flask test-client API tests (sessions round-trip, traversal blocking, fluid clamps), sim `--headless` **and** unit-level body-stability/SimConfig tests; **`node tests/frontend_smoke.mjs` green** (both frontend modules load and populate under a DOM stub); CLI menus 1, 7, 9, 10 run. (The old `test/` sample folder was removed by user decision; all verification uses `Sturgeon from Elsa and Ted\midlines` and synthetic sine waves.)

*(Note: bugs #1–4, #9–10 were fixed in functions that were later removed as dead code — `compare_number_of_joints_brute_force_fish_data`, `compare_starting_point_grow_segments_area_fish_data` and the sine-wave comparison helpers no longer exist (§6.6), so those fixes now live only in this log.)*

**Also updated**: `AGENT_README.md` — the "metrics MUTATE joints" warning is replaced with the pure-metrics contract, the padding note documents per-orientation padding (generation = tail extension, render = zeros), and P2 additions: the method registry (§6.4), the pytest suite (new Testing section), env-var config for the web server, and the packaging/console scripts.

---

## 6. Systemic / architectural issues

### 6.1 No real data model — raw nested lists, two orientations
~~Every module hand-built and hand-transposed `[point][frame]` ↔ `[frame][point]` arrays~~ — **fixed 2026-08-11**: `core.Midline` is now the data model. It is a thin list-compatible wrapper over the generation format (indexing, `len`, iteration and `np.asarray` behave exactly like the raw list, so all algorithms work unchanged) and adds: a **cached canonical frame-major numpy view** (`.frame_major`, used by the vectorized metrics via `to_frame_major`), a **lazy render-format conversion** (`.render`, zero-padded per frame), and the **per-frame real point counts** (`.real_lengths`) the loaders know but the raw list dropped. All generation-format loaders (`load_midline_data`, `load_3d_csv_data(for_generation=True)`, `load_data_file`) and the sine-wave generator return a `Midline`. The render path still returns plain frame-major lists (viewer-internal). Covered by `tests/test_midline.py` (5 tests). Still open: validation (shape assertions) and a numpy-backed canonical storage instead of the wrapped list.

### 6.2 The "grow" algorithms are duplicated per error metric
~~`grow_segments`, `grow_segments_binary_search`, `grow_segments_binary_search_midpoint_only`, `grow_segments_from_inflection` were ~800 combined lines duplicated across the two generation modules~~ — **fixed 2026-08-11**: all methods now live once in `generation_methods.py`, parametrised by `metric='area'|'linear'` through a `METRICS` strategy dict (segment scorer + point scorer); the two old modules are thin wrappers (~95 lines each). Verified that every wrapper produces **byte-identical joint configs** to the old implementations on real sturgeon data. Adding a third metric = adding one strategy pair.

### 6.3 Error accumulation is copy-pasted 4+ times
~~`find_total_error`, `find_total_area_error`, `find_total_linear_error`, and `per_frame_error` (web_viewer) all re-implemented the same double loop~~ — **fixed 2026-08-11**: `calculate_error.py` now has one vectorized `find_total_error` (both metrics) + `find_per_frame_error`; the area/linear variants are wrappers; `web_viewer.per_frame_error` delegates to `ce.find_per_frame_error`. The only remaining metric-loop is the per-frame search inside `grow_segments_binary_search_midpoint_only` (intentionally scalar, see §7).

### 6.4 Method registry is defined in 6 places
~~The method→id→param-type mapping was duplicated across `main.py` menus, `web_viewer.py:77`, and a JS copy in `web/index.html`~~ — **fixed 2026-08-11**: `method_registry.py` is now the **single source of truth** (entries `(id, label, func, param name, param label, type)` + `call()`/`call_by_id()` dispatch). The CLI menus (`visualise_generation_cli`, `compare_methods_cli`, `pick_method_and_save_all`, `visualize_data_cli`, `visualize_3d_animation_cli`) and the web server (`/api/methods`, `/api/generate`, `/api/fluid/simulate`) all consume it — adding a method is one registry line. The frontend dropdowns are generated from `/api/methods` (`web/js/methods.js`, with a static fallback for when the server is unreachable), so the last static Python→JS method-list copy is gone.

### 6.5 Circular import and import-time side effects
~~`main.py` imported `gather_data` and vice-versa; `web_viewer.py`/`midline_fluid_sim.py` imported `main`~~ — **fixed 2026-08-11**: all shared helpers moved to a neutral `core.py`; `gather_data`, `web_viewer` and `midline_fluid_sim` import from `core` only, so the only remaining edge is the one-directional `main` → `gather_data` import (fine). The dead `gather_data()` entry (which called `main.set_data_folder`) was removed. **`web_viewer.py` config is now testable**: `FISH_DATA_FOLDER`/`FISH_PORT` env vars override the positional args, which override the defaults (no more `sys.argv` reading that could not be overridden).

### 6.6 Dead code
~~`plot_3d_midline`, `plot_3d_midline_with_joints`, `plot_2d_projection_with_z` (main.py) were never called, and `gather_data.py` had ~460 lines of unreachable helpers (sine-wave sandbox/comparisons, brute-force benchmarks)~~ — **removed 2026-08-11**: the three 3D/2D plot functions and the `gather_data` dead functions (`sinewave_sandbox`, `compare_visual_sinewaves`, `compare_all_methods_linear_error_sinewave`, `compare_method_sinewave_*`, `compare_linear_and_area_error`, `compare_number_of_joints_brute_force_fish_data`, `compare_starting_point_grow_segments_area_fish_data`, `compare_area_method_with_brute_force_joint_count`) are deleted; `gather_data.py` is 605 → 144 lines and keeps only the two CLI-used functions. Verified by grep (zero external callers) and by CLI menu 7/1 runs.

### 6.7 Fluid simulation coupling
~~`midline_fluid_sim.py` duplicated `body_cells` + `rasterize` verbatim between `FishBody` and `RigidChainFish`, and the "tuned" body parameters that made the web endpoint stable lived only in `web_viewer.py` while the CLI defaults were known-unstable~~ — **fixed 2026-08-11**: the rasterisers are now one shared pair of module functions (`_body_cells`/`_rasterize`), and a **`SimConfig` dataclass** in `midline_fluid_sim.py` holds every stability-critical constant (spring params, cell radius, velocity clips, the `amp·freq` envelope) with the **tuned stable values as the single default**. The web endpoint passes `SimConfig()`; the CLI gets the same defaults. The `LBGrid.step` unused-`nx` local and the keep-alive animation references were also cleaned (ruff F841).

### 6.8 Frontend is a 1,500-line inline script
~~`web/index.html` was one HTML file containing all JS~~ — **fixed 2026-08-11**: the HTML pages are now markup only (`index.html` 1,454 → 183 lines, `fluid.html` 379 → 106), each importing a single ES module: `web/js/app.js` (viewer logic), `web/js/fluid.js` (fluid page), plus the shared `helpers.js` (pure DOM-free helpers) and `methods.js` (method list from `/api/methods`, static fallback). Method dropdowns are generated from the API. The frontend is now covered by `tests/frontend_smoke.mjs` — a node harness that loads the real modules under a minimal DOM + fetch stub and asserts both pages populate end-to-end (6 checks). Still open: splitting `app.js`'s ~1,300 lines internally (drawing vs. state vs. wiring) — testable now that it's a module.

### 6.9 No packaging, tests, lint, or CI
- ~~No `requirements.txt` / `pyproject.toml`~~ — **added 2026-08-11**: `requirements.txt` (runtime + pytest) and `pyproject.toml` (setuptools, flat-module package, console scripts `fishseg-cli` / `fishseg-web` / `fishseg-sim`; `pip install -e .` verified).
- ~~Only ad-hoc smoke tests~~ — **pytest suite added 2026-08-11** (`tests/`, 31 tests, ~5 s): metric purity + vectorized≡scalar equivalence, every registry method on synthetic sine midlines (invariants: head→tail configs), one end-to-end run on the real sturgeon `.xls` (auto-skips if data absent), Flask API test-client smoke tests (`/api/meta`, `/api/methods`, `/api/files`, `/api/data`, `/api/generate`, `/api/fluid/simulate`, incl. 400/404 paths), and a headless sim run.
- ~~No lint / CI~~ — **added 2026-08-11**: `[tool.ruff]` gates on `E9`+`F` (syntax/runtime errors + pyflakes) and `.github/workflows/ci.yml` runs `ruff check .` + `python -m pytest -q` + `node tests/frontend_smoke.mjs` on push for Python 3.11/3.12. The first lint pass caught bug #12 (a live `NameError` in menu 10) and cleaned ~25 unused-import/unused-local/placeholder-f-string issues. Still open: type checking / formatter config.
- ~~`gather_data.py` opens CSVs with bare `open()`~~ — **fixed 2026-08-11**: all CSV writes use `with open(...)` context managers (the surviving `compare_method_error`; every other `open()` lived in now-deleted dead functions, §6.6).

### 6.10 Repository hygiene
- ~~120 MB `.avi` + ~24 MB CSVs sit untracked with no guidance~~ — **2026-08-11**: `.gitattributes` declares git-lfs patterns (`*.avi`, `*.xls`, `*.xlsx`, `*.csv`) and `.gitignore` excludes the data folders + `*.avi`; README documents the data location. The data itself is intentionally not committed.
- Two READMEs (`README.md`, `AGENT_README.md`) with overlapping content — **2026-08-11**: roles are now explicit — README = user-facing overview, AGENT_README = authoritative technical reference, each cross-linking the other.

---

## 7. Performance observations

- **Error computation is now vectorized.** ~~All error metrics were scalar Python loops~~ — **fixed 2026-08-11**: `calculate_error.py` gained numpy scorers (`area_scores`, `linear_sum_scores`, `linear_max_scores`, `linear_distances`) over a frame-major array, used by both the total/per-frame metrics and the growth methods (`generation_methods.py` converts `[point][frame]` → `(frames, points, 2)` once per call). Measured on a synthetic midline with the same size as the real CSVs (598 frames × 216 points), old scalar vs new vectorized:

  | Operation | Old (scalar) | New (vectorized) | Speedup |
  |---|---|---|---|
  | `grow_segments` (area) | 8.66 s | 0.041 s | ~210× |
  | `grow_segments_binary_search` (area) | 17.39 s | 0.063 s | ~275× |
  | `grow_segments` (linear) | 1.65 s | 0.031 s | ~53× |
  | `grow_segments_binary_search` (linear) | 0.49 s | 0.046 s | ~11× |
  | `find_total_error` | ~1.0 s | 0.025 s | ~40× |

  Outputs were verified identical to the scalar implementations (joint configs byte-identical; metrics equal within 1e-9). Web slider latency on the 598-frame CSVs is now well under the frame budget.
- **Still scalar**: the per-frame binary search inside `grow_segments_binary_search_midpoint_only` (frame searches diverge, so they don't batch across frames) and the brute-force removal methods.
- **Brute-force removal** (`generate_segments_to_quantity`, `generate_segments_to_max_area_error`) does `copy.deepcopy(joints)` + a full `find_total_area_error` per candidate joint per removal — O(n³)-ish and documented as slow (now ~40× faster per metric call, but still the slowest algorithms). The `resolution_division` hack exists for this reason; a proper fix is incremental error updates when deleting a joint, or relegation to a "reference" algorithm.
- **Redundant work**: `Simulation.step` → `lb.velocity()` is computed, then the animation loop calls `flow_speed()` (another full `velocity()`) every frame; minor, but the LBM is the hot loop.
- **Caching in the web server is good** (display/midline/generation caches) — just cap `CACHE` at 128 entries by clearing wholesale; an LRU would be kinder under heavy slider use.

---

## 8. Recommended target architecture

*Status 2026-08-11: everything on the roadmap is implemented at the repo root: `core.py` (incl. the `Midline` data model), `calculate_error.py`, `generation_methods.py`, `method_registry.py`, `SimConfig` in `midline_fluid_sim.py`, `pyproject.toml`, `tests/` (pytest + node frontend smoke), `.github/workflows/ci.yml`, and the `web/js/` ES modules. The §8 package-dir layout (moving everything under a `fishseg/` package, renames like `cli.py`/`server.py`) remains the documented target but is not required — the flat-module package installs and runs.*

```
fishseg/                        # package (pyproject.toml, entry points)
  core.py                       # Midline data model (one orientation), padding mask,
                                #   tracing (midline_segments/trace_indices), is_pad
  io.py                         # single loader: xls + csv (feature-detect), -> Midline
  metrics.py                    # pure find_area_error/find_linear_error,
                                #   total_error(midline, joints, metric), per_frame_error
  methods.py                    # strategies parametrised by metric + METHOD registry {id,...}
  cli.py                        # menu driven by registry
  benchmark.py                  # gather_data equivalents (fixed bugs), context-managed files
  simulation.py                 # LBM + bodies, shared rasterize, SimConfig dataclass
  server.py                     # Flask app (factory, config via env/args, registry-driven)
web/                            # index.html, fluid.html, shared js modules
tests/                          # pytest: metrics (pure), methods on synthetic midlines,
                                #   padding handling, server smoke tests (Flask test client)
```

Key invariants to enforce:
- One loader, one orientation; convert at module boundaries only.
- Metrics are pure (never mutate `joints`).
- Methods take `metric` explicitly; no linear/area duplication.
- Registry is the single source of truth for CLI + web + `/api/methods`.

---

## 9. Prioritised roadmap

| Priority | Item | Effort | Why |
|---|---|---|---|---|
| ~~P0~~ ✅ | ~~Fix `gather_data` bugs (range, `all()`, hardcoded 199, wrong filename)~~ — **done 2026-08-11** (bugs #1–4, plus #9–10 found during testing) | S | Wrong benchmark data is silently produced today |
| ~~P0~~ ✅ | ~~Make error metrics pure (no `joints` mutation)~~ — **done 2026-08-11** (bug #8; metrics also consolidated) | S | Removes the worst footgun; enables safe caching |
| ~~P0~~ ✅ | ~~Fix padding handling on the generation path~~ — **done 2026-08-11** (bug #5: generation format now pads with each frame's tail point; render path unchanged) | M | The CSVs generated measurably wrong tail joints |
| ~~P1~~ ✅ | ~~Collapse linear/area method duplication via metric parameter~~ — **done 2026-08-11** (all methods once in `generation_methods.py`, metric strategy dict, thin wrappers; joint-identical output verified) | M | Biggest structural win; killed ~800 duplicated lines |
| ~~P1~~ ✅ | ~~Move the (now single) loader into a neutral `core` module~~ — **done 2026-08-11** (shared helpers in `core.py`; `main`↔`gather_data` cycle broken; `web_viewer`/`midline_fluid_sim` import `core` only) | M | Breaks the circular import |
| ~~P1~~ ✅ | ~~Vectorise error computation~~ — **done 2026-08-11** (numpy scorers; ~40–275× faster; metrics equal to scalar within 1e-9, joint configs identical) | M | Speeds up CLI sweeps and web slider UX |
| ~~P2~~ ✅ | ~~Method registry + `/api/methods` endpoint~~ — **done 2026-08-11**: `method_registry.py` is the single source of truth (id, label, func, param name/label/type); CLI menus and web server consume it; `/api/methods` serves the list (§6.4) | M | Adding a method becomes one-line |
| ~~P2~~ ✅ | ~~pytest suite~~ — **done 2026-08-11**: 87 tests in `tests/` (~14 s): metrics (+edge cases), all methods on synthetic + tiny/1-frame/straight data, `Midline`, core tracing/loaders, registry, **CLI menus with scripted input**, gather_data, Flask test-client API tests (sessions, traversal, clamps), sim `--headless` + unit stability; plus `node tests/frontend_smoke.mjs` for the frontend modules (§6.9) | M | Zero automated protection → 87 regression guards + 6 frontend checks |
| ~~P2~~ ✅ | ~~Packaging (`pyproject.toml`, `requirements.txt`, console scripts)~~ — **done 2026-08-11**: flat-module setuptools package; `fishseg-cli` / `fishseg-web` / `fishseg-sim`; `pip install -e .` verified (§6.9) | S | Reproducible installs |
| ~~P3~~ ✅ | ~~Frontend: generate method dropdowns from `/api/methods`, then split the 1,500-line inline JS into modules~~ — **done 2026-08-11**: dropdowns come from `/api/methods` via `web/js/methods.js` (static fallback); the inline scripts moved into `web/js/` ES modules (`app.js`, `fluid.js`, `helpers.js`); pages are markup-only (§6.8). The internal split of `app.js`'s ~1,300 lines (drawing vs. state vs. wiring) is the only optional follow-up. | S/L | Removes the last static method-list copy; testability |
| ~~P3~~ ✅ | ~~CI (lint + test on push)~~ — **done 2026-08-11**: `.github/workflows/ci.yml` (ruff `E9`+`F`, pytest, node frontend smoke on Python 3.11/3.12); the first lint pass caught bug #12 and cleaned 25+ issues (§6.9) | S | — |
| ~~P3~~ ✅ | ~~`SimConfig` dataclass with stable defaults; dedupe body rasterisation~~ — **done 2026-08-11**: `SimConfig` holds all tuned stability constants (single default for CLI + web); `_body_cells`/`_rasterize` shared by both fish bodies (§6.7) | M | Removes known-unstable defaults trap |

---

## 10. Appendix — strengths to preserve

- `AGENT_README.md` — keep updating it; it is the project's best asset.
- The web viewer's dependency-free approach and live-regeneration UX.
- The documented stability envelopes in the fluid sim (they represent real debugging effort).
- The `.xls`/`.csv` feature-detection and the unordered-point tracing — this is what makes real tracking data usable at all.
