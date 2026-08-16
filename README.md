# Automatic Fish Midline Segmentation

This is a vibe coded branch of the original Automatic Fish Midline Segmentation, with the difference being an addition of web interface to visualize and compare data. Also includes a new fluid simulation to test and measure swimming effectiveness of generated midlines. Models used: deepseek v4 flash / deepseek v4 pro

> **For contributors / agents**: `AGENT_README.md` is the authoritative technical reference (data model, architecture, web viewer internals, fluid-sim stability tuning). This README is the user-facing overview.

Generates **joints** (straight segments) for robotic fish from real or synthetic **fish midlines** — curved body shapes captured over time. A single joint configuration approximates every frame of a midline within an error threshold, which is the segmentation a robotic fish would need.

## Features

- **Joint generation methods** — incremental growth, binary search, inflection-point, equal/diminishing segment counts, and brute-force area methods, using either linear (cm) or area (cm²) error.
- **CLI** — pick a method, tune the threshold, and get an SVG plot per data file (`main.py`).
- **Benchmarking** — `gather_data.py` sweeps error thresholds / amplitudes / resolutions and writes comparison CSVs.
- **Web viewer** — interactive multi-view browser app (no CDN, dependency-free Canvas JS):
  - **dark mode**: a toggle on every page (viewer, fluid, steps) switches between light and dark, persisted in `localStorage` and following the OS preference by default; canvas charts are theme-aware
  - one container per dataset, each with its own data source, method, threshold/segments sliders, 2D/3D modes, frame scrubber and play animation (ping-pong, not loop)
  - **error analysis**: per-frame error chart with area/linear toggle, stats with max errors per frame, and a segment-length table → **heatmap** view showing how each segment's chord length changes over time
  - **compare panel**: four mini line+scatter charts (avg linear error, avg area error, generation time, joints) with one curve per container so results can be compared side-by-side as you drag sliders
  - **link controls**: link frame, threshold, segments, data source, play, and algorithm across containers and drive them all from one master window
  - **export**: per-container CSV (per-frame errors + segment lengths) and PNG snapshot, plus a comparison-history CSV for the compare panel
  - **session save/load**: one click saves the whole session (containers, links, comparison curves) into a new `sessions/session_<timestamp>/` folder in the project root
- **Fluid simulation** — `midline_fluid_sim.py` runs a 2D Lattice-Boltzmann fluid-structure simulation of an articulated rigid fish body (no springs), in two movement modes: **tethered** (head oscillated in a background current) and **free swim** (the joints are actuated with a traveling wave like a robotic fish's servos and the fish propels itself through the water, reporting speed, thrust, efficiency, Strouhal number and more). The web viewer has a **fluid simulation page** (`/fluid`) that runs a generated joint configuration through it live, plus a **compare panel** that runs several generation algorithms through the *same* free-swim simulation and ranks their swimming performance. The domain has no walls — waves pass the edges as if the fish were in a larger pool.

## Project layout

| File / folder | Role |
|---|---|
| `main.py` | CLI menu, plotting/animations, step viewer |
| `core.py` | `Midline` data model + shared data utilities: loading (`.xls`/`.csv`), tracing, sine-wave synthesis, basic plotting |
| `calculate_error.py` | Error metrics (area + linear), vectorized |
| `generation_methods.py` | All joint-generation methods, parametrised by error metric |
| `generation_methods_linear_error.py` | Thin wrappers around `generation_methods.py` (linear-error threshold, cm) |
| `generation_methods_area_error.py` | Thin wrappers around `generation_methods.py` (area-error threshold, cm²) + brute-force removal |
| `method_registry.py` | Single source of truth for the methods (web API + CLI + tests) |
| `gather_data.py` | Benchmark/comparison functions → CSV |
| `midline_fluid_sim.py` | 2D Lattice-Boltzmann fluid-structure simulation (`SimConfig` holds the tuned stability constants) — tethered flume mode + **free-swim traveling-wave mode** with swimming performance metrics |
| `web_viewer.py` + `web/index.html` + `web/fluid.html` + `web/steps.html` + `web/js/` | Interactive web viewer (Flask + dependency-free JS modules) with a fluid-simulation page and a step-through growth-methods page |
| `tests/` | pytest suite (106 tests: metrics, methods, Midline, core, registry, CLI menus, gather_data, Flask API, sim incl. swim propulsion) + `frontend_smoke.mjs` (node) |
| `.github/workflows/ci.yml` | CI: ruff lint + pytest + frontend smoke on push |
| `Sturgeon from Elsa and Ted/` | Real sturgeon tracking data (midline `.xls` files, graphs, source clips) |
| `Fish_NN_Values_3D.csv`, `Fish_Raw_Values_3D.csv` | Real sturgeon tracking data (long format) |
| `sessions/` | Saved web-viewer sessions (created on demand, gitignored) |

## Data formats

- `.xls` midlines: rows = points along the body, column pairs = frames (`x y | x y | ...`).
- `.csv` tracking data: long format, columns auto-detected (`Midline_X/Y` with optional `Midline_Z`, or `Relative_X/Y`), NaN dropped, short frames padded.
- Synthetic midlines: `generate_midline_from_sinewave(...)` in `core.py`.

## Dependencies

`numpy`, `matplotlib`, `pandas`, `xlrd` (for `.xls`), `flask` (web viewer only), `pytest` (tests only). Install all at once with `pip install -r requirements.txt`, or `pip install .` for the `fishseg-cli` / `fishseg-web` / `fishseg-sim` console scripts.

## Quick start

```bash
pip install -r requirements.txt

# CLI: generate joints for all midlines in a folder, save plots
python main.py "Sturgeon from Elsa and Ted\midlines"

# quick smoke run on one of the bundled sturgeon files
python main.py "Sturgeon from Elsa and Ted\midlines" results

# fluid simulation (headless diagnostic)
python midline_fluid_sim.py --headless --steps 500

# free-swimming simulation with performance metrics
python midline_fluid_sim.py --swim --headless --steps 500

# interactive web viewer (auto-opens a browser)
python web_viewer.py "Sturgeon from Elsa and Ted\midlines" 8123

# run the test suite
python -m pytest

# frontend smoke test (needs node)
node tests/frontend_smoke.mjs

# lint gate (syntax/runtime errors + pyflakes)
pip install ruff && ruff check .
```

## CLI usage

`python main.py [data_folder] [save_folder]` shows a numbered menu:

1-4 area grow methods · 5-6 equal/diminishing segments · 7 compare method vs error · 8 2D frame animation · 9 step through joint generation · 10 3D animation · `q` quit.

## Web viewer

`python web_viewer.py [data_folder] [port]` (port defaults to 8123). Add a view per dataset with **+ Add view**, then:

- pick the growth algorithm and drag the threshold or segment-count slider — a new joint configuration is generated live
- scrub frames or hit **Play** (loops from the end back to the start, or rewinds forward-and-back when the per-card **loop** checkbox is off)
- open **Compare results** to watch the four metric charts grow per container
- use the link icons / master window to drive frame, threshold, segments, data source, play, and algorithm across containers
- hit **Save session** to store everything in a new `sessions/` folder, and **Load session** to restore it later

## Fluid simulation page

`python web_viewer.py` serves both pages; click **Fluid sim** in the main viewer header (or go to `/fluid`):

- pick the data file, growth algorithm, and threshold/segment count, plus the flow speed, **movement mode**, wave amplitude (0–40 cells) / frequency (0.0005–0.05 cycles/step, set with a **log-scale slider or typed exactly** in its text box) / wave cycles along the body (0.4–10, swim mode), and sim length (100–2400 steps)
- **movement modes**: *tethered* keeps the fish in place and oscillates its head in a background current (a flume test); *free swim* actuates the joints with a **traveling wave** of joint angles — exactly how a robotic fish's servos swim — and the fish propels itself through the periodic domain. The commanded wave is drawn as a cyan ghost over the deformed body, and the diagnostics panel reports the measured **swimming metrics**: speed (cells/step and body lengths/s), thrust, actuation power, Froude efficiency, cost of transport, Strouhal number, tail-beat amplitude, and wave-tracking error
- a **resolution toggle (low / medium / high / max)** sets the lattice size (`260×160` / `320×200` / `400×240` / `800×480` cells — low is the original default, max is double the high resolution on both axes); higher resolutions give a finer flow field and a longer body in cells, at a runtime cost
- **Run simulation** generates the joints and runs them through the 2D Lattice-Boltzmann fluid solver with the **generated segments as the fish body**; a **progress bar** under the header shows the simulation progress (step fraction, or per-run aggregate while comparing) — the heavy work runs on a background thread while the page polls
- the velocity field is shown as a colored flow map with the deformed body, the rest (segments) shape, and joints marked; **Play** animates the sim — looping end-to-start or rewinding ping-pong via the **loop** checkbox
- **Compare joint configurations**: tick any set of generation algorithms, set the shared threshold / segment-count sliders, and hit **Compare now** — the backend runs every ticked algorithm through the *identical* free-swim simulation (**in parallel threads**) and returns the results side by side: a summary table with the best value per column highlighted (speed, thrust, efficiency, cost, Strouhal, tail amplitude, tracking error), speed-vs-time and thrust-vs-time charts with one colored line per algorithm, and a run card per algorithm — click a card to load that swim into the main viewer and play it back frame by frame
- directory navigation (folder input + subfolder chips) works like the main page, and every container on the main page has a **sim** button that opens the fluid page prefilled with its data, method, and parameters

## Step-through page

Go to `/steps` to watch several growth methods build their joint configurations at the same time:

- pick the data file, set the threshold / segment-count sliders, and tick the methods to compare (brute-force methods are off by default)
- **Run** records the joint configuration after every generation step of each method (the backend replays each method's `step_callback` and returns the snapshots via `/api/steps`)
- **Forward** and **Skip 5** step all charts forward together; **Back** / **Back 5** rewind, and the step slider scrubs to any point
- the frame slider shows the joints on any frame of the fish; charts label the current joint count and final average errors, and runs that hit the step cap are flagged
