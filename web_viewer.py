"""
Interactive web viewer for fish midline joint generation.

Serves a local web page (web/index.html) plus a JSON API. The page lets you
pick a data folder / file, run any joint generation method with a live
parameter slider, scrub through frames, inspect error per frame, and view the
data in 2D or 3D - without any GUI backend or extra pip packages.

Usage:
    python web_viewer.py [data_folder] [port]

    data_folder  defaults to the current directory
    port         defaults to 8123

Endpoints:
    /api/files            ?folder=   -> {"folder", "files", "dirs"}
    /api/data             ?file=     -> downsampled midline for display
    /api/generate         ?file=&method=&param= -> joints + errors
    /api/steps            ?file=&methods=&threshold=&segments= -> joint snapshots
                            after every generation step, for the step-through page
    /api/sessions                     -> {"sessions": [...]} saved session folders
    /api/session/save     (POST)      -> creates sessions/session_<timestamp>/session.json
    /api/session/load     ?name=      -> the saved session JSON
    /fluid                            -> fluid simulation page (web/fluid.html)
    /steps                            -> step-through growth-methods page (web/steps.html)
    /api/fluid/simulate   ?file=&method=&param=&flow=&steps=&mode=&amp=&freq=&cycles= -> joints run through the LBM fluid sim
                            ('tethered' head-oscillation flume test, or 'swim' free
                            swimming with a traveling-wave joint actuation + metrics)
    /api/fluid/compare    ?file=&methods=&threshold=&segments=&flow=&steps=&amp=&freq=
                            -> runs several generation methods through the free-swim
                            simulation under identical conditions and returns per-run
                            performance metrics, time series and compact animations
"""

__author__ = "Alex R.d Silva"

import glob
import json
import math
import os
import sys
import threading
import time
import uuid
import webbrowser
import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import matplotlib
matplotlib.use("Agg")  # server must never try to open a GUI window

from flask import Flask, jsonify, request, send_file, send_from_directory

from core import load_3d_csv_data, load_3d_render_data, load_midline_data, trace_indices
import calculate_error as ce
import method_registry as reg
import midline_fluid_sim as mfs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# config: FISH_DATA_FOLDER / FISH_PORT env vars override argv, which overrides
# defaults. argv is read defensively so importing the module from a test runner
# (whose argv is the runner's own) never crashes.
def _argv_path():
    """First non-flag command line argument as the data folder, or None."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return os.path.abspath(sys.argv[1])
    return None


def _argv_port():
    """Second command line argument as the port, or None."""
    try:
        return int(sys.argv[2])
    except (IndexError, ValueError):
        return None


DATA_FOLDER = os.environ.get("FISH_DATA_FOLDER") or _argv_path() or os.getcwd()
PORT = int(os.environ.get("FISH_PORT") or _argv_port() or 8123)
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")  # saved viewer sessions, one folder each
MAX_FRAMES = 240  # frames kept for display / animation
EPS = 1e-9

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# last results of /api/generate, keyed by (file, method, param)
CACHE = {}
CACHE_MAX = 128

# last results of /api/fluid/compare, keyed by the full condition tuple
COMPARE_CACHE = {}
COMPARE_CACHE_MAX = 16

# background simulation jobs (async=1 fluid requests): the heavy generation +
# LB simulation runs in a worker thread while the page polls /api/fluid/progress
SIM_JOBS = {}
SIM_JOBS_LOCK = threading.Lock()
SIM_JOBS_MAX = 16
# worker pool for background jobs; the compare endpoint additionally runs its
# per-method simulations on a thread pool of its own (numpy releases the GIL
# for the vectorized LB passes, so the runs genuinely overlap)
SIM_EXECUTOR = ThreadPoolExecutor(max_workers=4)
SIM_WORKERS = 6  # max parallel simulations inside one compare request

# per-file caches so repeated calls don't re-read the (slow) CSV files
DISPLAY_CACHE = {}
MIDLINE_CACHE = {}


def cached(fn, store, key):
    """Returns store[key], computing and caching it with fn on a miss."""
    if key in store:
        return store[key]
    value = fn()
    store[key] = value
    return value

# methods come from the shared registry (method_registry.py) - single source of truth
METHODS = reg.METHODS
METHODS_BY_ID = reg.METHODS_BY_ID


def r(x, digits=3):
    """Rounds a number for JSON output."""
    return round(float(x), digits)


def _start_job(fn, message):
    """
    Registers a new background job and runs `fn(job_id)` on the worker pool.
    The worker must call _job_progress/_job_message to report progress and end
    with a JSON-serialisable result; exceptions become job status "error".
    :param fn: the work function, called as fn(job_id)
    :param message: initial status message
    :return: the job id
    """
    job_id = uuid.uuid4().hex[:12]
    with SIM_JOBS_LOCK:
        if len(SIM_JOBS) >= SIM_JOBS_MAX:
            # evict the oldest finished jobs first (dicts keep insertion order)
            for key in list(SIM_JOBS):
                if SIM_JOBS[key]["status"] in ("done", "error"):
                    del SIM_JOBS[key]
                    if len(SIM_JOBS) < SIM_JOBS_MAX:
                        break
        SIM_JOBS[job_id] = {"status": "running", "progress": 0.0,
                            "message": message, "result": None, "error": None,
                            "runs": {}}

    def runner():
        try:
            result = fn(job_id)
            with SIM_JOBS_LOCK:
                job = SIM_JOBS.get(job_id)
                if job:
                    job["status"] = "done"
                    job["progress"] = 1.0
                    job["message"] = "done"
                    job["result"] = result
        except Exception as e:
            with SIM_JOBS_LOCK:
                job = SIM_JOBS.get(job_id)
                if job:
                    job["status"] = "error"
                    job["error"] = str(e)

    SIM_EXECUTOR.submit(runner)
    return job_id


def _job_message(job_id, message):
    """Updates a running job's status message (no-op for sync runs, job_id None)."""
    if job_id is None:
        return
    with SIM_JOBS_LOCK:
        job = SIM_JOBS.get(job_id)
        if job and job["status"] == "running":
            job["message"] = message


def _job_run_progress(job_id, run_idx, total_runs, fraction):
    """Records one sub-run's progress and aggregates the overall fraction."""
    if job_id is None:
        return
    with SIM_JOBS_LOCK:
        job = SIM_JOBS.get(job_id)
        if not job or job["status"] != "running":
            return
        job["runs"][run_idx] = fraction
        job["progress"] = min(0.99, sum(job["runs"].values()) / total_runs)


def _job_progress(job_id, fraction):
    """Records the overall progress of a single-run job."""
    if job_id is None:
        return
    with SIM_JOBS_LOCK:
        job = SIM_JOBS.get(job_id)
        if job and job["status"] == "running":
            job["progress"] = min(0.99, fraction)


def is_pad(p):
    """True if a [x, y, z] point is zero padding (short frame filler)."""
    return all(abs(c) < EPS for c in p)


def frames_to_show(n, max_frames=MAX_FRAMES):
    """Indices of the frames to keep for display, plus the stride used."""
    if n <= max_frames:
        return list(range(n)), 1
    stride = math.ceil(n / max_frames)
    return list(range(0, n, stride)), stride


def load_display_data(file_path):
    """
    Loads a data file into the render format [frame][point][x, y, z] and
    downsamples frames for the browser.
    :param file_path: path to the .xls / .csv file
    :return: dict of display-ready data, or None on failure
    """
    render = load_3d_render_data(file_path)
    if render is None:
        return None

    num_frames = len(render)
    sel, stride = frames_to_show(num_frames)
    max_points = max(len(fr) for fr in render)

    frames, traces, actual, has_z = [], [], [], False
    for f in sel:
        fr = render[f]
        last = -1
        for k in range(len(fr)):
            if not is_pad(fr[k]):
                last = k
        n = last + 1
        pts = [[r(fr[k][0]), r(fr[k][1]), r(fr[k][2])] for k in range(n)]
        if any(p[2] != 0 for p in pts):
            has_z = True
        traces.append(trace_indices(pts))
        frames.append(pts)
        actual.append(n)

    return {
        "name": os.path.basename(file_path),
        "num_frames": num_frames,
        "num_points": max_points,
        "stride": stride,
        "frames": frames,
        "trace": traces,
        "actual": actual,
        "has_z": has_z,
    }


def resolve_file(folder, rel):
    """Maps a relative file name to an absolute path, staying inside folder."""
    abs_folder = os.path.abspath(folder)
    abs_path = os.path.abspath(os.path.join(abs_folder, rel))
    if os.path.commonpath([abs_folder, abs_path]) != abs_folder:
        return None
    return abs_path


@app.route("/")
def index():
    """Serves the main viewer page."""
    return send_file(os.path.join(BASE_DIR, "web", "index.html"))


@app.route("/fluid")
def fluid():
    """Serves the fluid simulation page."""
    return send_file(os.path.join(BASE_DIR, "web", "fluid.html"))


@app.route("/steps")
def steps():
    """Serves the step-through growth-methods page."""
    return send_file(os.path.join(BASE_DIR, "web", "steps.html"))


@app.route("/js/<path:filename>")
def web_js(filename):
    """Serves the frontend ES modules from web/js/ (traversal-safe)."""
    return send_from_directory(os.path.join(BASE_DIR, "web", "js"), filename)


@app.route("/api/meta")
def api_meta():
    """Returns the configured data folder and port."""
    return jsonify({"folder": DATA_FOLDER, "port": PORT})


@app.route("/api/methods")
def api_methods():
    """Returns the joint generation method registry for the frontend."""
    return jsonify({"methods": [
        {"id": m[0], "label": m[1], "param": m[3], "param_label": m[4], "type": m[5]}
        for m in METHODS
    ]})


@app.route("/api/files")
def api_files():
    """Lists the .xls/.csv files and subdirectories of a folder."""
    folder = request.args.get("folder") or DATA_FOLDER
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return jsonify({"error": "folder not found: " + folder}), 404

    files = sorted(glob.glob(os.path.join(folder, "*.xls")) + glob.glob(os.path.join(folder, "*.csv")))
    files = [os.path.basename(f) for f in files]
    dirs = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    return jsonify({"folder": folder, "files": files, "dirs": dirs})


@app.route("/api/data")
def api_data():
    """Returns the downsampled display data of a file (frames, traces, bounds)."""
    folder = request.args.get("folder") or DATA_FOLDER
    rel = request.args.get("file", "")
    path = resolve_file(folder, rel)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404

    data = cached(lambda: load_display_data(path), DISPLAY_CACHE, path)
    if data is None:
        return jsonify({"error": "failed to load file"}), 400
    data["file"] = rel
    data["folder"] = os.path.abspath(folder)
    return jsonify(data)


@app.route("/api/generate")
def api_generate():
    """Runs a registry method on a file and returns joints + per-frame errors."""
    folder = request.args.get("folder") or DATA_FOLDER
    rel = request.args.get("file", "")
    method_id = request.args.get("method", "")
    try:
        param = float(request.args.get("param", "0"))
    except ValueError:
        return jsonify({"error": "bad param"}), 400

    path = resolve_file(folder, rel)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    method = METHODS_BY_ID.get(method_id)
    if method is None:
        return jsonify({"error": "unknown method"}), 400

    _, _, _, _, _, ptype = method
    key = (os.path.abspath(path), method_id, param)
    cached_result = CACHE.get(key)
    if cached_result is not None:
        return jsonify(cached_result)

    if path.lower().endswith(".csv"):
        midline = cached(lambda: load_3d_csv_data(path, midline_type="midline", for_generation=True),
                         MIDLINE_CACHE, path)
    else:
        midline = cached(lambda: load_midline_data(path), MIDLINE_CACHE, path)
    if midline is None:
        return jsonify({"error": "failed to load midline"}), 400

    if ptype == "count":
        param = int(round(param))

    start = time.perf_counter()
    try:
        joints = reg.call(method, midline, param)
    except Exception as e:  # keep the server alive on method edge cases
        return jsonify({"error": f"generation failed: {e}"}), 500
    elapsed = time.perf_counter() - start

    linear, area = ce.find_per_frame_error(joints, midline)
    nf = len(midline[0])
    avg_lin = sum(linear) / nf if nf else 0.0
    avg_area = sum(area) / nf if nf else 0.0
    sel, stride = frames_to_show(nf)

    result = {
        "method": method_id,
        "param": param,
        "ptype": ptype,
        "time": r(elapsed, 4),
        "joints": [[r(x), r(y), int(p)] for x, y, p in joints],
        "count": len(joints),
        "avg_linear": r(avg_lin, 4),
        "avg_area": r(avg_area, 4),
        "stride": stride,
        "per_linear": [r(linear[i], 4) for i in sel],
        "per_area": [r(area[i], 4) for i in sel],
    }
    if len(CACHE) >= CACHE_MAX:
        CACHE.clear()
    CACHE[key] = result
    return jsonify(result)


class _StopRecording(Exception):
    """Raised by the step recorder when a method reaches its snapshot cap."""


def _record_steps(method, midline, param, max_steps):
    """
    Runs a registry method while capturing the joint configuration after every
    step (one joint added or removed) through its step_callback, so the frontend
    can replay the generation one step at a time.
    :param method: a registry METHOD entry
    :param midline: midline in generation format [point][frame][x, y]
    :param param: the method's parameter value
    :param max_steps: cap on the number of snapshots kept
    :return: (steps, truncated) - list of joint snapshots, True if the cap hit
    """
    steps = []

    def recorder(joints, midline_ref, step_index):
        if len(steps) >= max_steps:
            raise _StopRecording()
        steps.append(copy.deepcopy(joints))

    try:
        reg.call(method, midline, param, step_callback=recorder)
    except _StopRecording:
        return steps, True
    return steps, False


@app.route("/api/steps")
def api_steps():
    """
    Runs generation methods with step recording so the step-through page can
    play the joint configurations back one step at a time.
    :return: per-method joint snapshots plus the final error stats
    """
    folder = request.args.get("folder") or DATA_FOLDER
    rel = request.args.get("file", "")
    try:
        threshold = float(request.args.get("threshold", "0.5"))
        segments = max(1, int(request.args.get("segments", "10")))
        max_steps = max(1, int(request.args.get("max_steps", "400")))
    except ValueError:
        return jsonify({"error": "bad parameter"}), 400
    threshold = max(1e-9, threshold)

    path = resolve_file(folder, rel)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404

    ids = [m.strip() for m in request.args.get("methods", "").split(",") if m.strip()]
    methods = [METHODS_BY_ID[i] for i in ids if i in METHODS_BY_ID]
    if not methods:
        methods = list(METHODS)  # default: every registry method

    if path.lower().endswith(".csv"):
        midline = cached(lambda: load_3d_csv_data(path, midline_type="midline", for_generation=True),
                         MIDLINE_CACHE, path)
    else:
        midline = cached(lambda: load_midline_data(path), MIDLINE_CACHE, path)
    if midline is None:
        return jsonify({"error": "failed to load midline"}), 400

    nf = len(midline[0])
    results = []
    for method in methods:
        method_id = method[0]
        ptype = method[5]
        param = int(round(segments)) if ptype == "count" else threshold
        try:
            steps, truncated = _record_steps(method, midline, param, max_steps)
        except Exception as e:
            results.append({"id": method_id, "label": method[1], "error": f"generation failed: {e}"})
            continue
        joints = steps[-1] if steps else []
        linear, area = ce.find_per_frame_error(joints, midline)
        results.append({
            "id": method_id,
            "label": method[1],
            "param": param,
            "ptype": ptype,
            "steps": steps,
            "truncated": truncated,
            "count": len(joints),
            "avg_linear": r(sum(linear) / nf) if nf else 0.0,
            "avg_area": r(sum(area) / nf) if nf else 0.0,
        })
    return jsonify({"file": rel, "methods": results, "max_steps": max_steps})


@app.route("/api/sessions")
def api_sessions():
    """Lists saved session folders (each contains a session.json)."""
    if not os.path.isdir(SESSIONS_DIR):
        return jsonify({"sessions": []})
    names = sorted(
        d for d in os.listdir(SESSIONS_DIR)
        if os.path.isdir(os.path.join(SESSIONS_DIR, d))
        and os.path.isfile(os.path.join(SESSIONS_DIR, d, "session.json"))
    )
    return jsonify({"sessions": names})


@app.route("/api/session/save", methods=["POST"])
def api_session_save():
    """
    Saves a viewer session (cards, links, compare history) into a newly
    generated folder sessions/session_<timestamp>/ in the project root.
    :return: {"name", "path"}
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "bad session payload"}), 400
    name = "session_" + time.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(SESSIONS_DIR, name)
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "session.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
    except Exception as e:
        return jsonify({"error": "save failed: " + str(e)}), 500
    return jsonify({"name": name, "path": os.path.abspath(folder)})


@app.route("/api/session/load")
def api_session_load():
    """Loads a saved session by folder name (no path separators allowed)."""
    name = request.args.get("name", "")
    if (not name or name.startswith(".") or os.path.sep in name
            or (os.path.altsep and os.path.altsep in name)):
        return jsonify({"error": "session not found"}), 404
    path = os.path.join(SESSIONS_DIR, name, "session.json")
    if not os.path.isfile(path):
        return jsonify({"error": "session not found"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": "load failed: " + str(e)}), 500


def resample_polyline(pts, nodes):
    """
    Resamples a polyline to a fixed number of evenly spaced nodes, preserving the
    endpoints. Used to turn a joint configuration (straight segments) into a
    continuous chain of body nodes for the fluid simulation.
    :param pts: array shaped (n, 2)
    :param nodes: desired number of output nodes
    :return: array shaped (nodes, 2)
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    targets = np.linspace(0.0, total, nodes)
    out = np.empty((nodes, 2))
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum) - 2 and t > cum[j + 1]:
            j += 1
        frac = (t - cum[j]) / seg_len[j] if seg_len[j] > 1e-9 else 0.0
        out[i] = pts[j] + frac * seg[j]
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def smooth_chain(pts, window=11):
    """
    Moving-average smoothing of a polyline, used to remove the high-frequency
    tracking noise of unordered CSV data before building the sim body.
    :param pts: array shaped (n, 2)
    :param window: averaging window size
    :return: smoothed array of the same shape
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= window + 2:
        return pts
    out = np.zeros_like(pts)
    half = window // 2
    for k in range(n):
        a, b = max(0, k - half), min(n, k + half + 1)
        out[k] = pts[a:b].mean(axis=0)
    return out


def build_sim_body(midline, method_id, param, nx, ny):
    """
    Generates a joint configuration for a midline and builds the scaled sim
    body plus the joint->node mapping, shared by the single-run and compare
    endpoints. The fish body rest shape is built from the generated straight
    segments (or from the smoothed traced chain when the data is unordered).
    :param midline: midline in generation format [point][frame][x, y]
    :param method_id: registry id of the generation method
    :param param: the method's parameter value
    :param nx, ny: lattice size (body spans 55% of the smaller dimension)
    :return: dict with joints, joint_pts, rest, joint_idx, body_kind, total_cm
    :raises ValueError: on degenerate configurations / unknown methods
    """
    method = METHODS_BY_ID.get(method_id)
    if method is None:
        raise ValueError("unknown method")
    _, _, _, _, _, ptype = method
    if ptype == "count":
        param = int(round(param))
    joints = reg.call(method, midline, param)

    # order joints along the body: frame-0 points are chained by nearest
    # neighbour (CSV tracking data can be unordered; .xls rows are head->tail),
    # then joints are sorted by their position in that chain. If the generation
    # order disagrees with the chain (unordered data), the body is built from
    # the smoothed traced chain instead and the joints are kept as markers.
    # Generation-format data has NO zero padding (short frames repeat the tail),
    # so eps=0: a real point at the origin must not be dropped as "padding".
    frame0 = [midline[p][0] for p in range(len(midline))]
    chain = trace_indices(frame0, eps=0.0)
    chain_pos = {row: i for i, row in enumerate(chain)}
    seq = [chain_pos[j[2]] for j in joints if j[2] in chain_pos]
    ordered_joints = sorted(((chain_pos[j[2]], midline[j[2]][0][0], midline[j[2]][0][1])
                             for j in joints if j[2] in chain_pos))
    jpts = np.asarray([[x, y] for _, x, y in ordered_joints], dtype=float)
    if len(jpts) < 2:
        raise ValueError("generation produced fewer than 2 joints on the body")
    ordered_frac = 1.0
    if len(seq) >= 4:
        ordered_frac = sum(1 for a, b in zip(seq, seq[1:]) if b > a) / (len(seq) - 1)
    if ordered_frac < 0.75:
        chain_pts = np.asarray([frame0[r] for r in chain], dtype=float)
        body = resample_polyline(smooth_chain(chain_pts, max(9, len(chain) // 20)), 48)
        joint_pts = jpts
        body_kind = "traced"
    else:
        body = jpts
        joint_pts = jpts
        body_kind = "segments"
    total_cm = float(np.linalg.norm(np.diff(body, axis=0), axis=1).sum())
    if total_cm <= 0:
        raise ValueError("degenerate joint configuration")

    # scale the body from cm into the lattice domain
    fish_length = 0.55 * min(nx, ny)
    scale = fish_length / total_cm
    pts = body * scale
    head = (nx * 0.25, ny / 2.0)
    pts[:, 0] += head[0] - pts[0, 0]
    pts[:, 1] += head[1] - pts[0, 1]
    joint_pts = joint_pts * scale
    joint_pts[:, 0] += head[0] - body[0, 0] * scale
    joint_pts[:, 1] += head[1] - body[0, 1] * scale

    node_count = min(32, max(12, 8 * (len(jpts) - 1)))
    rest = resample_polyline(pts, node_count)
    spread_x = float(rest[:, 0].max() - rest[:, 0].min())
    spread_y = float(rest[:, 1].max() - rest[:, 1].min())
    if not np.isfinite(rest).all() or (spread_x < 2.0 and spread_y < 2.0):
        raise ValueError("degenerate body shape after scaling")

    # articulation points on the body: the rest node nearest to every joint.
    # The frontend draws the joint markers at these nodes of the DEFORMED
    # body each frame, so the dots sit exactly where the body bends.
    joint_idx = [int(np.argmin(np.linalg.norm(rest - jp, axis=1))) for jp in joint_pts]
    return {
        "joints": joints,
        "joint_pts": joint_pts,
        "rest": rest,
        "joint_idx": joint_idx,
        "body_kind": body_kind,
        "total_cm": total_cm,
    }


@app.route("/api/fluid/progress")
def api_fluid_progress():
    """
    Polls a background fluid job started with async=1: returns its status,
    progress fraction (0..1), message and - once finished - the full result
    (the same payload the synchronous endpoint would have returned).
    :return: {"status", "progress", "message", "result"/"error"}
    """
    job_id = request.args.get("job", "")
    with SIM_JOBS_LOCK:
        job = SIM_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "unknown job"}), 404
        body = {"status": job["status"], "progress": job["progress"], "message": job["message"]}
        if job["status"] == "done":
            body["result"] = job["result"]
            del SIM_JOBS[job_id]  # the result is delivered exactly once
        elif job["status"] == "error":
            body["error"] = job["error"]
            del SIM_JOBS[job_id]
    return jsonify(body)


def _run_sim_body(body, flow, steps, amp, freq, cycles, mode, job_id=None, label=None):
    """
    Builds the Simulation for an already-generated body and runs it, returning
    the full /api/fluid/simulate payload. Progress is reported through the job
    when job_id is given.
    """
    sim_cfg = mfs.SimConfig()  # tuned stability constants, single source of truth
    sim = mfs.Simulation(body["nx"], body["ny"], body["rest"], joint_indices=body["joint_idx"],
                         config=sim_cfg, target_speed=flow, osc_amp=amp,
                         osc_freq=freq, mode=mode, cycles=cycles)

    field_stride = max(4, int(round(body["nx"] / 48)))
    cb = None
    if job_id is not None:
        msg = (label + " - " if label else "") + "running fluid simulation"
        _job_message(job_id, msg)
        cb = lambda frac: _job_progress(job_id, frac)
    frames, frame_stride = mfs.collect_frames(
        sim, steps, body["joint_idx"], field_stride, frame_cap=600,
        include_target=(mode == "swim"), progress_cb=cb)

    metrics = sim.analyze() if mode == "swim" else None
    nx, ny = body["nx"], body["ny"]
    xs = list(range(0, nx, field_stride))
    ys = list(range(0, ny, field_stride))
    result = {
        "method": body["method"], "param": body["param"], "ptype": body["ptype"],
        "joints": [[r(x), r(y), int(p)] for x, y, p in body["joints"]],
        "joint_pts": np.round(body["joint_pts"], 2).tolist(),
        "rest": np.round(body["rest"], 2).tolist(),
        "body_kind": body["body_kind"],
        "nx": nx, "ny": ny, "xs": xs, "ys": ys, "stride": field_stride,
        "frames": frames, "frame_stride": frame_stride, "steps": steps,
        "sim_dt": sim.sim_dt, "total_cm": r(body["total_cm"], 2),
        "count": len(body["joints"]),
        "flow_target": flow, "osc_amp": amp, "osc_freq": freq,
        "mode": mode, "cycles": cycles,
    }
    if metrics is not None:
        result["metrics"] = metrics
    return result


@app.route("/api/fluid/simulate")
def api_fluid_simulate():
    """
    Generates a joint configuration for a data file and runs it through the 2D
    Lattice-Boltzmann fluid simulation (midline_fluid_sim). Two modes:
    'tethered' (default, head oscillated in a background current) and 'swim'
    (free swimming: the joints are actuated with a traveling wave, the fish
    propels itself and swimming metrics are reported).
    With async=1 the heavy work runs on a background thread: the response is
    {"job": id} immediately, and /api/fluid/progress?job=id is polled for
    progress and the final result (drives the page's progress bar).
    :return: downsampled animation data: velocity field per frame, body position,
             diagnostics (+ commanded wave ghost and metrics in swim mode).
    """
    folder = request.args.get("folder") or DATA_FOLDER
    rel = request.args.get("file", "")
    method_id = request.args.get("method", "")
    async_flag = request.args.get("async") == "1"
    try:
        param = float(request.args.get("param", "0"))
    except ValueError:
        return jsonify({"error": "bad param"}), 400
    try:
        flow = float(request.args.get("flow", "0.035"))
        steps = int(request.args.get("steps", "600"))
        nx = int(request.args.get("nx", "260"))
        ny = int(request.args.get("ny", "160"))
        amp = float(request.args.get("amp", "2.0"))
        freq = float(request.args.get("freq", "0.008"))
        cycles = float(request.args.get("cycles", "1.0"))
    except ValueError:
        return jsonify({"error": "bad simulation parameter"}), 400
    mode = request.args.get("mode", "tethered")
    if mode not in ("tethered", "swim"):
        return jsonify({"error": "bad mode (expected 'tethered' or 'swim')"}), 400
    flow = max(0.0, min(0.15, flow))
    steps = max(50, min(2400, steps))
    nx = max(100, min(800, nx))
    ny = max(80, min(480, ny))
    amp = max(0.0, min(40.0, amp))
    freq = max(0.0005, min(0.05, freq))
    cycles = max(0.4, min(10.0, cycles))
    # actuation envelope: amp * 2*pi * freq must stay below ~0.31 cells/step
    # or the moving bounce-back in the LB solver becomes unstable
    if amp > 0 and amp * freq > mfs.SimConfig().max_amp_freq:
        freq = mfs.SimConfig().max_amp_freq / amp

    path = resolve_file(folder, rel)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    method = METHODS_BY_ID.get(method_id)
    if method is None:
        return jsonify({"error": "unknown method"}), 400

    if path.lower().endswith(".csv"):
        midline = cached(lambda: load_3d_csv_data(path, midline_type="midline", for_generation=True),
                         MIDLINE_CACHE, path)
    else:
        midline = cached(lambda: load_midline_data(path), MIDLINE_CACHE, path)
    if midline is None:
        return jsonify({"error": "failed to load midline"}), 400

    try:
        body = build_sim_body(midline, method_id, param, nx, ny)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    body["method"] = method_id
    body["param"] = param
    body["ptype"] = method[5]
    body["nx"] = nx
    body["ny"] = ny

    if async_flag:
        job_id = _start_job(
            lambda jid: _run_sim_body(body, flow, steps, amp, freq, cycles, mode, jid),
            "generating joints and running fluid simulation")
        return jsonify({"job": job_id, "steps": steps})

    try:
        result = _run_sim_body(body, flow, steps, amp, freq, cycles, mode)
    except Exception as e:
        return jsonify({"error": f"simulation failed: {e}"}), 500
    return jsonify(result)


def _run_one_compare(midline, method, threshold, segments, flow, steps, nx, ny,
                     amp, freq, cycles, field_stride, xs, ys, sim_cfg,
                     job_id, run_idx, total_runs):
    """Runs one method of a compare request (executed on a worker thread)."""
    method_id = method[0]
    ptype = method[5]
    param = int(round(segments)) if ptype == "count" else threshold
    body = build_sim_body(midline, method_id, param, nx, ny)
    sim = mfs.Simulation(nx, ny, body["rest"], joint_indices=body["joint_idx"],
                         config=sim_cfg, target_speed=flow, osc_amp=amp,
                         osc_freq=freq, mode="swim", cycles=cycles)
    cb = None
    if job_id is not None:
        cb = lambda frac: _job_run_progress(job_id, run_idx, total_runs, frac)
    frames, frame_stride = mfs.collect_frames(
        sim, steps, body["joint_idx"], field_stride, frame_cap=120,
        include_target=True, progress_cb=cb)
    metrics = sim.analyze()
    series_stride = max(1, steps // 200)
    return {
        "id": method_id, "label": method[1], "param": param, "ptype": ptype,
        "count": len(body["joints"]), "total_cm": r(body["total_cm"], 2),
        "body_kind": body["body_kind"], "metrics": metrics,
        "series": {k: [r(v, 6) for v in sim.metrics[k][::series_stride]]
                   for k in ("t", "speed", "thrust", "power", "tail_y", "flow")},
        "frames": frames, "frame_stride": frame_stride,
        "joint_pts": np.round(body["joint_pts"], 2).tolist(),
        "rest": np.round(body["rest"], 2).tolist(),
        "nx": nx, "ny": ny, "xs": xs, "ys": ys, "stride": field_stride,
        "sim_dt": sim.sim_dt,
    }


def _run_compare(midline, entries, threshold, segments, flow, steps, nx, ny,
                 amp, freq, cycles, job_id=None):
    """
    Runs every method through the free-swim simulation and returns the compare
    result dict. The per-method simulations run on a thread pool in PARALLEL
    (numpy releases the GIL for the vectorized LB passes); progress is
    aggregated into the background job when job_id is given.
    """
    total = len(entries)
    field_stride = max(4, int(round(nx / 48)))
    xs = list(range(0, nx, field_stride))
    ys = list(range(0, ny, field_stride))
    sim_cfg = mfs.SimConfig()
    runs = [None] * total
    done = set()

    with ThreadPoolExecutor(max_workers=min(SIM_WORKERS, total)) as pool:
        futures = {}
        for idx, method in enumerate(entries):
            fut = pool.submit(_run_one_compare, midline, method, threshold,
                              segments, flow, steps, nx, ny, amp, freq, cycles,
                              field_stride, xs, ys, sim_cfg, job_id, idx, total)
            futures[fut] = idx
        for fut, idx in futures.items():
            try:
                runs[idx] = fut.result()
            except Exception as e:
                method = entries[idx]
                param = int(round(segments)) if method[5] == "count" else threshold
                runs[idx] = {"id": method[0], "label": method[1], "param": param,
                             "ptype": method[5], "error": f"{e}"}
            done.add(idx)
            if job_id is not None:
                _job_message(job_id, f"running {len(done)}/{total} free-swim simulations")

    return {
        "nx": nx, "ny": ny, "flow_target": flow, "osc_amp": amp,
        "osc_freq": freq, "cycles": cycles, "steps": steps,
        "threshold": threshold, "segments": segments,
        "runs": runs,
    }


@app.route("/api/fluid/compare")
def api_fluid_compare():
    """
    Runs several joint-generation methods through the FREE-SWIMMING fluid
    simulation under IDENTICAL conditions and returns per-run performance
    metrics, time series and compact animations, so joint configurations from
    different algorithms (or parameter values) can be compared quantitatively.
    The per-method simulations execute on a thread pool in parallel.
    With async=1 the heavy work runs on a background thread: the response is
    {"job": id} immediately, and /api/fluid/progress?job=id is polled for
    progress and the final result (drives the page's progress bar).
    :return: runs[] with metrics/series/frames, plus the shared conditions
    """
    folder = request.args.get("folder") or DATA_FOLDER
    rel = request.args.get("file", "")
    methods_csv = request.args.get("methods", "")
    async_flag = request.args.get("async") == "1"
    try:
        threshold = float(request.args.get("threshold", "0.5"))
        segments = int(request.args.get("segments", "10"))
        flow = float(request.args.get("flow", "0.0"))
        steps = int(request.args.get("steps", "400"))
        nx = int(request.args.get("nx", "260"))
        ny = int(request.args.get("ny", "160"))
        amp = float(request.args.get("amp", "2.5"))
        freq = float(request.args.get("freq", "0.012"))
        cycles = float(request.args.get("cycles", "1.0"))
    except ValueError:
        return jsonify({"error": "bad parameter"}), 400
    threshold = max(1e-9, threshold)
    segments = max(2, segments)
    flow = max(0.0, min(0.15, flow))
    steps = max(50, min(1600, steps))
    nx = max(100, min(800, nx))
    ny = max(80, min(480, ny))
    amp = max(0.5, min(40.0, amp))
    freq = max(0.0005, min(0.05, freq))
    cycles = max(0.4, min(10.0, cycles))
    if amp > 0 and amp * freq > mfs.SimConfig().max_amp_freq:
        freq = mfs.SimConfig().max_amp_freq / amp

    path = resolve_file(folder, rel)
    if not path or not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404

    ids = [m.strip() for m in methods_csv.split(",") if m.strip()]
    entries = [METHODS_BY_ID[i] for i in ids if i in METHODS_BY_ID][:8]
    if not entries:
        return jsonify({"error": "no valid methods requested"}), 400

    cache_key = (os.path.abspath(path), methods_csv, threshold, segments, flow,
                 steps, nx, ny, amp, freq, cycles)
    if cache_key in COMPARE_CACHE:
        return jsonify(COMPARE_CACHE[cache_key])

    if path.lower().endswith(".csv"):
        midline = cached(lambda: load_3d_csv_data(path, midline_type="midline", for_generation=True),
                         MIDLINE_CACHE, path)
    else:
        midline = cached(lambda: load_midline_data(path), MIDLINE_CACHE, path)
    if midline is None:
        return jsonify({"error": "failed to load midline"}), 400

    if async_flag:
        def work(jid):
            result = _run_compare(midline, entries, threshold, segments, flow,
                                  steps, nx, ny, amp, freq, cycles, jid)
            result["file"] = rel
            return result
        job_id = _start_job(work, f"running {len(entries)} free-swim simulations")
        return jsonify({"job": job_id, "runs_total": len(entries), "steps": steps})

    result = _run_compare(midline, entries, threshold, segments, flow,
                          steps, nx, ny, amp, freq, cycles)
    result["file"] = rel
    if len(COMPARE_CACHE) >= COMPARE_CACHE_MAX:
        COMPARE_CACHE.clear()
    COMPARE_CACHE[cache_key] = result
    return jsonify(result)


def main():
    """Prints the URL and starts the Flask dev server (browser opens itself)."""
    url = f"http://127.0.0.1:{PORT}"
    print("Fish Midline Segmentation Viewer")
    print(f"  data folder : {DATA_FOLDER}")
    print(f"  open        : {url}")

    def open_browser():
        """Opens the viewer URL in the default browser (best-effort)."""
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
