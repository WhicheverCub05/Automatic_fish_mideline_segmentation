"""Flask API smoke tests via the test client (no real server needed)."""

import threading
import time

import web_viewer as wv

client = wv.app.test_client()
client.testing = True
def test_api_meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.get_json()
    assert "folder" in body and "port" in body


def test_web_js_modules():
    for name in ("helpers.js", "methods.js", "app.js", "fluid.js", "steps.js"):
        r = client.get("/js/" + name)
        assert r.status_code == 200, name
        assert b"import" in r.data or b"export" in r.data
    assert client.get("/js/helpers.js/../..").status_code in (400, 404)


def test_api_methods():
    r = client.get("/api/methods")
    assert r.status_code == 200
    methods = r.get_json()["methods"]
    assert len(methods) == len(wv.METHODS)
    assert {m["id"] for m in methods} == {m[0] for m in wv.METHODS}
    for m in methods:
        assert m["type"] in ("err", "count")


def test_api_files(synthetic_csv):
    r = client.get("/api/files", query_string={"folder": synthetic_csv["folder"]})
    assert r.status_code == 200
    assert synthetic_csv["file"] in r.get_json()["files"]


def test_api_files_missing_folder():
    r = client.get("/api/files", query_string={"folder": "C:\\definitely\\missing\\folder"})
    assert r.status_code == 404


def test_api_data(synthetic_csv):
    r = client.get("/api/data", query_string=synthetic_csv)
    assert r.status_code == 200
    body = r.get_json()
    assert body["num_frames"] > 1 and body["num_points"] > 1
    assert isinstance(body["frames"], list) and len(body["frames"]) > 0


def test_api_generate(synthetic_csv):
    params = dict(synthetic_csv, method="grow_bs_area", param="5")
    r = client.get("/api/generate", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] >= 2
    assert len(body["joints"]) == body["count"]
    assert "avg_area" in body and "avg_linear" in body


def test_api_generate_unknown_method(synthetic_csv):
    params = dict(synthetic_csv, method="no_such_method", param="5")
    assert client.get("/api/generate", query_string=params).status_code == 400


def test_api_generate_missing_file():
    params = {"folder": "C:\\definitely\\missing\\folder", "file": "x.csv",
              "method": "grow_area", "param": "5"}
    assert client.get("/api/generate", query_string=params).status_code == 404


def test_api_fluid_simulate(synthetic_csv):
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80")
    r = client.get("/api/fluid/simulate", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] >= 2
    # per-frame joints sit on the deformed body at the articulation points
    frame = body["frames"][0]
    assert len(frame["joints"]) == body["count"]
    assert len(frame["joints"][0]) == 2


def test_pages_are_served():
    assert client.get("/").status_code == 200
    assert client.get("/fluid").status_code == 200
    assert client.get("/steps").status_code == 200
    assert b"<html" in client.get("/").data.lower()


def test_api_data_missing_file():
    r = client.get("/api/data", query_string={"folder": "C:\\no\\such", "file": "x.csv"})
    assert r.status_code == 404


def test_api_data_bad_param(synthetic_csv):
    r = client.get("/api/data", query_string=dict(synthetic_csv, file="missing.csv"))
    assert r.status_code == 404


def test_api_generate_bad_param(synthetic_csv):
    params = dict(synthetic_csv, method="grow_area", param="not-a-number")
    assert client.get("/api/generate", query_string=params).status_code == 400


def test_api_steps_fast_methods(synthetic_csv):
    """Step recording returns a growing joint snapshot per generation step."""
    params = dict(synthetic_csv, methods="grow_area,equal_segments",
                  threshold="5", segments="10")
    r = client.get("/api/steps", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    by_id = {m["id"]: m for m in body["methods"]}
    assert set(by_id) == {"grow_area", "equal_segments"}
    for m in body["methods"]:
        assert "error" not in m, m.get("error")
        assert isinstance(m["steps"], list) and len(m["steps"]) >= 1
        assert all(isinstance(s, list) and len(s) >= 1 and len(s[0]) == 3 for s in m["steps"])
    # each step adds a joint, so consecutive snapshots differ
    grow = by_id["grow_area"]["steps"]
    assert len(grow) > 2
    assert grow[0] != grow[1] and grow[-1] != grow[0]
    # count matches the last snapshot
    assert by_id["grow_area"]["count"] == len(grow[-1])
    # equal segments: segment_count + 1 joints -> 11 snapshots
    assert len(by_id["equal_segments"]["steps"]) == 11


def test_api_steps_truncation_cap(synthetic_csv):
    """max_steps caps the snapshots and marks the run truncated."""
    params = dict(synthetic_csv, methods="grow_area", max_steps="2")
    r = client.get("/api/steps", query_string=params)
    body = r.get_json()
    m = body["methods"][0]
    assert m["truncated"] is True
    assert len(m["steps"]) == 2


def test_api_steps_unknown_ids_default_to_all(synthetic_csv):
    """Unknown method ids fall back to the full registry (fast under max_steps=1)."""
    params = dict(synthetic_csv, methods="no_such_method", max_steps="1")
    r = client.get("/api/steps", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["methods"]) == len(wv.METHODS)
    for m in body["methods"]:
        assert "steps" in m or "error" in m


def test_api_steps_missing_file():
    params = {"folder": "C:\\definitely\\missing\\folder", "file": "x.csv",
              "methods": "grow_area"}
    assert client.get("/api/steps", query_string=params).status_code == 404


def test_api_steps_bad_param(synthetic_csv):
    params = dict(synthetic_csv, methods="grow_area", threshold="not-a-number")
    assert client.get("/api/steps", query_string=params).status_code == 400


def test_api_steps_method_error_isolated(synthetic_csv, monkeypatch):
    """A failing method appears as an error entry without killing the request."""
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(wv.reg, "call", boom)
    params = dict(synthetic_csv, methods="grow_area,equal_segments", max_steps="1")
    r = client.get("/api/steps", query_string=params)
    assert r.status_code == 200
    assert all("error" in m for m in r.get_json()["methods"])


def test_api_fluid_bad_params(synthetic_csv):
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80", flow="not-a-number")
    assert client.get("/api/fluid/simulate", query_string=params).status_code == 400


def test_api_fluid_lattice_resolution(synthetic_csv):
    """nx/ny select the lattice resolution and are clamped to the bounds."""
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="400", ny="240")
    body = client.get("/api/fluid/simulate", query_string=params).get_json()
    assert body["nx"] == 400 and body["ny"] == 240

    # the 'max' resolution (double of high on both axes) is accepted
    params.update(nx="800", ny="480")
    body = client.get("/api/fluid/simulate", query_string=params).get_json()
    assert body["nx"] == 800 and body["ny"] == 480

    params.update(nx="9999", ny="1")
    body = client.get("/api/fluid/simulate", query_string=params).get_json()
    assert body["nx"] == 800 and body["ny"] == 80


def test_path_traversal_blocked(synthetic_csv):
    # resolve_file must keep requests inside the data folder
    r = client.get("/api/data", query_string={
        "folder": synthetic_csv["folder"],
        "file": "../" + synthetic_csv["file"]})
    assert r.status_code in (400, 404)


def test_session_save_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(wv, "SESSIONS_DIR", str(tmp_path))
    payload = {"data_folder": "C:\\data", "cards": [{"file": "a.xls", "method": "grow_area"}], "links": {}}
    r = client.post("/api/session/save", json=payload)
    assert r.status_code == 200
    name = r.get_json()["name"]
    assert name.startswith("session_")

    loaded = client.get("/api/session/load", query_string={"name": name})
    assert loaded.status_code == 200
    assert loaded.get_json() == payload

    listed = client.get("/api/sessions").get_json()["sessions"]
    assert name in listed


def test_session_load_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(wv, "SESSIONS_DIR", str(tmp_path))
    assert client.get("/api/session/load", query_string={"name": "..\\..\\x"}).status_code == 404
    assert client.get("/api/session/load", query_string={"name": ".."}).status_code == 404
    assert client.get("/api/session/load", query_string={"name": ""}).status_code == 404


def test_session_save_rejects_non_dict():
    assert client.post("/api/session/save", json=[1, 2, 3]).status_code == 400


def test_fluid_amp_freq_envelope_clamped(synthetic_csv):
    """amp * freq > SimConfig.max_amp_freq must be clamped, not rejected."""
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80", amp="8", freq="0.02")
    r = client.get("/api/fluid/simulate", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert body["osc_amp"] * body["osc_freq"] <= wv.mfs.SimConfig().max_amp_freq + 1e-9


def test_fluid_amp_freq_wide_ranges(synthetic_csv):
    """amp goes up to 40, freq up to 0.05, cycles up to 10, steps up to 2400
    (the amp*freq stability envelope still applies to freq)."""
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80", amp="15", freq="0.002")
    body = client.get("/api/fluid/simulate", query_string=params).get_json()
    assert body["osc_amp"] == 15.0  # amp is not envelope-clamped, only bounded
    assert body["osc_freq"] == 0.002

    # a huge amp*freq combination is clamped down to the stability envelope
    params.update(amp="40", freq="0.05", cycles="10", steps="3000")
    body = client.get("/api/fluid/simulate", query_string=params).get_json()
    assert body["osc_amp"] == 40.0
    assert body["osc_amp"] * body["osc_freq"] <= wv.mfs.SimConfig().max_amp_freq + 1e-9
    assert body["cycles"] == 10.0
    assert body["steps"] == 2400  # the doubled max steps cap


def test_api_fluid_simulate_swim_mode(synthetic_csv):
    """mode=swim runs the free-swimming wave and reports metrics + wave ghost."""
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="60", nx="100", ny="80", mode="swim",
                  amp="2", freq="0.014", cycles="1.0")
    r = client.get("/api/fluid/simulate", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert body["mode"] == "swim"
    assert body["cycles"] == 1.0
    assert "metrics" in body
    for key in ("speed", "speed_bl", "thrust", "power", "efficiency", "cost",
                "strouhal", "tail_amp", "track_rms", "yaw_rms", "wake", "finite"):
        assert key in body["metrics"], key
    frame = body["frames"][0]
    assert "target" in frame and len(frame["target"]) == len(frame["body"])
    assert body["metrics"]["finite"] is True


def test_api_fluid_simulate_bad_mode(synthetic_csv):
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80", mode="flapping")
    assert client.get("/api/fluid/simulate", query_string=params).status_code == 400


def test_api_fluid_compare(synthetic_csv):
    """compare runs several methods in swim mode and returns metrics/series/frames."""
    params = dict(synthetic_csv, methods="equal_segments,diminishing_segments",
                  segments="6", steps="60", nx="100", ny="80",
                  amp="2", freq="0.014")
    r = client.get("/api/fluid/compare", query_string=params)
    assert r.status_code == 200
    body = r.get_json()
    assert body["steps"] == 60
    assert len(body["runs"]) == 2
    for run in body["runs"]:
        assert "error" not in run, run.get("error")
        assert run["count"] >= 2
        assert run["metrics"]["finite"] is True
        assert set(run["series"]) >= {"t", "speed", "thrust", "power", "tail_y", "flow"}
        assert len(run["series"]["t"]) >= 1
        assert len(run["frames"]) >= 1
        assert "target" in run["frames"][0]


def test_api_fluid_compare_error_isolated(synthetic_csv, monkeypatch):
    """A crashing method appears as an error run without killing the request."""
    orig = wv.build_sim_body

    def flaky(midline, method_id, param, nx, ny):
        if method_id == "diminishing_segments":
            raise ValueError("boom")
        return orig(midline, method_id, param, nx, ny)

    monkeypatch.setattr(wv, "build_sim_body", flaky)
    params = dict(synthetic_csv, methods="equal_segments,diminishing_segments",
                  segments="6", steps="60", nx="100", ny="80",
                  amp="2", freq="0.014")
    r = client.get("/api/fluid/compare", query_string=params)
    assert r.status_code == 200
    runs = r.get_json()["runs"]
    assert len(runs) == 2
    errors = [run for run in runs if "error" in run]
    assert len(errors) == 1 and "boom" in errors[0]["error"]
    assert any("error" not in run for run in runs)


def test_api_fluid_compare_bad_params(synthetic_csv):
    params = dict(synthetic_csv, methods="equal_segments", segments="not-a-number")
    assert client.get("/api/fluid/compare", query_string=params).status_code == 400
    params = dict(synthetic_csv, methods="", segments="6")
    assert client.get("/api/fluid/compare", query_string=params).status_code == 400


def _poll_job(job, tries=150):
    """Polls /api/fluid/progress until done/error; returns the job body."""
    for _ in range(tries):
        time.sleep(0.1)
        body = client.get("/api/fluid/progress", query_string={"job": job}).get_json()
        if body["status"] in ("done", "error"):
            return body
    return body


def test_api_fluid_async_simulate_progress(synthetic_csv):
    """async=1 returns a job id immediately; polling delivers progress + result."""
    params = dict(synthetic_csv, method="equal_segments", param="6",
                  steps="50", nx="100", ny="80", mode="swim", amp="2", freq="0.014")
    r = client.get("/api/fluid/simulate", query_string={**params, "async": "1"})
    assert r.status_code == 200
    job = r.get_json()["job"]
    body = _poll_job(job)
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    result = body["result"]
    assert result["count"] >= 2
    assert result["mode"] == "swim"
    # the result is delivered exactly once
    assert client.get("/api/fluid/progress", query_string={"job": job}).status_code == 404


def test_api_fluid_async_compare(synthetic_csv):
    """async=1 compare runs on a background thread and resolves via polling."""
    params = dict(synthetic_csv, methods="equal_segments,diminishing_segments",
                  segments="6", steps="50", nx="100", ny="80", amp="2", freq="0.014")
    r = client.get("/api/fluid/compare", query_string={**params, "async": "1"})
    assert r.status_code == 200
    job = r.get_json()["job"]
    body = _poll_job(job)
    assert body["status"] == "done"
    result = body["result"]
    assert len(result["runs"]) == 2
    assert all("error" not in run for run in result["runs"])
    assert result["file"] == synthetic_csv["file"]


def test_api_fluid_progress_unknown_job():
    assert client.get("/api/fluid/progress", query_string={"job": "nope"}).status_code == 404


def test_compare_runs_in_parallel_threads(synthetic_csv, monkeypatch):
    """The per-method simulations of one compare must overlap on distinct
    threads (threading is used whenever multiple sims are computed)."""
    thread_ids = []
    orig = wv.mfs.collect_frames

    def wrapped(*a, **k):
        thread_ids.append(threading.get_ident())
        time.sleep(0.3)  # hold the worker so the two runs must overlap
        return orig(*a, **k)

    monkeypatch.setattr(wv.mfs, "collect_frames", wrapped)
    params = dict(synthetic_csv, methods="equal_segments,diminishing_segments",
                  segments="6", steps="50", nx="100", ny="80", amp="2", freq="0.014")
    r = client.get("/api/fluid/compare", query_string=params)
    assert r.status_code == 200
    assert len(set(thread_ids)) == 2, f"expected 2 distinct worker threads, got {thread_ids}"
