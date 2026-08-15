"""CLI menu regression tests: every interactive menu runs end-to-end with
scripted input and a headless matplotlib backend (conftest sets Agg)."""

import glob
import os

import main as mn


def test_menu1_pick_method_and_save_all(synthetic_data_dir, script_input, tmp_path):
    """Menu 1: pick grow_segments (area), a threshold, a file, then quit."""
    data_dir, save_dir = synthetic_data_dir
    script_input(["1", "5", "1", "q"])
    mn.pick_method_and_save_all(data_dir, save_dir)
    svgs = glob.glob(os.path.join(save_dir, "*.svg"))
    assert svgs, "no SVG plot was saved"


def test_menu1_invalid_option_keeps_running(synthetic_data_dir, script_input):
    """Menu 1: an invalid option must not crash; q still quits."""
    data_dir, save_dir = synthetic_data_dir
    script_input(["99", "q"])
    mn.pick_method_and_save_all(data_dir, save_dir)


def test_menu7_compare_method_error(synthetic_data_dir, script_input):
    """Menu 7: compare grow_segments against increasing thresholds -> CSV + plots."""
    data_dir, save_dir = synthetic_data_dir
    script_input(["1", "q"])
    mn.compare_methods_cli(data_dir, save_dir)
    csvs = glob.glob(os.path.join(save_dir, "*_All_Data.csv"))
    assert csvs, "no comparison CSV was written"
    assert os.path.getsize(csvs[0]) > 0


def test_menu8_visualize_data_cli(synthetic_data_dir, script_input, monkeypatch):
    """Menu 8: pick a file, skip generation, skip animation."""
    data_dir, save_dir = synthetic_data_dir
    animated = []
    monkeypatch.setattr(mn, "animate_frames", lambda midline, joints, interval: animated.append(interval))
    script_input(["1", "n", "0"])
    mn.visualize_data_cli(glob.glob(os.path.join(data_dir, "*.csv")), save_dir)
    assert animated == [0.0]


def test_menu8_with_generation(synthetic_data_dir, script_input, monkeypatch):
    """Menu 8: apply grow_segments (area) with a threshold before animating."""
    data_dir, save_dir = synthetic_data_dir
    animated = []
    monkeypatch.setattr(mn, "animate_frames", lambda midline, joints, interval: animated.append(len(joints)))
    script_input(["1", "y", "1", "5", "0"])
    mn.visualize_data_cli(glob.glob(os.path.join(data_dir, "*.csv")), save_dir)
    assert animated and animated[0] >= 2


def test_menu9_step_through_generation(synthetic_data_dir, script_input, monkeypatch):
    """Menu 9: step viewer runs the full pipeline (regression for bug #7 / the
    registry-driven method list). Enter is pressed once per step."""
    data_dir, save_dir = synthetic_data_dir
    monkeypatch.setattr(mn, "animate_frames", lambda *a, **k: None)
    monkeypatch.setattr(mn.plt, "show", lambda *a, **k: None)
    # method 1 (grow_area), threshold 5, pause 0 (Enter per step), every step,
    # file 1, then one Enter per generation step, then skip the animation
    script_input(["1", "5", "0", "1", "1"] + [""] * 30 + ["0"])
    mn.visualise_generation_cli(glob.glob(os.path.join(data_dir, "*.csv")), save_dir)


def test_menu10_visualize_3d_animation_cli(synthetic_data_dir, script_input, monkeypatch):
    """Menu 10: 3D animation with a generation method applied (regression for
    bug #12 — this menu used to reference removed gm_a/gm_l imports)."""
    data_dir, save_dir = synthetic_data_dir
    rendered = []
    monkeypatch.setattr(mn, "render_3d_animation", lambda datasets, interval: rendered.append(len(datasets)))
    script_input(["1", "y", "1", "5", "0"])
    mn.visualize_3d_animation_cli(glob.glob(os.path.join(data_dir, "*.csv")))
    assert rendered == [1]


def test_step_viewer_stopped_handled(synthetic_data_dir, script_input, monkeypatch):
    """Menu 9 with a step callback that raises StepViewerStopped must exit cleanly."""
    data_dir, save_dir = synthetic_data_dir

    def stopping_viewer(*a, **k):
        raise mn.StepViewerStopped

    monkeypatch.setattr(mn, "make_step_viewer", lambda *a, **k: stopping_viewer)
    monkeypatch.setattr(mn.plt, "show", lambda *a, **k: None)
    script_input(["1", "5", "0", "1", "1"])
    mn.visualise_generation_cli(glob.glob(os.path.join(data_dir, "*.csv")), save_dir)
