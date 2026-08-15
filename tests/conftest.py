"""Shared fixtures: synthetic midlines, synthetic CSV data, real data location."""

import builtins
import os

import matplotlib

matplotlib.use("Agg")  # headless backend for every test (CLI/plot paths included)

import pandas as pd
import pytest

from core import generate_midline_from_sinewave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_DIR = os.path.join(REPO_ROOT, "Sturgeon from Elsa and Ted", "midlines")


def make_midline(resolution=120, frames=12):
    """A wavy midline in generation format [point][frame][x, y]."""
    return generate_midline_from_sinewave(cycles=1.5, amplitude=20, length_cm=100,
                                          phase_difference=0.15, frames=frames,
                                          resolution=resolution)


@pytest.fixture
def sine_midline():
    return make_midline()


@pytest.fixture
def small_midline():
    # small enough that the O(n^3) brute-force methods stay fast in tests
    return make_midline(resolution=40, frames=6)


def write_tracking_csv(path, midline):
    """Writes a long-format tracking CSV (Frame, Midline_X, Midline_Y) from a
    generation-format midline [point][frame][x, y]."""
    num_points = len(midline)
    num_frames = len(midline[0])
    rows = [(f, midline[p][f][0], midline[p][f][1])
            for f in range(num_frames) for p in range(num_points)]
    pd.DataFrame(rows, columns=["Frame", "Midline_X", "Midline_Y"]).to_csv(path, index=False)


@pytest.fixture
def synthetic_csv(tmp_path):
    midline = make_midline(resolution=80, frames=8)
    path = tmp_path / "sine.csv"
    write_tracking_csv(path, midline)
    return {"folder": str(tmp_path), "file": "sine.csv"}


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """A data folder with one synthetic CSV + an (empty) save folder, for CLI tests."""
    midline = make_midline(resolution=80, frames=8)
    data_dir = tmp_path / "data"
    save_dir = tmp_path / "save"
    data_dir.mkdir()
    save_dir.mkdir()
    write_tracking_csv(data_dir / "sine.csv", midline)
    return str(data_dir), str(save_dir)


@pytest.fixture
def real_data_dir():
    if not os.path.isdir(REAL_DATA_DIR):
        pytest.skip("real sturgeon data not present")
    return REAL_DATA_DIR


class ScriptedInput:
    """Feeds scripted answers to input(); raises if the script runs dry."""

    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, prompt=""):
        if not self.answers:
            raise AssertionError(f"input() called with no scripted answer left (prompt: {prompt!r})")
        return self.answers.pop(0)


@pytest.fixture
def script_input(monkeypatch):
    """Monkeypatches input() with a queue of answers. Usage: script_input(["1", "5", "q"])."""

    def _set(answers):
        monkeypatch.setattr(builtins, "input", ScriptedInput(answers))

    return _set
