"""Fluid simulation smoke tests."""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_headless_run():
    result = subprocess.run(
        [sys.executable, "midline_fluid_sim.py", "--headless", "--steps", "20"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
