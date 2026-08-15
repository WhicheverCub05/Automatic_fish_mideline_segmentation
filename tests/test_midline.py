"""Midline data-model tests: list compatibility + the added adapters."""

import numpy as np
import pandas as pd

import calculate_error as ce
from core import Midline, load_3d_csv_data, load_midline_data


def make_gen_midline(points=8, frames=3):
    return [[[p, f] for f in range(frames)] for p in range(points)]


def test_list_compatibility():
    gen = make_gen_midline()
    m = Midline(gen)
    assert len(m) == len(gen) == 8
    assert m[0] == gen[0]
    assert m[3][2][0] == 3 and m[3][2][1] == 2
    assert len(m[0]) == 3
    assert list(m) == gen
    assert np.asarray(m, dtype=float).shape == (8, 3, 2)
    assert m.num_points == 8 and m.num_frames == 3
    assert m.shape == (8, 3, 2)
    assert m.real_lengths is None


def test_frame_major_is_canonical():
    gen = make_gen_midline()
    m = Midline(gen)
    expected = np.asarray(gen, dtype=float).transpose(1, 0, 2)
    assert np.array_equal(m.frame_major, expected)
    assert np.array_equal(ce.to_frame_major(m), expected)
    assert m.frame_major is m.frame_major  # cached


def test_render_conversion_and_padding():
    gen = make_gen_midline()
    m = Midline(gen, real_lengths=[8, 5, 8])
    render = m.render
    assert len(render) == 3 and len(render[0]) == 8
    assert render[0][7] == [7.0, 0.0, 0.0]  # real point
    assert render[1][5] == [0.0, 0.0, 0.0]  # zero-padded beyond frame 1's real length
    assert all(abs(c) < 1e-9 for c in render[2][7]) is False


def test_csv_loader_returns_midline_with_real_lengths(tmp_path):
    rows = []
    for f in range(2):
        n = 5 if f == 0 else 3
        for p in range(n):
            rows.append((f, float(p), float(f)))
    path = tmp_path / "ragged.csv"
    pd.DataFrame(rows, columns=["Frame", "Midline_X", "Midline_Y"]).to_csv(path, index=False)

    m = load_3d_csv_data(str(path), midline_type="midline", for_generation=True)
    assert isinstance(m, Midline)
    assert m.num_frames == 2 and m.num_points == 5
    assert m.real_lengths == [5, 3]
    # tail-extension padding: frame 1's last two points repeat its tail (point 2)
    assert m[3][1] == m[2][1]
    assert m[4][1] == m[2][1]


def test_xls_loader_returns_midline(real_data_dir, tmp_path):
    import glob
    import os
    xls_files = sorted(glob.glob(os.path.join(real_data_dir, "*.xls")))
    assert xls_files
    m = load_midline_data(xls_files[0])
    assert isinstance(m, Midline)
    assert m.num_points > 10 and m.num_frames > 1
    assert m.real_lengths is None  # xls frames are rectangular, no padding
