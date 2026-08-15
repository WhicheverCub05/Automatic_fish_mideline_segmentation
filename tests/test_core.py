"""core.py unit tests: tracing, loaders round-trip, plot helpers."""


import pandas as pd

from core import (is_padding_point, joints_to_length, load_3d_csv_data,
                  load_3d_render_data, midline_segments, plot_midline, trace_all_frames)


def test_is_padding_point():
    assert is_padding_point([0.0, 0.0, 0.0])
    assert is_padding_point((1e-12, 0.0, -1e-12))
    assert not is_padding_point([0.1, 0.0, 0.0])


def test_midline_segments_orders_unordered_points():
    # deliberately scrambled row order; padding points must be skipped
    pts = [[10.0, 10.0], [4.0, 4.0], [8.0, 8.0], [0.0, 0.0], [6.0, 6.0], [0.0, 0.0], [2.0, 2.0]]
    chain = midline_segments(pts)
    assert chain == [[10.0, 10.0], [8.0, 8.0], [6.0, 6.0], [4.0, 4.0], [2.0, 2.0]]
    assert all(not is_padding_point(p) for p in chain)  # zero padding never included


def test_trace_all_frames_with_padding():
    frames = [
        [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0], [0.0, 0.0]],
        [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
    ]
    traces = trace_all_frames(frames)
    assert len(traces) == 2
    assert traces[0] == [[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]
    assert traces[1] == [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]


def test_csv_render_format_round_trip(tmp_path):
    rows = []
    for f in range(2):
        n = 4 if f == 0 else 2
        for p in range(n):
            rows.append((f, float(p), float(f)))
    path = tmp_path / "r.csv"
    pd.DataFrame(rows, columns=["Frame", "Midline_X", "Midline_Y"]).to_csv(path, index=False)

    gen = load_3d_csv_data(str(path), midline_type="midline", for_generation=True)
    render = load_3d_csv_data(str(path), midline_type="midline", for_generation=False)
    assert len(render) == 2 and len(render[0]) == 4
    assert render[1][2] == [0.0, 0.0, 0.0]  # frame 1 padded with zeros
    assert not is_padding_point(render[1][1])
    # generation format tail-extension agrees with the real frame-1 points
    assert gen[3][1] == gen[1][1]

    # the Midline.render adapter must produce the same render format
    assert render == gen.render


def test_load_3d_render_data_csv(tmp_path):
    rows = [(f, float(p), float(f)) for f in range(2) for p in range(3)]
    path = tmp_path / "r.csv"
    pd.DataFrame(rows, columns=["Frame", "Midline_X", "Midline_Y"]).to_csv(path, index=False)
    data = load_3d_render_data(str(path))
    assert data is not None
    assert len(data) == 2 and len(data[0]) == 3 and len(data[0][0]) == 3


def test_plot_midline_and_joints_to_length_smoke(sine_midline):
    plot_midline(sine_midline, 0)
    joints = [[0.0, 0.0, 0], [5.0, 5.0, 40], [10.0, 10.0, 80], [15.0, 15.0, 119]]
    joints_to_length(joints, True)
    joints_to_length(joints)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_load_missing_file_returns_none(tmp_path):
    assert load_3d_csv_data(str(tmp_path / "missing.csv")) is None
    assert load_3d_render_data(str(tmp_path / "missing.csv")) is None
