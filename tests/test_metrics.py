"""Error metric tests: purity, vectorized == scalar equivalence, known values."""

import numpy as np
import pytest

import calculate_error as ce
from core import generate_midline_from_sinewave


def make_straight():
    return generate_midline_from_sinewave(cycles=0, amplitude=0, length_cm=100,
                                          phase_difference=0, frames=6, resolution=60)


def joints_for(points):
    return [[0.0, 0.0, p] for p in points]


def test_find_total_error_is_pure():
    """Metrics must never mutate the caller's joints list (bug #8 regression)."""
    m = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 12, 120)
    joints = joints_for([0, 20, 45, 70, 100, 119])
    snapshot = [list(j) for j in joints]
    ce.find_total_area_error(joints, m)
    ce.find_total_linear_error(joints, m)
    ce.find_total_error(joints, m)
    assert joints == snapshot


def test_straight_midline_zero_error():
    m = make_straight()
    joints = joints_for([0, 30, 59])
    area, linear = ce.find_total_error(joints, m)
    assert area == pytest.approx(0.0)
    assert linear == pytest.approx(0.0)


def test_find_per_frame_error_components():
    m = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 12, 120)
    joints = joints_for([0, 25, 55, 90, 119])
    per_linear, per_area = ce.find_per_frame_error(joints, m)
    nf = len(m[0])
    assert len(per_linear) == nf and len(per_area) == nf
    # find_total_error returns [linear, area] per-frame means; the per-frame
    # lists must average to them
    total_linear, total_area = ce.find_total_error(joints, m)
    assert sum(per_area) / nf == pytest.approx(total_area, rel=1e-6)
    assert sum(per_linear) / nf == pytest.approx(total_linear, rel=1e-6)
    assert all(v >= 0 for v in per_area + per_linear)


def test_area_scores_matches_scalar():
    m = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 12, 120)
    arr = ce.to_frame_major(m)
    for b, e in [(0, 10), (10, 40), (0, 119), (50, 90)]:
        vec = ce.area_scores(arr, b, e)
        for f in range(len(m[0])):
            assert vec[f] == pytest.approx(ce.find_area_error(b, e, f, m))


def test_linear_scorers_match_scalar():
    m = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 12, 120)
    arr = ce.to_frame_major(m)
    b, e = 10, 60
    dists = ce.linear_distances(arr, b, e)
    for f in range(len(m[0])):
        scalar = [ce.find_linear_error(m[b][f], m[e][f], m[p][f]) for p in range(b, e)]
        assert dists[f] == pytest.approx(np.array(scalar))
    mx = max(ce.find_linear_error(m[b][f], m[e][f], m[p][f])
             for f in range(len(m[0])) for p in range(b, e))
    assert ce.linear_max_scores(arr, b, e).max() == pytest.approx(mx)
    assert ce.linear_sum_scores(arr, b, e)[3] == pytest.approx(
        sum(ce.find_linear_error(m[b][3], m[e][3], m[p][3]) for p in range(b, e)))


def test_vertical_segment_area_error():
    """A vertical joint segment (x identical) must be scored like the scalar path.
    Joints at points 1 and 4 share x=0; the interior points 2, 3 deviate at x=1."""
    m = [[[0.0 if p in (1, 4) else 1.0, float(p)] for f in range(3)] for p in range(5)]
    joints = joints_for([1, 4])
    area, _ = ce.find_total_error(joints, m)
    assert area == pytest.approx(2.0)  # 2.0 per frame, averaged over the 3 frames
    arr = ce.to_frame_major(m)
    for f in range(3):
        assert ce.area_scores(arr, 1, 4)[f] == pytest.approx(ce.find_area_error(1, 4, f, m))


def test_degenerate_joint_segment_linear_error():
    """A zero-length joint segment falls back to the point distance."""
    m = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 6, 60)
    a = b = [3.0, 4.0]
    assert ce.find_linear_error(a, b, [0.0, 0.0]) == pytest.approx(5.0)
    assert ce.find_linear_error(a, b, a) == pytest.approx(0.0)
    assert ce.find_linear_error(m[10][0], m[10][0], m[12][0]) == pytest.approx(
        np.hypot(*(np.array(m[12][0]) - np.array(m[10][0]))))


def test_empty_and_single_point_edge_cases():
    m = [[[0.0, 0.0]]]  # 1 point x 1 frame
    area, linear = ce.find_total_error(joints_for([0]), m)
    assert area == 0.0 and linear == 0.0
    arr = ce.to_frame_major(m)
    assert ce.area_scores(arr, 0, 0).shape == (1,)
    assert ce.linear_sum_scores(arr, 0, 0).tolist() == [0.0]
