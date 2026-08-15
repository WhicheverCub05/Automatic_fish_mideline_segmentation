"""Joint generation method tests driven by the shared registry."""

import glob
import os

import pytest

import generation_methods as gm
import generation_methods_area_error as gm_a
import generation_methods_linear_error as gm_l
import method_registry as reg
from core import load_midline_data

ALL_IDS = [m[0] for m in reg.METHODS]
ERR_IDS = [m[0] for m in reg.METHODS if m[5] == "err"]
COUNT_IDS = [m[0] for m in reg.METHODS if m[5] == "count"]
BRUTE_IDS = ["brute_quantity", "brute_max_area"]


def assert_valid_joints(joints, num_points):
    """Every method must return head -> tail joint configs with real indices."""
    assert isinstance(joints, list) and len(joints) >= 2
    indices = [int(j[2]) for j in joints]
    assert indices[0] == 0
    assert indices[-1] == num_points - 1
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)


def assert_valid_joints_partial(joints, num_points):
    """Brute-force methods stop early by design, so the tail joint is optional."""
    assert isinstance(joints, list) and len(joints) >= 1
    indices = [int(j[2]) for j in joints]
    assert indices[0] == 0
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < num_points for i in indices)


@pytest.mark.parametrize("method_id", [i for i in ERR_IDS if i not in BRUTE_IDS])
def test_error_threshold_methods_on_sine(sine_midline, method_id):
    joints = reg.call_by_id(method_id, sine_midline, 1.0)
    assert_valid_joints(joints, len(sine_midline))
    # a tighter threshold must never produce fewer joints
    tight = reg.call_by_id(method_id, sine_midline, 0.1)
    assert len(tight) >= len(joints)


@pytest.mark.parametrize("method_id", [i for i in COUNT_IDS if i not in BRUTE_IDS])
def test_count_methods_on_sine(sine_midline, method_id):
    joints = reg.call_by_id(method_id, sine_midline, 8)
    assert_valid_joints(joints, len(sine_midline))


@pytest.mark.parametrize("method_id", BRUTE_IDS)
def test_brute_force_methods_on_small(small_midline, method_id):
    value = 4 if method_id == "brute_quantity" else 2000.0
    joints = reg.call_by_id(method_id, small_midline, value)
    assert_valid_joints_partial(joints, len(small_midline))


def test_metric_wrappers_match_unified_implementation(sine_midline):
    for unified, wrapper in [(gm.grow_segments, gm_a.grow_segments),
                             (gm.grow_segments_binary_search, gm_a.grow_segments_binary_search),
                             (gm.grow_segments_binary_search_midpoint_only,
                              gm_a.grow_segments_binary_search_midpoint_only),
                             (gm.grow_segments_from_inflection, gm_a.grow_segments_from_inflection)]:
        assert wrapper(midline=sine_midline, error_threshold=2.0) == \
            unified(midline=sine_midline, error_threshold=2.0, metric="area")
    for unified, wrapper in [(gm.grow_segments, gm_l.grow_segments),
                             (gm.grow_segments_binary_search, gm_l.grow_segments_binary_search),
                             (gm.grow_segments_binary_search_midpoint_only,
                              gm_l.grow_segments_binary_search_midpoint_only),
                             (gm.grow_segments_from_inflection, gm_l.grow_segments_from_inflection)]:
        assert wrapper(midline=sine_midline, error_threshold=2.0) == \
            unified(midline=sine_midline, error_threshold=2.0, metric="linear")


def test_step_callback_fires(sine_midline):
    calls = []
    reg.call_by_id("grow_area", sine_midline, 1.0,
                   step_callback=lambda joints, m, step: calls.append(step))
    assert calls, "step callback never fired"


def test_registry_is_consistent(sine_midline):
    ids = [m[0] for m in reg.METHODS]
    assert len(ids) == len(set(ids))
    for m in reg.METHODS:
        assert reg.METHODS_BY_ID[m[0]] is m
        assert callable(m[2])
        assert m[5] in ("err", "count")
    assert reg.call_by_id("equal_segments", sine_midline, 4) == \
        gm_l.create_equal_segments(midline=sine_midline, segment_count=4)


def test_generation_on_real_data(real_data_dir):
    xls_files = sorted(glob.glob(os.path.join(real_data_dir, "*.xls")))
    assert xls_files, "no .xls files in the real data folder"
    m = load_midline_data(xls_files[0])
    assert m is not None
    assert len(m) > 10 and len(m[0]) > 1
    joints = reg.call_by_id("grow_bs_area", m, 10.0)
    assert_valid_joints(joints, len(m))


def test_two_point_midline():
    """On a 2-point midline every method can only produce head + tail."""
    from core import generate_midline_from_sinewave
    tiny = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 4, 2)
    for method_id in [i for i in ERR_IDS if i not in BRUTE_IDS] + ["equal_segments", "diminishing_segments"]:
        value = 1 if reg.METHODS_BY_ID[method_id][5] == "count" else 1.0
        joints = reg.call_by_id(method_id, tiny, value)
        assert len(joints) == 2
        assert [int(j[2]) for j in joints] == [0, 1]


def test_single_frame_midline():
    from core import generate_midline_from_sinewave
    one_frame = generate_midline_from_sinewave(1.5, 20, 100, 0.15, 1, 40)
    for method_id in [i for i in ERR_IDS if i not in BRUTE_IDS]:
        joints = reg.call_by_id(method_id, one_frame, 1.0)
        assert_valid_joints(joints, 40)
    # brute force stops early by design (threshold in total area)
    brute = reg.call_by_id("brute_max_area", one_frame, 1.0)
    assert_valid_joints_partial(brute, 40)


def test_equal_segments_extremes(sine_midline):
    one = reg.call_by_id("equal_segments", sine_midline, 1)
    assert len(one) == 2  # 1 segment -> head + tail only
    many = reg.call_by_id("equal_segments", sine_midline, 50)
    assert len(many) == 51  # more segments than points saturate at every point
    assert_valid_joints(many, len(sine_midline))


def test_diminishing_segments_saturates_at_tail(sine_midline):
    joints = reg.call_by_id("diminishing_segments", sine_midline, 100)
    assert_valid_joints(joints, len(sine_midline))
    assert len(joints) <= 12  # ~log2(120) joints, no duplicates


def test_brute_force_with_resolution_division(small_midline):
    joints = reg.call_by_id("brute_quantity", small_midline, 4)
    divided = reg.METHODS_BY_ID["brute_quantity"][2](
        small_midline, 4, 2)  # resolution_division=2 speeds up the O(n^3) search
    assert_valid_joints_partial(joints, len(small_midline))
    assert_valid_joints_partial(divided, len(small_midline))


def test_straight_midline_generation():
    """amplitude=0: a straight body should need very few joints."""
    from core import generate_midline_from_sinewave
    straight = generate_midline_from_sinewave(cycles=0, amplitude=0, length_cm=100,
                                              phase_difference=0, frames=6, resolution=60)
    for method_id in ("grow_area", "grow_bs_area", "grow_linear", "grow_bs_linear"):
        joints = reg.call_by_id(method_id, straight, 0.5)
        assert len(joints) == 2, f"{method_id} on a straight line produced {len(joints)} joints"
