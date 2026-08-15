"""method_registry tests: the single source of truth stays consistent."""

import pytest

import method_registry as reg
import generation_methods_linear_error as gm_l


def test_ids_are_unique_and_consistent():
    ids = [m[0] for m in reg.METHODS]
    assert len(ids) == len(set(ids))
    for m in reg.METHODS:
        assert reg.METHODS_BY_ID[m[0]] is m
        assert callable(m[2])
        assert isinstance(m[1], str) and m[1]
        assert m[3] in ("error_threshold", "segment_count", "max_number_of_joints", "total_area_max")
        assert m[4] and m[5] in ("err", "count")


def test_call_dispatches_with_correct_kwarg(sine_midline):
    direct = gm_l.create_equal_segments(midline=sine_midline, segment_count=4)
    via_registry = reg.call_by_id("equal_segments", sine_midline, 4)
    assert via_registry == direct
    # string id and entry both work
    assert reg.call("equal_segments", sine_midline, 4) == direct
    assert reg.call(reg.METHODS_BY_ID["equal_segments"], sine_midline, 4) == direct


def test_unknown_id_raises():
    with pytest.raises(KeyError):
        reg.call_by_id("no_such_method", [], 1)


def test_every_method_accepts_step_callback(small_midline):
    calls = []
    cb = lambda joints, midline, step: calls.append(step)
    for method_id, value in (("grow_area", 1.0), ("grow_bs_linear", 1.0),
                             ("equal_segments", 4), ("brute_quantity", 4),
                             ("brute_max_area", 3000.0)):
        reg.call_by_id(method_id, small_midline, value, step_callback=cb)
    assert calls, "no step callback fired for any method"


def test_ptype_matches_param_family():
    for m in reg.METHODS:
        if m[5] == "count":
            assert m[3] in ("segment_count", "max_number_of_joints")
        else:
            assert m[3] in ("error_threshold", "total_area_max")
