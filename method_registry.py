"""
Single source of truth for the joint generation methods.

Every method is one entry: (id, label, function, param name, param label,
param type). The web viewer (/api/methods, /api/generate, /api/fluid/simulate),
the CLI menus (main.py) and the tests all consume this registry, so adding a
new method is a one-line change here.
"""

__author__ = "Alex R.d Silva"

import generation_methods_area_error as gm_a
import generation_methods_linear_error as gm_l

# (id, label, function, param name, param label, param type)
METHODS = [
    ("grow_area", "grow_segments (area)", gm_a.grow_segments, "error_threshold", "error threshold (cm^2)", "err"),
    ("grow_bs_area", "grow_segments_binary_search (area)", gm_a.grow_segments_binary_search,
     "error_threshold", "error threshold (cm^2)", "err"),
    ("grow_bs_mp_area", "grow_segments_binary_search_midpoint_only (area)", gm_a.grow_segments_binary_search_midpoint_only,
     "error_threshold", "error threshold (cm^2)", "err"),
    ("grow_inflect_area", "grow_segments_from_inflection (area)", gm_a.grow_segments_from_inflection,
     "error_threshold", "error threshold (cm^2)", "err"),
    ("grow_linear", "grow_segments (linear)", gm_l.grow_segments, "error_threshold", "error threshold (cm)", "err"),
    ("grow_bs_linear", "grow_segments_binary_search (linear)", gm_l.grow_segments_binary_search,
     "error_threshold", "error threshold (cm)", "err"),
    ("grow_bs_mp_linear", "grow_segments_binary_search_midpoint_only (linear)", gm_l.grow_segments_binary_search_midpoint_only,
     "error_threshold", "error threshold (cm)", "err"),
    ("grow_inflect_linear", "grow_segments_from_inflection (linear)", gm_l.grow_segments_from_inflection,
     "error_threshold", "error threshold (cm)", "err"),
    ("equal_segments", "create_equal_segments", gm_l.create_equal_segments, "segment_count", "number of segments", "count"),
    ("diminishing_segments", "create_diminishing_segments", gm_l.create_diminishing_segments,
     "segment_count", "number of segments", "count"),
    ("brute_quantity", "generate_segments_to_quantity (area, brute force)", gm_a.generate_segments_to_quantity,
     "max_number_of_joints", "target number of joints", "count"),
    ("brute_max_area", "generate_segments_to_max_area_error (area, brute force)", gm_a.generate_segments_to_max_area_error,
     "total_area_max", "max total area error", "err"),
]

METHODS_BY_ID = {m[0]: m for m in METHODS}


def call(method, midline, value, step_callback=None):
    """
    Runs a registry method entry with the correct keyword argument.
    :param method: a METHOD entry (or its id)
    :param midline: midline in generation format [point][frame][x, y]
    :param value: the method's parameter value
    :param step_callback: optional callback passed to methods that support it
    :return: the joint configuration returned by the method
    """
    if isinstance(method, str):
        method = METHODS_BY_ID[method]
    _, _, func, param_name, _, _ = method
    return func(midline=midline, step_callback=step_callback, **{param_name: value})


def call_by_id(method_id, midline, value, step_callback=None):
    """Shorthand for call() when the registry id is known."""
    return call(METHODS_BY_ID[method_id], midline, value, step_callback=step_callback)
