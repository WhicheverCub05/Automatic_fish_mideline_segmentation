"""
Area-error generation methods (metric-specific wrappers).

The growth methods are implemented once in generation_methods.py and
parametrised by metric; this module re-exports them with metric='area'
so existing call sites keep working unchanged.
"""

__author__ = "Alex R.d Silva"
__version__ = '1.1'

import generation_methods as gm


def grow_segments(midline, error_threshold, step_callback=None):
    """Area-error variant of generation_methods.grow_segments."""
    return gm.grow_segments(midline, error_threshold, metric='area', step_callback=step_callback)


def grow_segments_binary_search(midline, error_threshold, step_callback=None):
    """Area-error variant of generation_methods.grow_segments_binary_search."""
    return gm.grow_segments_binary_search(midline, error_threshold, metric='area',
                                          step_callback=step_callback)


def grow_segments_binary_search_midpoint_only(midline, error_threshold, step_callback=None):
    """Area-error variant of generation_methods.grow_segments_binary_search_midpoint_only."""
    return gm.grow_segments_binary_search_midpoint_only(midline, error_threshold, metric='area',
                                                        step_callback=step_callback)


def grow_segments_from_inflection(midline, error_threshold, step_callback=None):
    """Area-error variant of generation_methods.grow_segments_from_inflection."""
    return gm.grow_segments_from_inflection(midline, error_threshold, metric='area',
                                            step_callback=step_callback)


def generate_segments_to_max_area_error(midline, total_area_max, *resolution_division, step_callback=None):
    """Area-error brute-force variant of generation_methods.generate_segments_to_max_area_error."""
    return gm.generate_segments_to_max_area_error(midline, total_area_max, *resolution_division,
                                                  step_callback=step_callback)


def generate_segments_to_quantity(midline, max_number_of_joints, *resolution_division, step_callback=None):
    """Area-error brute-force variant of generation_methods.generate_segments_to_quantity."""
    return gm.generate_segments_to_quantity(midline, max_number_of_joints, *resolution_division,
                                            step_callback=step_callback)
