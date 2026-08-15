"""
Linear-error generation methods (metric-specific wrappers).

The growth methods are implemented once in generation_methods.py and
parametrised by metric; this module re-exports them with metric='linear'
so existing call sites keep working unchanged. The fixed-count methods
(create_equal_segments, create_diminishing_segments) are metric-free and
only re-exported here for API compatibility.
"""

__author__ = "Alex R.d Silva"
__version__ = '1.4'

import generation_methods as gm


def create_equal_segments(midline, segment_count, *frame, step_callback=None):
    """Metric-free: re-export of generation_methods.create_equal_segments."""
    return gm.create_equal_segments(midline, segment_count, *frame, step_callback=step_callback)


def create_diminishing_segments(midline, segment_count, *frame, step_callback=None):
    """Metric-free: re-export of generation_methods.create_diminishing_segments."""
    return gm.create_diminishing_segments(midline, segment_count, *frame, step_callback=step_callback)


def grow_segments(midline, error_threshold, step_callback=None):
    """Linear-error variant of generation_methods.grow_segments."""
    return gm.grow_segments(midline, error_threshold, metric='linear', step_callback=step_callback)


def grow_segments_binary_search(midline, error_threshold, step_callback=None):
    """Linear-error variant of generation_methods.grow_segments_binary_search."""
    return gm.grow_segments_binary_search(midline, error_threshold, metric='linear',
                                          step_callback=step_callback)


def grow_segments_binary_search_midpoint_only(midline, error_threshold, step_callback=None):
    """Linear-error variant of generation_methods.grow_segments_binary_search_midpoint_only."""
    return gm.grow_segments_binary_search_midpoint_only(midline, error_threshold, metric='linear',
                                                        step_callback=step_callback)


def grow_segments_from_inflection(midline, error_threshold, step_callback=None):
    """Linear-error variant of generation_methods.grow_segments_from_inflection."""
    return gm.grow_segments_from_inflection(midline, error_threshold, metric='linear',
                                            step_callback=step_callback)
