"""
Error metrics between a joint configuration and the fish midline.

Provides the scalar reference implementations (find_area_error,
find_linear_error) plus the vectorized per-segment scorers (area_scores,
linear_sum_scores, linear_max_scores) and the total/per-frame error
aggregators used by the generation methods, the CLI and the web viewer.
"""

__author__ = "Alex R.d Silva"
__version__ = '1.2'

import numpy as np


def find_area_error(segment_beginning, segment_end, frame, midline):
    """
    Finds the total area between the joint segment and the actual midline.
    The error is the sum of the absolute gaps between the midline and the joint
    for each consecutive pair of points, so it grows monotonically with the
    length of the segment.
    :param segment_beginning: The start of the joint
    :param segment_end: The end of the joint
    :param frame: which column of midline data
    :param midline: The fish midline data
    :return: The total area error between the joints and midline
    """
    if segment_end <= segment_beginning:
        return 0

    joint_s = midline[segment_beginning][frame]
    joint_e = midline[segment_end][frame]

    x_diff = joint_e[0] - joint_s[0]

    if x_diff == 0:
        # vertical joint segment: error is the area between the midline and the vertical line
        total_area = 0
        for i in range(segment_beginning, segment_end):
            mp_s = midline[i][frame]
            mp_e = midline[i + 1][frame]
            gap_s = abs(mp_s[0] - joint_s[0])
            gap_e = abs(mp_e[0] - joint_s[0])
            y_diff = abs(mp_e[1] - mp_s[1])
            total_area += ((gap_s + gap_e) / 2) * y_diff
        return total_area

    gradient = (joint_e[1] - joint_s[1]) / x_diff
    intercept = joint_s[1] - gradient * joint_s[0]

    total_area = 0
    for i in range(segment_beginning, segment_end):
        mp_s = midline[i][frame]
        mp_e = midline[i + 1][frame]

        # y values of the joint segment at the x positions of the midline points
        y_joint_s = gradient * mp_s[0] + intercept
        y_joint_e = gradient * mp_e[0] + intercept

        gap_s = abs(mp_s[1] - y_joint_s)
        gap_e = abs(mp_e[1] - y_joint_e)
        delta_x = abs(mp_e[0] - mp_s[0])

        total_area += ((gap_s + gap_e) / 2) * delta_x

    return total_area


def find_linear_error(segment_beginning, segment_end, midline_point):
    """
    Finds the perpendicular distance between the joint segment and the midline point
    :param segment_beginning: The start of the joint
    :param segment_end: The end of the joint
    :param midline_point: the part of the midline we are calculating error for
    :return: The perpendicular distance (error) between the midline point and the joint segment
    """
    x1, y1 = segment_beginning[0], segment_beginning[1]
    x2, y2 = segment_end[0], segment_end[1]
    x0, y0 = midline_point[0], midline_point[1]

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        # degenerate joint segment: distance to the single point
        return float(np.hypot(x0 - x1, y0 - y1))

    # perpendicular distance from the midline point to the line through the joint
    return float(abs(dx * (y1 - y0) - dy * (x1 - x0)) / np.hypot(dx, dy))


def to_frame_major(midline):
    """
    Converts the generation format midline [point][frame][x, y] into a frame-major
    numpy array (frames, points, 2) for vectorized scoring.
    :param midline: the midline data of the fish in generation format (or a core.Midline)
    :return: numpy array shaped (frames, points, 2)
    """
    if hasattr(midline, "frame_major"):
        return midline.frame_major
    return np.asarray(midline, dtype=float).transpose(1, 0, 2)


def linear_distances(arr, b, e):
    """
    Perpendicular distances from the interior midline points (b, e) to the
    straight segment (b, e), for every frame. Mirrors find_linear_error.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames, e - b) array of distances
    """
    seg = arr[:, e] - arr[:, b]
    dx, dy = seg[:, 0], seg[:, 1]
    seg_len = np.hypot(dx, dy)
    rel = arr[:, b:e, :] - arr[:, None, b, :]
    cross = dx[:, None] * rel[:, :, 1] - dy[:, None] * rel[:, :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.abs(cross) / np.maximum(seg_len[:, None], 1e-300)
    # degenerate segments (zero length): distance to the single point
    degen = seg_len == 0
    if np.any(degen):
        dist = np.where(degen[:, None], np.hypot(rel[:, :, 0], rel[:, :, 1]), dist)
    return dist


def linear_sum_scores(arr, b, e):
    """
    Per-frame sum of the perpendicular distances of the interior midline points
    to the segment (b, e). Used for the linear component of total error.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames,) array of summed distances
    """
    if e <= b:
        return np.zeros(arr.shape[0])
    return linear_distances(arr, b, e).sum(axis=1)


def linear_max_scores(arr, b, e):
    """
    Per-frame maximum perpendicular distance of the interior midline points to
    the segment (b, e). Used by the linear growth methods.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames,) array of maximum distances
    """
    if e <= b:
        return np.zeros(arr.shape[0])
    return linear_distances(arr, b, e).max(axis=1)


def area_scores(arr, b, e):
    """
    Per-frame trapezoid area between the segment (b, e) and the midline points,
    for every frame. Mirrors find_area_error, including vertical segments.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames,) array of area errors
    """
    if e <= b:
        return np.zeros(arr.shape[0])
    js = arr[:, b]
    je = arr[:, e]
    xdiff = je[:, 0] - js[:, 0]

    seg_x = arr[:, b:e + 1, 0]
    seg_y = arr[:, b:e + 1, 1]

    # non-vertical joint segments: gap to the line y = gradient*x + intercept
    gradient = np.divide(je[:, 1] - js[:, 1], xdiff, out=np.zeros_like(xdiff), where=xdiff != 0)
    intercept = js[:, 1] - gradient * js[:, 0]
    y_joint = gradient[:, None] * seg_x + intercept[:, None]
    gap = np.where(xdiff[:, None] == 0, np.abs(seg_x - js[:, None, 0]), np.abs(seg_y - y_joint))

    # horizontal distance between consecutive midline points for non-vertical
    # segments, vertical distance for vertical ones (matches find_area_error)
    step = np.where(xdiff[:, None] == 0, np.abs(np.diff(seg_y, axis=1)), np.abs(np.diff(seg_x, axis=1)))
    pair_area = ((gap[:, :-1] + gap[:, 1:]) / 2.0) * step
    return pair_area.sum(axis=1)


def joints_with_tail(joints, midline):
    """
    Returns a local copy of the joint configuration with the tail point appended.
    Error metrics evaluate against this copy, so the caller's joints list is never
    mutated (the tail point is added internally to close the last segment).
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: a copy of joints ending with the tail point
    """
    n = len(midline)
    pts = list(joints)
    pts.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
    return pts


def find_total_area_error(joints, midline):
    """
    Finds the total area error across all joints and the midline
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: The avg total error for each frame
    """
    return find_total_error(joints, midline)[1]


def find_total_linear_error(joints, midline):
    """
    Finds the total linear error across all joints and the midline
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: the avg total linear error for each frame
    """
    return find_total_error(joints, midline)[0]


def _score_segments(joints, midline):
    """
    Accumulates the linear and area error of every joint segment across all
    frames, with the tail point appended to close the last segment. The caller's
    joints list is never mutated.
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: (linear_per_frame, area_per_frame) numpy arrays, length len(midline[0])
    """
    pts = joints_with_tail(joints, midline)
    arr = to_frame_major(midline)
    linear = np.zeros(arr.shape[0])
    area = np.zeros(arr.shape[0])
    for joint in range(len(pts) - 1):
        b, e = pts[joint][2], pts[joint + 1][2]
        linear += linear_sum_scores(arr, b, e)
        area += area_scores(arr, b, e)
    return linear, area


def find_total_error(joints, midline):
    """
    Finds the total linear and area error across all joints and the midline.
    Does not mutate the joints list.
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: an array with [total linear error, total area error]
    """
    linear, area = _score_segments(joints, midline)
    return [float(linear.mean()), float(area.mean())]


def find_per_frame_error(joints, midline):
    """
    Linear and area error for every frame, without mutating the joints list.
    :param joints: the joint configuration of the fish
    :param midline: the midline data of the fish
    :return: (linear_per_frame, area_per_frame) lists of length len(midline[0])
    """
    linear, area = _score_segments(joints, midline)
    return linear.tolist(), area.tolist()
