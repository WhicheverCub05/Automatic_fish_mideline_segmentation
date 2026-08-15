"""
Joint generation methods for fish midline segmentation.

The four growth methods (grow_segments, grow_segments_binary_search,
grow_segments_binary_search_midpoint_only, grow_segments_from_inflection) are
implemented ONCE here and parametrised by error metric ('area' or 'linear'),
instead of being duplicated per metric. The fixed-count methods and the
area-only brute-force methods live here too.

generation_methods_area_error.py and generation_methods_linear_error.py are
thin metric-specific wrappers around this module, so existing call sites keep
working.

All growth methods take `midline` (generation format [point][frame][x, y]) plus
one parameter, and an optional `step_callback(joints, midline, step_index)`
used to visualise the process. Every method returns a configuration that spans
the whole body head -> tail (the tail point is always the final joint).
"""

__author__ = "Alex R.d Silva"

import copy

import numpy as np

import calculate_error as ce


def _np_midline(midline):
    """
    Converts the generation format [point][frame][x, y] to a frame-major numpy
    array (frames, points, 2) so the scoring helpers can be vectorized.
    :param midline: the fish midline data (or a core.Midline)
    :return: numpy array shaped (frames, points, 2)
    """
    return ce.to_frame_major(midline)


def _area_segment_score(arr, b, e):
    """
    Per-frame area error of the segment (b, e).
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames,) array of area errors
    """
    return ce.area_scores(arr, b, e)


def _linear_segment_score(arr, b, e):
    """
    Per-frame maximum perpendicular distance between the segment (b, e) and the
    midline points between the joints.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: segment end point index
    :return: (frames,) array of maximum distances
    """
    return ce.linear_max_scores(arr, b, e)


def _area_point_score(arr, b, e, mid):
    """
    Per-frame area error of the sub-segment (b, mid), used by the midpoint-only
    search. The `e` argument is unused for the area metric.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: current search end point index (unused)
    :param mid: candidate point index
    :return: (frames,) array of area errors
    """
    return ce.area_scores(arr, b, mid)


def _linear_point_score(arr, b, e, mid):
    """
    Per-frame perpendicular distance of the point `mid` from the segment (b, e),
    used by the midpoint-only search.
    :param arr: frame-major midline array (frames, points, 2)
    :param b: segment beginning point index
    :param e: current search end point index
    :param mid: candidate point index
    :return: (frames,) array of distances
    """
    seg = arr[:, e] - arr[:, b]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    rel = arr[:, mid, :] - arr[:, b, :]
    cross = seg[:, 0] * rel[:, 1] - seg[:, 1] * rel[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.abs(cross) / np.maximum(seg_len, 1e-300)
    degen = seg_len == 0
    if np.any(degen):
        dist = np.where(degen, np.hypot(rel[:, 0], rel[:, 1]), dist)
    return dist


# each metric provides (segment_score, point_score) for the growth methods
METRICS = {
    'area': (_area_segment_score, _area_point_score),
    'linear': (_linear_segment_score, _linear_point_score),
}


def create_equal_segments(midline, segment_count, *frame, step_callback=None):
    """
    implementation of equally divided segments
    :param midline: the midline of the fish
    :param segment_count: number of segments to create
    :param frame: which midline wave/column to create segments
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each segment is added, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    joints = []
    column = 0

    if frame:
        column = frame[0]

    n = len(midline)
    if segment_count < 1:
        return joints

    # segment_count segments: joints placed at equal spacing from the head to
    # the tail, inclusive, so the configuration covers the whole body
    for i in range(segment_count + 1):
        increment = int(round(i * (n - 1) / segment_count))
        x = midline[increment][column][0]
        y = midline[increment][column][1]
        joints.append([x, y, increment])
        if step_callback is not None:
            step_callback(joints, midline, increment)

    return joints


def create_diminishing_segments(midline, segment_count, *frame, step_callback=None):
    """
    create segments of diminishing size but add up to the midline length
    :param midline: the midline of the fish
    :param segment_count: number of segments to create
    :param frame: which midline wave/column to create segments
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each segment is added, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    joints = []
    n = len(midline)
    increment = 0
    column = 0

    if frame:
        column = frame[0]

    while len(joints) < segment_count:
        joints.append([midline[increment][column][0], midline[increment][column][1], increment])
        if step_callback is not None:
            step_callback(joints, midline, increment)
        if increment >= n - 1:
            break
        # each step covers half of the remaining body, so segments shrink
        increment += max(1, (n - increment) // 2)
        increment = min(increment, n - 1)

    if joints[-1][2] != n - 1:
        joints.append([midline[n - 1][column][0], midline[n - 1][column][1], n - 1])
        if step_callback is not None:
            step_callback(joints, midline, n - 1)
    return joints


def grow_segments(midline, error_threshold, metric='area', step_callback=None):
    """
    Growth method from Dr.Otar's paper. An increment is made and compared for error for each frame.
    If the avg error is below the threshold, add an increment and compare avg error again.
    :param midline: the midline of the fish
    :param error_threshold: the maximum error between the midline and segment
                            (cm^2 for metric='area', cm for metric='linear')
    :param metric: 'area' (default) or 'linear'
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is placed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    segment_score, _ = METRICS[metric]
    arr = _np_midline(midline)
    n = len(midline)

    joints = [[midline[0][0][0], midline[0][0][1], 0]]  # contains x, y, and increment

    increments = 2  # start at 2 as first increment is going to have 0 error

    while increments < n:
        total_error = float(segment_score(arr, joints[-1][2], increments).mean())

        if total_error < error_threshold:
            increments += 1

        else:
            increments -= 1

            if increments <= joints[-1][2]:
                print(f"stuck on increment: {increments} error: {total_error}")
                break
            else:
                joints.append([midline[increments][0][0],
                               midline[increments][0][1], increments])
                if step_callback is not None:
                    step_callback(joints, midline, increments)

    # cover the whole body: the tail point closes the last segment
    if joints[-1][2] != n - 1:
        joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
        if step_callback is not None:
            step_callback(joints, midline, n - 1)

    return joints


def grow_segments_binary_search(midline, error_threshold, metric='area', step_callback=None):
    """
    Grows the segments but uses a binary search technique.
    From the current joint, a binary search finds the farthest point whose average
    error across all frames stays below the threshold. A joint is placed there
    and the process repeats until the tail is reached.
    :param midline: the midline of the fish
    :param error_threshold: the maximum error between the midline and segment
                            (cm^2 for metric='area', cm for metric='linear')
    :param metric: 'area' (default) or 'linear'
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is placed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    segment_score, _ = METRICS[metric]
    arr = _np_midline(midline)
    n = len(midline)

    joints = [[midline[0][0][0], midline[0][0][1], 0]]  # contains x, y, and increment

    def avg_error(start_idx, end_idx):
        """
        Average error between the segment and the midline across all frames.
        :param start_idx: index of the segment beginning
        :param end_idx: index of the segment end
        :return: the average error
        """
        return float(segment_score(arr, start_idx, end_idx).mean())

    completed = False

    while not completed:
        last_joint = joints[-1][2]

        # if the tail is already within the threshold, we are done
        if avg_error(last_joint, n - 1) < error_threshold:
            joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
            if step_callback is not None:
                step_callback(joints, midline, n - 1)
            break

        low = last_joint
        high = n - 1

        # binary search for the farthest point from the current joint
        # whose average error stays below the error threshold
        while low < high:
            mid = (low + high + 1) // 2
            if avg_error(last_joint, mid) < error_threshold:
                low = mid
            else:
                high = mid - 1

        if low <= last_joint:
            # no progress possible; still close the body at the tail
            if joints[-1][2] != n - 1:
                joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
                if step_callback is not None:
                    step_callback(joints, midline, n - 1)
            break

        joints.append([midline[low][0][0], midline[low][0][1], low])
        if step_callback is not None:
            step_callback(joints, midline, low)

    return joints


def grow_segments_binary_search_midpoint_only(midline, error_threshold, metric='area', step_callback=None):
    """
    Works like the binary search generation method but is greedy and
    only finds the error from one value (the middle value)
    :param midline: the midline of the fish
    :param error_threshold: the maximum error between the midline and segment
                            (cm^2 for metric='area', cm for metric='linear')
    :param metric: 'area' (default) or 'linear'
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is placed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    _, point_score = METRICS[metric]
    arr = _np_midline(midline)
    n = len(midline)

    joints = [[midline[0][0][0], midline[0][0][1], 0]]  # contains x, y, and increment

    segment_beginning = [0, 0, 0]  # x, y, midline row index
    segment_end = [0, 0, 0]

    completed = False

    while not completed:

        tmp_joints = []
        avg_joint = 0
        avg_end_error = 0
        for f in range(len(midline[0])):

            segment_beginning[0] = midline[joints[-1][2]][f][0]
            segment_beginning[1] = midline[joints[-1][2]][f][1]
            segment_beginning[2] = joints[-1][2]

            segment_end[0] = midline[n - 1][f][0]
            segment_end[1] = midline[n - 1][f][1]
            segment_end[2] = n - 1

            start = segment_beginning[2]
            end = n - 1

            segment_built = False

            while not segment_built:
                error_index = (segment_end[2] + joints[-1][2]) // 2
                error = float(point_score(arr, joints[-1][2], segment_end[2], error_index)[f])

                mid = (start + end) // 2

                if end <= start:
                    segment_built = True
                    tmp_joints.append(segment_end)
                    avg_joint += segment_end[2]

                if end == n - 1:
                    avg_end_error += error

                if error >= error_threshold:
                    end = mid - 1
                    segment_end[2] = end
                    segment_end[1] = midline[segment_end[2]][f][1]
                    segment_end[0] = midline[segment_end[2]][f][0]

                elif error < error_threshold:
                    start = mid + 1
                    if start < n:
                        segment_end[2] = start
                        segment_end[1] = midline[segment_end[2]][f][1]
                        segment_end[0] = midline[segment_end[2]][f][0]

        avg_joint = avg_joint // len(tmp_joints)

        # guard against making no progress (would loop forever)
        if avg_joint <= joints[-1][2]:
            if joints[-1][2] != n - 1:
                joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
                if step_callback is not None:
                    step_callback(joints, midline, n - 1)
            break

        joints.append([midline[avg_joint][0][0], midline[avg_joint][0][1], avg_joint])
        if step_callback is not None:
            step_callback(joints, midline, avg_joint)

        if (avg_end_error / len(midline[0])) < error_threshold:
            completed = True

    # cover the whole body: the tail point closes the last segment
    if joints[-1][2] != n - 1:
        joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
        if step_callback is not None:
            step_callback(joints, midline, n - 1)

    return joints


def grow_segments_from_inflection(midline, error_threshold, metric='area', step_callback=None):
    """
    Finds a point with the highest gradient from current joint and tries
    to add a joint if the avg error for all frames is less than threshold.
    If the joint can't be added due to high error, try out previous midline points until the joint can be added
    :param midline: the midline of the fish
    :param error_threshold: the maximum error between the midline and segment
                            (cm^2 for metric='area', cm for metric='linear')
    :param metric: 'area' (default) or 'linear'
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is placed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    segment_score, _ = METRICS[metric]
    arr = _np_midline(midline)
    n = len(midline)

    joints = [[midline[0][0][0], midline[0][0][1], 0]]

    segment_beginning = [0, 0, 0]  # x, y, midline row index
    segment_end = [0, 0, 0]

    completed = False

    while not completed:
        inflection_point = joints[-1][2] + 1
        avg_inflection_point_array = []
        for f in range(len(midline[0])):
            max_gradient = 0
            inflection_point = joints[-1][2] + 1
            segment_beginning = [midline[joints[-1][2]][f][0],
                                 midline[joints[-1][2]][f][1]]

            # find inflection and error for it
            for j in range(joints[-1][2], n - 1):

                segment_end = [midline[inflection_point][f][0],
                               midline[inflection_point][f][1]]

                if abs(segment_end[0] - segment_beginning[0]) > 0:
                    tmp_gradient = abs(
                        (segment_end[1] - segment_beginning[1]) / (segment_end[0] - segment_beginning[0]))
                else:
                    tmp_gradient = 0
                # find max gradient which is an inflection
                if tmp_gradient + 0.0005 >= max_gradient:  # added as a threshold to mitigate noise
                    max_gradient = tmp_gradient
                    inflection_point += 1
                else:
                    inflection_point -= 1
                    avg_inflection_point_array.append(inflection_point)
                    break

        joint_built = False
        if len(avg_inflection_point_array) > 0:
            avg_inflection_point = sum(avg_inflection_point_array) // len(avg_inflection_point_array)
        else:
            avg_inflection_point = 0
            joint_built = True
            completed = True

        while not joint_built:

            for j in reversed(range(joints[-1][2], avg_inflection_point + 1)):
                # try a previous segment until the error works for all frames
                total_error = float(segment_score(arr, joints[-1][2], j).mean())

                if total_error < error_threshold:
                    joint_built = True
                    break

        if inflection_point >= n:
            completed = True
        else:
            joints.append([midline[avg_inflection_point][0][0], midline[avg_inflection_point][0][1],
                           avg_inflection_point])
            if step_callback is not None:
                step_callback(joints, midline, avg_inflection_point)

    # cover the whole body: the tail point closes the last segment
    if joints[-1][2] != n - 1:
        joints.append([midline[n - 1][0][0], midline[n - 1][0][1], n - 1])
        if step_callback is not None:
            step_callback(joints, midline, n - 1)

    return joints


def generate_segments_to_max_area_error(midline, total_area_max, *resolution_division, step_callback=None):
    """
    Creates a joint configuration by starting with a joint for each midline point and greedily removing
    the joint that would increase error by the smallest amount until the total error is over the threshold
    :param midline: the midline of the fish
    :param total_area_max: the maximum total error of the joint configuration
    :param resolution_division: optional variable to reduce the resolution of the midline data
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is removed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    joints = []

    if resolution_division:
        for j in range(0, len(midline), resolution_division[0]):
            joints.append([midline[j][0][0], midline[j][0][1], j])
    else:
        for j in range(len(midline)):
            joints.append([midline[j][0][0], midline[j][0][1], j])

    total_area_error = 0

    while total_area_error <= total_area_max:
        lowest_joint_error = total_area_max
        lowest_joint_error_index = 0
        # has to start at joint that is not the start or else it'll start eating up the joints
        for joint_index in range(1, len(joints)):
            tmp_joints = copy.deepcopy(joints)
            del tmp_joints[joint_index]
            tmp_area_error = ce.find_total_area_error(tmp_joints, midline)

            if lowest_joint_error > tmp_area_error:
                lowest_joint_error = tmp_area_error
                lowest_joint_error_index = joint_index

        if lowest_joint_error_index != 0:
            del joints[lowest_joint_error_index]
        else:
            break

        if step_callback is not None:
            step_callback(joints, midline, lowest_joint_error_index)

        total_area_error = ce.find_total_error(joints, midline)[1]

    return joints


def generate_segments_to_quantity(midline, max_number_of_joints, *resolution_division, step_callback=None):
    """
    Creates a joint configuration by starting off with a joint for each midline point, and removing
    the joints that would increase the error by the least amount until there are only a given amount of joints left.
    This can take a long time so a resolution division reduces data points to speed things up at the cost of accuracy
    :param midline: the midline of the fish
    :param max_number_of_joints: the maximum number of segments to build
    :param resolution_division: optional variable to reduce the resolution of the midline data
    :param step_callback: optional function called with (joints, midline, step_index)
                          after each joint is removed, used to visualise the process
    :return: array of where the joints should be along the midline
    """
    joints = []

    if resolution_division:
        for j in range(0, len(midline), resolution_division[0]):
            joints.append([midline[j][0][0], midline[j][0][1], j])
    else:
        for j in range(len(midline)):
            joints.append([midline[j][0][0], midline[j][0][1], j])

    while len(joints) > max_number_of_joints:
        lowest_joint_error = 1000
        lowest_joint_error_index = 0
        # has to start at joint that is not the start or else it'll start eating up the joints
        for joint_index in range(1, len(joints)):
            tmp_joints = copy.deepcopy(joints)
            del tmp_joints[joint_index]
            tmp_area_error = ce.find_total_area_error(tmp_joints, midline)

            if lowest_joint_error > tmp_area_error:
                lowest_joint_error = tmp_area_error
                lowest_joint_error_index = joint_index

        if lowest_joint_error_index != 0:
            del joints[lowest_joint_error_index]
        else:
            break

        if step_callback is not None:
            step_callback(joints, midline, lowest_joint_error_index)

    return joints
