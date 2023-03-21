# file contains methods calculating error
import numpy as np


# finds the area between a joint and midline
def find_area_error(start_index, end_index, frame, midline):
    # first, find area under midline
    total_area_under_midline = 0
    for m in range(end_index, start_index - 1):
        mp_s = midline[start_index][frame]
        mp_e = midline[end_index][frame]
        x_diff = abs(mp_s[0] - mp_e[0])

        if mp_s[1] < mp_e[1]:
            area_major = mp_s[1] * x_diff
        else:
            area_major = mp_e[1] * x_diff

        area_minor = (abs(mp_s[0] - mp_e[0]) * abs(mp_s[1] - mp_e[1])) / 2
        total_area_under_midline += (area_major + area_minor)

    # then find area under the line that intersects the midline
    mp_s = midline[start_index][frame]
    mp_e = midline[end_index][frame]
    x_diff = abs(mp_s[0] - mp_e[0])

    if mp_s[1] < mp_e[1]:
        total_area_under_line = mp_s[1] * x_diff
    else:
        total_area_under_line = mp_e[1] * x_diff

    # determine difference between midline area and segment line area which is the error
    if total_area_under_line > total_area_under_midline:
        area_difference = total_area_under_line - total_area_under_midline
    else:
        area_difference = total_area_under_midline - total_area_under_line

    return area_difference


# finds error by calculating distance between perpendicular of joint and midline area
def find_linear_error(segment_end, segment_beginning, midline_point):
    if segment_end[1] - segment_beginning[1] == 0 or segment_end[0] - segment_beginning[0] == 0:
        # gradient is 0 so perpendicular line is undefined
        return 0

    gradient = (segment_end[1] - segment_beginning[1]) / (segment_end[0] - segment_beginning[0])
    c = segment_end[1] - (gradient * segment_end[0])

    perpendicular_gradient = -1 / gradient
    perpendicular_c = midline_point[1] - (perpendicular_gradient * midline_point[0])

    x = abs((c - perpendicular_c) / (gradient - perpendicular_gradient))
    y = (gradient * x) + c
    # distance between 2 points on a graph
    error = abs(np.sqrt((x - midline_point[0]) ** 2 + (y - midline_point[1]) ** 2))
    return error
