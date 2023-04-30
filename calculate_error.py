# file contains methods calculating error
import numpy as np
import copy


def find_area_error(segment_beginning, segment_end, frame, midline):
    lowest_point = 0
    area_under_midline = 0
    area_under_joint = 0

    # find lowest point on the y axis and shift frame so all y values above 0
    for i in range(len(midline)):
        if midline[i][frame][1] < lowest_point:
            lowest_point = midline[i][frame][1]

    lowest_point = abs(lowest_point)

    for i in range(segment_beginning, segment_end):
        mp_s = copy.deepcopy(midline[i][frame])
        mp_e = copy.deepcopy(midline[i + 1][frame])
        x_diff = abs(mp_e[0] - mp_s[0])

        mp_s[1] += lowest_point
        mp_e[1] += lowest_point

        if mp_s[1] < mp_e[1]:
            area_major = mp_s[1] * x_diff
        else:
            area_major = mp_e[1] * x_diff

        area_minor = (abs(mp_s[0] - mp_e[0]) * abs(mp_s[1] - mp_e[1])) / 2
        area_under_midline += (area_major + area_minor)

    # find area under line
    joint_s = copy.deepcopy(midline[segment_beginning][frame])
    joint_e = copy.deepcopy(midline[segment_end][frame])

    joint_s[1] += lowest_point
    joint_e[1] += lowest_point

    joint_x_diff = abs(joint_s[0] - joint_e[0])

    if joint_s[1] < joint_e[1]:
        joint_area_major = joint_s[1] * joint_x_diff
    else:
        joint_area_major = joint_e[1] * joint_x_diff

    joint_area_minor = (abs(joint_s[1] - joint_e[1]) * abs(joint_s[0] - joint_e[0])) / 2

    area_under_joint = joint_area_major + joint_area_minor

    error = abs(area_under_midline - area_under_joint)

    return error


# finds error by calculating distance between perpendicular of joint and midline area
def find_linear_error(segment_beginning, segment_end, midline_point):
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


def find_total_error(joints, midline):
    # for each joint starting from the tip to the end, find the cumalitve error
    total_linear_error = 0
    total_area_error = 0

    # add joint to end to accurately assess all error
    joints.append([0, 0, (len(midline) - 1)])

    # print(f"len midline: {len(midline)}, len midline[0]: {len(midline[0])}, len midline[0][0]: {len(midline[0][0])}")

    for frame in range(len(midline[0])):
        frame_linear_error = 0
        frame_area_error = 0
        for joint in range(len(joints) - 1):
            for m in range(joints[joint][2], joints[joint + 1][2]):
                frame_linear_error += find_linear_error(joints[joint], joints[joint + 1], midline[m][frame])

            frame_area_error += find_area_error(joints[joint][2], joints[joint + 1][2], frame, midline)

        # print(f"frame {frame} error - linear: {frame_linear_error}, area:{frame_area_error}")
        total_linear_error += frame_linear_error
        total_area_error += frame_area_error

    avg_linear_frame_error = total_linear_error / len(midline[0])
    avg_area_frame_error = total_area_error / len(midline[0])
    # print(
    #    f"avg linear error: {avg_linear_frame_error:.3f}, "
    #    f"avg area error: {avg_area_frame_error:.3f}")
    # print(
    #    f"total linear error: {total_linear_error:.3f}, total area error: {total_area_error:.3f}, difference: "
    #    f"{total_linear_error / total_area_error:.3f}")

    joints.pop()

    return [avg_linear_frame_error, avg_area_frame_error]
