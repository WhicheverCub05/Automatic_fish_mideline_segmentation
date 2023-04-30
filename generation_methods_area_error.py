# file contains methods of creating segments using area
import calculate_error as ce


# growth method from Dr.Otar's paper. An increment is made and compared for max error for each frame.
# If the avg error is below the threshold, add an increment and compare avg max error.
def grow_segments(midline, error_threshold_area):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([midline[0][0][0], midline[0][0][1], 0])  # contains x, y, and increment

    segment_beginning = [0, 0]
    segment_end = [0, 0]
    increments = 2  # start at 2 as first increment is going to have 0 error

    while increments < len(midline):
        total_error = 0

        for f in range(len(midline[0])):
            frame_error = 0

            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]

            segment_end[0] = midline[increments][f][0]
            segment_end[1] = midline[increments][f][1]
            # get error between joint and increments

            frame_error = ce.find_area_error(joints[len(joints)-1][2], increments, f, midline)

            total_error += frame_error

        total_error /= len(midline[0])  # avg of error for that joint, for all frames.

        if total_error < error_threshold_area:
            # print("ye: ", increments, " f: ", f, " error: ", total_error)
            increments += 1

        elif total_error >= error_threshold_area:
            increments -= 1

            if increments <= joints[len(joints) - 1][2]:
                print("stuck on increment: ", increments, "error: ", total_error, "segment_beginning: ",
                      segment_beginning, "segment_end: ", segment_end)
                break
            else:
                joints.append([midline[increments][0][0],
                               midline[increments][0][1], increments])
                # print("Adding joint: ", joints[len(joints) - 1])

        else:
            print("Houston, we have a problem")

    return joints


# Grows the segments but uses a binary search technique.
# Joints are compared from start to the end of the midline, and halved if max error is over threshold
def grow_segments_binary_search(midline, error_threshold_area):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([midline[0][0][0], midline[0][0][1], 0])  # contains x, y, and increment

    segment_beginning = [0, 0, 0]  # x, y, midline row index
    segment_end = [0, 0, 0]

    completed = False

    while not completed:

        tmp_joints = [[0 for _ in range(3)] for _ in range(0)]
        avg_joint = 0
        avg_end_error = 0
        for f in range(len(midline[0])):

            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]
            segment_beginning[2] = joints[len(joints) - 1][2]

            segment_end[0] = midline[len(midline) - 1][f][0]
            segment_end[1] = midline[len(midline) - 1][f][1]
            segment_end[2] = len(midline) - 1

            start = segment_beginning[2]
            end = len(midline) - 1

            divisions = 1

            segment_built = False

            while not segment_built:
                error = ce.find_area_error(joints[len(joints)-1][2], (len(midline) - 1), f, midline)

                mid = (start + end) // 2

                if end <= start:
                    segment_built = True
                    tmp_joints.append(segment_end)
                    avg_joint += segment_end[2]

                if end == len(midline) - 1:
                    avg_end_error += error

                if error >= error_threshold_area:
                    end = mid - 1
                    segment_end[2] = end  # was int(midline_range / 2 ** divisions)
                    segment_end[1] = midline[segment_end[2]][f][1]
                    segment_end[0] = midline[segment_end[2]][f][0]
                    divisions += 1

                elif error < error_threshold_area:
                    start = mid + 1
                    if start < len(midline):
                        segment_end[2] = start  # was int(midline_range / 2 ** divisions)
                        segment_end[1] = midline[segment_end[2]][f][1]
                        segment_end[0] = midline[segment_end[2]][f][0]
                        divisions += 1

        avg_joint = avg_joint // len(tmp_joints)

        joints.append([midline[avg_joint][0][0], midline[avg_joint][0][1], avg_joint])

        # print("Adding joint:", joints[len(joints) - 1])

        if (avg_end_error / len(midline[0])) < error_threshold_area:
            # print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
            completed = True

    # joints.pop()
    return joints


# another option - for each joint, generate segments for 1 frame. try segment on other frames and reduce size as needed

# optimises the generation method by using greedy binary search
def grow_segments_binary_search_midpoint_only(midline, error_threshold_area):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([midline[0][0][0], midline[0][0][1], 0])  # contains x, y, and increment

    segment_beginning = [0, 0, 0]  # x, y, midline row index
    segment_end = [0, 0, 0]

    completed = False

    while not completed:

        tmp_joints = [[0 for _ in range(3)] for _ in range(0)]
        avg_joint = 0
        avg_end_error = 0
        for f in range(len(midline[0])):

            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]
            segment_beginning[2] = joints[len(joints) - 1][2]

            segment_end[0] = midline[len(midline) - 1][f][0]
            segment_end[1] = midline[len(midline) - 1][f][1]
            segment_end[2] = len(midline) - 1

            start = segment_beginning[2]
            end = len(midline) - 1

            divisions = 1

            segment_built = False

            while not segment_built:
                error_index = (segment_end[2] + joints[len(joints) - 1][2]) // 2
                error = ce.find_area_error(joints[len(joints) - 1][2], error_index, f, midline)

                mid = (start + end) // 2

                if end <= start:
                    segment_built = True
                    tmp_joints.append(segment_end)
                    avg_joint += segment_end[2]

                if end == len(midline) - 1:
                    avg_end_error += error

                if error >= error_threshold_area:
                    end = mid - 1
                    segment_end[2] = end  # was int(midline_range / 2 ** divisions)
                    segment_end[1] = midline[segment_end[2]][f][1]
                    segment_end[0] = midline[segment_end[2]][f][0]
                    divisions += 1

                elif error < error_threshold_area:
                    start = mid + 1
                    if start < len(midline):
                        segment_end[2] = start  # was int(midline_range / 2 ** divisions)
                        segment_end[1] = midline[segment_end[2]][f][1]
                        segment_end[0] = midline[segment_end[2]][f][0]
                        divisions += 1

        avg_joint = avg_joint // len(tmp_joints)

        joints.append([midline[avg_joint][0][0], midline[avg_joint][0][1], avg_joint])

        # print("Adding joint:", joints[len(joints) - 1])

        if (avg_end_error / len(midline[0])) < error_threshold_area:
            # print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
            completed = True
    # joints.pop()

    return joints


# finds a point with the highest gradient from current joint
# and tries to add a joint if the avg error for all frames is less than threshold.
# if the joint can't be added due to high error, try out previous midline points until the joint can be added
def grow_segments_from_inflection(midline, error_threshold_area):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([midline[0][0][0], midline[0][0][1], 0])

    segment_beginning = [0, 0, 0]  # x, y, midline row index
    segment_end = [0, 0, 0]

    completed = False

    while not completed:
        inflection_point = joints[len(joints) - 1][2] + 1
        total_error = 0
        tmp_error = 0
        avg_inflection_point_array = []
        for f in range(len(midline[0])):
            frame_max_error = 0
            max_gradient = 0
            inflection_point = joints[len(joints) - 1][2] + 1
            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]

            # find inflection and error for it
            for j in range(joints[len(joints) - 1][2], len(midline) - 1):

                segment_end[0] = midline[inflection_point][f][0]
                segment_end[1] = midline[inflection_point][f][1]

                if abs(segment_end[0] - segment_beginning[0]) > 0:
                    tmp_gradient = abs((segment_end[1] - segment_beginning[1])/(segment_end[0] - segment_beginning[0]))
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
            continue

        joint_built = False
        # print("avg_inflection_point_array:", avg_inflection_point_array)
        if len(avg_inflection_point_array) > 0:
            avg_inflection_point = sum(avg_inflection_point_array) // len(avg_inflection_point_array)
        else:
            avg_inflection_point = 0
            joint_built = True
            completed = True

        while not joint_built:

            total_error = 0
            for j in reversed(range(joints[len(joints) - 1][2], avg_inflection_point + 1)):
                # try a previous segment until the error works for all frames
                for f in range(len(midline[0])):
                    segment_beginning = midline[joints[(len(joints)-1)][2]][f]
                    segment_end = midline[j][f]

                    frame_max_error = ce.find_area_error(joints[len(joints)-1][2], j, f, midline)

                    total_error += frame_max_error

                total_error /= len(midline[0])

                if total_error < error_threshold_area:
                    joint_built = True
                    break

        if inflection_point >= len(midline):
            completed = True
        else:
            joints.append([midline[inflection_point][0][0], midline[inflection_point][0][1], avg_inflection_point])
            # print("inflection_point to add joint:", avg_inflection_point)

    return joints
