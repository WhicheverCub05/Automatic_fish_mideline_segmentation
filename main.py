# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# implementation of growth algorithm


# get data from every other line for now
def load_midline_data(location):
    file_data = pd.read_excel(location)
    dimensions = file_data.shape
    print("Loading:", location)
    print("shape: ", dimensions)

    midline = [[[0 for _ in range(2)] for _ in range(dimensions[1] // 2)] for _ in range(dimensions[0])]

    for column in range(0, dimensions[1], 2):
        for row in range(dimensions[0]):
            x = file_data.iat[row, column]
            y = file_data.iat[row, column + 1]
            midline[row][column // 2][0] = x
            midline[row][column // 2][1] = y

    return midline


def plot_midline(midline, *columns):
    if columns:
        for c in columns:
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][c][0])
                y.append(midline[s][c][1])
            plt.plot(x, y)

    else:
        for c in range(len(midline[0])):
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][c][0])
                y.append(midline[s][c][1])
            plt.plot(x, y)

    plt.xlabel("x")
    plt.ylabel("y")


def generate_segments(midline, error_threshold):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    joints.append([midline[0][0][0], midline[0][0][1], 0])  # contains x, y, and increment

    segment_beginning = [0, 0]
    segment_end = [0, 0]
    increments = 1

    while increments < len(midline):  # number of rows
        error = 0.0

        for f in range(1):  # len(midline[0]) all columns

            segment_beginning[0] = midline[joints[len(joints) - 1][2]][f][0]
            segment_beginning[1] = midline[joints[len(joints) - 1][2]][f][1]

            segment_end[0] = midline[increments][f][0]
            segment_end[1] = midline[increments][f][1]

            for i in range(joints[len(joints) - 1][2], increments, 1):
                a = abs((segment_end[1] - segment_beginning[1]) * midline[i][f][0]
                        - (segment_end[0] - segment_beginning[0])
                        * midline[i][f][1] + segment_beginning[0] * segment_end[1] - segment_end[0] *
                        segment_beginning[1])
                b = math.sqrt((segment_end[0] - segment_beginning[0])
                              ** 2 + (segment_end[1] - segment_beginning[1]) ** 2)

                if error < abs(a / b) / 100 and (a != 0.0 and b != 0.0):
                    error = abs(a / b) / 100

                plt.scatter(i, error)

        # error /= 3
        # error /= len(midline[0])

        if error < error_threshold:
            print("ye: ", increments, " error: ", error)
            increments += 1

        elif error >= error_threshold:
            increments -= 1

            if increments <= joints[len(joints) - 1][2]:
                print("stuck on increment: ", increments, "error: ", error, "Sb: ", segment_beginning,
                      "Se: ", segment_end)
                break
            else:
                joints.append([midline[increments][0][0],
                               midline[increments][0][1], increments])
                # print("Adding joint: ", joints[len(joints) - 1])

        else:
            print("Houston, we have a problem")

    return joints


# implementation of equally divided segments
def create_equal_segments(segment_count, midline, *frame):
    print("\n--Generating equally spaced segments--\n")
    joints = [[0 for _ in range(3)] for _ in range(0)]
    column = 0

    if frame:
        column = frame

    for i in range(segment_count):
        increment = int((i / segment_count) * len(midline))
        x = midline[increment][column][0]
        y = midline[increment][column][1]
        joints.append([x, y, increment])

    return joints


# create segments of diminishing size but add up to 1
def create_diminishing_segments(segment_count, midline, frame):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    length = len(midline)
    increment = 0
    for i in range(segment_count):
        x = midline[increment][frame][0]
        y = midline[increment][frame][1]
        joints.append([x, y, increment])
        increment += length // 2
        length = length // 2
        print("increment: ", increment)
    return joints


def math_test_bench():
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    # calculate max distance between points

    data = [[0.7, 1.0], [1.7, 1.5], [2.3, 2.7], [2.8, 3.7], [3.7, 4]]

    for d in data:
        plt.plot(d[0], d[1], 'bo', ls='--')

    # calculate error if data[2] was Se

    joints = [0]
    joints[0] = data[0]

    print("joints: ", joints)

    start = 0
    end = 4

    Sb = data[start]
    Se = data[end]

    intersection = [0, 0]

    y1 = Sb[1]
    y2 = Se[1]

    x1 = Sb[0]
    x2 = Se[0]

    print("y1:", y1, " y2:", y2, " x1:", x1, " x2:", x2)
    gr = (y2 - y1) / (x2 - x1)

    c = y1 - (gr * x1)

    x = np.linspace(0, 4, 100)
    y = (gr * x) + c
    plt.plot(x, y, '-r')

    # iterate from Sb index to Se index, incrementing Mp index
    for i in range(start + 1, end, 1):
        Mp = data[i]

        ym = ((-1 / gr) * x) + Mp[1] - ((-1 / gr) * Mp[0])

        plt.plot(x, ym, '-b')

        error = find_error(Se, Sb, Mp)

        # error is the distance between the two points

    plt.show()


def find_error(Se, Sb, Mp):
    gradient = (Se[1] - Sb[1]) / (Se[0] - Sb[0])
    c = Se[1] - (gradient * Se[0])

    if gradient == 0:
        # gradient is 0 so perpendicular line is undefined
        return 0

    perpendicular_gradient = -1 / gradient

    perpendicular_c = Mp[1] - (perpendicular_gradient * Mp[0])

    x = abs((c - perpendicular_c) / (gradient - perpendicular_gradient))
    y = (gradient * x) + c

    error = abs(np.sqrt((x - Mp[0]) ** 2 + (y - Mp[1]) ** 2))

    # print("Sb: ", Sb, " Se: ", Se, "Mp: ", Mp, "intersection x:", x, "y:", y, " error: ", error)

    # plt.scatter(x, y)

    return error


# another option - for each joint, generate segments for 1 frame. try segment on other frames and reduce size as needed

# optimises the generation method by using greedy binary search
def grow_segments_divide_and_conquer(midline, error_threshold):
    print("\n--Generating segments by growing divide & conquer--\n")

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
            segment_beginning[2] = joints[len(joints)-1][2]

            segment_end[0] = midline[len(midline)-1][f][0]
            segment_end[1] = midline[len(midline)-1][f][1]
            segment_end[2] = len(midline)-1

            start = segment_beginning[2]
            end = len(midline) - 1

            divisions = 1

            segment_built = False

            while not segment_built:
                error_index = (segment_end[2] + joints[len(joints)-1][2]) // 2
                error = find_error(segment_end, segment_beginning, midline[error_index][f])

                mid = (start + end) // 2

                if end <= start:
                    segment_built = True
                    tmp_joints.append(segment_end)
                    avg_joint += segment_end[2]

                if end == len(midline)-1:
                    avg_end_error += error
                
                if error >= error_threshold:
                    end = mid - 1
                    segment_end[2] = end  # was int(midline_range / 2 ** divisions)
                    segment_end[1] = midline[segment_end[2]][f][1]
                    segment_end[0] = midline[segment_end[2]][f][0]
                    divisions += 1

                elif error < error_threshold:
                    start = mid + 1
                    if start < len(midline):
                        segment_end[2] = start  # was int(midline_range / 2 ** divisions)
                        segment_end[1] = midline[segment_end[2]][f][1]
                        segment_end[0] = midline[segment_end[2]][f][0]
                        divisions += 1

        avg_joint = avg_joint // len(tmp_joints)

        joints.append([midline[avg_joint][0][0], midline[avg_joint][0][1], avg_joint])

        # print("Adding joint:", joints[len(joints)-1])

        if (avg_end_error / len(midline[0])) < error_threshold:
            print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
            completed = True

    joints.pop()

    return joints


def grow_segments(midline, error_threshold):
    print("\n--Generating segments by growing--\n")
    
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

            for i in range(joints[len(joints) - 1][2], increments, 1):
                tmp_error = find_error(segment_end, segment_beginning, midline[i][f])

                if frame_error < tmp_error:
                    frame_error = tmp_error

            total_error += frame_error

        total_error /= len(midline[0])  # total frames

        if total_error < error_threshold:
            # print("ye: ", increments, " f: ", f, " error: ", total_error)
            increments += 1

        elif total_error >= error_threshold:
            increments -= 1

            if increments <= joints[len(joints) - 1][2]:
                print("stuck on increment: ", increments, "error: ", total_error, "Sb: ", segment_beginning,
                      "Se: ", segment_end)
                break
            else:
                joints.append([midline[increments][0][0],
                               midline[increments][0][1], increments])
                # print("Adding joint: ", joints[len(joints) - 1])

        else:
            print("Houston, we have a problem")

    return joints


# turn joint data to actual lengths
def joints_to_length(joints):
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='black')

    for i in range(len(joints)-1):
        # length = √((x2 – x1)² + (y2 – y1)²)
        start = joints[i]
        end = joints[i+1]
        length += math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        length_difference = length - segments[i]
        segments.append(length)
        plt.scatter(length, 0, color='black')
        plt.annotate('(%d)' % joints[i+1][2], (length, 1))
        print("i:", i, " start:", round(start[0], 3), round(start[1], 3), " end:", round(end[0], 3), round(end[1], 3),
              " length:", round(length, 3), " difference:", round(length_difference, 3))
    return segments


def use_all_data(error_threshold):
    file_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    all_filenames = ["Acipenser_brevirostrum.Conte.110cm.1BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.110cm.1BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.104cm.3BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.104cm.3BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.102cm.350BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.98cm.4BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.98cm.4BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.93cm.350BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.93cm.350BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.91cm.150BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.91cm.150BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.88cm.150BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.79cm.450BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.79cm.450BL.s02.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.79cm.350BL.s01.avi_CURVES.xls",
                     "Acipenser_brevirostrum.Conte.79cm.350BL.s02.avi_CURVES.xls"]

    for f in range(len(all_filenames)-1):
        fish_midline = load_midline_data(file_path+all_filenames[f])
        joints = grow_segments(fish_midline, error_threshold)

        for i in range(len(fish_midline[0])):  # all: fish_midline[0])
            for j in range(len(joints)):
                plt.scatter(fish_midline[joints[j][2]][i][0],
                            fish_midline[joints[j][2]][i][1], color='green')

        plot_midline(fish_midline)

        joints_to_length(joints)

        plt.title(all_filenames[f])

        plt.show()



# run code
def main():
    error_threshold = 1

    file_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    # file_path = r"C:\Users\Which\Desktop\uni\Y3\Main_assignment\Data\Sturgeon from Elsa and Ted\midlines/"
    file_name = "Acipenser_brevirostrum.Conte.79cm.350BL.s01.avi_CURVES.xls"

    fish_midline = load_midline_data(file_path + file_name)

    # joints = create_equal_segments(10, fish_midline, 1)

    # joints = create_diminishing_segments(10, fish_midline, 0)

    # plot_joints = [5, 6, 7, 8]  # 6, 10, 12

    joints = grow_segments(fish_midline, error_threshold)

    print("number of joints (green):", len(joints))

    for i in range(len(fish_midline[0])):  # all: fish_midline[0])
        for j in range(len(joints)):
            plt.scatter(fish_midline[joints[j][2]][i][0],
                        fish_midline[joints[j][2]][i][1], color='green')

    """
    joints = grow_segments_divide_and_conquer(fish_midline, error_threshold)
    print("number of joints (red):", len(joints))
    for i in range(len(plot_joints)):  # all: fish_midline[0])
        for j in range(len(joints)):
            plt.scatter(fish_midline[joints[j][2]][plot_joints[i]][0],
                        fish_midline[joints[j][2]][plot_joints[i]][1], color='red
    """

    print("number of joints = ", len(joints), ", Joints: ", joints)

    plot_midline(fish_midline)

    joints_to_length(joints)

    plt.show()


# main()

use_all_data(0.5)
