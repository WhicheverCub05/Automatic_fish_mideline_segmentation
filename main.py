# first: use matplotlib to generate waves of different wavelengths/amplitudes so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
from os import path
import sys
import glob  # used to find


# implementation of growth algorithm


# get data from every other line for now
def load_midline_data(location):
    print("Loading:", location)
    try:
        file_data = pd.read_excel(location)
        dimensions = file_data.shape
        print("shape: ", dimensions)

        midline = [[[0 for _ in range(2)] for _ in range(dimensions[1] // 2)] for _ in range(dimensions[0])]

        for column in range(0, dimensions[1], 2):
            for row in range(dimensions[0]):
                x = file_data.iat[row, column]
                y = file_data.iat[row, column + 1]
                midline[row][column // 2][0] = x
                midline[row][column // 2][1] = y
        return midline

    except FileNotFoundError:
        print("the file is not found")


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
def create_equal_segments(midline, segment_count, *frame):
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
def create_diminishing_segments(midline, segment_count, *frame):
    joints = [[0 for _ in range(3)] for _ in range(0)]
    length = len(midline)
    increment = 0

    column = 0

    if frame:
        column = frame

    for i in range(segment_count):
        x = midline[increment][column][0]
        y = midline[increment][column][1]
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
    if Se[1] - Sb[1] == 0 or Se[0] - Sb[0] == 0:
        # gradient is 0 so perpendicular line is undefined
        return 0

    gradient = (Se[1] - Sb[1]) / (Se[0] - Sb[0])
    c = Se[1] - (gradient * Se[0])

    perpendicular_gradient = -1 / gradient

    perpendicular_c = Mp[1] - (perpendicular_gradient * Mp[0])

    x = abs((c - perpendicular_c) / (gradient - perpendicular_gradient))
    y = (gradient * x) + c

    error = abs(np.sqrt((x - Mp[0]) ** 2 + (y - Mp[1]) ** 2))

    # print("Sb: ", Sb, " Se: ", Se, "Mp: ", Mp, "intersection x:", x, "y:", y, " error: ", error)

    return error


# another option - for each joint, generate segments for 1 frame. try segment on other frames and reduce size as needed

# optimises the generation method by using greedy binary search
def grow_segments_divide_and_conquer(midline, error_threshold):
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
                error = find_error(segment_end, segment_beginning, midline[error_index][f])

                mid = (start + end) // 2

                if end <= start:
                    segment_built = True
                    tmp_joints.append(segment_end)
                    avg_joint += segment_end[2]

                if end == len(midline) - 1:
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

        print("Adding joint:", joints[len(joints) - 1])

        if (avg_end_error / len(midline[0])) < error_threshold:
            print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
            completed = True

    joints.pop()

    return joints


def grow_segments(midline, error_threshold):
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
                print("Adding joint: ", joints[len(joints) - 1])

        else:
            print("Houston, we have a problem")

    return joints


# turn joint data to actual lengths
def joints_to_length(joints):
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='red')

    for i in range(len(joints) - 1):
        # length = √((x2 – x1)² + (y2 – y1)²)
        start = joints[i]
        end = joints[i + 1]
        length += math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        length_difference = length - segments[i]
        segments.append(length)
        plt.scatter(length, 0, color='black', label='(%d)' % joints[i + 1][2])
        # plt.annotate('(%d)' % joints[i + 1][2], (length, i % 2))
        plt.legend(loc="upper right")
        # print("i:", i, " start:", round(start[0], 3), round(start[1], 3), " end:", round(end[0], 3), round(end[1], 3),
        #       " length:", round(length, 3), " difference:", round(length_difference, 3))
    return segments


def use_all_data(generation_method, folder_path, **parameters):
    # folder_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
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

    for f in range(len(all_filenames) - 1):

        fish_midline = load_midline_data(folder_path + "/" + all_filenames[f])

        if 'error_threshold' in parameters:
            joints = generation_method(midline=fish_midline, error_threshold=parameters['error_threshold'])
        elif 'segment_count' in parameters:
            joints = generation_method(midline=fish_midline, segment_count=parameters['segment_count'])
        else:
            joints = generation_method(midline=fish_midline)

        print("--Generation method: ", generation_method.__name__, "--")

        for i in range(len(fish_midline[0])):  # all: fish_midline[0])
            for j in range(len(joints)):
                plt.scatter(fish_midline[joints[j][2]][i][0],
                            fish_midline[joints[j][2]][i][1], color='green')

        plot_midline(fish_midline)

        joints_to_length(joints)

        plt.title(generation_method.__name__ + str(parameters) + all_filenames[f][28:45:1])
        plt.xlabel('x')
        plt.ylabel('y')

        filename = '/mnt/chromeos/MyFiles/Y3_Project/Fish data/Graphs/' + generation_method.__name__ \
                   + str(parameters) + all_filenames[f][28:45:1] + '.svg'
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: "
                  + folder_path + filename)

        plt.cla()


def pick_method_and_save_all(folder_path):
    error_threshold = 0
    segment_count = 0

    # let user pick method and error and save

    generation_dictionary = {
        "sg": "grow_segments",
        'sg_dc': "grow_segments_divide_and_conquer",
        'es': "create_equal_segments",
        'ds': "create_diminishing_segments",
        'mb': "math_test_bench",
        'q': 'exit script'
    }

    user_selection = ""

    while user_selection != 'q':

        for option in generation_dictionary:
            print(option, ": ", generation_dictionary[option])

        user_selection = input("please select the method: ")
        if user_selection not in generation_dictionary:
            print("try again b")

        if user_selection == 'sg' or user_selection == 'sg_dc':
            while 1:
                try:
                    error_threshold = float(input("please input an error threshold value: "))
                except ValueError:
                    print("please input a numerical value")

                if error_threshold <= 0:
                    print("please input a value larger than 0")
                else:
                    break

            if user_selection == 'sg_dc':
                use_all_data(grow_segments_divide_and_conquer, folder_path, error_threshold=error_threshold)
            elif user_selection == 'sg':
                use_all_data(grow_segments, folder_path, error_threshold=error_threshold)

        elif user_selection == 'es' or user_selection == 'ds':
            while 1:
                try:
                    segment_count = int(input("please input the number of segments: "))
                except ValueError:
                    print("please input a numerical value")

                if segment_count <= 0:
                    print("please input a value larger than 0")
                else:
                    break

            if user_selection == 'es':
                use_all_data(create_equal_segments, folder_path, segment_count=segment_count)
            elif user_selection == 'ds':
                use_all_data(create_diminishing_segments, folder_path, segment_count=segment_count)

        elif user_selection == 'mb':
            math_test_bench()
        elif user_selection == 'q':
            exit()
        else:
            print("Invalid selection, please try again")

        print("\n===== Operation complete =====\n")


def set_data_folder():
    # get file path from user, load data
    print("Set the file location of the database:")

    files_found = False
    folder_path = ""

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        print("sys.arg", sys.argv)

    while not files_found:
        if folder_path == "":
            folder_path = input("please input the file location:")

        if path.exists(folder_path):
            files = glob.glob(folder_path + '/*.xls')
            if len(files) > 0:
                print("Excel files found:", files)
                files_found = True
            else:
                print("Cannot find any excel files (.xls)")
                folder_path = ""
        else:
            print("Folder not found or no permissions to access it")
            print("FP:", folder_path)
            folder_path = ""

    print("Using folder path:", folder_path)

    return folder_path


# run code only when called as a script
if __name__ == "__main__":
    """
    j = load_midline_data("/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and "
                               "Ted/midlines/Acipenser_brevirostrum.Conte.98cm.4BL.s01.avi_CURVES.xls")
    m = grow_segments(j, 0.5)

    plot_midline(m)

    joints_to_length(j)

    plt.show()
    """

    directory = set_data_folder()
    pick_method_and_save_all(folder_path=directory)
