# first: use matplotlib to generate waves of different wavelengths/amplitudes, so you can visualise segment accuracy
# second: generate fish spine segments that will fit wave
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
from os import path
import sys
import glob  # used to find
import csv


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


# asks the user to set a filepath to save data
def get_user_save_path(data_path, *save_path):
    while 1:
        if save_path:
            folder_path = save_path
        elif len(sys.argv) > 2:
            folder_path = sys.argv[2]
        else:
            folder_path = input("input the location you want to save ('nf' makes a new file called 'results'): ")
            if folder_path == 'nf':
                if os.path.exists(data_path + '/results'):
                    folder_path = data_path + '/results'
                    break
                else:
                    try:
                        os.mkdir(data_path + '/results')
                        folder_path = data_path + '/results'
                        break
                    except FileNotFoundError or FileExistsError:
                        print("results file can't be made, please check your data permissions folder:",
                              data_path)

        if os.path.exists(folder_path):
            break
        else:
            if input("Folder doesn't exist. Create one? (y/n)").capitalize() == "Y":
                print("The folder you tried:", folder_path)
                try:
                    os.mkdir(folder_path)
                    break
                except FileNotFoundError:
                    print("Path does not exists. Please try again")
                    input("Hit enter to continue")
                except PermissionError:
                    print("Permission is denied. Please try another path")
                    input("Hit enter to continue")
            else:
                pass

    return folder_path


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


def math_test_bench():
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)

    data = [[0.7, 1.0], [1.7, 1.5], [2.3, 2.7], [2.8, 3.7], [3.7, 4]]

    for d in data:
        plt.plot(d[0], d[1], 'bo', ls='--')

    start = 0
    end = 4

    segment_beginning = data[start]
    segment_end = data[end]

    y1 = segment_beginning[1]
    y2 = segment_end[1]

    x1 = segment_beginning[0]
    x2 = segment_end[0]

    print("y1:", y1, " y2:", y2, " x1:", x1, " x2:", x2)
    gr = (y2 - y1) / (x2 - x1)

    c = y1 - (gr * x1)

    x = np.linspace(0, 10, 100)
    y = (gr * x) + c
    plt.plot(x, y, '-b')

    # iterate from segment_beginning index to segment_end index, incrementing midline_point index
    for i in range(start + 1, end, 1):
        midline_point = data[i]

        ym = ((-1 / gr) * x) + midline_point[1] - ((-1 / gr) * midline_point[0])

        error = find_error(segment_end, segment_beginning, midline_point)

        print("error:", error)

        c_intersection = midline_point[1] - ((-1 / gr) * midline_point[0])
        print("C_intersection:", c_intersection)

        x_intersection = abs((c - c_intersection) / (gr - (-1 / gr)))
        y_intersection = ((-1 / gr) * x_intersection) + c_intersection

        print("intersection:", x_intersection, ",", y_intersection)
        plt.scatter(x_intersection, y_intersection, color='green', label='(%d,%d)' % (x_intersection, y_intersection))
        plt.annotate(str([x_intersection, y_intersection]), [x_intersection, y_intersection])

        plt.plot(x, ym, '-r', label='(%d)' % error)

    plt.legend(loc="upper right")
    plt.ylim(0, 5)
    plt.xlim(0, 5)
    plt.show()


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


def find_area_error(segment_beginning, segment_end, midline_function):
    midline_point = int((segment_end[2] - segment_beginning[2]) / 2)

    # get function of midline curve


def find_error(segment_end, segment_beginning, midline_point):
    if segment_end[1] - segment_beginning[1] == 0 or segment_end[0] - segment_beginning[0] == 0:
        # gradient is 0 so perpendicular line is undefined
        return 0

    gradient = (segment_end[1] - segment_beginning[1]) / (segment_end[0] - segment_beginning[0])
    c = segment_end[1] - (gradient * segment_end[0])

    perpendicular_gradient = -1 / gradient
    perpendicular_c = midline_point[1] - (perpendicular_gradient * midline_point[0])

    x = abs((c - perpendicular_c) / (gradient - perpendicular_gradient))
    y = (gradient * x) + c

    error = abs(np.sqrt((x - midline_point[0]) ** 2 + (y - midline_point[1]) ** 2))
    return error


def grow_segments_binary_search(midline, error_threshold):
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
                error = 0
                for j in range(start + 1, end - 1, 1):
                    tmp_error = find_error(segment_end, segment_beginning, midline[j][f])
                    if tmp_error > error:
                        error = tmp_error
                    if error >= error_threshold:
                        break

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

        # print("Adding joint:", joints[len(joints) - 1])

        if (avg_end_error / len(midline[0])) < error_threshold:
            # print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
            completed = True

    joints.pop()

    return joints


# another option - for each joint, generate segments for 1 frame. try segment on other frames and reduce size as needed

# optimises the generation method by using greedy binary search
def grow_segments_binary_search_midpoint_only(midline, error_threshold):
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

        # print("Adding joint:", joints[len(joints) - 1])

        if (avg_end_error / len(midline[0])) < error_threshold:
            # print("avg_end_error:", avg_end_error / len(midline[0]), " avg_joint:", avg_joint)
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


# turn joint data to actual lengths
def joints_to_length(joints):
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='red', label="start of head")
    plt.xlim(-10, 150)
    plt.ylim(-15, 15)
    for i in range(len(joints) - 1):
        start = joints[i]
        end = joints[i + 1]
        length += math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)  # length = √((x2 – x1)² + (y2 – y1)²)
        length_difference = length - segments[i]
        segments.append(length)
        plt.scatter(length, 0, color='black', label=f'{joints[i + 1][2]} ({length_difference:.2f}cm)')
        # plt.annotate('(%d)' % joints[i + 1][2], (length, i % 2))
        plt.legend(loc="upper right")
    return segments


def use_all_data(generation_method, data_path, save_path, **parameters):
    all_files = glob.glob(data_path + '/*.xls')
    print("all_files: ", all_files)

    for f in range(len(all_files) - 1):
        fish_midline = load_midline_data(all_files[f])

        if 'error_threshold' in parameters:
            joints = generation_method(midline=fish_midline, error_threshold=parameters['error_threshold'])
        elif 'segment_count' in parameters:
            joints = generation_method(midline=fish_midline, segment_count=parameters['segment_count'])
        else:
            joints = generation_method(midline=fish_midline)

        print("- Generation method: ", generation_method.__name__, " -")

        for i in range(len(fish_midline[0])):
            for j in range(len(joints)):
                plt.scatter(fish_midline[joints[j][2]][i][0],
                            fish_midline[joints[j][2]][i][1], color='green')

        plot_midline(fish_midline)

        joints_to_length(joints)

        plt.title(all_files[f][len(all_files[f]) - 30:])
        plt.xlabel('x')
        plt.ylabel('y')

        filename = save_path + "/" + generation_method.__name__ + str(parameters) + \
                   all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1] + '.svg'
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: ",
                  filename)

        plt.cla()


def compare_error(generation_method, data_path, save_path):
    # folder_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    all_files = glob.glob(data_path + '/*.xls')

    # write data to csv files

    user_save_path = save_path

    csv_file = open(user_save_path + "/" + generation_method.__name__ \
                    + "_" + "All_Data" + '.csv', 'w')

    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(['error_threshold', 'number of joints, fish_info'])

    plt.cla()
    for f in range(len(all_files) - 1):

        csv_file_writer.writerow([''])

        fish_midline = load_midline_data(all_files[f])

        for i in range(40):
            # matplotlib.animation
            error_threshold = (i + 1) * 0.05
            joints = generation_method(midline=fish_midline, error_threshold=error_threshold)
            csv_file_writer.writerow([error_threshold, len(joints),
                                      all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1]])
            plt.scatter(error_threshold, len(joints))

        plt.xlabel("error threshold")
        plt.ylabel("number of joints")
        plt.title(os.path.basename(all_files[f]))
        plt.ylim(0, 25)
        plot_name = save_path + "/" + generation_method.__name__ + "." \
                    + all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1] + '.svg'
        try:
            plt.savefig(plot_name)
            print("saved plot:", plot_name)
        except FileNotFoundError:
            print("\nSomething is up with the filename or directory. Please check that the following file exists: "
                  + plot_name + "\n")
            break

        plt.cla()

        print("--Generation method: ", generation_method.__name__, "--")


def pick_method_and_save_all(data_path, *save_path):
    error_threshold = 0
    segment_count = 0

    # let user pick method and error and save

    ui_dictionary = {
        "sg": "grow_segments",
        "sg_bs": "grow_segments_binary_search",
        'sg_bs_mp': "grow_segments_binary_search_midpoint_only",
        'es': "create_equal_segments",
        'ds': "create_diminishing_segments",
        'mb': "math_test_bench",
        'ce': "compare method with increasing error",
        'q': 'exit script'
    }

    if save_path:
        user_save_path = get_user_save_path(data_path, save_path)
    else:
        user_save_path = get_user_save_path(data_path)

    user_selection = ""

    # menu user menu selection

    while user_selection != 'q':

        for option in ui_dictionary:
            print(option, ": ", ui_dictionary[option])

        user_selection = input("Select the method: ")
        if user_selection not in ui_dictionary:
            print("\noption not available\n")

        if user_selection == 'sg' or user_selection == 'sg_bs' or user_selection == 'sg_bs_mp':
            while 1:
                try:
                    error_threshold = float(input("Input an error threshold value: "))
                except ValueError:
                    print("\nplease input a numerical value\n")

                if error_threshold <= 0:
                    print("\nplease input a value larger than 0\n")
                else:
                    break

            if user_selection == 'sg':
                use_all_data(grow_segments, data_path, user_save_path, error_threshold=error_threshold)
            elif user_selection == 'sg_bs':
                use_all_data(grow_segments_binary_search, data_path, user_save_path,
                             error_threshold=error_threshold)
            elif user_selection == 'sg_bs_mp':
                use_all_data(grow_segments_binary_search_midpoint_only, data_path, user_save_path,
                             error_threshold=error_threshold)

        elif user_selection == 'es' or user_selection == 'ds':
            while 1:
                try:
                    segment_count = int(input("input the number of segments: "))
                except ValueError:
                    print("\nplease input a numerical value\n")
                if segment_count <= 0:
                    print("\nplease input a value larger than 0\n")
                else:
                    break

            if user_selection == 'es':
                use_all_data(create_equal_segments, data_path, user_save_path, segment_count=segment_count)
            elif user_selection == 'ds':
                use_all_data(create_diminishing_segments, data_path, user_save_path,
                             segment_count=segment_count)

        elif user_selection == 'mb':
            math_test_bench()
        elif user_selection == 'ce':
            while 1:
                user_method = input("generation method (q:quit): ")
                if user_method == "sg":
                    compare_error(grow_segments, data_path, user_save_path)
                elif user_method == "sg_bs":
                    compare_error(grow_segments_binary_search, data_path, user_save_path)
                elif user_method == "sg_bs_mp":
                    compare_error(grow_segments_binary_search_midpoint_only, data_path, user_save_path)
                elif user_method == "es":
                    compare_error(create_equal_segments, data_path, user_save_path)
                elif user_method == "ds":
                    compare_error(create_diminishing_segments, data_path, user_save_path)
                elif user_method == 'q':
                    break
                else:
                    print("\nplease input another generation method\n")

        elif user_selection == 'q':
            exit()
        else:
            print("\nInvalid selection, please try again\n")


def set_data_folder():
    # get file path from user, load data
    print("-Set the file location of the database-")

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
    directory = set_data_folder()
    pick_method_and_save_all(data_path=directory)
