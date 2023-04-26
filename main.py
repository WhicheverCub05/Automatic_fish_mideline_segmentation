# importing local project libraries
import generation_methods as gm
import gather_data as gd

# import other libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path
import sys
import glob
import time


# creates a 2D array of x,y values for each frame in the Excel file
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


def generate_midline_from_sinewave(frequency, amplitude, phase_difference, frames, resolution):
    midline = [[[0 for _ in range(2)] for _ in range(frames)] for _ in range(resolution)]
    x_values = np.linspace(0, 100, num=resolution)  # describes a 100cm sinewave

    phase = 0

    for f in range(frames):
        for r in range(resolution):
            midline[r][f][0] = x_values[r]
            midline[r][f][1] = np.sin((((x_values * frequency) / 100) * 2 * np.pi) + phase)[r] * amplitude
        phase += phase_difference

    # plot_midline(midline)
    return midline


# asks the user to set a filepath to save their data to
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


# each frame of the midline
def plot_midline(midline, *frames):
    if frames:
        for f in frames:
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][f][0])
                y.append(midline[s][f][1])
            plt.plot(x, y)

    else:
        for f in range(len(midline[0])):
            x = []
            y = []
            for s in range(len(midline)):
                x.append(midline[s][f][0])
                y.append(midline[s][f][1])
            plt.plot(x, y)

    plt.xlabel("x / cm")
    plt.ylabel("y / cm")


# turn joint data to actual lengths
def joints_to_length(joints, *plot_on_first_frame):
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='red', label="start of head")

    tmp_joints_x = []
    tmp_joints_y = []
    for j in range(len(joints)):
        tmp_joints_x.append(joints[j][0])
        tmp_joints_y.append(joints[j][1])

    tmp_joints_x.append(100)
    tmp_joints_y.append(0)

    plt.plot(tmp_joints_x, tmp_joints_y)

    for i in range(len(joints) - 1):
        start = joints[i]
        end = joints[i + 1]
        length += math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)  # length = √((x2 – x1)² + (y2 – y1)²)
        length_difference = length - segments[i]
        segments.append(length)
        plt.scatter(joints[i + 1][0], 0, color='gray')
        if plot_on_first_frame:
            if plot_on_first_frame[0]:
                plt.scatter(joints[i + 1][0], joints[i + 1][1], color='black',
                            label=f'{joints[i + 1][2]} ({length_difference:.2f}cm)')
        else:
            plt.scatter(joints[i + 1][0], 0, color='black', label=f'{joints[i + 1][2]} ({length_difference:.2f}cm)')
        # plt.annotate('(%d)' % joints[i + 1][2], (length, i % 2))
        plt.annotate('%d' % joints[i + 1][2], (joints[i + 1][0] + 0.1, joints[i + 1][1] + 0.05))
        # plt.legend(loc="best")
    return segments


# This function is used to generate .svg graphs using all the Excel data in a directory
# and then saves them to the user's path
def use_all_folder_data(generation_method, data_path, save_path, **parameters):
    all_files = glob.glob(data_path + '/*.xls')
    print("all_files: ", all_files)

    for f in range(len(all_files) - 1):
        fish_midline = load_midline_data(all_files[f])
        start_time = time.perf_counter()

        if 'error_threshold' in parameters:
            joints = generation_method(midline=fish_midline, error_threshold=parameters['error_threshold'])
        elif 'segment_count' in parameters:
            joints = generation_method(midline=fish_midline, segment_count=parameters['segment_count'])
        else:
            joints = generation_method(midline=fish_midline)

        generation_time = time.perf_counter() - start_time

        print("- Generation method: ", generation_method.__name__, f" time: {generation_time:.4f}s", " -")

        for i in [0, 7]:
            for j in range(len(joints)):
                plt.scatter(fish_midline[joints[j][2]][i][0],
                            fish_midline[joints[j][2]][i][1], color='green')

        plot_midline(fish_midline, 0, 7)

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


def pick_method_and_save_all(data_path, *save_path):
    error_threshold = 0
    segment_count = 0

    # let user pick method and error and save

    ui_dictionary = {
        'sg': "grow_segments",
        'sg_i': "grow_segments_from_inflection",
        'sg_bs': "grow_segments_binary_search",
        'sg_bs_mp': "grow_segments_binary_search_midpoint_only",
        'es': "create_equal_segments",
        'ds': "create_diminishing_segments",
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

        if user_selection == 'sg' or user_selection == 'sg_bs' \
                or user_selection == 'sg_bs_mp' or user_selection == 'sg_i':
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
                use_all_folder_data(gm.grow_segments, data_path, user_save_path, error_threshold=error_threshold)
            elif user_selection == 'sg_i':
                use_all_folder_data(gm.grow_segments_from_inflection, data_path, user_save_path,
                                    error_threshold=error_threshold)
            elif user_selection == 'sg_bs':
                use_all_folder_data(gm.grow_segments_binary_search, data_path, user_save_path,
                                    error_threshold=error_threshold)
            elif user_selection == 'sg_bs_mp':
                use_all_folder_data(gm.grow_segments_binary_search_midpoint_only, data_path, user_save_path,
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
                use_all_folder_data(gm.create_equal_segments, data_path, user_save_path, segment_count=segment_count)
            elif user_selection == 'ds':
                use_all_folder_data(gm.create_diminishing_segments, data_path, user_save_path,
                                    segment_count=segment_count)

        elif user_selection == 'ce':
            while 1:
                user_method = input("generation method (q:quit): ")
                if user_method == "sg":
                    gd.compare_method_error(gm.grow_segments, data_path, user_save_path)
                elif user_method == "sg_bs":
                    gd.compare_method_error(gm.grow_segments_binary_search, data_path, user_save_path)
                elif user_method == "sg_bs_mp":
                    gd.compare_method_error(gm.grow_segments_binary_search_midpoint_only, data_path, user_save_path)
                elif user_method == "es":
                    gd.compare_method_error(gm.create_equal_segments, data_path, user_save_path)
                elif user_method == "ds":
                    gd.compare_method_error(gm.create_diminishing_segments, data_path, user_save_path)
                elif user_method == 'q':
                    break
                else:
                    print("\nplease input another generation method\n")

        elif user_selection == 'q':
            exit()
        else:
            print("\nInvalid selection, please try again\n")


# sets the folder location of the Excel data that we use
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
