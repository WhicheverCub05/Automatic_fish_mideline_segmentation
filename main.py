# importing local project libraries
import numpy as np

import generation_methods as gm
import demonstration as de

# import other libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path
import sys
import glob
import csv
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

    plot_midline(midline)
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


# turn joint data to actual lengths
def joints_to_length(joints):
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='red', label="start of head")
    # plt.xlim(-10, 150)
    # plt.ylim(-15, 15)
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


# function uses a generation method and compares the number of joints and time to generate
# for error between 0.05 and 2, in increments of 0.05. This is all saved to a .csv file
def compare_method_error(generation_method, data_path, save_path):
    # folder_path = "/mnt/chromeos/MyFiles/Y3_Project/Fish data/Data/Sturgeon from Elsa and Ted/midlines/"
    all_files = glob.glob(data_path + '/*.xls')

    # write data to csv files

    user_save_path = save_path

    csv_file = open(user_save_path + "/" + generation_method.__name__ \
                    + "_" + "All_Data" + '.csv', 'w')

    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(['error_threshold', 'number of joints', 'time to generate', 'fish_info'])

    plt.cla()
    for f in range(len(all_files) - 1):

        csv_file_writer.writerow([''])

        fish_midline = load_midline_data(all_files[f])

        for i in range(40):
            # matplotlib.animation
            error_threshold = (i + 1) * 0.2
            start_time = time.perf_counter()
            joints = generation_method(midline=fish_midline, error_threshold=error_threshold)
            generation_time = time.perf_counter() - start_time
            csv_file_writer.writerow([error_threshold, len(joints), generation_time,
                                      all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1]])
            plt.scatter(error_threshold, len(joints))

        plt.xlabel("error threshold")
        plt.ylabel("number of joints")
        plt.title(os.path.basename(all_files[f]))
        plt.ylim(0, 25)
        plot_name = save_path + "/" + generation_method.__name__ + "." \
                    + all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1] + '1.svg'
        try:
            plt.savefig(plot_name)
            print("saved plot:", plot_name)
        except FileNotFoundError:
            print("\nSomething is up with the filename or directory. Please check that the following file exists: "
                  + plot_name + "\n")
            break

        plt.cla()

        print("--Generation method: ", generation_method.__name__, "--")


def sinewave_sandbox(save_path, *generation_method):
    amplitude = 5
    frequency = 2.5
    phase = 10
    frames = 1
    res = 200
    sin_midline = generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency,
                                                 phase_difference=phase, frames=frames, resolution=res)
    error = 0
    method = gm.grow_segments

    if generation_method:
        method = generation_method[0]

    for i in range(2):
        plt.cla()
        plot_midline(sin_midline)
        start_time = time.perf_counter()
        error += 0.05
        error = round(error, 2)
        joint_array = method(midline=sin_midline, error_threshold=error)
        end_time = time.perf_counter()
        print("method:", method.__name__, " joints:", len(joint_array), " error:", error,
              " time:", end_time - start_time)

        for f in range(len(sin_midline[0])):
            for j in range(len(joint_array)):
                plt.scatter(sin_midline[joint_array[j][2]][f][0],
                            sin_midline[joint_array[j][2]][f][1], color='green')

        joints_to_length(joint_array)

        plot_title = f"{method.__name__}, e: {error}, wave-> A:{amplitude}, λ:{frequency}, ϕ:{phase}, res:{res}"

        plt.title(plot_title)
        plt.annotate(f"time:{end_time - start_time:.4f}, joints:{len(joint_array)}", (1, 1))

        filename = f"{save_path}/{method.__name__}(e:{error},A:{amplitude},λ:{frequency},ϕ:{phase}" \
                   f",res:{res}).svg"
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: ",
                  filename)


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
                    compare_method_error(gm.grow_segments, data_path, user_save_path)
                elif user_method == "sg_bs":
                    compare_method_error(gm.grow_segments_binary_search, data_path, user_save_path)
                elif user_method == "sg_bs_mp":
                    compare_method_error(gm.grow_segments_binary_search_midpoint_only, data_path, user_save_path)
                elif user_method == "es":
                    compare_method_error(gm.create_equal_segments, data_path, user_save_path)
                elif user_method == "ds":
                    compare_method_error(gm.create_diminishing_segments, data_path, user_save_path)
                elif user_method == 'q':
                    break
                else:
                    print("\nplease input another generation method\n")

        elif user_selection == 'q':
            exit()
        else:
            print("\nInvalid selection, please try again\n")


def compare_method_sinewave_frequency(generation_method, frequency_min, frequency_max, frequency_interval, save_path):
    csv_file = open(
        save_path + "/" + f"frequency_range({frequency_min}, {frequency_max}, {frequency_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time'])

    frequency = frequency_min
    amplitude = 2  # find avg max amplitude of a fish
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames
    error_threshold = 1
    iterations = int((frequency_max - frequency_min) / frequency_interval)
    joints = []
    print(f"iterations size:{iterations}, interval:{frequency_interval}")
    # create csv file to write data
    # for a fixed amplitude and length (length = 100cm)
    # compare for each interval
    for f in range(iterations):
        sinewave_set = generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency,
                                                      phase_difference=phase_difference, frames=frames,
                                                      resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, phase_difference, frames, error_threshold, len(joints), time_taken])
        # write time taken, number of joints, error, total_error to csv file
        # joints_to_length(joints)
        # plt.show()
        frequency += frequency_interval


def compare_method_sinewave_amplitude(generation_method, amplitude_min, amplitude_max, amplitude_interval, save_path):
    csv_file = open(
        save_path + "/" + f"amplitude_range({amplitude_min}, {amplitude_max}, {amplitude_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time'])

    frequency = 5  # avg frequency of fish from data
    amplitude = amplitude_min
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames
    error_threshold = 1
    iterations = int((amplitude_max - amplitude_min) / amplitude_interval)
    # create csv file
    # fixed frequency and number of waves
    for a in range(iterations):
        sinewave_set = generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency,
                                                      phase_difference=phase_difference, frames=frames,
                                                      resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, phase_difference, frames, error_threshold, len(joints), time_taken])
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")
        amplitude += amplitude_interval


def compare_method_sinewave_resolution(generation_method, resolution_min, resolution_max, resolution_interval,
                                       save_path):
    csv_file = open(
        save_path + "/" + f"resolution_range({resolution_min}, {resolution_max}, {resolution_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time'])

    frequency = 5  # avg frequency of fish from data
    amplitude = 2  # find avg max amplitude of a fish
    resolution = resolution_min
    frames = 10
    phase_difference = 2 * np.pi / frames
    error_threshold = 1
    iterations = int((resolution_max - resolution_min) / resolution_interval)
    # create csv file to write data
    # for a fixed amplitude and length (length = 100cm)
    # compare for each interval
    for r in range(iterations):
        sinewave_set = generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency,
                                                      phase_difference=phase_difference, frames=frames,
                                                      resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, phase_difference, frames, error_threshold, len(joints), time_taken])
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")

        resolution += resolution_interval


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
    # save_dir = set_data_folder() + "/results"
    # sinewave_sandbox(save_dir, gm.grow_segments)
    directory = set_data_folder()
    pick_method_and_save_all(data_path=directory)

    print("----- compare frequency -----")
    # compare_method_sinewave_frequency(gm.grow_segments_from_inflection, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    # compare_method_sinewave_amplitude(gm.grow_segments_from_inflection, 0.1, 20, 0.1, save_dir)

    print("----- compare resolution -----")
    # compare_method_sinewave_resolution(gm.grow_segments_from_inflection, 10, 2000, 10, save_dir) # min - bs: 26, mp: 25
