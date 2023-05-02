"""
This module has functions to gather data to .csv files. either of these functions can be called by main.py
"""

__author__ = "Alex R.d Silva"
__version__ = '0.1'

# importing local project libraries
import generation_methods_linear_error as gm_l
import generation_methods_area_error as gm_a
import calculate_error as ce
import main as mn

# import other libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import glob
import os
import csv
import inspect


def compare_method_sinewave_frequency(generation_method, error_threshold, cycles_min, cycles_max,
                                      cycles_interval, save_path):
    """
    For a given generation method and threshold,
     create sine waves of varying frequency and save the results onto a csv file
    :param generation_method: the generation method to compare
    :param error_threshold: the fixed error threshold
    :param cycles_min: the minimum number of waves
    :param cycles_max: the maximum number of waves
    :param cycles_interval: the difference between subsequent frequencies
    :param save_path: the file path to save the csv file
    :return: None
    """
    csv_file = open(
        save_path + "/" + f"cycles_range({cycles_min}, {cycles_max}, {cycles_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['cycles', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    cycles = cycles_min
    amplitude = 22  # find avg max amplitude of a fish
    length = 110
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames

    iterations = int((cycles_max - cycles_min) / cycles_interval)

    print(f"iterations size:{iterations}, interval:{cycles_interval}")

    for f in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, cycles=cycles, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        print(
            f"λ:{cycles}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")
        csv_file_writer.writerow(
            [cycles, amplitude, resolution, round(phase_difference, 3), frames, error_threshold, len(joints),
             time_taken,
             avg_frame_error[0], avg_frame_error[1]])

        cycles += cycles_interval
        cycles = round(cycles, 1)


def compare_method_sinewave_amplitude(generation_method, error_threshold, amplitude_min, amplitude_max,
                                      amplitude_interval, save_path):
    """
    For a given generation method and error threshold,
     create sine waves of varying amplitudes and save the results onto a csv file
    :param generation_method: the generation method to create the joints from the midline
    :param error_threshold: the error threshold for the generation method
    :param amplitude_min: the minimum height (y axis) of the waves in cm
    :param amplitude_max: the maximum height of the wave in cm
    :param amplitude_interval: the difference between subsequent amplitudes
    :param save_path: the file path to save the reults to in a csv file
    :return: None
    """
    csv_file = open(
        save_path + "/" + f"amplitude_range({amplitude_min}, {amplitude_max}, {amplitude_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['cycles', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    cycles = 1.7  # avg cycles of fish from data
    amplitude = amplitude_min
    length = 110
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames
    iterations = int((amplitude_max - amplitude_min) / amplitude_interval)

    for a in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, cycles=cycles, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        csv_file_writer.writerow(
            [cycles, amplitude, resolution, round(phase_difference, 3), frames, error_threshold, len(joints),
             time_taken,
             avg_frame_error[0], avg_frame_error[1]])
        print(
            f"λ:{cycles}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")

        amplitude += amplitude_interval
        amplitude = round(amplitude, 1)


def compare_method_sinewave_resolution(generation_method, error_threshold, resolution_min, resolution_max,
                                       resolution_interval,
                                       save_path):
    """
    For a given generation method and error threshold,
    create sine waves of varying frequency and save the results onto a csv file
    :param generation_method: the generation method to create the joints from the midline
    :param error_threshold: the error threshold for the generation method
    :param resolution_min: the minimum number of waves
    :param resolution_max: the maximum number of waves
    :param resolution_interval: the difference between subsequent frequencies
    :param save_path: the file path to save the csv file
    :return: None
    """
    csv_file = open(
        save_path + "/" + f"resolution_range({resolution_min}, {resolution_max}, {resolution_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['cycles', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    cycles = 1.7  # avg cycles of fish from data
    length = 110
    amplitude = 22  # find avg max amplitude of a fish
    resolution = resolution_min
    frames = 10
    phase_difference = 2 * np.pi / frames
    iterations = int((resolution_max - resolution_min) / resolution_interval)

    for r in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, cycles=cycles, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(sinewave_set, error_threshold)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        csv_file_writer.writerow(
            [cycles, amplitude, resolution, phase_difference, frames, error_threshold, len(joints), time_taken,
             avg_frame_error[0], avg_frame_error[1]])
        print(
            f"λ:{cycles}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")

        resolution += resolution_interval


def compare_method_error(generation_method, error_threshold_min, error_threshold_max, error_threshold_interval,
                         data_path, save_path):
    """
    For a generation method, increase the error by an interval and test with fish data from an Excel file
    :param generation_method: the generation method to create the joints from the midline
    :param error_threshold_min: minimum error threshold
    :param error_threshold_max: maximum error threshold
    :param error_threshold_interval: spacing between max and min error threshold
    :param data_path: location of the midlines. Must be Excel files (.xls)
    :param save_path: location to save the results to as a csv file
    :return: None
    """

    all_files = glob.glob(data_path + '/*.xls')

    user_save_path = save_path

    csv_file = open(user_save_path + "/" + generation_method.__name__ \
                    + "_" + "All_Data" + '.csv', 'w')

    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(['error_threshold', 'number of joints', 'time to generate', 'fish_info'])

    plt.cla()
    for f in range(len(all_files) - 1):

        csv_file_writer.writerow([''])

        fish_midline = mn.load_midline_data(all_files[f])

        error_threshold = error_threshold_min
        for i in range(int((error_threshold_max - error_threshold_min) / error_threshold_interval)):
            error_threshold += error_threshold_interval
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
    """
    This function exists for me to test out generation methods on a sinewave and visualise it
    :param save_path: where any outputs (.csv) are saved to
    :param generation_method: the generation method to test out.
    :return: None
    """
    amplitude = 5
    cycles = 2.5
    length = 110
    phase = 10
    frames = 1
    res = 200
    sin_midline = mn.generate_midline_from_sinewave(amplitude=amplitude, cycles=cycles, length_cm=length,
                                                    phase_difference=phase, frames=frames, resolution=res)
    error = 0
    method = gm_l.grow_segments

    if generation_method:
        method = generation_method[0]

    for i in range(2):
        plt.cla()
        mn.plot_midline(sin_midline)
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

        mn.joints_to_length(joint_array)

        plot_title = f"{method.__name__}, e: {error}, wave-> A:{amplitude}, λ:{cycles}, ϕ:{phase}, res:{res}"

        plt.title(plot_title)
        plt.annotate(f"time:{end_time - start_time:.4f}, joints:{len(joint_array)}", (1, 1))

        filename = f"{save_path}/{method.__name__}(e:{error},A:{amplitude},λ:{cycles},ϕ:{phase}" \
                   f",res:{res}).svg"
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: ",
                  filename)


def compare_visual_sinewaves():
    """
    Function is used to create multiple matplotlib graphs with the midline and joints created for it
    :return: None
    """
    cycles = 1
    amplitude = 2
    length = 110
    frames = 10
    resolution = 200
    phase_difference = (np.pi * 2 / frames)
    error_threshold = 0.5
    growth_method = gm_l.grow_segments

    for i in range(1, 7):
        test_midline = mn.generate_midline_from_sinewave(i, amplitude, length, phase_difference, frames, resolution)
        test_joints = gm_l.grow_segments(test_midline, error_threshold)
        print(test_joints)
        total_error = ce.find_total_error(test_joints, test_midline)
        mn.plot_midline(test_midline, 0)
        mn.joints_to_length(test_joints, 1)
        plt.title(f"{growth_method.__name__}(e:{error_threshold}, A:{amplitude}, λ:{i}, ϕ:{phase_difference:.3f}, "
                  f"frames:{frames}, res:{resolution})")
        plt.annotate(f"avg frame error \nlinear:{total_error[0]:.2f} \narea:{total_error[1]:.2f}", (0, amplitude * -1))
        plt.show()
        plt.cla()


def compare_all_methods_linear_error_sinewave(save_dir):
    """
    Function is used to run several functions to gather data about all current linear generation methods
    using sine waves as the midlines.
    :param save_dir: directory to save the results to, as .csv files
    :return: None
    """
    print("----- compare frequency -----")
    compare_method_sinewave_frequency(gm_l.grow_segments, 4, 0.1, 20, 0.1, save_dir)
    compare_method_sinewave_frequency(gm_l.grow_segments_binary_search, 4, 0.1, 20, 0.1, save_dir)
    compare_method_sinewave_frequency(gm_l.grow_segments_binary_search_midpoint_only, 4, 0.1, 20, 0.1, save_dir)
    compare_method_sinewave_frequency(gm_l.grow_segments_from_inflection, 4, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    compare_method_sinewave_amplitude(gm_l.grow_segments, 4, 0.5, 100, 0.5, save_dir)
    compare_method_sinewave_amplitude(gm_l.grow_segments_binary_search, 4, 0.5, 100, 0.5, save_dir)
    compare_method_sinewave_amplitude(gm_l.grow_segments_binary_search_midpoint_only, 4, 0.5, 100, 0.5, save_dir)
    compare_method_sinewave_amplitude(gm_l.grow_segments_from_inflection, 4, 0.5, 100, 0.5, save_dir)

    print("----- compare resolution -----")
    compare_method_sinewave_resolution(gm_l.grow_segments, 4, 30, 2000, 10, save_dir)  # min - bs: 26, mp: 25
    compare_method_sinewave_resolution(gm_l.grow_segments_binary_search, 4, 30, 2000, 10,
                                       save_dir)
    compare_method_sinewave_resolution(gm_l.grow_segments_binary_search_midpoint_only, 4, 30, 2000, 10,
                                       save_dir)
    compare_method_sinewave_resolution(gm_l.grow_segments_from_inflection, 4, 30, 2000, 10,
                                       save_dir)


def compare_linear_and_area_error(data_path, save_dir, *generation_methods):
    """
    Function is used to compare the total error, time and number of joints
    with generation methods that use area and linear error thresholds. We use sinewave data and fish midline data
    :param data_path: directory to the data that the fish midlines are in
    :param save_dir: directory where the results of the data are saved to, as .csv files
    :param generation_methods: an array of the generation methods that are compared to each other
    :return: None
    """
    all_midline_files = glob.glob(data_path + '/*.xls')

    # write data to csv files

    csv_file = open(save_dir + "compare_linear_and_area_error" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    # compare joints and error for eel sine wave
    eel_midline = mn.generate_midline_from_sinewave(1.7, 22, 110, (np.pi * 2 / 10), 10, 200)

    error_threshold_area = 0
    error_threshold_linear = 0
    csv_data = []

    column_labels = ['linear error threshold', 'area error threshold', '']
    for g in range(len(generation_methods)):
        column_labels.append(generation_methods[g].__name__ + " time")
        column_labels.append(generation_methods[g].__name__ + " no. of joints")
        column_labels.append(generation_methods[g].__name__ + " total area error")
        column_labels.append('')

    csv_file_writer.writerow(column_labels)

    for e in range(100):

        csv_data = []

        error_threshold = 0
        error_threshold_linear += 0.5
        error_threshold_area += 10

        csv_data.append(error_threshold_linear)
        csv_data.append(error_threshold_area)

        print(f"linear error threshold: {error_threshold_linear}, area error threshold: {error_threshold_area}")

        for g in range(len(generation_methods)):

            generation_method = generation_methods[g]

            if 'error_threshold_area' in (str(inspect.signature(generation_method))):
                error_threshold = error_threshold_area
            else:
                error_threshold = error_threshold_linear

            time_start = time.perf_counter()
            joints = generation_method(eel_midline, error_threshold)
            total_time = time.perf_counter() - time_start
            total_error = ce.find_total_error(joints, eel_midline)

            csv_data.append('')
            csv_data.append(total_time)
            csv_data.append(len(joints))
            csv_data.append(total_error[1])

        csv_file_writer.writerow(csv_data)

    # compare fish data and see if number of joints is the same. also compare joint positions for a few sample fish.
    # for a method, say generate_segments, compare variables from sinewaves like amplitude, cycles, resolution.

    eel_joints_area = gm_a.grow_segments_binary_search(eel_midline, 65)
    eel_joints_linear = gm_l.grow_segments_binary_search(eel_midline, 4)  # (a:65 and l:4) ratio is 16.5

    total_error_area = ce.find_total_error(eel_joints_area, eel_midline)
    total_error_linear = ce.find_total_error(eel_joints_linear, eel_midline)

    print(f"eel joints area ({len(eel_joints_area)}):{eel_joints_area}")
    print("eel_joints_area total: ", total_error_area)

    print(f"eel joints linear ({len(eel_joints_linear)}):{eel_joints_linear}")
    print("eel_joints_linear total: ", total_error_linear)


def use_all_folder_data(generation_method, data_path, save_path, **parameters):
    """
    This function is used to generate .svg graphs using all the Excel data in a directory
    and then saves them to the user's path
    :param generation_method: the generation method to create the joints
    :param data_path: directory of the fish midline data
    :param save_path: directory to save the graphs to
    :param parameters: arguments that are passed into the generation method function
    :return: None
    """
    all_files = glob.glob(data_path + '/*.xls')
    print("all_files: ", all_files)

    for f in range(len(all_files) - 1):
        fish_midline = mn.load_midline_data(all_files[f])
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

        mn.plot_midline(fish_midline, 0, 7)

        mn.joints_to_length(joints)

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


def compare_number_of_joints_fish_data(data_path, save_dir, *resolution_division):
    """
    Compares method of generating joints by removing data points until targeted number of joints is achieved.
    Saves data like time and total area error to a .csv file
    :param data_path: directory of the fish midline data
    :param save_dir: directory of where to save the results to
    :param resolution_division: option to divide resolution of fish data to increase processing speed
    :return: None
    """
    all_files = glob.glob(data_path + '/*.xls')
    print("all_files: ", all_files)

    csv_file = open(save_dir + "compare_number_of_joints_fish_data" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(["Midline filename", "", "Number of joints", "Total_area_error", "Time taken"])

    for f in range(len(all_files)-1):
        fish_midline = mn.load_midline_data(all_files[f])
        for j in range(4, 7):
            start_time = time.perf_counter()
            joints = gm_a.generate_segments_to_quantity(fish_midline, j, resolution_division[0])
            total_time = time.perf_counter() - start_time

            total_area_error = ce.find_total_area_error(joints, fish_midline)

            csv_file_writer.writerow([all_files[f], "", len(joints), total_area_error, total_time])

            print(f"file:{all_files[f]}, number of joints:{j}, total_area_error:{total_area_error:.2f}, "
                  f"time taken:{total_time:.3f}")

        csv_file_writer.writerow("")
        csv_file_writer.writerow("")



def gather_data():
    """
    This Method is where I arrange which methods to run. This Function is then called in main.py
    :return: None
    """
    eel_midline = mn.generate_midline_from_sinewave(1.7, 22, 110, (np.pi * 2 / 10), 10, 50)

    # save_path = mn.set_data_folder() + "/results/"

    # compare_method_sinewave_resolution(gm_l.grow_segments, 4, 30, 2030, 10, save_path)
    # compare_method_sinewave_resolution(gm_a.grow_segments, 65, 30, 2040, 10, save_path)

    # joints = gm_a.generate_segments_to_max_area_error(eel_midline, 200)
    # joints = gm_a.generate_segments_to_quantity(eel_midline, 3)
    # print("final joints: ", joints)
    # mn.plot_midline(eel_midline, 0)
    # mn.joints_to_length(joints, 1)
    # plt.show()

    data_path = mn.set_data_folder()
    save_dir = mn.set_data_folder() + "/results/"
    compare_number_of_joints_fish_data(data_path, save_dir, 10)
