"""
This module has functions to gather data to .csv files. either of these functions can be called by main.py
"""

__author__ = "Alex R.d Silva"
__version__ = '1.0'

# importing local project libraries
import generation_methods_linear_error as gm_l
import generation_methods_area_error as gm_a
import calculate_error as ce
import main as mn

# import other libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os
import csv


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
    cycles = 1.7
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


def compare_linear_and_area_error(data_path, save_dir, generation_method_area, generation_method_linear):
    """
    Function is used to compare the total error, time and number of joints
    with generation methods that use area and linear error thresholds. We use sinewave data and fish midline data
    :param generation_method_area: the generation method that uses area in cm^2 as its threshold
    :param generation_method_linear: the generation method that uses linear distance in cm as its error threshold
    :param data_path: directory to the data that the fish midlines are in
    :param save_dir: directory where the results of the data are saved to, as .csv files
    :return: None
    """
    all_midline_files = glob.glob(data_path + '/*.xls')

    # write data to csv files

    csv_file = open(save_dir + "compare_linear_and_area_error" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    # compare joints and error for eel sine wave
    # eel_midline = mn.generate_midline_from_sinewave(1.7, 22, 110, (np.pi * 2 / 10), 10, 200)

    midline = mn.load_midline_data(all_midline_files[0])

    print(f"using all_midline_files[0]: {os.path.basename(all_midline_files[0])}")

    error_threshold_area = 0
    error_threshold_linear = 0

    column_labels = ['linear error threshold', 'area error threshold', '']
    for g in range(2):
        column_labels.append("time")
        column_labels.append("no. of joints")
        column_labels.append("total area error")
        column_labels.append('')

    csv_file_writer.writerow(column_labels)

    for e in range(100):

        csv_data = []

        error_threshold_linear += 0.05
        error_threshold_area += 0.2

        error_threshold_linear = round(error_threshold_linear, 3)
        error_threshold_area = round(error_threshold_area, 3)

        csv_data.append(error_threshold_linear)
        csv_data.append(error_threshold_area)

        time_start_area = time.perf_counter()
        joints_area = generation_method_area(midline, error_threshold_area)
        total_time_area = time.perf_counter() - time_start_area
        total_error_area = ce.find_total_area_error(joints_area, midline)

        time_start_linear = time.perf_counter()
        joints_linear = generation_method_linear(midline, error_threshold_linear)
        total_time_linear = time.perf_counter() - time_start_linear
        total_error_linear = ce.find_total_area_error(joints_linear, midline)

        print(f"linear threshold: {error_threshold_linear}, area threshold: {error_threshold_area}. "
              f"joints = L:{len(joints_linear)}, A:{len(joints_area)}")

        csv_data.append('')
        csv_data.append(total_time_linear)
        csv_data.append(len(joints_linear))
        csv_data.append(total_error_linear)

        csv_data.append('')
        csv_data.append(total_time_area)
        csv_data.append(len(joints_area))
        csv_data.append(total_error_area)

        csv_file_writer.writerow(csv_data)


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

        filename = save_path + "/" + generation_method.__name__ + str(parameters).replace(':', '_') + \
                   all_files[f][len(all_files[f]) - 30:len(all_files[f]) - 15:1] + '.svg'
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: ",
                  filename)

        plt.cla()


def compare_number_of_joints_brute_force_fish_data(data_path, save_dir, *resolution_division):
    """
    Compares method of generating joints by removing data points until targeted number of joints is achieved.
    Saves data like time and total area error to a .csv file
    :param data_path: directory of the fish midline data
    :param save_dir: directory of where to save the results to
    :param resolution_division: option to divide resolution of fish data to increase processing speed
    :return: None
    """

    division = 1

    if resolution_division:
        division = resolution_division[0]

    all_files = glob.glob(data_path + '/*.xls')

    csv_file = open(save_dir + f"compare_number_of_joints_fish_data(division:{division})" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(["Midline filename", "", "Number of joints", "Total_area_error", "Time taken",
                              "resolution division"])

    for f in range(all(all_files)):
        fish_midline = mn.load_midline_data(all_files[f])
        for j in range(3, 8):
            start_time = time.perf_counter()
            joints = gm_a.generate_segments_to_quantity(fish_midline, j, division)
            total_time = time.perf_counter() - start_time

            total_area_error = ce.find_total_area_error(joints, fish_midline)

            csv_file_writer.writerow([os.path.basename(all_files[f]), "", len(joints), total_area_error, total_time, division])

            print(f"file:{os.path.basename(all_files[f])}, number of joints:{j}, total_area_error:{total_area_error:.2f}, "
                  f"time taken:{total_time:.3f}, resolution division:{division}")

        csv_file_writer.writerow("")
        csv_file_writer.writerow("")


def compare_starting_point_grow_segments_area_fish_data(data_path, save_dir):
    """
    Visual tests to compare joint configuration when starting from hear or tail. Results are saved
    :param save_dir: directory of where to save the results to
    :param data_path: directory of the fish midline data
    :return: None
    """
    csv_file = open(save_dir + f"compare_starting_point_grow_segments_area_fish_data()" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(["Midline filename", "area error threshold", "forward number of joints",
                              "forward total area error", "reversed number of joints", "reversed total area error"])

    all_files = glob.glob(data_path + '/*.xls')

    midline = mn.load_midline_data(all_files[0])

    midline_reversed = [[[0 for _ in range(2)] for _ in range(len(midline[0]))] for _ in range(len(midline))]

    for f in range(len(midline[0])):
        for r in reversed(range(len(midline))):
            midline_reversed[abs(r - 199)][f][0] = midline[r][f][0]
            midline_reversed[abs(r - 199)][f][1] = midline[r][f][1]

    for error_threshold in range(2, 200, 2):
        joints = gm_a.grow_segments(midline, error_threshold)
        joints_reversed = gm_a.grow_segments(midline_reversed, error_threshold)

        joints_total_error = ce.find_total_area_error(joints, midline)
        joints_reversed_total_error = ce.find_total_area_error(joints_reversed, midline)
        print(f"total area error:{joints_total_error:.2f}, "
              f"reversed:{joints_reversed_total_error:.2f}")
        csv_file_writer.writerow([os.path.basename(all_files[0]), error_threshold, len(joints), joints_total_error, len(joints_reversed),
                                  joints_reversed_total_error])


def compare_area_method_with_brute_force_joint_count(generation_method, error_threshold, data_path, save_dir,
                                                     *resolution_division):
    """
    Compares the total area error for the same number of joints made by the generation method
    to the method generate_segments_to_quantity()
    :param generation_method: The generation method to be compared
    :param error_threshold: the error threshold of the generation method
    :param data_path: directory of the fish midline data
    :param save_dir: directory of where to save the results to
    :param resolution_division: optional variable to reduce the resolution of the midline data for the brute force method
    :return:
    """

    csv_file = open(save_dir + f"compare_starting_point_grow_segments_area_fish_data()" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)

    csv_file_writer.writerow(["Midline filename", "area error threshold", "brute force resolution division",
                              f"{generation_method.__name__} number of joints",
                              f"{generation_method.__name__} total area error", "brute force number of joints",
                              "brute force total area error"])

    all_files = glob.glob(data_path + '/*.xls')

    error_threshold = error_threshold

    if resolution_division:
        resolution_division = resolution_division[0]
    else:
        resolution_division = 1

    for midline_file in all_files:
        midline = mn.load_midline_data(midline_file)

        print(f"using file: {os.path.basename(midline_file)}")

        generation_method_joints = generation_method(error_threshold=error_threshold, midline=midline)

        print(f"File:{os.path.basename(midline_file)} ,number of joints:{len(generation_method_joints)}")

        brute_force_joints = gm_a.generate_segments_to_quantity(midline, len(generation_method_joints),
                                                                resolution_division)

        total_area_generation_method = ce.find_total_area_error(generation_method_joints, midline)
        total_area_brute_force = ce.find_total_area_error(brute_force_joints, midline)

        csv_file_writer.writerow([os.path.basename(midline_file), error_threshold, resolution_division,
                                  len(generation_method_joints), total_area_generation_method, len(brute_force_joints),
                                  total_area_brute_force])


def gather_data():
    """
    This Method is where I arrange which methods to run. This Function is then called in main.py
    :return: None
    """
    eel_midline = mn.generate_midline_from_sinewave(1.7, 22, 110, (np.pi * 2 / 10), 10, 200)

    data_path = mn.set_data_folder()
    save_path = data_path + "/results/"

    compare_area_method_with_brute_force_joint_count(gm_a.grow_segments, 2, data_path, save_path, 2)
