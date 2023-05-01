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


def compare_method_sinewave_frequency(generation_method, error_threshold, frequency_min, frequency_max,
                                      frequency_interval, save_path):
    csv_file = open(
        save_path + "/" + f"frequency_range({frequency_min}, {frequency_max}, {frequency_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    frequency = frequency_min
    amplitude = 22  # find avg max amplitude of a fish
    length = 110
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames

    iterations = int((frequency_max - frequency_min) / frequency_interval)
    joints = []
    print(f"iterations size:{iterations}, interval:{frequency_interval}")
    # create csv file to write data
    # for a fixed amplitude and length (length = 100cm)
    # compare for each interval
    for f in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, round(phase_difference, 3), frames, error_threshold, len(joints),
             time_taken,
             avg_frame_error[0], avg_frame_error[1]])
        # write time taken, number of joints, error, avg_frame_error to csv file
        # joints_to_length(joints)
        # plt.show()
        frequency += frequency_interval
        frequency = round(frequency, 1)


def compare_method_sinewave_amplitude(generation_method, error_threshold, amplitude_min, amplitude_max,
                                      amplitude_interval, save_path):
    csv_file = open(
        save_path + "/" + f"amplitude_range({amplitude_min}, {amplitude_max}, {amplitude_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    frequency = 1.7  # avg frequency of fish from data
    amplitude = amplitude_min
    length = 110
    resolution = 200  # default resolution for a midline
    frames = 10
    phase_difference = 2 * np.pi / frames
    iterations = int((amplitude_max - amplitude_min) / amplitude_interval)
    # create csv file
    # fixed frequency and number of waves
    for a in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, round(phase_difference, 3), frames, error_threshold, len(joints),
             time_taken,
             avg_frame_error[0], avg_frame_error[1]])
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")

        amplitude += amplitude_interval
        amplitude = round(amplitude, 1)


def compare_method_sinewave_resolution(generation_method, error_threshold, resolution_min, resolution_max,
                                       resolution_interval,
                                       save_path):
    csv_file = open(
        save_path + "/" + f"resolution_range({resolution_min}, {resolution_max}, {resolution_interval}) " + generation_method.__name__ \
        + "_" + "sinewaves" + '.csv', 'w')
    csv_file_writer = csv.writer(csv_file)
    csv_file_writer.writerow(
        ['frequency', 'amplitude', 'resolution', 'phase_difference', 'frames', 'error_threshold', 'no. of joints',
         'time', 'avg_linear_error', 'avg_area_error'])

    frequency = 1.7  # avg frequency of fish from data
    length = 110
    amplitude = 22  # find avg max amplitude of a fish
    resolution = resolution_min
    frames = 10
    phase_difference = 2 * np.pi / frames
    iterations = int((resolution_max - resolution_min) / resolution_interval)
    # create csv file to write data
    # for a fixed amplitude and length (length = 100cm)
    # compare for each interval
    for r in range(iterations):
        sinewave_set = mn.generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency, length_cm=length,
                                                         phase_difference=phase_difference, frames=frames,
                                                         resolution=resolution)
        start_time = time.perf_counter()
        joints = generation_method(error_threshold=error_threshold, midline=sinewave_set)
        time_taken = time.perf_counter() - start_time
        avg_frame_error = ce.find_total_error(joints, sinewave_set)
        csv_file_writer.writerow(
            [frequency, amplitude, resolution, phase_difference, frames, error_threshold, len(joints), time_taken,
             avg_frame_error[0], avg_frame_error[1]])
        print(
            f"λ:{frequency}, A:{amplitude}, R:{resolution}, ϕ:{phase_difference:.3f}, fr:{frames}, er:{error_threshold}, #j:{len(joints)}, t:{time_taken}")

        resolution += resolution_interval


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

        fish_midline = mn.load_midline_data(all_files[f])

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
    length = 110
    phase = 10
    frames = 1
    res = 200
    sin_midline = mn.generate_midline_from_sinewave(amplitude=amplitude, frequency=frequency, length_cm=length,
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


def compare_visual_sinewaves():
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
    print("----- compare frequency -----")
    compare_method_sinewave_frequency(gm_l.grow_segments, 4, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    compare_method_sinewave_amplitude(gm_l.grow_segments, 4, 0.5, 100, 0.5, save_dir)

    print("----- compare resolution -----")
    compare_method_sinewave_resolution(gm_l.grow_segments, 4, 20, 2000, 10, save_dir)  # min - bs: 26, mp: 25

    # -------------------------------

    print("----- compare frequency -----")
    compare_method_sinewave_frequency(gm_l.grow_segments_binary_search, 4, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    compare_method_sinewave_amplitude(gm_l.grow_segments_binary_search, 4, 0.5, 100, 0.5, save_dir)

    print("----- compare resolution -----")
    compare_method_sinewave_resolution(gm_l.grow_segments_binary_search, 4, 26, 2000, 10,
                                       save_dir)  # min - bs: 26, mp: 25

    # -------------------------------

    print("----- compare frequency -----")
    compare_method_sinewave_frequency(gm_l.grow_segments_binary_search_midpoint_only, 4, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    compare_method_sinewave_amplitude(gm_l.grow_segments_binary_search_midpoint_only, 4, 0.5, 100, 0.5, save_dir)

    print("----- compare resolution -----")
    compare_method_sinewave_resolution(gm_l.grow_segments_binary_search_midpoint_only, 4, 25, 2000, 10,
                                       save_dir)  # min - bs: 26, mp: 25

    # -------------------------------

    print("----- compare frequency -----")
    compare_method_sinewave_frequency(gm_l.grow_segments_from_inflection, 4, 0.1, 20, 0.1, save_dir)

    print("----- compare amplitude -----")
    compare_method_sinewave_amplitude(gm_l.grow_segments_from_inflection, 4, 0.5, 100, 0.5, save_dir)

    print("----- compare resolution -----")
    compare_method_sinewave_resolution(gm_l.grow_segments_from_inflection, 4, 20, 2000, 10,
                                       save_dir)  # min - bs: 26, mp: 25


def compare_linear_and_area_error(data_path, save_dir, *generation_methods):
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
    # for a method, say generate_segments, compare variables from sinewaves like amplitude, frequency, resolution.

    eel_joints_area = gm_a.grow_segments_binary_search(eel_midline, 65)
    eel_joints_linear = gm_l.grow_segments_binary_search(eel_midline, 4)  # (a:65 and l:4) ratio is 16.5

    total_error_area = ce.find_total_error(eel_joints_area, eel_midline)
    total_error_linear = ce.find_total_error(eel_joints_linear, eel_midline)

    print(f"eel joints area ({len(eel_joints_area)}):{eel_joints_area}")
    print("eel_joints_area total: ", total_error_area)

    print(f"eel joints linear ({len(eel_joints_linear)}):{eel_joints_linear}")
    print("eel_joints_linear total: ", total_error_linear)
