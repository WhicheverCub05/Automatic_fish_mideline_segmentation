"""
This module has functions to gather data to .csv files. either of these functions can be called by main.py

Only the two functions used by the CLI are kept (compare_method_error, use_all_folder_data);
the sine-wave sandbox/visual comparison helpers were dead code and were removed 2026-08-11.
"""

__author__ = "Alex R.d Silva"
__version__ = '1.0'

# importing local project libraries
from core import (joints_to_length, load_3d_csv_data, load_midline_data, plot_midline)

# import other libraries
import matplotlib.pyplot as plt
import time
import glob
import os
import csv


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

    with open(save_path + "/" + generation_method.__name__
              + "_" + "All_Data" + '.csv', 'w', newline='') as csv_file:
        csv_file_writer = csv.writer(csv_file)

        csv_file_writer.writerow(['error_threshold', 'number of joints', 'time to generate', 'fish_info'])

        plt.cla()
        for f in range(len(all_files)):

            csv_file_writer.writerow([''])

            fish_midline = load_midline_data(all_files[f])

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
                        + os.path.splitext(os.path.basename(all_files[f]))[0] + '1.svg'
            try:
                plt.savefig(plot_name)
                print("saved plot:", plot_name)
            except FileNotFoundError:
                print("\nSomething is up with the filename or directory. Please check that the following file exists: "
                      + plot_name + "\n")
                break

            plt.cla()

            print("--Generation method: ", generation_method.__name__, "--")


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
    # Look for both .xls and .csv files
    xls_files = glob.glob(data_path + '/*.xls')
    csv_files = glob.glob(data_path + '/*.csv')
    all_files = xls_files + csv_files
    print("all_files: ", all_files)

    for f in all_files:
        # Check if file is CSV or Excel
        if f.endswith('.csv'):
            # For CSV files, load data for generation methods (transposed format)
            fish_midline = load_3d_csv_data(f, midline_type='midline', for_generation=True)
            if fish_midline is None:
                print(f"Failed to load CSV file: {f}")
                continue
        else:
            fish_midline = load_midline_data(f)
        start_time = time.perf_counter()

        if 'error_threshold' in parameters:
            joints = generation_method(midline=fish_midline, error_threshold=parameters['error_threshold'])
        elif 'segment_count' in parameters:
            joints = generation_method(midline=fish_midline, segment_count=parameters['segment_count'])
        else:
            joints = generation_method(midline=fish_midline)

        generation_time = time.perf_counter() - start_time

        print("- Generation method: ", generation_method.__name__, f" time: {generation_time:.4f}s", " -")

        # plot a couple of representative frames (clamped to the available range)
        plot_frames = [i for i in [0, 7] if i < len(fish_midline[0])]
        for i in plot_frames:
            for j in range(len(joints)):
                plt.scatter(fish_midline[joints[j][2]][i][0],
                            fish_midline[joints[j][2]][i][1], color='green')

        plot_midline(fish_midline, *plot_frames)

        joints_to_length(joints)

        plt.title(os.path.basename(f))
        plt.xlabel('x')
        plt.ylabel('y')

        filename = save_path + "/" + generation_method.__name__ + str(parameters).replace(':', '_') + \
                   os.path.splitext(os.path.basename(f))[0] + '.svg'
        try:
            plt.savefig(filename)
            print("saved file:", filename)
        except FileNotFoundError:
            print("Something is up with the filename or directory. Please check that the following file exists: ",
                  filename)

        plt.cla()
