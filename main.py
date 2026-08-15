"""
This file contains functions for a CLI interface which lets the user try different functions and save the results.
Data loading, sine-wave synthesis and basic plotting live in core.py.
"""

__author__ = "Alex R.d Silva"
__version__ = '1.0'

# importing local project libraries
import calculate_error as ce
import gather_data as gd
import method_registry as reg
from core import (is_interactive_backend, is_padding_point, joints_to_length,
                  load_3d_render_data, load_data_file, midline_segments, plot_midline,
                  trace_all_frames)

# import other libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
from os import path
import sys
import glob
import time

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.font_manager import FontProperties

# registry ids used by the small CLI menus (compare / pick-and-save)
CLI_COMPARE_IDS = ["grow_area", "grow_bs_area", "grow_bs_mp_area", "grow_inflect_area",
                   "equal_segments", "diminishing_segments"]

# half the default title font size, used for every plot title
TITLE_HALF_SIZE = FontProperties(size=matplotlib.rcParams['axes.titlesize']).get_size() / 2


def print_usage():
    """
    Prints command line usage information.
    :return: None
    """
    print("""
Automatic Fish Midline Segmentation

Usage:
  python main.py [data_folder] [save_folder]

Arguments:
  data_folder   folder containing the fish midline data (.xls or .csv files).
                If omitted, you will be prompted for it.
  save_folder   folder where plots and results are saved. If omitted, you will
                be prompted ('nf' creates a 'results' folder in the data folder).

Examples:
  python main.py "Sturgeon from Elsa and Ted\\midlines"
  python main.py data results

If you see 'non-interactive backend detected', install a GUI toolkit for
matplotlib (e.g. tkinter) or run with:  set MPLBACKEND=TkAgg
""")


def set_data_folder():
    """
    Resolves the data folder from the command line or the user, retrying until
    a folder containing .xls/.csv files is found (also handles -h/--help).
    :return: the data folder path
    """
    # get file path from user, load data
    if '-h' in sys.argv or '--help' in sys.argv:
        print_usage()
        exit()

    print("-Set the file location of the database-")

    files_found = False
    folder_path = ""

    if len(sys.argv) > 1:
        folder_path = sys.argv[1]

    while not files_found:
        if folder_path == "":
            folder_path = input("please input the file location:")

        if path.exists(folder_path):
            xls_files = glob.glob(folder_path + '/*.xls')
            csv_files = glob.glob(folder_path + '/*.csv')
            total_files = len(xls_files) + len(csv_files)

            if total_files > 0:
                if len(xls_files) > 0:
                    print("Excel files found:", xls_files)
                if len(csv_files) > 0:
                    print("CSV files found:", csv_files)
                files_found = True
            else:
                print("Cannot find any excel files (.xls) or CSV files (.csv)")
                folder_path = ""
        else:
            print("Folder not found or no permissions to access it")
            print("FP:", folder_path)
            folder_path = ""

    print("Using folder path:", folder_path)

    return folder_path


def get_user_save_path(data_path, *save_path):
    """
    Gets save path from user that results of data like graphs or .csv files can be saved to.
    Also saves the results in a folder called 'results' which it can create if
    :param data_path: directory of the fish midline data
    :param save_path: directory of a folder that can be saved to
    :return: None
    """
    while 1:
        if save_path:
            folder_path = save_path[0]
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
                    except (FileNotFoundError, FileExistsError):
                        print("results file can't be made, please check your data permissions folder:",
                              data_path)

        if os.path.exists(folder_path):
            break
        else:
            if input("Folder doesn't exist. Create one? (y/n)").strip().lower() == "y":
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


def run_generation(midline, method, params, save_path, filename):
    """
    Runs a generation method on a midline, prints a summary and saves a plot of the result.
    :param midline: the fish midline data
    :param method: the generation method function
    :param params: keyword arguments passed to the generation method
    :param save_path: directory to save the plot to
    :param filename: base name used for the saved svg file
    :return: the joints generated
    """
    start_time = time.perf_counter()
    joints = method(midline=midline, **params)
    generation_time = time.perf_counter() - start_time

    total_error = ce.find_total_error(joints, midline)

    print(f"\n  method: {method.__name__} {params}")
    print(f"  time: {generation_time:.4f}s")
    print(f"  joints: {len(joints)}")
    print(f"  joint indices: {[j[2] for j in joints]}")
    print(f"  avg linear error: {total_error[0]:.4f}")
    print(f"  avg area error: {total_error[1]:.4f}")

    plt.cla()
    if len(midline[0]) > 1:
        plot_midline(midline, 0, len(midline[0]) // 2)
    else:
        plot_midline(midline, 0)
    for j in joints:
        plt.scatter(midline[j[2]][0][0], midline[j[2]][0][1], color='green')
    joints_to_length(joints, True)
    plt.title(f"{method.__name__} {params} - {filename}")
    plt.xlabel('x')
    plt.ylabel('y')

    try:
        out_path = f"{save_path}/{method.__name__}_{filename}.svg"
        plt.savefig(out_path)
        print(f"  saved: {out_path}")
    except FileNotFoundError:
        print(f"  could not save plot to: {save_path}")
    finally:
        plt.close()

    return joints


def pick_data_file(files, prompt="Select a file"):
    """
    Lets the user pick one of the available data files.
    :param files: list of file paths
    :param prompt: prompt text for the input
    :return: the selected file path, 'all' for every file, or None to go back
    """
    print("\nData files:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    print("  [a] all files")
    print("  [q] back")
    choice = input(prompt + ": ").strip().lower()

    if choice == 'q':
        return None
    if choice == 'a':
        return 'all'
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
    except ValueError:
        pass
    print("  invalid selection")
    return None


def get_number(prompt, allow_float):
    """
    Gets a positive number from the user.
    :param prompt: prompt text
    :param allow_float: True for error thresholds, False for segment counts
    :return: the number entered
    """
    while True:
        try:
            value = input(prompt)
            value = float(value) if allow_float else int(value)
        except ValueError:
            print("  please enter a valid number")
            continue
        if value <= 0:
            print("  please enter a value larger than 0")
            continue
        return value


def pick_quick_method():
    """
    Lets the user pick one of the three quick-pick generation methods and its
    parameter value. Used by the visualize menus to add joints before animating.
    :return: (method_id, parameter_value) or None if the selection was invalid
    """
    quick_pick = {'1': 'grow_area', '2': 'grow_bs_area', '3': 'equal_segments'}
    print("  [1] grow_segments (area)")
    print("  [2] grow_segments_binary_search (area)")
    print("  [3] create_equal_segments")
    method_id = quick_pick.get(input("  Select a method: ").strip())
    if method_id is None:
        print("  invalid method")
        return None
    allow_float = reg.METHODS_BY_ID[method_id][5] == 'err'
    value = get_number("  error threshold (cm^2): " if allow_float else "  number of segments: ",
                       allow_float=allow_float)
    return method_id, value


def get_frame_interval(default, skip_allowed=True):
    """
    Asks for the frame-animation interval in seconds; Enter uses the default.
    :param default: interval used when the user presses Enter
    :param skip_allowed: True if entering 0 skips the animation, False if 0
                         falls back to the default
    :return: the interval in seconds
    """
    hint = f"Enter = {default}, 0 to skip" if skip_allowed else f"Enter = {default}"
    try:
        interval = float(input(f"  frame interval in seconds ({hint}): ").strip() or default)
    except ValueError:
        interval = default
    if interval <= 0:
        return 0.0 if skip_allowed else default
    return interval


def compare_methods_cli(data_path, save_path):
    """
    Compares a generation method against increasing error thresholds.
    :param data_path: directory of the fish midline data
    :param save_path: directory to save the results to
    :return: None
    """
    print("\n--- Compare method with increasing error ---")
    methods = {str(i): reg.METHODS_BY_ID[method_id] for i, method_id in enumerate(CLI_COMPARE_IDS, 1)}
    for key, method in methods.items():
        print(f"  [{key}] {method[1]}")
    print("  [q] back")

    while True:
        key = input("Select a method (q: back): ").strip().lower()
        if key == 'q':
            return
        if key not in methods:
            print("  invalid method")
            continue
        gd.compare_method_error(methods[key][2], 2, 40, 2, data_path, save_path)
        print("  comparison saved to", save_path)


def visualize_data_cli(files, save_path):
    """
    Lets the user pick any data file (.xls or .csv), optionally apply a generation
    method, then animate through every frame with a fixed scale and fading echoes.
    :param files: list of available data files
    :param save_path: directory to save plots to
    :return: None
    """
    print("\n--- Visualize data (frame animation) ---")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    try:
        idx = int(input("Select a data file: ").strip()) - 1
        if not (0 <= idx < len(files)):
            print("  invalid selection")
            return
    except ValueError:
        print("  invalid selection")
        return

    midline = load_data_file(files[idx])
    if midline is None:
        return

    if len(midline[0]) <= 1:
        print(f"  {os.path.basename(files[idx])} has only 1 frame, nothing to animate")
        return

    joints = []
    if input("Apply a joint generation method? (y/n): ").strip().lower() == 'y':
        choice = pick_quick_method()
        if choice is None:
            return
        joints = reg.call_by_id(choice[0], midline, choice[1])
        total_error = ce.find_total_error(joints, midline)
        print(f"  {len(joints)} joints, avg area error {total_error[1]:.4f}, "
              f"avg linear error {total_error[0]:.4f}")

    animate_frames(midline, joints, get_frame_interval(0.05))


def visualize_3d_animation_cli(files):
    """
    Lets the user pick one data file or all of them, then renders the fish midlines
    in 3D. Multiple datasets are arranged in a grid on one page and animate at the
    same time until the user pauses, steps forward, or closes the window.
    :param files: list of available data files
    :return: None
    """
    print("\n--- Visualize data (3D animation) ---")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    print("  [a] all files in the folder")
    print("  [q] back")
    choice = input("Select a data file (or 'a' for all): ").strip().lower()
    if choice == 'q':
        return

    selected = []
    if choice == 'a':
        selected = files
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected = [files[idx]]
            else:
                print("  invalid selection")
                return
        except ValueError:
            print("  invalid selection")
            return

    datasets = []
    for f in selected:
        data3d = load_3d_render_data(f)
        if data3d is None:
            print(f"  failed to load {os.path.basename(f)}")
            continue
        if len(data3d) <= 1:
            print(f"  {os.path.basename(f)} has only 1 frame, skipping")
            continue
        datasets.append({'name': os.path.basename(f), 'path': f, 'data': data3d, 'joints': []})

    if not datasets:
        print("  no datasets with more than one frame loaded")
        return

    if input("Apply a joint generation method to the dataset(s)? (y/n): ").strip().lower() == 'y':
        choice = pick_quick_method()
        if choice is not None:
            method_id, value = choice
            for ds in datasets:
                midline2d = load_data_file(ds['path'])
                if midline2d is not None:
                    ds['joints'] = reg.call_by_id(method_id, midline2d, value)

    print(f"\n  rendering {len(datasets)} dataset(s) in 3D ...")
    render_3d_animation(datasets, get_frame_interval(0.04, skip_allowed=False))


class StepViewerStopped(Exception):
    """
    Raised by the step viewer callback when the user chooses to stop early.
    """
    pass


def make_step_viewer(midline, pause_time, step_every, method_name):
    """
    Creates a step callback that plots the current joint configuration on matplotlib
    after each iteration of a generation method.
    :param midline: the fish midline data
    :param pause_time: seconds to pause between steps, 0 to press Enter instead
    :param step_every: show every Nth step (1 = every step)
    :param method_name: name shown in the plot title
    :return: a step callback for a generation method
    """
    counter = {"n": 0}

    def viewer(joints, midline_ref, step_index):
        """Step callback: plots the current joints, then pauses or waits for Enter."""
        counter["n"] += 1
        if counter["n"] % step_every != 0:
            return

        plt.cla()

        # draw the midline for the relevant frames as a series of line plots
        frames_to_draw = [0]
        if len(midline[0]) > 1:
            frames_to_draw.append(len(midline[0]) // 2)
        traced_head = None
        traced_tail = None
        for ff in frames_to_draw:
            traced = midline_segments([midline[p][ff] for p in range(len(midline))])
            for i in range(len(traced) - 1):
                plt.plot([traced[i][0], traced[i + 1][0]],
                         [traced[i][1], traced[i + 1][1]], color='b', linewidth=1.0)
            if traced and traced_head is None:
                traced_head = traced[0]
                traced_tail = traced[-1]

        # draw the segments between joints
        for i in range(len(joints) - 1):
            xs = [midline[joints[i][2]][0][0], midline[joints[i + 1][2]][0][0]]
            ys = [midline[joints[i][2]][0][1], midline[joints[i + 1][2]][0][1]]
            plt.plot(xs, ys, 'r-', linewidth=1.5)

        for j in joints:
            plt.scatter(midline[j[2]][0][0], midline[j[2]][0][1], color='green', s=20)
        if traced_head is not None:
            plt.scatter(traced_head[0], traced_head[1], color='red', label='head')
            plt.scatter(traced_tail[0], traced_tail[1], color='orange', label='tail')

        plt.title(f"{method_name} - step {counter['n']}: {len(joints)} joints", fontsize=TITLE_HALF_SIZE)
        plt.legend(loc="best")
        plt.draw()

        if pause_time > 0:
            plt.pause(pause_time)
        else:
            answer = input("  press Enter to continue, 'q' to stop: ").strip().lower()
            if answer == 'q':
                raise StepViewerStopped()

    return viewer


def animate_frames(midline, joints, interval, echo_frames=8):
    """
    Animates the joint configuration across every frame of the midline, showing
    where each joint lies on the fish as it moves. Each frame leaves a fading echo
    of the last few frames that gets lighter the older it is and disappears after
    echo_frames frames. The axis scale stays identical across every frame.
    :param midline: the fish midline data [point][frame][x, y]
    :param joints: the joint configuration [[x, y, point_index], ...]
    :param interval: seconds between frames, 0 to skip the animation
    :param echo_frames: number of recent frames kept as a fading shadow
    :return: None
    """
    num_frames = len(midline[0])
    if num_frames <= 1 or interval <= 0:
        return

    # precompute the traced midline for every frame (skips padding, traces the body)
    frames_pts = [[midline[p][f] for p in range(len(midline))] for f in range(num_frames)]
    traced_frames = trace_all_frames(frames_pts)

    # fixed axis limits so the scale stays identical across every frame (ignoring padding)
    all_pts = [midline[p][f] for f in range(num_frames) for p in range(len(midline))
               if not is_padding_point(midline[p][f])]
    all_x = [p[0] for p in all_pts]
    all_y = [p[1] for p in all_pts]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    x_pad = x_range * 0.05 if x_range > 0 else 1.0
    y_pad = y_range * 0.05 if y_range > 0 else 1.0
    x_limits = (min(all_x) - x_pad, max(all_x) + x_pad)
    y_limits = (min(all_y) - y_pad, max(all_y) + y_pad)

    history = []  # most recent frame states, oldest first

    for f in range(num_frames):
        traced = traced_frames[f]
        joint_points = [midline[j[2]][f] for j in joints]

        history.append((traced, joint_points))
        if len(history) > echo_frames:
            history.pop(0)

        plt.cla()

        # fading echo trail of the previous frames (older = lighter)
        echo_jx = []
        echo_jy = []
        echo_jc = []
        for age, (htraced, hjp) in enumerate(reversed(history[:-1]), start=1):
            alpha = (echo_frames - age) / echo_frames
            if alpha <= 0:
                continue
            if len(htraced) > 1:
                segs = [((htraced[i][0], htraced[i][1]), (htraced[i + 1][0], htraced[i + 1][1]))
                        for i in range(len(htraced) - 1)]
                plt.gca().add_collection(LineCollection(segs, colors=(0.5, 0.5, 0.5, alpha), linewidths=1.2))
            if hjp:
                echo_jx.extend(p[0] for p in hjp)
                echo_jy.extend(p[1] for p in hjp)
                echo_jc.extend([(0.5, 0.5, 0.5, alpha)] * len(hjp))
        if echo_jx:
            plt.scatter(echo_jx, echo_jy, c=echo_jc, s=10)

        # the fish midline for this frame (full opacity, series of line segments)
        if traced:
            segs = [((traced[i][0], traced[i][1]), (traced[i + 1][0], traced[i + 1][1]))
                    for i in range(len(traced) - 1)]
            plt.gca().add_collection(LineCollection(segs, colors='b', linewidths=1.5))
            plt.scatter(traced[0][0], traced[0][1], color='red', label='head')
            plt.scatter(traced[-1][0], traced[-1][1], color='orange', label='tail')

        # the joints at this frame's position on the midline
        if len(joint_points) > 1:
            jsegs = [((joint_points[i][0], joint_points[i][1]),
                      (joint_points[i + 1][0], joint_points[i + 1][1]))
                     for i in range(len(joint_points) - 1)]
            plt.gca().add_collection(LineCollection(jsegs, colors='r', linewidths=1.5))
        if joint_points:
            plt.scatter([p[0] for p in joint_points], [p[1] for p in joint_points], color='green', s=20)

        plt.title(f"Frame {f + 1} / {num_frames} - {len(joints)} joints", fontsize=TITLE_HALF_SIZE)
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc="upper right")
        if interval > 0:
            plt.pause(interval)

    plt.show()  # keep the final frame visible


def render_3d_animation(datasets, interval=0.04, echo_frames=8):
    """
    Renders 3D midline animation(s). Each dataset is a dict with:
        name   : label shown in the title
        data   : [frame][point][x, y, z]
        joints : joint configuration [[x, y, point_index], ...] or []
    A single dataset uses one 3D axes; multiple datasets are arranged in a grid,
    all animating at the same time with a fixed scale. Recent frames leave a
    fading echo that gets lighter the older it is and disappears after
    echo_frames frames. Space pauses/resumes, arrow keys step one frame when
    paused, and closing the window exits.
    :param datasets: list of dataset dictionaries
    :param interval: seconds between frames
    :param echo_frames: number of recent frames kept as a fading shadow
    :return: None
    """
    if not datasets:
        return

    # rotate the axes: each data point [x, y, z] is displayed as [y, z, x],
    # so the body length (x) runs on the z axis, the width (y) on the x axis,
    # and the depth (z) on the y axis.
    for ds in datasets:
        ds['data'] = [[[p[1], p[2], p[0]] for p in frame] for frame in ds['data']]

    num_datasets = len(datasets)
    if num_datasets == 1:
        fig = plt.figure(figsize=(7, 6))
        axes = [fig.add_subplot(111, projection='3d')]
    else:
        cols = int(np.ceil(np.sqrt(num_datasets)))
        rows = int(np.ceil(num_datasets / cols))
        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        axes = [fig.add_subplot(rows, cols, k + 1, projection='3d') for k in range(rows * cols)]
        for ax in axes[num_datasets:]:
            ax.set_visible(False)

    # fixed axis limits per dataset so the scale does not change between frames (ignoring padding)
    limits = []
    for ds in datasets:
        all_pts = [p for f in ds['data'] for p in f if not is_padding_point(p)]
        all_x = [p[0] for p in all_pts]
        all_y = [p[1] for p in all_pts]
        all_z = [p[2] for p in all_pts]
        cx = (min(all_x) + max(all_x)) / 2
        cy = (min(all_y) + max(all_y)) / 2
        cz = (min(all_z) + max(all_z)) / 2
        span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)) / 2
        if span <= 0:
            span = 1.0
        limits.append((cx - span, cx + span, cy - span, cy + span, cz - span, cz + span))

    # precompute traced midlines for every frame of every dataset (skips padding, traces the body)
    traced_data = [trace_all_frames([ds['data'][f] for f in range(len(ds['data']))]) for ds in datasets]

    max_frames = max(len(ds['data']) for ds in datasets)
    state = {'paused': False, 'frame': 0}
    histories = [[] for _ in range(num_datasets)]  # most recent frame states per dataset

    def draw_frame(f):
        """Renders one animation frame for every dataset onto its 3D axes."""
        for idx, ds in enumerate(datasets):
            ax = axes[idx]
            n = len(ds['data'])
            pts = ds['data'][f % n]
            traced = traced_data[idx][f % n]
            joint_points = [pts[j[2]] for j in ds['joints']]

            histories[idx].append((traced, joint_points))
            if len(histories[idx]) > echo_frames:
                histories[idx].pop(0)

            ax.clear()

            # fading echo trail of the previous frames (older = lighter)
            echo_jx = []
            echo_jy = []
            echo_jz = []
            echo_jc = []
            for age, (ht, hjp) in enumerate(reversed(histories[idx][:-1]), start=1):
                alpha = (echo_frames - age) / echo_frames
                if alpha <= 0:
                    continue
                if len(ht) > 1:
                    hsegs = [((ht[i][0], ht[i][1], ht[i][2]),
                              (ht[i + 1][0], ht[i + 1][1], ht[i + 1][2]))
                             for i in range(len(ht) - 1)]
                    ax.add_collection(Line3DCollection(hsegs, colors=(0.5, 0.5, 0.5, alpha), linewidths=1.2))
                for p in hjp:
                    echo_jx.append(p[0])
                    echo_jy.append(p[1])
                    echo_jz.append(p[2])
                    echo_jc.append((0.5, 0.5, 0.5, alpha))
            if echo_jx:
                ax.scatter(echo_jx, echo_jy, echo_jz, c=echo_jc, s=10)

            # midline drawn as a series of line segments (traced order, no padding)
            if traced:
                segs = [((traced[i][0], traced[i][1], traced[i][2]),
                         (traced[i + 1][0], traced[i + 1][1], traced[i + 1][2]))
                        for i in range(len(traced) - 1)]
                ax.add_collection(Line3DCollection(segs, colors='b', linewidths=1.5))
                ax.scatter(traced[0][0], traced[0][1], traced[0][2], color='red', s=60)
                ax.scatter(traced[-1][0], traced[-1][1], traced[-1][2], color='orange', s=60)
            if joint_points:
                if len(joint_points) > 1:
                    jsegs = [((joint_points[i][0], joint_points[i][1], joint_points[i][2]),
                              (joint_points[i + 1][0], joint_points[i + 1][1], joint_points[i + 1][2]))
                             for i in range(len(joint_points) - 1)]
                    ax.add_collection(Line3DCollection(jsegs, colors='r', linewidths=1.5))
                ax.scatter([p[0] for p in joint_points], [p[1] for p in joint_points],
                           [p[2] for p in joint_points], color='green', s=20)
            ax.set_xlim(limits[idx][0], limits[idx][1])
            ax.set_ylim(limits[idx][2], limits[idx][3])
            ax.set_zlim(limits[idx][4], limits[idx][5])
            ax.set_xlabel('Y', fontsize=TITLE_HALF_SIZE)
            ax.set_ylabel('Z', fontsize=TITLE_HALF_SIZE)
            ax.set_zlabel('X', fontsize=TITLE_HALF_SIZE)
            ax.set_title(f"{ds['name']} - frame {(f % n) + 1}/{n} - {len(ds['joints'])} joints",
                         fontsize=TITLE_HALF_SIZE)
        fig.canvas.draw_idle()

    def on_key(event):
        """Keyboard handler: space toggles pause, arrows step one frame."""
        if event.key == ' ':
            state['paused'] = not state['paused']
        elif event.key in ('right', 'n', '.'):
            state['frame'] = (state['frame'] + 1) % max_frames
            draw_frame(state['frame'])
        elif event.key in ('left', 'b', ','):
            state['frame'] = (state['frame'] - 1) % max_frames
            draw_frame(state['frame'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.suptitle("space: pause/resume, arrows: step when paused, close window to exit",
                 fontsize=TITLE_HALF_SIZE)

    draw_frame(0)

    # a non-interactive backend cannot animate, so show the first frame only
    # (note: tkagg contains 'agg' as a substring, so use a real interactive check)
    if not is_interactive_backend():
        print("  non-interactive backend detected; install a GUI toolkit (e.g. tkinter) or run with\n"
              "  MPLBACKEND=TkAgg to see the animated window. Showing the first frame only.")
        return

    try:
        while plt.fignum_exists(fig.number):
            if not state['paused']:
                state['frame'] = (state['frame'] + 1) % max_frames
                draw_frame(state['frame'])
            plt.pause(interval)
    except Exception:
        pass
    plt.close(fig)


def visualise_generation_cli(files, save_path):
    """
    Lets the user pick a generation method and a data file, then watches the joint
    configuration build up one iteration at a time on a matplotlib plot.
    :param files: list of available data files
    :param save_path: directory to save a final plot to
    :return: None
    """
    print("\n--- Step through joint generation (visual) ---")
    methods = reg.METHODS  # (id, label, func, param name, param label, param type)
    for i, m in enumerate(methods, 1):
        print(f"  [{i}] {m[1]}")
    print("  [q] back")

    try:
        choice = int(input("Select a method: ").strip())
        if not (1 <= choice <= len(methods)):
            print("  invalid selection")
            return
    except ValueError:
        print("  invalid selection")
        return

    _, method_name, _, _, param_label, param_type = methods[choice - 1]
    value = get_number(f"  {param_label}: ", allow_float=param_type == 'err')

    try:
        pause_time = float(input("  pause between steps in seconds (0 = press Enter to advance): ").strip())
        if pause_time < 0:
            pause_time = 0
    except ValueError:
        pause_time = 0

    try:
        step_every = int(input("  show every Nth step (1 = every step): ").strip())
        if step_every < 1:
            step_every = 1
    except ValueError:
        step_every = 1

    file_path = pick_data_file(files, "Select a file to step through")
    if file_path is None or file_path == 'all':
        print("  please pick a single file")
        return

    midline = load_data_file(file_path)
    if midline is None:
        return

    print(f"\n  stepping through {method_name} ...")
    viewer = make_step_viewer(midline, pause_time, step_every, method_name)
    try:
        joints = reg.call(methods[choice - 1], midline, value, step_callback=viewer)
    except StepViewerStopped:
        print("\n  step viewing stopped early")
        return

    plt.show()  # keep the final step window open
    total_error = ce.find_total_error(joints, midline)
    print(f"  finished: {len(joints)} joints, avg area error {total_error[1]:.4f}, "
          f"avg linear error {total_error[0]:.4f}")

    if len(midline[0]) > 1:
        animate_frames(midline, joints, get_frame_interval(0.05))


def pick_method_and_save_all(data_path, *save_path):
    """
    CLI interface that lets the user select data and a generation method, run it,
    and save the resulting plots.
    :param data_path: directory of fish midline data
    :param save_path: directory to save the graphs
    :return: None
    """
    if save_path:
        user_save_path = get_user_save_path(data_path, *save_path)
    else:
        user_save_path = get_user_save_path(data_path)

    files = sorted(glob.glob(data_path + '/*.xls') + glob.glob(data_path + '/*.csv'))
    if not files:
        print("No data files (.xls or .csv) found in", data_path)
        return

    while True:
        print("\n" + "=" * 62)
        print("AUTOMATIC FISH MIDLINE SEGMENTATION")
        print("=" * 62)
        print(f"  data folder : {data_path}")
        print(f"  save folder : {user_save_path}")
        print(f"  data files  : {len(files)}")
        print()
        print("  Generation methods:")
        for i, method_id in enumerate(CLI_COMPARE_IDS, 1):
            print(f"    [{i}] {reg.METHODS_BY_ID[method_id][1]}")
        print()
        print("    [7] compare method with increasing error")
        print("    [8] visualize data (frame animation)")
        print("    [9] step through joint generation (visual)")
        print("    [10] visualize data (3D animation)")
        print("    [q] quit")
        print()

        choice = input("Select an option: ").strip().lower()

        if choice == 'q':
            break

        method_map = {str(i): method_id for i, method_id in enumerate(CLI_COMPARE_IDS, 1)}

        if choice == '7':
            compare_methods_cli(data_path, user_save_path)
            continue
        if choice == '8':
            visualize_data_cli(files, user_save_path)
            continue
        if choice == '9':
            visualise_generation_cli(files, user_save_path)
            continue
        if choice == '10':
            visualize_3d_animation_cli(files)
            continue
        if choice not in method_map:
            print("\n  invalid option")
            continue

        _, _, method, param_name, param_label, param_type = reg.METHODS_BY_ID[method_map[choice]]
        print(f"\n  {method.__name__} uses {param_label}")
        value = get_number(f"  {param_label}: ", allow_float=param_type == 'err')
        params = {param_name: value}

        target = pick_data_file(files, "Select a file to process")
        if target is None:
            continue
        if target == 'all':
            gd.use_all_folder_data(method, data_path, user_save_path, **params)
        else:
            midline = load_data_file(target)
            if midline is None:
                print("  failed to load", target)
                continue
            run_generation(midline, method, params, user_save_path, os.path.basename(target))


def main():
    """Console-script entry point for the CLI."""
    directory = set_data_folder()
    pick_method_and_save_all(data_path=directory)


# run code only when called as a script
if __name__ == "__main__":
    main()

