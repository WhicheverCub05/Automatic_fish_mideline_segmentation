"""
Shared data utilities for the fish midline project: loading, tracing, sine-wave
synthesis and basic plotting. Keeping these in a neutral module lets
main.py, gather_data.py, web_viewer.py and midline_fluid_sim.py use them
without importing the CLI module (which would create import cycles).
"""

__author__ = "Alex R.d Silva"

import math
import traceback

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Midline:
    """
    Thin, list-compatible wrapper for the generation-format midline
    [point][frame][x, y] (point-major, used by every generation method and error
    metric). Indexing it (midline[p][f], len(midline), iteration, np.asarray)
    behaves exactly like the raw list, so existing code works unchanged, while
    the extra properties give the canonical frame-major numpy view (cached), a
    lazy render-format conversion and the per-frame real point counts (the
    padding information the loaders know but the raw list drops).
    """

    def __init__(self, gen, real_lengths=None):
        """Wraps a point-major midline list with optional per-frame real lengths."""
        self._gen = gen
        self._real_lengths = real_lengths  # per-frame real (non-padding) point count
        self._frame_major = None
        self._render = None

    # --- list protocol (generation format: point-major) ---
    def __len__(self):
        """Number of points along the body."""
        return len(self._gen)

    def __getitem__(self, p):
        """Returns a point's frame list (list-compatible indexing)."""
        return self._gen[p]

    def __iter__(self):
        """Iterates over points (list-compatible)."""
        return iter(self._gen)

    def __repr__(self):
        """Short shape description for the console."""
        return f"Midline({self.num_points} points x {self.num_frames} frames)"

    @property
    def num_points(self):
        """Number of points along the body."""
        return len(self._gen)

    @property
    def num_frames(self):
        """Number of frames (0 for an empty midline)."""
        return len(self._gen[0]) if self._gen else 0

    @property
    def shape(self):
        """(points, frames, 2) tuple describing the generation format."""
        return (self.num_points, self.num_frames, 2)

    @property
    def real_lengths(self):
        """Per-frame real (non-padding) point counts; None when there is no padding."""
        return self._real_lengths

    @property
    def frame_major(self):
        """
        Canonical frame-major numpy array (frames, points, 2), built once and
        cached. This is the working representation for the vectorized metrics.
        """
        if self._frame_major is None:
            self._frame_major = np.asarray(self._gen, dtype=float).transpose(1, 0, 2)
        return self._frame_major

    @property
    def render(self):
        """
        Lazy render-format conversion [frame][point][x, y, z]. Points beyond a
        frame's real length are zero-padded (matching what the render loaders
        produce), so the viewer can detect padding with is_padding_point().
        """
        if self._render is None:
            nf = self.num_frames
            np_ = self.num_points
            render = [[[0.0, 0.0, 0.0] for _ in range(np_)] for _ in range(nf)]
            for f in range(nf):
                n = self._real_lengths[f] if self._real_lengths else np_
                for p in range(n):
                    x, y = self._gen[p][f]
                    render[f][p][0] = x
                    render[f][p][1] = y
            self._render = render
        return self._render


def is_padding_point(p, eps=1e-9):
    """
    True if a point is a zero padding point added when a frame had fewer points.
    :param p: a point coordinate list or tuple
    :return: True if all coordinates are ~0
    """
    return all(abs(c) < eps for c in p)


def trace_indices(points, eps=1e-9):
    """
    Orders a set of midline points into a nearest-neighbour chain, skipping zero
    padding points, and returns the ORIGINAL list indices in traced order. The
    caller can use these to reference the source data (e.g. joint rows) instead
    of coordinates. The distances are computed with numpy, which runs without
    the GIL and is therefore a good candidate to run in parallel threads.
    :param points: list of point coordinate lists [x, y, ...] (2D or 3D)
    :param eps: coordinates below this magnitude count as padding
    :return: list of original indices in traced order
    """
    real = [i for i, p in enumerate(points) if not is_padding_point(p, eps)]
    if not real:
        return []
    arr = np.asarray([points[i] for i in real], dtype=float)
    n = len(arr)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    last = arr[0]
    for _ in range(1, n):
        d = np.sum((arr - last) ** 2, axis=1)
        d[visited] = np.inf
        k = int(np.argmin(d))
        order.append(k)
        visited[k] = True
        last = arr[k]
    return [real[k] for k in order]


def midline_segments(points):
    """
    Traces a set of midline points into a nearest-neighbour chain (see
    trace_indices) and returns the coordinates in that order, so the body can be
    drawn as a series of line segments instead of one polyline that would
    zigzag through the unordered data.
    :param points: list of point coordinate lists [x, y, ...] (2D or 3D)
    :return: points in traced order
    """
    return [points[k] for k in trace_indices(points)]


def trace_all_frames(frames):
    """
    Traces every frame into a nearest-neighbour chain, processing the frames in
    parallel threads. numpy releases the GIL, so these vectorized computations
    genuinely use multiple cores / hyperthreading.
    :param frames: list of per-frame point lists
    :return: list of traced point lists, in the same order as frames
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as ex:
        return list(ex.map(midline_segments, frames))


def load_midline_data(location):
    """
    Loads .xls files and creates a 3D array of rows, columns and [x, y] values that define the midline
    :param location: location of Excel file with midlines
    :return: the midline from the Excel file
    """
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
        return Midline(midline)

    except FileNotFoundError:
        print("the file is not found")


def load_3d_csv_data(location, midline_type='midline', for_generation=False):
    """
    Loads 3D CSV files (Fish_NN_Values_3D.csv or Fish_Raw_Values_3D.csv) and creates a 3D array
    of rows, columns and [x, y, z] values that define the midline.
    :param location: location of CSV file with 3D midline data
    :param midline_type: 'midline' for Fish_NN_Values_3D.csv or 'raw' for Fish_Raw_Values_3D.csv
    :param for_generation: if True, return data in format [point][frame][x,y] for generation methods
                          if False, return data in format [frame][point][x,y,z] for visualization
    :return: the midline from the CSV file as a 3D array
    """
    print("Loading 3D data from:", location)
    try:
        df = pd.read_csv(location)

        # Check which columns are available
        has_midline_cols = 'Midline_X' in df.columns and 'Midline_Y' in df.columns
        has_relative_cols = 'Relative_X' in df.columns and 'Relative_Y' in df.columns
        z_col = 'Midline_Z' if 'Midline_Z' in df.columns else None

        if midline_type == 'midline':
            if has_midline_cols:
                # Traditional format with Midline_X, Midline_Y (Midline_Z optional -> z = 0)
                return csv_data_to_midline(df, 'Midline_X', 'Midline_Y', z_col, for_generation)
            if has_relative_cols:
                # Fish_NN_Values_3D.csv format - uses Relative_X, Relative_Y (treated as midline data)
                return csv_data_to_midline(df, 'Relative_X', 'Relative_Y', None, for_generation)
            print("Error: CSV file does not contain expected columns (Midline_X/Y/Z or Relative_X/Y)")
            return None

        elif midline_type == 'raw':
            if has_relative_cols:
                # Fish_Raw_Values_3D.csv format - uses Relative_X, Relative_Y
                return csv_data_to_midline(df, 'Relative_X', 'Relative_Y', None, for_generation)
            if has_midline_cols:
                # Fish_Raw_Values_3D.csv format - uses Midline_X, Midline_Y, Midline_Z (despite the name)
                return csv_data_to_midline(df, 'Midline_X', 'Midline_Y', z_col, for_generation)
            print("Error: CSV file does not contain Relative_X/Y or Midline_X/Y/Z columns for raw data")
            return None

    except FileNotFoundError:
        print("The file is not found")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        traceback.print_exc()
        return None


def csv_data_to_midline(df, x_col, y_col, z_col, for_generation):
    """
    Converts a long-format tracking DataFrame into a midline array. Rows with NaN
    coordinates are dropped. Frames with fewer points than the longest frame are
    padded: the generation format repeats each frame's last real point (its tail)
    so generation methods and error metrics never evaluate against synthetic zero
    points; the render format is zero-padded so is_padding_point() can detect it.
    :param df: tracking data DataFrame with a Frame column
    :param x_col: name of the x coordinate column
    :param y_col: name of the y coordinate column
    :param z_col: name of the z coordinate column, or None for 2D data (z = 0)
    :param for_generation: True for [point][frame][x, y], False for [frame][point][x, y, z]
    :return: the midline array
    """
    subset = [x_col, y_col] if z_col is None else [x_col, y_col, z_col]
    df_clean = df.dropna(subset=subset)

    frames = sorted(df_clean['Frame'].unique())
    num_frames = len(frames)

    # per-frame lists of real (non-NaN) points
    frame_pts = []
    for frame in frames:
        frame_data = df_clean[df_clean['Frame'] == frame].reset_index(drop=True)
        if z_col is None:
            frame_pts.append([[frame_data.iloc[k][x_col], frame_data.iloc[k][y_col], 0.0]
                              for k in range(len(frame_data))])
        else:
            frame_pts.append([[frame_data.iloc[k][x_col], frame_data.iloc[k][y_col], frame_data.iloc[k][z_col]]
                              for k in range(len(frame_data))])

    num_points = max((len(fp) for fp in frame_pts), default=0)

    if for_generation:
        # Transpose to [point][frame][x, y]; rows beyond a frame's real length
        # repeat that frame's tail point, so the tail joint is a real point in
        # every frame and no segment is ever scored against zero padding
        midline = [[[0.0 for _ in range(2)] for _ in range(num_frames)] for _ in range(num_points)]
        real_lengths = []
        for f, fp in enumerate(frame_pts):
            real = len(fp)
            real_lengths.append(real)
            tail_x = fp[-1][0]
            tail_y = fp[-1][1]
            for p in range(num_points):
                if p < real:
                    midline[p][f][0] = fp[p][0]
                    midline[p][f][1] = fp[p][1]
                else:
                    midline[p][f][0] = tail_x
                    midline[p][f][1] = tail_y
        print(f"Loaded {num_frames} frames with up to {num_points} points each (for generation methods)")
        return Midline(midline, real_lengths)

    # Render format [frame][point][x, y, z], zero-padded so short frames stay rectangular
    temp_midline = [[[0 for _ in range(3)] for _ in range(num_points)] for _ in range(num_frames)]
    for f, fp in enumerate(frame_pts):
        for p, pt in enumerate(fp):
            temp_midline[f][p][0] = pt[0]
            temp_midline[f][p][1] = pt[1]
            temp_midline[f][p][2] = pt[2]
    print(f"Loaded {num_frames} frames with up to {num_points} points each (3D midline data)")
    return temp_midline


def load_data_file(file_path):
    """
    Loads a midline data file (.xls or .csv) into the format used by generation methods
    :param file_path: path to the data file
    :return: the midline data
    """
    if file_path.lower().endswith('.csv'):
        return load_3d_csv_data(file_path, midline_type='midline', for_generation=True)
    return load_midline_data(file_path)


def load_3d_render_data(file_path):
    """
    Loads a data file into the 3D format [frame][point][x, y, z] used for 3D rendering.
    CSV files keep their stored coordinates; Excel files are 2D and get z = 0.
    :param file_path: path to the data file
    :return: the midline as [frame][point][x, y, z], or None on failure
    """
    if file_path.lower().endswith('.csv'):
        return load_3d_csv_data(file_path, midline_type='midline')
    midline = load_midline_data(file_path)
    if midline is None:
        return None
    num_points = len(midline)
    num_frames = len(midline[0])
    return [[[midline[p][f][0], midline[p][f][1], 0.0] for p in range(num_points)] for f in range(num_frames)]


def generate_midline_from_sinewave(cycles, amplitude, length_cm, phase_difference, frames, resolution):
    """
    Function creates midlines from a sine wave that match the 3D data structure of a midline for this application.
    the x and y values for each sinewave are in an array and multiple sine waves are in the midline.
    e.g. midline with: resolution = 200, 4 waves -> len(midline) = 200, len(midline[0]) = 4, midline[0][0] = [x ,y]
    :param cycles: number of complete cycles
    :param amplitude: max length in y and -y in cm
    :param length_cm: total length of the sine wave in cm
    :param phase_difference: phase difference between subsequent sine waves
    :param frames: number of different sine waves in the midline
    :param resolution: number of data points that describe a sine wave
    :return: 3D array of each sine wave x and y position for each frame.
    """
    midline = [[[0 for _ in range(2)] for _ in range(frames)] for _ in range(resolution)]
    x_values = np.linspace(0, length_cm, num=resolution)
    base_phase = (x_values * cycles) / length_cm * 2 * np.pi

    phase = 0

    for f in range(frames):
        wave = np.sin(base_phase + phase) * amplitude
        for r in range(resolution):
            midline[r][f][0] = x_values[r]
            midline[r][f][1] = wave[r]
        phase += phase_difference

    return Midline(midline)


def is_interactive_backend():
    """
    True if the current matplotlib backend can show windows (TkAgg, QtAgg, ...).
    :return: True if the backend is interactive
    """
    try:
        from matplotlib.backends import backend_registry
        from matplotlib.backends.backend_registry import BackendFilter
        interactive = {b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    except Exception:
        from matplotlib import rcsetup
        interactive = {b.lower() for b in rcsetup.interactive_bk}
    return plt.get_backend().lower() in interactive


def plot_midline(midline, *frames):
    """
    Function that allows for the midline and selected frames to be plotted using matplotlib
    :param midline: The fish midline to plot
    :param frames: Which frame or column of midline data to plot
    :return: None
    """
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


def joints_to_length(joints, *plot_on_first_frame):
    """
    Turns joint data into actual lengths that can be used to create real segments for a robot fish
    :param joints: The joint configuration data -> [[x, y, midline_row], ...]
    :param plot_on_first_frame: Option to plot the joints on the first frame of the midline instead of a straight line
    :return: array of lengths between the joints
    """
    segments = [0]
    length = 0
    plt.scatter(length, 0, color='red', label="start of head")

    tmp_joints_x = []
    tmp_joints_y = []
    for j in range(len(joints)):
        tmp_joints_x.append(joints[j][0])
        tmp_joints_y.append(joints[j][1])

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
        plt.legend(loc="best")
    return segments
