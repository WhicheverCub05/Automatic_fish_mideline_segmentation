"""gather_data tests: the two CLI-used functions write their outputs."""

import glob
import os

import gather_data as gd
import method_registry as reg


def test_compare_method_error_writes_csv_and_plots(real_data_dir, tmp_path):
    # compare_method_error only handles .xls files, so it needs the real data
    # folder (skipped on CI where the data is absent)
    gd.compare_method_error(reg.METHODS_BY_ID["grow_area"][2], 2, 10, 2, real_data_dir, str(tmp_path))
    csvs = glob.glob(os.path.join(str(tmp_path), "*_All_Data.csv"))
    assert csvs
    content = open(csvs[0], encoding="utf-8").read()
    assert "error_threshold" in content
    assert "number of joints" in content
    assert os.path.getsize(csvs[0]) > 100  # header + one row per xls file
    assert glob.glob(os.path.join(str(tmp_path), "*.svg")), "no comparison plots saved"


def test_use_all_folder_data_processes_every_file(synthetic_data_dir, tmp_path):
    data_dir, save_dir = synthetic_data_dir
    gd.use_all_folder_data(reg.METHODS_BY_ID["grow_bs_area"][2], data_dir, save_dir,
                           error_threshold=5.0)
    svgs = glob.glob(os.path.join(save_dir, "*.svg"))
    assert svgs, "no SVGs saved"
    assert len(svgs) == 1, "expected one SVG for the one data file"


def test_use_all_folder_data_segment_count_parameter(synthetic_data_dir):
    data_dir, save_dir = synthetic_data_dir
    gd.use_all_folder_data(reg.METHODS_BY_ID["equal_segments"][2], data_dir, save_dir,
                           segment_count=6)
    assert glob.glob(os.path.join(save_dir, "*.svg"))


def test_use_all_folder_data_missing_folder():
    """A missing data folder must not raise."""
    gd.use_all_folder_data(reg.METHODS_BY_ID["grow_area"][2], "C:\\no\\such\\folder", "C:\\tmp\\nowhere")
