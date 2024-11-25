import os
import pytest
import numpy as np
import pandas as pd
import tempfile
from unittest.mock import patch
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.preprocess.segment_split import (
    split_segment,
    save_segment,
)  # Replace with actual module name


@pytest.fixture
def generate_signal_data():
    """Generate test signal data."""
    timestamps = np.linspace(0, 100, 1000)  # 100 seconds of data at 10 Hz
    signal = np.sin(2 * np.pi * 0.1 * timestamps)  # A sine wave
    return pd.DataFrame({"time": timestamps, "signal": signal})


def test_split_segment_time_based(generate_signal_data):
    """Test splitting signal by time with no overlap."""
    signal = generate_signal_data
    segments, milestones = split_segment(
        signal, sampling_rate=10, duration=10, split_type=0
    )

    assert len(segments) == 10  # 10 segments of 10 seconds each
    assert milestones.shape == (10, 2)  # Start and end indices for each segment


def test_split_segment_time_based_with_overlap(generate_signal_data):
    """Test splitting signal by time with overlap."""
    signal = generate_signal_data
    segments, milestones = split_segment(
        signal, sampling_rate=10, duration=10, overlapping=5, split_type=0
    )

    assert len(segments) == 20  # Overlap reduces step size
    assert milestones.iloc[0, 0] == 0
    assert milestones.iloc[1, 0] == 50  # Next segment starts at 50% overlap


def test_split_segment_beat_based():
    """Test splitting signal by beats."""
    test_data_path = "tests/test_data/ppg_smartcare.csv"
    assert os.path.exists(test_data_path), f"Test data file not found: {test_data_path}"
    signal = pd.read_csv(test_data_path, usecols=["PLETH"])
    print(signal)
    segments, milestones = split_segment(
        signal, sampling_rate=100, duration=30, split_type=1, wave_type="PPG"
    )

    assert len(segments) == 111
    assert milestones.shape == (111, 2)  # Start and end indices for each segment


def test_split_segment_invalid_input(generate_signal_data):
    """Test invalid input for split_segment."""
    signal = generate_signal_data
    with pytest.raises(AssertionError):
        split_segment(signal, sampling_rate="invalid", duration=10)


def test_save_segment(generate_signal_data):
    """Test saving segments to CSV and images."""
    signal = generate_signal_data
    segments, _ = split_segment(signal, sampling_rate=10, duration=10, split_type=0)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_segment(
            segments,
            save_file_folder=temp_dir,
            save_image=False,
            save_img_folder=temp_dir,
        )

        # Check saved files
        saved_csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
        # saved_img_files = [f for f in os.listdir(temp_dir) if f.endswith(".png")]

        assert len(saved_csv_files) == len(segments)  # All segments saved as CSV
        # assert len(saved_img_files) == len(segments)  # All segments saved as PNG


def test_save_segment_invalid_input():
    """Test invalid input for save_segment."""
    with pytest.raises(AssertionError):
        save_segment(segment_list="invalid")


def test_split_segment_edge_case_empty_signal():
    """Test splitting an empty signal."""
    empty_signal = pd.DataFrame(columns=["time", "signal"])
    with pytest.raises(ValueError):
        split_segment(empty_signal, sampling_rate=10, duration=10)


def test_split_segment_edge_case_short_signal(generate_signal_data):
    """Test splitting a signal shorter than the segment duration."""
    signal = generate_signal_data.iloc[:50]  # Short signal with only 50 samples
    segments, milestones = split_segment(signal, sampling_rate=10, duration=10)

    assert len(segments) == 1  # Only one segment is possible
    assert milestones.iloc[0, 1] == len(signal)  # End index matches signal length


def test_split_segment_with_invalid_peak_detector(generate_signal_data):
    """Test split_segment with an invalid peak detector."""
    signal = generate_signal_data
    with pytest.raises(AssertionError):
        split_segment(signal, sampling_rate=10, duration=10, peak_detector=99)
