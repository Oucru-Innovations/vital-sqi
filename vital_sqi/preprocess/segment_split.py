"""Splitting long recordings into segments with optional overlapping:
- By duration
- By beat
"""

import warnings
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
import os
from vital_sqi.common.utils import cut_segment, check_signal_format
from vital_sqi.common.rpeak_detection import PeakDetector
import logging


def save_segment(
    segment_list,
    segment_name="segment",
    save_file_folder=None,
    save_image=False,
    save_img_folder=None,
):
    """
    Saves segments of waveforms to .csv files and optionally plots them to image files.

    Parameters
    ----------
    segment_list : list
        List of segments (arrays or DataFrames).
    segment_name : str, optional
        Base filename for saved files (default is "segment").
    save_file_folder : str, optional
        Directory to save .csv files (default is current working directory).
    save_image : bool, optional
        If True, saves images of each segment (default is False).
    save_img_folder : str, optional
        Directory to save image files (default is current working directory).

    Returns
    -------
    None
    """
    assert isinstance(
        segment_list, (list, np.ndarray, pd.Series)
    ), "Expected a list-liked type of signal segments."

    save_file_folder = save_file_folder or os.getcwd()
    save_img_folder = save_img_folder or os.getcwd()
    num_segments = len(segment_list)
    extension_len = len(str(num_segments))

    for i, segment in enumerate(tqdm(segment_list, desc="Saving segments"), start=1):
        filename_suffix = str(i).zfill(extension_len)
        saved_filename = f"{segment_name}-{filename_suffix}"

        try:
            if save_image:
                fig = go.Figure(
                    go.Scatter(x=np.arange(len(segment)), y=segment, mode="lines")
                )
                fig.update_layout(autosize=True)
                fig.write_image(os.path.join(save_img_folder, f"{saved_filename}.png"))

            if type(segment) is np.ndarray:
                np.savetxt(
                    os.path.join(save_file_folder, f"{saved_filename}.csv"),
                    segment,
                    delimiter=",",
                )
            else:
                segment.to_csv(os.path.join(save_file_folder, f"{saved_filename}.csv"))
        except Exception as e:
            # print(f"Failed to save segment {i} due to: {e}")
            logging.error(f"Failed to save segment {i} due to: {e}")


def split_segment(
    s,
    sampling_rate,
    split_type=0,
    duration=30.0,
    overlapping=0,
    peak_detector=6,
    wave_type="PPG",
):
    """
    Splits a long signal into segments based on time or beat, with optional overlap.

    Parameters
    ----------
    s : pd.DataFrame
        Signal data with timestamps as the first column and signal values as the second.
    sampling_rate : float or int
        Sampling rate of the signal.
    split_type : int, optional
        0: split by time; 1: split by beat (default is 0).
    duration : float, optional
        Segment length in seconds (if split_type=0) or in beats (if split_type=1, default is 30).
    overlapping : float or int, optional
        Overlap in seconds (if split_type=0) or in beats (if split_type=1, default is 0).
    peak_detector : int, optional
        Type of peak detector for beat-based segmentation (default is 6 - vitalDSP method).
    wave_type : str, optional
        Type of signal, either 'PPG' or 'ECG' (default is 'PPG').

    Returns
    -------
    segments : list
        List of segmented DataFrames.
    milestones : pd.DataFrame
        DataFrame containing start and end indices of each segment.

    Examples
    --------
    >>> from vital_sqi.common.utils import generate_timestamp
    >>> s = np.arange(100000)
    >>> timestamps = generate_timestamp(None, 100, len(s))
    >>> df = pd.DataFrame({'time': timestamps, 'signal': s})
    >>> segments, milestones = split_segment(df, sampling_rate=100, duration=5)
    """
    # Ensure input is not empty
    if s is None or len(s) == 0:
        raise ValueError("Input signal is empty or None. Cannot perform segmentation.")

    check_signal_format(s)
    sampling_rate = sampling_rate or 100  # Default sampling rate if None
    overlapping = overlapping or 0  # Default to no overlap if None

    assert isinstance(sampling_rate, (int, float)), "Expected a numeric sampling rate."
    assert split_type in [0, 1], "Expected split_type to be 0 (time) or 1 (beat)."
    assert isinstance(duration, (int, float)), "Expected duration to be numeric."
    assert isinstance(overlapping, (int, float)), "Expected overlapping to be numeric."
    assert (
        isinstance(peak_detector, int) and 0 <= peak_detector <= 7
    ), "Invalid peak_detector."
    assert wave_type in ["PPG", "ECG"], "Expected wave_type to be 'PPG' or 'ECG'."

    if split_type == 0:
        chunk_size = int(duration * sampling_rate)
        chunk_step = chunk_size - int(overlapping * sampling_rate)
        chunk_indices = [
            [i, min(i + chunk_size, len(s))] for i in range(0, len(s), chunk_step)
        ]
    else:
        numeric_columns = s.select_dtypes(include=["float", "int"]).columns
        if numeric_columns.empty:
            raise ValueError("No column with numeric type found in the DataFrame.")
        sig = np.array(s[numeric_columns[0]])
        detector = PeakDetector(wave_type=wave_type)
        if wave_type == "PPG":
            _, chunk_indices = detector.ppg_detector(
                np.array(sig), detector_type=peak_detector, get_session=True
            )
        else:
            _, chunk_indices = detector.ecg_detector(np.array(sig), get_session=True)
    # Handle case when chunk_indices is empty
    if not chunk_indices:
        warnings.warn("No segments could be created; returning empty lists.")
        return [], pd.DataFrame(columns=["start", "end"])

    milestones = pd.DataFrame(chunk_indices, columns=["start", "end"])
    segments = cut_segment(s, milestones)
    return segments, milestones
