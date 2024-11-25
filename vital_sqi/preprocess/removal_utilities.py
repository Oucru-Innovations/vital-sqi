"""Signal Processing Utilities for Removing Noise and Interpolating Missing Data."""

import numpy as np
import pandas as pd
import warnings
import pmdarima as pm
from vital_sqi.common.utils import cut_segment, format_milestone, check_signal_format


def get_start_end_points(start_cut_pivot, end_cut_pivot, length_df):
    """
    Determines the start and end points for each retained signal segment.

    Parameters
    ----------
    start_cut_pivot : array-like
        Array of starting points of removed segments.
    end_cut_pivot : array-like
        Array of corresponding ending points of removed segments.
    length_df : int
        Length of the original signal.

    Returns
    -------
    tuple
        Arrays of start and end milestones for retained segments.
    """
    start_cut_pivot, end_cut_pivot = np.array(start_cut_pivot), np.array(end_cut_pivot)

    if 0 not in start_cut_pivot:
        start_milestone = np.hstack((0, end_cut_pivot + 1))
        end_milestone = (
            np.hstack((start_cut_pivot - 1, length_df - 1))
            if length_df - 1 not in end_cut_pivot
            else start_cut_pivot - 1
        )
    else:
        start_milestone = end_cut_pivot + 1
        end_milestone = np.hstack((start_cut_pivot[1:] - 1, length_df - 1))

    return start_milestone, end_milestone


def remove_unchanged(s, sampling_rate, duration=10, output_signal=True):
    """
    Removes flat (unchanged) segments of the signal considered as noise.

    Parameters
    ----------
    s : pd.DataFrame
        Signal with first column as pd.Timestamp and second as float.
    sampling_rate : int or float
        Sampling rate of the signal.
    duration : int or float, optional
        Duration of flat signal (default 10 seconds) to be considered as noise.
    output_signal : bool, optional
        If True, returns processed signal along with milestones.

    Returns
    -------
    tuple
        Processed signal (optional) and DataFrame of retained segment milestones.
    """
    check_signal_format(s)
    assert isinstance(sampling_rate, (int, float)), "Expected a numeric sampling rate."
    assert isinstance(
        duration, (int, float, type(None))
    ), "Expected numeric duration or None."
    assert isinstance(output_signal, bool), "Expected boolean for output_signal."

    duration = duration or 0
    num_removed_instances = sampling_rate * duration
    signal_array = s.select_dtypes(include=[np.number]).iloc[:, 0].values
    unchanged_indices = np.where(np.diff(signal_array) == 0)[0]

    if len(unchanged_indices) == 0:
        # Entire signal is valid
        milestones = pd.DataFrame({"start": [0], "end": [len(s)]})
    else:
        continuous_segments = np.split(
            unchanged_indices, np.where(np.diff(unchanged_indices) != 1)[0] + 1
        )
        start_cut_pivot, end_cut_pivot = [], []

        for segment in continuous_segments:
            if len(segment) >= num_removed_instances:
                start_cut_pivot.append(segment[0])
                end_cut_pivot.append(segment[-1] + 1)

        # Handle edge cases at start and end of the signal
        if start_cut_pivot and start_cut_pivot[0] == 0:
            start_cut_pivot.pop(0)
        if end_cut_pivot and end_cut_pivot[-1] >= len(signal_array):
            end_cut_pivot.pop(-1)

        # If no valid segments remain after filtering
        if not start_cut_pivot and not end_cut_pivot:
            milestones = pd.DataFrame(columns=["start", "end"])
            if output_signal:
                return [], milestones
            return milestones

        start_milestone, end_milestone = get_start_end_points(
            start_cut_pivot, end_cut_pivot, len(s)
        )
        milestones = format_milestone(start_milestone, end_milestone)

    if output_signal:
        processed_s = cut_segment(s, milestones)
        return processed_s, milestones
    return milestones


def remove_invalid_smartcare(s, info, output_signal=True):
    """
    Filters out invalid signal samples based on Smartcare oximeter data.

    Parameters
    ----------
    s : pd.DataFrame
        Signal with first column as pd.Timestamp and second as float.
    info : pd.DataFrame
        Info containing "SPO2_PCT", "PERFUSION_INDEX", and "PULSE_BPM" columns.
    output_signal : bool, optional
        If True, returns processed signal along with milestones.

    Returns
    -------
    tuple
        Processed signal (optional) and DataFrame of retained segment milestones.
    """
    check_signal_format(s)
    assert isinstance(info, pd.DataFrame), "Expected a pd.DataFrame for info."
    assert isinstance(output_signal, bool), "Expected boolean for output_signal."

    required_columns = {"SPO2_PCT", "PERFUSION_INDEX", "PULSE_BPM"}
    if not required_columns.issubset(info.columns):
        warnings.warn(
            "Info does not contain all Smartcare columns. Using signal alone for filtering."
        )
        valid_indices = np.where(s.iloc[:, 1] != 0)[0]
    else:
        valid_indices = np.where(
            (s.iloc[:, 1] != 0)
            & (info["SPO2_PCT"] >= 80)
            & (info["PULSE_BPM"] <= 255)
            & (info["PERFUSION_INDEX"] >= 0.1)
        )[0]

    start_milestone, end_milestone = _identify_segments(valid_indices)
    milestones = format_milestone(start_milestone, end_milestone)

    if output_signal:
        processed_s = cut_segment(s, milestones)
        return processed_s, milestones
    return milestones


def _identify_segments(indices):
    """Helper function to identify start and end milestones based on valid indices."""
    segment_diff = np.diff(indices)
    segment_breaks = np.where(segment_diff > 1)[0]
    start_milestone = [indices[0]] + list(indices[segment_breaks + 1])
    end_milestone = list(indices[segment_breaks]) + [indices[-1]]
    return start_milestone, end_milestone


def trim_signal(s, sampling_rate, duration_left=300, duration_right=300):
    """
    Trims noise from the beginning and end of the signal.

    Parameters
    ----------
    s : pd.DataFrame
        Signal with first column as pd.Timestamp and second as float.
    sampling_rate : int or float
        Sampling rate of the signal.
    duration_left : int or float, optional
        Seconds to trim from the start (default 300).
    duration_right : int or float, optional
        Seconds to trim from the end (default 300).

    Returns
    -------
    pd.DataFrame
        Trimmed signal.
    """
    check_signal_format(s)
    assert isinstance(sampling_rate, (int, float)), "Expected numeric sampling rate."
    assert isinstance(
        duration_left, (int, float, type(None))
    ), "Expected numeric duration or None."
    assert isinstance(
        duration_right, (int, float, type(None))
    ), "Expected numeric duration or None."

    duration_left, duration_right = duration_left or 0, duration_right or 0
    trim_length = int((duration_left + duration_right) * sampling_rate * 2)

    if trim_length > len(s):
        warnings.warn(
            "Trimming length exceeds signal length. Returning original signal."
        )
        return s

    return s.iloc[
        int(duration_left * sampling_rate) : -int(duration_right * sampling_rate)
    ]


def interpolate_signal(s, missing_index, missing_len, method="arima", lag_ratio=10):
    """
    Interpolates missing signal segments using ARIMA.

    Parameters
    ----------
    s : pd.DataFrame
        Signal with first column as pd.Timestamp and second as float.
    missing_index : list or array-like
        Starting indices of missing segments.
    missing_len : list or array-like
        Lengths of missing segments corresponding to each starting index.
    method : str, optional
        Interpolation method (only 'arima' supported, default).
    lag_ratio : int, optional
        Multiplier for the ARIMA lag window size (default 10).

    Returns
    -------
    pd.DataFrame
        Signal with interpolated segments.
    """
    check_signal_format(s)
    assert isinstance(
        missing_index, (list, np.ndarray)
    ), "Expected list or array for missing_index."
    assert isinstance(
        missing_len, (list, np.ndarray)
    ), "Expected list or array for missing_len."
    assert method == "arima", "Only 'arima' method is currently supported."
    assert isinstance(lag_ratio, (int, float)), "Expected numeric lag_ratio."

    filled_signal = s.iloc[:, 1].values.tolist()
    for pos, num_missing in zip(missing_index, missing_len):
        seg_len = int(num_missing * lag_ratio)
        start_seg = max(0, pos - seg_len)
        ts_segment = s.iloc[start_seg:pos, 1]

        if method == "arima":
            if len(ts_segment) < 3:
                # Skip interpolation if the segment is too short for ARIMA
                warnings.warn(
                    f"Insufficient data for ARIMA interpolation at position {pos}. Filling with linear interpolation."
                )
                forecast = np.linspace(
                    filled_signal[pos - 1],
                    filled_signal[pos + num_missing],
                    num=num_missing,
                ).tolist()
            else:
                try:
                    model = pm.auto_arima(
                        ts_segment,
                        seasonal=True,
                        m=max(1, len(ts_segment) // 65),
                        max_p=3,
                        max_q=3,
                        max_d=3,
                        start_p=1,
                        start_q=1,
                        start_d=0,
                        suppress_warnings=True,
                        error_action="ignore",
                    )
                    forecast = model.predict(n_periods=num_missing).tolist()
                except Exception as e:
                    # Fallback to linear interpolation if ARIMA fails
                    warnings.warn(
                        f"ARIMA failed for position {pos} due to {e}. Using linear interpolation as fallback."
                    )
                    forecast = np.linspace(
                        filled_signal[pos - 1],
                        filled_signal[pos + num_missing],
                        num=num_missing,
                    ).tolist()

            # Replace any NaN in forecast
            forecast = [0 if np.isnan(val) else val for val in forecast]
            if len(forecast) != num_missing:
                # Ensure forecast matches the expected length
                forecast = forecast[:num_missing]

            filled_signal = (
                filled_signal[:pos] + forecast + filled_signal[pos + num_missing :]
            )

    s.iloc[:, 1] = filled_signal
    return s
