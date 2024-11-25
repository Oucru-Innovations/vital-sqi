"""Heart rate variability SQIs for time and frequency domain analysis."""

import numpy as np
from vital_sqi.common.power_spectrum import calculate_psd
import warnings
from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_nn_intervals,
    get_csi_cvi_features,
    get_geometrical_features,
)
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import HiddenPrints


def nn_mean_sqi(nn_intervals):
    """
    Calculates the mean of NN intervals.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        The mean of NN intervals or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> nn_mean_sqi(nn_intervals)
    815.0
    """
    try:
        if not nn_intervals:
            warnings.warn("Empty NN intervals provided.")
            return np.nan
        return np.mean(nn_intervals)
    except Exception as e:
        warnings.warn(f"Error in nn_mean_sqi: {e}")
        return np.nan


def sdnn_sqi(nn_intervals):
    """
    Calculates the standard deviation of NN intervals.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        The standard deviation of NN intervals or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> sdnn_sqi(nn_intervals)
    12.47
    """
    try:
        if not nn_intervals:
            warnings.warn("Empty NN intervals provided.")
            return np.nan
        return np.std(nn_intervals, ddof=1)
    except Exception as e:
        warnings.warn(f"Error in sdnn_sqi: {e}")
        return np.nan


def sdsd_sqi(nn_intervals):
    """
    Calculates the standard deviation of successive NN interval differences.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        The standard deviation of successive NN differences or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> sdsd_sqi(nn_intervals)
    10.0
    """
    try:
        if len(nn_intervals) < 2:
            warnings.warn("Insufficient NN intervals for SDSD calculation.")
            return np.nan
        return np.std(np.diff(nn_intervals))
    except Exception as e:
        warnings.warn(f"Error in sdsd_sqi: {e}")
        return np.nan


def rmssd_sqi(nn_intervals):
    """
    Calculates the root mean square of successive NN interval differences.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        The RMSSD value or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> rmssd_sqi(nn_intervals)
    10.0
    """
    try:
        if len(nn_intervals) < 2:
            warnings.warn("Insufficient NN intervals for RMSSD calculation.")
            return np.nan
        return np.sqrt(np.mean(np.diff(nn_intervals) ** 2))
    except Exception as e:
        warnings.warn(f"Error in rmssd_sqi: {e}")
        return np.nan


def cvsd_sqi(nn_intervals):
    """
    Calculates the coefficient of variation for successive differences.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        Coefficient of variation for successive differences or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> cvsd_sqi(nn_intervals)
    0.0123
    """
    try:
        mean_nn = nn_mean_sqi(nn_intervals)
        rmssd = rmssd_sqi(nn_intervals)
        return rmssd / mean_nn if mean_nn else np.nan
    except Exception as e:
        warnings.warn(f"Error in cvsd_sqi: {e}")
        return np.nan


def cvnn_sqi(nn_intervals):
    """
    Calculates the coefficient of variation for NN intervals.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        Coefficient of variation for NN intervals or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> cvnn_sqi(nn_intervals)
    0.015
    """
    try:
        mean_nn = nn_mean_sqi(nn_intervals)
        sdnn = sdnn_sqi(nn_intervals)
        return sdnn / mean_nn if mean_nn else np.nan
    except Exception as e:
        warnings.warn(f"Error in cvnn_sqi: {e}")
        return np.nan


def mean_nn_sqi(nn_intervals):
    """Calculates the mean of NN intervals."""
    return nn_mean_sqi(nn_intervals)


def median_nn_sqi(nn_intervals):
    """
    Calculates the median of NN intervals.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    float
        The median of NN intervals or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> median_nn_sqi(nn_intervals)
    815.0
    """
    try:
        if not nn_intervals:
            warnings.warn("Empty NN intervals provided.")
            return np.nan
        return np.median(nn_intervals)
    except Exception as e:
        warnings.warn(f"Error in median_nn_sqi: {e}")
        return np.nan


def pnn_sqi(nn_intervals, threshold=50):
    """
    Calculates the percentage of NN intervals that exceed a given threshold (e.g., pNN50).

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.
    threshold : float
        Threshold in milliseconds (default is 50ms).

    Returns:
    -------
    float
        Percentage of NN intervals exceeding the threshold or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> pnn_sqi(nn_intervals, threshold=50)
    0.0
    """
    try:
        if len(nn_intervals) < 2:
            warnings.warn("Insufficient NN intervals for pNN calculation.")
            return np.nan
        differences = np.abs(np.diff(nn_intervals))
        count_exceeds = np.sum(differences >= threshold)
        return (count_exceeds / len(differences)) * 100
    except Exception as e:
        warnings.warn(f"Error in pnn_sqi: {e}")
        return np.nan


def hr_sqi(nn_intervals, stat="mean"):
    """
    Generalized function to calculate heart rate (HR) statistics.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.
    stat : str
        The statistic to compute: 'mean', 'median', 'min', 'max', 'std'.

    Returns:
    -------
    float
        The computed HR statistic or NaN if input is invalid.

    Raises:
    ------
    ValueError
        If the `stat` parameter is not one of the allowed values.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> hr_sqi(nn_intervals, stat="mean")
    74.07
    >>> hr_sqi(nn_intervals, stat="min")
    72.29
    >>> hr_sqi(nn_intervals, stat="std")
    0.86
    """
    try:
        if stat not in ["mean", "median", "min", "max", "std"]:
            raise ValueError(
                "Invalid statistic requested: choose from 'mean', 'median', 'min', 'max', or 'std'."
            )

        if not nn_intervals:
            warnings.warn("Empty NN intervals provided.")
            return np.nan

        hr_values = 60000 / np.array(nn_intervals)

        if stat == "mean":
            return np.mean(hr_values)
        elif stat == "median":
            return np.median(hr_values)
        elif stat == "min":
            return np.min(hr_values)
        elif stat == "max":
            return np.max(hr_values)
        elif stat == "std":
            return np.std(hr_values)
    except Exception as e:
        warnings.warn(f"Error in hr_sqi: {e}")
        return np.nan


def hr_range_sqi(nn_intervals, range_min=40, range_max=200):
    """
    Calculates the percentage of HR values falling outside a specified range.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.
    range_min : float
        Minimum acceptable HR value (default is 40).
    range_max : float
        Maximum acceptable HR value (default is 200).

    Returns:
    -------
    float
        Percentage of HR values outside the specified range or NaN if input is invalid.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830]
    >>> hr_range_sqi(nn_intervals, range_min=50, range_max=150)
    0.0
    """
    try:
        if not nn_intervals:
            warnings.warn("Empty NN intervals provided.")
            return np.nan
        hr_values = 60000 / np.array(nn_intervals)
        out_of_range_count = np.sum((hr_values < range_min) | (hr_values > range_max))
        return (out_of_range_count / len(hr_values)) * 100
    except Exception as e:
        warnings.warn(f"Error in hr_range_sqi: {e}")
        return np.nan


def frequency_sqi(nn_intervals, freq_min=0.04, freq_max=0.15, metric="peak"):
    """Calculates frequency domain features in a specified frequency band."""
    # Validate metric first
    valid_metrics = ["peak", "absolute", "log", "normalized", "relative"]
    if metric not in valid_metrics:
        raise ValueError(
            "Invalid metric requested: choose from 'peak', \
            'absolute', 'log', 'normalized', or 'relative'."
        )

    if not isinstance(nn_intervals, (list, np.ndarray)) or len(nn_intervals) < 3:
        warnings.warn("Insufficient NN intervals for frequency analysis.")
        return np.nan

    try:
        freqs, powers = calculate_psd(nn_intervals)
        if len(freqs) == 0 or len(powers) == 0:
            raise ValueError("PSD calculation failed; insufficient data points.")
    except Exception as e:
        warnings.warn(f"Error during PSD calculation: {e}")
        return np.nan

    band_powers = powers[(freqs >= freq_min) & (freqs < freq_max)]

    if metric == "peak":
        return freqs[np.argmax(band_powers)] if band_powers.size > 0 else np.nan
    elif metric == "absolute":
        return np.sum(band_powers)
    elif metric == "log":
        return np.sum(np.log(band_powers + 1e-10))
    elif metric == "normalized":
        return np.sum(np.linalg.norm(band_powers + 1e-10))
    elif metric == "relative":
        total_power = np.sum(powers)
        return np.sum(band_powers) / total_power if total_power > 0 else np.nan


def lf_hf_ratio_sqi(nn_intervals, lf_range=(0.04, 0.15), hf_range=(0.15, 0.4)):
    """Calculates the LF/HF power ratio in frequency domain."""
    if not isinstance(nn_intervals, (list, np.ndarray)):
        warnings.warn("Invalid input: nn_intervals must be a list or numpy array.")
        return np.nan

    if not nn_intervals or len(nn_intervals) < 3:
        warnings.warn("Insufficient NN intervals for LF/HF ratio calculation.")
        return np.nan

    try:
        freqs, powers = calculate_psd(nn_intervals)
        if len(freqs) == 0 or len(powers) == 0:
            raise ValueError("PSD calculation failed; insufficient data points.")
    except Exception as e:
        warnings.warn(f"Error during PSD calculation: {e}")
        return np.nan

    lf_power = np.sum(powers[(freqs >= lf_range[0]) & (freqs < lf_range[1])])
    hf_power = np.sum(powers[(freqs >= hf_range[0]) & (freqs < hf_range[1])])

    if hf_power <= 0:
        warnings.warn("HF power is zero or negative, cannot compute LF/HF ratio.")
        return np.nan

    return lf_power / hf_power


def poincare_features_sqi(nn_intervals):
    """
    Calculates Poincare features: SD1, SD2, area, and SD1/SD2 ratio.

    Parameters:
    ----------
    nn_intervals : list or np.ndarray
        List of NN intervals in milliseconds.

    Returns:
    -------
    dict
        A dictionary containing the following keys:
        - "sd1": Standard deviation perpendicular to the line of identity.
        - "sd2": Standard deviation along the line of identity.
        - "area": Area of the ellipse formed by SD1 and SD2.
        - "ratio": Ratio of SD1 to SD2.

    Example:
    -------
    >>> nn_intervals = [800, 810, 820, 830, 800, 790]
    >>> poincare_features_sqi(nn_intervals)
    {'sd1': 7.07, 'sd2': 14.14, 'area': 312.21, 'ratio': 0.5}

    Notes:
    ------
    - SD1 and SD2 are derived from the Poincare plot of NN intervals.
    - Area represents the ellipse formed by SD1 and SD2.
    - Ratio is SD1 divided by SD2 and provides insights into heart rate variability.

    Warnings:
    ---------
    Insufficient NN intervals will result in NaN values for all metrics.
    """
    try:
        if len(nn_intervals) < 2:
            warnings.warn("Insufficient NN intervals for Poincare analysis.")
            return {"sd1": np.nan, "sd2": np.nan, "area": np.nan, "ratio": np.nan}

        differences = np.diff(nn_intervals)
        sd1 = np.sqrt(np.std(differences, ddof=1) ** 2 / 2)
        sd2 = np.sqrt(2 * np.std(nn_intervals, ddof=1) ** 2 - sd1**2)
        area = np.pi * sd1 * sd2
        ratio = sd1 / sd2 if sd2 != 0 else np.nan

        return {"sd1": sd1, "sd2": sd2, "area": area, "ratio": ratio}
    except Exception as e:
        warnings.warn(f"Error in poincare_features_sqi: {e}")
        return {"sd1": np.nan, "sd2": np.nan, "area": np.nan, "ratio": np.nan}


def get_all_features_hrva(signal, sample_rate=100, rpeak_method=6, wave_type="ECG"):
    """Extracts HRV features using peak detection and returns a comprehensive set of metrics."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")
    detector = PeakDetector(wave_type=wave_type)
    try:
        peak_list = (
            detector.ppg_detector(signal, detector_type=rpeak_method)
            if wave_type == "PPG"
            else detector.ecg_detector(signal)
        )
    except Exception as e:
        warnings.warn(f"Error during peak detection: {e}")
        return {}, {}, {}, {}

    if not isinstance(peak_list, (list, np.ndarray)) or len(peak_list) < 2:
        warnings.warn(
            "Peak Detector cannot find sufficient peaks or returned invalid data."
        )
        return {}, {}, {}, {}

    try:
        rr_intervals = np.diff(peak_list) * (1000 / sample_rate)
    except Exception as e:
        warnings.warn(f"Error during RR interval computation: {e}")
        return {}, {}, {}, {}

    try:
        with HiddenPrints():
            nn_intervals = get_nn_intervals(rr_intervals)
            time_features = get_time_domain_features(nn_intervals)
            freq_features = get_frequency_domain_features(nn_intervals)
            geometric_features = get_geometrical_features(nn_intervals)
            csi_cvi_features = get_csi_cvi_features(nn_intervals)
    except Exception as e:
        warnings.warn(f"Error during HRV feature extraction: {e}")
        return {}, {}, {}, {}

    return time_features, freq_features, geometric_features, csi_cvi_features
