"""Signal Quality Indexes based on R peak detection.
This module includes various R peak detection-based metrics for evaluating ECG and PPG signals.
Metrics:
- Ratio of ectopic beats
- Correlogram
- MSQ: Consistency evaluation between two R-peak detectors
- Interpolation-based SQI
"""

import numpy as np
from vital_sqi.common.utils import HiddenPrints

# from hrvanalysis.preprocessing import (
#     remove_outliers,
#     remove_ectopic_beats,
#     interpolate_nan_values,
# )
import warnings
from statsmodels.tsa.stattools import acf
from vital_sqi.common.rpeak_detection import PeakDetector
from vitalDSP.transforms.beats_transformation import RRTransformation


def ectopic_sqi(
    s,
    rule_index=1,
    sample_rate=100,
    rpeak_detector=6,
    wave_type="PPG",
    low_rri=300,
    high_rri=2000,
):
    """
    Evaluate the ratio of ectopic (invalid) R-R intervals in a signal based on HRV rules.

    Parameters
    ----------
    s : array-like
        Input signal.
    rule_index : int, optional
        Index of HRV rule for ectopic beat removal. Default is 1 (Adaptive).
        1: Adaptive, 2: Linear, 3: Spline
    sample_rate : int, optional
        Sampling frequency in Hz. Default is 100.
    rpeak_detector : int, optional
        Type of R peak detector algorithm to use. Default is 6.
    wave_type : str, optional
        Type of signal (ECG or PPG). Default is "PPG".
    low_rri : int, optional
        Minimum acceptable R-R interval in ms. Default is 300.
    high_rri : int, optional
        Maximum acceptable R-R interval in ms. Default is 2000.

    Returns
    -------
    float
        Ectopic beat ratio.

    Notes
    -----
    This function computes the ectopic beat ratio using R-R intervals derived
    from peak detection. Invalid intervals are identified using the specified HRV rules.
    """
    rules = ["adaptive", "linear", "spline"]

    try:
        # Validate rule_index
        if rule_index not in range(0, len(rules) + 1):
            raise ValueError(
                f"Invalid rule_index: {rule_index}. Must be between 0 and {len(rules)}."
            )

        # Initialize RRTransformation for the signal
        with HiddenPrints():
            transformer = RRTransformation(
                signal=s, fs=sample_rate, signal_type=wave_type
            )
            rr_intervals = transformer.compute_rr_intervals()

        if len(rr_intervals) < 2:
            raise ValueError("Insufficient RR intervals for analysis.")

        # Remove outliers based on RRI bounds and calculate outlier ratio
        rr_intervals_cleaned = transformer.remove_invalid_rr_intervals(
            rr_intervals, min_rri=low_rri / 1000, max_rri=high_rri / 1000
        )
        number_outliers = np.isnan(rr_intervals_cleaned).sum()
        total_rr_intervals = len(rr_intervals_cleaned)
        outlier_ratio = number_outliers / max(total_rr_intervals - number_outliers, 1)

        if rule_index == 0:
            return outlier_ratio

        # Impute invalid intervals for ectopic beat removal
        interpolated_rr_intervals = transformer.impute_rr_intervals(
            rr_intervals_cleaned
        )
        selected_rule = rules[rule_index - 1]
        nn_intervals = transformer.remove_ectopic_beats(
            interpolated_rr_intervals, method=selected_rule
        )
        number_ectopics = np.isnan(nn_intervals).sum()
        ectopic_ratio = number_ectopics / max(len(nn_intervals) - number_ectopics, 1)

        return ectopic_ratio

    except ValueError as e:
        warnings.warn(str(e))
        return np.nan

    except Exception as e:
        warnings.warn(f"Unexpected error in ectopic_sqi: {e}")
        return np.nan

    except ValueError as e:
        warnings.warn(f"Error in ectopic_sqi: {e}")
        return np.nan

    except Exception as e:
        warnings.warn(f"Unexpected error in ectopic_sqi: {e}")
        return np.nan

    except Exception as e:
        warnings.warn(
            f"No peaks detected in the signal. RR interval computation failed: {e}"
        )
        return np.nan


def correlogram_sqi(s, sample_rate=100, wave_type="PPG", time_lag=3, n_selection=3):
    """
    Computes the Correlogram Signal Quality Index (SQI) based on signal autocorrelation.

    Parameters
    ----------
    s : array-like
        Input signal.
    sample_rate : int, optional
        Sampling frequency in Hz. Default is 100.
    wave_type : str, optional
        Type of signal (ECG or PPG). Default is "PPG".
    time_lag : int, optional
        Time lag in seconds for autocorrelation. Default is 3.
    n_selection : int, optional
        Number of peaks to select from the autocorrelation function. Default is 3.

    Returns
    -------
    list
        List of indices and values of selected peaks in the autocorrelation function,
        or an empty list if no peaks are found.
    """
    try:
        nlags = time_lag * sample_rate
        if len(s) < nlags:
            raise ValueError("Signal length is too short for the specified time lag.")

        # Compute autocorrelation
        corr = acf(s, nlags=nlags, fft=True)

        # Find peaks in the autocorrelation function
        # corr_peaks_idx = signal.find_peaks(corr)[0]
        # corr_peaks_value = corr[corr_peaks_idx]
        detector = PeakDetector(wave_type=wave_type)

        if wave_type == "PPG":
            corr_peaks_idx, _ = detector.ppg_detector(s)
        else:
            corr_peaks_idx, _ = detector.ecg_detector(s)
        corr_peaks_value = corr[corr_peaks_idx]
        if len(corr_peaks_idx) == 0:
            warnings.warn("No peaks detected in the autocorrelation function.")
            return []

        # Select top peaks
        n_selection = min(n_selection, len(corr_peaks_value))
        corr_sqi = list(corr_peaks_idx[:n_selection]) + list(
            corr_peaks_value[:n_selection]
        )
        return corr_sqi

    except Exception as e:
        warnings.warn(f"Error in correlogram_sqi: {e}")
        return []


def interpolation_sqi(s):
    """
    Interpolation-based SQI to assess consistency in R-R intervals.

    Parameters
    ----------
    s : array-like
        Input signal.

    Returns
    -------
    float
        Interpolation SQI value.
    """
    # Placeholder, no clear implementation available
    return 0.0


def msq_sqi(s, peak_detector_1=7, peak_detector_2=6, wave_type="PPG"):
    """
    Computes the Modified Signal Quality (MSQ) SQI based on agreement between two R-peak detectors.
    This SQI is used to evaluate the consistency of peaks detected by different algorithms.

    Parameters
    ----------
    s : array-like
        Input signal.
    peak_detector_1 : int, optional
        Type of the primary peak detection algorithm. Default is 7 (Billauer).
    peak_detector_2 : int, optional
        Type of the secondary peak detection algorithm. Default is 6 (Scipy).
    wave_type : str, optional
        Type of signal, either 'PPG' or 'ECG'. Default is 'PPG'.

    Returns
    -------
    float
        MSQ SQI value indicating consistency between detectors.
    """
    if not isinstance(s, (np.ndarray, list)) or len(s) == 0:
        warnings.warn("Input signal is empty or invalid.")
        return np.nan

    try:
        detector = PeakDetector(wave_type=wave_type)

        if wave_type == "PPG":
            peaks_1, _ = detector.ppg_detector(s, detector_type=peak_detector_1)
            peaks_2, _ = detector.ppg_detector(
                s, detector_type=peak_detector_2, preprocess=False
            )
        else:
            peaks_1, _ = detector.ecg_detector(s)
            peaks_2, _ = detector.ecg_detector(s)

        # Check if either detector found no peaks
        if len(peaks_1) == 0 or len(peaks_2) == 0:
            warnings.warn("No peaks detected by one or both detectors.")
            return 0.0

        # Calculate the percentage of matching peaks between the two detectors
        intersection_peaks = np.intersect1d(peaks_1, peaks_2)
        peak1_dom = len(intersection_peaks) / max(len(peaks_1), 1)
        peak2_dom = len(intersection_peaks) / max(len(peaks_2), 1)

        # Return minimum of two ratios for robustness
        return min(peak1_dom, peak2_dom)
    except Exception as e:
        warnings.warn(f"Error in msq_sqi: {e}")
        return np.nan
