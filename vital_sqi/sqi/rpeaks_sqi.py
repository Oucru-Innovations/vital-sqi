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
from scipy import signal
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
    Compute the ratio of ectopic (invalid) beats in a signal.

    The function first removes out-of-range RR intervals (outliers), then
    applies an interpolation-based ectopic detection method to the remaining
    intervals.

    Parameters
    ----------
    s : array-like
        Raw waveform signal (PPG or ECG).
    rule_index : int, optional
        Controls which step of ectopic analysis to perform:

        * ``0`` — return the outlier ratio only (no ectopic removal)
        * ``1`` — apply *adaptive* ectopic removal after outlier filtering
        * ``2`` — apply *linear* ectopic removal after outlier filtering
        * ``3`` — apply *spline* ectopic removal after outlier filtering

        Default is ``1``.
    sample_rate : int, optional
        Sampling frequency in Hz (default ``100``).
    rpeak_detector : int, optional
        Peak detector index used by ``RRTransformation`` (default ``6``).
        This parameter is passed through but the actual detector is managed
        internally by ``vitalDSP``.
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'`` (default ``'PPG'``).
    low_rri : int, optional
        Minimum acceptable RR interval in milliseconds (default ``300``).
    high_rri : int, optional
        Maximum acceptable RR interval in milliseconds (default ``2000``).

    Returns
    -------
    float
        A ratio in ``[0, 1]`` where ``0`` means no ectopic/outlier beats and
        ``1`` means all beats are invalid.  Returns ``np.nan`` on error.
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
            rr_intervals, min_rr=low_rri / 1000, max_rr=high_rri / 1000
        )
        number_outliers = np.isnan(rr_intervals_cleaned).sum()
        total_rr_intervals = len(rr_intervals_cleaned)
        outlier_ratio = number_outliers / max(total_rr_intervals, 1)

        if rule_index == 0:
            return outlier_ratio

        # Impute invalid intervals for ectopic beat removal
        interpolated_rr_intervals = transformer.impute_rr_intervals(
            rr_intervals_cleaned
        )
        selected_rule = rules[rule_index - 1]
        nn_intervals = remove_ectopic_beats(
            interpolated_rr_intervals, method=selected_rule
        )
        number_ectopics = np.isnan(nn_intervals).sum()
        ectopic_ratio = number_ectopics / max(len(nn_intervals), 1)

        return ectopic_ratio

    except Exception as e:
        warnings.warn(f"Unexpected error in ectopic_sqi: {e}")
        return np.nan


def remove_ectopic_beats(rr_intervals, method="adaptive"):
    """
    Mark ectopic beats in an RR-interval array as NaN.

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR intervals in seconds, possibly containing NaN values from a prior
        outlier-removal step.
    method : str, optional
        Detection algorithm.  One of:

        * ``'adaptive'`` — compares each interval to a 5-point running mean;
          deviations > 20 % are flagged.
        * ``'linear'`` — fits a linear trend to valid intervals; deviations
          > 15 % are flagged.
        * ``'spline'`` — fits a smoothing spline; deviations > 10 % are
          flagged.

        Default is ``'adaptive'``.

    Returns
    -------
    np.ndarray
        Copy of *rr_intervals* with ectopic positions set to ``NaN``.
        If an error occurs the original array is returned unchanged.

    Example
    -------
    >>> import numpy as np
    >>> from vital_sqi.sqi.rpeaks_sqi import remove_ectopic_beats
    >>> rr = np.array([0.8, 1.2, 1.0, 2.5, 0.9, 0.85])
    >>> clean = remove_ectopic_beats(rr, method="adaptive")
    """
    try:
        if method not in ["adaptive", "linear", "spline"]:
            raise ValueError(
                f"Invalid method: {method}. Choose 'adaptive', 'linear', or 'spline'."
            )

        rr_intervals_cleaned = np.copy(rr_intervals)
        valid_intervals = rr_intervals_cleaned[~np.isnan(rr_intervals_cleaned)]

        if len(valid_intervals) < 3:
            raise ValueError("Not enough valid RR intervals for ectopic detection.")

        if method == "adaptive":
            # Detect ectopic beats using a running mean and threshold
            running_mean = np.convolve(valid_intervals, np.ones(5) / 5, mode="same")
            deviations = np.abs(valid_intervals - running_mean)
            threshold = 0.2 * running_mean  # 20% deviation considered ectopic
            ectopic_mask = deviations > threshold
            rr_intervals_cleaned[~np.isnan(rr_intervals_cleaned)] = np.where(
                ectopic_mask, np.nan, valid_intervals
            )

        elif method == "linear":
            # Use linear interpolation to fit a trend and remove ectopic beats
            linear_fit = np.polyval(
                np.polyfit(np.arange(len(valid_intervals)), valid_intervals, 1),
                np.arange(len(valid_intervals)),
            )
            deviations = np.abs(valid_intervals - linear_fit)
            threshold = 0.15 * linear_fit  # 15% deviation considered ectopic
            ectopic_mask = deviations > threshold
            rr_intervals_cleaned[~np.isnan(rr_intervals_cleaned)] = np.where(
                ectopic_mask, np.nan, valid_intervals
            )

        elif method == "spline":
            # Use spline fitting to detect and remove ectopic beats
            from scipy.interpolate import UnivariateSpline

            spline = UnivariateSpline(
                np.arange(len(valid_intervals)), valid_intervals, s=0.1
            )
            spline_fit = spline(np.arange(len(valid_intervals)))
            deviations = np.abs(valid_intervals - spline_fit)
            threshold = 0.1 * spline_fit  # 10% deviation considered ectopic
            ectopic_mask = deviations > threshold
            rr_intervals_cleaned[~np.isnan(rr_intervals_cleaned)] = np.where(
                ectopic_mask, np.nan, valid_intervals
            )

        return rr_intervals_cleaned

    except Exception as e:
        warnings.warn(f"Error in remove_ectopic_beats: {e}")
        return rr_intervals


def correlogram_sqi(s, sample_rate=100, wave_type="PPG", time_lag=3, n_selection=3):
    """
    Compute the Correlogram SQI from the autocorrelation function (ACF).

    The ACF is computed up to *time_lag* seconds.  The top *n_selection* ACF
    peaks are selected and their mean is returned as a single quality score.
    A high score (close to 1) indicates a highly periodic signal; a low score
    indicates poor periodicity or noise.

    Parameters
    ----------
    s : array-like
        Raw waveform signal (PPG or ECG).
    sample_rate : int, optional
        Sampling frequency in Hz (default ``100``).
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'`` (default ``'PPG'``).  Currently unused but
        retained for API consistency with other SQI functions.
    time_lag : int, optional
        Duration in seconds over which ACF is computed (default ``3``).
        The signal must be at least ``time_lag * sample_rate`` samples long.
    n_selection : int, optional
        Number of top ACF peaks to average (default ``3``).

    Returns
    -------
    float
        Mean ACF value of the top *n_selection* peaks, in ``[-1, 1]``.
        Returns ``np.nan`` if the signal is too short, no peaks are found,
        or an error occurs.
    """
    try:
        nlags = time_lag * sample_rate
        if len(s) < nlags:
            raise ValueError("Signal length is too short for the specified time lag.")

        # Compute autocorrelation
        corr = acf(s, nlags=nlags, fft=True)

        # Find peaks in the autocorrelation function
        corr_peaks_idx = signal.find_peaks(corr)[0]
        corr_peaks_value = corr[corr_peaks_idx]

        if len(corr_peaks_idx) == 0:
            warnings.warn("No peaks detected in the autocorrelation function.")
            return np.nan

        # Select top peaks based on autocorrelation values
        top_values = np.sort(corr_peaks_value)[
            -n_selection:
        ]  # Select top `n_selection` values

        # Compute SQI as the mean of the top peak values
        sqi_value = np.mean(top_values)
        return sqi_value

    except Exception as e:
        warnings.warn(f"Error in correlogram_sqi: {e}")
        return np.nan


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

    if wave_type == "ECG":
        warnings.warn(
            "msq_sqi for ECG currently uses two PPG-style detectors as a proxy. "
            "For a meaningful ECG MSQ, two distinct detectors (e.g. Pan-Tompkins "
            "and Hamilton) are needed. Returning NaN until Phase-2 ECG detectors "
            "are available.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.nan

    try:
        detector = PeakDetector(wave_type=wave_type)

        # Both wave types use ppg_detector with two different algorithm indices
        # because ecg_detector does not expose a detector_type parameter.
        peaks_1, _ = detector.ppg_detector(s, detector_type=peak_detector_1)
        peaks_2, _ = detector.ppg_detector(
            s, detector_type=peak_detector_2, preprocess=False
        )

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


def amplitude_consistency_sqi(s, sample_rate=100, wave_type="PPG", peak_detector=6):
    """
    Measure beat-to-beat amplitude variability as a signal quality indicator.

    High beat-to-beat variability in peak amplitude indicates motion artifact
    or poor electrode contact.  The metric is the coefficient of variation
    (std / mean) of detected peak amplitudes.  A clean signal has low CV
    (< 0.1); an artifact-corrupted signal has high CV (> 0.3).

    Parameters
    ----------
    s : array_like
        Raw waveform signal (PPG or ECG).
    sample_rate : int, optional
        Sampling frequency in Hz (default ``100``).
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'`` (default ``'PPG'``).
    peak_detector : int, optional
        Peak detector index (default ``6``).

    Returns
    -------
    float
        Coefficient of variation of peak amplitudes, or ``np.nan`` if fewer
        than 2 peaks are detected.
    """
    try:
        detector = PeakDetector(wave_type=wave_type)
        if wave_type == "PPG":
            peaks, _ = detector.ppg_detector(s, detector_type=peak_detector)
        else:
            peaks, _ = detector.ecg_detector(s)

        s = np.asarray(s, dtype=float)
        if len(peaks) < 2:
            warnings.warn("amplitude_consistency_sqi: fewer than 2 peaks detected.")
            return np.nan

        amplitudes = s[np.asarray(peaks, dtype=int)]
        mean_amp = np.mean(amplitudes)
        if mean_amp == 0:
            return np.nan
        return float(np.std(amplitudes, ddof=1) / np.abs(mean_amp))
    except Exception as e:
        warnings.warn(f"Error in amplitude_consistency_sqi: {e}")
        return np.nan


# from vitalDSP.utils.synthesize_data import generate_ecg_signal
# if __name__ == "__main__":
#     sfecg = 256
#     N = 100
#     Anoise = 0.05
#     hrmean = 70
#     ecg_signal = generate_ecg_signal(
#         sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean
#     )
#     # result_correlogram = correlogram_sqi(ecg_signal, sample_rate=sfecg, wave_type="ECG")
#     # print(result_correlogram)

#     result_ectopic = ectopic_sqi(ecg_signal, sample_rate=sfecg, wave_type="ECG")
#     print(result_ectopic)
