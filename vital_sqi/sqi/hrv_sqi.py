"""Heart rate variability SQIs for time and frequency domain analysis."""

import numpy as np
from vital_sqi.common.power_spectrum import calculate_psd
import warnings
from vitalDSP.physiological_features.hrv_analysis import HRVFeatures

# from hrvanalysis import (
#     get_time_domain_features,
#     get_frequency_domain_features,
#     get_nn_intervals,
#     get_csi_cvi_features,
#     get_geometrical_features,
# )
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
        if len(nn_intervals) == 0:
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
        if len(nn_intervals) == 0:
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
        return np.std(np.diff(nn_intervals), ddof=1)
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
        if len(nn_intervals) == 0:
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
        count_exceeds = np.sum(differences > threshold)
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

        if len(nn_intervals) == 0:
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
            return np.std(hr_values, ddof=1)
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
        if len(nn_intervals) == 0:
            warnings.warn("Empty NN intervals provided.")
            return np.nan
        hr_values = 60000 / np.array(nn_intervals)
        out_of_range_count = np.sum((hr_values < range_min) | (hr_values > range_max))
        return (out_of_range_count / len(hr_values)) * 100
    except Exception as e:
        warnings.warn(f"Error in hr_range_sqi: {e}")
        return np.nan


def frequency_sqi(nn_intervals, freq_min=0.04, freq_max=0.15, metric="peak"):
    """
    Compute a frequency-domain feature in a specified band.

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds. Requires at least 3 values.
    freq_min : float, optional
        Lower bound of the frequency band in Hz (default ``0.04``).
    freq_max : float, optional
        Upper bound of the frequency band in Hz (default ``0.15``).
    metric : str, optional
        Feature to extract.  One of:

        * ``'peak'`` — frequency of the spectral peak in the band.
        * ``'absolute'`` — sum of power in the band.
        * ``'log'`` — sum of log power in the band.
        * ``'normalized'`` — L2 norm of power in the band.
        * ``'relative'`` — band power / total power.

        Default is ``'peak'``.

    Returns
    -------
    float
        The requested feature, or ``np.nan`` on invalid input or error.

    Raises
    ------
    ValueError
        If *metric* is not one of the accepted strings.
    """
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

    mask = (freqs >= freq_min) & (freqs < freq_max)
    band_freqs = freqs[mask]
    band_powers = powers[mask]

    if metric == "peak":
        return band_freqs[np.argmax(band_powers)] if band_powers.size > 0 else np.nan
    elif metric == "absolute":
        return np.sum(band_powers)
    elif metric == "log":
        return np.sum(np.log(band_powers + 1e-10))
    elif metric == "normalized":
        return np.linalg.norm(band_powers + 1e-10)
    elif metric == "relative":
        total_power = np.sum(powers)
        return np.sum(band_powers) / total_power if total_power > 0 else np.nan


def lf_hf_ratio_sqi(nn_intervals, lf_range=(0.04, 0.15), hf_range=(0.15, 0.4)):
    """
    Compute the LF/HF power ratio from NN intervals.

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds.  Requires at least 3 values.
    lf_range : tuple of float, optional
        Low-frequency band ``(min_Hz, max_Hz)`` (default ``(0.04, 0.15)``).
    hf_range : tuple of float, optional
        High-frequency band ``(min_Hz, max_Hz)`` (default ``(0.15, 0.4)``).

    Returns
    -------
    float
        LF power divided by HF power, or ``np.nan`` if HF power is zero or
        insufficient data are provided.
    """
    if not isinstance(nn_intervals, (list, np.ndarray)):
        warnings.warn("Invalid input: nn_intervals must be a list or numpy array.")
        return np.nan

    if len(nn_intervals) < 3:
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
    """
    Extract a comprehensive set of HRV features from a raw waveform.

    Parameters
    ----------
    signal : array-like
        Raw PPG or ECG signal.
    sample_rate : int, optional
        Sampling frequency in Hz (default ``100``).
    rpeak_method : int, optional
        Peak detector index (0–7) passed to
        :class:`~vital_sqi.common.rpeak_detection.PeakDetector` (default ``6``).
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'`` (default ``'ECG'``).  Controls which detector
        branch is used.

    Returns
    -------
    dict
        HRV features as returned by
        :meth:`vitalDSP.physiological_features.hrv_analysis.HRVFeatures.compute_all_features`.
        Returns an empty dict on detection failure or insufficient peaks.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive number.")
    detector = PeakDetector(wave_type=wave_type)
    try:
        peak_list = (
            detector.ppg_detector(signal, detector_type=rpeak_method)
            if wave_type == "PPG"
            else detector.ecg_detector(signal)
        )
        if isinstance(peak_list, tuple) and peak_list:
            peak_list = peak_list[0]
    except Exception as e:
        warnings.warn(f"Error during peak detection: {e}")
        return {}

    if not isinstance(peak_list, (list, np.ndarray)) or len(peak_list) < 2:
        warnings.warn(
            "Peak Detector cannot find sufficient peaks or returned invalid data."
        )
        return {}

    try:
        rr_intervals = np.diff(peak_list) * (1000 / sample_rate)
    except Exception as e:
        warnings.warn(f"Error during RR interval computation: {e}")
        return {}

    try:
        hrv = HRVFeatures(rr_intervals, signal, sample_rate)
        return hrv.compute_all_features()
    except Exception as e:
        warnings.warn(f"Error during HRV feature extraction: {e}")
        return {}

    # try:
    #     with HiddenPrints():
    #         nn_intervals = get_nn_intervals(rr_intervals)
    #         time_features = get_time_domain_features(nn_intervals)
    #         freq_features = get_frequency_domain_features(nn_intervals)
    #         geometric_features = get_geometrical_features(nn_intervals)
    #         csi_cvi_features = get_csi_cvi_features(nn_intervals)
    # except Exception as e:
    #     warnings.warn(f"Error during HRV feature extraction: {e}")
    #     return {}, {}, {}, {}

    # return time_features, freq_features, geometric_features, csi_cvi_features


def rr_irregularity_sqi(nn_intervals):
    """
    Measure beat-to-beat RR interval irregularity.

    Computes the mean absolute deviation of successive RR differences
    normalised by the median RR interval.  More sensitive to ectopic beats
    and atrial fibrillation than plain SDNN.  A clean, regular signal has
    a value close to 0; a highly irregular signal approaches or exceeds 1.

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds.

    Returns
    -------
    float
        ``mean(|diff(nn)|) / median(nn)``, or ``np.nan`` on invalid input.
    """
    try:
        nn = np.asarray(nn_intervals, dtype=float)
        if len(nn) < 2:
            warnings.warn("Insufficient NN intervals for rr_irregularity_sqi.")
            return np.nan
        med = np.median(nn)
        if med == 0:
            return np.nan
        return float(np.mean(np.abs(np.diff(nn))) / med)
    except Exception as e:
        warnings.warn(f"Error in rr_irregularity_sqi: {e}")
        return np.nan


def sample_entropy_sqi(nn_intervals, m=2, r=None):
    """
    Compute Sample Entropy (SampEn) of the NN interval series.

    Sample entropy measures the regularity and complexity of a time series.
    Lower values indicate a more regular, predictable signal (good quality).
    Higher values indicate more randomness or noise.

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds.  Requires at least ``2*m + 2`` values.
    m : int, optional
        Template length (embedding dimension), default ``2``.
    r : float, optional
        Tolerance (similarity threshold).  Defaults to
        ``0.2 * std(nn_intervals)`` per Richman & Moorman (2000).

    Returns
    -------
    float
        Sample entropy value >= 0, or ``np.nan`` on invalid input or when
        no template matches are found.

    References
    ----------
    Richman J.S. & Moorman J.R. (2000). Am J Physiol Heart Circ Physiol.
    """
    try:
        nn = np.asarray(nn_intervals, dtype=float)
        n = len(nn)
        if n < 2 * m + 2:
            warnings.warn("Insufficient NN intervals for sample_entropy_sqi.")
            return np.nan
        if r is None:
            r = 0.2 * np.std(nn, ddof=1)
        if r <= 0:
            return np.nan

        def _count_matches(seq, length, tol):
            count = 0
            for i in range(len(seq) - length):
                template = seq[i: i + length]
                for j in range(i + 1, len(seq) - length):
                    if np.max(np.abs(seq[j: j + length] - template)) < tol:
                        count += 1
            return count

        A = _count_matches(nn, m + 1, r)
        B = _count_matches(nn, m, r)
        if B == 0:
            return np.nan
        return float(-np.log(A / B))
    except Exception as e:
        warnings.warn(f"Error in sample_entropy_sqi: {e}")
        return np.nan


def dfa_sqi(nn_intervals, scale_min=4, scale_max=None, n_scales=10):
    """
    Detrended Fluctuation Analysis (DFA) short-range scaling exponent (alpha1).

    DFA alpha1 quantifies short-range (4-16 beat) fractal correlations in the
    RR interval series.  Healthy sinus rhythm gives alpha1 ~ 1.0-1.2.
    White noise gives alpha1 ~ 0.5.  Values near 0.5 indicate disorganised,
    noisy intervals (poor signal or AF).

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds.  Requires at least 32 values for a
        reliable estimate.
    scale_min : int, optional
        Minimum DFA window size in beats (default ``4``).
    scale_max : int, optional
        Maximum DFA window size in beats.  Defaults to ``len(nn) // 4``.
    n_scales : int, optional
        Number of logarithmically spaced window sizes (default ``10``).

    Returns
    -------
    float
        DFA alpha1 scaling exponent, or ``np.nan`` on insufficient data.

    References
    ----------
    Peng C.K. et al. (1995). Chaos 5(1):82-87.
    """
    try:
        nn = np.asarray(nn_intervals, dtype=float)
        n = len(nn)
        if n < 16:
            warnings.warn("Insufficient NN intervals for dfa_sqi (need >= 16).")
            return np.nan

        if scale_max is None:
            scale_max = max(n // 4, scale_min + 1)
        scale_max = min(scale_max, n // 2)
        if scale_max <= scale_min:
            return np.nan

        scales = np.unique(
            np.round(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)).astype(int)
        )
        scales = scales[scales >= scale_min]

        profile = np.cumsum(nn - np.mean(nn))
        fluctuations = []
        for s in scales:
            n_blocks = n // s
            if n_blocks < 2:
                continue
            blocks = profile[: n_blocks * s].reshape(n_blocks, s)
            x = np.arange(s)
            rms = []
            for block in blocks:
                coef = np.polyfit(x, block, 1)
                trend = np.polyval(coef, x)
                rms.append(np.sqrt(np.mean((block - trend) ** 2)))
            fluctuations.append((s, np.mean(rms)))

        if len(fluctuations) < 2:
            return np.nan
        log_s = np.log10([f[0] for f in fluctuations])
        log_f = np.log10([f[1] for f in fluctuations])
        alpha, _ = np.polyfit(log_s, log_f, 1)
        return float(alpha)
    except Exception as e:
        warnings.warn(f"Error in dfa_sqi: {e}")
        return np.nan


def hurst_sqi(nn_intervals):
    """
    Estimate the Hurst exponent of the NN interval series via R/S analysis.

    The Hurst exponent (H) quantifies long-range correlations:

    * H > 0.5 — persistent, structured series (healthy sinus rhythm)
    * H ~ 0.5 — uncorrelated (white noise, artifact)
    * H < 0.5 — anti-persistent series

    Parameters
    ----------
    nn_intervals : list or np.ndarray
        NN intervals in milliseconds.  Requires at least 20 values.

    Returns
    -------
    float
        Hurst exponent estimate in ``[0, 1]``, or ``np.nan`` on error.
    """
    try:
        nn = np.asarray(nn_intervals, dtype=float)
        n = len(nn)
        if n < 20:
            warnings.warn("Insufficient NN intervals for hurst_sqi (need >= 20).")
            return np.nan

        # R/S analysis across multiple sub-series lengths
        lags = np.unique(
            np.round(np.logspace(np.log2(4), np.log2(n // 2), 8, base=2)).astype(int)
        )
        lags = lags[(lags >= 4) & (lags <= n // 2)]
        if len(lags) < 2:
            return np.nan

        rs_values = []
        for lag in lags:
            rs_per_lag = []
            for start in range(0, n - lag, lag):
                sub = nn[start: start + lag]
                mean_sub = np.mean(sub)
                devs = np.cumsum(sub - mean_sub)
                r = np.max(devs) - np.min(devs)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_per_lag.append(r / s)
            if rs_per_lag:
                rs_values.append((lag, np.mean(rs_per_lag)))

        if len(rs_values) < 2:
            return np.nan
        log_n = np.log10([v[0] for v in rs_values])
        log_rs = np.log10([v[1] for v in rs_values])
        H, _ = np.polyfit(log_n, log_rs, 1)
        return float(np.clip(H, 0.0, 1.0))
    except Exception as e:
        warnings.warn(f"Error in hurst_sqi: {e}")
        return np.nan


# from vitalDSP.utils.synthesize_data import generate_ecg_signal

# if __name__ == "__main__":
#     sfecg = 256
#     N = 100
#     Anoise = 0.05
#     hrmean = 70
#     ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
#     # result_correlogram = correlogram_sqi(ecg_signal, sample_rate=sfecg, wave_type="ECG")
#     # print(result_correlogram)

#     result_ectopic = get_all_features_hrva(
#         ecg_signal, sample_rate=sfecg, rpeak_method=6, wave_type="ECG"
#     )
#     print(result_ectopic)
