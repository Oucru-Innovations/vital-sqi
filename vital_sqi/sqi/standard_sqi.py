"""
Signal Quality Index (SQI) calculations for photoplethysmogram (PPG) signals.

These SQIs are based on the paper by Elgendi, Mohamed:
"Optimal Signal Quality Index for Photoplethysmogram Signals", Bioengineering.
"""

import numpy as np
import warnings
from scipy.stats import kurtosis, skew, entropy
import scipy.signal as sn


def perfusion_sqi(x, y):
    """
    Calculates the perfusion index, a measure of pulsatile blood flow relative to static blood flow.

    Parameters
    ----------
    x : array_like
        Raw PPG signal.
    y : array_like
        Filtered PPG signal.

    Returns
    -------
    float
        Perfusion SQI, calculated as [(max(y) - min(y)) / abs(mean(x))] * 100.
    """
    if np.abs(np.mean(x)) < 1e-10:
        return np.nan
    return ((np.max(y) - np.min(y)) / np.abs(np.mean(x))) * 100


def kurtosis_sqi(x, axis=0, fisher=True, bias=True, nan_policy="propagate"):
    """
    Calculates the kurtosis of the signal, a measure of tail heaviness in a distribution.

    Parameters
    ----------
    x : array_like
        Input signal.
    axis : int, optional
        Axis along which kurtosis is calculated (default is 0).
    fisher : bool, optional
        If True, Fisher’s definition is used (default is True).
    bias : bool, optional
        If False, the calculations are corrected for statistical bias (default is True).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaNs (default is 'propagate').

    Returns
    -------
    float or ndarray
        Kurtosis value(s) of the signal.
    """
    if np.all(x == 0):
        return 0
    return kurtosis(x, axis=axis, fisher=fisher, bias=bias, nan_policy=nan_policy)


def skewness_sqi(x, axis=0, bias=True, nan_policy="propagate"):
    """
    Calculates the skewness of the signal, a measure of asymmetry in a distribution.

    Parameters
    ----------
    x : array_like
        Input signal.
    axis : int, optional
        Axis along which skewness is calculated (default is 0).
    bias : bool, optional
        If False, calculations are corrected for statistical bias (default is True).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaNs (default is 'propagate').

    Returns
    -------
    float or ndarray
        Skewness value(s) of the signal.
    """
    if np.all(x == 0):
        return 0
    return skew(x, axis=axis, bias=bias, nan_policy=nan_policy)


def entropy_sqi(x, qk=None, base=None, axis=0):
    """
    Calculates the entropy of the signal, representing the randomness in the data distribution.

    Parameters
    ----------
    x : array_like
        Input signal.
    qk : array_like, optional
        Array against which relative entropy is calculated (default is None).
    base : float, optional
        Logarithmic base to use for entropy (default is None).
    axis : int, optional
        Axis along which entropy is calculated (default is 0).

    Returns
    -------
    float or ndarray
        Entropy value(s) of the signal.
    """
    x = np.array(x, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("Input signal is empty; cannot compute entropy.")
    counts, _ = np.histogram(x, bins="auto")
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    prob_dist = counts / counts.sum()
    return entropy(prob_dist, qk=None, base=base)


def signal_to_noise_sqi(a, axis=0, ddof=0):
    """
    Calculates the Signal-to-Noise Ratio (SNR) SQI, comparing signal level to noise level.

    Parameters
    ----------
    a : array_like
        Input signal.
    axis : int, optional
        Axis along which SNR is calculated (default is 0).
    ddof : int, optional
        Delta degrees of freedom for standard deviation (default is 0).

    Returns
    -------
    float or ndarray
        SNR value(s) of the signal.
    """
    if not isinstance(a, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array.")
    a = np.asanyarray(a)
    m = np.abs(a.mean(axis=axis))  # Take absolute value of the mean
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)  # Avoid division by zero


def zero_crossings_rate_sqi(y, threshold=1e-10, ref_magnitude=None, axis=-1):
    """
    Calculates the zero-crossing rate, the rate of sign changes in the signal.

    .. note::
        This counts crossings of the **absolute zero** level. For signals with
        a non-zero DC baseline (e.g. raw PPG/ECG), the signal may never cross
        zero, making this metric meaningless. Use
        :func:`mean_crossing_rate_sqi` instead, which subtracts the mean first.

    Parameters
    ----------
    y : array_like
        Input signal. Should be mean-centred before calling this function
        for physiologically meaningful results.
    threshold : float, optional
        Threshold for clipping values close to zero (default is 1e-10).
    ref_magnitude : float, optional
        Reference magnitude for scaling threshold (default is None).
    axis : int, optional
        Axis along which zero-crossings are calculated (default is -1).

    Returns
    -------
    float
        Zero-crossing rate of the signal.
    """
    if ref_magnitude is not None:
        threshold *= (
            ref_magnitude if not callable(ref_magnitude) else ref_magnitude(np.abs(y))
        )
    y_clipped = np.where(
        np.abs(y) <= threshold, 0, y
    )  # Clip values within threshold to zero
    zero_crossings = np.diff(np.sign(y_clipped), axis=axis) != 0
    return np.mean(zero_crossings)


def mean_crossing_rate_sqi(y, threshold=1e-10, ref_magnitude=None, pad=True, axis=-1):
    """
    Calculates the mean-crossing rate, the rate at which the signal crosses its mean value.

    Parameters
    ----------
    y : array_like
        Input signal.
    threshold : float, optional
        Threshold for clipping values close to zero (default is 1e-10).
    ref_magnitude : float, optional
        Reference magnitude for scaling threshold (default is None).
    pad : bool, optional
        If True, y[0] is considered a valid crossing (default is True).
    axis : int, optional
        Axis along which mean-crossing rate is calculated (default is -1).

    Returns
    -------
    float
        Mean-crossing rate of the signal.
    """
    if not isinstance(y, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array.")
    y = np.asanyarray(y)
    mean_shifted_y = y - np.mean(y, axis=axis, keepdims=True)
    return zero_crossings_rate_sqi(
        mean_shifted_y, threshold=threshold, ref_magnitude=ref_magnitude, axis=axis
    )


def clipping_sqi(signal, eps_pct=0.001):
    """
    Estimate the fraction of samples at or near the signal's amplitude rail.

    Clipped or saturated sensors produce runs of samples stuck at the minimum
    or maximum value.  A clean signal should have fewer than ~1% clipped
    samples; heavily saturated signals can exceed 5-10%.

    Parameters
    ----------
    signal : array_like
        Input signal.
    eps_pct : float, optional
        Tolerance as a fraction of the peak-to-peak range used to define
        "near the rail" (default ``0.001``, i.e. 0.1% of range).

    Returns
    -------
    float
        Fraction of samples in ``[0, 1]`` that are within *eps_pct* of the
        minimum or maximum value.  Returns ``0.0`` for constant signals and
        ``np.nan`` for empty input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.nan
    rng = np.max(signal) - np.min(signal)
    if rng == 0:
        return 0.0
    eps = eps_pct * rng
    clipped = np.sum(
        (signal <= np.min(signal) + eps) | (signal >= np.max(signal) - eps)
    )
    return float(clipped) / signal.size


def baseline_wander_sqi(signal, sampling_rate=100):
    """
    Quantify low-frequency baseline drift relative to total signal energy.

    Baseline wander (< 0.5 Hz) is caused by respiration, motion, and
    electrode movement.  High baseline wander indicates a noisy recording.
    The metric is the ratio of LF STFT energy to total STFT energy.

    Parameters
    ----------
    signal : array_like
        Input signal.
    sampling_rate : int, optional
        Sampling frequency in Hz (default ``100``).

    Returns
    -------
    float
        Ratio of energy below 0.5 Hz to total energy, in ``[0, 1]``.
        Returns ``np.nan`` if total energy is zero or the signal is too short.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 4:
        return np.nan
    nperseg = min(256, signal.size)
    f, _, spec = sn.stft(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=True,
    )
    power = np.abs(spec) ** 2
    total = np.sum(power)
    if total == 0:
        return np.nan
    lf_idx = f < 0.5
    lf = np.sum(power[lf_idx])
    return float(lf / total)


def spectral_snr_sqi(signal, sampling_rate=100, signal_band=None):
    """
    Estimate signal-to-noise ratio as in-band power over out-of-band power.

    Uses the STFT to separate physiological-band energy from noise-band
    energy.  Defaults assume a PPG signal (0.5-4 Hz physiological band).
    For ECG use ``signal_band=[0.5, 40]``.

    Parameters
    ----------
    signal : array_like
        Input signal.
    sampling_rate : int, optional
        Sampling frequency in Hz (default ``100``).
    signal_band : list of float, optional
        ``[low_hz, high_hz]`` of the physiological frequency band.
        Defaults to ``[0.5, 4.0]`` (PPG heart rate band).

    Returns
    -------
    float
        SNR in dB (``10 * log10(in_band / out_of_band)``).
        Returns ``np.nan`` if out-of-band power is zero or signal is too short.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 4:
        return np.nan
    if signal_band is None:
        signal_band = [0.5, 4.0]
    nperseg = min(256, signal.size)
    f, _, spec = sn.stft(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=True,
    )
    power = np.sum(np.abs(spec) ** 2, axis=1)
    in_band = np.sum(power[(f >= signal_band[0]) & (f <= signal_band[1])])
    out_band = np.sum(power[(f < signal_band[0]) | (f > signal_band[1])])
    if out_band == 0:
        return np.nan
    return float(10 * np.log10(in_band / out_band))
