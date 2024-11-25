"""
Signal Quality Index (SQI) calculations for photoplethysmogram (PPG) signals.

These SQIs are based on the paper by Elgendi, Mohamed:
"Optimal Signal Quality Index for Photoplethysmogram Signals", Bioengineering.
"""

import numpy as np
from scipy.stats import kurtosis, skew, entropy


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
    x = np.array(x)
    x_shifted = x - np.min(x)  # Shift x to non-negative
    if np.sum(x_shifted) == 0:
        raise ValueError("The sum of the input signal is zero; cannot compute entropy.")
    prob_dist = x_shifted / np.sum(x_shifted)  # Normalize to probability distribution
    return entropy(prob_dist, qk=qk, base=base, axis=axis)


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

    Parameters
    ----------
    y : array_like
        Input signal.
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
