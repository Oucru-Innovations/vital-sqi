import numpy as np
import pandas as pd
from scipy import signal
from typing import Optional
from vital_sqi.common.generate_template import squeeze_template


def taper_signal(
    s: np.ndarray, window: Optional[np.ndarray] = None, shift_min_to_zero: bool = True
) -> np.ndarray:
    """
    Applies a tapering window to the signal and optionally shifts the minimum value to zero.

    Parameters
    ----------
    s : np.ndarray
        Input signal as a 1D array of floats.
    window : np.ndarray, optional
        Window shape to apply, defaults to Tukey window if None.
    shift_min_to_zero : bool, optional
        If True, shifts the signal minimum value to zero.

    Returns
    -------
    np.ndarray
        Tapered and optionally shifted signal.
    """
    if shift_min_to_zero:
        s = s - np.min(s)
    if window is None:
        window = signal.windows.tukey(len(s), 0.9)
    return s * window


def smooth_signal(
    s: np.ndarray, window_len: int = 5, window: str = "flat"
) -> np.ndarray:
    """
    Smooths the signal using a specified window.

    Parameters
    ----------
    s : np.ndarray
        1D signal array.
    window_len : int, optional
        Size of the smoothing window, must be greater than 2.
    window : str, optional
        Window type for smoothing. Options: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if not isinstance(window_len, int) or window_len < 3:
        raise ValueError("window_len must be an integer greater than or equal to 3.")

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "window must be 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'."
        )

    if s.ndim != 1:
        raise ValueError("smoothing only supports 1D arrays.")

    if len(s) < window_len:
        raise ValueError("Input signal length must be greater than window size.")

    # Extend the signal at both ends for smoothing
    s_ext = np.r_[s[window_len - 1 : 0 : -1], s, s[-2 : -window_len - 1 : -1]]

    if window == "flat":
        w = np.ones(window_len)
    else:
        w = getattr(np, window)(window_len)

    return np.convolve(w / w.sum(), s_ext, mode="same")[
        window_len - 1 : -window_len + 1
    ]


def scale_pattern(s: np.ndarray, window_size: int) -> np.ndarray:
    """
    Scales or resamples the signal to a specified window size for comparison with a template.

    Parameters
    ----------
    s : np.ndarray
        Input signal as a 1D array of floats.
    window_size : int
        The desired size of the output signal.

    Returns
    -------
    np.ndarray
        Resampled and smoothed signal to match the desired window size.
    """
    if len(s) == window_size:
        return s
    elif len(s) < window_size:
        # Use linear interpolation for efficient upsampling
        scaled_signal = np.interp(
            np.linspace(0, len(s) - 1, window_size), np.arange(len(s)), s
        )
    else:
        # Downsample using a predefined squeeze function
        scaled_signal = squeeze_template(s, window_size)

    # Ensure the smoothing window length is valid
    window_len = min(5, len(scaled_signal))
    return smooth_signal(scaled_signal, window_len=window_len)
