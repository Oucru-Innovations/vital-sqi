import pytest
import numpy as np
from scipy.signal.windows import tukey
from vital_sqi.preprocess.preprocess_signal import (
    taper_signal,
    smooth_signal,
    scale_pattern,
)
from vital_sqi.common.generate_template import squeeze_template


def test_taper_signal_no_window():
    """Test tapering a signal with no window specified."""
    signal = np.array([1, 2, 3, 4, 5])
    tapered_signal = taper_signal(signal, shift_min_to_zero=True)
    expected_window = tukey(len(signal), 0.9)
    expected_result = (signal - np.min(signal)) * expected_window
    np.testing.assert_almost_equal(tapered_signal, expected_result)


def test_taper_signal_with_window():
    """Test tapering a signal with a custom window."""
    signal = np.array([1, 2, 3, 4, 5])
    custom_window = np.array([0.1, 0.5, 0.8, 0.5, 0.1])
    tapered_signal = taper_signal(signal, window=custom_window, shift_min_to_zero=False)
    expected_result = signal * custom_window
    np.testing.assert_almost_equal(tapered_signal, expected_result)


def test_smooth_signal_flat_window():
    """Test smoothing a signal with a flat window."""
    signal = np.array([1, 2, 3, 4, 5, 6, 7])
    smoothed_signal = smooth_signal(signal, window_len=3, window="flat")
    extended_signal = np.r_[signal[2::-1], signal, signal[:-3:-1]]
    expected_result = np.convolve(np.ones(3) / 3, extended_signal, mode="valid")[
        2 : len(signal) + 2
    ]
    tolerance = 1e2
    assert np.all(np.abs(smoothed_signal - expected_result) < tolerance), (
        f"Smoothed signal deviates from expected result by more than {tolerance}.\n"
        f"Smoothed Signal: {smoothed_signal}\nExpected Result: {expected_result}"
    )


def test_smooth_signal_hamming_window():
    """Test smoothing a signal with a Hamming window."""
    signal = np.array([1, 2, 3, 4, 5, 6, 7])
    smoothed_signal = smooth_signal(signal, window_len=3, window="hamming")
    hamming_window = np.hamming(3)

    # Extend the signal symmetrically for smoothing
    extended_signal = np.r_[signal[2::-1], signal, signal[:-3:-1]]

    # Convolve and trim to match the original signal length
    expected_result = np.convolve(
        hamming_window / hamming_window.sum(), extended_signal, mode="valid"
    )[
        2 : len(signal) + 2
    ]  # Adjust indices to match the input signal length

    tolerance = 1e2
    assert np.all(np.abs(smoothed_signal - expected_result) < tolerance), (
        f"Smoothed signal deviates from expected result by more than {tolerance}.\n"
        f"Smoothed Signal: {smoothed_signal}\nExpected Result: {expected_result}"
    )


def test_smooth_signal_invalid_params():
    """Test smooth_signal with invalid parameters."""
    with pytest.raises(ValueError, match="window_len must be an integer"):
        smooth_signal(np.array([1, 2, 3]), window_len=2)

    with pytest.raises(ValueError, match="window must be"):
        smooth_signal(np.array([1, 2, 3]), window_len=3, window="invalid_window")

    with pytest.raises(ValueError, match="smoothing only supports 1D arrays"):
        smooth_signal(np.array([[1, 2], [3, 4]]), window_len=3)

    with pytest.raises(
        ValueError, match="Input signal length must be greater than window size"
    ):
        smooth_signal(np.array([1, 2]), window_len=3)


def test_scale_pattern_upsampling():
    """Test scaling a signal to a larger window size."""
    signal = np.array([1, 2, 3])
    scaled_signal = scale_pattern(signal, window_size=6)
    expected_result = np.interp(
        np.linspace(0, len(signal) - 1, 6), np.arange(len(signal)), signal
    )
    smoothed_result = smooth_signal(expected_result, window_len=5)
    np.testing.assert_almost_equal(scaled_signal, smoothed_result)


def test_scale_pattern_downsampling():
    """Test scaling a signal to a smaller window size."""
    signal = np.array([1, 2, 3, 4, 5, 6])
    window_size = 3
    scaled_signal = scale_pattern(signal, window_size=window_size)
    expected_result = squeeze_template(
        signal, len(scaled_signal)
    )  # Replace with actual squeeze logic
    tolerance = 1e2
    assert np.all(np.abs(scaled_signal - expected_result) < tolerance), (
        f"Smoothed signal deviates from expected result by more than {tolerance}.\n"
        f"Smoothed Signal: {scaled_signal}\nExpected Result: {expected_result}"
    )


def test_scale_pattern_no_change():
    """Test scaling when the signal length equals the desired window size."""
    signal = np.array([1, 2, 3, 4])
    scaled_signal = scale_pattern(signal, window_size=4)
    np.testing.assert_almost_equal(scaled_signal, signal)


def test_scale_pattern_invalid_window_size():
    """Test scale_pattern with invalid window size."""
    signal = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        scale_pattern(signal, window_size=0)
