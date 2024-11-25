import pytest
import numpy as np
import logging
from vital_sqi.common.power_spectrum import (
    calculate_band_power,
    interpolate_rr_intervals,
    compute_time_and_bpm,
    calculate_psd,
    calculate_spectrogram,
    calculate_wavelet_power,
)

# Mock RR intervals for testing
mock_rr_intervals = np.array([800, 820, 780, 790, 810, 850])  # in ms
mock_freq = np.linspace(0, 1, 100)  # Frequency range for band power calculation
mock_power = np.random.rand(100)  # Mock power values for testing


def test_calculate_band_power():
    # Test within normal band
    power = calculate_band_power(mock_freq, mock_power, fmin=0.2, fmax=0.4)
    assert isinstance(power, float), "Expected band power to be a float"
    assert power > 0, "Band power should be greater than zero within the band range"

    # Edge case: out-of-range frequency band
    power = calculate_band_power(mock_freq, mock_power, fmin=1.2, fmax=1.5)
    assert power == 0, "Expected band power to be zero for out-of-range frequencies"


def test_interpolate_rr_intervals(caplog):
    ts_rr, bpm_list = compute_time_and_bpm(mock_rr_intervals)
    interpolated_bpm = interpolate_rr_intervals(ts_rr, bpm_list, sampling_frequency=4)
    assert interpolated_bpm.size > 0, "Interpolated BPM should not be empty"
    assert (
        len(interpolated_bpm) > 1
    ), "Interpolated BPM length should be greater than one"

    # Edge case: insufficient data points
    with caplog.at_level(logging.ERROR):
        interpolate_rr_intervals(np.array([1]), np.array([60]), sampling_frequency=4)
    assert (
        "Insufficient data points for interpolation." in caplog.text
    ), "Expected log message for insufficient data points."


def test_compute_time_and_bpm():
    ts_rr, bpm_list = compute_time_and_bpm(mock_rr_intervals)
    assert isinstance(ts_rr, np.ndarray) and isinstance(
        bpm_list, np.ndarray
    ), "Expected numpy arrays"
    assert len(ts_rr) == len(mock_rr_intervals), "Timestamp array length mismatch"
    assert len(bpm_list) == len(mock_rr_intervals), "BPM array length mismatch"
    assert bpm_list[0] > 0, "BPM values should be positive"

    # Edge case: empty RR intervals
    ts_rr, bpm_list = compute_time_and_bpm(np.array([]))
    assert (
        ts_rr.size == 0 and bpm_list.size == 0
    ), "Expected empty arrays for empty input"


@pytest.mark.parametrize("method", ["welch", "lomb", "ar"])
def test_calculate_psd(method, caplog):
    # Run the main calculation and validate the output
    freq, psd = calculate_psd(mock_rr_intervals, method=method)
    assert isinstance(freq, np.ndarray) and isinstance(
        psd, np.ndarray
    ), "Expected numpy arrays"
    assert (
        freq.size >= 0 and psd.size >= 0
    ), "Frequency and PSD arrays should not be None"

    # Edge case: invalid method, expecting a log message instead of a raised error
    with caplog.at_level(logging.ERROR):
        calculate_psd(mock_rr_intervals, method="invalid")
    assert (
        "Invalid method. Choose from 'welch', 'lomb', or 'ar'." in caplog.text
    ), "Expected log message for invalid method."


def test_calculate_spectrogram():
    freq, psd, time_segments = calculate_spectrogram(
        mock_rr_intervals, hr_sampling_frequency=4
    )
    assert isinstance(freq, np.ndarray), "Expected frequency array to be numpy array"
    assert isinstance(psd, np.ndarray), "Expected PSD array to be numpy array"
    assert isinstance(
        time_segments, np.ndarray
    ), "Expected time segments array to be numpy array"
    assert (
        len(freq) > 0 and len(psd) > 0 and len(time_segments) > 0
    ), "Outputs should not be empty"

    # Edge case: insufficient RR interval data
    freq, psd, time_segments = calculate_spectrogram(
        np.array([800]), hr_sampling_frequency=4
    )
    assert (
        freq.size == 0 and psd.size == 0 and time_segments.size == 0
    ), "Expected empty arrays for insufficient data"


@pytest.mark.parametrize("wavelet_type", ["gaussian", "paul", "mexican_hat", "morlet"])
def test_calculate_wavelet_power(wavelet_type, caplog):
    freqs, power = calculate_wavelet_power(
        mock_rr_intervals, heart_rate=4, mother_wave=wavelet_type
    )
    assert isinstance(freqs, np.ndarray) and isinstance(
        power, np.ndarray
    ), "Expected numpy arrays"
    assert (
        freqs.size > 0 and power.size > 0
    ), "Frequency and power arrays should not be empty"

    # Edge case: invalid wavelet type
    with caplog.at_level(logging.ERROR):
        freqs, power = calculate_wavelet_power(
            mock_rr_intervals, mother_wave="invalid_wavelet"
        )
    assert (
        "Invalid wavelet type" in caplog.text
    ), "Expected log message for invalid wavelet type"
    # assert freqs.size > 0 and power.size > 0, "Default behavior: fall back to Morlet wavelet"
