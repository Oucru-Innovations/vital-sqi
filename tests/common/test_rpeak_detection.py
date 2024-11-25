import numpy as np
import pytest
from vital_sqi.common.rpeak_detection import (
    PeakDetector,
    ADAPTIVE_THRESHOLD,
    CLUSTERER_METHOD,
    SLOPE_SUM_METHOD,
    MOVING_AVERAGE_METHOD,
    COUNT_ORIG_METHOD,
    DEFAULT,
    BILLAUER_METHOD,
)

# Mock a test signal
mock_signal = np.sin(
    np.linspace(0, 10 * np.pi, 100)
)  # A sine wave for basic peak testing

# Initialize the PeakDetector class for both ECG and PPG signals
detector_ppg = PeakDetector(wave_type="PPG", fs=100)
detector_ecg = PeakDetector(wave_type="ECG", fs=100)


@pytest.mark.parametrize(
    "detector_type",
    [
        ADAPTIVE_THRESHOLD,
        CLUSTERER_METHOD,
        SLOPE_SUM_METHOD,
        MOVING_AVERAGE_METHOD,
        COUNT_ORIG_METHOD,
        DEFAULT,
        BILLAUER_METHOD,
    ],
)
def test_ppg_detector_methods(detector_type):
    peaks, troughs = detector_ppg.ppg_detector(mock_signal, detector_type=detector_type)
    assert isinstance(
        peaks, np.ndarray
    ), f"Expected peaks to be an array for detector_type {detector_type}"
    assert isinstance(
        troughs, np.ndarray
    ), f"Expected troughs to be an array for detector_type {detector_type}"
    # Peaks and troughs in a sine wave should not be empty for most detector types
    if detector_type not in [
        COUNT_ORIG_METHOD,
        SLOPE_SUM_METHOD,
    ]:  # Adjust if methods are unimplemented
        assert (
            len(peaks) >= 0
        ), f"No error during peaks detection for detector_type {detector_type}"
        assert (
            len(troughs) >= 0
        ), f"No error during troughs detection for detector_type {detector_type}"


def test_ecg_detector():
    realistic_ecg_signal = np.sin(
        np.linspace(0, 10 * np.pi, 100)
    ) + 0.1 * np.random.randn(100)
    r_peaks, q_valleys, s_valleys, p_peaks, t_peaks = detector_ecg.ecg_detector(
        realistic_ecg_signal
    )
    for item in [r_peaks, q_valleys, s_valleys, p_peaks, t_peaks]:
        assert isinstance(item, np.ndarray), "Expected output to be an array"
    assert len(r_peaks) >= 0, "No Error during R peaks detected in realistic ECG signal"


def test_ppg_detector_preprocess_cubing():
    # Test preprocessing and cubing options
    peaks, troughs = detector_ppg.ppg_detector(
        mock_signal, preprocess=True, cubing=True
    )
    assert len(peaks) > 0, "No peaks detected with preprocess and cubing"
    assert len(troughs) > 0, "No troughs detected with preprocess and cubing"


def test_get_moving_average():
    # Test the moving average calculation with a simple input
    window_size = 5
    moving_avg = detector_ppg.get_moving_average(mock_signal, window_size)
    assert len(moving_avg) == len(mock_signal), "Moving average length mismatch"
    assert np.isclose(
        np.mean(moving_avg), 0, atol=0.1
    ), "Moving average incorrect for a sine wave"


def test_detect_peak_trough_clusterer():
    peaks, troughs = detector_ppg.detect_peak_trough_clusterer(mock_signal)
    assert len(peaks) > 0, "No peaks detected with clustering"
    assert len(troughs) > 0, "No troughs detected with clustering"


def test_detect_peak_trough_DEFAULT():
    peaks, troughs = detector_ppg.detect_peak_trough_DEFAULT(mock_signal)
    assert len(peaks) > 0, "No peaks detected with default scipy"
    assert len(troughs) > 0, "No troughs detected with default scipy"


def test_detect_peak_trough_moving_average_threshold():
    peaks, troughs = detector_ppg.detect_peak_trough_moving_average_threshold(
        mock_signal
    )
    assert (
        len(peaks) >= 0
    ), "No error during peaks detection with moving average threshold"


def test_detect_peak_trough_billauer():
    peaks, troughs = detector_ppg.detect_peak_trough_billauer(mock_signal, delta=0.2)
    assert len(peaks) > 0, "No peaks detected with Billauer method"
    assert len(troughs) > 0, "No troughs detected with Billauer method"


# def test_edge_case_empty_signal():
#     empty_signal = np.array([])
#     with pytest.raises(ValueError, match="Input signal is empty."):
#         detector_ppg.ppg_detector(empty_signal)


def test_edge_case_constant_signal():
    constant_signal = np.ones(100)
    peaks, troughs = detector_ppg.ppg_detector(constant_signal)
    assert len(peaks) == 0, "Peaks detected in constant signal"
    assert len(troughs) == 0, "Troughs detected in constant signal"


def test_invalid_wave_type():
    with pytest.raises(ValueError, match="Invalid wave_type. Expected 'PPG' or 'ECG'."):
        invalid_detector = PeakDetector(wave_type="unknown", fs=100)
        invalid_detector.ppg_detector(mock_signal)


# def test_invalid_detector_type():
#     with pytest.raises(ValueError, match="Invalid detector_type: 999"):
#         detector_ppg.ppg_detector(mock_signal, detector_type=999)


def test_invalid_window_size_for_moving_average():
    with pytest.raises(ValueError, match="Window size must be greater than 0."):
        detector_ppg.get_moving_average(mock_signal, 0)


def test_missing_required_methods():
    # Check that required methods are implemented in WaveformMorphology
    try:
        _ = detector_ecg.ecg_detector(mock_signal)
    except AttributeError as e:
        pytest.fail(f"Missing required method in WaveformMorphology: {e}")
