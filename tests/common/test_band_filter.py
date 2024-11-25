import pytest
import numpy as np
from vital_sqi.common.band_filter import BandpassFilter


@pytest.fixture
def mock_signal():
    """Generate a mock signal for testing."""
    t = np.linspace(0, 1, 500, endpoint=False)  # 1 second of data at 500 Hz
    return np.sin(2 * np.pi * 10 * t) + np.sin(
        2 * np.pi * 50 * t
    )  # Mix of 10 Hz and 50 Hz


def test_lowpass_filter_butter(mock_signal):
    """Test Butterworth low-pass filter."""
    filter = BandpassFilter(band_type="butter", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=20, order=4)
    assert len(filtered_signal) == len(mock_signal), "Filtered signal length mismatch."
    assert np.max(filtered_signal) < np.max(mock_signal), "Low-pass filter ineffective."


def test_highpass_filter_cheby1(mock_signal):
    """Test Chebyshev Type I high-pass filter."""
    filter = BandpassFilter(band_type="cheby1", fs=500)
    filtered_signal = filter.signal_highpass_filter(mock_signal, cutoff=30, order=4)
    assert len(filtered_signal) == len(mock_signal), "Filtered signal length mismatch."
    assert np.max(filtered_signal) < np.max(
        mock_signal
    ), "High-pass filter ineffective."


def test_lowpass_filter_invalid_cutoff(mock_signal, caplog):
    """Test low-pass filter with invalid cutoff frequency."""
    filter = BandpassFilter(band_type="butter", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=-10, order=4)

    # Verify the log contains the error message
    assert "Cutoff frequency must be between 0 and Nyquist frequency" in caplog.text

    # Verify the returned signal is an empty array
    assert filtered_signal.size == 0


def test_highpass_filter_invalid_order(mock_signal, caplog):
    """Test high-pass filter with invalid order."""
    filter = BandpassFilter(band_type="cheby2", fs=500)
    filtered_signal = filter.signal_highpass_filter(mock_signal, cutoff=20, order=-1)
    # Verify the log contains the actual error message
    assert "Filter order must be a positive integer" in caplog.text
    # Verify the returned signal is an empty array
    assert filtered_signal.size == 0


def test_bessel_lowpass(mock_signal):
    """Test Bessel low-pass filter."""
    filter = BandpassFilter(band_type="bessel", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=15, order=3)
    assert len(filtered_signal) == len(mock_signal), "Filtered signal length mismatch."


def test_ellip_highpass(mock_signal):
    """Test Elliptic high-pass filter."""
    filter = BandpassFilter(band_type="ellip", fs=500)
    filtered_signal = filter.signal_highpass_filter(mock_signal, cutoff=20, order=4)
    assert len(filtered_signal) == len(mock_signal), "Filtered signal length mismatch."


def test_cheby2_lowpass(mock_signal):
    """Test Chebyshev Type II low-pass filter."""
    filter = BandpassFilter(band_type="cheby2", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=30, order=3)
    assert len(filtered_signal) == len(mock_signal), "Filtered signal length mismatch."


def test_filter_invalid_band_type(mock_signal, caplog):
    """Test filter initialization with an invalid band type."""
    filter = BandpassFilter(band_type="invalid", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=20, order=3)
    # Verify the log contains the actual error message
    assert "Invalid band type: invalid" in caplog.text
    # Verify the returned signal is an empty array
    assert filtered_signal.size == 0


def test_highpass_filter_large_cutoff(mock_signal, caplog):
    """Test high-pass filter with cutoff frequency above Nyquist."""
    filter = BandpassFilter(band_type="butter", fs=500)
    filtered_signal = filter.signal_highpass_filter(mock_signal, cutoff=300, order=3)
    # Verify the log contains the actual error message
    assert "Cutoff frequency must be between 0 and Nyquist frequency" in caplog.text
    # Verify the returned signal is an empty array
    assert filtered_signal.size == 0


def test_lowpass_filter_zero_order(mock_signal, caplog):
    """Test low-pass filter with zero filter order."""
    filter = BandpassFilter(band_type="butter", fs=500)
    filtered_signal = filter.signal_lowpass_filter(mock_signal, cutoff=20, order=0)
    # Verify the log contains the actual error message
    assert "Filter order must be a positive integer" in caplog.text
    # Verify the returned signal is an empty array
    assert filtered_signal.size == 0
