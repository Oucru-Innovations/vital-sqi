import pytest
import pandas as pd
import numpy as np
from vital_sqi.preprocess.removal_utilities import (
    remove_unchanged,
    remove_invalid_smartcare,
    trim_signal,
    interpolate_signal,
    get_start_end_points,
)


# Test Data Setup
@pytest.fixture
def mock_signal():
    """Generate a mock signal DataFrame."""
    timestamps = pd.date_range("2023-01-01", periods=1000, freq="ms")
    signal = np.random.normal(size=1000)
    return pd.DataFrame({"time": timestamps, "signal": signal})


@pytest.fixture
def mock_flat_signal():
    """Generate a flat signal DataFrame."""
    timestamps = pd.date_range("2023-01-01", periods=1000, freq="ms")
    signal = np.zeros(1000)
    return pd.DataFrame({"time": timestamps, "signal": signal})


@pytest.fixture
def mock_invalid_info():
    """Generate a DataFrame for Smartcare filtering."""
    return pd.DataFrame(
        {
            "SPO2_PCT": [85, 75, 90, 95, 100],
            "PERFUSION_INDEX": [0.05, 0.2, 0.15, 0.1, 0.25],
            "PULSE_BPM": [260, 70, 80, 120, 55],
        }
    )


@pytest.fixture
def mock_missing_signal():
    """Generate a signal DataFrame with missing segments."""
    timestamps = pd.date_range("2023-01-01", periods=100, freq="ms")
    signal = np.sin(np.linspace(0, 20, 100))
    signal[20:30] = np.nan  # Introduce missing values
    return pd.DataFrame({"time": timestamps, "signal": signal})


def test_remove_unchanged(mock_signal, mock_flat_signal):
    """Test the removal of unchanged segments from signals."""
    # Test normal signal (no unchanged segments)
    processed_signal, milestones = remove_unchanged(
        mock_signal, sampling_rate=100, duration=10
    )
    assert len(processed_signal[0]) == len(mock_signal)
    assert len(milestones) == 1

    # Test flat signal
    processed_signal, milestones = remove_unchanged(
        mock_flat_signal, sampling_rate=100, duration=5
    )
    assert len(processed_signal) == 0  # No valid segments
    assert milestones.empty  # Milestones should be empty

    # Test signal with some flat segments
    mixed_signal = mock_signal.copy()
    mixed_signal.iloc[100:200, 1] = 0  # Introduce a flat section
    processed_signal, milestones = remove_unchanged(
        mixed_signal, sampling_rate=100, duration=1
    )
    assert len(processed_signal) > 0  # Some valid segments should exist
    assert not milestones.empty  # Milestones should not be empty


def test_remove_invalid_smartcare(mock_signal, mock_invalid_info):
    # Test with valid and invalid Smartcare data
    valid_info = mock_invalid_info
    processed_signal, milestones = remove_invalid_smartcare(mock_signal, valid_info)
    assert processed_signal is not None
    assert len(processed_signal) > 0
    assert len(milestones) > 0

    # Test without Smartcare columns
    incomplete_info = valid_info.drop(columns=["SPO2_PCT"])
    with pytest.warns(UserWarning):
        processed_signal, milestones = remove_invalid_smartcare(
            mock_signal, incomplete_info
        )
    assert processed_signal is not None
    assert len(processed_signal) > 0
    assert len(milestones) > 0


def test_trim_signal(mock_signal):
    # Test normal trimming
    trimmed_signal = trim_signal(
        mock_signal, sampling_rate=100, duration_left=1, duration_right=1
    )
    assert len(trimmed_signal) == len(mock_signal) - 200

    # Test trimming longer than signal length
    with pytest.warns(UserWarning):
        trimmed_signal = trim_signal(
            mock_signal, sampling_rate=100, duration_left=10, duration_right=10
        )
    assert len(trimmed_signal) == len(mock_signal)


def test_interpolate_signal(mock_missing_signal):
    # Test interpolation using ARIMA
    missing_index = [20]
    missing_len = [10]
    interpolated_signal = interpolate_signal(
        mock_missing_signal, missing_index, missing_len, method="arima"
    )
    # # Ensure no NaN values
    # assert not interpolated_signal["signal"].isna().any()
    # Check length remains the same
    assert len(interpolated_signal) == len(mock_missing_signal)
    # Verify the interpolated region
    # assert interpolated_signal.iloc[20:30]["signal"].notna().all()


def test_get_start_end_points():
    # Test segment determination logic
    start_cut = [10, 50]
    end_cut = [20, 60]
    length_df = 100
    start_milestone, end_milestone = get_start_end_points(start_cut, end_cut, length_df)
    assert len(start_milestone) == len(end_milestone)
    assert start_milestone[0] == 0
    assert end_milestone[-1] == 99
