import pytest
import numpy as np
from vital_sqi.sqi.hrv_sqi import (
    nn_mean_sqi,
    sdnn_sqi,
    sdsd_sqi,
    rmssd_sqi,
    cvsd_sqi,
    cvnn_sqi,
    median_nn_sqi,
    pnn_sqi,
    hr_sqi,
    hr_range_sqi,
    frequency_sqi,
    lf_hf_ratio_sqi,
    poincare_features_sqi,
    get_all_features_hrva,
)


class TestHRVSQIs:
    @pytest.fixture
    def valid_nn_intervals(self):
        return [800, 810, 820, 830, 800, 790]

    @pytest.fixture
    def empty_nn_intervals(self):
        return []

    @pytest.fixture
    def short_nn_intervals(self):
        return [800]

    def test_nn_mean_sqi(self, valid_nn_intervals, empty_nn_intervals):
        assert nn_mean_sqi(valid_nn_intervals) == pytest.approx(808.33, rel=1e-2)
        assert np.isnan(nn_mean_sqi(empty_nn_intervals))

    def test_sdnn_sqi(self, valid_nn_intervals, empty_nn_intervals):
        assert sdnn_sqi(valid_nn_intervals) == pytest.approx(14.58, rel=1e-2)
        assert np.isnan(sdnn_sqi(empty_nn_intervals))

    def test_sdsd_sqi(self, valid_nn_intervals, short_nn_intervals):
        """Test SDSD calculation with valid and insufficient NN intervals."""
        # Test with valid NN intervals
        expected_sdsd = np.std(np.diff(valid_nn_intervals))
        assert sdsd_sqi(valid_nn_intervals) == pytest.approx(expected_sdsd, rel=1e-2)

        # Test with insufficient NN intervals
        assert np.isnan(sdsd_sqi(short_nn_intervals))

        # Test with empty NN intervals
        assert np.isnan(sdsd_sqi([]))

        # Test with invalid input
        assert np.isnan(sdsd_sqi(["invalid_input"]))

        # # Test with invalid input
        # with pytest.raises(ValueError, match="diff requires input that is at least one dimensional"):
        #     sdsd_sqi("invalid_input")

    def test_rmssd_sqi(self, valid_nn_intervals, short_nn_intervals):
        # Test with valid NN intervals
        expected_rmssd = np.std(np.diff(valid_nn_intervals))
        assert rmssd_sqi(valid_nn_intervals) == pytest.approx(expected_rmssd, rel=1e-2)

        # Test with insufficient NN intervals
        assert np.isnan(rmssd_sqi(short_nn_intervals))

        # Test with empty NN intervals
        assert np.isnan(rmssd_sqi([]))

        # Test with invalid input
        assert np.isnan(rmssd_sqi(["invalid_input"]))

        # Test with invalid input
        # with pytest.raises(ValueError, match="diff requires input that is at least one dimensional"):
        #     rmssd_sqi("invalid_input")

    def test_cvsd_sqi(self, valid_nn_intervals):
        assert cvsd_sqi(valid_nn_intervals) == pytest.approx(0.01994, rel=1e-1)

    def test_cvnn_sqi(self, valid_nn_intervals):
        assert cvnn_sqi(valid_nn_intervals) == pytest.approx(0.0182, rel=1e-1)

    def test_median_nn_sqi(self, valid_nn_intervals, empty_nn_intervals):
        assert median_nn_sqi(valid_nn_intervals) == pytest.approx(810, rel=1e1)
        assert np.isnan(median_nn_sqi(empty_nn_intervals))

    def test_pnn_sqi(self, valid_nn_intervals, short_nn_intervals):
        assert pnn_sqi(valid_nn_intervals, threshold=10) == pytest.approx(
            100.0, rel=1e-2
        )
        assert np.isnan(pnn_sqi(short_nn_intervals))

    def test_hr_sqi(self, valid_nn_intervals):
        assert hr_sqi(valid_nn_intervals, stat="mean") == pytest.approx(74.26, rel=1e-2)
        assert hr_sqi(valid_nn_intervals, stat="min") == pytest.approx(72.29, rel=1e-2)
        assert hr_sqi(valid_nn_intervals, stat="max") == pytest.approx(75.95, rel=1e-2)
        assert hr_sqi(valid_nn_intervals, stat="median") == pytest.approx(
            74.68, rel=1e-2
        )
        assert hr_sqi(valid_nn_intervals, stat="std") == pytest.approx(1.2283, rel=1e-2)

        # Test invalid statistic
        with pytest.warns(UserWarning, match="Invalid statistic requested"):
            result = hr_sqi(valid_nn_intervals, stat="invalid")
            assert np.isnan(result)  # Ensure the function returns NaN for invalid stats

    def test_hr_range_sqi(self, valid_nn_intervals):
        assert hr_range_sqi(valid_nn_intervals, range_min=60, range_max=90) == 0.0
        assert hr_range_sqi(
            valid_nn_intervals, range_min=80, range_max=90
        ) == pytest.approx(100.0, rel=1e-2)

    def test_frequency_sqi(self, valid_nn_intervals):
        """Test frequency domain SQI calculation."""
        # Valid case
        result = frequency_sqi(
            valid_nn_intervals, freq_min=0.04, freq_max=0.15, metric="peak"
        )
        assert result >= 0 or np.isnan(
            result
        )  # Ensure valid peak or NaN if band_powers is empty

        # Invalid input cases
        invalid_inputs = [[], [800], "invalid", None]
        for input_data in invalid_inputs:
            assert np.isnan(
                frequency_sqi(input_data, freq_min=0.04, freq_max=0.15, metric="peak")
            )

        # Invalid metric
        with pytest.raises(ValueError, match="Invalid metric requested"):
            frequency_sqi(
                valid_nn_intervals, freq_min=0.04, freq_max=0.15, metric="unknown"
            )

        # Edge case: No power in specified band
        nn_intervals_with_no_power = [
            800,
            810,
            820,
        ]  # Example where band_powers might be empty
        assert np.isnan(
            frequency_sqi(
                nn_intervals_with_no_power, freq_min=0.4, freq_max=0.5, metric="peak"
            )
        )

    def test_lf_hf_ratio_sqi(self, valid_nn_intervals, empty_nn_intervals):
        """Test LF/HF ratio calculation."""
        # Valid case
        result = lf_hf_ratio_sqi(valid_nn_intervals)
        assert np.isnan(result) or result >= 0  # Ensure valid ratio or NaN

        # Empty NN intervals
        assert np.isnan(lf_hf_ratio_sqi(empty_nn_intervals))

        # Insufficient NN intervals
        assert np.isnan(lf_hf_ratio_sqi([800, 810]))

        # Invalid input
        invalid_inputs = [None, "invalid", 123]
        for input_data in invalid_inputs:
            assert np.isnan(lf_hf_ratio_sqi(input_data))

        # Edge case: LF or HF power is zero
        nn_intervals_no_hf = [
            800,
            810,
            820,
            830,
            840,
        ]  # Example that might lead to zero HF power
        assert np.isnan(lf_hf_ratio_sqi(nn_intervals_no_hf))

        # Invalid frequency range
        assert np.isnan(
            lf_hf_ratio_sqi(
                valid_nn_intervals, lf_range=(0.5, 1.0), hf_range=(1.0, 2.0)
            )
        )

    def test_poincare_features_sqi(self, valid_nn_intervals, short_nn_intervals):
        features = poincare_features_sqi(valid_nn_intervals)
        assert features["sd1"] >= 0
        assert features["sd2"] >= 0
        assert features["area"] >= 0
        assert features["ratio"] >= 0
        features = poincare_features_sqi(short_nn_intervals)
        for key in features:
            assert np.isnan(features[key])

    def test_get_all_features_hrva(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000))  # Simulated signal
        sample_rate = 100

        # Valid case
        features = (
            get_all_features_hrva(signal, sample_rate=sample_rate)
        )
        assert isinstance(features, dict)

        # Edge case: Invalid peak detection
        invalid_signal = [0] * 1000  # Flat signal, no peaks
        features = (
            get_all_features_hrva(invalid_signal, sample_rate=sample_rate)
        )
        assert features == {}

        # Edge case: Invalid wave type
        with pytest.raises(
            Exception, match="Invalid wave_type. Expected 'PPG' or 'ECG'."
        ):
            get_all_features_hrva(signal, sample_rate=sample_rate, wave_type="INVALID")

        # Edge case: Invalid sample rate
        with pytest.raises(Exception, match="Sample rate must be a positive number."):
            get_all_features_hrva(signal, sample_rate=-100)
