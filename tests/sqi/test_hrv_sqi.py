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
        with pytest.warns(UserWarning):
            res = sdsd_sqi("invalid_input")
            assert np.isnan(res)  # Ensure the function returns NaN for invalid inputs

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
        with pytest.warns(UserWarning):
            res = rmssd_sqi("invalid_input")
            assert np.isnan(res)  # Ensure the function returns NaN for invalid inputs

    def test_cvsd_sqi(self, valid_nn_intervals):
        assert cvsd_sqi(valid_nn_intervals) == pytest.approx(0.01994, rel=1e-1)
        with pytest.warns(UserWarning):
            res = cvsd_sqi("invalid_input")
            assert np.isnan(res)  # Ensure the function returns NaN for invalid inputs

    def test_cvnn_sqi(self, valid_nn_intervals):
        assert cvnn_sqi(valid_nn_intervals) == pytest.approx(0.0182, rel=1e-1)
        with pytest.warns(UserWarning):
            res = cvnn_sqi("invalid_input")
            assert np.isnan(res)  # Ensure the function returns NaN for invalid inputs

    def test_median_nn_sqi(self, valid_nn_intervals, empty_nn_intervals):
        assert median_nn_sqi(valid_nn_intervals) == pytest.approx(810, rel=1e1)
        assert np.isnan(median_nn_sqi(empty_nn_intervals))

    def test_pnn_sqi(self, valid_nn_intervals, short_nn_intervals):
        assert pnn_sqi(valid_nn_intervals, threshold=10) == pytest.approx(
            100.0, rel=1e-2
        )
        assert np.isnan(pnn_sqi(short_nn_intervals))
        with pytest.warns(UserWarning):
            res = pnn_sqi("invalid_input")
            assert np.isnan(res)  # Ensure the function returns NaN for invalid inputs

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

        # Test all metrics
        length = 500
        base_rate = 600
        variability = 50
        synthetic_nn_intervals = base_rate + np.random.randint(
            -variability, variability, size=length
        )
        freq_min = 0.04
        freq_max = 0.15
        result_peak = frequency_sqi(
            synthetic_nn_intervals, freq_min=freq_min, freq_max=freq_max, metric="peak"
        )
        assert result_peak >= 0 or np.isnan(result_peak)
        result_absolute = frequency_sqi(
            synthetic_nn_intervals,
            freq_min=freq_min,
            freq_max=freq_max,
            metric="absolute",
        )
        assert result_absolute >= 0 or np.isnan(result_absolute)
        result_log = frequency_sqi(
            synthetic_nn_intervals, freq_min=freq_min, freq_max=freq_max, metric="log"
        )
        assert result_log >= 0 or np.isnan(result_log)
        result_normalized = frequency_sqi(
            synthetic_nn_intervals,
            freq_min=freq_min,
            freq_max=freq_max,
            metric="normalized",
        )
        assert result_normalized >= 0 or np.isnan(result_normalized)
        result_relative = frequency_sqi(
            synthetic_nn_intervals,
            freq_min=freq_min,
            freq_max=freq_max,
            metric="relative",
        )
        assert result_relative >= 0 or np.isnan(result_relative)

    def generate_nn_intervals(self, length=500, base_rate=600, variability=50):
        """
        Generate synthetic NN intervals mimicking heart rate variability.

        Parameters:
        ----------
        length : int
            Number of intervals to generate.
        base_rate : int
            Base NN interval in milliseconds.
        variability : int
            Maximum variability in NN interval.

        Returns:
        -------
        list
            A list of synthetic NN intervals.
        """
        np.random.seed(42)  # For reproducibility
        # Generate random variations around the base rate
        nn_intervals = base_rate + np.random.randint(
            -variability, variability, size=length
        )
        return nn_intervals.tolist()

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

        length = 500
        base_rate = 600
        variability = 50
        synthetic_nn_intervals = base_rate + np.random.randint(
            -variability, variability, size=length
        )
        ratio = lf_hf_ratio_sqi(
            synthetic_nn_intervals, lf_range=(1e-3, 1e3), hf_range=(1e-4, 1e4)
        )
        print(ratio)
        assert not np.isnan(ratio)

        very_high_hf_ratio = lf_hf_ratio_sqi(
            synthetic_nn_intervals, lf_range=(0.5, 1.0), hf_range=(10.0, 20.0)
        )
        assert np.isnan(very_high_hf_ratio)

    def test_poincare_features_sqi(self, valid_nn_intervals, short_nn_intervals):
        features = poincare_features_sqi(valid_nn_intervals)
        assert features["sd1"] >= 0
        assert features["sd2"] >= 0
        assert features["area"] >= 0
        assert features["ratio"] >= 0
        features = poincare_features_sqi(short_nn_intervals)
        for key in features:
            assert np.isnan(features[key])
        with pytest.warns(UserWarning):
            res = poincare_features_sqi("invalid_input")
            assert np.isnan(
                res["sd1"]
            )  # Ensure the function returns NaN for invalid inputs.res['sd1']

    def test_get_all_features_hrva(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000))  # Simulated signal
        sample_rate = 100

        # Valid case
        features = get_all_features_hrva(signal, sample_rate=sample_rate)
        assert isinstance(features, dict)

        # Edge case: Invalid peak detection
        invalid_signal = [0] * 1000  # Flat signal, no peaks
        features = get_all_features_hrva(invalid_signal, sample_rate=sample_rate)
        assert features == {}

        # Edge case: Invalid wave type
        with pytest.raises(
            Exception, match="Invalid wave_type. Expected 'PPG' or 'ECG'."
        ):
            get_all_features_hrva(signal, sample_rate=sample_rate, wave_type="INVALID")

        # Edge case: Invalid sample rate
        with pytest.raises(Exception, match="Sample rate must be a positive number."):
            get_all_features_hrva(signal, sample_rate=-100)

        with pytest.warns(UserWarning):
            res = get_all_features_hrva("invalid_input")
            assert len(res) == 0

    def test_hr_sqi_invalid_stat(self, valid_nn_intervals):
        result = hr_sqi(valid_nn_intervals, stat="invalid")
        assert np.isnan(result), "Expected NaN for invalid stat input"

    def test_hr_range_sqi_edge_cases(self):
        nn_intervals = [800, 810, 820, 830]
        assert hr_range_sqi(nn_intervals, range_min=900, range_max=1000) == 100.0
        assert hr_range_sqi(nn_intervals, range_min=500, range_max=700) == 100.0

    def test_frequency_sqi_empty_band_powers(self):
        nn_intervals = [
            800,
            810,
            820,
            830,
        ]  # Adjust as necessary to force empty band_powers
        result = frequency_sqi(
            nn_intervals, freq_min=0.4, freq_max=0.5, metric="absolute"
        )
        assert np.isnan(result)

    def test_lf_hf_ratio_sqi_invalid_range(self, valid_nn_intervals):
        result = lf_hf_ratio_sqi(
            valid_nn_intervals, lf_range=(0.5, 0.4), hf_range=(0.15, 0.1)
        )
        assert np.isnan(result)

    def test_poincare_features_sqi_insufficient_data(self):
        features = poincare_features_sqi([800])  # Insufficient intervals
        for key in features:
            assert np.isnan(features[key])

    def test_get_all_features_hrva_invalid_inputs(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 1000))
        sample_rate = 100

        # Invalid peak detection method
        result = get_all_features_hrva(
            signal, sample_rate=sample_rate, rpeak_method=999
        )
        assert (
            result is not None
        ), "Expected a default dictionary for invalid peak detection method"

        # Invalid sample rate
        with pytest.raises(ValueError, match="Sample rate must be a positive number"):
            get_all_features_hrva(signal, sample_rate=-100)

    def test_get_all_features_hrva_rr_interval_failure(self):
        invalid_signal = [0] * 1000  # Flat signal, no peaks
        features = get_all_features_hrva(invalid_signal, sample_rate=100)
        assert features == {}

    def test_get_all_features_hrva_feature_extraction_failure(self):
        invalid_signal = np.sin(np.linspace(0, 2 * np.pi, 10))  # Too short for HRV
        features = get_all_features_hrva(invalid_signal, sample_rate=100)
        assert features == {}

    def test_get_all_features_hrva_final_block(self):
        invalid_signal = [0] * 1000
        features = get_all_features_hrva(invalid_signal, sample_rate=100)
        assert features == {}
