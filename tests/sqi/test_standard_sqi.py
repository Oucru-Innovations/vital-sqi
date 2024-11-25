import pytest
import numpy as np
from vital_sqi.sqi.standard_sqi import (
    perfusion_sqi,
    kurtosis_sqi,
    skewness_sqi,
    entropy_sqi,
    signal_to_noise_sqi,
    zero_crossings_rate_sqi,
    mean_crossing_rate_sqi,
)


class TestPerfusionSqi:
    def test_on_perfusion_sqi(self):
        raw_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        filtered_signal = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        result = perfusion_sqi(raw_signal, filtered_signal)
        assert result > 0
        assert result < np.inf

    def test_perfusion_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            perfusion_sqi("invalid", "invalid")


class TestKurtosisSqi:
    def test_on_kurtosis_sqi(self):
        raw_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = kurtosis_sqi(raw_signal)
        assert result == pytest.approx(-1.224, rel=1e-3)

    def test_kurtosis_sqi_edge_case(self):
        zero_signal = np.zeros(10)
        assert kurtosis_sqi(zero_signal) == 0

    def test_kurtosis_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            kurtosis_sqi("invalid")


class TestSkewnessSqi:
    def test_on_skewness_sqi(self):
        raw_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = skewness_sqi(raw_signal)
        assert result == pytest.approx(0, rel=1e-3)

    def test_skewness_sqi_edge_case(self):
        zero_signal = np.zeros(10)
        assert skewness_sqi(zero_signal) == 0

    def test_skewness_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            skewness_sqi("invalid")


class TestEntropySqi:
    def test_on_entropy_sqi(self):
        high_entropy_signal = np.random.randint(0, 10, 100)
        result = entropy_sqi(high_entropy_signal)
        assert result > 0
        assert result < np.log(len(high_entropy_signal))

    def test_entropy_sqi_zero_sum(self):
        zero_signal = np.zeros(10)
        with pytest.raises(ValueError, match="The sum of the input signal is zero"):
            entropy_sqi(zero_signal)

    def test_entropy_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            entropy_sqi("invalid")


class TestSignalToNoiseSqi:
    def test_on_signal_to_noise_sqi(self):
        random_signal = np.random.randn(100)
        result = signal_to_noise_sqi(random_signal)
        assert result > 0

    def test_signal_to_noise_sqi_edge_case(self):
        zero_signal = np.zeros(10)
        assert signal_to_noise_sqi(zero_signal) == 0

    def test_signal_to_noise_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            signal_to_noise_sqi("invalid")


class TestZeroCrossingRateSqi:
    def test_on_zero_crossings_rate_sqi(self):
        random_signal = np.random.randn(100)
        result = zero_crossings_rate_sqi(random_signal)
        assert result > 0

    def test_zero_crossings_rate_sqi_edge_case(self):
        zero_signal = np.zeros(10)
        assert zero_crossings_rate_sqi(zero_signal) == 0

    def test_zero_crossings_rate_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            zero_crossings_rate_sqi("invalid")


class TestMeanCrossingRateSqi:
    def test_on_mean_crossing_rate_sqi(self):
        random_signal = np.random.randn(100)
        result = mean_crossing_rate_sqi(random_signal)
        assert result > 0

    def test_mean_crossing_rate_sqi_edge_case(self):
        zero_signal = np.zeros(10)
        assert mean_crossing_rate_sqi(zero_signal) == 0

    def test_mean_crossing_rate_sqi_invalid_inputs(self):
        with pytest.raises(TypeError):
            mean_crossing_rate_sqi("invalid")
