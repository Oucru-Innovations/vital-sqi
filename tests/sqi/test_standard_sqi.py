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
    clipping_sqi,
    baseline_wander_sqi,
    spectral_snr_sqi,
)


class TestPerfusionSqi:
    def test_on_perfusion_sqi(self):
        raw_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        filtered_signal = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        result = perfusion_sqi(raw_signal, filtered_signal)
        assert result > 0
        assert result < np.inf

    def test_perfusion_sqi_zero_mean_returns_nan(self):
        # AC-coupled / high-pass filtered signals have ~zero mean
        raw_signal = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        filtered_signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        result = perfusion_sqi(raw_signal, filtered_signal)
        assert np.isnan(result)

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
        rng = np.random.default_rng(42)
        high_entropy_signal = rng.integers(0, 10, 100).astype(float)
        result = entropy_sqi(high_entropy_signal)
        assert result > 0
        # Maximum possible entropy for the number of histogram bins is bounded
        assert np.isfinite(result)

    def test_entropy_constant_signal_is_zero(self):
        # A constant signal has exactly one histogram bin → zero entropy
        constant = np.full(20, 5.0)
        result = entropy_sqi(constant)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_entropy_sqi_zero_sum(self):
        # Flat/constant signal has zero entropy — should return 0.0, not raise
        zero_signal = np.zeros(10)
        assert entropy_sqi(zero_signal) == 0.0

    def test_entropy_sqi_invalid_inputs(self):
        with pytest.raises((TypeError, ValueError)):
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


# ---------------------------------------------------------------------------
# clipping_sqi
# ---------------------------------------------------------------------------

class TestClippingSqi:
    def test_clean_signal_near_zero(self):
        s = np.sin(np.linspace(0, 2 * np.pi, 500))
        assert clipping_sqi(s) < 0.05

    def test_fully_clipped_signal(self):
        s = np.ones(100)
        assert clipping_sqi(s) == 0.0  # constant → range==0

    def test_hard_clipped_signal(self):
        s = np.linspace(-1, 1, 200)
        s[:20] = -1.0
        s[-20:] = 1.0
        assert clipping_sqi(s) > 0.1

    def test_empty_returns_nan(self):
        assert np.isnan(clipping_sqi([]))

    def test_return_type_is_float(self):
        assert isinstance(clipping_sqi(np.random.randn(100)), float)


# ---------------------------------------------------------------------------
# baseline_wander_sqi
# ---------------------------------------------------------------------------

class TestBaselineWanderSqi:
    def test_clean_signal_low_wander(self):
        fs = 100
        t = np.linspace(0, 10, fs * 10)
        s = np.sin(2 * np.pi * 1.0 * t)   # 1 Hz — well above LF threshold
        result = baseline_wander_sqi(s, sampling_rate=fs)
        assert 0.0 <= result <= 1.0

    def test_low_freq_dominated_signal_high_wander(self):
        fs = 100
        t = np.linspace(0, 10, fs * 10)
        s = np.sin(2 * np.pi * 0.1 * t)   # 0.1 Hz — below 0.5 Hz cutoff
        result = baseline_wander_sqi(s, sampling_rate=fs)
        assert result > 0.5

    def test_short_signal_returns_nan(self):
        assert np.isnan(baseline_wander_sqi(np.array([1.0, 2.0]), sampling_rate=100))

    def test_zero_signal_returns_nan(self):
        assert np.isnan(baseline_wander_sqi(np.zeros(500), sampling_rate=100))


# ---------------------------------------------------------------------------
# spectral_snr_sqi
# ---------------------------------------------------------------------------

class TestSpectralSnrSqi:
    def test_tonal_signal_positive_snr(self):
        fs = 100
        t = np.linspace(0, 10, fs * 10)
        s = np.sin(2 * np.pi * 1.2 * t)   # 1.2 Hz — inside PPG band 0.5–4 Hz
        result = spectral_snr_sqi(s, sampling_rate=fs)
        assert result > 0

    def test_out_of_band_signal_negative_snr(self):
        fs = 100
        t = np.linspace(0, 10, fs * 10)
        s = np.sin(2 * np.pi * 20.0 * t)  # 20 Hz — outside PPG band
        result = spectral_snr_sqi(s, sampling_rate=fs)
        assert result < 0

    def test_short_signal_returns_nan(self):
        assert np.isnan(spectral_snr_sqi(np.array([1.0, 2.0]), sampling_rate=100))

    def test_custom_band(self):
        fs = 256
        t = np.linspace(0, 5, fs * 5)
        s = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz inside ECG band [0.5, 40]
        result = spectral_snr_sqi(s, sampling_rate=fs, signal_band=[0.5, 40.0])
        assert isinstance(result, float)
