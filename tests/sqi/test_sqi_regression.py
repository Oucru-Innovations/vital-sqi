"""
P5.B3 — SQI numerical regression test matrix.

Each SQI is exercised on a deterministic synthetic signal with a known
expected output value.  The purpose is to catch formula regressions —
not just type/shape correctness.

Conventions
-----------
- All synthetic signals are generated with a fixed seed so results are
  reproducible across platforms.
- Expected values are computed analytically or from a reference
  implementation and embedded here as ground truth.
- Tolerances are `rel=1e-3` unless the SQI is inherently imprecise
  (e.g. frequency-domain metrics on short signals).
"""

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

# SQIs under test
from vital_sqi.sqi.standard_sqi import (
    perfusion_sqi,
    kurtosis_sqi,
    skewness_sqi,
    entropy_sqi,
    signal_to_noise_sqi,
    zero_crossings_rate_sqi,
    mean_crossing_rate_sqi,
)
from vital_sqi.sqi.hrv_sqi import (
    nn_mean_sqi,
    sdnn_sqi,
    sdsd_sqi,
    rmssd_sqi,
    poincare_features_sqi,
)
from vital_sqi.sqi.dtw_sqi import dtw_distance


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NN = [800.0, 810.0, 780.0, 820.0, 790.0, 830.0, 800.0, 810.0]
NN_arr = np.array(NN)


# ---------------------------------------------------------------------------
# standard_sqi
# ---------------------------------------------------------------------------

class TestPerfusionSqiRegression:
    def test_known_value(self):
        # x = [1..10]: mean = 5.5
        # y = [2..11]: max-min = 9.0
        # PI = (9 / 5.5) * 100
        x = np.arange(1.0, 11.0)
        y = np.arange(2.0, 11.0)  # 9 elements; max-min = 7.0 → PI = (7/5.5)*100
        # Use simple fixed values for clarity
        x = np.array([5.0, 5.0, 5.0])   # mean = 5
        y = np.array([0.0, 1.0, 0.0])    # max-min = 1
        expected = (1.0 / 5.0) * 100     # = 20.0
        assert perfusion_sqi(x, y) == pytest.approx(expected, rel=1e-6)

    def test_zero_mean_guard(self):
        assert np.isnan(perfusion_sqi(np.zeros(10), np.ones(10)))


class TestKurtosisSqiRegression:
    def test_sine_wave_kurtosis(self):
        # Pure sine has kurtosis = -1.5 (Fisher, excess)
        t = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        s = np.sin(t)
        result = kurtosis_sqi(s)
        assert result == pytest.approx(-1.5, rel=1e-2)

    def test_gaussian_kurtosis_near_zero(self):
        rng = np.random.default_rng(42)
        s = rng.normal(0, 1, 100_000)
        result = kurtosis_sqi(s)
        assert result == pytest.approx(0.0, abs=0.1)


class TestSkewnessSqiRegression:
    def test_symmetric_signal_zero_skewness(self):
        # Perfect symmetric distribution → skewness = 0
        rng = np.random.default_rng(99)
        s = rng.normal(0, 1, 100_000)
        assert skewness_sqi(s) == pytest.approx(0.0, abs=0.05)

    def test_right_skewed_positive(self):
        # Log-normal is right-skewed
        rng = np.random.default_rng(1)
        s = rng.lognormal(0, 1, 1000)
        assert skewness_sqi(s) > 0

    def test_left_skewed_negative(self):
        rng = np.random.default_rng(2)
        s = -rng.lognormal(0, 1, 1000)
        assert skewness_sqi(s) < 0


class TestEntropySqiRegression:
    def test_uniform_distribution_maximum_entropy(self):
        # Uniformly distributed signal → highest entropy relative to a bimodal one
        rng = np.random.default_rng(0)
        uniform = rng.integers(0, 100, 10000).astype(float)
        bimodal = np.concatenate([
            rng.integers(0, 10, 5000),
            rng.integers(90, 100, 5000),
        ]).astype(float)
        assert entropy_sqi(uniform) > entropy_sqi(bimodal)

    def test_constant_signal_zero_entropy(self):
        assert entropy_sqi(np.full(50, 3.14)) == pytest.approx(0.0, abs=1e-10)

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError, match="empty"):
            entropy_sqi(np.array([]))


class TestZeroCrossingSqiRegression:
    def test_sine_wave_rate(self):
        # 1 Hz sine at fs=100 → 2 zero crossings per second → rate = 2/100 = 0.02
        t = np.linspace(0, 1, 101, endpoint=False)
        s = np.sin(2 * np.pi * 1 * t)
        rate = zero_crossings_rate_sqi(s)
        # 100 diffs, ~2 sign changes → 0.02
        assert rate == pytest.approx(2 / 100, abs=0.01)

    def test_mean_crossing_dc_offset(self):
        # Sine with large DC offset: zero_crossings = 0, mean_crossing > 0
        t = np.linspace(0, 1, 100, endpoint=False)
        s = np.sin(2 * np.pi * t) + 1000
        assert zero_crossings_rate_sqi(s) == pytest.approx(0.0, abs=1e-10)
        assert mean_crossing_rate_sqi(s) > 0


class TestSignalToNoiseSqiRegression:
    def test_pure_signal_high_snr(self):
        # Constant signal: mean = 5, std = 0 → SNR = 0 by convention (no noise)
        s = np.full(100, 5.0)
        assert signal_to_noise_sqi(s) == pytest.approx(0.0)

    def test_known_snr(self):
        # mean = 2, std = 1 → SNR = 2
        rng = np.random.default_rng(7)
        s = rng.normal(loc=2.0, scale=1.0, size=100_000)
        assert signal_to_noise_sqi(s) == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# hrv_sqi
# ---------------------------------------------------------------------------

class TestNNMeanSqiRegression:
    def test_known_mean(self):
        assert nn_mean_sqi(NN) == pytest.approx(float(np.mean(NN_arr)), rel=1e-6)


class TestSDNNSqiRegression:
    def test_known_sdnn(self):
        expected = float(np.std(NN_arr, ddof=1))
        assert sdnn_sqi(NN) == pytest.approx(expected, rel=1e-6)


class TestSDSDSqiRegression:
    def test_known_sdsd(self):
        expected = float(np.std(np.diff(NN_arr), ddof=1))
        assert sdsd_sqi(NN) == pytest.approx(expected, rel=1e-6)


class TestRMSSDSqiRegression:
    def test_known_rmssd(self):
        diffs = np.diff(NN_arr)
        expected = float(np.sqrt(np.mean(diffs ** 2)))
        assert rmssd_sqi(NN) == pytest.approx(expected, rel=1e-6)

    def test_rmssd_ne_sdsd_for_nonzero_mean_diff(self):
        # Monotonically increasing intervals: RMSSD ≠ SDSD
        intervals = [800.0, 820.0, 840.0, 860.0, 880.0]
        arr = np.array(intervals)
        diffs = np.diff(arr)  # [20, 20, 20, 20]
        rmssd = float(np.sqrt(np.mean(diffs ** 2)))
        sdsd = float(np.std(diffs, ddof=1))
        assert rmssd > sdsd  # RMSSD = 20.0, SDSD = 0.0
        assert rmssd_sqi(intervals) == pytest.approx(rmssd, rel=1e-6)


class TestPoincareRegression:
    def test_sd2_no_nan(self):
        result = poincare_features_sqi(NN)
        assert np.isfinite(result["sd1"])
        assert np.isfinite(result["sd2"])
        assert result["sd2"] >= 0

    def test_sd1_formula(self):
        # sd1 = std(diff) / sqrt(2)
        diffs = np.diff(NN_arr)
        expected_sd1 = float(np.std(diffs, ddof=1)) / np.sqrt(2)
        result = poincare_features_sqi(NN)
        assert result["sd1"] == pytest.approx(expected_sd1, rel=1e-5)


# ---------------------------------------------------------------------------
# dtw
# ---------------------------------------------------------------------------

class TestDTWDistanceRegression:
    def test_identical_sequences_zero_distance(self):
        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert dtw_distance(s, s) == pytest.approx(0.0, abs=1e-10)

    def test_distance_is_symmetric(self):
        s1 = np.array([1.0, 2.0, 3.0])
        s2 = np.array([1.5, 2.5, 2.0])
        assert dtw_distance(s1, s2) == pytest.approx(dtw_distance(s2, s1), rel=1e-6)

    def test_known_small_example(self):
        # seq1 = [1, 2, 3], seq2 = [1, 2, 3] → 0
        # seq1 = [0], seq2 = [1] → 1
        assert dtw_distance(np.array([0.0]), np.array([1.0])) == pytest.approx(1.0)

    def test_positive_for_different_sequences(self):
        s1 = np.zeros(10)
        s2 = np.ones(10)
        assert dtw_distance(s1, s2) > 0
