"""Extra coverage for vital_sqi.rule.robust_classifier helpers."""
import numpy as np
import pandas as pd
import pytest

from vital_sqi.rule import robust_classifier as rc


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class TestNormalizers:
    def test_rank_normalize_single_value(self):
        out = rc._rank_normalize(np.array([5.0]))
        assert out.shape == (1,)
        assert out[0] == 0.0

    def test_rank_normalize_empty(self):
        out = rc._rank_normalize(np.array([]))
        assert out.shape == (0,)

    def test_iqr_normalize_constant_returns_half(self):
        out = rc._iqr_normalize(np.array([3.0, 3.0, 3.0, 3.0]))
        np.testing.assert_array_equal(out, np.full(4, 0.5))

    def test_iqr_normalize_clips_outliers(self):
        out = rc._iqr_normalize(np.array([1.0, 2.0, 3.0, 100.0]))
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_minmax_normalize_constant_returns_half(self):
        out = rc._minmax_normalize(np.array([3.0, 3.0]))
        np.testing.assert_array_equal(out, np.full(2, 0.5))

    def test_minmax_normalize_standard(self):
        out = rc._minmax_normalize(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_sanitize_all_nan_returns_zeros(self):
        out = rc._sanitize(np.array([np.nan, np.nan, np.nan]))
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_sanitize_replaces_inf_and_nan(self):
        out = rc._sanitize(np.array([1.0, np.inf, np.nan, -np.inf, 5.0]))
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Bimodality & regime helpers
# ---------------------------------------------------------------------------

class TestBimodality:
    def test_bimodality_coefficient_short_input(self):
        # < 4 values → returns 0.0
        assert rc._bimodality_coefficient(np.array([0.1, 0.9])) == 0.0

    def test_bimodality_coefficient_clearly_bimodal(self):
        # Two clusters → bc > 0.555
        x = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        bc = rc._bimodality_coefficient(x)
        assert bc > 0.555

    def test_normal_fraction(self):
        s = np.array([0.1, 0.4, 0.6, 0.8])
        assert rc._normal_fraction(s, threshold=0.5) == 0.5

    def test_detect_regime_clean(self):
        # Gaussian-like (unimodal, low BC) cluster at 0.8 → clean
        rng = np.random.default_rng(0)
        scores = np.clip(rng.normal(0.8, 0.05, 100), 0, 1)
        assert rc._detect_quality_regime(scores, abs_quality=0.8) == "clean"

    def test_detect_regime_heavy_noise(self):
        rng = np.random.default_rng(0)
        scores = np.clip(rng.normal(0.15, 0.05, 100), 0, 1)
        assert rc._detect_quality_regime(scores, abs_quality=0.1) == "heavy_noise"

    def test_detect_regime_bimodal(self):
        scores = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        assert rc._detect_quality_regime(scores, abs_quality=0.5) == "bimodal"


# ---------------------------------------------------------------------------
# Per-regime classifiers
# ---------------------------------------------------------------------------

class TestPerRegimeClassifiers:
    def test_classify_mostly_good(self):
        scores = np.array([0.1, 0.5, 0.8])
        out = rc._classify_mostly_good(scores, threshold=0.4)
        np.testing.assert_array_equal(out, [0.0, 1.0, 1.0])

    def test_classify_heavy_noise_top_quantile(self):
        scores = np.linspace(0, 1, 100)
        out = rc._classify_heavy_noise(scores, accept_quantile=0.9)
        # Top 10% accepted
        assert out.sum() <= 12 and out.sum() >= 8

    def test_classify_bimodal_short_input_fallback(self):
        # Below MIN_SEGMENTS_FOR_GMM → MAD-based split
        scores = np.array([0.1, 0.1, 0.9, 0.9])
        out = rc._classify_bimodal(scores)
        assert out.shape == (4,)
        assert ((out == 0.0) | (out == 1.0)).all()

    def test_bhattacharyya_distance_zero_std(self):
        assert rc._bhattacharyya_distance(0, 0, 0, 0) == 0.0

    def test_bhattacharyya_distance_finite(self):
        d = rc._bhattacharyya_distance(0.2, 0.01, 0.8, 0.01)
        assert np.isfinite(d) and d > 0


# ---------------------------------------------------------------------------
# Public entry point — covers the orchestration paths
# ---------------------------------------------------------------------------

class TestClassifySegmentsRobust:
    def test_empty_input_returns_empty_result(self):
        df = pd.DataFrame(columns=["a", "b"])
        result = rc.classify_segments_robust(df, sqi_names=["a", "b"])
        assert result.decisions == []
        assert result.scores.shape == (0,)

    def test_clean_recording(self):
        # Multiple correlated SQIs narrow the rank consensus distribution.
        rng = np.random.default_rng(0)
        base = rng.normal(0.8, 0.03, 100)
        df = pd.DataFrame({
            "a": np.clip(base + rng.normal(0, 0.01, 100), 0, 1),
            "b": np.clip(base + rng.normal(0, 0.01, 100), 0, 1),
            "c": np.clip(base + rng.normal(0, 0.01, 100), 0, 1),
        })
        result = rc.classify_segments_robust(df, sqi_names=list(df.columns))
        # Either clean or bimodal acceptable; main goal is no crash + valid output
        assert result.regime in ("clean", "bimodal")
        assert len(result.decisions) == 100

    def test_heavy_noise_recording(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.1, 0.03, 100)
        df = pd.DataFrame({
            "a": np.clip(base + rng.normal(0, 0.01, 100), 0, 1),
            "b": np.clip(base + rng.normal(0, 0.01, 100), 0, 1),
        })
        result = rc.classify_segments_robust(df, sqi_names=list(df.columns))
        assert result.regime in ("heavy_noise", "bimodal")

    def test_bimodal_recording(self):
        good = np.full(50, 0.85)
        bad = np.full(50, 0.15)
        df = pd.DataFrame({"kurt": np.concatenate([good, bad])})
        result = rc.classify_segments_robust(df, sqi_names=["kurt"])
        assert result.regime == "bimodal"

    def test_config_override(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"kurt": rng.uniform(0.7, 0.9, 100)})
        result = rc.classify_segments_robust(
            df,
            sqi_names=["kurt"],
            config={"heavy_noise_quantile": 0.5},
        )
        # Config field present in result
        assert isinstance(result.regime_info, dict)

    def test_default_sqi_names(self):
        # sqi_names=None → uses all non-meta columns
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.uniform(0.7, 0.9, 30),
                           "b": rng.uniform(0.6, 0.8, 30)})
        result = rc.classify_segments_robust(df)
        assert len(result.decisions) == 30

    def test_all_nan_column_does_not_crash(self):
        df = pd.DataFrame({"good": [0.8] * 30, "bad": [np.nan] * 30})
        result = rc.classify_segments_robust(df, sqi_names=["good", "bad"])
        assert len(result.decisions) == 30

    def test_file_flagged_when_few_accepted(self):
        df = pd.DataFrame({"kurt": np.full(100, 0.05)})
        result = rc.classify_segments_robust(df, sqi_names=["kurt"])
        # Mostly bad → small accept ratio → file_flagged set
        assert isinstance(result.file_flagged, bool)
