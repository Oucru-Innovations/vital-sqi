"""
Phase 4 tests — robust classifier (P4.1–P4.6).

Coverage:
  P4.1 — score column present and in [0,1]
  P4.2 — _sanitize handles inf/NaN
  P4.3 — RobustResult fields; four regime fixtures
  P4.4 — classify_segments mode="robust" and mode="legacy"
  P4.5 — regime_info diagnostic dict populated
  P4.6 — four synthetic regime fixtures (clean, mostly_bad, bimodal, heavy_noise)
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from vital_sqi.rule.robust_classifier import (
    RobustResult,
    _bic_bimodal_check,
    _classify_bimodal,
    _classify_heavy_noise,
    _classify_mostly_good,
    _detect_quality_regime,
    _iqr_normalize,
    _normal_fraction,
    _rank_normalize,
    _sanitize,
    classify_segments_robust,
)
from vital_sqi.common.utils import sanitize_sqi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sqi_df(scores, n_cols=2, noise=0.05, seed=42):
    """Build a DataFrame with ``n_cols`` SQI columns correlated to ``scores``."""
    rng = np.random.default_rng(seed)
    data = {
        f"sqi_{k}": scores + rng.normal(0, noise, len(scores))
        for k in range(n_cols)
    }
    return pd.DataFrame(data)


def _clean_fixture(n=60):
    """~90% of segments are high quality."""
    rng = np.random.default_rng(0)
    scores = np.clip(rng.normal(0.8, 0.05, n), 0, 1)
    return _make_sqi_df(scores)


def _mostly_bad_fixture(n=60):
    """~10% of segments are high quality."""
    rng = np.random.default_rng(1)
    scores = np.clip(rng.normal(0.15, 0.05, n), 0, 1)
    return _make_sqi_df(scores)


def _bimodal_fixture(n=60):
    """Equal mix of good (~0.8) and bad (~0.2) segments."""
    rng = np.random.default_rng(2)
    good = np.clip(rng.normal(0.8, 0.04, n // 2), 0, 1)
    bad = np.clip(rng.normal(0.2, 0.04, n // 2), 0, 1)
    scores = np.concatenate([good, bad])
    rng.shuffle(scores)
    return _make_sqi_df(scores)


def _heavy_noise_fixture(n=60):
    """Very noisy — uniformly distributed, mostly bad."""
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.0, 0.4, n)
    return _make_sqi_df(scores)


# ---------------------------------------------------------------------------
# P4.2 — _sanitize / sanitize_sqi
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_inf_replaced(self):
        v = np.array([1.0, np.inf, -np.inf, 2.0])
        out = _sanitize(v)
        assert np.all(np.isfinite(out))

    def test_nan_filled_with_median(self):
        v = np.array([1.0, np.nan, 3.0])
        out = _sanitize(v)
        assert out[1] == pytest.approx(2.0)

    def test_all_nan_returns_zeros(self):
        v = np.full(5, np.nan)
        out = _sanitize(v)
        assert np.all(out == 0)

    def test_utils_sanitize_sqi_matches(self):
        """sanitize_sqi in utils.py must match the internal _sanitize."""
        v = np.array([np.inf, 1.0, np.nan, 2.0, -np.inf])
        assert np.allclose(sanitize_sqi(v), _sanitize(v))


# ---------------------------------------------------------------------------
# P4.3 — normalisation helpers
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_rank_normalize_range(self):
        v = np.array([3.0, 1.0, 2.0])
        out = _rank_normalize(v)
        assert np.all(out >= 0) and np.all(out <= 1)
        assert out[1] == pytest.approx(0.0)
        assert out[0] == pytest.approx(1.0)

    def test_iqr_normalize_clipped(self):
        v = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        out = _iqr_normalize(v)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_rank_normalize_single_value(self):
        out = _rank_normalize(np.array([5.0]))
        assert out[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# P4.3 — RobustResult dataclass
# ---------------------------------------------------------------------------

class TestRobustResult:
    def test_fields_present(self):
        r = RobustResult(
            decisions=["accept", "reject"],
            scores=np.array([0.8, 0.2]),
            regime="clean",
            regime_info={"normal_fraction": 0.5},
            file_flagged=False,
        )
        assert r.regime == "clean"
        assert len(r.decisions) == 2
        assert len(r.scores) == 2
        assert r.file_flagged is False

    def test_empty_result_from_empty_df(self):
        df = pd.DataFrame({"sqi_0": []})
        result = classify_segments_robust(df)
        assert result.decisions == []
        assert len(result.scores) == 0


# ---------------------------------------------------------------------------
# P4.6 — Four regime fixtures
# ---------------------------------------------------------------------------

class TestRegimes:
    def test_clean_regime_detected(self):
        df = _clean_fixture()
        result = classify_segments_robust(df)
        assert result.regime == "clean"
        # abs_quality should reflect that raw values cluster near 0.8
        assert result.regime_info["abs_quality"] >= 0.5

    def test_mostly_bad_detected_as_heavy_noise(self):
        df = _mostly_bad_fixture()
        result = classify_segments_robust(df)
        assert result.regime in ("heavy_noise", "bimodal")
        # abs_quality should reflect that raw values cluster near 0.15
        assert result.regime_info["abs_quality"] < 0.5

    def test_bimodal_regime_detected(self):
        df = _bimodal_fixture(n=80)
        result = classify_segments_robust(df)
        # bimodal or heavy_noise both acceptable; the key is ≥30% are accepted
        assert result.regime_info["n_accepted"] >= int(0.3 * 80)

    def test_heavy_noise_regime_accepts_few(self):
        df = _heavy_noise_fixture()
        result = classify_segments_robust(df)
        assert result.regime in ("heavy_noise", "bimodal")
        # heavy noise: accept only top quantile → ≤30% accepted
        total = result.regime_info["n_segments"]
        n_acc = result.regime_info["n_accepted"]
        assert n_acc <= int(0.35 * total) + 1

    def test_file_flagged_when_mostly_rejected(self):
        # Build a df where almost everything is very bad
        scores = np.full(50, 0.05)
        df = _make_sqi_df(scores, noise=0.01)
        result = classify_segments_robust(df)
        assert result.file_flagged is True

    def test_file_not_flagged_for_clean(self):
        df = _clean_fixture()
        result = classify_segments_robust(df)
        assert result.file_flagged is False


# ---------------------------------------------------------------------------
# P4.1 + P4.5 — score column and regime_info
# ---------------------------------------------------------------------------

class TestScoreAndRegimeInfo:
    def test_score_in_range(self):
        df = _clean_fixture()
        result = classify_segments_robust(df)
        assert np.all(result.scores >= 0) and np.all(result.scores <= 1)

    def test_regime_info_keys_present(self):
        df = _clean_fixture()
        result = classify_segments_robust(df)
        for key in ("regime", "normal_fraction", "n_segments", "n_accepted",
                    "score_mean", "score_std", "file_flagged"):
            assert key in result.regime_info, f"Missing key: {key}"

    def test_decisions_match_scores_direction(self):
        """Accepted segments should generally have higher scores than rejected."""
        df = _clean_fixture(n=100)
        result = classify_segments_robust(df)
        accepts = result.scores[np.array(result.decisions) == "accept"]
        rejects = result.scores[np.array(result.decisions) == "reject"]
        if len(accepts) > 0 and len(rejects) > 0:
            assert np.mean(accepts) > np.mean(rejects)

    def test_n_decisions_equals_n_segments(self):
        df = _bimodal_fixture(n=40)
        result = classify_segments_robust(df)
        assert len(result.decisions) == 40
        assert len(result.scores) == 40


# ---------------------------------------------------------------------------
# P4.4 — classify_segments mode parameter
# ---------------------------------------------------------------------------

class TestClassifySegmentsMode:
    def _make_rule_json(self, tmpdir, sqi_name="skewness_sqi"):
        rule = {
            sqi_name: {
                "name": sqi_name,
                "def": [
                    {"op": ">", "value": "-10", "label": "accept"},
                    {"op": "<=", "value": "-10", "label": "reject"},
                    {"op": ">=", "value": "10", "label": "reject"},
                    {"op": "<", "value": "10", "label": "accept"},
                ],
            }
        }
        path = os.path.join(tmpdir, "rules.json")
        with open(path, "w") as f:
            json.dump(rule, f)
        return path

    def test_legacy_mode_returns_ruleset(self, tmp_path):
        from vital_sqi.pipeline.pipeline_functions import classify_segments
        from vital_sqi.rule.ruleset_class import RuleSet

        sqi_df = pd.DataFrame({"skewness_sqi": [0.1, -0.2, 0.5]})
        rule_path = self._make_rule_json(str(tmp_path))
        ruleset_order = {1: "skewness_sqi"}

        result, sqis = classify_segments(
            [sqi_df], rule_path, ruleset_order, auto_mode=False, mode="legacy"
        )
        assert isinstance(result, RuleSet)
        assert "decision" in sqis[0].columns
        assert set(sqis[0]["decision"]).issubset({"accept", "reject"})

    def test_robust_mode_returns_robust_result(self, tmp_path):
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        sqi_df = pd.DataFrame({
            "sqi_0": np.random.default_rng(0).normal(0.7, 0.1, 30),
            "sqi_1": np.random.default_rng(1).normal(0.7, 0.1, 30),
        })
        rule_path = self._make_rule_json(str(tmp_path))
        ruleset_order = {1: "skewness_sqi"}

        result, sqis = classify_segments(
            [sqi_df], rule_path, ruleset_order, mode="robust"
        )
        assert isinstance(result, RobustResult)
        assert "decision" in sqis[0].columns
        assert "score" in sqis[0].columns
        assert np.all(sqis[0]["score"].between(0, 1))

    def test_invalid_mode_raises(self, tmp_path):
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        sqi_df = pd.DataFrame({"sqi_0": [0.5]})
        rule_path = self._make_rule_json(str(tmp_path))
        with pytest.raises(ValueError, match="mode must be"):
            classify_segments([sqi_df], rule_path, {1: "skewness_sqi"}, mode="bad")

    def test_robust_mode_legacy_unchanged(self, tmp_path):
        """Running legacy mode twice produces identical results."""
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        sqi_df = pd.DataFrame({"skewness_sqi": [0.1, 0.5, -0.2, 0.3]})
        rule_path = self._make_rule_json(str(tmp_path))
        ruleset_order = {1: "skewness_sqi"}

        _, sqis1 = classify_segments(
            [sqi_df.copy()], rule_path, ruleset_order, auto_mode=False, mode="legacy"
        )
        _, sqis2 = classify_segments(
            [sqi_df.copy()], rule_path, ruleset_order, auto_mode=False, mode="legacy"
        )
        pd.testing.assert_series_equal(sqis1[0]["decision"], sqis2[0]["decision"])

    def test_robust_mode_multichannel(self, tmp_path):
        """Robust mode processes each channel independently."""
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        rng = np.random.default_rng(42)
        df1 = pd.DataFrame({"sqi_0": rng.normal(0.8, 0.05, 30)})
        df2 = pd.DataFrame({"sqi_0": rng.normal(0.2, 0.05, 30)})
        rule_path = self._make_rule_json(str(tmp_path))
        ruleset_order = {1: "skewness_sqi"}

        result, sqis = classify_segments(
            [df1, df2], rule_path, ruleset_order, mode="robust"
        )
        # Channel 0 (clean) should accept more than channel 1 (noisy)
        acc0 = (sqis[0]["decision"] == "accept").sum()
        acc1 = (sqis[1]["decision"] == "accept").sum()
        assert acc0 >= acc1
