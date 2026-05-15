"""Unit tests for vital_sqi.rule.auto_threshold."""

from __future__ import annotations

import numpy as np
import pytest

from vital_sqi.rule.auto_threshold import (
    Band,
    DEGENERATE_BAND_HALF_WIDTH,
    per_rule_quantile,
    quantile_band,
    strictest_columns,
    tuned_bands,
)


# ---------------------------------------------------------------------------
# quantile_band
# ---------------------------------------------------------------------------


class TestQuantileBand:
    def test_returns_band_with_expected_quantiles(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 10_000)
        band = quantile_band("kurt", values, lower_pct=0.05, upper_pct=0.95)
        assert band is not None
        # The lower/upper should approximate -1.645 / 1.645 for N(0,1).
        assert band.lower == pytest.approx(-1.645, abs=0.1)
        assert band.upper == pytest.approx(1.645, abs=0.1)

    def test_returns_none_for_constant_column(self):
        assert quantile_band("flat", np.full(100, 0.5)) is None

    def test_returns_none_for_all_nan(self):
        assert quantile_band("dead", np.full(50, np.nan)) is None

    def test_returns_none_for_too_few_values(self):
        assert quantile_band("tiny", [0.5]) is None

    def test_returns_none_when_band_narrower_than_epsilon(self):
        values = np.array([0.5, 0.5 + DEGENERATE_BAND_HALF_WIDTH / 10])
        assert quantile_band("almost_flat", values) is None

    def test_drops_inf_and_nan(self):
        # The function must drop ±inf and NaN before computing quantiles.
        values = np.concatenate([
            np.linspace(-1, 1, 100), [np.inf, -np.inf, np.nan],
        ])
        band = quantile_band("ok", values)
        assert band is not None
        assert np.isfinite(band.lower) and np.isfinite(band.upper)

    def test_rejects_invalid_quantile_ranges(self):
        values = np.linspace(0, 1, 100)
        with pytest.raises(ValueError):
            quantile_band("x", values, lower_pct=0.6, upper_pct=0.9)  # lower > 0.5
        with pytest.raises(ValueError):
            quantile_band("x", values, lower_pct=0.05, upper_pct=0.4)  # upper < 0.5
        with pytest.raises(ValueError):
            quantile_band("x", values, lower_pct=-0.1, upper_pct=0.95)


# ---------------------------------------------------------------------------
# per_rule_quantile + tuned_bands
# ---------------------------------------------------------------------------


class TestPerRuleQuantile:
    @pytest.mark.parametrize("n_rules", [1, 3, 5, 10])
    def test_independence_recovers_target(self, n_rules):
        target = 0.85
        q = per_rule_quantile(target, n_rules=n_rules)
        keep = 1 - 2 * q
        # Joint accept under independence should equal the target exactly.
        assert keep ** n_rules == pytest.approx(target, rel=1e-6)

    def test_quantile_shrinks_as_rules_grow(self):
        # More rules → tighter per-rule trim is needed.
        prev = per_rule_quantile(0.85, 1)
        for n in (2, 3, 5, 10):
            curr = per_rule_quantile(0.85, n)
            assert curr < prev
            prev = curr

    def test_clamps_target_to_valid_range(self):
        # 0 → degenerate clamp; should not raise or return NaN.
        q = per_rule_quantile(0.0, 5)
        assert 0 <= q <= 0.5

    def test_n_rules_validation(self):
        with pytest.raises(ValueError):
            per_rule_quantile(0.85, 0)


class TestTunedBands:
    def test_joint_accept_hits_target(self):
        rng = np.random.default_rng(0)
        cols = {f"sqi_{i}": rng.normal(0, 1, 5_000) for i in range(5)}
        bands = tuned_bands(cols, target_accept_rate=0.85)
        assert len(bands) == 5
        # Empirically verify joint accept rate.
        joint = np.ones(5_000, dtype=bool)
        for band in bands:
            v = cols[band.column]
            joint &= (v > band.lower) & (v < band.upper)
        # Within 3 percentage points of target — the independence
        # approximation is exact under truly independent draws.
        assert joint.mean() == pytest.approx(0.85, abs=0.03)

    def test_drops_degenerate_columns_silently(self):
        rng = np.random.default_rng(1)
        cols = {
            "good": rng.normal(0, 1, 1000),
            "constant": np.full(1000, 0.5),
            "all_nan": np.full(1000, np.nan),
        }
        bands = tuned_bands(cols, target_accept_rate=0.85)
        # Only "good" should survive.
        names = [b.column for b in bands]
        assert names == ["good"]

    def test_returns_empty_when_no_columns_survive(self):
        cols = {
            "a": np.full(100, 1.0),
            "b": np.full(100, 2.0),
        }
        assert tuned_bands(cols) == []

    def test_band_records_metadata(self):
        rng = np.random.default_rng(2)
        cols = {"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)}
        bands = tuned_bands(cols, target_accept_rate=0.85)
        for band in bands:
            assert isinstance(band, Band)
            # Symmetric trim → quantile_lo + quantile_hi == 1.
            assert band.quantile_lo + band.quantile_hi == pytest.approx(1.0)
            assert "auto-tuned" in band.note


# ---------------------------------------------------------------------------
# strictest_columns
# ---------------------------------------------------------------------------


class TestStrictestColumns:
    def test_returns_empty_when_few_rules(self):
        # Need at least 3 to talk about an outlier.
        assert strictest_columns({"a": 5, "b": 50}) == []

    def test_returns_empty_when_no_outlier(self):
        counts = {"a": 10, "b": 11, "c": 9, "d": 10}
        assert strictest_columns(counts) == []

    def test_flags_clear_outlier(self):
        counts = {"a": 10, "b": 11, "c": 9, "d": 100}
        flagged = strictest_columns(counts)
        assert "d" in flagged
        assert flagged[0] == "d"   # worst offender first

    def test_handles_zero_variance(self):
        # If every rule rejects the same number of segments, no outlier
        # exists and the function must not divide by zero.
        assert strictest_columns({"a": 10, "b": 10, "c": 10}) == []


# ---------------------------------------------------------------------------
# Integration with classify_segments
# ---------------------------------------------------------------------------


class TestClassifySegmentsTuneMode:
    def test_tune_mode_hits_target_on_uniform_distribution(self, tmp_path):
        import json
        import pandas as pd
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        rng = np.random.default_rng(7)
        # Five independent gaussian SQIs over 1000 segments.
        n = 1000
        cols = {f"sqi_{i}": rng.normal(0, 1, n) for i in range(5)}
        sqi_df = pd.DataFrame(cols)

        # Write a rule_dict with placeholder bounds (auto-tune ignores them).
        rule_dict_path = tmp_path / "rule_dict.json"
        rule_dict_path.write_text(json.dumps({
            name: {
                "name": name,
                "def": [
                    {"op": ">",  "value": "0",   "label": "accept"},
                    {"op": "<=", "value": "0",   "label": "reject"},
                    {"op": ">=", "value": "1",   "label": "reject"},
                    {"op": "<",  "value": "1",   "label": "accept"},
                ],
            } for name in cols
        }))

        ruleset_order = {i + 1: name for i, name in enumerate(cols)}
        _, out = classify_segments(
            [sqi_df.copy()], str(rule_dict_path), ruleset_order,
            auto_mode="tune", target_accept_rate=0.80,
        )
        decisions = list(out[0]["decision"])
        accept_rate = sum(1 for d in decisions if d == "accept") / len(decisions)
        # Under independent draws the realised rate should land within
        # 5 percentage points of the target.
        assert accept_rate == pytest.approx(0.80, abs=0.05)

    def test_quantile_mode_unchanged_on_legacy_signature(self, tmp_path):
        # Calling classify_segments(auto_mode=True, lower_bound, upper_bound)
        # exactly as before must still work.  Regression guard for the API.
        import json, pandas as pd
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        rng = np.random.default_rng(3)
        sqi_df = pd.DataFrame({"kurtosis_sqi": rng.normal(0, 1, 500)})
        rule_dict_path = tmp_path / "rule_dict.json"
        rule_dict_path.write_text(json.dumps({
            "kurtosis_sqi": {
                "name": "kurtosis_sqi",
                "def": [
                    {"op": ">",  "value": "-1", "label": "accept"},
                    {"op": "<=", "value": "-1", "label": "reject"},
                    {"op": ">=", "value": "1",  "label": "reject"},
                    {"op": "<",  "value": "1",  "label": "accept"},
                ],
            }
        }))
        _, out = classify_segments(
            [sqi_df.copy()],
            str(rule_dict_path),
            {1: "kurtosis_sqi"},
            auto_mode=True, lower_bound=0.05, upper_bound=0.95,
        )
        assert "decision" in out[0].columns
        assert len(out[0]) == 500
