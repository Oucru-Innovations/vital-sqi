"""Extra coverage for classify_segments — error paths and rare branches.

The happy paths (manual / quantile / tune on synthetic data) are
covered in ``test_auto_threshold.py``.  These tests target the gaps:

* invalid ``auto_mode`` values
* file-not-found errors
* unknown-rule errors
* the "every rule was degenerate" warning path that hands back an
  all-accept channel
* the manual-mode branch from inside the loop
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vital_sqi.pipeline.pipeline_functions import classify_segments


def _rule_def(lower: float, upper: float):
    return [
        {"op": ">",  "value": f"{lower:.6g}", "label": "accept"},
        {"op": "<=", "value": f"{lower:.6g}", "label": "reject"},
        {"op": ">=", "value": f"{upper:.6g}", "label": "reject"},
        {"op": "<",  "value": f"{upper:.6g}", "label": "accept"},
    ]


@pytest.fixture
def rule_dict_path(tmp_path):
    path = tmp_path / "rule_dict.json"
    payload = {
        "kurtosis_sqi": {"name": "kurtosis_sqi", "def": _rule_def(-1.0, 1.0)},
        "perfusion_sqi": {"name": "perfusion_sqi", "def": _rule_def(0.0, 10.0)},
    }
    path.write_text(json.dumps(payload))
    return path


class TestAutoModeValidation:
    def test_invalid_auto_mode_string_raises(self, rule_dict_path):
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="auto_mode"):
            classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "kurtosis_sqi"},
                auto_mode="nope",
            )

    def test_invalid_auto_mode_type_raises(self, rule_dict_path):
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="auto_mode"):
            classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "kurtosis_sqi"},
                auto_mode=42,
            )

    def test_invalid_mode_kwarg_raises(self, rule_dict_path):
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.1, 0.2, 0.3]})
        with pytest.raises(ValueError, match="mode must be"):
            classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "kurtosis_sqi"},
                mode="banana",
            )


class TestMissingFiles:
    def test_missing_rule_dict_raises(self, tmp_path):
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.1, 0.2]})
        with pytest.raises(FileNotFoundError, match="Rule dictionary"):
            classify_segments(
                [sqi_df.copy()],
                str(tmp_path / "missing.json"),
                {1: "kurtosis_sqi"},
            )

    def test_unknown_rule_name_raises(self, rule_dict_path):
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.1, 0.2]})
        with pytest.raises(KeyError, match="not found"):
            classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "made_up_sqi"},
            )


class TestManualMode:
    def test_manual_mode_applies_dict_bounds(self, rule_dict_path):
        # In manual mode the band is (-1, 1) from the rule_dict, so 0.5
        # accepts and 5.0 rejects.
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.5, 5.0, -2.0]})
        _, out = classify_segments(
            [sqi_df.copy()],
            str(rule_dict_path),
            {1: "kurtosis_sqi"},
            auto_mode="manual",
        )
        decisions = list(out[0]["decision"])
        assert decisions == ["accept", "reject", "reject"]


class TestDegenerateChannelFallback:
    def test_all_degenerate_rules_emit_warning_and_accept_all(self, rule_dict_path):
        # Constant SQI column → quantile_band returns None for every row.
        # The expected behaviour is a warning + every segment marked "accept".
        sqi_df = pd.DataFrame({"kurtosis_sqi": [0.5, 0.5, 0.5, 0.5]})
        with pytest.warns(UserWarning):
            _, out = classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "kurtosis_sqi"},
                auto_mode="quantile",
            )
        decisions = list(out[0]["decision"])
        assert decisions == ["accept", "accept", "accept", "accept"]

    def test_skipped_rules_warning_when_partial_degenerate(self, rule_dict_path):
        # One usable rule, one constant rule.  Should warn about the
        # constant one but still classify on the usable one.
        rng = np.random.default_rng(0)
        sqi_df = pd.DataFrame({
            "kurtosis_sqi": rng.normal(0, 0.3, 100),
            "perfusion_sqi": np.full(100, 5.0),  # constant → dropped
        })
        with pytest.warns(UserWarning, match="degenerate"):
            _, out = classify_segments(
                [sqi_df.copy()],
                str(rule_dict_path),
                {1: "kurtosis_sqi", 2: "perfusion_sqi"},
                auto_mode="quantile",
            )
        # Should still have a decision column.
        assert "decision" in out[0].columns
        assert len(out[0]) == 100
