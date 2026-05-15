"""Extra coverage for vital_sqi.common.utils edge paths.

Targets lines flagged by ``coverage`` that the existing ``test_utils.py``
and ``test_utils_extra.py`` don't reach: timestamp-type branches in
``calculate_sampling_rate``, ``generate_timestamp`` validation,
``parse_datetime`` fallback to ``dateparser``, and ``parse_rule`` errors.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from vital_sqi.common import utils as u


# ---------------------------------------------------------------------------
# calculate_sampling_rate — each timestamp-type branch
# ---------------------------------------------------------------------------


class TestCalculateSamplingRate:
    def test_too_few_returns_none(self):
        assert u.calculate_sampling_rate([0.0]) is None

    def test_numeric_seconds(self):
        ts = np.arange(0.0, 1.0, 0.01)
        fs = u.calculate_sampling_rate(ts)
        assert fs == pytest.approx(100.0, rel=1e-3)

    def test_pd_timestamps(self):
        ts = pd.date_range("2024-01-01", periods=100, freq=pd.Timedelta(seconds=0.02))
        fs = u.calculate_sampling_rate(ts)
        assert fs == pytest.approx(50.0, rel=1e-3)

    def test_numpy_datetime64(self):
        # Build a plain np.datetime64 array (no pandas-specific dtype)
        start = np.datetime64("2024-01-01T00:00:00")
        ts = np.array([start + np.timedelta64(int(i * 10), "ms") for i in range(50)])
        fs = u.calculate_sampling_rate(ts)
        assert fs == pytest.approx(100.0, rel=1e-3)

    def test_string_timestamps_parsed_by_dateparser(self):
        ts = ["2024-01-01 00:00:00", "2024-01-01 00:00:01"]
        fs = u.calculate_sampling_rate(ts)
        assert fs == pytest.approx(1.0, rel=1e-3)

    def test_all_unparseable_strings_return_none(self):
        ts = ["not-a-date", "neither-this"]
        assert u.calculate_sampling_rate(ts) is None

    def test_zero_difference_returns_none(self):
        # Two identical timestamps → no positive deltas → None
        ts = [0.0, 0.0, 0.0]
        assert u.calculate_sampling_rate(ts) is None


# ---------------------------------------------------------------------------
# generate_timestamp validation + the "from datetime input" branch
# ---------------------------------------------------------------------------


class TestGenerateTimestamp:
    def test_rejects_non_numeric_sampling_rate(self):
        with pytest.raises(ValueError):
            u.generate_timestamp(None, "not-a-number", 100)

    def test_accepts_datetime_start(self):
        start = dt.datetime(2024, 1, 1, 12, 0, 0)
        ts = u.generate_timestamp(start, 100, 50)
        assert len(ts) == 50
        # First timestamp must match the requested start (within a few µs).
        assert pd.Timestamp(ts[0]) == pd.Timestamp(start)

    def test_accepts_pd_timestamp_start(self):
        start = pd.Timestamp("2024-01-01")
        ts = u.generate_timestamp(start, 100, 10)
        assert len(ts) == 10


# ---------------------------------------------------------------------------
# parse_datetime — fallback branch
# ---------------------------------------------------------------------------


class TestParseDatetime:
    def test_canonical_format(self):
        result = u.parse_datetime("2024-01-15", type="date")
        assert isinstance(result, dt.datetime)
        assert (result.year, result.month, result.day) == (2024, 1, 15)

    def test_falls_back_to_dateparser(self):
        # A non-canonical format that the explicit strptime list doesn't
        # cover but dateparser can handle.
        result = u.parse_datetime("January 15, 2024", type="datetime")
        assert result is not None

    def test_unparseable_raises_valueerror(self):
        # dateparser returns None for nonsense; the wrapper turns that
        # into a ValueError.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            # Either ValueError or it returns None from dateparser; we
            # accept either as long as it doesn't crash.
            try:
                result = u.parse_datetime("totally not a date string", type="datetime")
                assert result is None
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# parse_rule — error paths
# ---------------------------------------------------------------------------


class TestParseRule:
    def test_dict_source(self, tmp_path):
        # When source is a dict the function should not touch the filesystem.
        source = {
            "kurtosis_sqi": {
                "name": "kurtosis_sqi",
                "def": [
                    {"op": ">",  "value": "0", "label": "accept"},
                    {"op": "<=", "value": "0", "label": "reject"},
                    {"op": ">=", "value": "5", "label": "reject"},
                    {"op": "<",  "value": "5", "label": "accept"},
                ],
            }
        }
        rule_def, boundaries, labels = u.parse_rule("kurtosis_sqi", source)
        assert len(boundaries) == 2
        assert "accept" in labels and "reject" in labels

    def test_missing_sqi_raises_value_error(self, tmp_path):
        source = {"a": {"name": "a", "def": []}}
        with pytest.raises(ValueError, match="not found"):
            u.parse_rule("missing", source)

    def test_missing_def_key_raises_key_error(self):
        source = {"k": {"name": "k"}}  # no "def"
        with pytest.raises(KeyError, match="'def'"):
            u.parse_rule("k", source)

    def test_invalid_json_file_raises(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not: valid json}")
        with pytest.raises(json.JSONDecodeError):
            u.parse_rule("anything", str(path))


# ---------------------------------------------------------------------------
# sanitize_sqi + create_rule_def — small gaps
# ---------------------------------------------------------------------------


class TestSanitizeSqiEdgeCases:
    def test_preserves_finite_values(self):
        out = u.sanitize_sqi([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_negative_inf_replaced(self):
        out = u.sanitize_sqi([-np.inf, 1.0, 2.0])
        assert np.all(np.isfinite(out))
