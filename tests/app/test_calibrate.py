"""Unit tests for the Calibrate view — layout, payload round-trip, atomic save."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from dash import dcc, html

from vital_sqi.app.views import calibrate as cal


def _walk(root):
    yield root
    children = getattr(root, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            yield from _walk(child)


def _ids(root):
    return {getattr(n, "id", None) for n in _walk(root) if getattr(n, "id", None)}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_required_ids_present(self):
        expected = {
            "calibrate-wave-type",
            "calibrate-n-segments",
            "calibrate-n-reject",
            "calibrate-duration",
            "calibrate-lower-pct",
            "calibrate-upper-pct",
            "calibrate-seed",
            "calibrate-run-btn",
            "calibrate-cancel-btn",
            "calibrate-status",
            "calibrate-progress",
            "calibrate-results-table",
            "calibrate-thresholds-store",
            "calibrate-save-btn",
            "calibrate-save-status",
        }
        missing = expected - _ids(cal.layout)
        assert not missing, f"Calibrate layout missing IDs: {missing}"

    def test_save_button_disabled_initially(self):
        import dash_bootstrap_components as dbc
        buttons = [n for n in _walk(cal.layout) if isinstance(n, dbc.Button)]
        save = next(b for b in buttons if getattr(b, "id", None) == "calibrate-save-btn")
        assert save.disabled is True

    def test_wave_type_defaults_to_ppg(self):
        drops = [n for n in _walk(cal.layout) if isinstance(n, dcc.Dropdown)]
        wt = next(d for d in drops if d.id == "calibrate-wave-type")
        assert wt.value == "PPG"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class _StubThreshold:
    """A duck-typed stand-in so we don't need to run the heavy calibrate()."""

    def __init__(self, sqi_name, lower, upper, calibrated=True):
        self.sqi_name = sqi_name
        self.lower = lower
        self.upper = upper
        self.accept_median = (lower + upper) / 2
        self.accept_std = (upper - lower) / 4
        self.reject_median = upper + 1
        self.n_accept = 100
        self.n_reject = 50
        self.calibrated = calibrated
        self.note = ""


class TestThresholdsToDataframe:
    def test_one_row_per_sqi(self):
        thresholds = {
            "kurtosis_sqi": _StubThreshold("kurtosis_sqi", 0.5, 5.0),
            "perfusion_sqi": _StubThreshold("perfusion_sqi", 10.0, 50.0),
            "broken_sqi":   _StubThreshold("broken_sqi", 0.0, 0.0, calibrated=False),
        }
        df = cal._thresholds_to_dataframe(thresholds)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "sqi" in df.columns
        assert "lower" in df.columns
        # Uncalibrated entries are still shown so the user sees what got skipped.
        assert (df["calibrated"] == False).any()  # noqa: E712


class TestThresholdsToPayload:
    def test_round_trip_through_store(self):
        thresholds = {"k": _StubThreshold("k", 0.5, 5.0)}
        payload = cal._thresholds_to_payload(thresholds)
        # Must be JSON-serialisable (dcc.Store goes through json under the hood).
        json.loads(json.dumps(payload))
        assert payload["k"]["lower"] == 0.5
        assert payload["k"]["upper"] == 5.0
        assert payload["k"]["calibrated"] is True


class TestThresholdsToTable:
    def test_returns_em_for_empty(self):
        out = cal._thresholds_to_table({})
        assert isinstance(out, html.Em)

    def test_returns_datatable_for_populated(self):
        from dash import dash_table
        thresholds = {"k": _StubThreshold("k", 0.5, 5.0)}
        out = cal._thresholds_to_table(thresholds)
        assert isinstance(out, dash_table.DataTable)


class TestSafeRound:
    def test_handles_none(self):
        assert cal._safe_round(None, 4) == ""

    def test_rounds_floats(self):
        assert cal._safe_round(3.14159, 2) == 3.14

    def test_rounds_ints(self):
        assert cal._safe_round(42, 4) == 42.0

    def test_handles_strings(self):
        assert cal._safe_round("not-a-number", 4) == ""


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


class TestAtomicSaveThresholds:
    def test_writes_rule_dict_and_sqi_dict(self, tmp_path):
        # Use a calibrated SQI that exists in the bundled sqi_dict template
        # so export_sqi_dict has a matching entry.  ``kurtosis_sqi`` is in the
        # template registry.
        payload = {
            "kurtosis_sqi": {
                "sqi_name": "kurtosis_sqi",
                "lower": 0.5,
                "upper": 5.0,
                "accept_median": 2.75,
                "accept_std": 1.0,
                "reject_median": 8.0,
                "n_accept": 200,
                "n_reject": 50,
                "calibrated": True,
                "note": "test",
            }
        }
        result = cal._atomic_save_thresholds(
            payload, output_dir=tmp_path, wave_type="PPG",
            lower_pct=5.0, upper_pct=95.0,
        )
        rule_path = tmp_path / "rule_dict.json"
        sqi_path = tmp_path / "sqi_dict.json"
        assert rule_path.exists()
        assert sqi_path.exists()

        with rule_path.open() as fh:
            rule_dict = json.load(fh)
        assert "kurtosis_sqi" in rule_dict
        assert result["n_rule_entries"] == 1

    def test_creates_backup_when_target_exists(self, tmp_path):
        # Seed the directory so the exporter has something to back up.
        (tmp_path / "rule_dict.json").write_text(json.dumps({"old": "rule"}))
        (tmp_path / "sqi_dict.json").write_text(json.dumps({"old": "sqi"}))

        payload = {
            "kurtosis_sqi": {
                "sqi_name": "kurtosis_sqi",
                "lower": 0.5, "upper": 5.0,
                "accept_median": 2.75, "accept_std": 1.0,
                "reject_median": 8.0,
                "n_accept": 200, "n_reject": 50,
                "calibrated": True, "note": "",
            }
        }
        cal._atomic_save_thresholds(
            payload, output_dir=tmp_path, wave_type="PPG",
            lower_pct=5.0, upper_pct=95.0,
        )
        # The exporters write timestamped backup files alongside the targets.
        backups = list(tmp_path.glob("rule_dict_backup_*.json"))
        assert backups, "expected a rule_dict_backup_*.json to be created"

    def test_skips_uncalibrated_entries(self, tmp_path):
        payload = {
            "kurtosis_sqi": {
                "sqi_name": "kurtosis_sqi",
                "lower": 0.5, "upper": 5.0,
                "accept_median": 2.75, "accept_std": 1.0,
                "reject_median": 8.0,
                "n_accept": 200, "n_reject": 50,
                "calibrated": True, "note": "",
            },
            "broken_sqi": {
                "sqi_name": "broken_sqi",
                "lower": float("nan"), "upper": float("nan"),
                "accept_median": float("nan"), "accept_std": float("nan"),
                "reject_median": float("nan"),
                "n_accept": 0, "n_reject": 0,
                "calibrated": False, "note": "all-NaN",
            },
        }
        result = cal._atomic_save_thresholds(
            payload, output_dir=tmp_path, wave_type="PPG",
            lower_pct=5.0, upper_pct=95.0,
        )
        # The uncalibrated entry must NOT appear in rule_dict (it has no bounds).
        rule_dict = json.loads((tmp_path / "rule_dict.json").read_text())
        assert "broken_sqi" not in rule_dict
        assert result["n_rule_entries"] == 1
