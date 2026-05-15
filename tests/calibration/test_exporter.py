"""Unit tests for vital_sqi.calibration.exporter."""
import json
import os
import numpy as np
import pytest

from vital_sqi.calibration.threshold_estimator import SQIThreshold
from vital_sqi.calibration.exporter import (
    _make_rule_def_entry,
    export_rule_dict,
    export_sqi_dict,
    export_diagnostics,
)


def _calibrated_threshold(name, lower=0.5, upper=5.0, **kw):
    return SQIThreshold(
        sqi_name=name,
        lower=lower, upper=upper,
        accept_median=(lower + upper) / 2,
        accept_std=(upper - lower) / 4,
        reject_median=upper + 1,
        n_accept=100, n_reject=50,
        calibrated=True, note=kw.get("note", ""),
    )


class TestMakeRuleDefEntry:
    def test_returns_dict_with_required_keys(self):
        t = _calibrated_threshold("kurtosis_sqi", 1.0, 10.0)
        entry = _make_rule_def_entry(t, "PPG", 5.0, 95.0)
        assert entry["name"] == "kurtosis_sqi"
        assert "def" in entry
        assert "desc" in entry
        assert entry["ref"] == "vital_sqi.calibration"

    def test_def_has_four_paired_rules(self):
        t = _calibrated_threshold("kurtosis_sqi", 1.0, 10.0)
        entry = _make_rule_def_entry(t, "PPG", 5.0, 95.0)
        ops = [r["op"] for r in entry["def"]]
        assert ops == [">", "<=", ">=", "<"]
        labels = [r["label"] for r in entry["def"]]
        assert labels == ["accept", "reject", "reject", "accept"]


class TestExportRuleDict:
    def test_writes_new_file(self, tmp_path):
        out = tmp_path / "rule_dict.json"
        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_rule_dict(thresholds, str(out), wave_type="PPG")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "kurtosis_sqi" in data
        assert data["kurtosis_sqi"]["name"] == "kurtosis_sqi"

    def test_uncalibrated_thresholds_are_skipped(self, tmp_path):
        out = tmp_path / "rule_dict.json"
        thresholds = {
            "good_sqi": _calibrated_threshold("good_sqi"),
            "bad_sqi": SQIThreshold(sqi_name="bad_sqi"),  # calibrated=False
        }
        export_rule_dict(thresholds, str(out), backup=False)
        data = json.loads(out.read_text())
        assert "good_sqi" in data
        assert "bad_sqi" not in data

    def test_merges_existing_entries(self, tmp_path):
        out = tmp_path / "rule_dict.json"
        # Pre-existing entry that the calibrator did not touch.
        pre = {"hand_curated_sqi": {"name": "hand_curated_sqi", "def": []}}
        out.write_text(json.dumps(pre))

        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_rule_dict(thresholds, str(out), backup=False)
        data = json.loads(out.read_text())
        assert "hand_curated_sqi" in data
        assert "kurtosis_sqi" in data

    def test_backup_created_when_file_exists(self, tmp_path):
        out = tmp_path / "rule_dict.json"
        out.write_text("{}")
        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_rule_dict(thresholds, str(out), backup=True)
        backups = list(tmp_path.glob("rule_dict_backup_*.json"))
        assert len(backups) == 1


class TestExportSqiDict:
    def test_writes_calibrated_sqis_only(self, tmp_path):
        out = tmp_path / "sqi_dict.json"
        thresholds = {
            "kurtosis_sqi": _calibrated_threshold("kurtosis_sqi"),
            "skewness_sqi": SQIThreshold(sqi_name="skewness_sqi"),  # not calibrated
        }
        export_sqi_dict(thresholds, str(out), backup=False)
        data = json.loads(out.read_text())
        assert "kurtosis_sqi" in data
        assert "skewness_sqi" not in data

    def test_entries_have_sqi_and_args(self, tmp_path):
        out = tmp_path / "sqi_dict.json"
        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_sqi_dict(thresholds, str(out), backup=False)
        data = json.loads(out.read_text())
        entry = data["kurtosis_sqi"]
        assert entry["sqi"] == "kurtosis_sqi"
        assert "args" in entry

    def test_poincare_subcolumns_emit_parent(self, tmp_path):
        out = tmp_path / "sqi_dict.json"
        # poincare returns dict with sd1/sd2/area/ratio columns; if any sub
        # column was calibrated, the poincare_sqi parent entry must appear.
        thresholds = {"sd1": _calibrated_threshold("sd1")}
        export_sqi_dict(thresholds, str(out), backup=False)
        data = json.loads(out.read_text())
        assert "poincare_sqi" in data

    def test_backup_created_when_file_exists(self, tmp_path):
        out = tmp_path / "sqi_dict.json"
        out.write_text("{}")
        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_sqi_dict(thresholds, str(out), backup=True)
        backups = list(tmp_path.glob("sqi_dict_backup_*.json"))
        assert len(backups) == 1


class TestExportDiagnostics:
    def test_writes_csv(self, tmp_path):
        out = tmp_path / "diag.csv"
        thresholds = {"kurtosis_sqi": _calibrated_threshold("kurtosis_sqi")}
        export_diagnostics(thresholds, str(out))
        assert out.exists()
        content = out.read_text()
        assert "kurtosis_sqi" in content
        assert "lower" in content
