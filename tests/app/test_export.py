"""Unit tests for the Export view — layout + every download builder."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from dash import dcc, html

from vital_sqi.app.views import export as ex


def _walk(node):
    yield node
    children = getattr(node, "children", None)
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
            "export-summary",
            "export-decisions-btn",
            "export-rule-dict-btn",
            "export-accepted-zip-btn",
            "export-html-btn",
            "export-decisions-download",
            "export-rule-dict-download",
            "export-accepted-zip-download",
            "export-html-download",
        }
        missing = expected - _ids(ex.layout)
        assert not missing, f"Export layout missing IDs: {missing}"

    def test_one_download_endpoint_per_button(self):
        downloads = [n for n in _walk(ex.layout) if isinstance(n, dcc.Download)]
        ids = {d.id for d in downloads}
        assert len(ids) == 4
        for required in (
            "export-decisions-download",
            "export-rule-dict-download",
            "export-accepted-zip-download",
            "export-html-download",
        ):
            assert required in ids


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSqiDfFromStore:
    def test_none_returns_none(self):
        assert ex._sqi_df_from_store(None) is None

    def test_empty_dict_returns_none(self):
        assert ex._sqi_df_from_store({}) is None

    def test_valid_payload(self):
        payload = {"k": {0: 0.1, 1: 0.2}}
        out = ex._sqi_df_from_store(payload)
        assert out is not None
        assert out.shape == (2, 1)


class TestAttachDecisions:
    def test_adds_decision_column(self):
        df = pd.DataFrame({"k": [0.1, 0.2, 0.3]})
        decisions = [
            {"decision": "accept"},
            {"decision": "reject"},
            {"decision": "accept"},
        ]
        out = ex._attach_decisions(df, decisions)
        assert list(out["decision"]) == ["accept", "reject", "accept"]
        # Source DataFrame must be untouched.
        assert "decision" not in df.columns

    def test_unknown_when_decisions_missing(self):
        df = pd.DataFrame({"k": [0.1, 0.2]})
        out = ex._attach_decisions(df, None)
        assert (out["decision"] == "unknown").all()

    def test_unknown_when_length_mismatch(self):
        df = pd.DataFrame({"k": [0.1, 0.2, 0.3]})
        decisions = [{"decision": "accept"}]
        out = ex._attach_decisions(df, decisions)
        assert (out["decision"] == "unknown").all()


class TestDecisionsToRuleDict:
    def test_quantile_mode_produces_one_entry_per_column(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"sqi_{i}": rng.normal(0, 1, 500) for i in range(3)})
        out = ex._decisions_to_rule_dict(
            df, decisions=None, mode="quantile",
            quantile_lo=0.05, quantile_hi=0.95,
        )
        assert set(out.keys()) == {"sqi_0", "sqi_1", "sqi_2"}
        # Each entry has the canonical "def" structure.
        for name, entry in out.items():
            assert entry["name"] == name
            assert len(entry["def"]) == 4
            assert {op["op"] for op in entry["def"]} == {">", "<=", ">=", "<"}

    def test_tune_mode_produces_wider_bands(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"sqi_{i}": rng.normal(0, 1, 500) for i in range(5)})
        q_out = ex._decisions_to_rule_dict(df, None, mode="quantile")
        t_out = ex._decisions_to_rule_dict(df, None, mode="tune", target_accept_rate=0.85)
        # Both should produce 5 entries.
        assert len(q_out) == 5 and len(t_out) == 5

    def test_manual_mode_returns_empty(self):
        # Manual mode delegates to the bundled rule_dict — no snapshot.
        out = ex._decisions_to_rule_dict(
            pd.DataFrame({"k": [0.1, 0.2]}), None, mode="manual",
        )
        assert out == {}

    def test_robust_mode_returns_empty(self):
        out = ex._decisions_to_rule_dict(
            pd.DataFrame({"k": [0.1, 0.2]}), None, mode="robust",
        )
        assert out == {}

    def test_drops_constant_columns(self):
        df = pd.DataFrame({
            "varying": np.linspace(0, 1, 100),
            "constant": np.full(100, 0.5),
        })
        out = ex._decisions_to_rule_dict(df, None, mode="quantile")
        assert "varying" in out
        assert "constant" not in out

    def test_selected_columns_whitelist(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"sqi_{i}": rng.normal(0, 1, 200) for i in range(3)})
        out = ex._decisions_to_rule_dict(
            df, None, mode="quantile", selected_columns=["sqi_1"],
        )
        assert set(out.keys()) == {"sqi_1"}


class TestBuildAcceptedZip:
    def _waveform_payload(self, n=300):
        return {
            "signal": np.arange(n, dtype=float).tolist(),
            "sampling_rate": 100.0,
            "wave_type": "PPG",
            "source_name": "test.csv",
            "timestamps": None,
        }

    def test_returns_none_when_payload_missing(self):
        assert ex._build_accepted_zip(None, None, None) is None
        assert ex._build_accepted_zip(self._waveform_payload(), None, None) is None

    def test_returns_none_when_lengths_mismatch(self):
        milestones = {"start": [0, 100], "end": [100, 200]}
        decisions = [{"decision": "accept"}]  # length 1 vs 2
        out = ex._build_accepted_zip(self._waveform_payload(), milestones, decisions)
        assert out is None

    def test_returns_none_when_zero_accepted(self):
        milestones = {"start": [0, 100], "end": [100, 200]}
        decisions = [{"decision": "reject"}, {"decision": "reject"}]
        out = ex._build_accepted_zip(self._waveform_payload(), milestones, decisions)
        assert out is None

    def test_zip_contains_one_csv_per_accepted_segment(self):
        milestones = {"start": [0, 100, 200], "end": [100, 200, 300]}
        decisions = [
            {"decision": "accept"},
            {"decision": "reject"},
            {"decision": "accept"},
        ]
        payload = ex._build_accepted_zip(self._waveform_payload(), milestones, decisions)
        assert payload is not None
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = zf.namelist()
        assert len(names) == 2
        assert all(n.endswith(".csv") for n in names)


class TestBuildHtmlReport:
    def _basic_inputs(self):
        df = pd.DataFrame({
            "kurtosis_sqi": [1.0, 2.0, 3.0],
            "perfusion_sqi": [10.0, 20.0, 30.0],
            "decision": ["accept", "reject", "accept"],
        })
        decisions = [{"decision": d} for d in df["decision"]]
        waveform = {
            "signal": list(range(300)),
            "sampling_rate": 100.0,
            "wave_type": "PPG",
            "source_name": "test.csv",
            "timestamps": None,
        }
        return df, decisions, waveform

    def test_returns_html_string(self):
        df, decisions, wav = self._basic_inputs()
        out = ex._build_html_report(df, wav, decisions,
                                    mode="quantile",
                                    selected_columns=["kurtosis_sqi"])
        assert isinstance(out, str)
        assert out.startswith("<!doctype html>")
        assert "vital_sqi report" in out

    def test_shows_decision_counts(self):
        df, decisions, wav = self._basic_inputs()
        out = ex._build_html_report(df, wav, decisions,
                                    mode="quantile",
                                    selected_columns=None)
        # 2 accept + 1 reject in the fixture.
        assert "2" in out  # accept count
        assert "1" in out  # reject count

    def test_lists_active_rules(self):
        df, decisions, wav = self._basic_inputs()
        out = ex._build_html_report(df, wav, decisions,
                                    mode="quantile",
                                    selected_columns=["kurtosis_sqi", "perfusion_sqi"])
        assert "kurtosis_sqi" in out and "perfusion_sqi" in out

    def test_works_without_waveform_payload(self):
        df, decisions, _ = self._basic_inputs()
        # Should still produce a valid report when raw waveform is absent
        # (e.g. user uploaded a pre-computed SQI table).
        out = ex._build_html_report(df, None, decisions,
                                    mode="quantile",
                                    selected_columns=None)
        assert "vital_sqi report" in out
