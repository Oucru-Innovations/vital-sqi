"""Unit tests for the Inspect view — layout structure and pure helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import dcc, html

from vital_sqi.app.views import inspect as v


def _find(component, kind):
    found = []
    if isinstance(component, kind):
        found.append(component)
    children = getattr(component, "children", None)
    if children is None:
        return found
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            found.extend(_find(child, kind))
    return found


# ---------------------------------------------------------------------------
# Layout structure
# ---------------------------------------------------------------------------


class TestLayout:
    def test_layout_is_div(self):
        assert isinstance(v.layout, html.Div)

    def test_has_required_ids(self):
        # The Inspect view's callbacks reference these IDs by name; the
        # layout MUST expose them so the callback graph remains valid even
        # before any data is loaded.  Note: ``inspect-decisions`` lives at
        # the app root (since Phase 5) so it's tested in test_app.py.
        expected = {
            "inspect-summary",
            "inspect-timeline",
            "inspect-segment-label",
            "inspect-segment-picker",
            "inspect-filter",
            "inspect-waveform",
            "inspect-sqi-table",
            "inspect-rule-trace",
        }
        # Collect every component with an ``id`` attribute, irrespective of
        # its component class.
        all_with_id = []

        def walk(node):
            if getattr(node, "id", None) is not None:
                all_with_id.append(node.id)
            children = getattr(node, "children", None)
            if children is None:
                return
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                if child is not None:
                    walk(child)

        walk(v.layout)
        missing = expected - set(all_with_id)
        assert not missing, f"Inspect layout missing IDs: {missing}"

    def test_filter_radio_default_is_all(self):
        radios = _find(v.layout, dcc.RadioItems)
        flt = next((r for r in radios if r.id == "inspect-filter"), None)
        assert flt is not None
        assert flt.value == "all"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSqiDfFromStore:
    def test_none_returns_none(self):
        assert v._sqi_df_from_store(None) is None

    def test_empty_dict_returns_none(self):
        assert v._sqi_df_from_store({}) is None

    def test_valid_payload(self):
        payload = {"kurtosis_sqi": {0: 0.1, 1: 0.2, 2: 0.3}}
        df = v._sqi_df_from_store(payload)
        assert df is not None
        assert df.shape == (3, 1)


class TestLoadRuleDict:
    def test_returns_bundled_dict(self):
        # Phase 5 dropped the rule-set-store, so _load_rule_dict takes no
        # arguments and always reads from vital_sqi/resource/rule_dict.json.
        result = v._load_rule_dict()
        assert isinstance(result, dict)
        assert len(result) > 0


class TestBuildRules:
    def test_builds_rule_for_matching_column(self):
        df = pd.DataFrame({"kurtosis_sqi": np.linspace(0.1, 0.9, 50)})
        rule_dict = {"kurtosis_sqi": {"name": "kurtosis_sqi"}}
        rules = v._build_rules(df, rule_dict)
        assert len(rules) == 1
        assert rules[0].name == "kurtosis_sqi"

    def test_skips_unknown_columns(self):
        df = pd.DataFrame({"kurtosis_sqi": np.linspace(0.1, 0.9, 50),
                           "made_up": np.linspace(0.1, 0.9, 50)})
        rule_dict = {"kurtosis_sqi": {"name": "kurtosis_sqi"}}
        rules = v._build_rules(df, rule_dict)
        assert [r.name for r in rules] == ["kurtosis_sqi"]

    def test_handles_columns_with_no_finite_values(self):
        # sanitize_sqi turns all-NaN columns into zeros, so _build_rules can
        # still construct a rule (the band collapses to (0, 0) — every
        # segment subsequently rejects, which is the desired safe default).
        df = pd.DataFrame({"kurtosis_sqi": [np.nan] * 10})
        rule_dict = {"kurtosis_sqi": {"name": "kurtosis_sqi"}}
        rules = v._build_rules(df, rule_dict)
        # Either skipped (legacy behaviour) or yields one degenerate rule;
        # both are acceptable so long as the call doesn't crash.
        assert isinstance(rules, list)


class TestClassify:
    def test_returns_one_entry_per_row(self):
        df = pd.DataFrame({"kurtosis_sqi": np.linspace(0.1, 0.9, 30)})
        rules = v._build_rules(df, {"kurtosis_sqi": {"name": "kurtosis_sqi"}})
        result = v._classify(df, rules)
        assert len(result) == 30
        assert all("decision" in r and "trace" in r for r in result)

    def test_extreme_outliers_get_rejected(self):
        # p5/p95 auto-mode → values *clearly* outside the bulk fall in the
        # reject region.  Use a noisy distribution centred around 0.5 with
        # two distant outliers so the empirical p5 and p95 sit well inside
        # [-100, 100].
        rng = np.random.default_rng(0)
        bulk = rng.normal(0.5, 0.05, 48)
        values = np.concatenate([bulk, [-100.0, 100.0]])
        df = pd.DataFrame({"kurtosis_sqi": values})
        rules = v._build_rules(df, {"kurtosis_sqi": {"name": "kurtosis_sqi"}})
        result = v._classify(df, rules)
        # Last two rows are the outliers — both must reject.
        assert result[-1]["decision"] == "reject"
        assert result[-2]["decision"] == "reject"

    def test_no_rules_means_all_accept(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = v._classify(df, rules=[])
        assert all(r["decision"] == "accept" for r in result)


class TestSegmentIndicesForFilter:
    @pytest.fixture
    def decisions(self):
        return [
            {"decision": "accept"},
            {"decision": "reject"},
            {"decision": "accept"},
            {"decision": "reject"},
        ]

    def test_all(self, decisions):
        assert v._segment_indices_for_filter(decisions, "all") == [0, 1, 2, 3]

    def test_accept_only(self, decisions):
        assert v._segment_indices_for_filter(decisions, "accept") == [0, 2]

    def test_reject_only(self, decisions):
        assert v._segment_indices_for_filter(decisions, "reject") == [1, 3]


class TestMakeTimelineFigure:
    def test_returns_plotly_figure(self):
        decisions = [{"decision": "accept"}, {"decision": "reject"}, {"decision": "accept"}]
        fig = v._make_timeline_figure(decisions)
        assert isinstance(fig, go.Figure)
        # Single bar trace, length 3
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 3

    def test_colors_match_decisions(self):
        decisions = [{"decision": "accept"}, {"decision": "reject"}]
        fig = v._make_timeline_figure(decisions)
        colors = list(fig.data[0].marker.color)
        # Two distinct colours, one for each decision.
        assert colors[0] != colors[1]


class TestMakeWaveformFigure:
    def test_normal_segment(self):
        signal = np.sin(np.linspace(0, 10, 1000))
        fig = v._make_waveform_figure(
            signal, sampling_rate=100, start_idx=200, end_idx=500, label="seg 2"
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 300

    def test_empty_segment(self):
        signal = np.zeros(100)
        fig = v._make_waveform_figure(
            signal, sampling_rate=100, start_idx=50, end_idx=50, label="seg"
        )
        assert isinstance(fig, go.Figure)

    def test_clipped_bounds(self):
        signal = np.arange(100, dtype=float)
        fig = v._make_waveform_figure(
            signal, sampling_rate=10, start_idx=-10, end_idx=200, label="clip"
        )
        # Should clip to the full available range, not crash.
        assert len(fig.data[0].x) == 100


class TestRulesPanelControls:
    """Phase-2 follow-up: quantile / tune / manual controls in the Rules panel."""

    def test_mode_radio_exists_with_expected_options(self):
        radios = _find(v.layout, dcc.RadioItems)
        mode = next((r for r in radios if r.id == "inspect-mode"), None)
        assert mode is not None
        values = {opt["value"] for opt in mode.options}
        # Phase 4 adds 'robust' alongside the original three.
        assert values == {"quantile", "tune", "manual", "robust"}
        assert mode.value == "quantile"

    def test_quantile_range_slider_exists(self):
        sliders = _find(v.layout, dcc.RangeSlider)
        ids = {getattr(s, "id", None) for s in sliders}
        assert "inspect-quantile-slider" in ids

    def test_tune_slider_exists(self):
        sliders = _find(v.layout, dcc.Slider)
        ids = {getattr(s, "id", None) for s in sliders}
        assert "inspect-tune-slider" in ids

    def test_drop_strictest_button_starts_disabled(self):
        import dash_bootstrap_components as dbc
        buttons = _find(v.layout, dbc.Button)
        drop = next(
            (b for b in buttons if getattr(b, "id", None) == "inspect-drop-strictest-btn"),
            None,
        )
        assert drop is not None
        assert drop.disabled is True


class TestPerRuleRejectCounts:
    def test_counts_first_reject_per_segment_correctly(self):
        decisions = [
            {"decision": "accept", "trace": [{"name": "k", "outcome": "accept"}]},
            {"decision": "reject", "trace": [
                {"name": "k", "outcome": "reject"},
                {"name": "p", "outcome": "accept"},
            ]},
            {"decision": "reject", "trace": [
                {"name": "k", "outcome": "accept"},
                {"name": "p", "outcome": "reject"},
            ]},
            {"decision": "reject", "trace": [
                {"name": "k", "outcome": "reject"},
                {"name": "p", "outcome": "reject"},
            ]},
        ]
        counts = v._per_rule_reject_counts(decisions)
        assert counts == {"k": 2, "p": 2}

    def test_no_rejects(self):
        decisions = [{"decision": "accept", "trace": []}, {"decision": "accept", "trace": []}]
        assert v._per_rule_reject_counts(decisions) == {}


class TestBuildRulesManual:
    def test_manual_rules_use_dict_bounds(self):
        import json
        from vital_sqi.app.views.inspect import _build_rules_manual

        rule_dict = {
            "kurtosis_sqi": {
                "name": "kurtosis_sqi",
                "def": [
                    {"op": ">",  "value": "0", "label": "accept"},
                    {"op": "<=", "value": "0", "label": "reject"},
                    {"op": ">=", "value": "1", "label": "reject"},
                    {"op": "<",  "value": "1", "label": "accept"},
                ],
            }
        }
        rules = _build_rules_manual(rule_dict, selected_columns=None)
        assert len(rules) == 1
        # Manual rule should accept 0.5 and reject -0.5 / 2.0
        assert rules[0].apply_rule(0.5) == "accept"
        assert rules[0].apply_rule(-0.5) == "reject"
        assert rules[0].apply_rule(2.0) == "reject"

    def test_whitelist_filters_rules(self):
        from vital_sqi.app.views.inspect import _build_rules_manual

        rule_dict = {
            f"sqi_{i}": {"name": f"sqi_{i}", "def": [
                {"op": ">",  "value": "0", "label": "accept"},
                {"op": "<=", "value": "0", "label": "reject"},
                {"op": ">=", "value": "1", "label": "reject"},
                {"op": "<",  "value": "1", "label": "accept"},
            ]} for i in range(3)
        }
        rules = _build_rules_manual(rule_dict, selected_columns=["sqi_1"])
        assert [r.name for r in rules] == ["sqi_1"]


class TestRobustMode:
    """Phase-4 tests: the Inspect view's Robust threshold mode."""

    def test_mode_radio_includes_robust(self):
        radios = _find(v.layout, dcc.RadioItems)
        mode = next(r for r in radios if r.id == "inspect-mode")
        values = {opt["value"] for opt in mode.options}
        assert "robust" in values

    def test_decisions_from_robust_result(self):
        # Construct a tiny synthetic RobustResult and verify the converter.
        from vital_sqi.rule import RobustResult
        result = RobustResult(
            decisions=["accept", "reject", "accept"],
            scores=np.array([0.8, 0.2, 0.7]),
            regime="bimodal",
            regime_info={"score_mean": 0.57, "file_flagged": False},
            file_flagged=False,
        )
        out = v._decisions_from_robust_result(result)
        assert len(out) == 3
        assert out[0]["decision"] == "accept"
        assert out[0]["regime"] == "bimodal"
        # Per-segment "trace" must be a single synthetic row so the
        # detail panel has something to render.
        assert out[0]["trace"][0]["name"] == "robust_score"

    def test_regime_banner_hidden_in_non_robust_mode(self):
        # Banner returns an empty string unless the mode is robust.
        out = v._render_regime_banner(
            "quantile",
            decisions=[{"decision": "accept"}],
        )
        assert out == ""

    def test_regime_banner_shows_badge_in_robust_mode(self):
        import dash_bootstrap_components as dbc
        out = v._render_regime_banner(
            "robust",
            decisions=[{
                "decision": "accept",
                "regime": "clean",
                "regime_info": {"score_mean": 0.83, "file_flagged": False},
            }],
        )
        assert isinstance(out, dbc.Badge)
        assert "clean" in str(out.children)


class TestBuildRulesTuneMode:
    def test_tune_mode_produces_wider_bands_than_quantile(self):
        # With 5 SQIs and target 0.85, per-rule trim ≈ 1.6% → bands are
        # noticeably wider than p5/p95.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            f"sqi_{i}": rng.normal(0, 1, 1000) for i in range(5)
        })
        rule_dict = {c: {"name": c} for c in df.columns}

        q_rules = v._build_rules(
            df, rule_dict, mode="quantile", quantile_lo=0.05, quantile_hi=0.95,
        )
        t_rules = v._build_rules(
            df, rule_dict, mode="tune", target_accept_rate=0.85,
        )
        # Both should produce a rule per column.
        assert len(q_rules) == len(t_rules) == 5
        # Tune-mode bands should be wider on average.
        q_widths = [
            rule.rule["boundaries"][1] - rule.rule["boundaries"][0]
            for rule in q_rules
        ]
        t_widths = [
            rule.rule["boundaries"][1] - rule.rule["boundaries"][0]
            for rule in t_rules
        ]
        assert np.mean(t_widths) > np.mean(q_widths)


class TestFormatSegmentLabel:
    def test_label_contains_idx_and_decision(self):
        label = v._format_segment_label(
            idx=7, start_idx=300, end_idx=600,
            sampling_rate=100, decision="accept",
        )
        assert "0007" in label
        assert "accept" in label
