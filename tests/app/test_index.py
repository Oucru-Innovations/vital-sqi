"""
Unit tests for vital_sqi.app.util.parsing — no browser required.

Tests parse_data, generate_rule, generate_rule_set, parse_rule_list,
and generate_boundaries purely in Python.
"""
import base64
import io
import json
import pytest
import pandas as pd
from dash import html

from vital_sqi.app.util.parsing import (
    parse_data,
    generate_rule,
    generate_rule_set,
    parse_rule_list,
    generate_boundaries,
)
from vital_sqi.rule.rule_class import Rule
from vital_sqi.rule.ruleset_class import RuleSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(content_bytes, mime="text/csv"):
    encoded = base64.b64encode(content_bytes).decode()
    return f"data:{mime};base64,{encoded}"


def _csv_content(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return _encode(buf.getvalue().encode())


def _json_content(obj):
    return _encode(json.dumps(obj).encode(), mime="application/json")


# ---------------------------------------------------------------------------
# parse_data
# ---------------------------------------------------------------------------

class TestParseData:
    def test_csv_returns_dict(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = parse_data(_csv_content(df), "signal.csv")
        assert isinstance(result, dict)
        assert "a" in result

    def test_json_returns_dict(self):
        obj = {"rule": [{"op": ">", "value": 0.5, "label": "accept"}]}
        result = parse_data(_json_content(obj), "rules.json")
        assert isinstance(result, dict)
        assert "rule" in result

    def test_txt_returns_dict(self):
        txt = "col1 col2\n1 2\n3 4\n"
        content = _encode(txt.encode(), mime="text/plain")
        result = parse_data(content, "data.txt")
        assert isinstance(result, dict)

    def test_unsupported_extension_returns_error_div(self):
        content = _encode(b"binary data", mime="application/octet-stream")
        result = parse_data(content, "file.xyz")
        assert isinstance(result, html.Div)

    def test_malformed_content_returns_error_div(self):
        result = parse_data("not_valid_base64_content", "file.csv")
        assert isinstance(result, html.Div)


# ---------------------------------------------------------------------------
# generate_rule
# ---------------------------------------------------------------------------

class TestGenerateRule:
    _rule_def = [
        {"op": "<=", "value": 0.5, "label": "reject"},
        {"op": ">",  "value": 0.5, "label": "accept"},
    ]

    def test_returns_rule_object(self):
        rule = generate_rule("kurtosis_sqi", self._rule_def)
        assert isinstance(rule, Rule)

    def test_rule_name_preserved(self):
        rule = generate_rule("kurtosis_sqi", self._rule_def)
        assert rule.name == "kurtosis_sqi"

    def test_rule_has_boundaries(self):
        rule = generate_rule("kurtosis_sqi", self._rule_def)
        assert rule.rule is not None
        assert "boundaries" in rule.rule

    def test_invalid_def_returns_none(self):
        result = generate_rule("bad_sqi", None)
        assert result is None


# ---------------------------------------------------------------------------
# generate_rule_set
# ---------------------------------------------------------------------------

class TestGenerateRuleSet:
    _rule_set_dict = [
        {
            "name": "kurtosis_sqi",
            "order": 1,
            "def": [
                {"op": "<=", "value": 0.5, "label": "reject"},
                {"op": ">",  "value": 0.5, "label": "accept"},
            ],
        },
        {
            "name": "skewness_sqi",
            "order": 2,
            "def": [
                {"op": "<=", "value": 2.0, "label": "accept"},
                {"op": ">",  "value": 2.0, "label": "reject"},
            ],
        },
    ]

    def test_returns_ruleset_object(self):
        rs = generate_rule_set(self._rule_set_dict)
        assert isinstance(rs, RuleSet)

    def test_ruleset_has_correct_number_of_rules(self):
        rs = generate_rule_set(self._rule_set_dict)
        assert len(rs.rules) == 2

    def test_empty_dict_returns_empty_ruleset(self):
        rs = generate_rule_set([])
        assert isinstance(rs, RuleSet)
        assert len(rs.rules) == 0

    def test_invalid_input_returns_none(self):
        result = generate_rule_set(None)
        assert result is None


# ---------------------------------------------------------------------------
# parse_rule_list
# ---------------------------------------------------------------------------

class TestParseRuleList:
    def test_returns_list_of_dicts(self):
        rule_def = [
            {"op": "<=", "value": 0.5, "label": "reject", "extra": "ignored"},
            {"op": ">",  "value": 0.5, "label": "accept", "extra": "ignored"},
        ]
        result = parse_rule_list(rule_def)
        assert isinstance(result, list)
        assert len(result) == 2
        assert set(result[0].keys()) == {"op", "value", "label"}

    def test_empty_list_returns_empty(self):
        assert parse_rule_list([]) == []

    def test_malformed_input_returns_empty(self):
        result = parse_rule_list("not a list")
        assert result == []


# ---------------------------------------------------------------------------
# generate_boundaries
# ---------------------------------------------------------------------------

class TestGenerateBoundaries:
    def test_single_boundary(self):
        result = generate_boundaries([5.0])
        assert len(result) == 2
        assert "[-inf" in result[0]
        assert "inf]" in result[-1]

    def test_multiple_boundaries(self):
        result = generate_boundaries([1.0, 3.0, 7.0])
        assert len(result) == 4  # n+1 intervals for n boundaries
        assert result[0] == "[-inf, 1.0]"
        assert result[-1] == "[7.0, inf]"

    def test_empty_boundaries_returns_empty(self):
        result = generate_boundaries([])
        assert result == []
