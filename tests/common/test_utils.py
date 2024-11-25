import pytest
import numpy as np
import pandas as pd
from vital_sqi.common.utils import (
    HiddenPrints,
    get_nn,
    check_valid_signal,
    calculate_sampling_rate,
    generate_timestamp,
    parse_datetime,
    parse_rule,
    generate_labels,
    sort_rule,
    decompose_operand,
    check_unique_pair,
    check_conflict,
    get_decision,
    get_inteveral_label_list,
    get_value_label_list,
    cut_segment,
    format_milestone,
    check_signal_format,
    create_rule_def,
)

# Mock constants
OPERAND_MAPPING_DICT = {">": 5, ">=": 4, "=": 3, "<=": 2, "<": 1}


# Fixtures
@pytest.fixture
def mock_signal():
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def mock_timestamps():
    return pd.date_range(start="2023-01-01", periods=5, freq="1S")


@pytest.fixture
def mock_rule_dict():
    return [
        {"op": ">", "value": "0.5", "label": "accept"},
        {"op": "<=", "value": "0.5", "label": "reject"},
    ]


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="S"),
            "signal": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def mock_df_with_labels():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="S"),
            "signal": [1, 2, 3, 4, 5],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
            "op": [">", "<=", ">", "<=", "="],
            "label": ["accept", "reject", "accept", "reject", "reject"],
        }
    )


# Tests
def test_hidden_prints(capsys):
    with HiddenPrints():
        print("This will not be printed.")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_get_nn(mock_signal):
    nn_intervals = get_nn(
        mock_signal, wave_type="PPG", sample_rate=100, remove_ectopic_beat=False
    )
    assert nn_intervals.size >= 0


def test_check_valid_signal(mock_signal):
    assert check_valid_signal(mock_signal) is True
    with pytest.raises(ValueError):
        check_valid_signal("invalid_signal")


def test_calculate_sampling_rate(mock_timestamps):
    sampling_rate = calculate_sampling_rate(mock_timestamps)
    assert sampling_rate == 1.0


def test_generate_timestamp():
    timestamps = generate_timestamp(pd.Timestamp("2023-01-01"), 1, 5)
    assert len(timestamps) == 5
    assert isinstance(timestamps[0], pd.Timestamp)


def test_parse_datetime():
    parsed_date = parse_datetime("2023-01-01", type="date")
    assert parsed_date == pd.Timestamp("2023-01-01")


def test_parse_rule(mock_rule_dict):
    rule_def, boundaries, labels = parse_rule(
        "mock_rule", {"mock_rule": {"def": mock_rule_dict}}
    )
    assert len(rule_def) == len(mock_rule_dict)


def test_generate_labels(mock_df_with_labels):
    boundaries = np.array([0.1, 0.3, 0.5])
    interval_labels, value_labels = generate_labels(mock_df_with_labels, boundaries)
    assert len(interval_labels) == len(boundaries) + 1
    assert len(value_labels) == len(boundaries)


def test_check_unique_pair():
    pair = pd.DataFrame([{"value": 1, "op": "="}])
    assert check_unique_pair(pair) is True
    with pytest.raises(AssertionError):
        duplicate_pair = pd.concat([pair, pair])  # Use pd.concat instead of append
        check_unique_pair(duplicate_pair)


def test_sort_rule(mock_rule_dict):
    sorted_df = sort_rule(mock_rule_dict)
    assert not sorted_df.empty


def test_decompose_operand(mock_rule_dict):
    decomposed_df = decompose_operand(mock_rule_dict)
    assert not decomposed_df.empty


def test_check_conflict():
    decision_lt = pd.DataFrame([{"value": 1, "op": "<", "label": "accept"}])
    decision_gt = pd.DataFrame([{"value": 1, "op": ">", "label": "accept"}])
    assert check_conflict(decision_lt, decision_gt) == "accept"

    conflicting_decision = pd.DataFrame([{"value": 1, "op": ">", "label": "reject"}])
    with pytest.raises(ValueError):
        check_conflict(decision_lt, conflicting_decision)


def test_get_decision(mock_df_with_labels):
    boundaries = np.array([0.1, 0.3, 0.5])
    decision = get_decision(mock_df_with_labels, boundaries, 0)
    assert decision is not None


def test_get_inteveral_label_list():
    mock_df_with_rules = pd.DataFrame(
        [
            {"value": -np.inf, "op": "<", "label": "neutral"},
            {"value": 0.1, "op": "<", "label": "reject"},
            {"value": 0.1, "op": "=", "label": "accept"},
            {"value": 0.3, "op": ">", "label": "accept"},
            {"value": 0.3, "op": "<=", "label": "reject"},
            {"value": 0.5, "op": ">", "label": "neutral"},
        ]
    )
    boundaries = np.array([0.1, 0.3, 0.5])
    interval_labels = get_inteveral_label_list(mock_df_with_rules, boundaries)
    assert len(interval_labels) == len(boundaries) + 1
    assert interval_labels[0] == "neutral"
    assert interval_labels[-1] == "neutral"


def test_get_value_label_list(mock_df_with_labels):
    boundaries = np.array([0.1, 0.3, 0.5])
    value_labels = get_value_label_list(mock_df_with_labels, boundaries, [None] * 4)
    assert len(value_labels) == len(boundaries)


def test_cut_segment(mock_df):
    milestones = pd.DataFrame({"start": [0, 2], "end": [2, 5]})
    segments = cut_segment(mock_df, milestones)
    assert len(segments) == len(milestones)


def test_format_milestone():
    start = [0, 1, 2]
    end = [2, 3]
    milestones = format_milestone(start, end)
    assert len(milestones) == len(end)


def test_check_signal_format(mock_signal):
    df = check_signal_format(mock_signal)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns


def test_create_rule_def():
    rule_def = create_rule_def("sqi", upper_bound=1, lower_bound=0)
    assert "sqi" in rule_def
    assert "def" in rule_def["sqi"]
