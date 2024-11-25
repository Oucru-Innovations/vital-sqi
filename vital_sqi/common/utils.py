import numpy as np
import os
import sys
import datetime as dt
import json
import pandas as pd
from pandas.core.dtypes.common import is_number
from vital_sqi.common.rpeak_detection import PeakDetector
import dateparser
import logging
from vitalDSP.transforms.beats_transformation import RRTransformation

# Set up logging configuration
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPERAND_MAPPING_DICT = {">": 5, ">=": 4, "=": 3, "<=": 2, "<": 1}


class HiddenPrints:
    """
    Context manager to suppress console output temporarily.

    Usage:
    ------
    with HiddenPrints():
        # Code that prints to console
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_nn(
    signal, wave_type="PPG", sample_rate=100, rpeak_method=6, remove_ectopic_beat=False
):
    """
    Calculate NN intervals from a PPG or ECG signal.

    Parameters
    ----------
    signal : array_like
        Input signal data.
    wave_type : str, optional
        Type of waveform ('PPG' or 'ECG'), by default 'PPG'.
    sample_rate : int or float, optional
        Sampling frequency in Hz, by default 100.
    rpeak_method : int, optional
        Method identifier for R-peak detection, by default 7.
    remove_ectopic_beat : bool, optional
        If True, removes ectopic beats, by default False.

    Returns
    -------
    np.ndarray
        Array of NN intervals in milliseconds.
    """
    try:
        transformer = RRTransformation(
            signal=signal, fs=sample_rate, signal_type=wave_type
        )
        rr_intervals = transformer.process_rr_intervals(
            impute_invalid=False, remove_invalid=remove_ectopic_beat
        )
        nn_intervals_non_na = np.where(np.isnan(rr_intervals), -1, rr_intervals)
        return nn_intervals_non_na
    except Exception as e:
        logging.error(f"Error in get_nn function: {e}")
        return np.array([])


def check_valid_signal(signal):
    """
    Validates that the input is a numeric array-like signal.

    Parameters
    ----------
    signal : array_like
        The signal to validate.

    Returns
    -------
    bool
        True if the signal is valid.

    Raises
    ------
    ValueError
        If the signal is invalid.
    """
    if not isinstance(signal, (list, np.ndarray, pd.Series)):
        raise ValueError(f"Expected array-like input, found {type(signal)}.")

    if len(signal) == 0:
        raise ValueError("Empty signal provided.")

    signal_array = np.asarray(signal)

    if len(signal_array.shape) > 1:
        raise ValueError("Signal must be one-dimensional.")

    if not np.issubdtype(signal_array.dtype, np.number):
        raise ValueError("Signal contains non-numeric data.")

    return True


def calculate_sampling_rate(timestamps):
    """
    Calculates the sampling rate from an array of timestamps.

    Parameters
    ----------
    timestamps : array_like
        Array of timestamps (float, pd.Timestamp, or np.datetime64).

    Returns
    -------
    float or None
        The calculated sampling rate in Hz, or None if calculation fails.
    """
    try:
        if len(timestamps) < 2:
            logging.error("Not enough timestamps to calculate sampling rate.")
            return None

        # Convert timestamps to seconds
        if isinstance(timestamps[0], (float, int)):
            timestamps_seconds = np.array(timestamps)
        elif isinstance(timestamps[0], pd.Timestamp):
            timestamps_seconds = (timestamps - timestamps[0]).total_seconds()
        elif isinstance(timestamps[0], np.datetime64):
            timestamps_seconds = (timestamps - timestamps[0]) / np.timedelta64(1, "s")
        else:
            # Attempt to parse timestamps using dateparser
            timestamps_parsed = [dateparser.parse(str(ts)) for ts in timestamps]
            if None in timestamps_parsed:
                logging.error("Failed to parse some timestamps.")
                return None
            timestamps_seconds = np.array(
                [
                    (ts - timestamps_parsed[0]).total_seconds()
                    for ts in timestamps_parsed
                ]
            )

        # Calculate time differences
        time_diffs = np.diff(timestamps_seconds)
        time_diffs = time_diffs[time_diffs > 0]  # Exclude zero or negative differences

        if len(time_diffs) == 0:
            logging.error("No positive time differences found.")
            return None

        min_step = np.min(time_diffs)
        sampling_rate = round(1 / min_step, 3)
        return sampling_rate
    except Exception as e:
        logging.error(f"Error calculating sampling rate: {e}")
        return None


def generate_timestamp(start_datetime, sampling_rate, signal_length):
    """
    Generates a sequence of timestamps for a signal.

    Parameters
    ----------
    start_datetime : datetime.datetime or pd.Timestamp or None
        The starting timestamp. If None, uses current time.
    sampling_rate : float
        The sampling rate in Hz.
    signal_length : int
        The number of timestamps to generate.

    Returns
    -------
    np.ndarray
        Array of pd.Timestamp objects.

    Raises
    ------
    ValueError
        If sampling_rate is not a real number.
    """
    if not isinstance(sampling_rate, (int, float)) or not np.isreal(sampling_rate):
        raise ValueError("Sampling rate must be a real number.")

    if start_datetime is None:
        start_datetime = pd.Timestamp.now()
    elif not isinstance(start_datetime, pd.Timestamp):
        start_datetime = pd.Timestamp(start_datetime)

    try:
        step_size = 1 / sampling_rate
        time_offsets = np.arange(signal_length) * step_size
        timestamps = start_datetime + pd.to_timedelta(time_offsets, unit="s")
        return timestamps
    except Exception as e:
        logging.error(f"Error generating timestamps: {e}")
        return np.array([])


def parse_datetime(string, type="datetime"):
    """
    A simple dateparser that detects common  datetime formats

    Parameters
    ----------
    string : str
        a date string in format as denoted below.

    Returns
    -------
    datetime.datetime
        datetime object of a time.

    """
    # some common formats.
    date_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%d %b %Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]
    datime_formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d.%m.%Y %H:%M:%S.%f",
        "%Y.%m.%d %H:%M:%S.%f",
        "%d %b %Y %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S.%f",
        "%Y-%m-%d %I:%M:%S.%f",
        "%d-%m-%Y %I:%M:%S.%f",
        "%d.%m.%Y %I:%M:%S.%f",
        "%Y.%m.%d %I:%M:%S.%f",
        "%d %b %Y %I:%M:%S.%f",
        "%Y/%m/%d %I:%M:%S.%f",
        "%d/%m/%Y %I:%M:%S.%f",
    ]
    if type == "date":
        formats = date_formats
    if type == "datetime":
        formats = datime_formats
    for i, f in enumerate(formats):
        try:
            return dt.datetime.strptime(string, f)
        except Exception as e:
            logging.exception(f"Error parsing date format: {e}")
            pass
    try:
        return dateparser.parse(string)
    except Exception as e:
        raise ValueError(
            f"{e} Datetime string must be of standard Python format "
            "(https://docs.python.org/3/library/time.html), "
            "e.g., `%d-%m-%Y`, eg. `24-01-2020`"
        )


def parse_rule(name, source):
    """
    Parses rule definitions from a JSON file or dictionary.

    Parameters
    ----------
    name : str
        Name of the SQI to retrieve.
    source : str or dict
        Path to the JSON file or dictionary containing rule definitions.

    Returns
    -------
    tuple
        A tuple containing rule definitions, boundaries, and label lists.
    """
    # If source is a dictionary, use it directly; otherwise, treat it as a file path
    if isinstance(source, dict):
        all_rules = source
    else:
        assert os.path.isfile(source), "Source file not found."
        with open(source, "r") as json_file:
            try:
                all_rules = json.load(json_file)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON file: {e}")
                raise

    try:
        sqi = all_rules[name]
    except KeyError:
        logging.error(f"SQI {name} not found in the source.")
        raise ValueError(f"SQI {name} not found.")

    return update_rule(sqi["def"], is_update=False)


def generate_labels(df, boundaries):
    """
    Efficiently generates interval and value labels for the boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of decomposed rule definitions.
    boundaries : np.ndarray
        Unique sorted boundary values.

    Returns
    -------
    tuple of np.ndarray
        Interval and value labels for the boundaries.
    """
    interval_labels = np.full(len(boundaries) + 1, None, dtype=object)
    value_labels = np.empty(len(boundaries), dtype=object)

    # Set initial and final interval labels
    interval_labels[0], interval_labels[-1] = df.iloc[0]["label"], df.iloc[-1]["label"]

    for idx, boundary in enumerate(boundaries[:-1]):
        decision = get_decision(df, boundaries, idx)
        interval_labels[idx + 1] = decision

    for idx, boundary in enumerate(boundaries):
        decision = df[(df["value"] == boundary) & (df["op"] == "=")]
        value_labels[idx] = (
            decision.iloc[0]["label"]
            if not decision.empty
            else interval_labels[idx + 1]
        )

    return interval_labels, value_labels


def update_rule(rule_def, threshold_list=None, is_update=True):
    """
    Updates rule definitions with new thresholds.

    Parameters
    ----------
    rule_def : list
        Existing rule definitions.
    threshold_list : list, optional
        List of new thresholds to add (default is empty).
    is_update : bool, optional
        Indicates whether to update or create new rules (default is True).

    Returns
    -------
    tuple
        Updated rule definitions, boundaries, and label lists.
    """
    if threshold_list is None:
        threshold_list = []
    all_rules = [] if rule_def is None or is_update else list(np.copy(rule_def))
    all_rules.extend(threshold_list)

    # Sort and decompose operands
    sorted_rules = sort_rule(all_rules)
    decomposed_rules = decompose_operand(sorted_rules.to_dict("records"))

    # Generate boundaries and labels efficiently
    boundaries = np.unique(decomposed_rules["value"])
    interval_labels, value_labels = generate_labels(decomposed_rules, boundaries)

    # Combine interval and value labels for final label list
    label_list = [val for pair in zip(interval_labels, value_labels) for val in pair]
    label_list.append(interval_labels[-1])
    return all_rules, boundaries, label_list


def sort_rule(rule_def):
    """
    Sorts rule definitions by value and operand order.

    Parameters
    ----------
    rule_def : list
        List of rule definitions.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame of rules.
    """
    df = pd.DataFrame(rule_def)
    # logging.debug(df)
    # Convert "NA" and "Inf" to np.nan and np.inf, then convert column to numeric
    df["value"].replace({"NA": np.nan, "Inf": np.inf, "-Inf": -np.inf}, inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["operand_order"] = df["op"].map(OPERAND_MAPPING_DICT)
    df.sort_values(
        by=["value", "operand_order"], ascending=True, inplace=True, ignore_index=True
    )
    return df


def decompose_operand(rule_dict):
    """
    Decomposes operands into distinct components for comparison.

    Parameters
    ----------
    rule_dict : dict
        Dictionary of rule definitions.

    Returns
    -------
    pd.DataFrame
        DataFrame of all operands after decomposition.
    """
    # Convert the input dictionary to DataFrame and apply mapping for operand order
    df = pd.DataFrame(rule_dict)
    # Convert "NA" and "Inf" to np.nan and np.inf, then convert column to numeric
    df["value"].replace({"NA": np.nan, "Inf": np.inf, "-Inf": -np.inf}, inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["operand_order"] = df["op"].map(OPERAND_MAPPING_DICT)

    # Initialize a list to collect decomposed operands
    decomposed_operands = []

    # Identify and handle single operands without decomposition
    single_operands = df[df["operand_order"].isin([5, 3, 1])]
    decomposed_operands.append(single_operands)

    # Handle 'gte' (>=) operands by creating '>' and '=' entries
    gte_operands = df[df["operand_order"] == 4]
    if not gte_operands.empty:
        gte_operands_equal = gte_operands.copy()
        gte_operands["op"] = ">"
        gte_operands_equal["op"] = "="
        decomposed_operands.extend([gte_operands, gte_operands_equal])

    # Handle 'lte' (<=) operands by creating '<' and '=' entries
    lte_operands = df[df["operand_order"] == 2]
    if not lte_operands.empty:
        lte_operands_equal = lte_operands.copy()
        lte_operands["op"] = "<"
        lte_operands_equal["op"] = "="
        decomposed_operands.extend([lte_operands, lte_operands_equal])

    # Concatenate all decomposed operands and sort
    all_operands = pd.concat(decomposed_operands, ignore_index=True)
    return sort_rule(all_operands.to_dict("records"))


def check_unique_pair(pair):
    """
    Checks that there are no duplicate decisions.

    Parameters
    ----------
    pair : pd.DataFrame
        DataFrame containing a pair of values.

    Returns
    -------
    bool
        True if the pair is unique.
    """
    assert len(pair) <= 1, f"Duplicated decision at '{pair['value']} {pair['op']}'"
    return True


def check_conflict(decision_lt, decision_gt):
    """
    Checks for conflicts between two decisions.

    Parameters
    ----------
    decision_lt : pd.DataFrame
        Decision with less-than operand.
    decision_gt : pd.DataFrame
        Decision with greater-than operand.

    Returns
    -------
    str
        Label if no conflict, otherwise raises ValueError.
    """
    label_lt = decision_lt["label"].values[0] if not decision_lt.empty else None
    label_gt = decision_gt["label"].values[0] if not decision_gt.empty else None

    if label_lt is None:
        return label_gt
    if label_gt is None:
        return label_lt
    if label_lt != label_gt:
        raise ValueError(
            f"Conflict detected: '{decision_lt.iloc[0]['op']} {decision_lt.iloc[0]['value']}' "
            f"is {label_lt}, but '{decision_gt.iloc[0]['op']} {decision_gt.iloc[0]['value']}' "
            f"is {label_gt}."
        )
    return label_gt


def get_decision(df, boundaries, idx):
    """
    Recursively fetches the decision for a given boundary index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of rule definitions.
    boundaries : np.ndarray
        Array of boundary values.
    idx : int
        Current index of the boundary.

    Returns
    -------
    str
        Decision label or None if not found.
    """
    start_value, end_value = boundaries[idx], boundaries[idx + 1]
    decision_lt = df[(df["value"] == end_value) & (df["op"] == "<")]
    decision_gt = df[(df["value"] == start_value) & (df["op"] == ">")]

    check_unique_pair(decision_lt)
    check_unique_pair(decision_gt)
    return check_conflict(decision_lt, decision_gt)


def get_inteveral_label_list(df, boundaries):
    inteveral_label_list = np.array([None] * (len(boundaries) + 1))

    assert (
        df["op"].iloc[0] == "<"
    ), "The rule is missing a decision from -inf to " + str(df["value"].iloc[0])
    inteveral_label_list[0] = df.iloc[0]["label"]
    for idx, val in enumerate(np.arange((len(boundaries) - 1))):
        decision = get_decision(df, boundaries, idx)
        inteveral_label_list[idx + 1] = decision
    assert df["op"].iloc[-1] == ">", (
        "The rule is missing a decision from " + str(df["value"].iloc[-1]) + " to inf"
    )
    inteveral_label_list[-1] = df.iloc[-1]["label"]
    return inteveral_label_list


def get_value_label_list(df, boundaries, interval_label_list):
    value_label_list = np.array([None] * (len(boundaries)))
    for idx, val in enumerate(np.arange(len(boundaries))):
        decision = df[(df["value"] == boundaries[idx]) & (df["op"] == "=")]
        check_unique_pair(decision)
        if len(decision) == 0:
            value_label_list[idx] = interval_label_list[idx + 1]
        else:
            value_label_list[idx] = decision.iloc[0]["label"]
    return value_label_list


def cut_segment(df, milestones):
    """
    Splits a DataFrame into segments based on the start and end indices provided in the milestones DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Signal DataFrame containing the full data to be segmented.
    milestones : pd.DataFrame
        DataFrame containing 'start' and 'end' columns, representing the start and end indices for each segment.

    Returns
    -------
    list of pd.DataFrame
        A list of DataFrame segments based on the specified start and end indices.

    Raises
    ------
    AssertionError
        If milestones is not a DataFrame or if 'start' and 'end' columns are missing.
        If start and end indices are out of bounds.
    """
    # Check milestones format and columns
    assert isinstance(milestones, pd.DataFrame), (
        "Milestones must be a DataFrame with 'start' and 'end' columns. "
        "Use `format_milestone` to prepare the milestone DataFrame if needed."
    )
    assert (
        "start" in milestones.columns and "end" in milestones.columns
    ), "Milestones DataFrame must contain 'start' and 'end' columns."

    # Initialize list to hold segmented DataFrames
    segmented_dfs = []

    # Loop over each milestone row to cut segments
    for _, row in milestones.iterrows():
        start, end = int(row["start"]), int(row["end"])

        # Check that start and end are within DataFrame bounds
        if start < 0 or end > len(df):
            raise ValueError(
                f"Segment index out of bounds: start={start}, end={end}, length={len(df)}"
            )
        if start >= end:
            raise ValueError(
                f"Start index must be less than end index: start={start}, end={end}"
            )

        # Slice DataFrame for the given segment and append to list
        segmented_dfs.append(df.iloc[start:end])

    return segmented_dfs


def format_milestone(start_milestone, end_milestone):
    """
    Formats start and end milestones as a DataFrame, trimming to the minimum length if necessary.

    Parameters
    ----------
    start_milestone : array-like
        Start indices of segments.
    end_milestone : array-like
        End indices of segments.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'start' and 'end' columns, truncated to minimum length if input lengths differ.
    """
    min_length = min(len(start_milestone), len(end_milestone))

    if len(start_milestone) != len(end_milestone):
        logging.warning(
            "Mismatched lengths for start and end milestones. Truncating to minimum length: %d",
            min_length,
        )

    # Truncate both lists to the minimum length
    start_milestone = start_milestone[:min_length]
    end_milestone = end_milestone[:min_length]

    return pd.DataFrame({"start": start_milestone, "end": end_milestone})


def check_signal_format(s):
    """
    Checks and converts the input signal to a DataFrame with a 'timestamp' column if missing.

    Parameters
    ----------
    s : pd.DataFrame, list, or np.ndarray
        Input signal data. Can be a DataFrame, list, or numpy array.

    Returns
    -------
    pd.DataFrame
        Validated and formatted DataFrame with 'timestamp' and 'signal' columns.
    """
    # Convert to DataFrame if necessary
    if not isinstance(s, pd.DataFrame):
        if isinstance(s, (list, np.ndarray)):
            s = pd.DataFrame(s, columns=["signal"])
        else:
            logging.error(
                "Invalid input type. Expected DataFrame, list, or numpy array."
            )
            return None  # or handle accordingly

    # Check or create 'timestamp' column
    if "timestamps" not in s.columns or not np.issubdtype(
        s["timestamps"].dtype, np.datetime64
    ):
        s.insert(0, "timestamps", pd.to_datetime(pd.Series(range(len(s))), unit="s"))
        logging.error(
            "No valid timestamps column found. Generated 'timestamps' column based on index."
        )

    # Ensure the second column is numerical
    if not np.issubdtype(s.iloc[:, 1].dtype, np.number):
        logging.error("The signal column contains non-numeric values.")
        return None  # or handle accordingly

    return s


def create_rule_def(sqi_name, upper_bound=0, lower_bound=1):
    """
    Creates a default rule definition for SQI.

    Parameters
    ----------
    sqi_name : str
        Name of the SQI.
    upper_bound : float, optional
        Upper bound for acceptance (default is 0).
    lower_bound : float, optional
        Lower bound for acceptance (default is 1).

    Returns
    -------
    dict
        JSON-style dictionary with rule definitions.
    """
    return {
        sqi_name: {
            "name": sqi_name,
            "def": [
                {"op": ">", "value": str(lower_bound), "label": "accept"},
                {"op": "<=", "value": str(lower_bound), "label": "reject"},
                {"op": ">=", "value": str(upper_bound), "label": "reject"},
                {"op": "<", "value": str(upper_bound), "label": "accept"},
            ],
            "desc": "",
            "ref": "",
        }
    }
