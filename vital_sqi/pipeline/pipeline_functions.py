"""Signal Quality Index (SQI) Processing and Classification Utilities"""

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.signal import resample
from vital_sqi.common.rpeak_detection import PeakDetector
import vital_sqi.sqi as sq
from vital_sqi.rule import RuleSet, Rule, update_rule
from vital_sqi.common.utils import get_nn, create_rule_def
from vital_sqi.preprocess.preprocess_signal import taper_signal
import warnings
import logging
import inspect
from vital_sqi.sqi import sqi_mapping
from joblib import Parallel, delayed


def classify_segments(
    sqis,
    rule_dict_filename,
    ruleset_order,
    auto_mode=True,
    lower_bound=0.05,
    upper_bound=0.95,
):
    """
    Classify each segment based on SQI thresholds and return the decision per segment.

    Parameters
    ----------
    sqis : DataFrame
        A DataFrame containing SQI values for each segment.
    rule_dict_filename : str
        Path to the JSON file defining thresholds for each SQI.
    ruleset_order : dict
        Specifies the order of the rules in the ruleset.
    auto_mode : bool
        Enables automatic threshold adjustment based on quantiles.
    lower_bound, upper_bound : float
        Quantiles for the lower and upper bounds of threshold adjustment.

    Returns
    -------
    rule_list : dict
        Dictionary containing rule names and corresponding Rule objects.
    sqis : DataFrame
        Updated DataFrame with decisions ('accept' or 'reject') for each segment.
    """
    # Load rule dictionary
    try:
        with open(rule_dict_filename, "r") as f:
            rule_dict = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Rule dictionary file not found: {rule_dict_filename}"
        ) from e

    # Validate all rule names exist before processing any channel
    for rule_name in ruleset_order.values():
        if rule_name not in rule_dict:
            raise KeyError(
                f"Rule '{rule_name}' not found in rule_dict. "
                f"Available rules: {list(rule_dict.keys())}"
            )

    # Build the list of SQI column names (not rule names) for DataFrame indexing.
    # rule_dict[rule_name]["name"] is the actual SQI column name in the DataFrame.
    selected_sqi = [rule_dict[rule_name]["name"] for rule_name in ruleset_order.values()]

    for i, sqi_df in enumerate(sqis):
        # Build per-channel rule list; in auto_mode thresholds are derived from
        # this channel's own distribution (not blindly from channel 0).
        rule_list = {}
        channel_rule_dict = {k: dict(v) for k, v in rule_dict.items()}

        for rule_order, rule_name in ruleset_order.items():
            sqi_name = channel_rule_dict[rule_name]["name"]

            if auto_mode:
                valid_values = (
                    sqi_df[sqi_name]
                    .replace([np.inf, -np.inf, np.nan], np.nan)
                    .dropna()
                )
                if len(valid_values) == 0:
                    warnings.warn(
                        f"No valid values for '{sqi_name}' in channel {i}; "
                        "skipping auto-mode for this rule."
                    )
                else:
                    lower_unit = np.quantile(valid_values, lower_bound)
                    upper_unit = np.quantile(valid_values, upper_bound)
                    sqi_rule = create_rule_def(
                        sqi_name, lower_bound=lower_unit, upper_bound=upper_unit
                    )
                    channel_rule_dict[rule_name]["def"] = sqi_rule[sqi_name]["def"]

            # Create the Rule using the SQI column name so that RuleSet.execute
            # can look up the value by rule.name in the SQI DataFrame.
            rule = generate_rule(sqi_name, channel_rule_dict[rule_name]["def"])
            rule_list[rule_order] = rule

        ruleset = RuleSet(rule_list)
        decisions = [
            ruleset.execute(
                pd.DataFrame(dict(sqi_df[selected_sqi].iloc[idx]), index=[0])
            )
            for idx in range(len(sqi_df))
        ]
        sqi_df["decision"] = decisions
        sqis[i] = sqi_df

    return ruleset, sqis


def get_reject_segments(segments, wave_type):
    """
    Return accept/reject decisions for each segment based on wave type.

    Parameters
    ----------
    segments : list
        List of signal DataFrames.
    wave_type : str
        Type of waveform ('PPG' or 'ECG').

    Returns
    -------
    Series
        Series with 'accept' or 'reject' for each segment.
    """
    return pd.Series(["accept"] * len(segments))


def map_decision(decision):
    """
    Map decision string to integer for processing.

    Parameters
    ----------
    decision : str
        'accept' or 'reject'

    Returns
    -------
    int
        0 for 'accept', 1 for 'reject'
    """
    return 0 if decision == "accept" else 1


def get_decision_segments(segments, decision, reject_decision):
    """
    Separate accepted and rejected segments based on decisions.

    Parameters
    ----------
    segments : list
        List of all segments.
    decision : list
        Decisions from SQI evaluation ('accept'/'reject').
    reject_decision : list
        Additional rejection criteria.

    Returns
    -------
    tuple of lists
        Accepted and rejected segments.
    """
    # Ensure inputs are of the same length
    if not (len(segments) == len(decision) == len(reject_decision)):
        raise ValueError(
            f"Length mismatch: segments={len(segments)}, decision={len(decision)}, reject_decision={len(reject_decision)}"
        )

    combined_decision = [
        "reject" if (d == "reject" or r == "reject") else "accept"
        for d, r in zip(decision, reject_decision)
    ]
    accepted = [seg for seg, d in zip(segments, combined_decision) if d == "accept"]
    rejected = [seg for seg, d in zip(segments, combined_decision) if d == "reject"]
    return accepted, rejected


def per_beat_sqi(
    sqi_func, troughs, signal, use_mean_beat, mean_resample_size, taper=False, **kwargs
):
    """
    Compute SQI per beat by dividing the signal based on trough indices.

    Parameters
    ----------
    sqi_func : function
        Function for calculating SQI per beat.
    troughs : array-like
        Indices marking the start of each beat.
    signal : array-like
        Signal values for a single segment.
    use_mean_beat : bool
        Whether to use a resampled mean beat for SQI calculation.
    mean_resample_size : int
        Resample size for mean beat.
    taper : bool
        Whether to taper each beat.

    Returns
    -------
    list
        SQI values per beat.
    """
    if len(troughs) < 2:
        logging.warning("Not enough troughs to compute beats.")
        return [-np.inf]

    sqi_vals = []
    beat_list = []

    for idx in range(len(troughs) - 1):
        single_beat = signal[troughs[idx] : troughs[idx + 1]]
        if len(single_beat) == 0:
            continue
        if taper:
            single_beat = taper_signal(single_beat)

        if use_mean_beat:
            beat_list.append(resample(single_beat, mean_resample_size))
        else:
            sqi = sqi_func(single_beat, **kwargs)
            sqi_vals.append(sqi)

    if use_mean_beat and beat_list:
        mean_beat = np.mean(np.array(beat_list), axis=0)
        sqi = sqi_func(mean_beat, **kwargs)
        sqi_vals.append(sqi)  # Single value for mean-beat mode

    if not sqi_vals:
        logging.warning("No valid beats found for SQI calculation.")
        return [-np.inf]

    return sqi_vals


def get_sqi_dict(sqis, sqi_name):
    """
    Map SQI name with computed values based on SQI type.

    Parameters
    ----------
    sqis : various
        SQI values.
    sqi_name : str
        Name of the SQI.

    Returns
    -------
    dict
        Mapped SQI values.
    """
    if sqi_name == "correlogram_sqi":
        # correlogram_sqi returns a scalar mean of top ACF peaks.
        # Expose as a single named column; no multi-value expansion.
        scalar = sqis[0] if isinstance(sqis, (list, np.ndarray)) else sqis
        return {"correlogram_sqi": scalar}

    if isinstance(sqis, dict):
        return sqis

    if isinstance(sqis, (float, int, np.floating, np.integer)):
        return {sqi_name: sqis}

    if isinstance(sqis, np.ndarray):
        sqis = sqis.tolist()

    if isinstance(sqis, list):
        if len(sqis) == 1:
            return {sqi_name: sqis[0]}
        return {
            f"{sqi_name}_mean_sqi": np.mean(sqis),
            f"{sqi_name}_median_sqi": np.median(sqis),
            f"{sqi_name}_std_sqi": np.std(sqis),
        }

    return {sqi_name: sqis}


def get_sqi(
    sqi_func,
    sqi_name,
    s,
    per_beat=False,
    use_mean_beat=True,
    mean_resample_size=100,
    wave_type="PPG",
    peak_detector=6,
    _signal_values=None,
    _peak_list=None,
    _trough_list=None,
    **kwargs,
):
    """
    Compute SQI for a signal segment using the specified function and configuration.

    Parameters
    ----------
    sqi_func : function
        Function to calculate SQI.
    sqi_name : str
        Name of the SQI.
    s : DataFrame
        Signal data.
    per_beat : bool, optional
        Whether to calculate SQI per beat.
    use_mean_beat : bool, optional
        Whether to use mean beat for SQI calculation.
    mean_resample_size : int, optional
        Resample size for mean beat.
    wave_type : str, optional
        Waveform type ('PPG' or 'ECG').
    peak_detector : int, optional
        Peak detector mode (1-7).
    _signal_values : np.ndarray, optional
        Pre-extracted signal array (skips DataFrame conversion when provided).
    _peak_list : array-like, optional
        Pre-computed peak indices (skips peak detection when provided).
    _trough_list : array-like, optional
        Pre-computed trough indices (skips peak detection when provided).

    Returns
    -------
    dict
        Calculated SQI values.
    """
    # P3.2: use pre-hoisted array when available, otherwise extract from s
    if _signal_values is not None:
        signal_values = _signal_values
    elif isinstance(s, pd.DataFrame):
        signal_values = s.iloc[:, 1].values
    elif isinstance(s, pd.Series):
        signal_values = s.values
    else:
        signal_values = np.asarray(s)

    if inspect.getfullargspec(sqi_func)[0][0] == "nn_intervals":
        signal_values = get_nn(signal_values)

    if per_beat:
        # P3.1: use cached peaks when available; only detect if not provided
        if _peak_list is not None and _trough_list is not None:
            trough_list = _trough_list
        else:
            detector = PeakDetector()
            if wave_type == "PPG":
                _peak_list, trough_list = detector.ppg_detector(
                    signal_values, peak_detector
                )
            else:
                result = detector.ecg_detector(signal_values, peak_detector)
                _peak_list, trough_list = result[0], result[2]  # r_peaks, s_valleys
        sqi_scores = per_beat_sqi(
            sqi_func,
            trough_list,
            signal_values,
            use_mean_beat,
            mean_resample_size,
            **kwargs,
        )
    else:
        if "wave_type" in inspect.getfullargspec(sqi_func)[0]:
            kwargs["wave_type"] = wave_type
        sqi_scores = sqi_func(signal_values, **kwargs)

    sqi_score_dict = get_sqi_dict(sqi_scores, sqi_name)
    return sqi_score_dict


def extract_segment_sqi(s, sqi_list, sqi_names, sqi_arg_list, wave_type):
    """
    Extract SQIs for a single segment.

    Parameters
    ----------
    s : DataFrame
        Segment signal data.
    sqi_list : list
        List of SQI functions.
    sqi_names : list
        Names of SQIs.
    sqi_arg_list : dict
        Arguments for each SQI.
    wave_type : str
        Type of waveform ('PPG' or 'ECG').

    Returns
    -------
    Series
        Calculated SQI values.
    """
    # P3.2: hoist DataFrame→array conversion once for the whole segment
    if isinstance(s, pd.DataFrame):
        signal_values = s.iloc[:, 1].values
    elif isinstance(s, pd.Series):
        signal_values = s.values
    else:
        signal_values = np.asarray(s)

    # P3.1: detect peaks once for the segment; reused by all per-beat SQIs
    peak_list, trough_list = None, None
    needs_peaks = any(
        sqi_arg_list.get(name, {}).get("per_beat", False) for name in sqi_names
    )
    if needs_peaks:
        try:
            detector = PeakDetector(wave_type=wave_type)
            if wave_type == "PPG":
                peak_list, trough_list = detector.ppg_detector(signal_values)
            else:
                result = detector.ecg_detector(signal_values)
                peak_list, trough_list = result[0], result[2]  # r_peaks, s_valleys
        except Exception as e:
            warnings.warn(f"Peak detection failed for segment: {e}")

    sqi_scores = {}
    for sqi_func, sqi_name in zip(sqi_list, sqi_names):
        args = sqi_arg_list.get(sqi_name, {}).copy()
        args["wave_type"] = wave_type
        # Pass pre-hoisted array and cached peaks into get_sqi
        args["_signal_values"] = signal_values
        if args.get("per_beat", False):
            args["_peak_list"] = peak_list
            args["_trough_list"] = trough_list

        try:
            if sqi_func.__name__ == "perfusion_sqi":
                # perfusion_sqi(x, y): x=raw signal (col 0), y=processed signal (col 1).
                # Inject y explicitly; x is passed as the first positional arg by get_sqi.
                args["y"] = np.array(s.iloc[:, 1]) if isinstance(s, pd.DataFrame) else signal_values
            sqi_scores.update(get_sqi(sqi_func, sqi_name, s, **args))
        except Exception as e:
            warnings.warn(f"{sqi_func.__name__} raised exception: {e}")

    return pd.Series(sqi_scores)


def extract_sqi(segments, milestones, sqi_dict_filename, wave_type="PPG", n_jobs=1):
    """
    Extract SQIs for multiple segments based on SQI dictionary.

    Parameters
    ----------
    segments : list
        List of segments.
    milestones : DataFrame
        Milestone indices for segments.
    sqi_dict_filename : str
        Path to SQI configuration file.
    wave_type : str, optional
        Type of waveform ('PPG' or 'ECG').
    n_jobs : int, optional
        Number of parallel jobs (default 1). Use -1 for all cores.
        Requires joblib.

    Returns
    -------
    DataFrame
        Extracted SQIs for each segment.
    """
    with open(sqi_dict_filename, "r") as arg_file:
        sqi_dict = json.load(arg_file)

    # Extract SQI function mappings, names, and arguments
    sqi_list = [sqi_mapping[sqi["sqi"]] for sqi in sqi_dict.values()]
    sqi_names = list(sqi_dict.keys())
    sqi_arg_list = {name: sqi["args"] for name, sqi in sqi_dict.items()}

    # P3.3: optional parallel execution; n_jobs=1 preserves serial behaviour
    if n_jobs == 1:
        sqi_rows = [
            extract_segment_sqi(seg, sqi_list, sqi_names, sqi_arg_list, wave_type)
            for seg in tqdm(segments)
        ]
    else:
        sqi_rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(extract_segment_sqi)(
                seg, sqi_list, sqi_names, sqi_arg_list, wave_type
            )
            for seg in tqdm(segments)
        )

    # P3.5: build DataFrame directly from collected rows (no per-segment append)
    df_sqi = pd.DataFrame(sqi_rows)

    # Add start and end indices from milestones
    df_sqi["start_idx"] = milestones.iloc[:, 0].values
    df_sqi["end_idx"] = milestones.iloc[:, 1].values

    return df_sqi


def generate_rule(rule_name, rule_def):
    """
    Generate a Rule object from rule definition.

    Parameters
    ----------
    rule_name : str
        Rule name.
    rule_def : dict
        Rule definitions.

    Returns
    -------
    Rule
        Created rule object.
    """
    rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
    return Rule(
        rule_name, {"def": rule_def, "boundaries": boundaries, "labels": label_list}
    )
