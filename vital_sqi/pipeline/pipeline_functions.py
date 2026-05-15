"""Signal Quality Index (SQI) Processing and Classification Utilities"""

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.signal import resample
from vital_sqi.common.rpeak_detection import PeakDetector, ECG_DEFAULT
import vital_sqi.sqi as sq
from vital_sqi.rule import RuleSet, Rule, update_rule
from vital_sqi.common.utils import get_nn, create_rule_def, sanitize_sqi
from vital_sqi.rule.robust_classifier import classify_segments_robust, RobustResult
from vital_sqi.preprocess.preprocess_signal import taper_signal
import warnings
import logging
import inspect
from joblib import Parallel, delayed
from vital_sqi.sqi import sqi_mapping

# Cache getfullargspec results per function — introspection is static and
# called once per SQI per segment, which adds up across thousands of segments.
_argspec_cache: dict = {}
_ARGSPEC_MISS = object()  # sentinel so empty arg lists [] are still cached


def _get_arg_names(func):
    """Memoised positional-arg list for *func*. Handles 0-arg functions correctly."""
    cached = _argspec_cache.get(func, _ARGSPEC_MISS)
    if cached is _ARGSPEC_MISS:
        cached = inspect.getfullargspec(func)[0] or []
        _argspec_cache[func] = cached
    return cached


def classify_segments(
    sqis,
    rule_dict_filename,
    ruleset_order,
    auto_mode=True,
    lower_bound=0.05,
    upper_bound=0.95,
    mode="legacy",
    robust_config=None,
):
    """
    Classify each segment as ``'accept'`` or ``'reject'`` using threshold rules.

    The classifier builds a :class:`~vital_sqi.rule.RuleSet` from the rules
    named in *ruleset_order*, then runs ``RuleSet.execute`` on every segment
    row.  Rules are evaluated in ascending integer key order; the first
    ``'reject'`` short-circuits evaluation for that segment (linear early-exit,
    not recursive).

    **Auto-mode** (``auto_mode=True``, the default):
        Before classification, each rule's threshold values are replaced with
        the empirical *lower_bound* and *upper_bound* quantiles of the SQI
        values observed across all segments.  This makes the classifier
        self-adapting: it accepts the middle 90% (or whatever quantile range
        you choose) of the recording's own distribution.  Use this when you
        trust the recording but not the pre-calibrated absolute bounds.

    **Manual mode** (``auto_mode=False``):
        Thresholds stored in *rule_dict_filename* are used exactly as written.
        Use this when you want to apply externally calibrated bounds without
        adapting them to the current recording.

    Parameters
    ----------
    sqis : list of DataFrame
        One DataFrame per segment produced by :func:`extract_sqi`.  Every
        DataFrame must have the SQI column names referenced in *ruleset_order*.
    rule_dict_filename : str
        Path to a ``rule_dict.json`` file.  Each entry must have keys
        ``"name"`` (SQI column name) and ``"def"`` (list of threshold
        conditions accepted by :func:`~vital_sqi.common.utils.update_rule`).
        The calibrated file at ``vital_sqi/resource/rule_dict.json`` is the
        default starting point.
    ruleset_order : dict
        Maps integer priority keys to rule names present in the rule file,
        e.g. ``{1: "kurtosis_sqi", 2: "perfusion_sqi"}``.  Lower key =
        evaluated first.  Only rules listed here participate in classification.
    auto_mode : bool, optional
        Adjust thresholds to observed quantiles before classifying
        (default ``True``).
    lower_bound : float, optional
        Lower quantile for auto-mode threshold adjustment (default ``0.05``).
    upper_bound : float, optional
        Upper quantile for auto-mode threshold adjustment (default ``0.95``).

    Returns
    -------
    ruleset : RuleSet
        The :class:`~vital_sqi.rule.RuleSet` used for classification.
    sqis : list of DataFrame
        The input list with an added ``"decision"`` column
        (``'accept'`` or ``'reject'``) in each DataFrame.

    Raises
    ------
    FileNotFoundError
        If *rule_dict_filename* does not exist.
    KeyError
        If a rule name from *ruleset_order* is absent from the rule file.

    Examples
    --------
    >>> ruleset_order = {1: "kurtosis_sqi", 2: "perfusion_sqi"}
    >>> ruleset, sqis = classify_segments(
    ...     sqis, "vital_sqi/resource/rule_dict.json",
    ...     ruleset_order, auto_mode=True
    ... )
    >>> decisions = [df["decision"].iloc[0] for df in sqis]
    """
    if mode not in ("legacy", "robust"):
        raise ValueError(f"mode must be 'legacy' or 'robust', got {mode!r}")

    # ── Robust mode: skip rule-dict entirely ────────────────────────────────
    if mode == "robust":
        sqi_names = list(sqis[0].columns) if sqis else []
        combined = pd.concat(sqis, ignore_index=True) if sqis else pd.DataFrame()
        robust_result = classify_segments_robust(
            combined, sqi_names=sqi_names, config=robust_config
        )
        decisions = robust_result.decisions
        scores = robust_result.scores
        idx = 0
        for i, sqi_df in enumerate(sqis):
            n = len(sqi_df)
            sqi_df = sqi_df.copy()
            sqi_df["decision"] = decisions[idx: idx + n]
            sqi_df["score"] = scores[idx: idx + n]
            sqis[i] = sqi_df
            idx += n
        return robust_result, sqis

    # ── Legacy mode ─────────────────────────────────────────────────────────
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

    ruleset = None
    for i, sqi_df in enumerate(sqis):
        # Build per-channel rule list; in auto_mode thresholds are derived from
        # this channel's own distribution (not blindly from channel 0).
        rule_list = {}
        channel_rule_dict = {k: dict(v) for k, v in rule_dict.items()}

        for rule_order, rule_name in ruleset_order.items():
            sqi_name = channel_rule_dict[rule_name]["name"]

            if auto_mode:
                clean = sanitize_sqi(sqi_df[sqi_name].values)
                valid_values = clean[np.isfinite(clean)]
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
        # selected_sqi: the SQI column names that rules actually reference
        selected_sqi = [channel_rule_dict[rn]["name"] for rn in ruleset_order.values()]
        subset = sqi_df[selected_sqi]
        decisions = [
            ruleset.execute(subset.iloc[[idx]])
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
    sqi_func : callable
        SQI function with signature ``f(beat_array, **kwargs) -> scalar``.
    troughs : array-like of int
        Indices marking the start of each beat (typically returned by
        :class:`~vital_sqi.common.rpeak_detection.PeakDetector`).
        Requires at least two entries to form one beat.
    signal : array-like
        Raw signal values for a single segment.
    use_mean_beat : bool
        If ``True``, resample every beat to *mean_resample_size* samples, average
        them into one mean beat, and apply *sqi_func* once.  The single result is
        then replicated to produce one value per beat interval.
        If ``False``, apply *sqi_func* independently to each beat.
    mean_resample_size : int
        Number of samples to use when resampling beats (only relevant when
        *use_mean_beat* is ``True``).
    taper : bool, optional
        If ``True``, apply :func:`~vital_sqi.preprocess.preprocess_signal.taper_signal`
        to each beat before SQI calculation (default ``False``).
    **kwargs
        Additional keyword arguments forwarded to *sqi_func*.

    Returns
    -------
    list of float
        One SQI value per beat interval (``len(troughs) - 1`` elements in the
        normal case).  Returns ``[-np.inf]`` when fewer than two troughs are
        found or when no valid beats remain after filtering.
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
    Package a raw SQI result into a ``{column_name: value}`` dict for DataFrame insertion.

    Parameters
    ----------
    sqis : float, int, np.floating, np.ndarray, list, or dict
        Raw value(s) returned by an SQI function.
    sqi_name : str
        Base column name for this SQI.

    Returns
    -------
    dict
        Mapping of column name(s) to value(s).  Rules:

        - ``correlogram_sqi`` → single ``{"correlogram_sqi": scalar}``.
        - ``dict`` input → returned unchanged.
        - Scalar (float / int / np.floating / np.integer) → ``{sqi_name: scalar}``.
        - 1-element list or ndarray → ``{sqi_name: value}``.
        - Multi-element list or ndarray → three columns:
          ``{sqi_name_mean_sqi, sqi_name_median_sqi, sqi_name_std_sqi}``.
    """
    if sqi_name == "correlogram_sqi":
        # correlogram_sqi returns a scalar mean of top ACF peaks.
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
    _nn_intervals=None,
    _signal_values=None,
    _peak_list=None,
    _trough_list=None,
    **kwargs,
):
    """
    Compute SQI for a single signal segment.

    Parameters
    ----------
    sqi_func : callable
        SQI function to apply.
    sqi_name : str
        Identifier for this SQI, used as the column name in the output dict.
    s : DataFrame, Series, or array-like
        Signal data.  When a DataFrame is passed the second column (index 1) is
        used as the signal; a Series is converted directly; anything else is
        coerced via ``np.asarray``.
    per_beat : bool, optional
        If ``True`` perform per-beat SQI computation via
        :func:`per_beat_sqi` (default ``False``).
    use_mean_beat : bool, optional
        Passed through to :func:`per_beat_sqi`; only relevant when
        *per_beat* is ``True`` (default ``True``).
    mean_resample_size : int, optional
        Passed through to :func:`per_beat_sqi`; only relevant when
        *per_beat* is ``True`` (default ``100``).
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'``.  Controls which peak detector branch is used
        when *per_beat* is ``True``, and is forwarded to SQI functions that
        accept a *wave_type* parameter (default ``'PPG'``).
    peak_detector : int, optional
        Peak detector index (0–7) passed to
        :class:`~vital_sqi.common.rpeak_detection.PeakDetector` when
        *per_beat* is ``True`` (default ``6``).
    _signal_values : np.ndarray, optional (internal)
        Pre-extracted signal array injected by :func:`extract_segment_sqi` to
        avoid redundant array conversion.  Not intended for direct use.
    _peak_list : array-like, optional (internal)
        Pre-computed peak indices injected by :func:`extract_segment_sqi`.
    _trough_list : array-like, optional (internal)
        Pre-computed trough indices injected by :func:`extract_segment_sqi`.
    **kwargs
        Additional keyword arguments forwarded to *sqi_func*.

    Returns
    -------
    dict
        Mapping of column name(s) to scalar SQI value(s), as produced by
        :func:`get_sqi_dict`.
    """
    # Use pre-hoisted signal array when available (injected by extract_segment_sqi)
    if _signal_values is not None:
        signal_values = _signal_values
    elif isinstance(s, pd.DataFrame):
        signal_values = s.iloc[:, 1].values
    elif isinstance(s, pd.Series):
        signal_values = s.values
    else:
        signal_values = np.asarray(s)

    # Use pre-computed nn_intervals if injected, otherwise compute from signal
    spec_args = _get_arg_names(sqi_func)
    if spec_args and spec_args[0] == "nn_intervals":
        signal_values = _nn_intervals if _nn_intervals is not None else get_nn(signal_values)

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
                ecg_det = peak_detector if peak_detector >= ECG_DEFAULT else ECG_DEFAULT
                result = detector.ecg_detector(signal_values, ecg_det)
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
        # Add wave_type to kwargs if needed
        if "wave_type" in spec_args:
            kwargs["wave_type"] = wave_type
        sqi_scores = sqi_func(signal_values, **kwargs)

    sqi_score_dict = get_sqi_dict(sqi_scores, sqi_name)
    return sqi_score_dict


def extract_segment_sqi(s, sqi_list, sqi_names, sqi_arg_list, wave_type):
    """
    Extract all SQIs for a single signal segment.

    Peak detection is performed once per segment and the results are reused
    across all per-beat SQI functions via the ``_peak_list`` / ``_trough_list``
    private keyword arguments injected into :func:`get_sqi`.

    Parameters
    ----------
    s : DataFrame
        Segment signal data.  Second column (index 1) must contain the raw
        waveform values.
    sqi_list : list of callable
        SQI functions to evaluate, in the same order as *sqi_names*.
    sqi_names : list of str
        Identifiers for each SQI, matched against keys in *sqi_arg_list*.
    sqi_arg_list : dict
        Mapping of SQI name → keyword-argument dict.  Each dict is forwarded
        to :func:`get_sqi` and ultimately to the underlying SQI function.
    wave_type : str
        ``'PPG'`` or ``'ECG'``; controls peak detector branch.

    Returns
    -------
    Series
        One entry per SQI column produced (multi-element SQIs generate
        ``_mean_sqi``, ``_median_sqi``, ``_std_sqi`` columns via
        :func:`get_sqi_dict`).
    """
    sqi_scores = {}
    signal_values = s.iloc[:, 1].values

    # Compute nn_intervals once and reuse across all nn_intervals-based SQIs
    _nn_cache = None
    # Peak lists computed lazily on first per_beat SQI; reused for subsequent ones
    peak_list = None
    trough_list = None

    for sqi_func, sqi_name in zip(sqi_list, sqi_names):
        args = sqi_arg_list.get(sqi_name, {}).copy()
        args["wave_type"] = wave_type
        # Pass pre-hoisted array and cached peaks into get_sqi
        args["_signal_values"] = signal_values
        if args.get("per_beat", False):
            if peak_list is None:
                detector = PeakDetector()
                if wave_type == "PPG":
                    peak_list, trough_list = detector.ppg_detector(signal_values, args.get("peak_detector", 6))
                else:
                    result = detector.ecg_detector(signal_values, args.get("peak_detector", ECG_DEFAULT))
                    peak_list, trough_list = result[0], result[2]
            args["_peak_list"] = peak_list
            args["_trough_list"] = trough_list

        try:
            if sqi_func.__name__ == "perfusion_sqi":
                args = {"y": signal_values}
                sqi_scores.update(get_sqi(sqi_func, sqi_name, s, **args))
                continue

            _spec_args = _argspec_cache.get(sqi_func) or _argspec_cache.setdefault(
                sqi_func, inspect.getfullargspec(sqi_func)[0] or []
            )
            first_arg = _spec_args[0] if _spec_args else ""
            if first_arg == "nn_intervals":
                if _nn_cache is None:
                    _nn_cache = get_nn(signal_values)
                args["_nn_intervals"] = _nn_cache

            sqi_scores.update(get_sqi(sqi_func, sqi_name, s, **args))
        except Exception as e:
            warnings.warn(f"{sqi_func.__name__} raised exception: {e}")

    return pd.Series(sqi_scores)


def extract_sqi(segments, milestones, sqi_dict_filename, wave_type="PPG", n_jobs=1):
    """
    Extract all configured SQIs for every segment and return a result DataFrame.

    This is the top-level entry point for batch SQI extraction.  Internally it
    calls :func:`extract_segment_sqi` for each segment, which handles:

    - Routing HRV SQIs through a single cached ``get_nn()`` call per segment.
    - Routing signal-level SQIs directly to the SQI function.
    - Catching per-SQI exceptions and returning NaN for failed SQIs.

    Column names in the output follow these rules:

    - Scalar SQIs → one column named by the ``sqi_dict`` key.
    - Dict-returning SQIs (e.g. ``poincare_sqi``) → one column per dict key
      (``sd1``, ``sd2``, ``area``, ``ratio``).
    - Per-beat SQIs returning a list → three columns:
      ``{key}_mean_sqi``, ``{key}_median_sqi``, ``{key}_std_sqi``.

    Parameters
    ----------
    segments : list of DataFrame
        Segmented signal DataFrames produced by
        :func:`~vital_sqi.preprocess.segment_split.split_segment`.
        Each DataFrame must have two columns: timestamps (column 0) and
        raw waveform values (column 1).
    milestones : DataFrame
        Two-column DataFrame with ``start_idx`` and ``end_idx`` (sample
        positions in the original recording) for each segment.
    sqi_dict_filename : str
        Path to the JSON SQI configuration file.  The calibrated default is
        ``vital_sqi/resource/sqi_dict.json``.

        Format::

            {
              "user_label":  {"sqi": "registered_function_name", "args": {...}},
              "kurtosis":    {"sqi": "kurtosis_sqi", "args": {"axis": 0}},
              "poincare":    {"sqi": "poincare_sqi", "args": {}}
            }

        ``"sqi"`` must be a key in :data:`~vital_sqi.sqi.sqi_mapping`.
        ``"args"`` are keyword arguments forwarded verbatim to the SQI function.

    wave_type : str, optional
        ``'PPG'`` (default) or ``'ECG'``.  Passed to every SQI that accepts
        a ``wave_type`` parameter and controls peak detector branch selection.

    Returns
    -------
    pd.DataFrame
        One row per segment.  Columns are SQI labels from the config file
        (expanded for multi-output SQIs) plus ``start_idx`` and ``end_idx``.

    Examples
    --------
    >>> from vital_sqi.pipeline.pipeline_functions import extract_sqi
    >>> sqi_df = extract_sqi(segments, milestones,
    ...                      "vital_sqi/resource/sqi_dict.json",
    ...                      wave_type="PPG")
    >>> print(sqi_df.columns.tolist())
    ['kurtosis', 'perfusion', 'sd1', 'sd2', 'area', 'ratio', ..., 'start_idx', 'end_idx']
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
