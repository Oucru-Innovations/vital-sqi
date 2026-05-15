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


_DEFAULT_FS = 100.0


def _infer_sample_rate(segment_df, args_or_argmap) -> float:
    """Best-effort sampling-rate detection for the current segment.

    Strategy (first match wins):

    1. Derive fs from the segment's first column, assumed to be a
       timestamp (``datetime64`` or numeric seconds).  This is the most
       reliable source because it reflects the actual recording — not the
       defaults baked into ``sqi_dict.json``.
    2. Fall back to any ``sample_rate`` / ``sampling_rate`` value passed
       in the SQI args dict.
    3. Last-resort default of ``100`` Hz.

    Always returns a positive float; callers should not need to guard
    against NaN.
    """
    if hasattr(segment_df, "iloc") and segment_df.shape[1] >= 2:
        try:
            ts = segment_df.iloc[:, 0]
            if pd.api.types.is_datetime64_any_dtype(ts):
                deltas = ts.diff().dropna().dt.total_seconds().to_numpy()
            else:
                deltas = np.diff(pd.to_numeric(ts, errors="coerce").dropna().to_numpy())
            deltas = deltas[deltas > 0]
            if deltas.size:
                step = float(np.median(deltas))
                if step > 0:
                    return 1.0 / step
        except Exception:
            pass
    fs = _scan_args_for_fs(args_or_argmap)
    if fs is not None and fs > 0:
        return float(fs)
    return _DEFAULT_FS


def _scan_args_for_fs(args_or_argmap) -> "float | None":
    """Return the first sample_rate / sampling_rate value found, or None."""
    if not isinstance(args_or_argmap, dict):
        return None
    # Case 1: flat kwargs dict (called from get_sqi with a single SQI's args).
    for key in ("sample_rate", "sampling_rate"):
        if key in args_or_argmap and isinstance(args_or_argmap[key], (int, float)):
            return float(args_or_argmap[key])
    # Case 2: nested map of {name: {kwargs}} as used by extract_segment_sqi.
    for inner in args_or_argmap.values():
        if isinstance(inner, dict):
            for key in ("sample_rate", "sampling_rate"):
                if key in inner and isinstance(inner[key], (int, float)):
                    return float(inner[key])
    return None


def classify_segments(
    sqis,
    rule_dict_filename,
    ruleset_order,
    auto_mode=True,
    lower_bound=0.05,
    upper_bound=0.95,
    mode="legacy",
    robust_config=None,
    target_accept_rate=0.85,
):
    """
    Classify each segment as ``'accept'`` or ``'reject'`` using threshold rules.

    The classifier builds a :class:`~vital_sqi.rule.RuleSet` from the rules
    named in *ruleset_order*, then runs ``RuleSet.execute`` on every segment
    row.  Rules are evaluated in ascending integer key order; the first
    ``'reject'`` short-circuits evaluation for that segment (linear early-exit,
    not recursive).

    Threshold-selection strategies (``auto_mode`` argument)
    ------------------------------------------------------

    ``auto_mode=False`` or ``"manual"``
        Thresholds stored in *rule_dict_filename* are used exactly as
        written.  Use this when you want to apply externally calibrated
        bounds without adapting them to the current recording.

    ``auto_mode=True`` or ``"quantile"`` *(default)*
        Replace each rule's bounds with the empirical *lower_bound* /
        *upper_bound* quantiles of the SQI values observed across all
        segments.  Simple and predictable, but with many independent
        rules the joint accept rate can be much lower than ``upper -
        lower`` would suggest because each rule trims its own tails.

    ``auto_mode="tune"``
        Auto-tune the per-rule quantile so the *joint* accept rate
        targets *target_accept_rate* (default ``0.85``).  Under the
        independence approximation each rule keeps
        ``target ** (1/n_rules)`` of its distribution, splitting the
        trim symmetrically across both tails.  Much more forgiving than
        plain ``"quantile"`` mode when several rules are active.  See
        :func:`vital_sqi.rule.auto_threshold.per_rule_quantile` for the
        underlying math.

    Degenerate rules — SQIs whose distribution collapses to a single
    value across the recording (e.g. ``zero_crossings_rate_sqi`` on
    mean-centred PPG) — are dropped from the rule set with a warning
    instead of producing a 0-width "reject everything" band.

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
    auto_mode : bool or str, optional
        See above.  ``True`` is an alias for ``"quantile"``;
        ``False`` is an alias for ``"manual"``.  Default ``True``.
    lower_bound : float, optional
        Lower quantile for ``"quantile"`` mode (default ``0.05``).
    upper_bound : float, optional
        Upper quantile for ``"quantile"`` mode (default ``0.95``).
    target_accept_rate : float, optional
        Joint accept rate target for ``"tune"`` mode (default ``0.85``).
        Ignored unless ``auto_mode == "tune"``.

    Returns
    -------
    ruleset : RuleSet
        The :class:`~vital_sqi.rule.RuleSet` used for classification.
        Only rules with usable (non-degenerate) bands are included.
    sqis : list of DataFrame
        The input list with an added ``"decision"`` column
        (``'accept'`` or ``'reject'``) in each DataFrame.

    Raises
    ------
    FileNotFoundError
        If *rule_dict_filename* does not exist.
    KeyError
        If a rule name from *ruleset_order* is absent from the rule file.
    ValueError
        If *auto_mode* is not one of the documented values.

    Examples
    --------
    >>> ruleset_order = {1: "kurtosis_sqi", 2: "perfusion_sqi"}
    >>> ruleset, sqis = classify_segments(
    ...     sqis, "vital_sqi/resource/rule_dict.json",
    ...     ruleset_order, auto_mode="tune", target_accept_rate=0.85,
    ... )
    >>> decisions = [df["decision"].iloc[0] for df in sqis]
    """
    if mode not in ("legacy", "robust"):
        raise ValueError(f"mode must be 'legacy' or 'robust', got {mode!r}")

    # Normalise auto_mode to a canonical string so the rest of the body
    # can switch on it cleanly.  Keep backwards-compatible bool aliases.
    if auto_mode is True:
        auto_mode_norm = "quantile"
    elif auto_mode is False:
        auto_mode_norm = "manual"
    elif isinstance(auto_mode, str) and auto_mode in ("manual", "quantile", "tune"):
        auto_mode_norm = auto_mode
    else:
        raise ValueError(
            f"auto_mode must be True/False or one of "
            "'manual', 'quantile', 'tune'; got {auto_mode!r}"
        )

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

    from vital_sqi.rule.auto_threshold import (
        quantile_band,
        tuned_bands,
    )

    ruleset = None
    for i, sqi_df in enumerate(sqis):
        # Build per-channel rule list; in auto modes the thresholds are
        # derived from this channel's own distribution (not blindly from
        # channel 0).  Auto-tune mode needs the full set of columns first
        # so it can pick a per-rule quantile that hits the joint accept
        # target — we pre-compute the bands before the rule-building
        # loop in that case.
        rule_list = {}
        channel_rule_dict = {k: dict(v) for k, v in rule_dict.items()}

        # ---- precompute bands when needed ---------------------------------
        bands_by_name = {}  # rule_name → Band ('manual' mode leaves this empty)
        if auto_mode_norm == "tune":
            sqi_name_per_rule = {
                rn: channel_rule_dict[rn]["name"] for rn in ruleset_order.values()
            }
            col_values = {
                rn: sanitize_sqi(sqi_df[name].values)
                for rn, name in sqi_name_per_rule.items()
                if name in sqi_df.columns
            }
            for band in tuned_bands(col_values, target_accept_rate=target_accept_rate):
                bands_by_name[band.column] = band
        elif auto_mode_norm == "quantile":
            for rule_name in ruleset_order.values():
                sqi_name = channel_rule_dict[rule_name]["name"]
                if sqi_name not in sqi_df.columns:
                    continue
                values = sanitize_sqi(sqi_df[sqi_name].values)
                band = quantile_band(
                    sqi_name, values,
                    lower_pct=lower_bound, upper_pct=upper_bound,
                )
                if band is not None:
                    bands_by_name[rule_name] = band

        # ---- build rule list, skipping degenerate / missing entries -------
        skipped_for_channel = []
        for rule_order, rule_name in ruleset_order.items():
            sqi_name = channel_rule_dict[rule_name]["name"]

            if auto_mode_norm == "manual":
                # Honour the bounds shipped in the rule_dict verbatim.
                rule = generate_rule(sqi_name, channel_rule_dict[rule_name]["def"])
                rule_list[rule_order] = rule
                continue

            band = bands_by_name.get(rule_name)
            if band is None:
                # Either the column was missing, fewer than 2 finite values,
                # or the band collapsed to zero width — skip rather than
                # produce a rule that rejects every segment.
                skipped_for_channel.append(sqi_name)
                continue

            sqi_rule = create_rule_def(
                sqi_name, lower_bound=band.lower, upper_bound=band.upper
            )
            channel_rule_dict[rule_name]["def"] = sqi_rule[sqi_name]["def"]
            rule = generate_rule(sqi_name, channel_rule_dict[rule_name]["def"])
            rule_list[rule_order] = rule

        if skipped_for_channel:
            warnings.warn(
                f"channel {i}: dropped {len(skipped_for_channel)} degenerate rule(s): "
                f"{skipped_for_channel}",
                stacklevel=2,
            )

        if not rule_list:
            warnings.warn(
                f"channel {i}: no usable rules; every segment will be 'accept'.",
                stacklevel=2,
            )
            sqi_df = sqi_df.copy()
            sqi_df["decision"] = ["accept"] * len(sqi_df)
            sqis[i] = sqi_df
            continue

        # Renumber rule_list so the keys are consecutive starting from 1 —
        # required by RuleSet's __setattr__ validator after we may have
        # dropped rules.
        compact = {
            new_order: rule
            for new_order, (_, rule) in enumerate(sorted(rule_list.items()), start=1)
        }
        ruleset = RuleSet(compact)
        selected_sqi = [r.name for r in compact.values()]
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
        if _nn_intervals is not None:
            signal_values = _nn_intervals
        else:
            # P3 fix: forward the caller's wave_type / fs so ECG-at-256-Hz
            # doesn't get processed through the PPG-100-Hz default.
            inferred_fs = _infer_sample_rate(s, kwargs)
            signal_values = get_nn(
                signal_values,
                wave_type=wave_type,
                sample_rate=inferred_fs,
            )

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
    # Sampling rate is needed for the underlying vitalDSP RR transformer.
    # We infer it from the segment's timestamp column once per segment so we
    # don't burn time on this per SQI.
    inferred_fs = _infer_sample_rate(s, sqi_arg_list)
    # Peak lists computed lazily on first per_beat SQI; reused for subsequent ones
    peak_list = None
    trough_list = None

    for sqi_func, sqi_name in zip(sqi_list, sqi_names):
        args = sqi_arg_list.get(sqi_name, {}).copy()
        args["wave_type"] = wave_type
        # Override any baked-in sample_rate / sampling_rate with the segment's
        # actual fs so SQIs computed on non-100 Hz recordings work correctly.
        if "sample_rate" in args:
            args["sample_rate"] = inferred_fs
        if "sampling_rate" in args:
            args["sampling_rate"] = inferred_fs
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

            _spec_args = _get_arg_names(sqi_func)
            first_arg = _spec_args[0] if _spec_args else ""
            if first_arg == "nn_intervals":
                if _nn_cache is None:
                    _nn_cache = get_nn(
                        signal_values,
                        wave_type=wave_type,
                        sample_rate=inferred_fs,
                    )
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
