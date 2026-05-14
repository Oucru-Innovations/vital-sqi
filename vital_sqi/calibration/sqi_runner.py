"""
Batch SQI computation over synthetic signal segments.

``compute_sqi_distributions`` is the main entry point.  It takes a list of
(signal, fs) tuples, wraps each into the DataFrame format expected by
``extract_segment_sqi``, runs all configured SQIs, and returns the raw
per-segment SQI values as a DataFrame — one row per segment.

The function handles three special cases transparently:
- dict-returning SQIs (e.g. ``poincare_sqi``) are flattened to multiple columns
- SQIs that raise exceptions emit NaN for that segment (logged, not re-raised)
- nn_intervals-based SQIs are detected and handled by the existing pipeline logic
"""

import numpy as np
import pandas as pd
import warnings
import logging
from tqdm import tqdm
from joblib import Parallel, delayed
from vital_sqi.common.utils import generate_timestamp
from vital_sqi.pipeline.pipeline_functions import extract_segment_sqi
from vital_sqi.sqi import sqi_mapping


# Default SQI configuration used when no custom sqi_arg_list is provided.
# Covers every function in sqi_mapping with sensible defaults.
DEFAULT_SQI_ARG_LIST = {
    "perfusion_sqi":        {},
    "kurtosis_sqi":         {"axis": 0, "fisher": True, "bias": True,
                              "nan_policy": "propagate"},
    "skewness_sqi":         {"axis": 0, "bias": True,
                              "nan_policy": "propagate"},
    "entropy_sqi":          {"qk": None, "base": None},
    "signal_to_noise_sqi":  {"axis": 0, "ddof": 0},
    "zero_crossings_rate_sqi": {"threshold": 1e-10, "ref_magnitude": None,
                                 "axis": -1},
    "mean_crossing_rate_sqi":  {"threshold": 1e-10, "ref_magnitude": None,
                                 "pad": True, "axis": -1},
    "ectopic_sqi":          {"rule_index": 0, "sample_rate": 100,
                              "rpeak_detector": 6, "low_rri": 300, "high_rri": 2000},
    "correlogram_sqi":      {"sample_rate": 100, "time_lag": 3, "n_selection": 3},
    "interpolation_sqi":    {},
    "msq_sqi":              {"peak_detector_1": 7, "peak_detector_2": 6},
    "band_energy_sqi":      {"sampling_rate": 100, "band": None},
    "lfe_sqi":              {"sampling_rate": 100, "band": [0, 0.5]},
    "qrse_sqi":             {"sampling_rate": 100, "band": [5, 25]},
    "hfe_sqi":              {"sampling_rate": 100, "band": [100, 1000]},
    "vhfp_sqi":             {"sampling_rate": 100, "band": [150, 1000]},
    "qrsa_sqi":             {"sampling_rate": 100},
    "dtw_sqi":              {"template_type": 0, "template_size": 100},
    "sdnn_sqi":             {},
    "sdsd_sqi":             {},
    "rmssd_sqi":            {},
    "cvsd_sqi":             {},
    "cvnn_sqi":             {},
    "mean_nn_sqi":          {},
    "median_nn_sqi":        {},
    "pnn_sqi":              {},
    "hr_mean_sqi":          {},
    "hr_median_sqi":        {},
    "hr_min_sqi":           {},
    "hr_max_sqi":           {},
    "hr_std_sqi":           {},
    "hr_range_sqi":         {"range_min": 40, "range_max": 200},
    "peak_frequency_sqi":   {"f_min": 0.04, "f_max": 0.15},
    "absolute_power_sqi":   {"f_min": 0.04, "f_max": 0.15},
    "log_power_sqi":        {"f_min": 0.04, "f_max": 0.15},
    "relative_power_sqi":   {"f_min": 0.04, "f_max": 0.15},
    "normalized_power_sqi": {"f_min": 0.04, "f_max": 0.15},
    "lf_hf_ratio_sqi":      {},
    "poincare_sqi":         {},
    # new SQIs (P3/P4)
    "clipping_sqi":             {},
    "baseline_wander_sqi":      {"sampling_rate": 100},
    "spectral_snr_sqi":         {"sampling_rate": 100, "signal_band": [0.5, 4.0]},
    "amplitude_consistency_sqi": {"sample_rate": 100},
    "rr_irregularity_sqi":      {},
    "sample_entropy_sqi":       {"m": 2},
    "dfa_sqi":                  {},
    "hurst_sqi":                {},
}


def _make_segment_df(signal: np.ndarray, fs: int) -> pd.DataFrame:
    """Wrap a raw signal array into the two-column DataFrame format."""
    timestamps = generate_timestamp(None, fs, len(signal))
    return pd.DataFrame({"time": timestamps, "signal": signal})


def _compute_one_segment(signal, fs, sqi_funcs, valid_names, valid_args, wave_type):
    """Compute all SQIs for a single segment. Used by joblib workers."""
    seg_df = _make_segment_df(signal, fs)
    patched_args = _patch_fs(valid_args, fs)
    try:
        return extract_segment_sqi(seg_df, sqi_funcs, valid_names, patched_args, wave_type)
    except Exception as e:
        logging.warning(f"extract_segment_sqi failed: {e}")
        return pd.Series({n: np.nan for n in valid_names})


def compute_sqi_distributions(
    segments,
    wave_type="PPG",
    sqi_names=None,
    sqi_arg_list=None,
    show_progress=True,
    n_jobs=1,
):
    """
    Compute SQIs for every segment and return a DataFrame of raw values.

    Parameters
    ----------
    segments : list of tuple
        Each element is ``(signal_array: np.ndarray, fs: int)`` as produced by
        :func:`~vital_sqi.calibration.signal_generator.generate_clean_ppg` or
        :func:`~vital_sqi.calibration.signal_generator.generate_clean_ecg`.
    wave_type : str, optional
        ``'PPG'`` or ``'ECG'`` (default ``'PPG'``).
    sqi_names : list of str, optional
        Subset of SQI names to compute.  Defaults to all keys in
        :data:`DEFAULT_SQI_ARG_LIST`.
    sqi_arg_list : dict, optional
        Custom argument dict keyed by SQI name.  Any key not present falls
        back to :data:`DEFAULT_SQI_ARG_LIST`.
    show_progress : bool
        Show tqdm progress bar (default ``True``).
    n_jobs : int, optional
        Number of parallel workers for joblib.  ``1`` (default) runs
        sequentially; ``-1`` uses all available CPU cores.

    Returns
    -------
    pd.DataFrame
        One row per segment, one column per SQI output.  Multi-value SQIs
        (poincare) produce multiple columns.  Failed/errored SQIs are NaN.
    """
    if sqi_names is None:
        sqi_names = list(DEFAULT_SQI_ARG_LIST.keys())

    merged_args = {**DEFAULT_SQI_ARG_LIST, **(sqi_arg_list or {})}

    # Build parallel lists expected by extract_segment_sqi
    sqi_funcs = []
    valid_names = []
    valid_args = {}
    for name in sqi_names:
        if name not in sqi_mapping:
            warnings.warn(f"SQI '{name}' not in sqi_mapping -- skipping.")
            continue
        sqi_funcs.append(sqi_mapping[name])
        valid_names.append(name)
        valid_args[name] = merged_args.get(name, {})

    if n_jobs == 1:
        rows = []
        iterator = tqdm(segments, desc="Computing SQIs", disable=not show_progress)
        for signal, fs in iterator:
            rows.append(_compute_one_segment(
                signal, fs, sqi_funcs, valid_names, valid_args, wave_type
            ))
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_one_segment)(
                signal, fs, sqi_funcs, valid_names, valid_args, wave_type
            )
            for signal, fs in tqdm(segments, desc="Computing SQIs", disable=not show_progress)
        )

    df = pd.DataFrame(rows).reset_index(drop=True)
    # Replace inf/-inf with NaN so percentile calculations are clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _patch_fs(sqi_arg_list: dict, fs: int) -> dict:
    """
    Update any ``sample_rate`` or ``sampling_rate`` argument to the actual fs.
    Returns a shallow copy — does not mutate the input dict.
    """
    patched = {}
    for name, args in sqi_arg_list.items():
        a = dict(args)
        if "sample_rate" in a:
            a["sample_rate"] = fs
        if "sampling_rate" in a:
            a["sampling_rate"] = fs
        patched[name] = a
    return patched
