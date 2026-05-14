"""
Export calibrated thresholds to rule_dict.json and sqi_dict.json.

rule_dict format (one entry per SQI)
-------------------------------------
Each entry uses the paired operator structure required by the rule engine:

    {
      "sqi_name": {
        "name": "sqi_name",
        "def": [
          {"op": ">",  "value": "<lower>", "label": "accept"},
          {"op": "<=", "value": "<lower>", "label": "reject"},
          {"op": ">=", "value": "<upper>", "label": "reject"},
          {"op": "<",  "value": "<upper>", "label": "accept"}
        ],
        "desc": "Calibrated from synthetic <wave_type> signals. Accept: p<lo>-p<hi>.",
        "ref":  "vital_sqi.calibration"
      }
    }

The exporter merges new calibrated entries INTO the existing rule_dict rather
than replacing it, so manually curated entries (for SQIs the calibrator
cannot reach) are preserved.
"""

import json
import os
import shutil
from datetime import datetime
from typing import Optional

from vital_sqi.calibration.threshold_estimator import SQIThreshold


# ---------------------------------------------------------------------------
# rule_dict exporter
# ---------------------------------------------------------------------------

def _make_rule_def_entry(threshold: SQIThreshold, wave_type: str,
                          lower_pct: float, upper_pct: float) -> dict:
    """Build a single rule_dict entry from a calibrated SQIThreshold."""
    lower_str = f"{threshold.lower:.6g}"
    upper_str = f"{threshold.upper:.6g}"
    desc = (
        f"Calibrated from synthetic {wave_type} signals. "
        f"Accept region: p{lower_pct:.0f}={threshold.lower:.4g} to "
        f"p{upper_pct:.0f}={threshold.upper:.4g} "
        f"(median={threshold.accept_median:.4g}, n={threshold.n_accept}). "
        f"{threshold.note}"
    ).strip()

    return {
        "name": threshold.sqi_name,
        "def": [
            {"op": ">",  "value": lower_str, "label": "accept"},
            {"op": "<=", "value": lower_str, "label": "reject"},
            {"op": ">=", "value": upper_str, "label": "reject"},
            {"op": "<",  "value": upper_str, "label": "accept"},
        ],
        "desc": desc,
        "ref": "vital_sqi.calibration",
    }


def export_rule_dict(
    thresholds: dict,
    output_path: str,
    wave_type: str = "PPG",
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    backup: bool = True,
) -> None:
    """
    Write or update a rule_dict.json file with calibrated thresholds.

    Existing entries whose SQI was NOT calibrated are preserved unchanged.
    Calibrated entries overwrite their previous counterparts.

    Parameters
    ----------
    thresholds : dict
        Output of :func:`~vital_sqi.calibration.threshold_estimator.estimate_thresholds`.
        Only entries with ``calibrated=True`` are written.
    output_path : str
        Destination file path (created if absent).
    wave_type : str
        Used in the ``desc`` field for documentation purposes.
    lower_pct, upper_pct : float
        Percentiles used during estimation (for documentation).
    backup : bool
        If ``True`` and the file already exists, a timestamped backup is made
        before overwriting (default ``True``).
    """
    # Load existing file (if any)
    existing = {}
    if os.path.isfile(output_path):
        if backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.replace(".json", f"_backup_{ts}.json")
            shutil.copy2(output_path, backup_path)
        with open(output_path) as f:
            existing = json.load(f)

    # Merge calibrated entries
    # Note: dict-returning SQIs (e.g. poincare_sqi -> sd1/sd2/area/ratio) produce
    # bare sub-column names.  We write them as-is so rule_dict entries match the
    # actual DataFrame column names produced by extract_sqi.
    n_updated = 0
    for name, t in thresholds.items():
        if not t.calibrated:
            continue
        existing[name] = _make_rule_def_entry(t, wave_type, lower_pct, upper_pct)
        n_updated += 1

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[exporter] rule_dict: {n_updated} entries written -> {output_path}")


# ---------------------------------------------------------------------------
# sqi_dict exporter
# ---------------------------------------------------------------------------

# Argument templates for each SQI — defines what goes in sqi_dict.json "args".
# Only SQIs that appear here will be emitted to sqi_dict.json.
_SQI_ARG_TEMPLATES = {
    "perfusion_sqi":           {},
    "kurtosis_sqi":            {"axis": 0, "fisher": True, "bias": True,
                                 "nan_policy": "propagate"},
    "skewness_sqi":            {"axis": 0, "bias": True,
                                 "nan_policy": "propagate"},
    "entropy_sqi":             {"qk": None, "base": None},
    "signal_to_noise_sqi":     {"axis": 0, "ddof": 0},
    "zero_crossings_rate_sqi": {"threshold": 1e-10, "ref_magnitude": None,
                                 "axis": -1},
    "mean_crossing_rate_sqi":  {"threshold": 1e-10, "ref_magnitude": None,
                                 "pad": True, "axis": -1},
    "ectopic_sqi":             {"rule_index": 0, "sample_rate": 100,
                                 "rpeak_detector": 6, "low_rri": 300, "high_rri": 2000},
    "correlogram_sqi":         {"sample_rate": 100, "time_lag": 3, "n_selection": 3},
    "msq_sqi":                 {"peak_detector_1": 7, "peak_detector_2": 6},
    "band_energy_sqi":         {"sampling_rate": 100, "band": None},
    "lfe_sqi":                 {"sampling_rate": 100, "band": [0, 0.5]},
    "qrse_sqi":                {"sampling_rate": 100, "band": [5, 25]},
    "hfe_sqi":                 {"sampling_rate": 100, "band": [100, 1000]},
    "vhfp_sqi":                {"sampling_rate": 100, "band": [150, 1000]},
    "qrsa_sqi":                {"sampling_rate": 100},
    "dtw_sqi":                 {"template_type": 0, "template_size": 100},
    "sdnn_sqi":                {},
    "sdsd_sqi":                {},
    "rmssd_sqi":               {},
    "cvsd_sqi":                {},
    "cvnn_sqi":                {},
    "mean_nn_sqi":             {},
    "median_nn_sqi":           {},
    "pnn_sqi":                 {},
    "hr_mean_sqi":             {},
    "hr_median_sqi":           {},
    "hr_min_sqi":              {},
    "hr_max_sqi":              {},
    "hr_std_sqi":              {},
    "hr_range_sqi":            {"range_min": 40, "range_max": 200},
    "peak_frequency_sqi":      {"f_min": 0.04, "f_max": 0.15},
    "absolute_power_sqi":      {"f_min": 0.04, "f_max": 0.15},
    "log_power_sqi":           {"f_min": 0.04, "f_max": 0.15},
    "relative_power_sqi":      {"f_min": 0.04, "f_max": 0.15},
    "normalized_power_sqi":    {"f_min": 0.04, "f_max": 0.15},
    "lf_hf_ratio_sqi":         {},
    "poincare_sqi":            {},
    # new SQIs (P3/P4)
    "clipping_sqi":                {},
    "baseline_wander_sqi":         {"sampling_rate": 100},
    "spectral_snr_sqi":            {"sampling_rate": 100, "signal_band": [0.5, 4.0]},
    "amplitude_consistency_sqi":   {"sample_rate": 100},
    "rr_irregularity_sqi":         {},
    "sample_entropy_sqi":          {"m": 2},
    "dfa_sqi":                     {},
    "hurst_sqi":                   {},
}


def export_sqi_dict(
    thresholds: dict,
    output_path: str,
    backup: bool = True,
) -> None:
    """
    Write a sqi_dict.json containing only successfully calibrated SQIs.

    Entries for SQIs that could not be calibrated are excluded so the output
    template is safe to use directly in ``extract_sqi``.

    Parameters
    ----------
    thresholds : dict
        Output of :func:`~vital_sqi.calibration.threshold_estimator.estimate_thresholds`.
    output_path : str
        Destination file path.
    backup : bool
        If the file exists, make a timestamped backup before overwriting.
    """
    if os.path.isfile(output_path) and backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.replace(".json", f"_backup_{ts}.json")
        shutil.copy2(output_path, backup_path)

    sqi_dict = {}
    for sqi_name, args in _SQI_ARG_TEMPLATES.items():
        # Include the SQI if it was successfully calibrated OR has a template
        # (poincare returns a dict so its sub-columns appear under different names)
        t = thresholds.get(sqi_name)
        calibrated = t is not None and t.calibrated

        # poincare_sqi returns a dict with keys sd1/sd2/area/ratio (bare, no prefix)
        if not calibrated and sqi_name == "poincare_sqi":
            calibrated = any(
                k in ("sd1", "sd2", "area", "ratio") and v.calibrated
                for k, v in thresholds.items()
            )

        if calibrated:
            sqi_dict[sqi_name] = {"sqi": sqi_name, "args": args}

    with open(output_path, "w") as f:
        json.dump(sqi_dict, f, indent=2)

    print(f"[exporter] sqi_dict: {len(sqi_dict)} entries written -> {output_path}")


# ---------------------------------------------------------------------------
# CSV diagnostics report
# ---------------------------------------------------------------------------

def export_diagnostics(thresholds: dict, output_path: str) -> None:
    """
    Write a human-readable CSV report of all calibration results.

    Parameters
    ----------
    thresholds : dict
        Output of :func:`~vital_sqi.calibration.threshold_estimator.estimate_thresholds`.
    output_path : str
        Destination CSV file path.
    """
    from vital_sqi.calibration.threshold_estimator import thresholds_to_dataframe
    df = thresholds_to_dataframe(thresholds)
    df.to_csv(output_path)
    print(f"[exporter] diagnostics CSV written -> {output_path}")
