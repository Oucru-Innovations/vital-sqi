"""
Derive accept/reject thresholds from empirical SQI distributions.

The core algorithm
------------------
1.  ``accept_df``  — SQI values from clean signals (noise_floor ≈ 0)
2.  ``reject_df``  — SQI values from heavily degraded signals

For each SQI column:

    lower = percentile(accept_df[col], lower_pct)   # e.g. p5
    upper = percentile(accept_df[col], upper_pct)   # e.g. p95

The accept region is the open interval ``(lower, upper)``.

Edge cases
----------
- All-NaN column         → SQI is skipped (not calibratable)
- Constant column        → warns and widens the band by a small epsilon
- Accept and reject      → the reject distribution is stored for reference
  distributions overlap    but does not change the thresholds (clean-signal
                           distribution is authoritative)
- Very small range       → epsilon guard prevents a zero-width rule
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional


_EPSILON = 1e-6   # minimum half-width for a non-degenerate band


@dataclass
class SQIThreshold:
    """
    Calibrated threshold for a single SQI column.

    Attributes
    ----------
    sqi_name : str
        Column name as it appears in the SQI DataFrame.
    lower : float
        Lower bound of the accept region (exclusive, ``>`` operator).
    upper : float
        Upper bound of the accept region (exclusive, ``<`` operator).
    accept_median : float
        Median of the accept (clean) distribution.
    accept_std : float
        Standard deviation of the accept distribution.
    reject_median : float or None
        Median of the reject distribution (NaN if not available).
    n_accept : int
        Number of valid (non-NaN) accept samples used.
    n_reject : int
        Number of valid (non-NaN) reject samples used.
    calibrated : bool
        False if the SQI could not be calibrated (all-NaN, constant, etc.).
    note : str
        Human-readable note about any special handling applied.
    """
    sqi_name: str
    lower: float = np.nan
    upper: float = np.nan
    accept_median: float = np.nan
    accept_std: float = np.nan
    reject_median: float = np.nan
    n_accept: int = 0
    n_reject: int = 0
    calibrated: bool = False
    note: str = ""


def estimate_thresholds(
    accept_df: pd.DataFrame,
    reject_df: Optional[pd.DataFrame] = None,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> dict:
    """
    Derive accept/reject bounds for every SQI column.

    Parameters
    ----------
    accept_df : pd.DataFrame
        SQI values from clean segments.  One row per segment, one column per
        SQI.  All-NaN columns are skipped.
    reject_df : pd.DataFrame, optional
        SQI values from degraded segments.  Used only for diagnostics (the
        reject distribution does not change the threshold values).
    lower_pct : float
        Lower percentile of the accept distribution that defines the accept
        lower bound (default ``5``).
    upper_pct : float
        Upper percentile defining the accept upper bound (default ``95``).

    Returns
    -------
    dict
        Mapping of ``sqi_name → SQIThreshold``.  Only calibratable SQIs are
        included.
    """
    thresholds = {}

    for col in accept_df.columns:
        accept_vals = accept_df[col].replace([np.inf, -np.inf], np.nan).dropna().values

        t = SQIThreshold(sqi_name=col)
        t.n_accept = len(accept_vals)

        if reject_df is not None and col in reject_df.columns:
            reject_vals = reject_df[col].replace([np.inf, -np.inf], np.nan).dropna().values
            t.n_reject = len(reject_vals)
            t.reject_median = float(np.median(reject_vals)) if len(reject_vals) > 0 else np.nan
        else:
            reject_vals = np.array([])

        if t.n_accept < 5:
            t.note = f"Skipped: only {t.n_accept} valid accept samples"
            continue

        t.accept_median = float(np.median(accept_vals))
        t.accept_std = float(np.std(accept_vals))

        lower = float(np.percentile(accept_vals, lower_pct))
        upper = float(np.percentile(accept_vals, upper_pct))

        # Guard: constant or near-constant column
        if upper - lower < _EPSILON:
            half = max(abs(t.accept_median) * 0.1, _EPSILON)
            lower = t.accept_median - half
            upper = t.accept_median + half
            t.note = "Constant/near-constant: band widened by epsilon"
            warnings.warn(
                f"SQI '{col}' has near-zero variance in accept distribution. "
                f"Widened band to ({lower:.4g}, {upper:.4g})."
            )

        # Guard: reject distribution overlaps — note it but keep clean bounds
        if len(reject_vals) > 0:
            reject_lower = float(np.percentile(reject_vals, lower_pct))
            reject_upper = float(np.percentile(reject_vals, upper_pct))
            overlap = not (upper < reject_lower or lower > reject_upper)
            if overlap and not t.note:
                t.note = (
                    f"Accept/reject distributions overlap. "
                    f"Accept p5-p95: ({lower:.3g}, {upper:.3g}), "
                    f"Reject p5-p95: ({reject_lower:.3g}, {reject_upper:.3g}). "
                    f"Consider using a more discriminative SQI or tighter noise conditions."
                )

        t.lower = lower
        t.upper = upper
        t.calibrated = True
        thresholds[col] = t

    return thresholds


def thresholds_to_dataframe(thresholds: dict) -> pd.DataFrame:
    """
    Convert a thresholds dict to a summary DataFrame for inspection.

    Parameters
    ----------
    thresholds : dict
        Output of :func:`estimate_thresholds`.

    Returns
    -------
    pd.DataFrame
        One row per SQI with columns: sqi_name, lower, upper, accept_median,
        accept_std, reject_median, n_accept, n_reject, calibrated, note.
    """
    rows = []
    for t in thresholds.values():
        rows.append({
            "sqi_name":      t.sqi_name,
            "lower":         t.lower,
            "upper":         t.upper,
            "accept_median": t.accept_median,
            "accept_std":    t.accept_std,
            "reject_median": t.reject_median,
            "n_accept":      t.n_accept,
            "n_reject":      t.n_reject,
            "calibrated":    t.calibrated,
            "note":          t.note,
        })
    return pd.DataFrame(rows).set_index("sqi_name")
