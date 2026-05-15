"""Strategies for picking accept-band thresholds from an SQI distribution.

Two policies live here:

* :func:`quantile_band` — pick a fixed quantile window around the
  observed distribution.  This is the classic ``auto_mode=True`` from
  :func:`vital_sqi.pipeline.pipeline_functions.classify_segments`: trim
  the bottom *lower_pct* and top *upper_pct* tails.

* :func:`tuned_band` — given a set of SQI columns, derive a per-column
  quantile that targets a joint accept rate.  Assumes rules are
  independent (a common simplifying assumption for orthogonal SQIs); the
  per-rule keep-rate is ``target ** (1/n)`` so the product of independent
  keep-rates equals the target.

Both strategies share a degenerate-band guard that returns ``None`` when
the observed distribution is too narrow to produce a meaningful rule.
Callers should drop those columns from the rule set rather than build a
band that rejects every segment.

This module is deliberately UI-free: it only deals with numbers.  The
Inspect view and ``classify_segments`` both consume it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


#: Bands narrower than this collapse to "reject everything" under
#: percentile-based auto-mode.  Empirically this is the same threshold
#: used by :mod:`vital_sqi.calibration.threshold_estimator` for its
#: epsilon guard.
DEGENERATE_BAND_HALF_WIDTH = 1e-6


@dataclass(frozen=True)
class Band:
    """An accept band ``(lower, upper)`` plus diagnostic provenance."""

    column: str
    lower: float
    upper: float
    quantile_lo: float
    quantile_hi: float
    note: str = ""

    @property
    def width(self) -> float:
        return self.upper - self.lower


# ---------------------------------------------------------------------------
# Policy 1 — fixed quantile window (the existing auto_mode=True behaviour)
# ---------------------------------------------------------------------------


def quantile_band(
    column: str,
    values: Sequence[float],
    *,
    lower_pct: float = 0.05,
    upper_pct: float = 0.95,
) -> Optional[Band]:
    """Compute an accept band from the empirical lower/upper quantiles.

    Parameters
    ----------
    column
        SQI name (only used in the returned :class:`Band` for diagnostics).
    values
        Observed SQI values; ``NaN`` / ``inf`` are dropped before quantile
        computation.
    lower_pct
        Lower quantile (e.g. ``0.05`` for p5).  Must be in ``[0, 0.5)``.
    upper_pct
        Upper quantile (e.g. ``0.95`` for p95).  Must be in ``(0.5, 1]``.

    Returns
    -------
    Band or None
        ``None`` when fewer than 2 finite values are available, or when
        the resulting band is narrower than
        :data:`DEGENERATE_BAND_HALF_WIDTH`.  Callers should treat both as
        "this SQI cannot contribute a useful rule".
    """
    if not (0.0 <= lower_pct < 0.5 < upper_pct <= 1.0):
        raise ValueError(
            f"Need 0 <= lower_pct ({lower_pct}) < 0.5 < upper_pct "
            f"({upper_pct}) <= 1."
        )
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    lo = float(np.quantile(finite, lower_pct))
    hi = float(np.quantile(finite, upper_pct))
    if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
        return None
    return Band(
        column=column,
        lower=lo,
        upper=hi,
        quantile_lo=lower_pct,
        quantile_hi=upper_pct,
    )


# ---------------------------------------------------------------------------
# Policy 2 — joint-accept-rate auto-tune
# ---------------------------------------------------------------------------


def per_rule_quantile(target_accept_rate: float, n_rules: int) -> float:
    """Symmetric per-rule trim that yields *target_accept_rate* jointly.

    Under the independence approximation, the joint accept rate is the
    product of per-rule accept rates.  Solving::

        target = keep ** n_rules
        keep   = target ** (1 / n_rules)
        trim   = 1 - keep              # split symmetrically across both tails
        lower_pct = trim / 2

    For ``target=0.85, n_rules=5`` this gives a per-rule keep-rate of
    ~0.968 → bands at p1.6/p98.4.  Much more forgiving than the legacy
    p5/p95 (which on 5 independent rules expects ~60% joint accept).

    Parameters
    ----------
    target_accept_rate
        Desired fraction of segments that should pass *all* rules.
        Clamped to ``(0, 1)``; values at the extremes give degenerate
        bands.
    n_rules
        Number of independent rules in the set.  ``1`` returns the
        symmetric split corresponding to a single quantile pair.

    Returns
    -------
    float
        ``lower_pct`` (the upper quantile is ``1 - lower_pct``).
    """
    if n_rules < 1:
        raise ValueError("n_rules must be at least 1")
    p = float(np.clip(target_accept_rate, 1e-3, 0.999))
    keep = p ** (1.0 / n_rules)
    trim = max(0.0, 1.0 - keep)
    return float(trim / 2.0)


def tuned_bands(
    column_values: "dict[str, Sequence[float]]",
    *,
    target_accept_rate: float = 0.85,
) -> List[Band]:
    """Per-column accept bands sized to hit *target_accept_rate* jointly.

    Degenerate columns are dropped silently — they don't count towards
    *n_rules*, so the per-rule quantile is recomputed only over the
    columns that actually contribute a band.  This keeps the auto-tune
    sensible when half the catalogue is constant.

    Two-pass algorithm:

    1. Filter to columns whose p5/p95 band is non-degenerate (cheap
       sanity check; bands narrower than that won't survive any tighter
       trim either).
    2. Compute the per-rule quantile from the surviving count, then
       compute each column's actual band at that quantile.

    Parameters
    ----------
    column_values
        Mapping from SQI column name to its observed values.
    target_accept_rate
        Desired joint accept rate in ``(0, 1)``.

    Returns
    -------
    list of Band
        One entry per surviving column, in iteration order of the input.
    """
    # Pre-filter: a column whose p5/p95 already collapses to zero width
    # will never produce a useful band, regardless of how tight we trim.
    survivors: List[Tuple[str, np.ndarray]] = []
    for column, values in column_values.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            continue
        lo = float(np.quantile(finite, 0.05))
        hi = float(np.quantile(finite, 0.95))
        if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
            continue
        survivors.append((column, finite))

    if not survivors:
        return []

    lower_pct = per_rule_quantile(target_accept_rate, n_rules=len(survivors))
    upper_pct = 1.0 - lower_pct

    bands: List[Band] = []
    for column, finite in survivors:
        lo = float(np.quantile(finite, lower_pct))
        hi = float(np.quantile(finite, upper_pct))
        if (hi - lo) < DEGENERATE_BAND_HALF_WIDTH:
            # The pre-filter accepted this column at p5/p95 but the
            # tighter trim flattened it.  Skip rather than poison the
            # rule set with a 0-width band.
            continue
        bands.append(
            Band(
                column=column,
                lower=lo,
                upper=hi,
                quantile_lo=lower_pct,
                quantile_hi=upper_pct,
                note=f"auto-tuned for joint accept ~{target_accept_rate:.0%}",
            )
        )
    return bands


# ---------------------------------------------------------------------------
# Strict-rule detector — flag rules that reject far more than the rest
# ---------------------------------------------------------------------------


def strictest_columns(
    per_rule_rejects: "dict[str, int]",
    *,
    mad_multiplier: float = 3.0,
) -> List[str]:
    """Return rule names whose rejection count is an upward outlier.

    Used by the Inspect view's "Drop strictest rule" button.

    Uses the **modified Z-score** (median + ``mad_multiplier`` × MAD)
    rather than the parametric ``mean + k·std``, because the latter is
    blown up by the very outlier we're trying to detect.  The classic
    modified Z-score from Iglewicz & Hoaglin (1993) flags a sample as
    an outlier when its rescaled deviation
    ``0.6745 * (x - median) / MAD`` exceeds 3.5 — we use a slightly
    looser cutoff (the ``mad_multiplier`` default of 3.0 in raw units)
    so the UI surfaces obviously-strict rules without nagging on
    borderline cases.

    Returns an empty list when fewer than 3 rules are supplied (with
    only 2 points one is always the "outlier") or when every rule
    rejects roughly the same number of segments (MAD == 0).

    Parameters
    ----------
    per_rule_rejects
        ``{rule_name: n_segments_rejected}``.
    mad_multiplier
        How many MADs above the median a count must be to count as an
        outlier.  Defaults to ``3.0`` (≈ the 99th percentile of a
        normal distribution).
    """
    if len(per_rule_rejects) < 3:
        return []
    counts = np.array(list(per_rule_rejects.values()), dtype=float)
    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med)))
    if mad < 1e-9:
        # Every rule rejects the same number — no meaningful outlier.
        return []
    threshold = med + mad_multiplier * mad
    flagged = [
        name for name, count in per_rule_rejects.items()
        if count > threshold
    ]
    flagged.sort(key=lambda n: per_rule_rejects[n], reverse=True)
    return flagged
