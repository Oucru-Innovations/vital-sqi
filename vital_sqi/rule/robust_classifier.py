"""
Robust SQI segment classifier.

Implements a three-regime quality classification strategy:
  - clean        : most segments are good (rank-IQR threshold)
  - bimodal      : clear good/bad split (GMM with Bhattacharyya fallback)
  - heavy_noise  : most segments are bad (conservative accept)

Public entry point::

    result = classify_segments_robust(sqis_df, sqi_names)
    # result.decisions   — list[str] "accept"/"reject"
    # result.scores      — np.ndarray float in [0,1]
    # result.regime      — str  "clean" | "bimodal" | "heavy_noise"
    # result.regime_info — dict with diagnostic fields
    # result.file_flagged — bool

Ported and simplified from frequency_resonance/src/core/signal_processing/sqi_scorer.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# GaussianMixture is optional — imported lazily so the rest of the module loads
# even if sklearn is absent (unit tests can still test the non-GMM paths).
_GMM_AVAILABLE = False
try:
    from sklearn.mixture import GaussianMixture
    _GMM_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RobustResult:
    decisions: List[str]
    scores: np.ndarray
    regime: str
    regime_info: dict = field(default_factory=dict)
    file_flagged: bool = False


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] via rank / (N-1); ties get average rank."""
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    ranks = stats.rankdata(values, method="average")
    return (ranks - 1) / (n - 1)


def _iqr_normalize(values: np.ndarray) -> np.ndarray:
    """Robust [0,1] scaling using IQR: (x - P25) / (P75 - P25)."""
    p25, p75 = np.percentile(values, [25, 75])
    iqr = p75 - p25
    if iqr == 0:
        return np.full(len(values), 0.5)
    return np.clip((values - p25) / iqr, 0, 1)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Scale to [0,1] using global min/max."""
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.full(len(values), 0.5)
    return (values - vmin) / (vmax - vmin)


def _sanitize(values: np.ndarray) -> np.ndarray:
    """Replace inf/-inf with NaN, then fill NaN with column median."""
    v = np.array(values, dtype=float)
    v[~np.isfinite(v)] = np.nan
    if np.all(np.isnan(v)):
        return np.zeros(len(v))
    med = np.nanmedian(v)
    v[np.isnan(v)] = med
    return v


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

_MIN_SEGMENTS_FOR_GMM = 20


def _normal_fraction(scores: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of segments with score >= threshold."""
    return float(np.mean(scores >= threshold))


def _bic_bimodal_check(scores: np.ndarray) -> bool:
    """Return True when a 2-component GMM fits better (lower BIC) than 1-component."""
    if not _GMM_AVAILABLE or len(scores) < _MIN_SEGMENTS_FOR_GMM:
        return False
    x = scores.reshape(-1, 1)
    try:
        bic1 = GaussianMixture(n_components=1, random_state=0).fit(x).bic(x)
        bic2 = GaussianMixture(n_components=2, random_state=0).fit(x).bic(x)
        return bic2 < bic1
    except Exception:
        return False


def _bimodality_coefficient(values: np.ndarray) -> float:
    """
    Bimodality coefficient (BC): BC > 5/9 ≈ 0.555 suggests bimodality.

    BC = (skewness² + 1) / (kurtosis + 3*(n-1)²/((n-2)*(n-3)))
    Reference: SAS documentation.
    """
    n = len(values)
    if n < 4:
        return 0.0
    sk = float(stats.skew(values))
    ku = float(stats.kurtosis(values, fisher=True))  # excess kurtosis
    # finite-sample correction
    ku_adj = ku + 3 * (n - 1) ** 2 / max((n - 2) * (n - 3), 1)
    return (sk ** 2 + 1) / (ku_adj + 1e-12)


def _detect_quality_regime(
    rank_scores: np.ndarray,
    abs_quality: float,
    normal_frac_threshold_clean: float = 0.7,
    normal_frac_threshold_bad: float = 0.3,
) -> str:
    """
    Classify the recording-level quality into one of three regimes.

    Strategy
    --------
    1. Bimodality check first (bimodality coefficient + GMM BIC on rank scores).
       A bimodal rank distribution means roughly half the segments are clearly
       better than the other half.
    2. If not bimodal, use the rank-score variance relative to a uniform
       baseline to detect "all-similar" recordings, then use ``abs_quality``
       (the cross-column mean of per-column raw-value medians, clipped to
       [0, 1]) to distinguish "all good" (clean) from "all bad" (heavy_noise).
    3. For intermediate rank variances, ``abs_quality`` is the tiebreaker.

    Parameters
    ----------
    rank_scores : np.ndarray
        Mean rank-normalised consensus score, one value per segment.
    abs_quality : float
        Scalar in [0, 1] summarising the absolute level of the raw SQI
        values (mean of per-column medians clipped to [0, 1]).

    Returns
    -------
    str
        "clean" | "bimodal" | "heavy_noise"
    """
    bc = _bimodality_coefficient(rank_scores)
    if bc > 0.555 or _bic_bimodal_check(rank_scores):
        return "bimodal"

    # Not bimodal — use abs_quality to separate clean from heavy_noise.
    # abs_quality >= 0.5 means the median raw SQI value sits in the "good"
    # half of [0, 1]; this correctly distinguishes a tight cluster at 0.8
    # (clean) from a tight cluster at 0.15 (heavy_noise).
    if abs_quality >= 0.5:
        return "clean"
    return "heavy_noise"


# ---------------------------------------------------------------------------
# Per-regime classifiers
# ---------------------------------------------------------------------------

def _classify_mostly_good(scores: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Threshold just below the median; accepts the bulk of segments."""
    return (scores >= threshold).astype(float)


def _bhattacharyya_distance(mu1, s1, mu2, s2) -> float:
    avg_s = (s1 + s2) / 2
    if avg_s == 0:
        return 0.0
    term1 = (mu1 - mu2) ** 2 / (4 * avg_s)
    if s1 <= 0 or s2 <= 0 or avg_s <= 0:
        return term1
    term2 = 0.5 * np.log(avg_s / (np.sqrt(s1 * s2) + 1e-12))
    return term1 + term2


def _classify_bimodal(scores: np.ndarray) -> np.ndarray:
    """
    Fit a 2-component GMM; label each segment to the higher-mean component.
    Falls back to MAD-based split when GMM is unavailable or fails.
    """
    if _GMM_AVAILABLE and len(scores) >= _MIN_SEGMENTS_FOR_GMM:
        try:
            x = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
            labels = gmm.predict(x)
            # "good" component = higher mean
            good_comp = int(np.argmax(gmm.means_.ravel()))
            return (labels == good_comp).astype(float)
        except Exception:
            pass

    # Fallback: MAD-robust split at median
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    threshold = med - mad if mad > 0 else med
    return (scores >= threshold).astype(float)


def _classify_heavy_noise(scores: np.ndarray, accept_quantile: float = 0.8) -> np.ndarray:
    """Accept only the top accept_quantile fraction."""
    threshold = np.quantile(scores, accept_quantile)
    return (scores >= threshold).astype(float)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_segments_robust(
    sqis_df: pd.DataFrame,
    sqi_names: Optional[List[str]] = None,
    config: Optional[dict] = None,
) -> RobustResult:
    """
    Classify signal segments using a rank+IQR consensus score and
    automatic regime detection.

    Parameters
    ----------
    sqis_df : pd.DataFrame
        One row per segment, one column per SQI. May contain inf/NaN.
    sqi_names : list of str, optional
        Columns to use. Defaults to all non-index columns.
    config : dict, optional
        Override default thresholds:
          clean_threshold   (default 0.7)
          bad_threshold     (default 0.3)
          heavy_noise_quantile (default 0.8)

    Returns
    -------
    RobustResult
    """
    cfg = {
        "clean_threshold": 0.7,
        "bad_threshold": 0.3,
        "heavy_noise_quantile": 0.8,
    }
    if config:
        cfg.update(config)

    if sqi_names is None:
        sqi_names = [c for c in sqis_df.columns if c not in ("start_idx", "end_idx", "decision")]

    if len(sqis_df) == 0 or not sqi_names:
        empty = RobustResult(
            decisions=[],
            scores=np.array([]),
            regime="clean",
            regime_info={"note": "empty input"},
        )
        return empty

    # --- Sanitize and build consensus score ---
    # Each column is rank-normalised within the recording to produce the
    # per-segment consensus score (relative quality ranking).
    # For regime detection we need an absolute quality level that is not
    # destroyed by within-recording normalisation.  We use the column-wise
    # IQR-normalised values whose *median* is then the absolute level signal:
    # a column where all values cluster near 0.8 will have median near 0.8
    # after IQR normalisation only when the distribution is tight — but
    # actually IQR normalisation still spreads tight distributions.
    # Instead we compute the per-column *raw median* (after sanitising)
    # and clip it to [0,1]; the cross-column average of these medians is
    # the absolute quality indicator used by the regime detector.
    col_rank_scores = []
    col_raw_medians = []
    for col in sqi_names:
        if col not in sqis_df.columns:
            continue
        clean = _sanitize(sqis_df[col].values)
        col_rank_scores.append(_rank_normalize(clean))
        # Raw median clipped to [0,1] as absolute quality proxy.
        # Works when SQI values are naturally in [0,1]; for unbounded SQIs
        # this is a rough signal but still captures very-high vs very-low.
        col_raw_medians.append(float(np.clip(np.median(clean), 0.0, 1.0)))

    if not col_rank_scores:
        n = len(sqis_df)
        scores = np.full(n, 0.5)
        abs_quality = 0.5
    else:
        scores = np.mean(np.column_stack(col_rank_scores), axis=1)
        abs_quality = float(np.mean(col_raw_medians))

    # --- Detect regime ---
    # abs_quality is a scalar summarising the absolute level of raw SQI values.
    # scores (rank-based) are used for bimodality detection.
    regime = _detect_quality_regime(
        scores,
        abs_quality,
        normal_frac_threshold_clean=cfg["clean_threshold"],
        normal_frac_threshold_bad=cfg["bad_threshold"],
    )

    # --- Apply per-regime classifier ---
    # All classifiers work on the rank-based consensus scores.
    if regime == "clean":
        accept_mask = _classify_mostly_good(scores)
    elif regime == "bimodal":
        accept_mask = _classify_bimodal(scores)
    else:  # heavy_noise
        accept_mask = _classify_heavy_noise(
            scores, accept_quantile=cfg["heavy_noise_quantile"]
        )

    decisions = ["accept" if a else "reject" for a in accept_mask]
    nf = _normal_fraction(scores)

    # Flag the file when 20 % or fewer of segments are accepted
    file_flagged = float(np.mean(accept_mask)) <= 0.2

    regime_info = {
        "regime": regime,
        "normal_fraction": round(nf, 4),
        "n_segments": len(sqis_df),
        "n_accepted": int(np.sum(accept_mask)),
        "score_mean": round(float(np.mean(scores)), 4),
        "score_std": round(float(np.std(scores)), 4),
        "abs_quality": round(abs_quality, 4),
        "rank_var": round(float(np.var(scores)), 4),
        "file_flagged": file_flagged,
        "gmm_available": _GMM_AVAILABLE,
    }

    return RobustResult(
        decisions=decisions,
        scores=scores,
        regime=regime,
        regime_info=regime_info,
        file_flagged=file_flagged,
    )
