"""
Top-level calibration experiment runner.

Usage (from repo root)
----------------------
    python -m vital_sqi.calibration.run_calibration --wave_type PPG
    python -m vital_sqi.calibration.run_calibration --wave_type ECG
    python -m vital_sqi.calibration.run_calibration --wave_type PPG --n_segments 500 --dry_run

What it does
------------
1. Generate ``n_segments`` clean signals (``noise_floor=0``).
2. For every noise profile in :data:`~vital_sqi.calibration.noise_injector.NOISE_PROFILES`
   that is labelled as clean, accumulate into the **accept** pool.
3. For every noise profile labelled as reject-level, generate ``n_segments``
   degraded signals and accumulate into the **reject** pool.
4. Compute SQIs over both pools via :func:`~vital_sqi.calibration.sqi_runner.compute_sqi_distributions`.
5. Estimate p5/p95 thresholds from the accept pool.
6. Export ``rule_dict.json`` and ``sqi_dict.json`` to ``output_dir``.
7. Write a diagnostics CSV alongside the outputs.

The script prints a summary table on completion showing which SQIs were
calibrated and their derived thresholds.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from vital_sqi.calibration.signal_generator import generate_clean_ppg, generate_clean_ecg
from vital_sqi.calibration.noise_injector import (
    inject_noise, NOISE_PROFILES, CLEAN_PROFILE_LABELS
)
from vital_sqi.calibration.sqi_runner import compute_sqi_distributions
from vital_sqi.calibration.threshold_estimator import estimate_thresholds, thresholds_to_dataframe
from vital_sqi.calibration.exporter import export_rule_dict, export_sqi_dict, export_diagnostics


def _resource_dir() -> str:
    """Path to vital_sqi/resource/ relative to this file."""
    return os.path.join(os.path.dirname(__file__), "..", "resource")


def calibrate(
    wave_type: str = "PPG",
    n_segments: int = 200,
    n_reject_segments: int = 50,
    duration: float = 30.0,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    output_dir: str = None,
    dry_run: bool = False,
    seed: int = 42,
    show_progress: bool = True,
) -> dict:
    """
    Run the full calibration experiment and (optionally) export results.

    Parameters
    ----------
    wave_type : str
        ``'PPG'`` or ``'ECG'``.
    n_segments : int
        Number of clean segments (and accept-pool segments per noise condition).
    n_reject_segments : int
        Number of segments per reject-pool noise condition.  Smaller than
        *n_segments* is fine — the reject pool is only used for diagnostics
        and overlap detection, not for the p5/p95 accept-band thresholds
        (default ``50``).
    duration : float
        Segment duration in seconds.
    lower_pct : float
        Lower percentile for accept band (default ``5``).
    upper_pct : float
        Upper percentile for accept band (default ``95``).
    output_dir : str, optional
        Directory to write ``rule_dict.json``, ``sqi_dict.json``, and the
        diagnostics CSV.  Defaults to ``vital_sqi/resource/``.
    dry_run : bool
        If ``True``, compute thresholds but do NOT write any files.
    seed : int
        Random seed for reproducibility (default ``42``).
    show_progress : bool
        Show tqdm progress bars.

    Returns
    -------
    dict
        Calibrated thresholds as returned by
        :func:`~vital_sqi.calibration.threshold_estimator.estimate_thresholds`.
    """
    if output_dir is None:
        output_dir = os.path.abspath(_resource_dir())

    rng = np.random.default_rng(seed)
    fs = 100 if wave_type == "PPG" else 256
    _gen = generate_clean_ppg if wave_type == "PPG" else generate_clean_ecg

    print(f"\n{'='*60}")
    print(f"  vital_sqi calibration - {wave_type}")
    print(f"  {n_segments} accept / {n_reject_segments} reject segments x {len(NOISE_PROFILES)} noise profiles")
    print(f"  Accept band: p{lower_pct:.0f} – p{upper_pct:.0f}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Step 1: Generate clean baseline segments
    # ------------------------------------------------------------------
    print("[1/4] Generating clean segments ...")
    clean_segments = _gen(
        n_segments=n_segments,
        duration=duration,
        sampling_rate=fs,
        noise_floor=0.0,
        rng=np.random.default_rng(seed),
    )
    if not clean_segments:
        raise RuntimeError(f"Signal generator returned 0 segments for {wave_type}.")
    print(f"      Generated {len(clean_segments)} clean segments.")

    # ------------------------------------------------------------------
    # Step 2: Build accept pool -clean + very-mild-noise segments
    # ------------------------------------------------------------------
    print("[2/4] Building accept (clean) pool ...")
    accept_segments = list(clean_segments)  # always include pure clean

    for label, noise_type, amplitude in NOISE_PROFILES:
        if label not in CLEAN_PROFILE_LABELS or label == "clean":
            continue
        noisy = _degrade_batch(clean_segments, noise_type, amplitude, fs, rng)
        accept_segments.extend(noisy)

    print(f"      Accept pool size: {len(accept_segments)} segments.")
    accept_df = compute_sqi_distributions(
        accept_segments,
        wave_type=wave_type,
        show_progress=show_progress,
    )

    # ------------------------------------------------------------------
    # Step 3: Build reject pool -moderate-to-severe noise conditions
    # Only n_reject_segments per profile (reject pool is diagnostics-only;
    # p5/p95 thresholds come from the accept pool alone).
    # ------------------------------------------------------------------
    print("[3/4] Building reject (noisy) pool ...")
    reject_segs_all = []
    reject_base = clean_segments[:n_reject_segments]  # subsample for speed
    for label, noise_type, amplitude in NOISE_PROFILES:
        if label in CLEAN_PROFILE_LABELS:
            continue  # skip clean profiles
        noisy = _degrade_batch(reject_base, noise_type, amplitude, fs, rng)
        reject_segs_all.extend(noisy)

    print(f"      Reject pool size: {len(reject_segs_all)} segments.")
    reject_df = compute_sqi_distributions(
        reject_segs_all,
        wave_type=wave_type,
        show_progress=show_progress,
    )

    # ------------------------------------------------------------------
    # Step 4: Estimate thresholds
    # ------------------------------------------------------------------
    print("[4/4] Estimating thresholds ...")
    thresholds = estimate_thresholds(
        accept_df=accept_df,
        reject_df=reject_df,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
    )

    n_calibrated = sum(1 for t in thresholds.values() if t.calibrated)
    print(f"\n      Calibrated: {n_calibrated} / {len(accept_df.columns)} SQIs")

    # Print summary table
    summary = thresholds_to_dataframe(thresholds)
    calibrated_summary = summary[summary["calibrated"]][["lower", "upper", "accept_median",
                                                           "accept_std", "n_accept", "note"]]
    print("\n" + calibrated_summary.to_string(float_format=lambda x: f"{x:.4g}"))

    # ------------------------------------------------------------------
    # Step 5: Export
    # ------------------------------------------------------------------
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

        rule_path = os.path.join(output_dir, "rule_dict.json")
        sqi_path  = os.path.join(output_dir, "sqi_dict.json")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(output_dir, f"calibration_report_{wave_type}_{ts}.csv")

        export_rule_dict(thresholds, rule_path, wave_type, lower_pct, upper_pct)
        export_sqi_dict(thresholds, sqi_path)
        export_diagnostics(thresholds, csv_path)

        print(f"\n[done] Files written to {output_dir}")
    else:
        print("\n[dry_run] No files written.")

    return thresholds


def _degrade_batch(clean_segments, noise_type, amplitude, fs, rng):
    """Apply a noise type to every segment in the batch and return new list."""
    degraded = []
    for sig, _ in clean_segments:
        noisy = inject_noise(sig, noise_type, amplitude, rng, fs=fs)
        degraded.append((noisy, fs))
    return degraded


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Calibrate SQI thresholds using synthetic physiological signals."
    )
    p.add_argument("--wave_type",   default="PPG",  choices=["PPG", "ECG"],
                   help="Signal type to calibrate (default: PPG)")
    p.add_argument("--n_segments",  default=200,    type=int,
                   help="Clean/accept segments per noise condition (default: 200)")
    p.add_argument("--n_reject_segments", default=50, type=int,
                   help="Reject segments per noise condition (default: 50)")
    p.add_argument("--duration",    default=30.0,   type=float,
                   help="Segment duration in seconds (default: 30)")
    p.add_argument("--lower_pct",   default=5.0,    type=float,
                   help="Lower percentile for accept band (default: 5)")
    p.add_argument("--upper_pct",   default=95.0,   type=float,
                   help="Upper percentile for accept band (default: 95)")
    p.add_argument("--output_dir",  default=None,
                   help="Output directory (default: vital_sqi/resource/)")
    p.add_argument("--dry_run",     action="store_true",
                   help="Compute thresholds but do not write files")
    p.add_argument("--seed",        default=42,     type=int,
                   help="Random seed (default: 42)")
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress progress bars")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    calibrate(
        wave_type=args.wave_type,
        n_segments=args.n_segments,
        n_reject_segments=args.n_reject_segments,
        duration=args.duration,
        lower_pct=args.lower_pct,
        upper_pct=args.upper_pct,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed,
        show_progress=not args.quiet,
    )
