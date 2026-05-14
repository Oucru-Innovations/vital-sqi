"""
Generate clean synthetic ECG and PPG segments for calibration.

Each function returns a list of (signal_array, fs) tuples, one per segment.
Segments vary heart rate across the physiological range so the SQI distribution
covers realistic within-subject variability.
"""

import numpy as np
import warnings


def generate_clean_ppg(
    n_segments=200,
    duration=30.0,
    sampling_rate=100,
    hr_range=(50, 110),
    noise_floor=0.0,
    rng=None,
):
    """
    Generate clean synthetic PPG segments.

    Parameters
    ----------
    n_segments : int
        Number of independent segments to generate.
    duration : float
        Length of each segment in seconds.
    sampling_rate : int
        Samples per second.
    hr_range : tuple of int
        (min_hr, max_hr) in bpm; each segment gets a randomly drawn HR.
    noise_floor : float
        Baseline noise amplitude (fraction of peak-to-peak). Use 0.0 for
        truly clean signals.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list of tuple
        Each element is ``(signal_array: np.ndarray, fs: int)``.
    """
    from vitalDSP.utils.data_processing.synthesize_data import generate_synthetic_ppg

    if rng is None:
        rng = np.random.default_rng(42)

    segments = []
    hr_values = rng.integers(hr_range[0], hr_range[1] + 1, size=n_segments)

    for hr in hr_values:
        try:
            _, sig = generate_synthetic_ppg(
                duration=duration,
                sampling_rate=sampling_rate,
                heart_rate=int(hr),
                noise_level=noise_floor,
                display=False,
            )
            sig = np.asarray(sig, dtype=float)
            segments.append((sig, sampling_rate))
        except Exception as e:
            warnings.warn(f"PPG generation failed for HR={hr}: {e}")

    return segments


def generate_clean_ecg(
    n_segments=200,
    duration=30.0,
    sampling_rate=256,
    hr_range=(50, 110),
    noise_floor=0.0,
    rng=None,
):
    """
    Generate clean synthetic ECG segments.

    Parameters
    ----------
    n_segments : int
        Number of independent segments to generate.
    duration : float
        Length of each segment in seconds.
    sampling_rate : int
        Samples per second (vitalDSP ECG uses ``sfecg``).
    hr_range : tuple of int
        (min_hr, max_hr) in bpm.
    noise_floor : float
        Additive noise amplitude (``Anoise`` in vitalDSP units). Use 0.0 for
        clean signals.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    list of tuple
        Each element is ``(signal_array: np.ndarray, fs: int)``.
    """
    from vitalDSP.utils.data_processing.synthesize_data import generate_ecg_signal

    if rng is None:
        rng = np.random.default_rng(42)

    segments = []
    hr_values = rng.integers(hr_range[0], hr_range[1] + 1, size=n_segments)

    # N in generate_ecg_signal is duration in beats (approx)
    for hr in hr_values:
        n_beats = max(1, int(duration * hr / 60))
        try:
            sig = generate_ecg_signal(
                sfecg=sampling_rate,
                N=n_beats,
                Anoise=noise_floor,
                hrmean=int(hr),
            )
            sig = np.asarray(sig, dtype=float)
            # Trim or pad to exact sample count
            n_samples = int(duration * sampling_rate)
            if len(sig) >= n_samples:
                sig = sig[:n_samples]
            else:
                sig = np.pad(sig, (0, n_samples - len(sig)), mode="edge")
            segments.append((sig, sampling_rate))
        except Exception as e:
            warnings.warn(f"ECG generation failed for HR={hr}: {e}")

    return segments
