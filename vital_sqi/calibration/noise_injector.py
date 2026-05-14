"""
Noise injection for calibration experiments.

Each noise function takes a clean signal array and returns a degraded copy.
All amplitudes are expressed as a fraction of the signal's peak-to-peak range
so they scale correctly regardless of the raw signal magnitude.

Available noise types
---------------------
- ``gaussian``        : additive white Gaussian noise
- ``baseline_wander`` : slow sinusoidal drift (0.05–0.5 Hz)
- ``motion``          : burst-style motion artifacts (random amplitude spikes)
- ``harmonic``        : sum of sine/cosine harmonics at multiples of a base freq
- ``powerline``       : 50 Hz or 60 Hz sinusoidal interference
- ``polynomial``      : low-degree polynomial trend added across the segment
- ``impulse``         : random isolated spike impulses
- ``colored``         : pink (1/f) or brown (1/f²) noise

``NOISE_PROFILES`` is a list of ``(name, noise_fn, amplitude)`` triples that
define the experiments run by the calibration pipeline.
"""

import numpy as np
from typing import Callable
from scipy.signal import resample as scipy_resample


# ---------------------------------------------------------------------------
# Individual noise functions
# ---------------------------------------------------------------------------

def _pp(signal: np.ndarray) -> float:
    """Peak-to-peak range; returns 1.0 if signal is flat to avoid division by zero."""
    pp = np.ptp(signal)
    return pp if pp > 0 else 1.0


def gaussian_noise(signal: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise scaled to ``amplitude`` × peak-to-peak."""
    return signal + rng.normal(0, amplitude * _pp(signal), size=len(signal))


def baseline_wander(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                    fs: int = 100) -> np.ndarray:
    """
    Slow sinusoidal drift at a random frequency in [0.05, 0.5] Hz,
    mimicking respiration-induced baseline wander.
    """
    t = np.arange(len(signal)) / fs
    freq = rng.uniform(0.05, 0.5)
    phase = rng.uniform(0, 2 * np.pi)
    drift = amplitude * _pp(signal) * np.sin(2 * np.pi * freq * t + phase)
    return signal + drift


def motion_artifact(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                    fs: int = 100, n_bursts: int = 3) -> np.ndarray:
    """
    Random burst artifacts: short high-amplitude segments injected at random
    positions, mimicking movement in wearable recordings.
    """
    degraded = signal.copy()
    burst_len = int(fs * rng.uniform(0.2, 1.0))  # 0.2 – 1.0 s bursts
    for _ in range(n_bursts):
        start = rng.integers(0, max(1, len(signal) - burst_len))
        end = min(len(signal), start + burst_len)
        degraded[start:end] += rng.normal(0, amplitude * _pp(signal), size=end - start)
    return degraded


def harmonic_interference(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                           fs: int = 100) -> np.ndarray:
    """
    Sum of sine and cosine harmonics at random base frequency [1–15 Hz] with
    up to 5 harmonics. Mimics periodic electrical interference or muscle noise.
    """
    t = np.arange(len(signal)) / fs
    base_freq = rng.uniform(1.0, 15.0)
    n_harmonics = rng.integers(2, 6)
    interference = np.zeros(len(signal))
    for k in range(1, n_harmonics + 1):
        a = rng.uniform(0.3, 1.0) / k          # amplitude decays with harmonic
        phase_sin = rng.uniform(0, 2 * np.pi)
        phase_cos = rng.uniform(0, 2 * np.pi)
        interference += a * np.sin(2 * np.pi * k * base_freq * t + phase_sin)
        interference += a * np.cos(2 * np.pi * k * base_freq * t + phase_cos)
    interference /= (np.max(np.abs(interference)) + 1e-9)  # normalise to [-1, 1]
    return signal + amplitude * _pp(signal) * interference


def powerline_noise(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                    fs: int = 100) -> np.ndarray:
    """
    50 Hz or 60 Hz sinusoidal powerline interference with a random phase.
    """
    t = np.arange(len(signal)) / fs
    freq = rng.choice([50.0, 60.0])
    phase = rng.uniform(0, 2 * np.pi)
    interference = amplitude * _pp(signal) * np.sin(2 * np.pi * freq * t + phase)
    return signal + interference


def polynomial_trend(signal: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Polynomial trend of degree 2–4 added across the segment.
    Mimics slow non-stationary drift or temperature effects.
    """
    n = len(signal)
    t = np.linspace(-1, 1, n)           # normalised to [-1, 1] for numerical stability
    degree = rng.integers(2, 5)
    coeffs = rng.uniform(-1, 1, size=degree + 1)
    trend = np.polyval(coeffs, t)
    trend /= (np.max(np.abs(trend)) + 1e-9)
    return signal + amplitude * _pp(signal) * trend


def impulse_noise(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                  rate: float = 0.02) -> np.ndarray:
    """
    Random isolated spike impulses at ``rate`` fraction of samples.
    Positive and negative spikes with equal probability.
    """
    degraded = signal.copy()
    n_impulses = max(1, int(len(signal) * rate))
    positions = rng.choice(len(signal), size=n_impulses, replace=False)
    signs = rng.choice([-1, 1], size=n_impulses)
    degraded[positions] += signs * amplitude * _pp(signal) * rng.uniform(3, 8, size=n_impulses)
    return degraded


def colored_noise(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                  exponent: float = 1.0) -> np.ndarray:
    """
    Colored noise with power spectrum ∝ 1/f^exponent.

    exponent=1 → pink noise, exponent=2 → brown noise.
    Generated via FFT shaping of white noise.
    """
    n = len(signal)
    white = rng.standard_normal(n)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero at DC
    power = freqs ** (-exponent / 2.0)
    power[0] = 0.0  # zero DC component
    shaped = np.fft.irfft(fft_white * power, n=n)
    shaped /= (np.std(shaped) + 1e-9)
    return signal + amplitude * _pp(signal) * shaped


def time_shift(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
               fs: int = 100) -> np.ndarray:
    """
    Circularly shift the signal by a random number of samples.

    Simulates late electrode attachment, trigger jitter, or sync misalignment.
    The shift magnitude is ``amplitude * fs`` samples (e.g. amplitude=0.05 at
    fs=100 means up to 5-sample / 50 ms shift).  Circular padding preserves
    signal length without introducing zero-padding edge artifacts.

    Parameters
    ----------
    signal : np.ndarray
    amplitude : float
        Maximum shift as a fraction of fs (in seconds).
    rng : np.random.Generator
    fs : int
    """
    max_shift = max(1, int(amplitude * fs))
    shift = rng.integers(-max_shift, max_shift + 1)
    return np.roll(signal, shift)


def clock_drift(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                fs: int = 100) -> np.ndarray:
    """
    Simulate clock drift by resampling the signal at a perturbed rate.

    The effective sampling rate is stretched or compressed by ``amplitude``
    fraction (e.g. amplitude=0.02 → ±2% clock error).  The signal is
    resampled to a perturbed length and then trimmed / zero-padded back to
    the original length, simulating what happens when data is labelled with
    the wrong fs.

    Parameters
    ----------
    signal : np.ndarray
    amplitude : float
        Maximum fractional clock error (e.g. 0.02 = ±2%).
    rng : np.random.Generator
    fs : int  (unused — kept for API uniformity)
    """
    n = len(signal)
    drift = rng.uniform(-amplitude, amplitude)
    new_len = max(2, int(round(n * (1 + drift))))
    resampled = scipy_resample(signal, new_len)
    if new_len >= n:
        return resampled[:n]
    # new_len < n: zero-pad at the end
    out = np.zeros(n, dtype=signal.dtype)
    out[:new_len] = resampled
    return out


def sample_dropout(signal: np.ndarray, amplitude: float, rng: np.random.Generator,
                   fs: int = 100) -> np.ndarray:
    """
    Randomly drop and duplicate samples to simulate packet loss and jitter.

    For each dropped sample, a neighbouring sample is duplicated to keep
    the signal length constant.  ``amplitude`` controls the fraction of
    samples affected (e.g. 0.05 → 5% dropout rate).

    Parameters
    ----------
    signal : np.ndarray
    amplitude : float
        Fraction of samples to drop (and replace by duplication).
    rng : np.random.Generator
    fs : int  (unused)
    """
    n = len(signal)
    n_drop = max(0, int(n * amplitude))
    if n_drop == 0:
        return signal.copy()
    out = signal.copy()
    # Drop random positions and fill with the preceding sample
    positions = np.sort(rng.choice(n - 1, size=n_drop, replace=False))
    for pos in positions:
        out[pos] = out[max(0, pos - 1)]  # replace with predecessor
    return out


# ---------------------------------------------------------------------------
# Composite injector
# ---------------------------------------------------------------------------

def inject_noise(
    signal: np.ndarray,
    noise_type: str,
    amplitude: float,
    rng: np.random.Generator,
    fs: int = 100,
) -> np.ndarray:
    """
    Apply a named noise type to a clean signal.

    Parameters
    ----------
    signal : np.ndarray
        Clean input signal.
    noise_type : str
        One of the keys in :data:`NOISE_PROFILES` or a raw noise name:
        ``'gaussian'``, ``'baseline_wander'``, ``'motion'``, ``'harmonic'``,
        ``'powerline'``, ``'polynomial'``, ``'impulse'``, ``'pink'``,
        ``'brown'``, ``'combined_mild'``, ``'combined_severe'``.
    amplitude : float
        Noise amplitude as a fraction of signal peak-to-peak.
    rng : np.random.Generator
        Random number generator.
    fs : int
        Sampling frequency in Hz (required by time-dependent noise types).

    Returns
    -------
    np.ndarray
        Degraded signal (same length as input).
    """
    _DISPATCH = {
        "gaussian": lambda s, a, r: gaussian_noise(s, a, r),
        "baseline_wander": lambda s, a, r: baseline_wander(s, a, r, fs),
        "motion": lambda s, a, r: motion_artifact(s, a, r, fs),
        "harmonic": lambda s, a, r: harmonic_interference(s, a, r, fs),
        "powerline": lambda s, a, r: powerline_noise(s, a, r, fs),
        "polynomial": lambda s, a, r: polynomial_trend(s, a, r),
        "impulse": lambda s, a, r: impulse_noise(s, a, r),
        "pink": lambda s, a, r: colored_noise(s, a, r, exponent=1.0),
        "brown": lambda s, a, r: colored_noise(s, a, r, exponent=2.0),
        "time_shift": lambda s, a, r: time_shift(s, a, r, fs),
        "clock_drift": lambda s, a, r: clock_drift(s, a, r, fs),
        "dropout": lambda s, a, r: sample_dropout(s, a, r, fs),
        "combined_mild": lambda s, a, r: polynomial_trend(
            baseline_wander(gaussian_noise(s, a * 0.5, r), a * 0.3, r, fs), a * 0.2, r
        ),
        "combined_severe": lambda s, a, r: impulse_noise(
            motion_artifact(
                harmonic_interference(
                    gaussian_noise(s, a * 0.5, r), a * 0.4, r, fs
                ), a * 0.6, r, fs
            ), a * 0.8, r
        ),
    }
    if noise_type not in _DISPATCH:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Choose from: {list(_DISPATCH.keys())}"
        )
    return _DISPATCH[noise_type](signal, amplitude, rng)


# ---------------------------------------------------------------------------
# Experiment profiles: (name, noise_type, amplitude)
# ---------------------------------------------------------------------------

#: Full set of noise conditions used in calibration experiments.
#: Each entry is ``(label, noise_type, amplitude)`` where amplitude is a
#: fraction of the signal peak-to-peak range.
NOISE_PROFILES = [
    # --- Clean / near-clean ---
    ("clean",              "gaussian",        0.00),
    ("gaussian_very_mild", "gaussian",        0.02),
    ("gaussian_mild",      "gaussian",        0.05),
    # --- Gaussian degradation sweep ---
    ("gaussian_moderate",  "gaussian",        0.15),
    ("gaussian_severe",    "gaussian",        0.40),
    ("gaussian_extreme",   "gaussian",        0.80),
    # --- Baseline wander ---
    ("bw_mild",            "baseline_wander", 0.05),
    ("bw_moderate",        "baseline_wander", 0.20),
    ("bw_severe",          "baseline_wander", 0.50),
    # --- Motion artifacts ---
    ("motion_mild",        "motion",          0.10),
    ("motion_moderate",    "motion",          0.40),
    ("motion_severe",      "motion",          0.80),
    # --- Harmonic interference ---
    ("harmonic_mild",      "harmonic",        0.05),
    ("harmonic_moderate",  "harmonic",        0.20),
    ("harmonic_severe",    "harmonic",        0.50),
    # --- Powerline ---
    ("powerline_mild",     "powerline",       0.05),
    ("powerline_severe",   "powerline",       0.30),
    # --- Polynomial trend ---
    ("poly_mild",          "polynomial",      0.10),
    ("poly_severe",        "polynomial",      0.50),
    # --- Impulse / spike noise ---
    ("impulse_mild",       "impulse",         0.05),
    ("impulse_severe",     "impulse",         0.30),
    # --- Colored noise ---
    ("pink_mild",          "pink",            0.05),
    ("pink_severe",        "pink",            0.30),
    ("brown_mild",         "brown",           0.05),
    ("brown_severe",       "brown",           0.30),
    # --- Combined realistic noise ---
    ("combined_mild",      "combined_mild",   0.10),
    ("combined_severe",    "combined_severe", 0.50),
    # --- Temporal / structural distortions ---
    ("shift_mild",         "time_shift",      0.05),   # up to 5% of fs in samples
    ("shift_severe",       "time_shift",      0.20),   # up to 20% of fs
    ("drift_mild",         "clock_drift",     0.02),   # ±2% clock error
    ("drift_severe",       "clock_drift",     0.05),   # ±5% clock error
    ("dropout_mild",       "dropout",         0.02),   # 2% packet loss
    ("dropout_severe",     "dropout",         0.10),   # 10% packet loss
]

#: Profiles considered "clean enough" to contribute to the accept distribution.
#: Any profile NOT in this set contributes to the reject distribution.
CLEAN_PROFILE_LABELS = {"clean", "gaussian_very_mild"}
