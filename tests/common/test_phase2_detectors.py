"""
Phase 2 peak-detection tests.

Covers all new detectors added in Phase 2:
  ECG: Pan-Tompkins, Hamilton, Engzee (+ ECG_DEFAULT regression)
  PPG: AMPD, LOCAL_MAX_IBI (+ dispatcher wiring for all methods)

Design principle (Rule 9): every test encodes WHY the behaviour matters.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _sine_ppg(freq_hz=1.0, fs=100, duration=15.0, noise=0.02):
    """Synthetic PPG: clean sine at known frequency."""
    np.random.seed(42)
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * freq_hz * t) + noise * np.random.randn(len(t))


def _synthetic_ecg(fs=256, duration=30.0, hr=70, noise=0.05):
    """
    Synthetic ECG: Gaussian QRS complexes + T-waves at regular HR.
    Returns (signal, true_r_peak_indices).
    """
    np.random.seed(0)
    N = int(fs * duration)
    s = np.zeros(N)
    rr = 60.0 / hr  # seconds per beat
    peak_times = np.arange(0.5, duration, rr)
    true_peaks = []
    for pt in peak_times:
        idx = int(pt * fs)
        if idx >= N:
            continue
        true_peaks.append(idx)
        # QRS Gaussian
        for i in range(max(0, idx - 20), min(N, idx + 20)):
            s[i] += 1.5 * np.exp(-0.5 * ((i - idx) / (0.01 * fs)) ** 2)
        # T-wave
        t_idx = idx + int(0.25 * fs)
        if t_idx < N:
            for i in range(max(0, t_idx - 30), min(N, t_idx + 30)):
                s[i] += 0.3 * np.exp(-0.5 * ((i - t_idx) / (0.05 * fs)) ** 2)
    s += noise * np.random.randn(N)
    return s, np.array(true_peaks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_dist(detected, ground_truth):
    """For each detected peak, find the distance to the nearest ground truth."""
    if len(detected) == 0 or len(ground_truth) == 0:
        return np.array([])
    return np.array([np.min(np.abs(ground_truth - p)) for p in detected])


# ===========================================================================
# ECG detectors
# ===========================================================================


class TestECGDetectors:
    """Validate ECG R-peak detectors against synthetic ground truth."""

    @pytest.fixture
    def ecg_30s(self):
        s, truth = _synthetic_ecg(fs=256, duration=30.0, hr=70, noise=0.05)
        return s, truth  # ~35 beats

    @pytest.fixture
    def detector(self):
        from vital_sqi.common.rpeak_detection import PeakDetector
        return PeakDetector(wave_type="ECG", fs=256)

    # -----------------------------------------------------------------------
    # Pan-Tompkins
    # -----------------------------------------------------------------------
    def test_pan_tompkins_count(self, detector, ecg_30s):
        """Pan-Tompkins must find approximately the correct number of beats."""
        from vital_sqi.common.rpeak_detection import PAN_TOMPKINS
        s, truth = ecg_30s
        r, *_ = detector.ecg_detector(s, detector_type=PAN_TOMPKINS)
        assert len(r) >= len(truth) - 3, (
            f"Pan-Tompkins found {len(r)} peaks, expected ~{len(truth)}. "
            "Too many false negatives."
        )
        assert len(r) <= len(truth) + 3, (
            f"Pan-Tompkins found {len(r)} peaks, expected ~{len(truth)}. "
            "Too many false positives."
        )

    def test_pan_tompkins_accuracy(self, detector, ecg_30s):
        """Pan-Tompkins detected peaks must be within 10 ms of true R-peaks."""
        from vital_sqi.common.rpeak_detection import PAN_TOMPKINS
        s, truth = ecg_30s
        r, *_ = detector.ecg_detector(s, detector_type=PAN_TOMPKINS)
        if len(r) == 0:
            pytest.skip("No peaks detected")
        dists = _nearest_dist(r, truth)
        tol = int(0.010 * 256)  # 10 ms at 256 Hz
        assert np.median(dists) <= tol, (
            f"Pan-Tompkins median peak error {np.median(dists)} samples "
            f"> {tol} samples (10 ms tolerance)."
        )

    def test_pan_tompkins_returns_critical_points(self, detector, ecg_30s):
        """Pan-Tompkins path must still return Q/S/P/T via WaveformMorphology."""
        from vital_sqi.common.rpeak_detection import PAN_TOMPKINS
        s, _ = ecg_30s
        r, q, sv, p, t = detector.ecg_detector(s, detector_type=PAN_TOMPKINS)
        assert len(q) > 0, "Q valleys must be detected via WaveformMorphology"
        assert len(sv) > 0, "S valleys must be detected"
        assert len(p) > 0, "P peaks must be detected"
        assert len(t) > 0, "T peaks must be detected"

    # -----------------------------------------------------------------------
    # Hamilton
    # -----------------------------------------------------------------------
    def test_hamilton_count(self, detector, ecg_30s):
        """Hamilton detector must find approximately the correct beat count."""
        from vital_sqi.common.rpeak_detection import HAMILTON
        s, truth = ecg_30s
        r, *_ = detector.ecg_detector(s, detector_type=HAMILTON)
        assert abs(len(r) - len(truth)) <= 4, (
            f"Hamilton found {len(r)} peaks, expected ~{len(truth)}."
        )

    def test_hamilton_accuracy(self, detector, ecg_30s):
        """Hamilton peaks must be within 15 ms of ground truth."""
        from vital_sqi.common.rpeak_detection import HAMILTON
        s, truth = ecg_30s
        r, *_ = detector.ecg_detector(s, detector_type=HAMILTON)
        if len(r) == 0:
            pytest.skip("No peaks detected")
        dists = _nearest_dist(r, truth)
        tol = int(0.015 * 256)
        assert np.median(dists) <= tol, (
            f"Hamilton median error {np.median(dists)} > {tol} samples."
        )

    def test_hamilton_returns_critical_points(self, detector, ecg_30s):
        """Hamilton path must return Q/S/P/T."""
        from vital_sqi.common.rpeak_detection import HAMILTON
        s, _ = ecg_30s
        r, q, sv, p, t = detector.ecg_detector(s, detector_type=HAMILTON)
        assert len(q) > 0 and len(sv) > 0 and len(p) > 0 and len(t) > 0

    # -----------------------------------------------------------------------
    # Engzee
    # -----------------------------------------------------------------------
    def test_engzee_finds_peaks(self, detector, ecg_30s):
        """Engzee must detect peaks on a clean ECG (may over-detect on T-waves)."""
        from vital_sqi.common.rpeak_detection import ENGZEE
        s, truth = ecg_30s
        r, *_ = detector.ecg_detector(s, detector_type=ENGZEE)
        # Engzee is known to over-detect; require at least true count detected
        assert len(r) >= len(truth) - 5, (
            f"Engzee only found {len(r)} peaks; likely missing true R-peaks."
        )

    def test_engzee_returns_critical_points(self, detector, ecg_30s):
        """Engzee path must also return Q/S/P/T via WaveformMorphology."""
        from vital_sqi.common.rpeak_detection import ENGZEE
        s, _ = ecg_30s
        r, q, sv, p, t = detector.ecg_detector(s, detector_type=ENGZEE)
        assert isinstance(q, np.ndarray)
        assert isinstance(t, np.ndarray)

    # -----------------------------------------------------------------------
    # ECG_DEFAULT (vitalDSP — regression guard)
    # -----------------------------------------------------------------------
    def test_ecg_default_returns_five_tuple(self, detector, ecg_30s):
        """ECG_DEFAULT must return (r, q, s, p, t) — five arrays."""
        from vital_sqi.common.rpeak_detection import ECG_DEFAULT
        s, _ = ecg_30s
        result = detector.ecg_detector(s, detector_type=ECG_DEFAULT)
        assert len(result) == 5, "ecg_detector must return 5-tuple"
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_invalid_ecg_detector_raises(self, detector, ecg_30s):
        """Passing an unknown ECG detector_type must raise ValueError."""
        s, _ = ecg_30s
        with pytest.raises(ValueError, match="Invalid ECG detector_type"):
            detector.ecg_detector(s, detector_type=999)

    def test_empty_ecg_raises(self, detector):
        """Empty signal must raise ValueError."""
        from vital_sqi.common.rpeak_detection import ECG_DEFAULT
        with pytest.raises(ValueError, match="empty"):
            detector.ecg_detector(np.array([]), detector_type=ECG_DEFAULT)

    # -----------------------------------------------------------------------
    # get_session interface
    # -----------------------------------------------------------------------
    def test_get_session_returns_two_tuple(self, detector, ecg_30s):
        """get_session=True must return (r_peaks, ecg_session) not a 5-tuple."""
        from vital_sqi.common.rpeak_detection import PAN_TOMPKINS
        s, _ = ecg_30s
        result = detector.ecg_detector(s, detector_type=PAN_TOMPKINS, get_session=True)
        assert len(result) == 2, "get_session must return 2-tuple"


# ===========================================================================
# PPG detectors — AMPD
# ===========================================================================


class TestAMPD:
    """AMPD (Automatic Multiscale Peak Detection) tests."""

    @pytest.fixture
    def ppg_1hz_30s(self):
        return _sine_ppg(freq_hz=1.0, fs=100, duration=30.0, noise=0.02)

    @pytest.fixture
    def detector(self):
        from vital_sqi.common.rpeak_detection import PeakDetector
        return PeakDetector(wave_type="PPG", fs=100)

    def test_ampd_count_clean(self, detector, ppg_1hz_30s):
        """AMPD must find ~30 peaks in a clean 30 s, 1 Hz PPG."""
        from vital_sqi.common.rpeak_detection import AMPD_METHOD
        peaks, troughs = detector.ppg_detector(ppg_1hz_30s, detector_type=AMPD_METHOD)
        assert len(peaks) >= 25, f"AMPD found only {len(peaks)} peaks on 30 s, 1 Hz PPG"
        assert len(peaks) <= 36, f"AMPD found too many peaks: {len(peaks)}"

    def test_ampd_peaks_are_local_maxima(self, detector, ppg_1hz_30s):
        """Every AMPD peak must be a local maximum of the raw signal."""
        from vital_sqi.common.rpeak_detection import AMPD_METHOD
        from scipy.signal import argrelmax
        peaks, _ = detector.ppg_detector(ppg_1hz_30s, detector_type=AMPD_METHOD)
        true_max = argrelmax(ppg_1hz_30s)[0]
        for p in peaks:
            dist = np.min(np.abs(true_max - p))
            assert dist <= 5, (
                f"AMPD peak at {p} is {dist} samples from nearest local maximum."
            )

    def test_ampd_troughs_between_peaks(self, detector, ppg_1hz_30s):
        """Each trough index must lie between the two peaks that bound it."""
        from vital_sqi.common.rpeak_detection import AMPD_METHOD
        peaks, troughs = detector.ppg_detector(ppg_1hz_30s, detector_type=AMPD_METHOD)
        if len(peaks) < 2:
            pytest.skip("Too few peaks for trough check")
        for i, tr in enumerate(troughs):
            assert peaks[i] <= tr <= peaks[i + 1], (
                f"Trough {tr} not between peaks {peaks[i]} and {peaks[i+1]}"
            )

    def test_ampd_returns_arrays(self, detector):
        """AMPD must return numpy arrays even on minimal input."""
        from vital_sqi.common.rpeak_detection import AMPD_METHOD
        s = _sine_ppg(freq_hz=1.0, fs=100, duration=5.0)
        peaks, troughs = detector.ppg_detector(s, detector_type=AMPD_METHOD)
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)

    def test_ampd_dispatcher_wired(self, detector, ppg_1hz_30s):
        """AMPD_METHOD constant must be accepted by ppg_detector dispatcher."""
        from vital_sqi.common.rpeak_detection import AMPD_METHOD
        peaks, troughs = detector.ppg_detector(ppg_1hz_30s, detector_type=AMPD_METHOD)
        assert isinstance(peaks, np.ndarray)


# ===========================================================================
# PPG detectors — LOCAL_MAX_IBI
# ===========================================================================


class TestLocalMaxIBI:
    """LOCAL_MAX_IBI (IBI-tracking local-maximum) detector tests."""

    @pytest.fixture
    def detector(self):
        from vital_sqi.common.rpeak_detection import PeakDetector
        return PeakDetector(wave_type="PPG", fs=100)

    def test_local_max_ibi_count_clean(self, detector):
        """LOCAL_MAX_IBI must find ~30 peaks in a clean 30 s, 1 Hz PPG."""
        from vital_sqi.common.rpeak_detection import LOCAL_MAX_IBI
        ppg = _sine_ppg(freq_hz=1.0, fs=100, duration=30.0, noise=0.02)
        peaks, _ = detector.ppg_detector(ppg, detector_type=LOCAL_MAX_IBI)
        assert len(peaks) >= 25 and len(peaks) <= 36, (
            f"LOCAL_MAX_IBI found {len(peaks)} peaks on 30 s, 1 Hz PPG."
        )

    def test_local_max_ibi_refractory_enforced(self, detector):
        """Consecutive peaks must be separated by at least 300 ms (200 BPM cap)."""
        from vital_sqi.common.rpeak_detection import LOCAL_MAX_IBI
        ppg = _sine_ppg(freq_hz=1.2, fs=100, duration=20.0, noise=0.05)
        peaks, _ = detector.ppg_detector(ppg, detector_type=LOCAL_MAX_IBI)
        if len(peaks) < 2:
            pytest.skip("Too few peaks")
        min_gap = np.min(np.diff(peaks))
        min_rr_samples = int(0.300 * 100)  # 300 ms at fs=100
        assert min_gap >= min_rr_samples, (
            f"Minimum inter-peak gap {min_gap} samples < {min_rr_samples} "
            "(300 ms refractory). Detector is not enforcing refractory period."
        )

    def test_local_max_ibi_noisy(self, detector):
        """LOCAL_MAX_IBI must remain robust to moderate noise (SNR ~10 dB)."""
        from vital_sqi.common.rpeak_detection import LOCAL_MAX_IBI
        np.random.seed(7)
        t = np.arange(0, 20.0, 0.01)
        ppg = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.random.randn(len(t))
        peaks, _ = detector.ppg_detector(ppg, detector_type=LOCAL_MAX_IBI)
        # With SNR ~10 dB, expect at least 15 of 20 true peaks detected
        assert len(peaks) >= 15, (
            f"LOCAL_MAX_IBI found only {len(peaks)} peaks under noise; "
            "too fragile for clinical use."
        )

    def test_local_max_ibi_troughs_bounded(self, detector):
        """Troughs must lie between adjacent peaks."""
        from vital_sqi.common.rpeak_detection import LOCAL_MAX_IBI
        ppg = _sine_ppg(freq_hz=1.0, fs=100, duration=15.0)
        peaks, troughs = detector.ppg_detector(ppg, detector_type=LOCAL_MAX_IBI)
        if len(peaks) < 2:
            pytest.skip("Too few peaks")
        for i, tr in enumerate(troughs):
            assert peaks[i] <= tr <= peaks[i + 1], (
                f"Trough {tr} not between peaks {peaks[i]} and {peaks[i+1]}"
            )

    def test_local_max_ibi_dispatcher_wired(self, detector):
        """LOCAL_MAX_IBI constant must be accepted by ppg_detector."""
        from vital_sqi.common.rpeak_detection import LOCAL_MAX_IBI
        ppg = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        peaks, troughs = detector.ppg_detector(ppg, detector_type=LOCAL_MAX_IBI)
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)


# ===========================================================================
# Dispatcher completeness
# ===========================================================================


class TestDispatcherCompleteness:
    """All method constants must be accepted by the dispatcher without raising."""

    def test_all_ppg_methods_accepted(self):
        """Every PPG method constant must route to a valid implementation."""
        from vital_sqi.common.rpeak_detection import (
            PeakDetector,
            ADAPTIVE_THRESHOLD, COUNT_ORIG_METHOD, CLUSTERER_METHOD,
            SLOPE_SUM_METHOD, MOVING_AVERAGE_METHOD, DEFAULT,
            BILLAUER_METHOD, AMPD_METHOD, LOCAL_MAX_IBI,
        )
        ppg = _sine_ppg(freq_hz=1.0, fs=100, duration=15.0)
        det = PeakDetector(wave_type="PPG", fs=100)
        methods = [
            ADAPTIVE_THRESHOLD, COUNT_ORIG_METHOD, CLUSTERER_METHOD,
            SLOPE_SUM_METHOD, MOVING_AVERAGE_METHOD, DEFAULT,
            BILLAUER_METHOD, AMPD_METHOD, LOCAL_MAX_IBI,
        ]
        for m in methods:
            try:
                peaks, troughs = det.ppg_detector(ppg, detector_type=m)
                assert isinstance(peaks, np.ndarray), f"Method {m} did not return ndarray"
            except Exception as e:
                pytest.fail(f"Method {m} raised unexpectedly: {e}")

    def test_all_ecg_methods_accepted(self):
        """Every ECG method constant must route without raising on valid input."""
        from vital_sqi.common.rpeak_detection import (
            PeakDetector, ECG_DEFAULT, PAN_TOMPKINS, HAMILTON, ENGZEE,
        )
        ecg, _ = _synthetic_ecg(fs=256, duration=20.0, hr=70)
        det = PeakDetector(wave_type="ECG", fs=256)
        for m in [ECG_DEFAULT, PAN_TOMPKINS, HAMILTON, ENGZEE]:
            try:
                result = det.ecg_detector(ecg, detector_type=m)
                assert len(result) == 5, f"Method {m} did not return 5-tuple"
            except Exception as e:
                pytest.fail(f"ECG method {m} raised unexpectedly: {e}")

    def test_ppg_invalid_type_raises(self):
        from vital_sqi.common.rpeak_detection import PeakDetector
        det = PeakDetector(wave_type="PPG", fs=100)
        with pytest.raises(ValueError, match="Invalid detector_type"):
            det.ppg_detector(_sine_ppg(), detector_type=999)

    def test_ecg_invalid_type_raises(self):
        from vital_sqi.common.rpeak_detection import PeakDetector
        det = PeakDetector(wave_type="ECG", fs=256)
        ecg, _ = _synthetic_ecg(fs=256, duration=10.0)
        with pytest.raises(ValueError, match="Invalid ECG detector_type"):
            det.ecg_detector(ecg, detector_type=999)
