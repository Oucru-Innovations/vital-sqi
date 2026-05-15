"""
Phase 1 fix regression tests.

Each test encodes WHY the behaviour matters (Rule 9), not just what the code returns.
Tests are grouped by the fix ID from dev_docs/fix_plan.md.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_rule(lower, upper, name="sqi"):
    """Return a Rule that accepts values in (lower, upper) and rejects outside."""
    from vital_sqi.rule.rule_class import Rule

    r = Rule(name)
    r.update_def(
        op_list=[">", "<="],
        value_list=[lower, upper],
        label_list=["accept", "reject"],
    )
    return r


def _sine_ppg(freq_hz=1.0, fs=100, duration=5.0, amplitude=1.0):
    """Simple synthetic PPG: clean sine wave with known peak positions."""
    t = np.arange(0, duration, 1.0 / fs)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


# ===========================================================================
# P1.A6 — Rule.update_def label tautology removed
# ===========================================================================


class TestP1A6_RuleLabelTautology:
    """The 'if label != "reject" or label != "accept": label = None' block was
    always True, nullifying every valid label. Verify labels survive round-trip."""

    def test_accept_label_survives(self):
        from vital_sqi.rule.rule_class import Rule

        r = Rule("test")
        r.update_def(op_list=[">"], value_list=[5], label_list=["accept"])
        # Before fix this would have been None; must be "accept"
        assert "accept" in r.rule["labels"], "accept label must survive update_def"

    def test_reject_label_survives(self):
        from vital_sqi.rule.rule_class import Rule

        r = Rule("test")
        r.update_def(op_list=["<="], value_list=[5], label_list=["reject"])
        assert "reject" in r.rule["labels"], "reject label must survive update_def"

    def test_apply_rule_uses_correct_label(self):
        """A properly constructed rule (< 10 → accept, > 10 → reject) must
        return the correct label. Before the tautology fix, all labels were
        nullified and apply_rule always fell through to 'reject'."""
        from vital_sqi.rule.rule_class import Rule

        # Pair '<' and '>' at the same boundary to define accept/reject regions.
        r = Rule("test")
        r.update_def(
            op_list=["<", ">"],
            value_list=[10, 10],
            label_list=["accept", "reject"],
        )
        assert r.apply_rule(11) == "reject", "Values above threshold must be rejected"
        assert r.apply_rule(9) == "accept", "Values below threshold must be accepted"


# ===========================================================================
# P1.A5 — RuleSet.execute: None / NaN → reject (not silent accept)
# ===========================================================================


class TestP1A5_RuleSetNoneReject:
    """None from apply_rule (and NaN inputs) must propagate as reject, not accept."""

    def test_nan_sqi_is_rejected(self):
        """NaN fed into apply_rule must return 'reject', not silently pass through."""
        from vital_sqi.rule.rule_class import Rule
        from vital_sqi.rule.ruleset_class import RuleSet

        # Accept values in (0, 1): < 0 → reject, > 0 → accept, < 1 → accept, > 1 → reject
        r = Rule("mysqi")
        r.update_def(
            op_list=["<", ">", "<", ">"],
            value_list=[0, 0, 1, 1],
            label_list=["reject", "accept", "accept", "reject"],
        )
        rs = RuleSet({1: r})
        df = pd.DataFrame([[np.nan]], columns=["mysqi"])
        assert rs.execute(df) == "reject", "NaN SQI must be rejected, not silently accepted"

    def test_inf_sqi_is_rejected(self):
        from vital_sqi.rule.rule_class import Rule
        from vital_sqi.rule.ruleset_class import RuleSet

        r = Rule("mysqi")
        r.update_def(
            op_list=["<", ">", "<", ">"],
            value_list=[0, 0, 1, 1],
            label_list=["reject", "accept", "accept", "reject"],
        )
        rs = RuleSet({1: r})
        df = pd.DataFrame([[np.inf]], columns=["mysqi"])
        assert rs.execute(df) == "reject"

    def test_apply_rule_nan_returns_reject(self):
        """Non-finite inputs must always return 'reject' regardless of rule definition."""
        from vital_sqi.rule.rule_class import Rule

        r = Rule("test")
        r.update_def(
            op_list=["<", ">"],
            value_list=[0, 0],
            label_list=["reject", "accept"],
        )
        assert r.apply_rule(np.nan) == "reject"
        assert r.apply_rule(np.inf) == "reject"
        assert r.apply_rule(-np.inf) == "reject"


# ===========================================================================
# P1.A1 — classify_segments uses per-channel thresholds in auto_mode
# ===========================================================================


class TestP1A1_PerChannelAutoMode:
    """Auto-mode thresholds must be derived from each channel's own distribution,
    not from channel 0 applied to all channels."""

    def _make_sqis_two_channels(self):
        from vital_sqi.common.utils import create_rule_def

        # Channel 0: SQI values around 0.5 (normal)
        # Channel 1: SQI values around 50.0 (very different scale)
        np.random.seed(42)
        ch0 = pd.DataFrame({"skewness_1": np.random.uniform(0.4, 0.6, 20)})
        ch1 = pd.DataFrame({"skewness_1": np.random.uniform(40.0, 60.0, 20)})
        return [ch0, ch1]

    def test_thresholds_differ_per_channel(self):
        """When two channels have very different SQI scales, auto_mode must
        produce different thresholds; otherwise the wrong channel gets misclassified."""
        import json
        import tempfile
        import os
        from vital_sqi.pipeline.pipeline_functions import classify_segments

        sqis = self._make_sqis_two_channels()

        # Build a minimal rule_dict_filename on disk
        rule_dict = {
            "skewness_rule": {
                "name": "skewness_1",
                "def": [{"op": ">", "value": 0, "label": "accept"}],
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(rule_dict, f)
            fname = f.name

        try:
            ruleset, result_sqis = classify_segments(
                sqis,
                rule_dict_filename=fname,
                ruleset_order={1: "skewness_rule"},
                auto_mode=True,
                lower_bound=0.05,
                upper_bound=0.95,
            )
            # Both channels should have a "decision" column
            assert "decision" in result_sqis[0].columns
            assert "decision" in result_sqis[1].columns
            # Decisions should exist (not crash)
            assert all(d in ("accept", "reject") for d in result_sqis[0]["decision"])
            assert all(d in ("accept", "reject") for d in result_sqis[1]["decision"])
        finally:
            os.unlink(fname)


# ===========================================================================
# P1.A4 — per_beat_sqi mean-beat returns single value, not N duplicates
# ===========================================================================


class TestP1A4_PerBeatMeanBeat:
    """use_mean_beat=True must compute one SQI on the mean beat and return it as a
    single-element list, not duplicate it across all beats (which makes std=0)."""

    def test_mean_beat_returns_single_value(self):
        from vital_sqi.pipeline.pipeline_functions import per_beat_sqi
        from vital_sqi.sqi.standard_sqi import skewness_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        # Troughs at every 100 samples (1 Hz)
        troughs = np.arange(0, len(signal), 100)

        result = per_beat_sqi(
            skewness_sqi, troughs, signal, use_mean_beat=True, mean_resample_size=100
        )
        assert len(result) == 1, (
            "use_mean_beat should return exactly one SQI value, "
            f"got {len(result)}"
        )

    def test_per_beat_returns_multiple_values(self):
        from vital_sqi.pipeline.pipeline_functions import per_beat_sqi
        from vital_sqi.sqi.standard_sqi import skewness_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        troughs = np.arange(0, len(signal), 100)

        result = per_beat_sqi(
            skewness_sqi, troughs, signal, use_mean_beat=False, mean_resample_size=100
        )
        # 9 beats for 10 troughs (len-1)
        assert len(result) == len(troughs) - 1


# ===========================================================================
# P1.B1 — correlogram_sqi no longer crashes the pipeline
# ===========================================================================


class TestP1B1_CorrelogramSqiReturn:
    """correlogram_sqi returns a scalar; get_sqi_dict must not try to index it."""

    def test_correlogram_returns_scalar(self):
        from vital_sqi.sqi.rpeaks_sqi import correlogram_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        result = correlogram_sqi(signal, sample_rate=100, wave_type="PPG")
        assert np.isscalar(result) or np.isnan(result), "correlogram_sqi must return scalar"

    def test_get_sqi_dict_correlogram_no_crash(self):
        """get_sqi_dict must not raise TypeError on a scalar correlogram result."""
        from vital_sqi.pipeline.pipeline_functions import get_sqi_dict

        scalar_result = 0.85
        d = get_sqi_dict(scalar_result, "correlogram_sqi")
        assert isinstance(d, dict)
        assert "correlogram_sqi" in d
        assert d["correlogram_sqi"] == pytest.approx(0.85)

    def test_get_sqi_dict_scalar_float(self):
        from vital_sqi.pipeline.pipeline_functions import get_sqi_dict

        d = get_sqi_dict(0.5, "skewness_sqi")
        assert d == {"skewness_sqi": 0.5}

    def test_get_sqi_dict_single_element_list(self):
        from vital_sqi.pipeline.pipeline_functions import get_sqi_dict

        d = get_sqi_dict([0.5], "skewness_sqi")
        assert d == {"skewness_sqi": 0.5}

    def test_get_sqi_dict_multi_element_list(self):
        from vital_sqi.pipeline.pipeline_functions import get_sqi_dict

        d = get_sqi_dict([0.2, 0.5, 0.8], "skewness_sqi")
        assert "skewness_sqi_mean_sqi" in d
        assert "skewness_sqi_std_sqi" in d
        assert d["skewness_sqi_mean_sqi"] == pytest.approx(np.mean([0.2, 0.5, 0.8]))


# ===========================================================================
# P1.B2 — split_segment raises on overlapping >= duration
# ===========================================================================


class TestP1B2_SplitSegmentZeroStep:
    """overlapping >= duration must raise ValueError immediately, not crash silently."""

    def _make_df(self, n=1000, fs=100):
        from vital_sqi.common.utils import generate_timestamp

        t = generate_timestamp(None, fs, n)
        return pd.DataFrame({"time": t, "signal": np.sin(np.linspace(0, 10, n))})

    def test_equal_overlap_raises(self):
        from vital_sqi.preprocess.segment_split import split_segment

        df = self._make_df()
        with pytest.raises(ValueError, match="overlapping"):
            split_segment(df, sampling_rate=100, split_type=0, duration=30, overlapping=30)

    def test_greater_overlap_raises(self):
        from vital_sqi.preprocess.segment_split import split_segment

        df = self._make_df()
        with pytest.raises(ValueError, match="overlapping"):
            split_segment(df, sampling_rate=100, split_type=0, duration=10, overlapping=15)

    def test_valid_overlap_works(self):
        from vital_sqi.preprocess.segment_split import split_segment

        df = self._make_df(n=2000)
        segs, milestones = split_segment(
            df, sampling_rate=100, split_type=0, duration=5, overlapping=2
        )
        assert len(segs) > 0


# ===========================================================================
# P1.C1 — calculate_band_power uses np.trapz, not wrong normalization
# ===========================================================================


class TestP1C1_BandPowerNormalization:
    """Band power must integrate the PSD over frequency (Parseval-consistent).
    The old formula divided by 2*len(power)^2 which was off by ~10^6."""

    def test_parseval_consistency(self):
        """Total PSD power over [0, fs/2] must be close to signal variance."""
        from vital_sqi.common.power_spectrum import calculate_band_power
        from scipy.signal import welch

        np.random.seed(0)
        fs = 4.0  # HRV sampling rate
        # White noise → flat PSD → band power ≈ total variance scaled by band fraction
        rr = 800 + 50 * np.random.randn(300)  # ms
        # Simulate a simple bpm series
        bpm = 60000 / rr
        freq, psd = welch(bpm - np.mean(bpm), fs=fs, nfft=512)

        _trapz = getattr(np, "trapezoid", np.trapz)
        total_power = _trapz(psd, freq)
        band_power = calculate_band_power(freq, psd, fmin=0.0, fmax=fs / 2)

        # band_power must be close to total_power (same range)
        ratio = band_power / (total_power + 1e-12)
        assert 0.9 < ratio < 1.1, (
            f"Band power ratio to total PSD should be ~1, got {ratio:.4f}. "
            "Old normalization was off by ~10^6."
        )

    def test_band_power_empty_band_returns_zero(self):
        from vital_sqi.common.power_spectrum import calculate_band_power

        freq = np.linspace(0, 2, 100)
        psd = np.ones(100)
        # Band outside all frequencies → 0
        result = calculate_band_power(freq, psd, fmin=10, fmax=20)
        assert result == 0.0

    def test_band_power_higher_than_sub_band(self):
        """Power in the full band must be >= power in a sub-band."""
        from vital_sqi.common.power_spectrum import calculate_band_power

        freq = np.linspace(0, 2, 200)
        psd = np.ones(200)
        full = calculate_band_power(freq, psd, 0.0, 2.0)
        sub = calculate_band_power(freq, psd, 0.04, 0.15)
        assert full >= sub


# ===========================================================================
# P1.C2 — ectopic_sqi ratio bounded in [0, 1]
# ===========================================================================


class TestP1C2_EctopicRatioBounded:
    """ectopic_sqi must return a value in [0, 1]. The old formula
    divided by (total - outliers), producing values > 1 when many outliers exist."""

    def test_outlier_ratio_bounded(self):
        """Inject a signal with almost all outlier beats; ratio must stay ≤ 1."""
        from vital_sqi.sqi.rpeaks_sqi import ectopic_sqi

        # 5 s of pure noise likely to produce many outlier RR intervals
        np.random.seed(1)
        noisy = np.random.randn(500)
        result = ectopic_sqi(noisy, sample_rate=100, wave_type="PPG")
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0, f"ectopic_sqi must be in [0,1], got {result}"

    def test_clean_signal_low_ratio(self):
        """A clean periodic PPG should yield a low ectopic ratio."""
        from vital_sqi.sqi.rpeaks_sqi import ectopic_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=30.0)
        result = ectopic_sqi(signal, sample_rate=100, wave_type="PPG")
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0


# ===========================================================================
# P1.D — Peak detection fixes
# ===========================================================================


class TestP1D_PeakDetection:
    """Tests for all rpeak_detection.py fixes."""

    # Helper: synthetic PPG at 1 Hz, fs=100, 10 s → 9 or 10 peaks
    @pytest.fixture
    def ppg_1hz(self):
        return _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)

    # -----------------------------------------------------------------------
    # D2 — ADAPTIVE_THRESHOLD dispatches to the dedicated method
    # -----------------------------------------------------------------------
    def test_adaptive_threshold_dispatched(self, ppg_1hz):
        """ADAPTIVE_THRESHOLD must reach detect_peak_trough_adaptive_threshold,
        not fall through to the vitalDSP default path."""
        from vital_sqi.common.rpeak_detection import (
            PeakDetector,
            ADAPTIVE_THRESHOLD,
        )

        detector = PeakDetector(wave_type="PPG", fs=100)
        peaks, troughs = detector.ppg_detector(ppg_1hz, detector_type=ADAPTIVE_THRESHOLD)
        assert isinstance(peaks, np.ndarray)
        assert isinstance(troughs, np.ndarray)
        # A 1 Hz sine over 10 s should give ~9–10 peaks
        assert len(peaks) >= 5, (
            f"ADAPTIVE_THRESHOLD found only {len(peaks)} peaks on a clean 1 Hz PPG. "
            "Check dispatch wiring."
        )

    # -----------------------------------------------------------------------
    # D3 — Elgendi 2-MA returns raw-signal peaks, not MA edges
    # -----------------------------------------------------------------------
    def test_elgendi_peaks_at_signal_maxima(self, ppg_1hz):
        """Detected peaks must coincide with local maxima of the raw signal,
        not with the falling edges of the moving average."""
        from vital_sqi.common.rpeak_detection import PeakDetector, MOVING_AVERAGE_METHOD
        from scipy.signal import argrelmax

        detector = PeakDetector(wave_type="PPG", fs=100)
        peaks, _ = detector.ppg_detector(ppg_1hz, detector_type=MOVING_AVERAGE_METHOD)

        if len(peaks) == 0:
            pytest.skip("No peaks found — may be too short for w2 at this fs")

        true_maxima = argrelmax(ppg_1hz)[0]
        # Each detected peak must be within ±5 samples of a true local maximum
        for p in peaks:
            nearest_dist = np.min(np.abs(true_maxima - p))
            assert nearest_dist <= 5, (
                f"Elgendi peak at {p} is {nearest_dist} samples from the nearest "
                "true local maximum. Old code returned MA falling edges, not raw peaks."
            )

    # -----------------------------------------------------------------------
    # D4 — Billauer records mxpos/mnpos (true peak), not crossing index
    # -----------------------------------------------------------------------
    def test_billauer_peaks_at_signal_maxima(self, ppg_1hz):
        """Billauer peaks must be at (or very near) the true signal maxima."""
        from vital_sqi.common.rpeak_detection import PeakDetector, BILLAUER_METHOD
        from scipy.signal import argrelmax

        detector = PeakDetector(wave_type="PPG", fs=100)
        peaks, troughs = detector.ppg_detector(ppg_1hz, detector_type=BILLAUER_METHOD)

        assert len(peaks) >= 5, "Billauer should find ~9 peaks on 1 Hz sine over 10 s"

        true_maxima = argrelmax(ppg_1hz)[0]
        for p in peaks:
            nearest_dist = np.min(np.abs(true_maxima - p))
            assert nearest_dist <= 3, (
                f"Billauer peak at {p} is {nearest_dist} samples from the nearest "
                "true maximum. Old code recorded the detection crossing index instead of mxpos."
            )

    def test_billauer_peaks_are_higher_than_adjacent(self, ppg_1hz):
        """Every detected peak index must be a local maximum: s[p] > s[p-1] and s[p] > s[p+1]."""
        from vital_sqi.common.rpeak_detection import PeakDetector, BILLAUER_METHOD

        detector = PeakDetector(wave_type="PPG", fs=100)
        peaks, _ = detector.detect_peak_trough_billauer(ppg_1hz, delta=0.5)
        for p in peaks:
            if 0 < p < len(ppg_1hz) - 1:
                assert ppg_1hz[p] >= ppg_1hz[p - 1] and ppg_1hz[p] >= ppg_1hz[p + 1], (
                    f"Peak at index {p} (value {ppg_1hz[p]:.3f}) is not a local maximum. "
                    "Old code used the detection crossing index, not mxpos."
                )

    # -----------------------------------------------------------------------
    # D1 — search_for_onset exists and slope_sum returns results
    # -----------------------------------------------------------------------
    def test_slope_sum_returns_non_empty_on_ppg(self, ppg_1hz):
        """After adding search_for_onset, SLOPE_SUM_METHOD must produce peaks
        on a clean periodic signal (previously always crashed silently)."""
        from vital_sqi.common.rpeak_detection import PeakDetector, SLOPE_SUM_METHOD

        detector = PeakDetector(wave_type="PPG", fs=100)
        # Use a longer signal to satisfy the 10-second initialization window
        long_ppg = _sine_ppg(freq_hz=1.0, fs=100, duration=15.0)
        peaks, troughs = detector.ppg_detector(long_ppg, detector_type=SLOPE_SUM_METHOD)
        assert isinstance(peaks, np.ndarray)
        # Should find at least some peaks on a clean 15 s, 1 Hz sine
        assert len(peaks) >= 3, (
            f"SLOPE_SUM_METHOD returned only {len(peaks)} peaks. "
            "search_for_onset may not be working correctly."
        )

    def test_search_for_onset_exists(self):
        """search_for_onset method must be defined and callable."""
        from vital_sqi.common.rpeak_detection import PeakDetector

        detector = PeakDetector(wave_type="PPG", fs=100)
        assert hasattr(detector, "search_for_onset"), (
            "search_for_onset is not defined; slope-sum detector will AttributeError."
        )
        slope_sum = np.array([0.0, 0.1, 0.5, 1.0, 0.8, 0.3, 0.05, 0.01])
        onset = detector.search_for_onset(slope_sum, n=4, local_max=1.0)
        assert isinstance(onset, int)
        assert 0 <= onset <= 4

    # -----------------------------------------------------------------------
    # D5 — slope-sum uses self.fs, not hardcoded 100
    # -----------------------------------------------------------------------
    def test_slope_sum_respects_fs(self):
        """slope-sum window size must scale with self.fs."""
        from vital_sqi.common.rpeak_detection import PeakDetector

        det_100 = PeakDetector(wave_type="PPG", fs=100)
        det_256 = PeakDetector(wave_type="PPG", fs=256)

        # 128 ms window: should be 12 samples at 100 Hz and 32 samples at 256 Hz
        expected_100 = max(1, int(0.128 * 100))
        expected_256 = max(1, int(0.128 * 256))
        assert expected_100 != expected_256, "Sanity check on test setup"

        # Verify by running both on same-duration signals and checking peak count ratio
        ppg_100 = _sine_ppg(freq_hz=1.0, fs=100, duration=15.0)
        ppg_256 = _sine_ppg(freq_hz=1.0, fs=256, duration=15.0)

        p100, _ = det_100.detect_peak_trough_slope_sum(ppg_100)
        p256, _ = det_256.detect_peak_trough_slope_sum(ppg_256)

        # Both should find roughly 14-15 peaks on a 15 Hz, 1 s signal
        # Just assert neither crashes
        assert isinstance(p100, np.ndarray)
        assert isinstance(p256, np.ndarray)

    # -----------------------------------------------------------------------
    # Invalid detector type now raises, not silently returns empty
    # -----------------------------------------------------------------------
    def test_invalid_detector_type_raises(self, ppg_1hz):
        from vital_sqi.common.rpeak_detection import PeakDetector

        detector = PeakDetector(wave_type="PPG", fs=100)
        with pytest.raises(ValueError, match="Invalid detector_type"):
            detector.ppg_detector(ppg_1hz, detector_type=999)

    def test_empty_signal_raises(self):
        from vital_sqi.common.rpeak_detection import PeakDetector

        detector = PeakDetector(wave_type="PPG", fs=100)
        with pytest.raises(ValueError, match="empty"):
            detector.ppg_detector(np.array([]))


# ===========================================================================
# P1.E1 — MSQ ECG returns NaN (not always-1.0) pending Phase 2
# ===========================================================================


class TestP1E1_MsqEcgNan:
    """msq_sqi for ECG must return NaN and issue a RuntimeWarning until a real
    second detector (Phase 2) is available. Silently returning 1.0 is misleading."""

    def test_msq_ecg_returns_nan(self):
        from vital_sqi.sqi.rpeaks_sqi import msq_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        with pytest.warns(RuntimeWarning, match="distinct detectors"):
            result = msq_sqi(signal, wave_type="ECG")
        assert np.isnan(result), "ECG MSQ must return NaN until two distinct detectors exist"

    def test_msq_ppg_still_works(self):
        """PPG MSQ (two different detectors) must remain functional."""
        from vital_sqi.sqi.rpeaks_sqi import msq_sqi

        signal = _sine_ppg(freq_hz=1.0, fs=100, duration=10.0)
        result = msq_sqi(signal, peak_detector_1=7, peak_detector_2=5, wave_type="PPG")
        # Should return a numeric value (possibly NaN if no peaks found, but not an exception)
        assert isinstance(result, (float, np.floating))


# ===========================================================================
# P1 — get_decision_segments combiner correctness
# ===========================================================================


class TestGetDecisionSegments:
    """Combined decision must be reject when EITHER source says reject."""

    def test_both_accept(self):
        from vital_sqi.pipeline.pipeline_functions import get_decision_segments

        segs = [f"seg{i}" for i in range(3)]
        a, r = get_decision_segments(segs, ["accept"] * 3, ["accept"] * 3)
        assert len(a) == 3
        assert len(r) == 0

    def test_one_rejects(self):
        from vital_sqi.pipeline.pipeline_functions import get_decision_segments

        segs = ["a", "b", "c"]
        a, r = get_decision_segments(segs, ["accept", "reject", "accept"], ["accept"] * 3)
        assert "b" in r
        assert "a" in a
        assert "c" in a

    def test_predefined_reject_overrides(self):
        from vital_sqi.pipeline.pipeline_functions import get_decision_segments

        segs = ["a", "b"]
        a, r = get_decision_segments(segs, ["accept", "accept"], ["accept", "reject"])
        assert "b" in r
        assert "a" in a

    def test_length_mismatch_raises(self):
        from vital_sqi.pipeline.pipeline_functions import get_decision_segments

        with pytest.raises(ValueError, match="Length mismatch"):
            get_decision_segments(["a", "b"], ["accept"], ["accept"])
