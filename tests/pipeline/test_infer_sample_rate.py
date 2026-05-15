"""Regression tests for fs/wave_type plumbing through extract_segment_sqi.

Background
----------
Before this fix, `get_nn` defaulted to ``wave_type='PPG'`` and
``sample_rate=100``.  Callers in ``pipeline_functions`` did NOT pass either
argument, so an ECG recording at 128 Hz was silently processed as PPG at
100 Hz — peak detection failed, every NN-derived SQI returned NaN, and the
console filled with ``ERROR:root:Error in get_nn function: ...``.

These tests pin down:

1. ``_infer_sample_rate`` reads fs from the segment's timestamp column.
2. The function falls back to ``sample_rate`` keys in the SQI arg dict.
3. It returns a positive default rather than NaN when both sources fail.
"""

import numpy as np
import pandas as pd
import pytest

from vital_sqi.pipeline.pipeline_functions import (
    _infer_sample_rate,
    _scan_args_for_fs,
)


def _make_segment(fs: float, n: int = 1000) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(seconds=1 / fs))
    return pd.DataFrame({"timestamps": timestamps, "signal": np.zeros(n)})


class TestInferSampleRate:
    def test_datetime_timestamps_yield_correct_fs(self):
        seg = _make_segment(fs=128.0)
        assert _infer_sample_rate(seg, {}) == pytest.approx(128.0, rel=1e-3)

    def test_numeric_seconds_timestamps(self):
        seg = pd.DataFrame(
            {
                "timestamps": np.arange(0, 10, 1 / 250.0),
                "signal": np.zeros(2500),
            }
        )
        assert _infer_sample_rate(seg, {}) == pytest.approx(250.0, rel=1e-3)

    def test_timestamps_win_over_baked_in_sample_rate(self):
        # The bug scenario: sqi_dict has sample_rate=100 baked in but the
        # actual segment is 128 Hz.  Inference must prefer the timestamps.
        seg = _make_segment(fs=128.0)
        baked = {"ectopic_sqi": {"sample_rate": 100}}
        result = _infer_sample_rate(seg, baked)
        assert result == pytest.approx(128.0, rel=1e-3)

    def test_fall_back_to_args_when_no_timestamps(self):
        # A degenerate DataFrame with only one column → can't infer from
        # timestamps; pull fs from the args map instead.
        seg = pd.DataFrame({"signal": np.zeros(100)})
        result = _infer_sample_rate(seg, {"x": {"sample_rate": 256}})
        assert result == 256.0

    def test_fall_back_to_default(self):
        seg = pd.DataFrame({"signal": np.zeros(100)})
        result = _infer_sample_rate(seg, {})
        assert result > 0  # default must be positive, not NaN
        assert np.isfinite(result)


class TestScanArgsForFs:
    def test_flat_kwargs(self):
        assert _scan_args_for_fs({"sample_rate": 256}) == 256.0

    def test_flat_kwargs_sampling_rate_alias(self):
        assert _scan_args_for_fs({"sampling_rate": 250}) == 250.0

    def test_nested_argmap(self):
        argmap = {
            "kurtosis_sqi": {"axis": 0},
            "ectopic_sqi": {"sample_rate": 128},
        }
        assert _scan_args_for_fs(argmap) == 128.0

    def test_no_match_returns_none(self):
        assert _scan_args_for_fs({"foo": "bar"}) is None
        assert _scan_args_for_fs({}) is None

    def test_non_dict_returns_none(self):
        assert _scan_args_for_fs(None) is None
        assert _scan_args_for_fs(42) is None


# ---------------------------------------------------------------------------
# Integration: extract_sqi must produce non-NaN HRV SQIs on a 128 Hz ECG.
# ---------------------------------------------------------------------------


class TestExtractSqiOnNonDefaultFs:
    def test_hrv_sqis_populated_at_128hz(self, tmp_path):
        """End-to-end check: synth 128 Hz ECG → extract_sqi → HRV SQIs are finite.

        This is the regression for the user's report of NaN HRV columns on
        the Oucru ECG file (128 Hz).
        """
        from vital_sqi.common.utils import generate_timestamp
        from vital_sqi.pipeline.pipeline_functions import extract_sqi
        from vital_sqi.preprocess.segment_split import split_segment

        fs = 128
        duration_s = 60
        n_samples = fs * duration_s
        # Crude synthetic ECG: train of sharp R-peaks @ 75 bpm.
        t = np.arange(n_samples) / fs
        signal = np.zeros(n_samples)
        beat_period = fs * 60 // 75  # ~102 samples between beats
        for i in range(0, n_samples, int(beat_period)):
            if i + 5 < n_samples:
                signal[i:i + 5] = [10, 80, 40, -20, 5]

        seg_df = pd.DataFrame({
            "timestamps": generate_timestamp(None, fs, n_samples),
            "signal": signal,
        })
        segments, milestones = split_segment(
            seg_df, sampling_rate=fs, split_type=0,
            duration=30, overlapping=0, wave_type="ECG",
        )
        assert len(segments) >= 1

        # Minimal sqi_dict — just the HRV columns that triggered the bug.
        import json
        sqi_dict = {
            "mean_nn_sqi": {"sqi": "mean_nn_sqi", "args": {}},
            "sdnn_sqi":    {"sqi": "sdnn_sqi",    "args": {}},
            "hr_mean_sqi": {"sqi": "hr_mean_sqi", "args": {}},
        }
        sqi_path = tmp_path / "sqi_dict.json"
        sqi_path.write_text(json.dumps(sqi_dict))

        result = extract_sqi(segments, milestones, str(sqi_path), wave_type="ECG")
        # At least one segment should yield finite HRV values; the test
        # signal is clean enough that all should succeed.
        for col in ("mean_nn_sqi", "sdnn_sqi", "hr_mean_sqi"):
            finite = result[col].dropna()
            assert len(finite) > 0, f"{col} was all NaN — fs/wave_type not plumbed"
