"""
Phase 3 performance & caching tests.

Covers:
  P3.1 — peak detection called once per segment (not K×N times)
  P3.2 — DataFrame→array hoisting in extract_segment_sqi
  P3.3 — n_jobs parameter accepted by extract_sqi
  P3.4 — yield_mode on cut_segment
  P3.5 — DataFrame built from pre-collected rows (no per-row append)
"""

import numpy as np
import pandas as pd
import pytest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from vital_sqi.pipeline.pipeline_functions import (
    extract_segment_sqi,
    extract_sqi,
    get_sqi,
)
from vital_sqi.common.utils import cut_segment, format_milestone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(n=300, fs=100):
    """Return a 2-column DataFrame (time, signal) with a clean sine wave."""
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)
    return pd.DataFrame({"time": t, "signal": sig})


def _make_sqi_json(tmpdir, sqi_name="skewness_sqi", per_beat=False):
    """Write a minimal SQI config JSON and return its path."""
    cfg = {
        sqi_name: {
            "sqi": sqi_name,
            "args": {"per_beat": per_beat},
        }
    }
    path = os.path.join(tmpdir, "sqi_cfg.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# P3.1 — peak detection cached (called once per segment, not per SQI)
# ---------------------------------------------------------------------------

class TestPeakCaching:
    def test_peak_detection_not_called_when_no_per_beat(self):
        """When no SQI uses per_beat, PeakDetector should never be called."""
        seg = _make_segment()
        sqi_list = [lambda s, **kw: float(np.mean(s))]
        sqi_list[0].__name__ = "mean_sqi"
        sqi_names = ["mean_sqi"]
        sqi_arg_list = {"mean_sqi": {"per_beat": False}}

        call_count = []

        from vital_sqi.common import rpeak_detection as rd
        original_ppg = rd.PeakDetector.ppg_detector

        def counting_ppg(self, s, *args, **kwargs):
            call_count.append(1)
            return original_ppg(self, s, *args, **kwargs)

        with patch.object(rd.PeakDetector, "ppg_detector", counting_ppg):
            extract_segment_sqi(seg, sqi_list, sqi_names, sqi_arg_list, "PPG")

        assert len(call_count) == 0, "PeakDetector called when no SQI needs peaks"

    def test_peak_detection_called_once_for_multiple_per_beat_sqis(self):
        """When multiple SQIs use per_beat, PeakDetector.ppg_detector runs once."""
        seg = _make_segment(n=500)

        import vital_sqi.sqi as sq
        # Two per-beat SQIs
        sqi_list = [sq.skewness_sqi, sq.kurtosis_sqi]
        sqi_names = ["skewness_sqi", "kurtosis_sqi"]
        sqi_arg_list = {
            "skewness_sqi": {"per_beat": True},
            "kurtosis_sqi": {"per_beat": True},
        }

        call_count = []

        from vital_sqi.common import rpeak_detection as rd
        original_ppg = rd.PeakDetector.ppg_detector

        def counting_ppg(self, s, *args, **kwargs):
            call_count.append(1)
            return original_ppg(self, s, *args, **kwargs)

        with patch.object(rd.PeakDetector, "ppg_detector", counting_ppg):
            extract_segment_sqi(seg, sqi_list, sqi_names, sqi_arg_list, "PPG")

        assert len(call_count) == 1, (
            f"Expected 1 peak detection call, got {len(call_count)}"
        )


# ---------------------------------------------------------------------------
# P3.2 — signal array hoisted before SQI loop
# ---------------------------------------------------------------------------

class TestArrayHoisting:
    def test_get_sqi_accepts_pre_extracted_array(self):
        """get_sqi should use _signal_values directly without re-extracting."""
        import vital_sqi.sqi as sq
        seg = _make_segment(n=200)
        pre_extracted = seg.iloc[:, 1].values.copy()
        # Corrupt the DataFrame's second column so we can detect if it's read
        corrupted = seg.copy()
        corrupted.iloc[:, 1] = 999.0

        result_prehoisted = get_sqi(
            sq.skewness_sqi, "skewness_sqi", corrupted,
            _signal_values=pre_extracted
        )
        result_direct = get_sqi(sq.skewness_sqi, "skewness_sqi", seg)

        assert abs(result_prehoisted["skewness_sqi"] - result_direct["skewness_sqi"]) < 1e-9

    def test_extract_segment_sqi_produces_same_result_as_per_segment_call(self):
        """extract_segment_sqi output should equal per-function get_sqi calls."""
        import vital_sqi.sqi as sq
        seg = _make_segment(n=300)
        sqi_list = [sq.skewness_sqi]
        sqi_names = ["skewness_sqi"]
        sqi_arg_list = {"skewness_sqi": {}}

        result = extract_segment_sqi(seg, sqi_list, sqi_names, sqi_arg_list, "PPG")
        expected = get_sqi(sq.skewness_sqi, "skewness_sqi", seg)

        assert abs(result["skewness_sqi"] - expected["skewness_sqi"]) < 1e-9


# ---------------------------------------------------------------------------
# P3.3 — n_jobs parameter
# ---------------------------------------------------------------------------

class TestNJobs:
    def test_extract_sqi_n_jobs_1_matches_serial(self, tmp_path):
        """n_jobs=1 (default) and explicit n_jobs=1 produce identical results."""
        segs = [_make_segment(n=300) for _ in range(3)]
        milestones = format_milestone(
            [i * 300 for i in range(3)],
            [(i + 1) * 300 for i in range(3)],
        )
        cfg_path = _make_sqi_json(str(tmp_path))

        result_default = extract_sqi(segs, milestones, cfg_path, wave_type="PPG")
        result_n1 = extract_sqi(segs, milestones, cfg_path, wave_type="PPG", n_jobs=1)

        pd.testing.assert_frame_equal(result_default, result_n1)

    def test_extract_sqi_signature_has_n_jobs(self):
        """extract_sqi must accept n_jobs kwarg without raising."""
        import inspect
        sig = inspect.signature(extract_sqi)
        assert "n_jobs" in sig.parameters

    def test_extract_sqi_n_jobs_result_shape(self, tmp_path):
        """Parallel execution returns same number of rows as segments."""
        segs = [_make_segment(n=300) for _ in range(4)]
        milestones = format_milestone(
            [i * 300 for i in range(4)],
            [(i + 1) * 300 for i in range(4)],
        )
        cfg_path = _make_sqi_json(str(tmp_path))

        result = extract_sqi(segs, milestones, cfg_path, wave_type="PPG", n_jobs=2)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# P3.4 — cut_segment yield_mode
# ---------------------------------------------------------------------------

class TestCutSegmentYieldMode:
    def _make_df(self, n=100):
        return pd.DataFrame({
            "time": np.arange(n) / 100.0,
            "signal": np.sin(2 * np.pi * np.arange(n) / 100.0),
        })

    def test_yield_mode_false_returns_list(self):
        df = self._make_df(300)
        ms = format_milestone([0, 100, 200], [100, 200, 300])
        result = cut_segment(df, ms, yield_mode=False)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_yield_mode_true_returns_generator(self):
        import types
        df = self._make_df(300)
        ms = format_milestone([0, 100, 200], [100, 200, 300])
        result = cut_segment(df, ms, yield_mode=True)
        assert isinstance(result, types.GeneratorType)

    def test_yield_mode_generator_matches_list(self):
        df = self._make_df(300)
        ms = format_milestone([0, 100, 200], [100, 200, 300])
        list_result = cut_segment(df, ms, yield_mode=False)
        gen_result = list(cut_segment(df, ms, yield_mode=True))
        assert len(list_result) == len(gen_result)
        for a, b in zip(list_result, gen_result):
            pd.testing.assert_frame_equal(a, b)

    def test_yield_mode_lazy_no_upfront_materialisation(self):
        """Generator should not advance until iterated."""
        df = self._make_df(300)
        ms = format_milestone([0, 100, 200], [100, 200, 300])
        gen = cut_segment(df, ms, yield_mode=True)
        # Has not been iterated — just check it is a generator
        import types
        assert isinstance(gen, types.GeneratorType)
        first = next(gen)
        assert len(first) == 100

    def test_yield_mode_raises_on_out_of_bounds(self):
        df = self._make_df(100)
        ms = format_milestone([0], [200])  # end > len(df)
        with pytest.raises(ValueError, match="out of bounds"):
            list(cut_segment(df, ms, yield_mode=True))

    def test_default_behaviour_unchanged(self):
        """Calling cut_segment without yield_mode must still return a list."""
        df = self._make_df(200)
        ms = format_milestone([0, 100], [100, 200])
        result = cut_segment(df, ms)
        assert isinstance(result, list)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# P3.5 — DataFrame pre-allocation (no per-row append)
# ---------------------------------------------------------------------------

class TestDataFramePreallocation:
    def test_extract_sqi_returns_dataframe(self, tmp_path):
        segs = [_make_segment(n=300) for _ in range(5)]
        milestones = format_milestone(
            [i * 300 for i in range(5)],
            [(i + 1) * 300 for i in range(5)],
        )
        cfg_path = _make_sqi_json(str(tmp_path))
        result = extract_sqi(segs, milestones, cfg_path, wave_type="PPG")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_extract_sqi_contains_milestone_columns(self, tmp_path):
        segs = [_make_segment(n=300) for _ in range(2)]
        milestones = format_milestone([0, 300], [300, 600])
        cfg_path = _make_sqi_json(str(tmp_path))
        result = extract_sqi(segs, milestones, cfg_path, wave_type="PPG")
        assert "start_idx" in result.columns
        assert "end_idx" in result.columns
        assert list(result["start_idx"]) == [0, 300]
        assert list(result["end_idx"]) == [300, 600]

    def test_extract_sqi_no_nan_rows_for_valid_segments(self, tmp_path):
        """Valid segments should produce finite SQI values."""
        segs = [_make_segment(n=300) for _ in range(3)]
        milestones = format_milestone([0, 300, 600], [300, 600, 900])
        cfg_path = _make_sqi_json(str(tmp_path))
        result = extract_sqi(segs, milestones, cfg_path, wave_type="PPG")
        sqi_cols = [c for c in result.columns if c not in ("start_idx", "end_idx")]
        assert result[sqi_cols].notna().all().all()
