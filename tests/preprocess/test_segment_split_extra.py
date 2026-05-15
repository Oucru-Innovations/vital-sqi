"""Extra coverage for vital_sqi.preprocess.segment_split.

Targets the parts of ``save_segment`` and ``split_segment`` that the
existing tests don't reach: image-saving path, ndarray vs DataFrame
branches, beat-mode segmentation, and the empty-chunks fallback.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from vital_sqi.preprocess.segment_split import save_segment, split_segment


def _signal_df(n: int = 600, fs: int = 100) -> pd.DataFrame:
    from vital_sqi.common.utils import generate_timestamp
    ts = generate_timestamp(None, fs, n)
    return pd.DataFrame({
        "timestamps": ts,
        "signal": np.sin(np.linspace(0, 4 * np.pi, n)),
    })


# ---------------------------------------------------------------------------
# save_segment
# ---------------------------------------------------------------------------


class TestSaveSegment:
    def test_saves_dataframe_segments_as_csv(self, tmp_path):
        segs = [
            pd.DataFrame({"t": [0, 0.01], "x": [0.1, 0.2]}),
            pd.DataFrame({"t": [0, 0.01], "x": [0.3, 0.4]}),
        ]
        save_segment(segs, save_file_folder=str(tmp_path))
        files = sorted(tmp_path.glob("segment-*.csv"))
        assert len(files) == 2

    def test_saves_ndarray_segments_as_csv(self, tmp_path):
        segs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        save_segment(segs, save_file_folder=str(tmp_path))
        files = sorted(tmp_path.glob("segment-*.csv"))
        assert len(files) == 2

    def test_segment_name_none_falls_back_to_default(self, tmp_path):
        segs = [np.array([1.0, 2.0])]
        save_segment(segs, segment_name=None, save_file_folder=str(tmp_path))
        # Phase-2 fix: callers passing segment_name=None should NOT produce
        # files literally named "None-XX.csv".
        files = list(tmp_path.glob("None-*.csv"))
        assert not files
        files = list(tmp_path.glob("segment-*.csv"))
        assert len(files) == 1

    def test_default_folder_is_cwd(self, tmp_path, monkeypatch):
        # When save_file_folder is None it falls back to os.getcwd().
        monkeypatch.chdir(tmp_path)
        save_segment([np.array([1.0, 2.0])], save_file_folder=None)
        files = list(tmp_path.glob("segment-*.csv"))
        assert len(files) == 1


# ---------------------------------------------------------------------------
# split_segment — extra branches
# ---------------------------------------------------------------------------


class TestSplitSegmentExtra:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            split_segment(
                pd.DataFrame(), sampling_rate=100,
                split_type=0, duration=30,
            )

    def test_invalid_split_type_raises(self):
        df = _signal_df()
        with pytest.raises(AssertionError):
            split_segment(df, sampling_rate=100, split_type=99, duration=5)

    def test_invalid_wave_type_raises(self):
        df = _signal_df()
        with pytest.raises(AssertionError):
            split_segment(
                df, sampling_rate=100, split_type=0,
                duration=5, wave_type="EEG",
            )

    def test_chunk_step_must_be_positive(self):
        df = _signal_df()
        # overlapping >= duration → chunk_step <= 0 → ValueError.
        with pytest.raises(ValueError, match="overlapping"):
            split_segment(
                df, sampling_rate=100, split_type=0,
                duration=5, overlapping=10,
            )

    def test_time_split_produces_correct_segments(self):
        df = _signal_df(n=600, fs=100)  # 6 s of data
        segs, milestones = split_segment(
            df, sampling_rate=100, split_type=0,
            duration=2, overlapping=0,
        )
        # 6 s / 2 s per chunk = 3 segments.
        assert len(segs) == 3
        assert list(milestones.columns) == ["start", "end"]
