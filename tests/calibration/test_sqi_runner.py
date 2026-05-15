"""Unit tests for vital_sqi.calibration.sqi_runner."""
import numpy as np
import pandas as pd
import pytest

from vital_sqi.calibration.sqi_runner import (
    _make_segment_df,
    _patch_fs,
    compute_sqi_distributions,
    DEFAULT_SQI_ARG_LIST,
)


@pytest.fixture
def fake_segments():
    """Three short synthetic sinusoids — enough to exercise SQI plumbing."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(3):
        t = np.linspace(0, 5, 500)
        sig = np.sin(2 * np.pi * 1.0 * t) + 0.05 * rng.standard_normal(500)
        out.append((sig, 100))
    return out


class TestMakeSegmentDf:
    def test_returns_two_column_dataframe(self):
        sig = np.linspace(0, 1, 100)
        df = _make_segment_df(sig, fs=100)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (100, 2)
        assert "time" in df.columns
        assert "signal" in df.columns


class TestPatchFs:
    def test_patches_sample_rate(self):
        out = _patch_fs({"foo": {"sample_rate": 100}}, fs=250)
        assert out["foo"]["sample_rate"] == 250

    def test_patches_sampling_rate(self):
        out = _patch_fs({"foo": {"sampling_rate": 100}}, fs=250)
        assert out["foo"]["sampling_rate"] == 250

    def test_does_not_add_missing_keys(self):
        out = _patch_fs({"foo": {"other": 1}}, fs=250)
        assert "sample_rate" not in out["foo"]
        assert "sampling_rate" not in out["foo"]

    def test_does_not_mutate_input(self):
        original = {"foo": {"sample_rate": 100}}
        _patch_fs(original, fs=250)
        assert original["foo"]["sample_rate"] == 100


class TestComputeSqiDistributions:
    def test_returns_dataframe(self, fake_segments):
        # Use a small subset of SQIs to keep the test fast.
        df = compute_sqi_distributions(
            fake_segments,
            wave_type="PPG",
            sqi_names=["kurtosis_sqi", "skewness_sqi"],
            show_progress=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(fake_segments)

    def test_replaces_inf_with_nan(self, fake_segments):
        df = compute_sqi_distributions(
            fake_segments,
            wave_type="PPG",
            sqi_names=["kurtosis_sqi"],
            show_progress=False,
        )
        # No infs should survive
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any()

    def test_unknown_sqi_name_warns(self, fake_segments):
        with pytest.warns(UserWarning, match="not in sqi_mapping"):
            compute_sqi_distributions(
                fake_segments,
                wave_type="PPG",
                sqi_names=["definitely_not_a_real_sqi"],
                show_progress=False,
            )

    def test_default_sqi_arg_list_present(self):
        # Sanity: DEFAULT_SQI_ARG_LIST must include the new SQIs added recently.
        for name in ("clipping_sqi", "baseline_wander_sqi",
                     "sample_entropy_sqi", "dfa_sqi", "hurst_sqi"):
            assert name in DEFAULT_SQI_ARG_LIST
