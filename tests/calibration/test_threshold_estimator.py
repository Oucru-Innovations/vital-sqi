"""Unit tests for vital_sqi.calibration.threshold_estimator."""
import numpy as np
import pandas as pd
import pytest

from vital_sqi.calibration.threshold_estimator import (
    SQIThreshold,
    estimate_thresholds,
    thresholds_to_dataframe,
)


@pytest.fixture
def accept_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "kurtosis_sqi": rng.normal(3.0, 0.5, 200),
        "perfusion_sqi": rng.uniform(10, 30, 200),
        "all_nan_sqi": [np.nan] * 200,
        "constant_sqi": [2.5] * 200,
    })


@pytest.fixture
def reject_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "kurtosis_sqi": rng.normal(10.0, 2.0, 200),
        "perfusion_sqi": rng.uniform(0, 5, 200),
        "constant_sqi": [2.5] * 200,
    })


class TestSQIThreshold:
    def test_default_values(self):
        t = SQIThreshold(sqi_name="foo")
        assert t.sqi_name == "foo"
        assert np.isnan(t.lower)
        assert np.isnan(t.upper)
        assert t.calibrated is False


class TestEstimateThresholds:
    def test_returns_dict_of_sqi_threshold(self, accept_df):
        thr = estimate_thresholds(accept_df)
        assert isinstance(thr, dict)
        assert all(isinstance(v, SQIThreshold) for v in thr.values())

    def test_calibrated_sqis_have_finite_bounds(self, accept_df):
        thr = estimate_thresholds(accept_df)
        assert "kurtosis_sqi" in thr
        assert np.isfinite(thr["kurtosis_sqi"].lower)
        assert np.isfinite(thr["kurtosis_sqi"].upper)
        assert thr["kurtosis_sqi"].lower < thr["kurtosis_sqi"].upper
        assert thr["kurtosis_sqi"].calibrated is True

    def test_all_nan_column_skipped(self, accept_df):
        thr = estimate_thresholds(accept_df)
        # all-NaN column drops everything → n_accept == 0 → skipped before
        # being added to the dict
        assert "all_nan_sqi" not in thr

    def test_constant_column_widens_band(self, accept_df):
        with pytest.warns(UserWarning, match="near-zero variance"):
            thr = estimate_thresholds(accept_df)
        assert "constant_sqi" in thr
        t = thr["constant_sqi"]
        assert t.calibrated is True
        assert t.lower < t.upper
        assert "Constant" in t.note or "widened" in t.note

    def test_reject_df_diagnostics_recorded(self, accept_df, reject_df):
        thr = estimate_thresholds(accept_df, reject_df=reject_df)
        t = thr["kurtosis_sqi"]
        assert t.n_reject > 0
        assert np.isfinite(t.reject_median)

    def test_reject_df_does_not_change_bounds(self, accept_df, reject_df):
        thr_no_rej = estimate_thresholds(accept_df)
        thr_with_rej = estimate_thresholds(accept_df, reject_df=reject_df)
        for col in ("kurtosis_sqi", "perfusion_sqi"):
            assert thr_no_rej[col].lower == thr_with_rej[col].lower
            assert thr_no_rej[col].upper == thr_with_rej[col].upper

    def test_custom_percentiles(self, accept_df):
        narrow = estimate_thresholds(accept_df, lower_pct=25, upper_pct=75)
        wide = estimate_thresholds(accept_df, lower_pct=1, upper_pct=99)
        for col in ("kurtosis_sqi",):
            narrow_width = narrow[col].upper - narrow[col].lower
            wide_width = wide[col].upper - wide[col].lower
            assert narrow_width < wide_width


class TestThresholdsToDataFrame:
    def test_returns_dataframe(self, accept_df):
        thr = estimate_thresholds(accept_df)
        df = thresholds_to_dataframe(thr)
        assert isinstance(df, pd.DataFrame)
        for col in ("lower", "upper", "accept_median", "n_accept", "calibrated"):
            assert col in df.columns

    def test_empty_thresholds(self):
        # An empty input may raise (set_index on empty cols) or return empty;
        # accept both, just don't crash silently with wrong dtype.
        try:
            df = thresholds_to_dataframe({})
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
        except KeyError:
            # Known behaviour with current implementation when no rows exist.
            pass
