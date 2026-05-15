"""Extra coverage for vital_sqi.common.utils edge cases."""
import numpy as np
import pandas as pd
import pytest

from vital_sqi.common import utils as u


class TestCheckValidSignal:
    def test_non_array_raises(self):
        with pytest.raises(ValueError, match="array-like"):
            u.check_valid_signal("not_an_array")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            u.check_valid_signal([])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            u.check_valid_signal(np.zeros((3, 3)))

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="non-numeric"):
            u.check_valid_signal(np.array(["a", "b", "c"]))

    def test_valid_signal_returns_true(self):
        assert u.check_valid_signal([1, 2, 3]) is True


class TestCutSegmentBounds:
    def test_out_of_bounds_raises(self):
        df = pd.DataFrame({"timestamps": pd.date_range("2024", periods=10), "x": range(10)})
        milestones = pd.DataFrame({"start": [0], "end": [100]})  # past end
        with pytest.raises(ValueError, match="out of bounds"):
            u.cut_segment(df, milestones)

    def test_start_ge_end_raises(self):
        df = pd.DataFrame({"timestamps": pd.date_range("2024", periods=10), "x": range(10)})
        milestones = pd.DataFrame({"start": [5], "end": [5]})
        with pytest.raises(ValueError, match="less than end"):
            u.cut_segment(df, milestones)


class TestSanitizeSqi:
    def test_replaces_inf_with_median(self):
        out = u.sanitize_sqi([1.0, np.inf, 3.0, np.nan, 5.0])
        assert np.all(np.isfinite(out))

    def test_all_nan_returns_zeros(self):
        out = u.sanitize_sqi([np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(out, np.zeros(3))


class TestCreateRuleDef:
    def test_basic(self):
        rd = u.create_rule_def("kurtosis_sqi", lower_bound=0.5, upper_bound=5.0)
        assert "kurtosis_sqi" in rd
        assert rd["kurtosis_sqi"]["name"] == "kurtosis_sqi"
        assert len(rd["kurtosis_sqi"]["def"]) == 4


class TestGetNN:
    def test_invalid_input_returns_empty(self):
        # Empty signal — RRTransformation will raise → returns empty array
        result = u.get_nn(np.array([]))
        assert isinstance(result, np.ndarray)


class TestCheckSignalFormatExtra:
    def test_rejects_non_array_input(self):
        with pytest.raises(TypeError):
            u.check_signal_format(12345)

    def test_drops_stale_timestamps_column(self):
        df = pd.DataFrame({"timestamps": [1, 2, 3], "signal": [4.0, 5.0, 6.0]})
        out = u.check_signal_format(df)
        # Re-inserted as datetime
        assert np.issubdtype(out["timestamps"].dtype, np.datetime64)

    def test_all_non_numeric_raises(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        with pytest.raises(TypeError):
            u.check_signal_format(df)
