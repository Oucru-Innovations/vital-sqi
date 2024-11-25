import pytest
import numpy as np
from vital_sqi.sqi.dtw_sqi import dtw_sqi


class TestDtwSqi:
    def setup_class(self):
        """Setup reusable test data."""
        self.valid_signal = [0, 1, 2, 3, 4, 5]
        self.short_signal = [0, 1]
        self.invalid_signal = ["a", "b", "c"]  # Non-numeric values
        self.empty_signal = []
        self.template_types = [0, 1, 2, 3]

    def test_on_invalid_template_type(self):
        """Test invalid template types."""
        invalid_template_types = [2.1, 4, -1, "invalid"]
        for template_type in invalid_template_types:
            with pytest.raises(ValueError, match="Invalid template type"):
                dtw_sqi(self.valid_signal, template_type)

    def test_on_valid_template_types(self):
        """Test valid template types with both modes."""
        for template_type in self.template_types:
            result_dtw = dtw_sqi(self.valid_signal, template_type, simple_mode=False)
            result_simple = dtw_sqi(self.valid_signal, template_type, simple_mode=True)
            assert isinstance(result_dtw, float)
            assert isinstance(result_simple, float)

    def test_on_empty_signal(self):
        """Test behavior with an empty signal."""
        for template_type in self.template_types:
            with pytest.raises(ValueError, match="Empty signal provided."):
                dtw_sqi(self.empty_signal, template_type)

    def test_on_short_signal(self):
        """Test behavior with very short signals."""
        for template_type in self.template_types:
            result = dtw_sqi(self.short_signal, template_type, simple_mode=True)
            assert isinstance(result, float)

    def test_on_non_numeric_signal(self):
        """Test behavior with non-numeric signals."""
        for template_type in self.template_types:
            with pytest.raises(ValueError, match="Signal contains non-numeric data."):
                dtw_sqi(self.invalid_signal, template_type)

    def test_on_high_template_size(self):
        """Test behavior with large template sizes."""
        large_template_size = 1000
        for template_type in self.template_types:
            result = dtw_sqi(
                self.valid_signal, template_type, template_size=large_template_size
            )
            assert isinstance(result, float)

    def test_on_low_template_size(self):
        """Test behavior with small template sizes."""
        small_template_size = 2
        for template_type in self.template_types:
            result = dtw_sqi(
                self.valid_signal, template_type, template_size=small_template_size
            )
            assert isinstance(result, float)

    def test_on_edge_case_template_size(self):
        """Test behavior with edge-case template sizes."""
        for template_type in self.template_types:
            with pytest.raises(
                ValueError, match="Number of samples must be greater than zero."
            ):
                dtw_sqi(self.valid_signal, template_type, template_size=0)

            with pytest.raises(
                ValueError, match="Number of samples must be greater than zero."
            ):
                dtw_sqi(self.valid_signal, template_type, template_size=-10)

    def test_on_equal_signal_and_template(self):
        """Test behavior when the signal matches the template."""
        # Using template_type=0 for simplicity
        result = dtw_sqi([1, 1, 1, 1], 0, simple_mode=False)
        assert isinstance(result, float)
        # assert result == pytest.approx(0.0, rel=1e-2)

    def test_on_signal_with_noise(self):
        """Test behavior with noisy signals."""
        noisy_signal = np.array(self.valid_signal) + np.random.normal(
            0, 0.1, len(self.valid_signal)
        )
        for template_type in self.template_types:
            result = dtw_sqi(noisy_signal, template_type, simple_mode=False)
            assert isinstance(result, float)

    def test_simple_mode_vs_dtw_mode(self):
        """Compare results of simple mode and DTW mode."""
        for template_type in self.template_types:
            result_simple = dtw_sqi(self.valid_signal, template_type, simple_mode=True)
            result_dtw = dtw_sqi(self.valid_signal, template_type, simple_mode=False)
            assert isinstance(result_simple, float)
            assert isinstance(result_dtw, float)
            assert result_simple >= 0
            assert result_dtw >= 0

    # def test_on_high_dimensional_signal(self):
    #     """Test behavior with multi-dimensional signals."""
    #     high_dim_signal = np.array([[0, 1], [2, 3]])
    #     for template_type in self.template_types:
    #         with pytest.raises(ValueError, match="Signal must be one-dimensional."):
    #             dtw_sqi(high_dim_signal, template_type)
