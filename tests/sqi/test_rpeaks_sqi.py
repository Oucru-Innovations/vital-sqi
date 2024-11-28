import pytest
import numpy as np
from scipy import signal
from vitalDSP.utils.synthesize_data import generate_ecg_signal
from vital_sqi.sqi.rpeaks_sqi import (
    ectopic_sqi,
    correlogram_sqi,
    interpolation_sqi,
    msq_sqi,
)


class TestRPeakSQI:
    @pytest.fixture
    def valid_signal(self):
        """Fixture for a valid synthetic signal."""
        np.random.seed(0)
        return np.sin(np.linspace(0, 2 * np.pi, 1000))

    @pytest.fixture
    def invalid_signal(self):
        """Fixture for an invalid signal."""
        return "invalid_signal"

    def test_ectopic_sqi(self):
        """Test the ectopic_sqi function."""
        # Create a simulated PPG-like signal
        sample_rate = 100
        t = np.linspace(0, 10, 10 * sample_rate)  # 10 seconds of signal
        valid_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

        # Test with valid signal and rules
        for rule_index in range(1, 4):  # Test adaptive, linear, spline
            result = ectopic_sqi(
                valid_signal,
                sample_rate=sample_rate,
                wave_type="PPG",
                rule_index=rule_index,
            )
            assert (
                np.isnan(result) or 0.0 <= result <= 1.0
            ), f"Unexpected ectopic ratio for rule {rule_index}: {result}"

        # Test with rule_index=0 (outlier ratio)
        result = ectopic_sqi(
            valid_signal, sample_rate=sample_rate, wave_type="PPG", rule_index=0
        )
        assert (
            np.isnan(result) or 0.0 <= result <= 1.0
        ), f"Unexpected outlier ratio: {result}"

        # Test with invalid rule_index
        with pytest.warns(UserWarning, match="Invalid rule_index"):
            assert np.isnan(
                ectopic_sqi(
                    valid_signal, sample_rate=sample_rate, wave_type="PPG", rule_index=4
                )
            )

        # Test with short signal
        with pytest.warns(
            UserWarning
            # , match="Insufficient RR intervals for analysis"
        ):
            assert np.isnan(ectopic_sqi(valid_signal[:10], sample_rate=sample_rate))

        # Test with empty signal
        with pytest.warns(
            UserWarning,
            # match="No peaks detected in the signal"
        ):
            assert np.isnan(ectopic_sqi([], sample_rate=sample_rate))

        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        for rule_index in range(1, 4):
            result = ectopic_sqi(
                ecg_signal, sample_rate=sfecg, wave_type="ECG", rule_index=rule_index
            )
            assert np.isreal(result)

    def test_correlogram_sqi(self):
        """Test the correlogram_sqi function."""
        # Create a simulated signal
        sample_rate = 100
        t = np.linspace(0, 10, 10 * sample_rate)  # 10 seconds of signal
        valid_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

        # Test with valid signal
        result = correlogram_sqi(
            valid_signal, sample_rate=sample_rate, time_lag=3, n_selection=3
        )
        assert result is not None, f"Expected {result} not None"

        # Test with short signal
        short_signal = valid_signal[:50]  # Signal shorter than time lag
        with pytest.warns(
            UserWarning, match="Signal length is too short for the specified time lag"
        ):
            assert np.isnan(
                correlogram_sqi(
                    short_signal, sample_rate=sample_rate, time_lag=3, n_selection=3
                )
            )

        # Test with flat signal (no peaks)
        flat_signal = np.zeros_like(valid_signal)
        with pytest.warns(
            UserWarning,
            # match="No peaks detected in the autocorrelation function."
        ):
            assert np.isnan(
                correlogram_sqi(
                    flat_signal, sample_rate=sample_rate, time_lag=3, n_selection=3
                )
            )

        sfecg = 256
        N = 100
        Anoise = 0.05
        hrmean = 70
        ecg_signal = generate_ecg_signal(sfecg=sfecg, N=N, Anoise=Anoise, hrmean=hrmean)
        result = correlogram_sqi(ecg_signal, sample_rate=sfecg, wave_type="ECG")
        assert result is not None

    def test_interpolation_sqi(self, valid_signal):
        """Test the interpolation_sqi function."""
        result = interpolation_sqi(valid_signal)
        assert isinstance(result, float)  # Placeholder function

    def test_msq_sqi(self, valid_signal):
        """Test the msq_sqi function."""
        # Create a simulated signal
        sample_rate = 100
        t = np.linspace(0, 10, 10 * sample_rate)  # 10 seconds of signal
        valid_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

        # Test with valid signal
        result = msq_sqi(
            valid_signal, peak_detector_1=7, peak_detector_2=6, wave_type="PPG"
        )
        assert 0.0 <= result <= 1.0, f"Unexpected MSQ value: {result}"

        # Test with empty signal
        with pytest.warns(UserWarning, match="Input signal is empty or invalid."):
            assert np.isnan(
                msq_sqi([], peak_detector_1=7, peak_detector_2=6, wave_type="PPG")
            )

        # Test with no peaks detected
        flat_signal = np.zeros_like(valid_signal)
        with pytest.warns(
            UserWarning, match="No peaks detected by one or both detectors."
        ):
            assert (
                msq_sqi(
                    flat_signal, peak_detector_1=7, peak_detector_2=6, wave_type="PPG"
                )
                == 0.0
            )

        # Test with invalid signal type
        with pytest.warns(UserWarning, match="Input signal is empty or invalid."):
            assert np.isnan(
                msq_sqi(
                    "invalid_signal",
                    peak_detector_1=7,
                    peak_detector_2=6,
                    wave_type="PPG",
                )
            )
