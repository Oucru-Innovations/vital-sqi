import pytest
from vital_sqi.sqi.waveform_sqi import *
from vital_sqi.data.signal_io import ECG_reader
import os


class TestBandEnergySqi(object):
    file_name = os.path.abspath("tests/test_data/example.edf")
    out = ECG_reader(file_name, "edf")

    def test_on_valid(self):
        out = band_energy_sqi(
            signal=self.out.signals.iloc[:, 1],
            sampling_rate=self.out.sampling_rate,
            band=[0, 0.5],
        )
        assert isinstance(out, float)
        out = band_energy_sqi(
            signal=self.out.signals.iloc[:, 1],
            sampling_rate=self.out.sampling_rate,
            band=[0, 0.5],
        )
        assert isinstance(out, float)

    def test_on_band(self):
        with pytest.raises(AssertionError) as exc_info:
            out = band_energy_sqi(
                signal=self.out.signals.iloc[:, 1],
                sampling_rate=self.out.sampling_rate,
                band="[0, 0.5]",
            )
        assert exc_info.match("Invalid band values")
        with pytest.raises(AssertionError) as exc_info:
            out = band_energy_sqi(
                signal=self.out.signals.iloc[:, 1],
                sampling_rate=self.out.sampling_rate,
                band=[0.5, 0],
            )
        assert exc_info.match("Invalid band values")

    def test_on_sampling_rate(self):
        with pytest.raises(AssertionError) as exc_info:
            out = band_energy_sqi(
                signal=self.out.signals.iloc[:, 1], sampling_rate="", band=[0.5, 0]
            )
        assert exc_info.match("Expected a numeric sampling rate value.")


import numpy as np
import warnings


# ---------------------------------------------------------------------------
# lf_energy_sqi / qrs_energy_sqi / hf_energy_sqi / vhf_norm_power_sqi / qrs_a_sqi
# ---------------------------------------------------------------------------

@pytest.fixture
def ppg_signal():
    fs = 100
    t = np.linspace(0, 10, fs * 10)
    return np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.default_rng(0).normal(size=len(t))


class TestLfEnergySqi:
    def test_returns_float(self, ppg_signal):
        assert isinstance(lf_energy_sqi(ppg_signal, sampling_rate=100), float)

    def test_value_non_negative(self, ppg_signal):
        assert lf_energy_sqi(ppg_signal, sampling_rate=100) >= 0


class TestQrsEnergySqi:
    def test_returns_float(self, ppg_signal):
        assert isinstance(qrs_energy_sqi(ppg_signal, sampling_rate=100), float)

    def test_value_non_negative(self, ppg_signal):
        assert qrs_energy_sqi(ppg_signal, sampling_rate=100) >= 0


class TestHfEnergySqi:
    def test_above_nyquist_returns_nan_with_warning(self, ppg_signal):
        with pytest.warns(UserWarning, match="Nyquist"):
            result = hf_energy_sqi(ppg_signal, sampling_rate=100)
        assert np.isnan(result)

    def test_valid_band_returns_float(self, ppg_signal):
        result = hf_energy_sqi(ppg_signal, sampling_rate=100, band=[20, 49])
        assert isinstance(result, float)


class TestVhfNormPowerSqi:
    def test_above_nyquist_returns_nan_with_warning(self, ppg_signal):
        with pytest.warns(UserWarning, match="Nyquist"):
            result = vhf_norm_power_sqi(ppg_signal, sampling_rate=100)
        assert np.isnan(result)

    def test_valid_band_returns_float(self, ppg_signal):
        result = vhf_norm_power_sqi(ppg_signal, sampling_rate=100, band=[20, 45])
        assert isinstance(result, float)

    def test_zero_signal_returns_nan(self):
        result = vhf_norm_power_sqi(np.zeros(500), sampling_rate=100, band=[20, 45])
        assert np.isnan(result)


class TestQrsASqi:
    def test_returns_float_or_nan(self, ppg_signal):
        result = qrs_a_sqi(ppg_signal, sampling_rate=100)
        assert isinstance(result, (float, np.floating)) or np.isnan(result)
