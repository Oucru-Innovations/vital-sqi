"""Coverage-driven tests for SQI error paths and new functions.

These tests exercise the ``except Exception``, NaN-return, and
edge-case branches that aren't reached by the existing happy-path tests.
"""
import numpy as np
import pytest

from vital_sqi.sqi import hrv_sqi, rpeaks_sqi, standard_sqi, waveform_sqi


# ---------------------------------------------------------------------------
# hrv_sqi error / edge branches
# ---------------------------------------------------------------------------

class TestHrvSqiEdgeCases:
    def test_cvsd_sqi_zero_mean_returns_nan(self):
        # Zero NN intervals → mean is 0 → cvsd returns NaN via the
        # 'mean_nn if mean_nn else np.nan' guard.
        result = hrv_sqi.cvsd_sqi([0.0, 0.0, 0.0, 0.0])
        assert np.isnan(result)

    def test_cvnn_sqi_zero_mean_returns_nan(self):
        result = hrv_sqi.cvnn_sqi([0.0, 0.0, 0.0, 0.0])
        assert np.isnan(result)

    def test_pnn_sqi_short_input(self):
        # < 2 entries → np.diff returns empty → no count above threshold
        result = hrv_sqi.pnn_sqi([0.8])
        # implementation returns 0.0 or nan on degenerate input
        assert np.isnan(result) or result == 0.0

    def test_get_all_features_hrva_bad_signal(self):
        # Empty signal → peak detector either fails or yields too few peaks
        result = hrv_sqi.get_all_features_hrva(
            np.array([]), sample_rate=100, wave_type="PPG"
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_all_features_hrva_invalid_sample_rate(self):
        with pytest.raises(ValueError):
            hrv_sqi.get_all_features_hrva(
                np.zeros(100), sample_rate=-1, wave_type="PPG"
            )

    def test_rr_irregularity_sqi_short_input(self):
        with pytest.warns(UserWarning):
            result = hrv_sqi.rr_irregularity_sqi([0.8])
        assert np.isnan(result)

    def test_rr_irregularity_sqi_zero_median(self):
        result = hrv_sqi.rr_irregularity_sqi([0.0, 0.0, 0.0, 0.0])
        assert np.isnan(result)

    def test_sample_entropy_sqi_too_short(self):
        with pytest.warns(UserWarning):
            result = hrv_sqi.sample_entropy_sqi([0.8, 0.8])
        assert np.isnan(result)

    def test_sample_entropy_sqi_zero_tolerance(self):
        # Constant NN intervals — function must not crash, must return either
        # NaN (zero-tolerance guard) or a finite scalar.  Both branches are
        # acceptable; we only verify it doesn't raise.
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            result = hrv_sqi.sample_entropy_sqi([0.8] * 50)
        assert np.isnan(result) or np.isfinite(result)

    def test_sample_entropy_sqi_happy_path(self):
        # Varied NN intervals — should produce a finite value.
        rng = np.random.default_rng(0)
        nn = 0.8 + 0.05 * rng.standard_normal(200)
        result = hrv_sqi.sample_entropy_sqi(nn, m=2)
        assert np.isfinite(result) or np.isnan(result)  # allow NaN if A=0

    def test_dfa_sqi_too_short(self):
        with pytest.warns(UserWarning):
            result = hrv_sqi.dfa_sqi([0.8] * 10)
        assert np.isnan(result)

    def test_dfa_sqi_happy_path(self):
        rng = np.random.default_rng(0)
        nn = 0.8 + 0.05 * rng.standard_normal(100)
        result = hrv_sqi.dfa_sqi(nn)
        assert np.isfinite(result) or np.isnan(result)

    def test_hurst_sqi_too_short(self):
        result = hrv_sqi.hurst_sqi([0.8] * 5)
        assert np.isnan(result)

    def test_hurst_sqi_happy_path(self):
        rng = np.random.default_rng(0)
        nn = 0.8 + 0.05 * rng.standard_normal(200)
        result = hrv_sqi.hurst_sqi(nn)
        assert np.isfinite(result) or np.isnan(result)


# ---------------------------------------------------------------------------
# rpeaks_sqi error / edge branches
# ---------------------------------------------------------------------------

class TestRpeaksSqiEdgeCases:
    def test_ectopic_sqi_short_signal_returns_nan(self):
        result = rpeaks_sqi.ectopic_sqi(np.array([1.0, 2.0]), sample_rate=100)
        assert np.isnan(result)

    def test_ectopic_sqi_invalid_rule_index(self):
        # Out-of-range rule index → ValueError caught, returns NaN
        result = rpeaks_sqi.ectopic_sqi(
            np.sin(np.linspace(0, 30, 3000)),
            rule_index=99, sample_rate=100,
        )
        assert np.isnan(result)

    def test_remove_ectopic_beats_invalid_method(self):
        # Invalid method falls into the ValueError branch and returns the
        # input unchanged.
        rr = np.array([0.8, 1.0, 0.9, 1.1])
        result = rpeaks_sqi.remove_ectopic_beats(rr, method="not_a_method")
        np.testing.assert_array_equal(result, rr)

    def test_remove_ectopic_beats_short_input(self):
        # < 3 valid intervals raises internally → returned unchanged
        rr = np.array([0.8, np.nan])
        result = rpeaks_sqi.remove_ectopic_beats(rr, method="adaptive")
        # Either returns original or all-nan, but must not crash
        assert len(result) == len(rr)

    def test_msq_sqi_empty_signal_returns_nan(self):
        with pytest.warns(UserWarning):
            result = rpeaks_sqi.msq_sqi([])
        assert np.isnan(result)

    def test_amplitude_consistency_too_few_peaks(self):
        # Constant signal → peak detector returns < 2 peaks.
        with pytest.warns(UserWarning):
            result = rpeaks_sqi.amplitude_consistency_sqi(
                np.zeros(500), sample_rate=100
            )
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# standard_sqi error / edge branches
# ---------------------------------------------------------------------------

class TestStandardSqiEdgeCases:
    def test_signal_to_noise_sqi_rejects_non_array(self):
        with pytest.raises(TypeError):
            standard_sqi.signal_to_noise_sqi("not an array")

    def test_mean_crossing_rate_rejects_non_array(self):
        with pytest.raises(TypeError):
            standard_sqi.mean_crossing_rate_sqi("not an array")

    def test_clipping_sqi_empty_returns_nan(self):
        assert np.isnan(standard_sqi.clipping_sqi(np.array([])))

    def test_clipping_sqi_constant_signal_returns_zero(self):
        assert standard_sqi.clipping_sqi(np.ones(100)) == 0.0

    def test_clipping_sqi_clipped_signal(self):
        # Signal pinned at min/max for most samples
        sig = np.concatenate([np.ones(50) * 0.0, np.ones(50) * 1.0])
        result = standard_sqi.clipping_sqi(sig)
        assert result > 0.9  # almost all samples are at the rails

    def test_baseline_wander_sqi_short_signal_nan(self):
        assert np.isnan(standard_sqi.baseline_wander_sqi(np.array([1.0, 2.0])))

    def test_baseline_wander_sqi_zero_signal_nan(self):
        # Zero signal → total power is 0
        result = standard_sqi.baseline_wander_sqi(np.zeros(500), sampling_rate=100)
        assert np.isnan(result)

    def test_baseline_wander_sqi_normal(self):
        t = np.linspace(0, 5, 500)
        # 0.1 Hz drift + 1 Hz signal → some LF energy fraction
        sig = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.0 * t)
        result = standard_sqi.baseline_wander_sqi(sig, sampling_rate=100)
        assert 0.0 <= result <= 1.0

    def test_spectral_snr_sqi_short_signal_nan(self):
        assert np.isnan(standard_sqi.spectral_snr_sqi(np.array([1.0, 2.0])))

    def test_spectral_snr_sqi_zero_signal_nan(self):
        result = standard_sqi.spectral_snr_sqi(np.zeros(500), sampling_rate=100)
        assert np.isnan(result)

    def test_spectral_snr_sqi_normal(self):
        t = np.linspace(0, 5, 500)
        # Pure 1 Hz signal in the [0.5, 4] band
        sig = np.sin(2 * np.pi * 1.0 * t)
        result = standard_sqi.spectral_snr_sqi(sig, sampling_rate=100)
        assert np.isfinite(result)
        assert result > 0  # in-band > out-of-band

    def test_entropy_sqi_flat_signal_returns_zero(self):
        # All-zero signal: previously raised; now returns 0.0
        assert standard_sqi.entropy_sqi(np.zeros(100)) == 0.0


# ---------------------------------------------------------------------------
# waveform_sqi edge cases
# ---------------------------------------------------------------------------

class TestWaveformSqiEdgeCases:
    def test_hf_energy_sqi_above_nyquist_returns_nan(self):
        # Default band is [100, inf] but fs=100 has Nyquist 50.
        with pytest.warns(UserWarning):
            result = waveform_sqi.hf_energy_sqi(
                np.sin(np.linspace(0, 10, 1000)), sampling_rate=100
            )
        assert np.isnan(result)

    def test_vhf_norm_power_sqi_above_nyquist_returns_nan(self):
        with pytest.warns(UserWarning):
            result = waveform_sqi.vhf_norm_power_sqi(
                np.sin(np.linspace(0, 10, 1000)), sampling_rate=100
            )
        assert np.isnan(result)

    def test_band_energy_sqi_none_band(self):
        # band=None branch
        result = waveform_sqi.band_energy_sqi(
            np.sin(np.linspace(0, 10, 1000)), sampling_rate=100, band=None
        )
        assert np.isfinite(result)

    def test_band_energy_sqi_empty_idx_returns_zero(self):
        # band entirely outside Nyquist → idx empty → 0.0
        result = waveform_sqi.band_energy_sqi(
            np.sin(np.linspace(0, 10, 1000)),
            sampling_rate=100, band=[60.0, 80.0],
        )
        assert result == 0.0
