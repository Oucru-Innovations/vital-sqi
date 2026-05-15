"""Unit tests for vital_sqi.calibration.noise_injector."""
import numpy as np
import pytest

from vital_sqi.calibration import noise_injector as ni


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def signal():
    # Sine wave with non-zero peak-to-peak.
    return np.sin(2 * np.pi * 1.0 * np.linspace(0, 10, 1000))


# ---------------------------------------------------------------------------
# Individual noise functions — verify shape preserved and output differs
# ---------------------------------------------------------------------------

class TestIndividualNoise:
    def test_gaussian_noise_shape_and_diff(self, signal, rng):
        out = ni.gaussian_noise(signal, 0.1, rng)
        assert out.shape == signal.shape
        assert not np.array_equal(out, signal)

    def test_gaussian_zero_amplitude_is_identity(self, signal, rng):
        out = ni.gaussian_noise(signal, 0.0, rng)
        np.testing.assert_allclose(out, signal)

    def test_baseline_wander(self, signal, rng):
        out = ni.baseline_wander(signal, 0.2, rng, fs=100)
        assert out.shape == signal.shape
        assert not np.array_equal(out, signal)

    def test_motion_artifact(self, signal, rng):
        out = ni.motion_artifact(signal, 0.3, rng, fs=100, n_bursts=2)
        assert out.shape == signal.shape

    def test_harmonic(self, signal, rng):
        out = ni.harmonic_interference(signal, 0.1, rng, fs=100)
        assert out.shape == signal.shape

    def test_powerline(self, signal, rng):
        out = ni.powerline_noise(signal, 0.05, rng, fs=100)
        assert out.shape == signal.shape

    def test_polynomial_trend(self, signal, rng):
        out = ni.polynomial_trend(signal, 0.2, rng)
        assert out.shape == signal.shape

    def test_impulse_noise(self, signal, rng):
        out = ni.impulse_noise(signal, 0.1, rng)
        assert out.shape == signal.shape

    def test_colored_noise_pink(self, signal, rng):
        out = ni.colored_noise(signal, 0.1, rng, exponent=1.0)
        assert out.shape == signal.shape

    def test_colored_noise_brown(self, signal, rng):
        out = ni.colored_noise(signal, 0.1, rng, exponent=2.0)
        assert out.shape == signal.shape

    def test_time_shift(self, signal, rng):
        out = ni.time_shift(signal, 0.05, rng, fs=100)
        assert out.shape == signal.shape

    def test_clock_drift(self, signal, rng):
        out = ni.clock_drift(signal, 0.02, rng, fs=100)
        assert out.shape == signal.shape

    def test_sample_dropout(self, signal, rng):
        out = ni.sample_dropout(signal, 0.05, rng, fs=100)
        assert out.shape == signal.shape


# ---------------------------------------------------------------------------
# _pp guard against flat signals
# ---------------------------------------------------------------------------

class TestPeakToPeakHelper:
    def test_flat_signal_returns_one(self):
        flat = np.zeros(100)
        assert ni._pp(flat) == 1.0

    def test_nonflat_signal_returns_range(self):
        s = np.array([1.0, 3.0, -2.0, 5.0])
        assert ni._pp(s) == 7.0


# ---------------------------------------------------------------------------
# inject_noise dispatch
# ---------------------------------------------------------------------------

class TestInjectNoise:
    @pytest.mark.parametrize("noise_type", [
        "gaussian", "baseline_wander", "motion", "harmonic", "powerline",
        "polynomial", "impulse", "pink", "brown",
        "time_shift", "clock_drift", "dropout",
        "combined_mild", "combined_severe",
    ])
    def test_each_dispatch_branch(self, signal, rng, noise_type):
        out = ni.inject_noise(signal, noise_type, 0.1, rng, fs=100)
        assert out.shape == signal.shape

    def test_unknown_noise_type_raises(self, signal, rng):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            ni.inject_noise(signal, "not_a_real_noise", 0.1, rng)


# ---------------------------------------------------------------------------
# NOISE_PROFILES sanity
# ---------------------------------------------------------------------------

class TestNoiseProfiles:
    def test_profiles_are_triples(self):
        for entry in ni.NOISE_PROFILES:
            assert len(entry) == 3

    def test_clean_profile_present(self):
        names = {n for n, _, _ in ni.NOISE_PROFILES}
        assert "clean" in names
        assert "clean" in ni.CLEAN_PROFILE_LABELS

    def test_all_noise_types_dispatchable(self, signal, rng):
        # Every noise_type referenced in NOISE_PROFILES must be invokable
        # via inject_noise without raising.
        for label, noise_type, amp in ni.NOISE_PROFILES:
            out = ni.inject_noise(signal, noise_type, amp, rng, fs=100)
            assert out.shape == signal.shape
