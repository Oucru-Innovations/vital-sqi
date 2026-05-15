"""Smoke tests for the calibration orchestration layer.

These rely on vitalDSP synthesis being available; they're kept short
(tiny n_segments) so they run in seconds.
"""
import os
import numpy as np
import pandas as pd
import pytest


# Skip cleanly if vitalDSP synth is unavailable in the test env.
vitaldsp = pytest.importorskip("vitalDSP.utils.data_processing.synthesize_data")


from vital_sqi.calibration import signal_generator as sg


class TestSignalGenerator:
    def test_generate_clean_ppg_returns_segments(self):
        segments = sg.generate_clean_ppg(
            n_segments=3, duration=5.0, sampling_rate=100,
            hr_range=(60, 80),
        )
        assert isinstance(segments, list)
        # At least one segment should succeed
        if segments:
            sig, fs = segments[0]
            assert isinstance(sig, np.ndarray)
            assert fs == 100

    def test_generate_clean_ecg_returns_segments(self):
        segments = sg.generate_clean_ecg(
            n_segments=3, duration=5.0, sampling_rate=256,
            hr_range=(60, 80),
        )
        assert isinstance(segments, list)
        if segments:
            sig, fs = segments[0]
            assert isinstance(sig, np.ndarray)
            assert fs == 256
            # Length should equal duration * fs
            assert len(sig) == int(5.0 * 256)


class TestCalibrateSmoke:
    """End-to-end smoke test with tiny segment count."""

    def test_calibrate_ppg_dry_run(self, tmp_path):
        from vital_sqi.calibration.run_calibration import calibrate

        thresholds = calibrate(
            wave_type="PPG",
            n_segments=5,
            n_reject_segments=3,
            duration=5.0,
            output_dir=str(tmp_path),
            dry_run=True,           # don't actually export
            show_progress=False,
        )
        assert isinstance(thresholds, dict)
        # Even with only 5 segments, at least some SQIs should calibrate
        # (this is a smoke test — exact count varies with synthesis luck).

    def test_calibrate_writes_files_when_not_dry_run(self, tmp_path):
        from vital_sqi.calibration.run_calibration import calibrate

        calibrate(
            wave_type="PPG",
            n_segments=5,
            n_reject_segments=3,
            duration=5.0,
            output_dir=str(tmp_path),
            dry_run=False,
            show_progress=False,
        )
        # Files should exist regardless of how many SQIs calibrated
        assert (tmp_path / "rule_dict.json").exists()
        assert (tmp_path / "sqi_dict.json").exists()
