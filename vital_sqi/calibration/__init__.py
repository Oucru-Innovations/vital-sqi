"""
vital_sqi.calibration
=====================
Empirical calibration of SQI thresholds using synthetic physiological signals.

Workflow
--------
1. :mod:`signal_generator`   — generate clean ECG/PPG signals via vitalDSP
2. :mod:`noise_injector`     — degrade signals with multiple noise models
3. :mod:`sqi_runner`         — compute all SQIs over batches of segments
4. :mod:`threshold_estimator`— derive p5/p95 accept bounds from distributions
5. :mod:`exporter`           — write calibrated rule_dict.json / sqi_dict.json

Quick start
-----------
>>> from vital_sqi.calibration import run_calibration
>>> run_calibration.calibrate(wave_type="PPG", output_dir="vital_sqi/resource")
"""

from vital_sqi.calibration.signal_generator import generate_clean_ppg, generate_clean_ecg
from vital_sqi.calibration.noise_injector import inject_noise, NOISE_PROFILES
from vital_sqi.calibration.sqi_runner import compute_sqi_distributions
from vital_sqi.calibration.threshold_estimator import estimate_thresholds
from vital_sqi.calibration.exporter import export_rule_dict, export_sqi_dict

__all__ = [
    "generate_clean_ppg",
    "generate_clean_ecg",
    "inject_noise",
    "NOISE_PROFILES",
    "compute_sqi_distributions",
    "estimate_thresholds",
    "export_rule_dict",
    "export_sqi_dict",
]
