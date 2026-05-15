"""
vital_sqi.rule
==============

Threshold-based and consensus-based segment classifiers.

Public surface
--------------
``Rule`` / ``RuleSet``
    Threshold-application objects: given a calibrated ``(lower, upper)``
    band per SQI, decide accept/reject per segment.

``classify_segments_robust`` / ``RobustResult``
    Three-regime classifier (clean / bimodal / heavy-noise) that does not
    require pre-calibrated thresholds.

``auto_threshold`` submodule
    Strategies for deriving the ``(lower, upper)`` band from an observed
    SQI distribution: :func:`~vital_sqi.rule.auto_threshold.quantile_band`
    (fixed p5/p95-style trim) and
    :func:`~vital_sqi.rule.auto_threshold.tuned_bands` (per-rule quantile
    that targets a joint accept rate).
"""

from vital_sqi.rule.ruleset_class import *
from vital_sqi.rule.rule_class import *
from vital_sqi.rule.robust_classifier import classify_segments_robust, RobustResult
from vital_sqi.rule import auto_threshold  # noqa: F401  - submodule re-export
