Rule engine (``vital_sqi.rule``)
================================

Three classifier families live here, plus a small math module:

* :class:`~vital_sqi.rule.Rule` / :class:`~vital_sqi.rule.RuleSet` —
  apply pre-computed accept-band thresholds, segment by segment.  Used
  by :func:`vital_sqi.pipeline.pipeline_functions.classify_segments`.
* :mod:`vital_sqi.rule.auto_threshold` — strategies for *deriving*
  those thresholds from the observed SQI distribution.  Both the
  programmatic pipeline and the web app's Inspect view share these
  helpers.
* :func:`~vital_sqi.rule.classify_segments_robust` — a separate
  non-rule-based three-regime classifier (clean / bimodal /
  heavy-noise) that needs no pre-calibrated thresholds at all.

.. contents::
   :local:
   :depth: 1


Auto-threshold strategies
-------------------------

.. automodule:: vital_sqi.rule.auto_threshold
   :members:
   :show-inheritance:


Single-SQI rule
---------------

.. automodule:: vital_sqi.rule.rule_class
   :members:
   :show-inheritance:


Composite rule set
------------------

.. automodule:: vital_sqi.rule.ruleset_class
   :members:
   :show-inheritance:


Robust auto-classifier
----------------------

.. automodule:: vital_sqi.rule.robust_classifier
   :members:
   :show-inheritance:
