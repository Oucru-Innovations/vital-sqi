Rule engine (``vital_sqi.rule``)
================================

The rule engine wraps SQI threshold definitions into composable
``Rule`` and ``RuleSet`` objects, plus a robust auto-classifier that
detects three regimes (clean / bimodal / heavy-noise) without needing
hand-picked thresholds.

.. contents::
   :local:
   :depth: 1


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
