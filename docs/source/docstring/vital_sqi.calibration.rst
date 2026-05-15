Calibration (``vital_sqi.calibration``)
=======================================

Automated derivation of SQI accept/reject thresholds. The calibration
workflow generates clean synthetic signals, injects parameterised noise,
computes SQIs over both pools, and writes a ``rule_dict.json`` and
``sqi_dict.json`` that can be loaded by the pipeline.

.. contents::
   :local:
   :depth: 1


Top-level runner
----------------

.. automodule:: vital_sqi.calibration.run_calibration
   :members:
   :show-inheritance:


Synthetic signal generation
---------------------------

.. automodule:: vital_sqi.calibration.signal_generator
   :members:
   :show-inheritance:


Noise injection
---------------

.. automodule:: vital_sqi.calibration.noise_injector
   :members:
   :show-inheritance:


Batch SQI computation
---------------------

.. automodule:: vital_sqi.calibration.sqi_runner
   :members:
   :show-inheritance:


Threshold estimator
-------------------

.. automodule:: vital_sqi.calibration.threshold_estimator
   :members:
   :show-inheritance:


Exporter
--------

.. automodule:: vital_sqi.calibration.exporter
   :members:
   :show-inheritance:
