Welcome to vital_sqi's documentation!
=====================================

.. image:: ./_static/imgs/logo.png
   :alt: vital_sqi logo
   :align: center

.. note::

   **vital_sqi** is a Python library for analysing physiological signals
   (ECG, PPG) and computing Signal Quality Indices (SQI). It bundles
   classical SQIs from the literature, a calibrated rule engine, a
   robust multi-regime classifier, and a calibration toolkit for deriving
   thresholds from synthetic noise sweeps.

Source code lives on `GitHub <https://github.com/Oucru-Innovations/vital-sqi>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage/installation
   usage/introduction
   usage/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage/pipeline
   usage/app
   usage/contributions
   usage/development

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   _examples/notebooks/Data_manipulation_ECG_PPG
   _examples/notebooks/SQI_pipeline

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   docstring/vital_sqi.sqi
   docstring/vital_sqi.pipeline
   docstring/vital_sqi.calibration
   docstring/vital_sqi.rule
   docstring/vital_sqi.preprocess
   docstring/vital_sqi.common
   docstring/vital_sqi.data
   docstring/vital_sqi.dataset

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
