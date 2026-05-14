SQI Pipeline and Calibration
============================

This page describes in full detail:

1. How raw physiological signals are segmented and passed through the SQI extraction pipeline.
2. How each SQI is computed, routed, and packaged into a result DataFrame.
3. How the rule-based classifier works and makes accept/reject decisions.
4. How the calibration system derives thresholds empirically from synthetic signals.

All source lives in ``vital_sqi/pipeline/pipeline_functions.py``,
``vital_sqi/rule/``, and ``vital_sqi/calibration/``.

---

.. contents:: Table of Contents
   :local:
   :depth: 2

---

Part 1 — SQI Extraction Pipeline
---------------------------------

Overview
~~~~~~~~

The extraction pipeline transforms a list of signal segments into a
structured DataFrame of quality metrics.  The call chain is::

    extract_sqi(segments, milestones, sqi_dict_filename)
        └── for each segment:
                extract_segment_sqi(segment, sqi_list, sqi_names, sqi_arg_list, wave_type)
                    ├── [once] get_nn(signal)          # nn_intervals cache
                    └── for each SQI:
                            get_sqi(sqi_func, sqi_name, segment, ...)
                                ├── [if per_beat] per_beat_sqi(...)
                                └── [else]         sqi_func(signal, **kwargs)
                                        └── get_sqi_dict(result, sqi_name)

Step 1 — ``extract_sqi`` (entry point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    df = extract_sqi(segments, milestones, "vital_sqi/resource/sqi_dict.json", wave_type="PPG")

**Inputs**

``segments``
    A list of DataFrames produced by
    :func:`~vital_sqi.preprocess.segment_split.split_segment`.
    Each DataFrame has two columns: ``time`` (timestamps) and ``signal``
    (raw waveform values).

``milestones``
    A two-column DataFrame with ``start_idx`` and ``end_idx`` (sample
    positions in the original recording) for each segment.

``sqi_dict_filename``
    Path to a JSON configuration file.  Each top-level key is a
    user-chosen label; each value is a dict with:

    - ``"sqi"`` — the function name registered in ``sqi_mapping``
      (e.g. ``"kurtosis_sqi"``).
    - ``"args"`` — keyword arguments forwarded to the SQI function.

    Example::

        {
          "perfusion":  {"sqi": "perfusion_sqi",  "args": {}},
          "kurtosis":   {"sqi": "kurtosis_sqi",   "args": {"axis": 0}},
          "poincare":   {"sqi": "poincare_sqi",   "args": {}}
        }

**What it does**

1. Loads ``sqi_dict_filename`` and resolves each ``"sqi"`` key to a
   callable from ``sqi_mapping``.
2. Iterates over every segment (with a ``tqdm`` progress bar).
3. Calls :func:`extract_segment_sqi` for each segment, collecting a
   ``pd.Series`` of SQI values.
4. Assembles all series into a ``pd.DataFrame`` (one row per segment).
5. Appends ``start_idx`` / ``end_idx`` columns from *milestones*.

**Output**

A DataFrame with one row per segment.  Columns are the SQI labels from
the config file.  Multi-value SQIs (e.g. ``poincare_sqi`` which returns
``sd1``, ``sd2``, ``area``, ``ratio``) expand into sub-columns.
Per-beat SQIs that return lists expand into ``_mean_sqi``,
``_median_sqi``, and ``_std_sqi`` columns.

Step 2 — ``extract_segment_sqi`` (per-segment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Called once per segment.  Responsible for:

- Extracting the raw signal array from the segment DataFrame once (avoiding
  repeated ``iloc`` calls inside the loop).
- Computing ``nn_intervals`` **once** and caching the result for reuse by
  all HRV SQIs (see :ref:`nn-caching`).
- Iterating over every (function, name) pair, calling
  :func:`get_sqi`, and collecting results.

Special handling:

- ``perfusion_sqi`` requires two signals (raw + filtered).  When it
  appears in the list, :func:`extract_segment_sqi` passes
  ``y=signal_values`` directly and skips the standard dispatch path.
- Any SQI that raises an exception is caught; a warning is emitted and
  that SQI column receives ``NaN`` for the segment.

.. _nn-caching:

nn_intervals Caching
^^^^^^^^^^^^^^^^^^^^^

HRV SQIs (``sdnn_sqi``, ``rmssd_sqi``, ``poincare_sqi``, etc.) take
``nn_intervals`` as their first positional argument rather than the raw
signal.  NN intervals are derived from peak detection, which involves a
Kalman filter and is expensive (~14 ms per segment at 100 Hz).

Without caching, ``get_nn`` would be called once per HRV SQI (12–18
calls per segment).  The optimisation: ``extract_segment_sqi`` inspects
the first argument name of each SQI function using
``inspect.getfullargspec`` (cached in ``_argspec_cache`` at module level
so introspection runs only once per function lifetime) and routes HRV
SQIs through a single shared ``_nn_cache``:

.. code-block:: python

    _nn_cache = None
    for sqi_func, sqi_name in zip(sqi_list, sqi_names):
        first_arg = _argspec_cache[sqi_func][0]
        if first_arg == "nn_intervals":
            if _nn_cache is None:
                _nn_cache = get_nn(signal_values)   # computed once
            args["_nn_intervals"] = _nn_cache       # injected into get_sqi

This reduces peak-detection calls from N (one per HRV SQI) to 1 per
segment, giving a ~3.6x speedup.

Step 3 — ``get_sqi`` (per-SQI dispatch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles two execution modes:

**Whole-segment mode** (``per_beat=False``, default)
    The SQI function is called on the full signal array (or on
    ``nn_intervals`` for HRV SQIs).  The result — scalar, dict, or list —
    is packaged by :func:`get_sqi_dict`.

**Per-beat mode** (``per_beat=True``)
    :func:`extract_segment_sqi` runs peak detection **once** per segment
    on the first per-beat SQI encountered and caches the resulting
    ``peak_list`` / ``trough_list``.  All subsequent per-beat SQIs in the
    same segment reuse the cached peaks without re-running the detector.
    :func:`get_sqi` accepts the cached peaks via the ``_peak_list`` /
    ``_trough_list`` private keyword arguments; :func:`per_beat_sqi` then
    slices the signal into individual beats and calls the SQI function on
    each.  If ``use_mean_beat=True`` (default), all beats are resampled to
    ``mean_resample_size`` samples, averaged, and the SQI is computed once
    on the mean beat (one value per segment).

``wave_type`` is forwarded to every SQI function whose signature includes
a ``wave_type`` parameter.  ``sampling_rate`` / ``sample_rate`` arguments
in ``sqi_arg_list`` are automatically updated to the actual signal
sampling rate before calling (via ``_patch_fs`` in the calibration
runner).

Step 4 — ``get_sqi_dict`` (result packaging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts whatever the SQI function returns into a ``{column: value}``
dict that maps cleanly to DataFrame columns:

=========================  ===================================================
Raw return type            Column(s) produced
=========================  ===================================================
``dict``                   Returned unchanged (e.g. ``poincare_sqi`` →
                           ``{"sd1": …, "sd2": …, "area": …, "ratio": …}``)
scalar (float/int)         ``{sqi_name: scalar}``
1-element list/ndarray     ``{sqi_name: value}``
multi-element list/ndarray ``{sqi_name_mean_sqi, sqi_name_median_sqi,``
                           ``sqi_name_std_sqi}``
=========================  ===================================================

---

Part 2 — SQI Catalogue
-----------------------

The following table lists every SQI registered in ``sqi_mapping``.
SQIs whose first argument is ``nn_intervals`` are automatically routed
through the cached NN-interval path.  All others receive the raw signal.

.. list-table:: Registered SQIs
   :header-rows: 1
   :widths: 30 15 55

   * - Function name
     - Input type
     - What it measures
   * - ``perfusion_sqi``
     - raw + filtered
     - (max−min of filtered) / |mean of raw| × 100
   * - ``kurtosis_sqi``
     - signal
     - Tail heaviness of the amplitude distribution
   * - ``skewness_sqi``
     - signal
     - Asymmetry of the amplitude distribution
   * - ``entropy_sqi``
     - signal
     - Shannon entropy of the normalised amplitude histogram
   * - ``signal_to_noise_sqi``
     - signal
     - |mean| / std — higher is cleaner
   * - ``zero_crossings_rate_sqi``
     - signal
     - Rate of sign changes — high value indicates noise
   * - ``mean_crossing_rate_sqi``
     - signal
     - Rate at which signal crosses its own mean
   * - ``clipping_sqi``
     - signal
     - Fraction of samples at the amplitude rail (saturation)
   * - ``baseline_wander_sqi``
     - signal
     - Low-frequency (<0.5 Hz) energy fraction — drift indicator
   * - ``spectral_snr_sqi``
     - signal
     - In-band vs out-of-band power ratio (dB)
   * - ``ectopic_sqi``
     - signal
     - Ratio of ectopic/outlier RR intervals
   * - ``correlogram_sqi``
     - signal
     - Mean of top ACF peaks — periodicity score
   * - ``msq_sqi``
     - signal
     - Peak-detector agreement ratio between two algorithms
   * - ``amplitude_consistency_sqi``
     - signal
     - Beat-to-beat peak amplitude coefficient of variation
   * - ``band_energy_sqi``
     - signal
     - Peak STFT time-marginal in a frequency band
   * - ``lfe_sqi``
     - signal
     - Low-frequency (0–0.5 Hz) STFT energy
   * - ``qrse_sqi``
     - signal
     - QRS-band (5–25 Hz) STFT energy
   * - ``hfe_sqi``
     - signal
     - High-frequency (100+ Hz) STFT energy
   * - ``vhfp_sqi``
     - signal
     - Very-high-frequency normalised power (>150 Hz)
   * - ``qrsa_sqi``
     - signal
     - Median QRS amplitude via WaveformMorphology
   * - ``dtw_sqi``
     - signal
     - DTW distance to a reference template (4 types)
   * - ``sdnn_sqi``
     - nn_intervals
     - Sample std of NN intervals
   * - ``sdsd_sqi``
     - nn_intervals
     - Sample std of successive NN differences
   * - ``rmssd_sqi``
     - nn_intervals
     - Root mean square of successive differences
   * - ``cvsd_sqi``
     - nn_intervals
     - RMSSD / mean NN (normalised short-term variability)
   * - ``cvnn_sqi``
     - nn_intervals
     - SDNN / mean NN (normalised long-term variability)
   * - ``mean_nn_sqi``
     - nn_intervals
     - Mean of NN intervals
   * - ``median_nn_sqi``
     - nn_intervals
     - Median of NN intervals
   * - ``pnn_sqi``
     - nn_intervals
     - % of successive differences strictly > threshold (pNN50)
   * - ``hr_mean_sqi``
     - nn_intervals
     - Mean heart rate in BPM
   * - ``hr_median_sqi``
     - nn_intervals
     - Median heart rate in BPM
   * - ``hr_min_sqi``
     - nn_intervals
     - Minimum instantaneous heart rate
   * - ``hr_max_sqi``
     - nn_intervals
     - Maximum instantaneous heart rate
   * - ``hr_std_sqi``
     - nn_intervals
     - Sample std of instantaneous heart rate
   * - ``hr_range_sqi``
     - nn_intervals
     - Fraction of HR values outside [range_min, range_max]
   * - ``peak_frequency_sqi``
     - nn_intervals
     - Spectral peak frequency in the LF band (0.04–0.15 Hz)
   * - ``absolute_power_sqi``
     - nn_intervals
     - Absolute power in the LF band
   * - ``log_power_sqi``
     - nn_intervals
     - Log power in the LF band
   * - ``relative_power_sqi``
     - nn_intervals
     - LF band power / total power
   * - ``normalized_power_sqi``
     - nn_intervals
     - LF band power / (total power − VLF power), per HRV Task Force 1996
   * - ``lf_hf_ratio_sqi``
     - nn_intervals
     - LF/HF power ratio (sympatho-vagal balance)
   * - ``poincare_sqi``
     - nn_intervals
     - SD1, SD2, ellipse area, SD1/SD2 ratio
   * - ``rr_irregularity_sqi``
     - nn_intervals
     - MAD of successive RR diffs / median RR
   * - ``sample_entropy_sqi``
     - nn_intervals
     - Sample entropy — complexity / regularity of RR series
   * - ``dfa_sqi``
     - nn_intervals
     - DFA alpha1 — short-range fractal scaling exponent
   * - ``hurst_sqi``
     - nn_intervals
     - Hurst exponent — long-range correlation (R/S analysis)

---

Part 3 — Rule-Based Classification
------------------------------------

Overview
~~~~~~~~

After SQI extraction, every segment receives an ``"accept"`` or
``"reject"`` decision by running a ``RuleSet`` over its SQI row.  The
workflow::

    classify_segments(sqis, rule_dict_filename, ruleset_order)
        ├── load rule_dict.json
        ├── [auto_mode] adjust thresholds to observed p5/p95
        ├── build Rule objects → RuleSet
        └── for each segment row:
                RuleSet.execute(row_df) → "accept" | "reject"

Rule Format
~~~~~~~~~~~

Rules are stored in ``rule_dict.json``.  Each entry has the structure:

.. code-block:: json

    {
      "kurtosis_sqi": {
        "name": "kurtosis_sqi",
        "def": [
          {"op": ">",  "value": "0.5",  "label": "accept"},
          {"op": "<=", "value": "0.5",  "label": "reject"},
          {"op": ">=", "value": "8.0",  "label": "reject"},
          {"op": "<",  "value": "8.0",  "label": "accept"}
        ],
        "desc": "Calibrated from synthetic PPG signals. Accept: p5=0.5 to p95=8.0.",
        "ref":  "vital_sqi.calibration"
      }
    }

The four-element ``"def"`` array encodes a half-open accept interval
``(lower, upper)``:

- Condition 1&2: value > lower → accept; value <= lower → reject
- Condition 3&4: value >= upper → reject; value < upper → accept

The ``Rule`` class parses this into a sorted ``boundaries`` array and a
``labels`` array, then uses ``bisect`` for O(log n) lookup.

The ``Rule`` class
~~~~~~~~~~~~~~~~~~

:class:`~vital_sqi.rule.Rule` represents a single threshold rule for one
SQI.

- ``load_def(path)`` — loads rule from JSON file by name.
- ``update_def(op_list, value_list, label_list)`` — programmatically sets
  new thresholds.
- ``apply_rule(x)`` — applies ``bisect_left`` on the sorted boundaries
  array to find the interval containing ``x``, returns its label.
- ``save_def(path, overwrite=False)`` — writes or merges the rule to JSON.

The ``RuleSet`` class
~~~~~~~~~~~~~~~~~~~~~

:class:`~vital_sqi.rule.RuleSet` manages an ordered dict of ``Rule``
objects and executes them sequentially.

``execute(value_df)``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    for order, rule in sorted(self.rules.items()):
        value = value_df.iloc[0][rule.name]
        decision = rule.apply_rule(value)
        if decision == "reject":
            return "reject"
    return "accept"

**This is a linear early-exit scan, not recursion.**  Rules are checked
in ascending integer order.  The moment any rule returns ``"reject"``,
the loop short-circuits and returns immediately without checking the
remaining rules.  Only if every rule returns ``"accept"`` does the
function return ``"accept"``.

This design is:

- **Optimal** — O(R) where R is the number of rules; cannot be done
  faster than linear since every rule must be checked in the worst case.
- **Ordered** — rule priority is controlled by the integer key; lower key
  = checked first.
- **Short-circuiting** — reject-heavy rules placed early (low key)
  minimize average rule evaluations per segment.

Auto-Mode Threshold Adjustment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``auto_mode=True`` (the default), ``classify_segments`` replaces
the stored thresholds with the empirical p5/p95 quantiles of the
observed SQI distribution across all segments *before* running the
classifier.  This makes the pipeline self-adapting: thresholds shift
to match the recording's actual signal quality distribution rather than
relying solely on the calibrated bounds from synthetic data.

Set ``auto_mode=False`` to use the calibrated thresholds from
``rule_dict.json`` exactly as stored.

``ruleset_order`` and ``get_decision_segments``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ruleset_order`` is a dict mapping integer priority keys to rule names::

    ruleset_order = {1: "kurtosis_sqi", 2: "perfusion_sqi", 3: "sdnn_sqi"}

Only the SQIs listed here participate in classification.  All other SQI
columns in the DataFrame are ignored.

After classification, ``get_decision_segments(segments, decision,
reject_decision)`` separates segments into accepted and rejected lists
by combining the rule-set decision with any pre-existing rejection
flags.

---

Part 4 — Peak Detection Algorithms
------------------------------------

Overview
~~~~~~~~

Peak detection is used by:

- All ``per_beat=True`` SQIs (DTW, skewness, kurtosis on individual beats).
- HRV SQIs — ``get_nn()`` calls the PPG/ECG detector to derive RR intervals.
- ``ectopic_sqi``, ``correlogram_sqi``, ``msq_sqi``, ``amplitude_consistency_sqi``.

The :class:`~vital_sqi.common.rpeak_detection.PeakDetector` class exposes two
independent method families: ``ppg_detector`` (9 algorithms) and
``ecg_detector`` (4 algorithms).

PPG Detectors
~~~~~~~~~~~~~

Select via the ``peak_detector`` argument in ``sqi_dict.json`` or via the
``detector_type`` constant imported from
``vital_sqi.common.rpeak_detection``.

.. list-table:: PPG detection algorithms
   :header-rows: 1
   :widths: 10 25 65

   * - Constant
     - Value
     - Description
   * - ``DEFAULT``
     - 6
     - vitalDSP ``WaveformMorphology.systolic_peaks`` — height + distance
       threshold on the band-passed signal.  **Recommended default.**
   * - ``ADAPTIVE_THRESHOLD``
     - 1
     - Threshold adapts to the local signal amplitude (running max).
       Suitable for recordings with slow baseline drift.
   * - ``COUNT_ORIG_METHOD``
     - 2
     - Local maxima above a ``0.75``-quantile × 0.2 threshold.
       Fast but sensitive to non-zero DC baselines.
   * - ``CLUSTERER_METHOD``
     - 3
     - KMeans (k=2) separates peaks from non-peaks.
       Non-deterministic; set ``random_state`` for reproducibility.
   * - ``SLOPE_SUM_METHOD``
     - 4
     - Zong 2003 slope-sum energy with onset back-search.
       Robust on noisy PPG; requires ``fs``-scaled windows.
   * - ``MOVING_AVERAGE_METHOD``
     - 5
     - Elgendi two-moving-average (MA\ :sub:`peak` > MA\ :sub:`beat`).
       Returns raw-signal peaks within each detected "block of interest".
   * - ``BILLAUER_METHOD``
     - 7
     - Billauer delta-based peak/trough tracker.
       Records ``mxpos`` (true peak sample) not the detection crossing.
   * - ``AMPD_METHOD``
     - 8
     - Automatic Multiscale Peak Detection (Scholkmann 2012).
       Parameter-free; works well on quasi-periodic PPG at 50–250 Hz.
   * - ``LOCAL_MAX_IBI``
     - 9
     - Local-maxima with inter-beat-interval gating.
       Filters candidate peaks by physiological IBI range (300–2000 ms).

ECG Detectors
~~~~~~~~~~~~~

Used by ``ecg_detector(s, detector_type=ECG_DEFAULT)``.  All algorithms
detect R-peaks; Q/S/P/T morphology extraction always uses vitalDSP
``WaveformMorphology``, anchored to the chosen R-peak indices.

.. list-table:: ECG detection algorithms
   :header-rows: 1
   :widths: 15 8 77

   * - Constant
     - Value
     - Description
   * - ``ECG_DEFAULT``
     - 10
     - vitalDSP ``WaveformMorphology`` derivative-square-MWI with
       height threshold.  Provides R + Q/S/P/T in a single call.
       **Recommended default.**
   * - ``PAN_TOMPKINS``
     - 11
     - Classic Pan-Tompkins 1985.  Bandpass 5–15 Hz → derivative →
       squaring → 150 ms MWI → adaptive dual threshold with 2 s
       learning period.  Back-projects within ±60 ms for true R-peak.
   * - ``HAMILTON``
     - 12
     - Hamilton-Tompkins simplified 2002.  Bandpass 8–16 Hz → first
       difference → 80 ms moving average → mean+0.5σ threshold.
       Single-pass, suitable for real-time use.
   * - ``ENGZEE``
     - 13
     - Engzee-Zeelenberg 1979.  High-pass filtered derivative with a
       dynamic threshold that adapts after each detected beat.

Usage example::

    from vital_sqi.common.rpeak_detection import (
        PeakDetector, ECG_DEFAULT, PAN_TOMPKINS, HAMILTON, ENGZEE,
        DEFAULT, AMPD_METHOD
    )

    ppg_detector = PeakDetector(wave_type="PPG", fs=100)
    peaks, troughs = ppg_detector.ppg_detector(ppg_signal)          # DEFAULT
    peaks, troughs = ppg_detector.ppg_detector(ppg_signal, AMPD_METHOD)

    ecg_detector = PeakDetector(wave_type="ECG", fs=256)
    r, q, s, p, t = ecg_detector.ecg_detector(ecg_signal)           # ECG_DEFAULT
    r, q, s, p, t = ecg_detector.ecg_detector(ecg_signal, PAN_TOMPKINS)

---

Part 5 — Calibration System
----------------------------

Purpose
~~~~~~~

The calibration system derives ``lower`` and ``upper`` threshold bounds
for every SQI empirically, using large pools of synthetic clean and
noisy signals.  Results are written to:

- ``vital_sqi/resource/rule_dict.json`` — threshold rules for the
  classifier.
- ``vital_sqi/resource/sqi_dict.json`` — SQI function registry (only
  successfully calibrated SQIs are included).

Architecture
~~~~~~~~~~~~

::

    run_calibration.calibrate()
        ├── 1. generate_clean_ppg / generate_clean_ecg   (signal_generator.py)
        ├── 2. inject_noise × NOISE_PROFILES              (noise_injector.py)
        │       ├── accept pool: clean + very-mild-noise segments
        │       └── reject pool: moderate-to-severe noise segments
        ├── 3. compute_sqi_distributions(accept + reject) (sqi_runner.py)
        ├── 4. estimate_thresholds(accept_df, reject_df)  (threshold_estimator.py)
        └── 5. export_rule_dict + export_sqi_dict         (exporter.py)

Step 1 — Signal Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vital_sqi.calibration.signal_generator.generate_clean_ppg` and
:func:`~vital_sqi.calibration.signal_generator.generate_clean_ecg`
produce ``n_segments`` clean synthetic waveforms using ``vitalDSP``'s
physiological signal generators.

- PPG: sampling rate 100 Hz, default segment duration 30 s.
- ECG: sampling rate 256 Hz, default segment duration 30 s.
- ``noise_floor=0.0`` — no noise added at generation time.
- Each segment is a ``(signal_array, fs)`` tuple.

Step 2 — Noise Injection
~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vital_sqi.calibration.noise_injector.inject_noise` applies one
of 11 noise types to a clean segment:

.. list-table:: Noise types
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``gaussian``
     - Additive white Gaussian noise scaled to ``amplitude × peak-to-peak``
   * - ``baseline_wander``
     - Sinusoidal drift at 0.05–0.5 Hz (respiration-like)
   * - ``motion``
     - 3 burst-style high-amplitude artifacts at random positions
   * - ``harmonic``
     - Sum of sine/cosine harmonics at a random base frequency 1–15 Hz
   * - ``powerline``
     - 50 Hz or 60 Hz sinusoidal interference
   * - ``polynomial``
     - Degree 2–4 polynomial trend (temperature / electrode drift)
   * - ``impulse``
     - Random isolated spikes at ~2% of samples
   * - ``pink``
     - 1/f colored noise via FFT shaping
   * - ``brown``
     - 1/f² colored noise (Brownian motion-like)
   * - ``time_shift``
     - Circular roll by a random number of samples (sync jitter)
   * - ``clock_drift``
     - Resample ±amplitude% then trim/pad (wrong fs label simulation)
   * - ``dropout``
     - Replace amplitude% of samples with their predecessor (packet loss)
   * - ``combined_mild``
     - Gaussian + baseline wander + polynomial (mild realistic composite)
   * - ``combined_severe``
     - Gaussian + harmonic + motion + impulse (severe realistic composite)

``NOISE_PROFILES`` defines 33 ``(label, noise_type, amplitude)`` triples.

The **accept pool** includes:

- All ``n_segments`` pure clean segments.
- All segments from profiles labelled ``clean`` or ``gaussian_very_mild``
  (amplitude 0.02 — barely perceptible noise).

The **reject pool** includes segments from all other profiles (amplitude
0.05–0.80, representing clinically poor signal quality).  The reject pool
uses only ``n_reject_segments`` (default 50) per profile for speed — it
is used for **diagnostics only** and does not influence the p5/p95 bounds.

Step 3 — SQI Distribution Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vital_sqi.calibration.sqi_runner.compute_sqi_distributions`
runs all 47 SQIs over every segment in both pools:

.. code-block:: python

    accept_df = compute_sqi_distributions(accept_segments, wave_type="PPG")
    reject_df  = compute_sqi_distributions(reject_segments, wave_type="PPG")

Each call returns a DataFrame with one row per segment and one column
per SQI output.  ``poincare_sqi`` expands to 4 sub-columns (sd1, sd2,
area, ratio).  Failed SQIs are NaN for that segment.

``_patch_fs`` automatically updates any ``sample_rate`` or
``sampling_rate`` keyword argument to the actual segment sampling rate.

Step 4 — Threshold Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vital_sqi.calibration.threshold_estimator.estimate_thresholds`
processes each SQI column independently:

.. code-block:: text

    for each column in accept_df:
        accept_vals = drop NaN and Inf
        if n_valid < 5:
            skip (mark calibrated=False)
        lower = percentile(accept_vals, lower_pct)   # default p5
        upper = percentile(accept_vals, upper_pct)   # default p95
        if upper - lower < 1e-6:
            widen band by epsilon (constant signal guard)
        if reject distribution overlaps accept band:
            record note (thresholds are NOT changed)
        mark calibrated=True

The result for each column is a :class:`~vital_sqi.calibration.threshold_estimator.SQIThreshold`
dataclass containing:

=========================================  ====================================
Field                                      Meaning
=========================================  ====================================
``lower``, ``upper``                       Accept region bounds
``accept_median``, ``accept_std``          Statistics of the clean distribution
``reject_median``                          Median of the noisy distribution
``n_accept``, ``n_reject``                 Sample counts used
``calibrated``                             False if SQI could not be calibrated
``note``                                   Human-readable diagnostic note
=========================================  ====================================

SQIs that cannot be calibrated (always NaN on synthetic data):

- ``interpolation_sqi`` — unimplemented stub, always 0.
- ``vhfp_sqi`` — band (>150 Hz) is above the Nyquist limit for 100/256 Hz signals.
- ``sample_entropy_sqi``, ``dfa_sqi``, ``hurst_sqi`` — require natural
  HRV variability; synthetic signals have constant RR intervals (std=0),
  making entropy/DFA/Hurst undefined.  These SQIs are fully functional on
  real recordings.

Step 5 — Export
~~~~~~~~~~~~~~~~

**rule_dict.json** — written by
:func:`~vital_sqi.calibration.exporter.export_rule_dict`.

Each calibrated SQI produces a four-element ``"def"`` array encoding the
accept interval ``(lower, upper)``::

    "kurtosis_sqi": {
      "name": "kurtosis_sqi",
      "def": [
        {"op": ">",  "value": "0.497", "label": "accept"},
        {"op": "<=", "value": "0.497", "label": "reject"},
        {"op": ">=", "value": "8.21",  "label": "reject"},
        {"op": "<",  "value": "8.21",  "label": "accept"}
      ],
      "desc": "Calibrated from synthetic PPG signals. Accept: p5=0.497 to p95=8.21."
    }

Existing entries for SQIs that were **not** calibrated in the current
run are **preserved unchanged** (merge, not overwrite).  This protects
manually curated thresholds.

**sqi_dict.json** — written by
:func:`~vital_sqi.calibration.exporter.export_sqi_dict`.

Only SQIs that were successfully calibrated AND appear in
``_SQI_ARG_TEMPLATES`` are emitted.  The output is a ready-to-use
configuration file for :func:`~vital_sqi.pipeline.pipeline_functions.extract_sqi`.

Running Calibration
~~~~~~~~~~~~~~~~~~~~

From the repo root::

    # PPG calibration (default: 200 accept segments, 50 reject per profile)
    python -m vital_sqi.calibration.run_calibration --wave_type PPG

    # ECG calibration
    python -m vital_sqi.calibration.run_calibration --wave_type ECG

    # Dry run — compute thresholds but do not write files
    python -m vital_sqi.calibration.run_calibration --wave_type PPG --dry_run

    # All options
    python -m vital_sqi.calibration.run_calibration \
        --wave_type PPG \
        --n_segments 500 \
        --n_reject_segments 100 \
        --duration 30 \
        --lower_pct 5 \
        --upper_pct 95 \
        --output_dir path/to/output \
        --seed 42

Or from Python:

.. code-block:: python

    from vital_sqi.calibration.run_calibration import calibrate

    thresholds = calibrate(
        wave_type="PPG",
        n_segments=200,
        n_reject_segments=50,
        duration=30.0,
        lower_pct=5.0,
        upper_pct=95.0,
        dry_run=False,
        seed=42,
    )

The returned ``thresholds`` dict maps SQI column names to
:class:`~vital_sqi.calibration.threshold_estimator.SQIThreshold` objects
and can be inspected with
:func:`~vital_sqi.calibration.threshold_estimator.thresholds_to_dataframe`.

---

Part 6 — Complete End-to-End Example
--------------------------------------

.. code-block:: python

    import vital_sqi.data.signal_io as sio
    from vital_sqi.preprocess.segment_split import split_segment
    from vital_sqi.pipeline.pipeline_functions import extract_sqi, classify_segments, get_decision_segments

    # 1. Load signal
    df = sio.read_ppg("recording.csv", sampling_rate=100)

    # 2. Segment into 30-second windows
    segments, milestones = split_segment(df, duration=30, fs=100)

    # 3. Extract SQIs
    sqi_df = extract_sqi(
        segments, milestones,
        "vital_sqi/resource/sqi_dict.json",
        wave_type="PPG",
    )

    # 4. Classify — each segment gets 'accept' or 'reject'
    ruleset_order = {1: "kurtosis_sqi", 2: "perfusion_sqi", 3: "correlogram_sqi"}
    ruleset, sqis_with_decisions = classify_segments(
        [sqi_df.iloc[[i]] for i in range(len(sqi_df))],
        rule_dict_filename="vital_sqi/resource/rule_dict.json",
        ruleset_order=ruleset_order,
        auto_mode=True,
    )

    # 5. Separate accepted / rejected segments
    decisions    = [df["decision"].iloc[0] for df in sqis_with_decisions]
    pre_reject   = ["accept"] * len(segments)   # or from get_reject_segments()
    accepted, rejected = get_decision_segments(segments, decisions, pre_reject)

    print(f"Accepted: {len(accepted)}  Rejected: {len(rejected)}")

---

See Also
--------

- :py:mod:`vital_sqi.sqi` — all SQI functions and ``sqi_mapping``
- :py:mod:`vital_sqi.rule` — ``Rule`` and ``RuleSet`` classes
- :py:mod:`vital_sqi.calibration` — calibration subpackage API
- :py:func:`vital_sqi.pipeline.pipeline_functions.extract_sqi`
- :py:func:`vital_sqi.pipeline.pipeline_functions.classify_segments`
- :py:func:`vital_sqi.calibration.run_calibration.calibrate`
