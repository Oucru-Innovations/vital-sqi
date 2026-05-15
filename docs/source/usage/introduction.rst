Introduction
============

Welcome to the **vital_sqi** library! This introduction provides an overview of the library's purpose, the physiological signals it processes, and the Signal Quality Indexes (SQIs) it computes. **vital_sqi** is optimized for use alongside our **VitalDSP** library, creating a powerful ecosystem for analyzing and processing physiological signals.

Background
----------
Signal quality is a crucial aspect of physiological signal processing. In real-world scenarios, signals such as ECG (Electrocardiogram) and PPG (Photoplethysmogram) are often corrupted by noise, artifacts, or other distortions that affect their utility for clinical or analytical purposes. **vital_sqi** quantifies signal quality using robust, automated methods to ensure reliable downstream analyses.

When combined with **VitalDSP**, you can perform advanced signal processing, such as filtering, transformations, and feature extraction, before applying **vital_sqi**'s comprehensive SQI metrics for signal quality assessment.

The library offers a suite of Signal Quality Indexes (SQIs) that fall into three domains:
1. **Statistical domain**: Evaluates statistical properties of the signal.
2. **Signal processing domain**: Assesses signal characteristics based on transformations and filters.
3. **Dynamic time warping domain**: Measures similarity between signals based on time-alignment.

---

ECG Signals
~~~~~~~~~~~
**Electrocardiogram (ECG)** signals measure the electrical activity of the heart and are widely used for diagnosing cardiovascular conditions. Key features of ECG signals include:
- **P wave**: Represents atrial depolarization.
- **QRS complex**: Represents ventricular depolarization and is critical for heart rate analysis.
- **T wave**: Represents ventricular repolarization.

ECG signals are prone to noise from muscle activity, electrode placement, and movement artifacts. **vital_sqi**, especially when used with **VitalDSP**, helps assess ECG signal quality for improved reliability in clinical settings.

---

PPG Signals
~~~~~~~~~~~
**Photoplethysmogram (PPG)** signals measure blood volume changes in the microvascular bed of tissue. They are commonly used in wearable devices to monitor heart rate, blood oxygen saturation, and other vital signs.

PPG signals are affected by:
- **Motion artifacts**: Caused by patient movement or device placement.
- **Ambient light**: Can distort readings in optical sensors.
- **Physiological noise**: Such as respiration and muscle activity.

By combining **VitalDSP**'s preprocessing and artifact removal tools with **vital_sqi**'s quality assessment metrics, you can ensure high-quality signals for more accurate physiological measurements.

---

SQI Indexes
-----------
**Signal Quality Indexes (SQIs)** quantify the reliability of physiological signals. These metrics fall into three domains:

1. **Statistical Domain**:
   - Metrics based on statistical properties of the signal.
   - Examples: Kurtosis, Skewness, Perfusion Index.

2. **Signal Processing Domain**:
   - Metrics derived from signal transformations and characteristics.
   - Examples: Entropy, Signal-to-Noise Ratio, Zero Crossing Rate, Mean Crossing Rate.

3. **Dynamic Time Warping Domain**:
   - Metrics based on time-alignment and similarity between signals.
   - Example: Dynamic Time Warping (DTW).

Below is a summary of the SQIs supported by the library:

============================== ============================================== ==================
Acronym                       Domain                                          Status
============================== ============================================== ==================
``perfusion``                 Statistical                                    Stable
``kurtosis``                  Statistical                                    Stable
``skewness``                  Statistical                                    Stable
``entropy``                   Statistical                                    Stable
``signal_to_noise``           Statistical                                    Stable
``zero_crossings_rate``       Statistical / Signal Processing                Stable
``mean_crossing_rate``        Statistical / Signal Processing                Stable
``clipping``                  Statistical / Signal Processing                Stable
``baseline_wander``           Signal Processing                              Stable
``spectral_snr``              Signal Processing                              Stable
``band_energy``               Signal Processing                              Stable
``lfe``                       Signal Processing                              Stable
``qrse``                      Signal Processing                              Stable
``hfe``                       Signal Processing                              Stable
``vhfp``                      Signal Processing                              Stable
``qrsa``                      Signal Processing                              Stable
``DTW``                       Dynamic Time Warping                           Stable
``correlogram``               R-peak / HRV                                   Stable
``ectopic``                   R-peak / HRV                                   Stable
``msq``                       R-peak / HRV                                   Stable
``amplitude_consistency``     R-peak / HRV                                   Stable
``rr_irregularity``           HRV (time domain)                              Stable
``interpolation``             R-peak / HRV                                   Stub (returns NaN)
``sdnn``                      HRV (time domain)                              Stable
``rmssd``                     HRV (time domain)                              Stable
``sdsd``                      HRV (time domain)                              Stable
``cvsd``                      HRV (time domain)                              Stable
``cvnn``                      HRV (time domain)                              Stable
``mean_nn``                   HRV (time domain)                              Stable
``median_nn``                 HRV (time domain)                              Stable
``pnn``                       HRV (time domain)                              Stable
``rr_irregularity``           HRV (time domain)                              Stable
``hr_mean``                   HRV (heart rate)                               Stable
``hr_median``                 HRV (heart rate)                               Stable
``hr_min``                    HRV (heart rate)                               Stable
``hr_max``                    HRV (heart rate)                               Stable
``hr_std``                    HRV (heart rate)                               Stable
``hr_range``                  HRV (heart rate)                               Stable
``peak_frequency``            HRV (frequency domain)                         Stable
``absolute_power``            HRV (frequency domain)                         Stable
``log_power``                 HRV (frequency domain)                         Stable
``relative_power``            HRV (frequency domain)                         Stable
``normalized_power``          HRV (frequency domain)                         Stable
``lf_hf_ratio``               HRV (frequency domain)                         Stable
``poincare``                  HRV (nonlinear)                                Stable
``sample_entropy``            HRV (nonlinear)                                Stable
``dfa``                       HRV (nonlinear)                                Stable
``hurst``                     HRV (nonlinear)                                Stable
============================== ============================================== ==================

References:
- [1] Optimal Signal Quality Index for Photoplethysmogram Signals, Mohamed Elgendi et al.
- [2] Dynamic Time Warping for Signal Quality Assessment
- [3] Statistical Metrics for ECG Signal Analysis

---

Detailed SQI Descriptions
--------------------------

- **Perfusion Index (`perfusion`)**:
   - Measures the ratio of pulsatile to non-pulsatile blood flow.
   - Useful for evaluating the strength of the PPG signal.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.perfusion_sqi`

- **Kurtosis (`kurtosis`)**:
   - Measures the sharpness or peakedness of the signal distribution.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.kurtosis_sqi`

- **Skewness (`skewness`)**:
   - Evaluates the asymmetry of the signal distribution.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.skewness_sqi`

- **Entropy (`entropy`)**:
   - Quantifies the randomness or complexity of the signal.
   - Commonly used for assessing the regularity of physiological signals.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.entropy_sqi`

- **Signal-to-Noise Ratio (`signal_to_noise`)**:
   - Measures the ratio of signal power to noise power.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.signal_to_noise_sqi`

- **Zero Crossings Rate (`zero_crossings_rate`)**:
   - Calculates the rate at which the signal crosses zero.
   - Indicative of high-frequency components or noise.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.zero_crossings_rate_sqi`

- **Mean Crossing Rate (`mean_crossing_rate`)**:
   - Similar to zero crossings but uses the mean as the reference point.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.mean_crossing_rate_sqi`

- **Dynamic Time Warping (`DTW`)**:
   - Aligns two signals temporally and computes a similarity score.
   - Particularly useful for time-series comparisons.
   - See: :py:func:`vital_sqi.sqi.dtw_sqi.dtw_sqi`

- **Clipping (`clipping`)**:
   - Fraction of samples at or near the amplitude rail (saturation detection).
   - Returns 0.0 for a clean signal; >0.05 indicates heavy saturation.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.clipping_sqi`

- **Baseline Wander (`baseline_wander`)**:
   - Ratio of sub-0.5 Hz STFT energy to total energy. High values indicate slow drift.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.baseline_wander_sqi`

- **Spectral SNR (`spectral_snr`)**:
   - 10*log10(in-band power / out-of-band power) in dB. Defaults to PPG band 0.5-4 Hz.
   - See: :py:func:`vital_sqi.sqi.standard_sqi.spectral_snr_sqi`

- **Amplitude Consistency (`amplitude_consistency`)**:
   - Coefficient of variation of beat-to-beat peak amplitudes. Low CV = stable beats.
   - See: :py:func:`vital_sqi.sqi.rpeaks_sqi.amplitude_consistency_sqi`

- **Band Energy SQIs (`lfe`, `qrse`, `hfe`, `vhfp`, `qrsa`)**:
   - Frequency-band energy metrics derived from the short-time Fourier transform.
   - ``lfe``: low-frequency (0–0.5 Hz) STFT energy — indicates baseline wander.
   - ``qrse``: QRS-band (5–25 Hz) STFT energy — captures ventricular activation.
   - ``hfe``: high-frequency (>100 Hz) STFT energy — sensitive to electrode noise.
   - ``vhfp``: very-high-frequency (>150 Hz) normalised power fraction.
   - ``qrsa``: median QRS amplitude via vitalDSP WaveformMorphology.
   - See: :py:mod:`vital_sqi.sqi.waveform_sqi`

- **RR Irregularity (`rr_irregularity`)**:
   - Mean absolute deviation of successive RR differences normalised by median RR.
   - More sensitive than SDNN for detecting ectopic beats and AF.
   - See: :py:func:`vital_sqi.sqi.hrv_sqi.rr_irregularity_sqi`

- **Sample Entropy (`sample_entropy`)**:
   - Complexity/regularity of the NN interval series (Richman & Moorman, 2000).
   - Lower SampEn = more regular = better quality signal.
   - See: :py:func:`vital_sqi.sqi.hrv_sqi.sample_entropy_sqi`

- **DFA (`dfa`)**:
   - Detrended Fluctuation Analysis short-range scaling exponent (alpha1).
   - Healthy sinus rhythm: alpha1 ~ 1.0-1.2; white noise: ~0.5.
   - See: :py:func:`vital_sqi.sqi.hrv_sqi.dfa_sqi`

- **Hurst (`hurst`)**:
   - Hurst exponent via R/S analysis. H > 0.5 = persistent structured signal.
   - See: :py:func:`vital_sqi.sqi.hrv_sqi.hurst_sqi`

---

Peak Detection
--------------

**vital_sqi** bundles its own ``PeakDetector`` class
(``vital_sqi.common.rpeak_detection``) so that peak-based SQIs work without
any additional dependencies beyond **vitalDSP**.

**PPG algorithms** (9 methods): ``DEFAULT`` (vitalDSP WaveformMorphology),
``ADAPTIVE_THRESHOLD``, ``COUNT_ORIG_METHOD``, ``CLUSTERER_METHOD``,
``SLOPE_SUM_METHOD``, ``MOVING_AVERAGE_METHOD``, ``BILLAUER_METHOD``,
``AMPD_METHOD`` (Scholkmann 2012), and ``LOCAL_MAX_IBI``.

**ECG algorithms** (4 methods): ``ECG_DEFAULT`` (vitalDSP WaveformMorphology),
``PAN_TOMPKINS`` (1985), ``HAMILTON`` (2002), ``ENGZEE`` (1979).
All ECG algorithms use vitalDSP for Q/S/P/T morphology extraction, anchored to
the R-peaks found by the selected algorithm.

.. note::
   ``msq_sqi`` currently returns ``NaN`` and emits a ``RuntimeWarning`` for
   ECG signals.  A meaningful ECG MSQ requires two independently calibrated
   R-peak detectors (e.g. Pan-Tompkins vs Hamilton) producing comparable
   indices.  A future release will wire the ECG detector-switching API into
   ``msq_sqi`` to enable true ECG peak-agreement scoring.

---

Classification modes
--------------------

After SQIs are computed, segments are labelled accept / reject by a
rule engine.  Three threshold-selection strategies are available
(:func:`vital_sqi.pipeline.pipeline_functions.classify_segments`,
``auto_mode`` argument):

* **Manual** — use thresholds shipped in ``rule_dict.json`` verbatim.
  Best when applying externally calibrated bounds without adapting
  them to the current recording.

* **Quantile** (default) — replace each rule's bounds with the
  empirical *lower / upper* quantiles (p5 / p95 by default) of the SQI
  values observed across all segments.  Self-adapting per recording.

* **Auto-tune** — pick the per-rule quantile so the *joint* accept
  rate hits a user-specified target (default 85 %) under the
  independence approximation.  Much more forgiving than plain Quantile
  when several rules are stacked.

A separate non-rule-based classifier — three-regime auto-detection
(:func:`~vital_sqi.rule.classify_segments_robust`) — is also
available for cases where you don't trust the rule definitions at all.

See :doc:`pipeline` for the full math and a worked example.


Using **vital_sqi** with VitalDSP
---------------------------------
While **vital_sqi** can be used as a standalone library, it works best when combined with **VitalDSP**. Here’s why:
- **Advanced Preprocessing**: Use **VitalDSP** to filter, detrend, and denoise ECG and PPG signals before computing SQIs.
- **Feature Extraction**: Combine **VitalDSP**'s feature engineering tools with SQIs for comprehensive signal analysis.
- **Seamless Integration**: Both libraries are designed to complement each other, enabling end-to-end signal processing workflows.

---

Conclusion
----------
**vital_sqi** is designed to provide robust tools for signal quality assessment, ensuring high reliability in both ECG and PPG analyses. When paired with **VitalDSP**, you gain access to a complete pipeline for signal processing and quality evaluation.

Ready to get started? Explore the installation and usage guides next!
