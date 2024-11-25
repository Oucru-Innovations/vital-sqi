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

======================= ============================================== ==================
Acronym                Domain                                          Status
======================= ============================================== ==================
``perfusion``          Statistical                                    Stable
``kurtosis``           Statistical                                    Stable
``skewness``           Statistical                                    Stable
``entropy``            Signal Processing                              Stable
``signal_to_noise``    Signal Processing                              Stable
``zero_crossings_rate`` Signal Processing                             Stable
``mean_crossing_rate`` Signal Processing                              Stable
``DTW``                Dynamic Time Warping                          Stable
======================= ============================================== ==================

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

---

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
