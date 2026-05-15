Quickstart
==========

Welcome to the **vital_sqi** quickstart guide! This section provides step-by-step instructions for setting up and using **vital_sqi** for signal quality assessment. By the end of this guide, you’ll be able to compute Signal Quality Indexes (SQIs) for ECG and PPG signals and integrate them into your workflow.

Prerequisites
-------------
Before starting, ensure you have the following:
- **Python 3.7+** installed.
- **vital_sqi** installed. Refer to the :doc:`installation` guide if needed.
- Sample ECG or PPG data in a supported format (e.g., `.csv`, `.txt`, or `.edf`).

If you plan to preprocess your data before assessing quality, install and configure the **VitalDSP** library for seamless integration.

---

Getting Started
---------------
Follow these steps to quickly set up and run your first SQI computation:

1. **Import the Library**:
   Import the required modules from **vital_sqi**:

   .. code-block:: python

      from vital_sqi.sqi.standard_sqi import perfusion_sqi, kurtosis_sqi
      from vital_sqi.sqi.dtw_sqi import dtw_sqi
      import numpy as np

2. **Load Your Data**:
   Load ECG or PPG data using your preferred method. For example, if your data is stored in a CSV file:

   .. code-block:: python

      import pandas as pd

      # Load data
      data = pd.read_csv("path/to/your_signal.csv")
      signal = data["PPG"]  # Assuming 'PPG' is the column name
      timestamps = data["Time"]  # Optional: timestamps for alignment

3. **Compute SQIs**:
   Use **vital_sqi** to calculate SQIs for the signal:

   .. code-block:: python

      # Example: Compute Perfusion Index
      perfusion_score = perfusion_sqi(signal)

      # Example: Compute Kurtosis
      kurtosis_score = kurtosis_sqi(signal)

      # Example: Compute Dynamic Time Warping (DTW) similarity
      reference_signal = np.sin(np.linspace(0, 10, len(signal)))  # Reference signal
      dtw_score = dtw_sqi(signal, reference_signal)

      print("Perfusion SQI:", perfusion_score)
      print("Kurtosis SQI:", kurtosis_score)
      print("DTW Score:", dtw_score)

4. **Visualize Results**:
   Visualize your signal alongside SQI results for better interpretation:

   .. code-block:: python

      import matplotlib.pyplot as plt

      plt.figure(figsize=(10, 5))
      plt.plot(timestamps, signal, label="PPG Signal")
      plt.title("PPG Signal with Computed SQIs")
      plt.xlabel("Time")
      plt.ylabel("Amplitude")
      plt.legend()
      plt.show()

---

Using Preprocessing with VitalDSP
---------------------------------
To improve SQI computations, preprocess your signals with **VitalDSP**. For example:

1. **Install VitalDSP**:
   .. code-block:: bash

      pip install vital-DSP

2. **Filter and Denoise Your Signal**:
   Use **VitalDSP** to apply a bandpass filter and remove noise:

   .. code-block:: python

      from vitalDSP.signal_processing.signal_filtering import SignalFiltering

      # Apply bandpass filter (0.5-5 Hz for PPG)
      sf = SignalFiltering(signal, fs=100)
      filtered_signal = sf.bandpass_filter(lowcut=0.5, highcut=5)

3. **Recompute SQIs**:
   Use the filtered signal with **vital_sqi**:

   .. code-block:: python

      perfusion_score = perfusion_sqi(filtered_signal)
      kurtosis_score = kurtosis_sqi(filtered_signal)
      print("Perfusion SQI (filtered):", perfusion_score)
      print("Kurtosis SQI (filtered):", kurtosis_score)

---

Example Use Case
----------------
Here’s a complete example workflow to compute and analyze SQIs:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from vitalDSP.signal_processing.signal_filtering import SignalFiltering
   from vital_sqi.sqi.standard_sqi import perfusion_sqi, kurtosis_sqi

   # Load signal
   data = pd.read_csv("path/to/your_signal.csv")
   signal = data["PPG"].values
   timestamps = data["Time"]

   # Preprocess the signal
   sf = SignalFiltering(signal, fs=100)
   filtered_signal = sf.bandpass_filter(lowcut=0.5, highcut=5)

   # Compute SQIs
   perfusion_score = perfusion_sqi(filtered_signal)
   kurtosis_score = kurtosis_sqi(filtered_signal)

   # Visualize results
   plt.figure(figsize=(10, 5))
   plt.plot(timestamps, signal, label="Raw Signal")
   plt.plot(timestamps, filtered_signal, label="Filtered Signal")
   plt.title("PPG Signal with SQI Analysis")
   plt.xlabel("Time")
   plt.ylabel("Amplitude")
   plt.legend()
   plt.show()

   print("Perfusion SQI:", perfusion_score)
   print("Kurtosis SQI:", kurtosis_score)

---

End-to-end pipeline with auto-tuned classifier
----------------------------------------------

The example above computes individual SQIs.  Most users want the full
pipeline: load → segment → extract every SQI → classify segments as
accept / reject.  That's four calls:

.. code-block:: python

   import pandas as pd
   from vital_sqi.common.utils import generate_timestamp
   from vital_sqi.preprocess.segment_split import split_segment
   from vital_sqi.pipeline.pipeline_functions import (
       extract_sqi, classify_segments,
   )

   # 1. Load + wrap into a DataFrame with a timestamps column.
   df = pd.read_csv("recording.csv")
   fs = 100
   df = pd.DataFrame({
       "timestamps": generate_timestamp(None, fs, len(df)),
       "signal":     df["PPG"].values,
   })

   # 2. Split into 30-second non-overlapping segments.
   segments, milestones = split_segment(
       df, sampling_rate=fs, split_type=0,
       duration=30, overlapping=0, wave_type="PPG",
   )

   # 3. Compute every SQI in the bundled catalogue.
   sqi_df = extract_sqi(
       segments, milestones,
       "vital_sqi/resource/sqi_dict.json",
       wave_type="PPG",
   )

   # 4. Classify each segment.  Auto-tune mode targets an 85 % joint
   #    accept rate; per-rule quantiles are picked accordingly under
   #    the independence approximation.
   ruleset_order = {
       1: "kurtosis_sqi",
       2: "perfusion_sqi",
       3: "correlogram_sqi",
       4: "msq_sqi",
       5: "dtw_sqi",
   }
   ruleset, sqis_with_decisions = classify_segments(
       [sqi_df.copy()],
       rule_dict_filename="vital_sqi/resource/rule_dict.json",
       ruleset_order=ruleset_order,
       auto_mode="tune",
       target_accept_rate=0.85,
   )

   decisions = list(sqis_with_decisions[0]["decision"])
   print(f"Accepted {decisions.count('accept')}/{len(decisions)} segments")

For the corresponding GUI workflow — drop a recording in the browser
and tweak the threshold mode interactively — see :doc:`app`.

---

Next Steps
----------
Congratulations! You've successfully computed Signal Quality Indexes for your physiological signals. To learn more:
- Explore the available SQIs in :doc:`introduction`.
- Read the full pipeline reference in :doc:`pipeline` for the end-to-end workflow.
- Check out `VitalDSP <https://vital-dsp.readthedocs.io/>`_ for advanced preprocessing.
- Browse the API docs in :doc:`../docstring/vital_sqi.pipeline`.

Happy coding!
