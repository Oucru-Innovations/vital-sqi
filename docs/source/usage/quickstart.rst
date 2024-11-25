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

      from vitalDSP.filters import bandpass_filter

      # Apply bandpass filter (0.5-5 Hz for PPG)
      filtered_signal = bandpass_filter(signal, fs=100, lowcut=0.5, highcut=5)

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
   from vitalDSP.filters import bandpass_filter
   from vital_sqi.sqi.standard_sqi import perfusion_sqi, kurtosis_sqi

   # Load signal
   data = pd.read_csv("path/to/your_signal.csv")
   signal = data["PPG"]
   timestamps = data["Time"]

   # Preprocess the signal
   filtered_signal = bandpass_filter(signal, fs=100, lowcut=0.5, highcut=5)

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

Next Steps
----------
Congratulations! You've successfully computed Signal Quality Indexes for your physiological signals. To learn more:
- Explore the available SQIs in :doc:`introduction`.
- Check out advanced workflows using :doc:`VitalDSP <../vitalDSP/introduction>` for preprocessing.
- Dive into documentation on :doc:`pipelines` for integrating SQIs into larger projects.

Happy coding!
