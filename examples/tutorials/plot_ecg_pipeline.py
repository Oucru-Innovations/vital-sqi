"""
ECG Pipeline Example
====================

End-to-end example: load an ECG signal, split into segments, compute SQIs
(including HRV-based metrics), and classify each segment.

"""

# %%
# Synthesise an ECG signal
# ------------------------

import numpy as np
import pandas as pd
from vital_sqi.common.utils import generate_timestamp


def _simple_ecg(n, fs=250, hr=70, noise=0.02):
    """Minimal synthetic ECG: sum of Gaussian QRS peaks."""
    t = np.arange(n) / fs
    rr = 60.0 / hr
    peak_times = np.arange(0, t[-1], rr)
    ecg = np.zeros(n)
    for pt in peak_times:
        idx = int(pt * fs)
        if idx < n:
            window = np.arange(max(0, idx - 10), min(n, idx + 10))
            ecg[window] += np.exp(-0.5 * ((window - idx) / 3) ** 2)
    ecg += noise * np.random.randn(n)
    return ecg


fs = 250
n = fs * 300  # 5 minutes
signal = _simple_ecg(n, fs=fs)
timestamps = generate_timestamp(None, fs, n)
df = pd.DataFrame({"time": timestamps, "ECG": signal})

print(f"Signal shape: {df.shape}")

# %%
# Segment the signal
# ------------------

from vital_sqi.preprocess.segment_split import split_segment

segments, milestones = split_segment(df, sampling_rate=fs, duration=30, wave_type="ECG")
print(f"Segments: {len(segments)}")

# %%
# Extract SQIs
# ------------

import os
from vital_sqi.pipeline.pipeline_functions import extract_sqi

sqi_dict_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "vital_sqi", "resource", "sqi_dict.json",
    )
)

sqi_df = extract_sqi(segments, milestones, sqi_dict_file, wave_type="ECG")
print(sqi_df[["perfusion", "kurtosis_1", "correlogram"]].head())

# %%
# Classify segments
# -----------------

from vital_sqi.pipeline.pipeline_functions import classify_segments

rule_dict_file = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "vital_sqi", "resource", "rule_dict.json",
    )
)

ruleset_order = {1: "perfusion", 2: "kurtosis"}
ruleset, classified = classify_segments(
    [sqi_df], rule_dict_file, ruleset_order, auto_mode=True
)

print(classified[0]["decision"].value_counts())

# %%
# Plot
# ----

import matplotlib.pyplot as plt

decisions = classified[0]["decision"]
colors = ["steelblue" if d == "accept" else "tomato" for d in decisions]

plt.figure(figsize=(12, 3))
plt.bar(range(len(decisions)), [1] * len(decisions), color=colors)
plt.xlabel("Segment index")
plt.title("ECG Segment Quality (blue=accept, red=reject)")
plt.tight_layout()
plt.show()
