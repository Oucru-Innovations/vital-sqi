"""
PPG Pipeline Example
====================

End-to-end example: load a PPG signal, split it into 30-second segments,
compute SQIs, and classify each segment as *accept* or *reject*.

"""

# %%
# Load signal data
# ----------------
# We generate a synthetic PPG-like signal for demonstration.

import numpy as np
import pandas as pd
from vital_sqi.common.utils import generate_timestamp

# 5 minutes of synthetic PPG at 100 Hz
fs = 100
duration_s = 300
n = fs * duration_s
t = np.linspace(0, duration_s, n)
signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n)

timestamps = generate_timestamp(None, fs, n)
df = pd.DataFrame({"time": timestamps, "PLETH": signal})

print(f"Signal shape: {df.shape}")

# %%
# Split into segments
# -------------------
# Each segment is 30 seconds with no overlap.

from vital_sqi.preprocess.segment_split import split_segment

segments, milestones = split_segment(df, sampling_rate=fs, duration=30)
print(f"Number of segments: {len(segments)}")

# %%
# Extract SQIs
# ------------
# Use the bundled SQI configuration template.

import os
from vital_sqi.pipeline.pipeline_functions import extract_sqi

sqi_dict_file = os.path.join(
    os.path.dirname(__file__), "..", "..", "vital_sqi", "resource", "sqi_dict.json"
)
sqi_dict_file = os.path.abspath(sqi_dict_file)

sqi_df = extract_sqi(segments, milestones, sqi_dict_file, wave_type="PPG")
print(sqi_df.head())

# %%
# Classify segments
# -----------------
# Use the bundled rule template with auto-mode quantile thresholds.

from vital_sqi.pipeline.pipeline_functions import classify_segments

rule_dict_file = os.path.join(
    os.path.dirname(__file__), "..", "..", "vital_sqi", "resource", "rule_dict.json"
)
rule_dict_file = os.path.abspath(rule_dict_file)

ruleset_order = {1: "perfusion", 2: "kurtosis"}

# classify_segments expects a list of DataFrames
ruleset, classified = classify_segments(
    [sqi_df], rule_dict_file, ruleset_order, auto_mode=True
)

print(classified[0]["decision"].value_counts())

# %%
# Visualise decisions
# -------------------

import matplotlib.pyplot as plt

decisions = classified[0]["decision"]
colors = ["green" if d == "accept" else "red" for d in decisions]

plt.figure(figsize=(12, 3))
plt.bar(range(len(decisions)), [1] * len(decisions), color=colors)
plt.xlabel("Segment index")
plt.ylabel("")
plt.title("PPG Segment Quality (green=accept, red=reject)")
plt.tight_layout()
plt.show()
