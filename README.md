# Vital_sqi: Signal quality control of physiological signals

<!-- [![Build Status](https://travis-ci.com/meta00/vital_sqi.svg?token=CDjcmJqzLe7opuWagsPJ&branch=main)](https://travis-ci.com/meta00/vital_sqi)
[![codecov](https://codecov.io/gh/meta00/vital_sqi/branch/main/graph/badge.svg?token=6RV5BUK340)](https://codecov.io/gh/meta00/vital_sqi)
[![Documentation Status](https://readthedocs.org/projects/vitalsqi/badge/?version=latest)](https://vitalsqi.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/vital-sqi.svg)](https://badge.fury.io/py/vital-sqi)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

![GitHub stars](https://img.shields.io/github/stars/Oucru-Innovations/vital-sqi?style=social)
![PyPI - Downloads](https://img.shields.io/pypi/dm/vitalSQI-toolkit)
![PyPI](https://img.shields.io/pypi/v/vitalSQI-toolkit)
![Build Status](https://github.com/Oucru-Innovations/vital-sqi/actions/workflows/ci.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/Oucru-Innovations/vital-sqi/badge.svg?branch=main)](https://coveralls.io/github/Oucru-Innovations/vital-sqi?branch=main)
![GitHub License](https://img.shields.io/github/license/Oucru-Innovations/vital-sqi)
![Python Versions](https://img.shields.io/badge/python-3.7%2B-blue)
[![Documentation Status](https://readthedocs.org/projects/vital-sqi/badge/?version=latest)](https://vital-sqi.readthedocs.io/en/latest/?badge=latest)
<!-- ![PyPI Downloads](https://img.shields.io/pypi/dm/vitalsqi)
[![PyPI version](https://badge.fury.io/py/vitalsqi.svg)](https://badge.fury.io/py/vitalsqi)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Oucru-Innovations/vital-sqi/blob/main/docs/source/notebooks/synthesize_data.ipynb) -->

---

# Overview

**Vital_sqi** is an open-source Python package for **signal quality index (SQI)** extraction and **quality classification** of ECG and PPG signals. The package enables streamlined signal quality control, essential for reliable health monitoring and clinical research.

## Key Features:
1. Support for **PPG and ECG signals** in various formats, including wearable device data.
2. Implementation of **state-of-the-art SQI extraction methods** (e.g., statistical, HRV, waveform-based).
3. Rule-based **quality classification** using user-defined thresholds.
4. **Pipelines** for end-to-end processing, from data ingestion to SQI classification.
5. A **GUI tool** for defining rules and visualizing results interactively.

---

# Prerequisites and Installation

The package supports Python `3.7` and `3.8`. Install Vital_sqi using pip:

```bash
pip install vitalSQI-toolkit
```

### Optional Dependencies:
- **Dash** for GUI applications.
- **Plotly** for visualization.
- **Matplotlib** for advanced plotting options.

For detailed dependencies, refer to the [documentation](https://vital-sqi.readthedocs.io/en/latest/).

---

# Getting Started

The core of the package is built around three main classes:

### **1. SignalSQI Class**
This class handles the main signal data and SQI extraction workflow.
Attributes:
- `signal`: Waveform data (as a pandas DataFrame).
- `sampling_rate`: Sampling rate of the signal (user-defined or auto-inferred).
- `sqis`: SQI values calculated for signal segments.
- `rules` and `ruleset`: Lists of applied rules and the ruleset used for classification.

### **2. Rule Class**
Defines individual SQI-based rules for classification.
Example rule structure in JSON:
```json
{
  "test_sqi": {
    "name": "test_sqi",
    "def": [
      {"op": ">", "value": "10", "label": "reject"},
      {"op": ">=", "value": "3", "label": "accept"},
      {"op": "<", "value": "3", "label": "reject"}
    ],
    "desc": "",
    "ref": ""
  }
}
```

### **3. Ruleset Class**
Groups rules and defines the sequence for applying them to signal segments. The ruleset is defined in JSON format and can be customized to fit specific needs.`ruleset` object of class `Ruleset` contains a set of selected `rules` (selected from the list of rules in `signal_obj.rule`) and the order to apply them in quality 
assignment (see schema below). Notice that this is not a tree-based classification.
  
![Example of a rule set](images/resize_sample_rule_chart.png "Example of a rule set")

# Pipelines

The package includes predefined pipelines for processing ECG and PPG signals. 

### Example: Extracting SQIs from ECG
```python
from vital_sqi.pipeline.pipeline_highlevel import *
from vital_sqi.data.signal_sqi_class import SignalSQI
import os

file_in = os.path.abspath('tests/test_data/example.edf')
sqi_dict = os.path.abspath('tests/test_data/sqi_dict.json')
segments, signal_sqi_obj = get_ecg_sqis(file_in, sqi_dict, 'edf')
```

### Example: Quality Classification for ECG
```python
from vital_sqi.pipeline.pipeline_highlevel import *
import os
import tempfile

file_in = os.path.abspath('tests/test_data/example.edf')
sqi_dict = os.path.abspath('tests/test_data/sqi_dict.json')
rule_dict_filename = os.path.abspath('tests/test_data/rule_dict_test.json')
ruleset_order = {3: 'skewness_1', 2: 'entropy', 1: 'perfusion'}
output_dir = tempfile.gettempdir()

signal_obj = get_qualified_ecg(
    file_name=file_in,
    sqi_dict_filename=sqi_dict,
    file_type='edf',
    duration=30,
    rule_dict_filename=rule_dict_filename,
    ruleset_order=ruleset_order,
    output_dir=output_dir
)
```

---

# GUI for Rules and Execution

Vital_sqi provides a GUI for creating rules, defining rulesets, and executing them interactively. The GUI helps users:
- Configure rules visually.
- Test and validate signal quality thresholds.
- Export results for further analysis.

[Click here for the GUI guide](https://vital-sqi.readthedocs.io/en/latest/docstring/vital_sqi.app.html#module-vital_sqi.app.app).


# Workflow Overview

### **1. Reading and Writing Signals**
Supported formats:
- ECG: `EDF`, `MIT`, `CSV`.
- PPG: `CSV`.

### **2. Preprocessing and Segmentation**
Available preprocessing steps:
- Trimming, tapering, and smoothing.
- Bandpass filtering.
- Segmentation by duration or by beat.

### **3. SQI Extraction**
Four types of SQIs:
1. **Statistical SQIs**: Kurtosis, skewness, entropy, etc.
2. **HRV-based SQIs**: SDNN, SDSD, RMSSD, etc.
3. **RR Interval-based SQIs**: Ectopic, correlogram, etc.
4. **Waveform-based SQIs**: DTW, QRS energy, etc.

The function `vital_sqi.pipeline_function.extract_sqi` is used to extract a number of SQIs from segments. The requested SQIs
are defined in a json file called SQI dictionary. We provide a dictionary template for all implemented SQIs, with default 
parameters, in `vital_sqi/resource/sqi_dict.json`.

### **4. Signal Quality Classification**
- Rule-based classification using user-defined thresholds.
- Optimized rule application for performance.

Templates for rules and rulesets are available in the `vital_sqi/resource` directory.

The package allows making rule set from SQIs and user-defined thresholds for quality classification. A segment assigned 
as `accept` pass all rules in the set, otherwise `reject`. Rules in the set have ordered, which might help to 
improve speed.

We ran brute force threshold searching for an in-house PPG dataset (generated with Smartcare, doubly annotated 
by domain experts) to obtain a set of recommended thresholds, as found in `resource/rule_dict.json`.

---

# Documentation

Find detailed tutorials, examples, and API references at:
🔗 [Vital_sqi Documentation](https://vital-sqi.readthedocs.io/en/latest/)

---

# Contributions

We welcome contributions from the community! Please refer to our [CONTRIBUTING.md](https://github.com/Oucru-Innovations/vital-sqi/blob/main/CONTRIBUTING.md) for guidelines.

---

# References

- [Optimal Signal Quality Index for Photoplethysmogram Signals](https://doi.org/10.xxxx)
- [Other relevant papers and research articles]

---

# License

Vital_sqi is licensed under the MIT License. See the [LICENSE](https://github.com/Oucru-Innovations/vital-sqi/blob/main/LICENSE) file for details.

---

Thank you for supporting Vital_sqi! For questions or issues, feel free to [open an issue](https://github.com/Oucru-Innovations/vital-sqi/issues) on GitHub.
