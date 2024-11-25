"""
vital_sqi.sqi
==============
A subpackage for signal quality index calculation, including:
- Standard: Statistical domain
- Dynamic template maching based
- Peak detection based
- Heart rate variability based

"""

from vital_sqi.sqi.dtw_sqi import dtw_sqi
from vital_sqi.sqi.standard_sqi import (
    perfusion_sqi,
    kurtosis_sqi,
    skewness_sqi,
    entropy_sqi,
    signal_to_noise_sqi,
    zero_crossings_rate_sqi,
    mean_crossing_rate_sqi,
)
from vital_sqi.sqi.rpeaks_sqi import (
    ectopic_sqi,
    correlogram_sqi,
    interpolation_sqi,
    msq_sqi,
)
from vital_sqi.sqi.hrv_sqi import (
    nn_mean_sqi,
    sdnn_sqi,
    sdsd_sqi,
    rmssd_sqi,
    cvsd_sqi,
    cvnn_sqi,
    mean_nn_sqi,
    median_nn_sqi,
    pnn_sqi,
    hr_sqi,
    hr_range_sqi,
    frequency_sqi,
    lf_hf_ratio_sqi,
    poincare_features_sqi,
    get_all_features_hrva,
)
from vital_sqi.sqi.waveform_sqi import (
    band_energy_sqi,
    lf_energy_sqi,
    qrs_energy_sqi,
    hf_energy_sqi,
    vhf_norm_power_sqi,
    qrs_a_sqi,
)

from vital_sqi.sqi.dtw_sqi import *
from vital_sqi.sqi.hrv_sqi import *
from vital_sqi.sqi.rpeaks_sqi import *
from vital_sqi.sqi.waveform_sqi import *
from vital_sqi.sqi.standard_sqi import *

# Exported mapping for SQI functions
sqi_mapping = {
    "perfusion_sqi": perfusion_sqi,
    "kurtosis_sqi": kurtosis_sqi,
    "skewness_sqi": skewness_sqi,
    "entropy_sqi": entropy_sqi,
    "signal_to_noise_sqi": signal_to_noise_sqi,
    "zero_crossings_rate_sqi": zero_crossings_rate_sqi,
    "mean_crossing_rate_sqi": mean_crossing_rate_sqi,
    "ectopic_sqi": ectopic_sqi,
    "correlogram_sqi": correlogram_sqi,
    "interpolation_sqi": interpolation_sqi,
    "msq_sqi": msq_sqi,
    "band_energy_sqi": band_energy_sqi,
    "lfe_sqi": lf_energy_sqi,
    "qrse_sqi": qrs_energy_sqi,
    "hfe_sqi": hf_energy_sqi,
    "vhfp_sqi": vhf_norm_power_sqi,
    "qrsa_sqi": qrs_a_sqi,
    "dtw_sqi": dtw_sqi,
    "nn_mean_sqi": nn_mean_sqi,
    "sdnn_sqi": sdnn_sqi,
    "sdsd_sqi": sdsd_sqi,
    "rmssd_sqi": rmssd_sqi,
    "cvsd_sqi": cvsd_sqi,
    "cvnn_sqi": cvnn_sqi,
    "mean_nn_sqi": mean_nn_sqi,
    "median_nn_sqi": median_nn_sqi,
    "pnn_sqi": pnn_sqi,
    "hr_mean_sqi": lambda nn_intervals: hr_sqi(nn_intervals, stat="mean"),
    "hr_median_sqi": lambda nn_intervals: hr_sqi(nn_intervals, stat="median"),
    "hr_min_sqi": lambda nn_intervals: hr_sqi(nn_intervals, stat="min"),
    "hr_max_sqi": lambda nn_intervals: hr_sqi(nn_intervals, stat="max"),
    "hr_std_sqi": lambda nn_intervals: hr_sqi(nn_intervals, stat="std"),
    "hr_range_sqi": hr_range_sqi,
    "peak_frequency_sqi": lambda freqs, pows: frequency_sqi(freqs, pows, metric="peak"),
    "absolute_power_sqi": lambda freqs, pows: frequency_sqi(
        freqs, pows, metric="absolute"
    ),
    "log_power_sqi": lambda freqs, pows: frequency_sqi(freqs, pows, metric="log"),
    "relative_power_sqi": lambda freqs, pows: frequency_sqi(
        freqs, pows, metric="relative"
    ),
    "normalized_power_sqi": lambda freqs, pows: frequency_sqi(
        freqs, pows, metric="normalized"
    ),
    "lf_hf_ratio_sqi": lf_hf_ratio_sqi,
    "poincare_sqi": poincare_features_sqi,
}
