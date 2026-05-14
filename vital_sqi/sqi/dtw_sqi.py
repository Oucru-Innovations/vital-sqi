"""Signal quality indexes based on dynamic template matching
"""

import numpy as np
from vital_sqi.common.generate_template import (
    ppg_absolute_dual_skewness_template,
    ppg_dual_double_frequency_template,
    ppg_nonlinear_dynamic_system_template,
    ecg_dynamic_template,
)
from vital_sqi.common.utils import check_valid_signal
from scipy.spatial.distance import euclidean
from scipy.signal import resample

# from librosa.sequence import dtw
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def dtw_distance(seq1, seq2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences in a vectorized manner.

    Parameters
    ----------
    seq1 : array_like
        The first sequence (e.g., the signal)
    seq2 : array_like
        The second sequence (e.g., the template)

    Returns
    -------
    float
        The DTW distance between the sequences
    """
    n, m = len(seq1), len(seq2)

    # Calculate cost matrix in a vectorized manner
    cost_matrix = cdist(seq1.reshape(-1, 1), seq2.reshape(-1, 1), metric="euclidean")

    # Initialize the DTW matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill the DTW matrix row by row.  The "insert" transition dtw[i, j-1]
    # must use the already-updated value from the current row, so a cell-by-
    # cell loop is required for correctness.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw_matrix[i, j] = cost_matrix[i - 1, j - 1] + min(
                dtw_matrix[i - 1, j],    # deletion
                dtw_matrix[i, j - 1],    # insertion (updated value)
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m]


def dtw_sqi(s, template_type, template_size=100, simple_mode=False):
    """
    Euclidean distance between signal and its template using DTW

    Parameters
    ----------
    s : array_like
        Signal containing int or float values.

    template_type : int
        Template type identifier (0-3 for different template types)

    template_size : int, optional
        Size of the template to resample the signal and template (default is 100)

    simple_mode : bool, optional
        If True, uses a simpler Euclidean distance instead of DTW

    Returns
    -------
    float
        Calculated DTW or Euclidean distance between the signal and the template.
    """
    check_valid_signal(s)

    if template_size <= 0:
        raise ValueError("Number of samples must be greater than zero.")

    s = resample(s, template_size).reshape(-1)
    if not isinstance(template_type, int) or not (0 <= template_type <= 3):
        raise ValueError("Invalid template type")

    if template_type == 0:
        reference = ppg_nonlinear_dynamic_system_template(template_size).reshape(-1)
    elif template_type == 1:
        reference = ppg_dual_double_frequency_template(template_size)
    elif template_type == 2:
        reference = ppg_absolute_dual_skewness_template(template_size)
    elif template_type == 3:
        # ecg_dynamic_template solves an ODE at hardcoded time points and
        # returns ~100 000 samples regardless of template_size.  Resample
        # to the requested size so the DTW comparison is meaningful.
        raw_template = np.array(ecg_dynamic_template(template_size)).reshape(-1)
        reference = resample(raw_template, template_size)
    else:
        raise ValueError("Invalid template type")

    scaler = MinMaxScaler(feature_range=(0, 1))
    s = scaler.fit_transform(s.reshape(-1, 1)).reshape(-1)
    reference = scaler.fit_transform(reference.reshape(-1, 1)).reshape(-1)

    if simple_mode:
        return np.mean(
            [euclidean([s[i]], [reference[i]]) for i in range(template_size)]
        )
    else:
        return dtw_distance(s, reference)

    # if simple_mode:
    #     cost = 0
    #     for i in range(template_size):
    #         cost += euclidean([s[i]], [reference[i]])
    #     dtw_cost = cost / template_size
    # else:
    #     beat = resample(s, template_size)
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     beat = scaler.fit_transform(beat.reshape(-1, 1)).reshape(-1)

    #     reference = resample(reference, template_size)
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     reference = scaler.fit_transform(reference.reshape(-1, 1)).reshape(-1)

    #     # Use custom DTW function instead of librosa
    #     dtw_cost = dtw_distance(beat, reference)

    # return dtw_cost
