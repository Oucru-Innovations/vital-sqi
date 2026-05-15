"""
Implementation of waveform-based SQIs (Signal Quality Indices):
- For ECG based on DiMarco2012.
- For PPG (to be extended).
"""

import warnings
import scipy.signal as sn
import numpy as np
from vitalDSP.physiological_features.waveform import WaveformMorphology


def band_energy_sqi(signal, sampling_rate=100, band=None, nperseg=2048):
    """
    Compute the peak value of the time marginal of the energy distribution in a frequency band.

    Parameters
    ----------
    signal : array-like
        The input signal.
    sampling_rate : int, optional
        Sampling rate of the signal. Default is 100 Hz.
    band : list, optional
        Frequency band [low, high]. If None, the entire spectrum is used. Default is None.
    nperseg : int, optional
        Length of each segment for the Short-Time Fourier Transform. Default is 2048.

    Returns
    -------
    float
        Maximum time marginal power in the specified frequency band.

    Raises
    ------
    AssertionError
        If the band is not a valid list or has invalid values.

    Example
    -------
    >>> import numpy as np
    >>> from vital_sqi.sqi.waveform_sqi import band_energy_sqi
    >>> signal = np.sin(2 * np.pi * 0.01 * np.arange(1000))
    >>> band_energy_sqi(signal, sampling_rate=100, band=[0.1, 0.5])
    0.3141592653589793
    """
    assert np.isreal(sampling_rate), "Expected a numeric sampling rate value."
    if len(signal) < nperseg:
        nperseg = len(signal)

    f, t, spec = sn.stft(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=(nperseg // 2),
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
    )

    if band is None:
        max_time_marginal = float(np.max(np.sum(np.abs(spec), axis=0)))
    else:
        assert isinstance(band, list) and band[0] <= band[1], "Invalid band values."
        idx = np.where((f > band[0]) & (f <= band[1]))[0]
        if len(idx) == 0:
            return 0.0
        max_time_marginal = float(np.max(np.sum(np.abs(spec[idx]), axis=0)))

    return max_time_marginal


def lf_energy_sqi(signal, sampling_rate, band=[0, 0.5]):
    """
    Low-Frequency Energy SQI.

    Parameters
    ----------
    signal : array-like
        The input signal.
    sampling_rate : int
        Sampling rate of the signal.
    band : list, optional
        Frequency band. Default is [0, 0.5].

    Returns
    -------
    float
        Low-frequency energy SQI.
    """
    return band_energy_sqi(signal, sampling_rate, band)


def qrs_energy_sqi(signal, sampling_rate, band=[5, 25]):
    """
    QRS Energy SQI.

    Parameters
    ----------
    signal : array-like
        The input signal.
    sampling_rate : int
        Sampling rate of the signal.
    band : list, optional
        Frequency band. Default is [5, 25].

    Returns
    -------
    float
        QRS energy SQI.
    """
    return band_energy_sqi(signal, sampling_rate, band)


def hf_energy_sqi(signal, sampling_rate, band=None):
    """
    High-Frequency Energy SQI.

    Parameters
    ----------
    signal : array-like
        The input signal.
    sampling_rate : int
        Sampling rate of the signal.
    band : list, optional
        Frequency band [low_hz, high_hz]. Defaults to ``[100, np.inf]``.
        For signals sampled at 100 Hz (Nyquist = 50 Hz) this band is above
        Nyquist and will always return 0; pass an appropriate band for the
        actual sampling rate (e.g. ``[20, 50]`` for 100 Hz data).

    Returns
    -------
    float
        High-frequency energy SQI.
    """
    if band is None:
        band = [100, np.inf]
    nyquist = sampling_rate / 2.0
    if band[0] >= nyquist:
        warnings.warn(
            f"hf_energy_sqi: band lower bound ({band[0]} Hz) is at or above Nyquist "
            f"({nyquist} Hz) for sampling_rate={sampling_rate}. "
            "Returning NaN. Pass a band appropriate for this sampling rate.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan
    return band_energy_sqi(signal, sampling_rate, band)


def vhf_norm_power_sqi(signal, sampling_rate, band=None, nperseg=2048):
    """
    Very High-Frequency Normalized Power SQI.

    Parameters
    ----------
    signal : array-like
        The input signal.
    sampling_rate : int
        Sampling rate of the signal.
    band : list, optional
        Frequency band [low_hz, high_hz]. Defaults to ``[150, np.inf]``.
        For signals sampled at 100 Hz (Nyquist = 50 Hz) this band is above
        Nyquist and will return NaN; pass a band within the Nyquist limit
        (e.g. ``[30, 50]`` for 100 Hz data).
    nperseg : int, optional
        Length of each segment for the Short-Time Fourier Transform. Default is 2048.

    Returns
    -------
    float
        Normalized power in the very high-frequency band, or NaN if the band
        contains no STFT bins.
    """
    if band is None:
        band = [150, np.inf]
    nyquist = sampling_rate / 2.0
    if band[0] >= nyquist:
        warnings.warn(
            f"vhf_norm_power_sqi: band lower bound ({band[0]} Hz) is at or above "
            f"Nyquist ({nyquist} Hz) for sampling_rate={sampling_rate}. "
            "Returning NaN. Pass a band appropriate for this sampling rate.",
            UserWarning,
            stacklevel=2,
        )
        return np.nan
    if len(signal) < nperseg:
        nperseg = len(signal)

    f, t, spec = sn.stft(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=(nperseg // 2),
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
    )

    idx = np.where((f > band[0]) & (f <= band[1]))[0]
    if len(idx) == 0:
        warnings.warn(
            f"vhf_norm_power_sqi: band {band} is above Nyquist ({sampling_rate / 2} Hz). "
            "Returning NaN."
        )
        return np.nan
    freq_marginal = np.sum(np.abs(spec[idx]), axis=0)
    peak = max(freq_marginal)
    if peak == 0:
        return np.nan
    np_vhf = float(np.median(freq_marginal) / peak)

    return np_vhf


def qrs_a_sqi(signal, sampling_rate):
    """
    QRS Amplitude SQI.

    Parameters
    ----------
    signal : array-like
        The input ECG signal.
    sampling_rate : int
        Sampling rate of the signal.

    Returns
    -------
    float
        Median peak-to-nadir amplitude difference of QRS complexes.

    Example
    -------
    >>> signal = np.random.randn(1000)
    >>> qrs_a_sqi(signal, sampling_rate=100)
    0.75
    """
    waveform = WaveformMorphology(signal, signal_type="ECG", fs=sampling_rate)
    r_peaks = waveform.r_peaks
    qrs_session = waveform.detect_qrs_session(r_peaks)
    qrs_a = np.median(
        [np.max(signal[qrs]) - np.min(signal[qrs]) for qrs in qrs_session]
    )
    return qrs_a
