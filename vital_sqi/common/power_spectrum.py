import numpy as np
from scipy import signal, interpolate
import pycwt as wavelet
from statsmodels.tsa.ar_model import AutoReg
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define dictionary of available wavelet types
MOTHER_WAVE_DICT = {
    "gaussian": wavelet.DOG(),
    "paul": wavelet.Paul(),
    "mexican_hat": wavelet.MexicanHat(),
}


def calculate_band_power(freq, power, fmin, fmax):
    """
    Calculate the absolute power within a specified frequency band.

    Parameters
    ----------
    freq : array-like
        Array of frequencies.
    power : array-like
        Array of power values corresponding to each frequency.
    fmin : float
        Lower bound of the frequency band.
    fmax : float
        Upper bound of the frequency band.

    Returns
    -------
    float
        Absolute power within the specified frequency band.
    """
    try:
        if power.ndim == 2:
            power = np.mean(power, axis=1)
        band_power = np.sum(power[(freq >= fmin) & (freq < fmax)]) / (
            2 * len(power) ** 2
        )
        return band_power
    except Exception as e:
        logging.error(f"Error calculating band power: {e}")
        return 0


def interpolate_rr_intervals(ts_rr, bpm_list, sampling_frequency, method="linear"):
    """
    Interpolate RR intervals to resample the heart rate data.

    Parameters
    ----------
    ts_rr : array-like
        Timestamps for RR intervals (in ms).
    bpm_list : array-like
        Heart rate values in bpm.
    sampling_frequency : int
        Sampling frequency for interpolation.
    method : str, optional
        Interpolation method, by default "linear".

    Returns
    -------
    numpy.ndarray
        Interpolated heart rate in bpm.
    """
    try:
        if len(ts_rr) < 2 or len(bpm_list) < 2:
            raise ValueError("Insufficient data points for interpolation.")

        interpolator = interpolate.interp1d(
            ts_rr, bpm_list, kind=method, fill_value="extrapolate"
        )
        time_offset = 1 / sampling_frequency
        ts_interpolated = np.arange(0, ts_rr[-1] - ts_rr[0], time_offset)
        return interpolator(ts_interpolated)
    except Exception as e:
        logging.error(f"Error in RR interval interpolation: {e}")
        return np.array([])


def compute_time_and_bpm(rr_intervals):
    """
    Generate timestamps and convert RR intervals to bpm.

    Parameters
    ----------
    rr_intervals : array-like
        List of RR intervals (in ms).

    Returns
    -------
    tuple
        Timestamps for each heartbeat (in seconds) and corresponding bpm values.
    """
    try:
        if len(rr_intervals) < 1:
            raise ValueError("RR intervals array is empty.")

        ts_rr = np.cumsum(rr_intervals) / 1000
        bpm_list = 60000 / rr_intervals  # Convert ms to bpm
        return ts_rr, bpm_list
    except Exception as e:
        logging.error(f"Error computing time and BPM from RR intervals: {e}")
        return np.array([]), np.array([])


def calculate_psd(
    rr_intervals,
    method="welch",
    hr_sampling_frequency=4,
    power_type="density",
    max_lag=3,
):
    """
    Calculate the power spectral density (PSD) from RR intervals.

    Parameters
    ----------
    rr_intervals : array-like
        List of RR intervals (in ms).
    method : str, optional
        Method to calculate PSD ('welch', 'lomb', or 'ar'), by default "welch".
    hr_sampling_frequency : int, optional
        Sampling frequency for heart rate, by default 4 Hz.
    power_type : str, optional
        Power type for AR method ('density' or 'spectrogram'), by default "density".
    max_lag : int, optional
        Maximum lag for autoregressive (AR) method, by default 3.

    Returns
    -------
    tuple
        Frequencies and power spectral density values.
    """
    try:
        ts_rr, bpm_list = compute_time_and_bpm(rr_intervals)

        if method == "welch":
            interpolated_bpm = interpolate_rr_intervals(
                ts_rr, bpm_list, hr_sampling_frequency
            )
            if interpolated_bpm.size == 0:
                raise ValueError("Interpolation failed, insufficient data points.")
            interpolated_bpm -= np.mean(interpolated_bpm)  # Normalize
            freq, psd = signal.welch(
                interpolated_bpm, fs=hr_sampling_frequency, window="hann", nfft=4096
            )

        elif method == "lomb":
            freq = np.linspace(0, hr_sampling_frequency, 256)
            angular_freq = 2 * np.pi * freq
            psd = signal.lombscargle(ts_rr, bpm_list, angular_freq, normalize=True)

        elif method == "ar":
            freq, psd_raw = signal.periodogram(
                bpm_list, fs=hr_sampling_frequency, window="boxcar", scaling=power_type
            )
            model = AutoReg(psd_raw, max_lag)
            res = model.fit()
            psd = model.predict(res.params)

        else:
            raise ValueError("Invalid method. Choose from 'welch', 'lomb', or 'ar'.")

        return freq, psd
    except Exception as e:
        logging.error(f"Error calculating PSD with method {method}: {e}")
        return np.array([]), np.array([])


def calculate_spectrogram(rr_intervals, hr_sampling_frequency=4):
    """
    Compute the spectrogram for the given RR intervals.

    Parameters
    ----------
    rr_intervals : array-like
        List of RR intervals (in ms).
    hr_sampling_frequency : int, optional
        Sampling frequency for heart rate, by default 4 Hz.

    Returns
    -------
    tuple
        Frequencies, power spectral density values, and time segments.
    """
    try:
        ts_rr, bpm_list = compute_time_and_bpm(rr_intervals)
        if len(bpm_list) < 2:
            raise ValueError("Insufficient data for spectrogram calculation.")

        freq, time_segments, psd = signal.spectrogram(
            bpm_list, fs=hr_sampling_frequency
        )
        return freq, psd, time_segments
    except Exception as e:
        logging.error(f"Error calculating spectrogram: {e}")
        return np.array([]), np.array([]), np.array([])


def calculate_wavelet_power(rr_intervals, heart_rate=4, mother_wave="morlet"):
    """
    Calculate spectral power using continuous wavelet transform (CWT).

    Parameters
    ----------
    rr_intervals : array-like
        List of RR intervals (in ms).
    heart_rate : int, optional
        Sampling rate for heart rate data, by default 4 Hz.
    mother_wave : str, optional
        Mother wavelet type ('gaussian', 'paul', 'mexican_hat', or 'morlet'), by default "morlet".

    Returns
    -------
    tuple
        Frequencies and wavelet power values.
    """
    try:
        dt = 1 / heart_rate
        if mother_wave not in MOTHER_WAVE_DICT and mother_wave != "morlet":
            logging.error(f"Invalid wavelet type: {mother_wave}")
            return np.array([]), np.array([])

        mother_wavelet = MOTHER_WAVE_DICT.get(mother_wave, wavelet.Morlet())
        wave, scales, freqs, _, _, _ = wavelet.cwt(
            rr_intervals, dt, wavelet=mother_wavelet
        )
        power = np.abs(wave) ** 2
        return freqs, power
    except Exception as e:
        logging.error(f"Error calculating wavelet power: {e}")
        return np.array([]), np.array([])
