"""Filtering of raw signals using bandpass filters."""

import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, lfilter, filtfilt
from scipy import signal
import logging


class BandpassFilter:
    """
    A class for bandpass filtering of signals using different filter types.

    Parameters
    ----------
    band_type : str, optional
        The type of bandpass filter to use, default is "butter".
        Options:
            - "butter": Butterworth filter
            - "cheby1": Chebyshev Type I filter
            - "cheby2": Chebyshev Type II filter
            - "ellip": Elliptic (Cauer) filter
            - "bessel": Bessel filter
    fs : int, optional
        Sampling frequency of the signal (default is 100 Hz).

    Examples
    --------
    >>> filter = BandpassFilter(band_type="butter", fs=100)
    >>> filtered_signal = filter.signal_lowpass_filter(data, cutoff=10, order=3)
    """

    def __init__(self, band_type="butter", fs=100):
        self.band_type = band_type
        self.fs = fs

    def _design_filter(self, cutoff, order, a_pass=3, rp=4, rs=40, btype="high"):
        """
        Designs a digital filter based on the specified parameters.

        Parameters
        ----------
        cutoff : float
            The cutoff frequency of the filter.
        order : int
            The order of the filter.
        a_pass : float, optional
            Passband maximum loss (only for Chebyshev Type I, default is 3).
        rp : float, optional
            Maximum ripple in the passband (only for elliptic filters, default is 4 dB).
        rs : float, optional
            Minimum stopband attenuation (only for elliptic filters, default is 40 dB).
        btype : str, optional
            Type of filter: "low", "high" (default is "high").

        Returns
        -------
        tuple
            Filter coefficients (b, a).
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq

        if normal_cutoff <= 0 or normal_cutoff >= 1:
            raise ValueError(
                "Cutoff frequency must be between 0 and Nyquist frequency."
            )

        if order <= 0:
            raise ValueError("Filter order must be a positive integer.")

        try:
            if self.band_type == "cheby1":
                b, a = cheby1(order, a_pass, normal_cutoff, btype=btype)
            elif self.band_type == "cheby2":
                b, a = cheby2(order, a_pass, normal_cutoff, btype=btype)
            elif self.band_type == "ellip":
                b, a = ellip(order, rp, rs, normal_cutoff, btype=btype)
            elif self.band_type == "bessel":
                b, a = bessel(order, normal_cutoff, btype=btype)
            elif self.band_type == "butter":
                b, a = butter(order, normal_cutoff, btype=btype)
            else:
                raise ValueError(f"Invalid band type: {self.band_type}")
        except Exception as e:
            logging.error(f"Error in filter design: {e}")
            raise ValueError("Filter design failed due to invalid parameters.")
        return b, a

    def signal_lowpass_filter(self, data, cutoff, order=3, a_pass=3, rp=4, rs=40):
        """
        Applies a low-pass filter to the input signal.

        Parameters
        ----------
        data : array_like
            The input signal to be filtered.
        cutoff : float
            The cutoff frequency of the filter.
        order : int, optional
            The order of the filter (default is 3).
        a_pass : float, optional
            Passband maximum loss (only for Chebyshev Type I, default is 3).
        rp : float, optional
            Maximum ripple in the passband (only for elliptic filters, default is 4 dB).
        rs : float, optional
            Minimum stopband attenuation (only for elliptic filters, default is 40 dB).

        Returns
        -------
        array_like
            The filtered signal.
        """
        try:
            b, a = self._design_filter(cutoff, order, a_pass, rp, rs, btype="low")
            return lfilter(b, a, data)
        except Exception as e:
            logging.error(f"Low-pass filtering failed: {e}")
            return np.array([])

    def signal_highpass_filter(self, data, cutoff, order=5, a_pass=3, rp=4, rs=40):
        """
        Applies a high-pass filter to the input signal.

        Parameters
        ----------
        data : array_like
            The input signal to be filtered.
        cutoff : float
            The cutoff frequency of the filter.
        order : int, optional
            The order of the filter (default is 5).
        a_pass : float, optional
            Passband maximum loss (only for Chebyshev Type I, default is 3).
        rp : float, optional
            Maximum ripple in the passband (only for elliptic filters, default is 4 dB).
        rs : float, optional
            Minimum stopband attenuation (only for elliptic filters, default is 40 dB).

        Returns
        -------
        array_like
            The filtered signal.
        """
        try:
            b, a = self._design_filter(cutoff, order, a_pass, rp, rs, btype="high")
            return filtfilt(b, a, data)  # Zero-phase filtering for better results
        except Exception as e:
            logging.error(f"High-pass filtering failed: {e}")
            return np.array([])
