"""R peak detection approaches for PPG and ECG."""

import numpy as np
from sklearn.cluster import KMeans
from scipy import signal
import warnings
import logging
from vital_sqi.common.band_filter import BandpassFilter
from vitalDSP.physiological_features.waveform import WaveformMorphology

# Set up logging configuration
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Detection method constants
ADAPTIVE_THRESHOLD = 1
COUNT_ORIG_METHOD = 2
CLUSTERER_METHOD = 3
SLOPE_SUM_METHOD = 4
MOVING_AVERAGE_METHOD = 5
DEFAULT = 6
BILLAUER_METHOD = 7


class PeakDetector:
    """
    Detects peaks in PPG and ECG signals using various algorithms.

    Parameters
    ----------
    wave_type : str, optional
        The type of waveform to detect peaks from, either 'PPG' or 'ECG' (default is 'PPG').
    fs : int, optional
        Sampling frequency of the signal (default is 100).

    Examples
    --------
    >>> detector = PeakDetector(wave_type="PPG", fs=100)
    >>> signal = np.random.randn(1000)
    >>> peaks, troughs = detector.ppg_detector(signal, detector_type=ADAPTIVE_THRESHOLD)
    """

    def __init__(self, wave_type="PPG", fs=100):
        if wave_type not in ["PPG", "ECG"]:
            raise ValueError("Invalid wave_type. Expected 'PPG' or 'ECG'.")
        self.wave_type = wave_type
        self.fs = fs

    def ecg_detector(self, s, get_session=False):
        """
        Detects R-peaks and other characteristic points in ECG signals.

        Parameters
        ----------
        s : array_like
            Input ECG signal.
        get_nadir : bool, optional
            Whether to detect nadir points as well (default is False).

        Returns
        -------
        tuple
            R-peaks, Q valleys, S valleys, P peaks, and T peaks.
        """
        try:
            waveform = WaveformMorphology(s, signal_type="ECG", fs=self.fs)
            r_peaks = waveform.r_peaks
            q_valleys = waveform.detect_q_valley(r_peaks)
            s_valleys = waveform.detect_s_valley(r_peaks)
            p_peaks = waveform.detect_p_peak(r_peaks, q_valleys)
            t_peaks = waveform.detect_t_peak(r_peaks, s_valleys)
            if get_session:
                ecg_session = waveform.detect_ecg_session(
                    p_peaks=p_peaks, t_peaks=t_peaks
                )
                return r_peaks, ecg_session
            return r_peaks, q_valleys, s_valleys, p_peaks, t_peaks
        except Exception as e:
            logging.error(f"ECG detection failed: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def ppg_detector(
        self,
        s,
        detector_type=DEFAULT,
        get_session=False,
        preprocess=False,
        cubing=False,
    ):
        """
        Detects peaks in PPG signals using specified detector type.

        Parameters
        ----------
        s : array_like
            Input PPG signal.
        detector_type : int, optional
            Method for peak detection (default is ADAPTIVE_THRESHOLD).
        preprocess : bool, optional
            Whether to apply filtering to the signal (default is False).
        cubing : bool, optional
            Whether to cube the signal for enhanced peak detection (default is False).

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        try:
            if detector_type not in [
                ADAPTIVE_THRESHOLD,
                COUNT_ORIG_METHOD,
                CLUSTERER_METHOD,
                SLOPE_SUM_METHOD,
                MOVING_AVERAGE_METHOD,
                DEFAULT,
                BILLAUER_METHOD,
            ]:
                raise ValueError(f"Invalid detector_type: {detector_type}")
            # Validate signal
            if len(s) == 0:
                raise ValueError("Input signal is empty.")
            if preprocess:
                filter = BandpassFilter(fs=self.fs)
                s = filter.signal_highpass_filter(s, cutoff=1, order=2)
                s = filter.signal_lowpass_filter(s, cutoff=12, order=2)
            if cubing:
                s = s**3

            if get_session:
                waveform = WaveformMorphology(s, signal_type="PPG", fs=self.fs)
                ppg_session = waveform.detect_ppg_session()
                return waveform.systolic_peaks, ppg_session

            if detector_type == CLUSTERER_METHOD:
                return self.detect_peak_trough_clusterer(s)
            elif detector_type == SLOPE_SUM_METHOD:
                return self.detect_peak_trough_slope_sum(s)
            elif detector_type == MOVING_AVERAGE_METHOD:
                return self.detect_peak_trough_moving_average_threshold(s)
            elif detector_type == COUNT_ORIG_METHOD:
                return self.detect_peak_trough_count_orig(s)
            elif detector_type == DEFAULT:
                waveform = WaveformMorphology(s, signal_type="PPG", fs=self.fs)
                return waveform.systolic_peaks, waveform.detect_troughs()
            elif detector_type == BILLAUER_METHOD:
                return self.detect_peak_trough_billauer(s)
            else:
                waveform = WaveformMorphology(s, signal_type="PPG", fs=self.fs)
                return waveform.systolic_peaks, waveform.detect_troughs()
        except Exception as e:
            logging.error(
                f"PPG detection failed with detector type {detector_type}: {e}"
            )
            return np.array([]), np.array([])

    def detect_peak_trough_slope_sum(self, s):
        """
        Detect peaks and troughs in a signal using the slope sum method.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs as lists of indices.
        """
        # Define constants and initialize lists
        fs = 100
        window_size = 12  # Approx. 128 ms at 100 Hz
        slope_sum = []
        onset_list = []
        peak_finalist = []
        trough_finalist = []

        # Compute slope sum for each sample using a sliding window
        for n in range(window_size + 1, len(s)):
            Zk = sum(max(0, s[k] - s[k - 1]) for k in range(n - window_size, n))
            slope_sum.append(Zk)
        slope_sum = np.array(slope_sum)

        # Establish adaptive threshold based on initial slope sum values
        initial_threshold = 3 * np.mean(slope_sum[: 10 * fs])
        threshold_base = initial_threshold

        # Identify threshold crossings and search for peaks and onsets
        for n, Z_value in enumerate(slope_sum):
            actual_threshold = threshold_base * 0.6
            if Z_value > actual_threshold:
                left, right = max(0, n - 15), min(len(slope_sum), n + 15)
                local_min, local_max = np.min(slope_sum[left:right]), np.max(
                    slope_sum[left:right]
                )

                if (local_max - local_min) > local_min * 2:
                    # Accept the pulse
                    onset = self.search_for_onset(slope_sum, n, local_max)
                    onset_list.append(onset)
                    threshold_base = (
                        local_max  # Update threshold base for next detection
                    )

        # Remove duplicates and sort onsets
        onset_list = np.unique(onset_list)

        # Detect peaks and troughs between consecutive onsets
        for i in range(1, len(onset_list)):
            left, right = onset_list[i - 1], onset_list[i]
            try:
                peak_finalist.append(np.argmax(s[left:right]) + left)
                trough_finalist.append(np.argmin(s[left:right]) + left)
            except ValueError as e:
                warnings.warn(f"Peak detection failed at index {i} due to {e}")

        return np.array(peak_finalist), np.array(trough_finalist)

    def detect_peak_trough_count_orig(self, s):
        """
        Detect peaks and troughs in a signal using local extrema and thresholding.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs as numpy arrays.
        """
        # Identify local extrema
        local_maxima = signal.argrelmax(s)[0]
        local_minima = signal.argrelmin(s)[0]

        # Define thresholds based on quantiles
        peak_threshold = np.quantile(s[local_maxima], 0.75) * 0.2
        trough_threshold = np.quantile(s[local_minima], 0.25) * 0.2

        # Filter extrema based on thresholds
        peak_shortlist = local_maxima[s[local_maxima] >= peak_threshold]
        trough_shortlist = local_minima[s[local_minima] <= trough_threshold]

        # Initialize lists for peaks and troughs
        peak_finalist = []
        trough_finalist = []

        # Traverse through troughs to find peaks
        left_trough = trough_shortlist[0]
        for right_trough in trough_shortlist[1:]:
            # Peaks between the current pair of troughs
            peaks_in_range = peak_shortlist[
                (peak_shortlist > left_trough) & (peak_shortlist < right_trough)
            ]

            if len(peaks_in_range) == 0:
                # No peaks found between the troughs
                left_trough = (
                    left_trough if s[left_trough] < s[right_trough] else right_trough
                )
            else:
                # Select the highest peak
                peak = peaks_in_range[np.argmax(s[peaks_in_range])]
                peak_finalist.append(peak)
                trough_finalist.append(left_trough)
                left_trough = right_trough

        # Add the final right trough
        trough_finalist.append(right_trough)

        return np.array(peak_finalist), np.array(trough_finalist)

    def detect_peak_trough_adaptive_threshold(
        self, s, adaptive_size=0.75, overlap=0, sliding=1
    ):
        """
        Detect peaks and troughs in a signal using an adaptive threshold approach.

        Parameters
        ----------
        s : array_like
            Input signal.
        adaptive_size : float, optional
            Window size for adaptive thresholding as a fraction of the sampling rate (default is 0.75).
        overlap : float, optional
            Overlapping ratio for the sliding window (default is 0).
        sliding : int, optional
            Step size for sliding window (default is 1).

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        adaptive_window = int(adaptive_size * self.fs)
        adaptive_threshold = self.get_moving_average(s, int(adaptive_window * 2 + 1))

        start_ROIs, end_ROIs = self.get_ROI(s, adaptive_threshold)

        # Detect peaks within the regions of interest
        peak_finalist = [
            np.argmax(s[start_ROI : end_ROI + 1]) + start_ROI
            for start_ROI, end_ROI in zip(start_ROIs, end_ROIs)
        ]

        # Detect troughs between the peaks
        trough_finalist = [
            np.argmin(s[peak_finalist[idx] : peak_finalist[idx + 1]])
            + peak_finalist[idx]
            for idx in range(len(peak_finalist) - 1)
        ]

        return np.array(peak_finalist), np.array(trough_finalist)

    def get_ROI(self, s, adaptive_threshold, margin=0.1):
        """
        Identify regions of interest (ROIs) in the signal where peaks or troughs are likely to occur.

        Parameters
        ----------
        s : array_like
            Input signal.
        adaptive_threshold : array_like
            Adaptive threshold values for the signal.
        margin : float, optional
            Margin (fraction of the signal range) to include before and after the ROI (default is 0.1).

        Returns
        -------
        tuple
            Two lists: start_ROIs and end_ROIs, which contain the start and end indices of the ROIs.

        Notes
        -----
        - ROIs are defined as contiguous regions where the signal exceeds the adaptive threshold.
        - Margins can be added to widen the ROIs for more inclusive peak/trough detection.

        Example
        -------
        >>> s = [0, 1, 3, 7, 5, 2, 0, 6, 9, 8, 4, 1, 0]
        >>> adaptive_threshold = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        >>> peak_detector = PeakDetection(fs=100)
        >>> start_ROIs, end_ROIs = peak_detector.get_ROI(s, adaptive_threshold)
        """
        # Identify regions where the signal exceeds the adaptive threshold
        above_threshold = s > adaptive_threshold

        # Find the start and end indices of these regions
        start_ROIs = []
        end_ROIs = []

        for i in range(1, len(above_threshold)):
            # Transition from below threshold to above threshold: start of an ROI
            if above_threshold[i] and not above_threshold[i - 1]:
                start_ROIs.append(i)

            # Transition from above threshold to below threshold: end of an ROI
            if not above_threshold[i] and above_threshold[i - 1]:
                end_ROIs.append(i - 1)

        # Handle case where the signal ends above the threshold
        if above_threshold[-1]:
            end_ROIs.append(len(s) - 1)

        # Apply margin to widen the ROIs
        signal_length = len(s)
        start_ROIs = [
            max(0, int(start - margin * signal_length)) for start in start_ROIs
        ]
        end_ROIs = [
            min(signal_length - 1, int(end + margin * signal_length))
            for end in end_ROIs
        ]

        return start_ROIs, end_ROIs

    def detect_peak_trough_moving_average_threshold(self, s):
        """
        Detects peaks using a moving average threshold approach.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        try:
            w1 = int(0.12 * self.fs)  # Shorter moving average window
            w2 = int(0.67 * self.fs)  # Longer moving average window

            ma_peak = self.get_moving_average(s, w1)
            ma_beat = self.get_moving_average(s, w2)

            # Align moving averages with the signal
            offset = len(s) - len(ma_peak)
            if offset > 0:
                ma_peak = np.pad(
                    ma_peak, (offset // 2, offset - offset // 2), mode="edge"
                )
            elif offset < 0:
                ma_peak = ma_peak[: len(s)]

            threshold = ma_beat + 0.02 * np.mean(s)

            peaks = np.where(
                (ma_peak > threshold[:-1]) & (ma_peak[1:] <= ma_peak[:-1])
            )[0]
            return np.array(peaks), np.array([])
        except Exception as e:
            logging.error(f"Moving average threshold-based detection failed: {e}")
            return np.array([]), np.array([])

    def get_moving_average(self, q, w):
        """
        Calculates the moving average of a sequence.

        Parameters
        ----------
        q : array_like
            Input sequence.
        w : int
            Window size for the moving average.

        Returns
        -------
        array_like
            Moving average of the sequence.
        """
        if w <= 0:
            raise ValueError("Window size must be greater than 0.")

        try:
            q_padded = np.pad(q, (w // 2, w - 1 - w // 2), mode="edge")
            return np.convolve(q_padded, np.ones(w) / w, "valid")
        except Exception as e:
            logging.error(f"Moving average calculation failed: {e}")
            return np.array([])

    def detect_peak_trough_clusterer(self, s):
        """
        Detects peaks and troughs in a signal using a clustering technique.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        try:
            local_maxima = signal.argrelmax(s)[0]
            local_minima = signal.argrelmin(s)[0]

            def cluster_extrema(extrema):
                features = np.vstack([s[extrema], np.gradient(s)[extrema]]).T
                kmeans = KMeans(n_clusters=2)
                labels = kmeans.fit_predict(features)
                cluster = labels[np.argmax(s[extrema])]  # Cluster with higher peak
                return extrema[labels == cluster]

            peaks = cluster_extrema(local_maxima)
            troughs = cluster_extrema(local_minima)
            return peaks, troughs
        except Exception as e:
            logging.error(f"Clustering-based peak/trough detection failed: {e}")
            return np.array([]), np.array([])

    def detect_peak_trough_DEFAULT(self, s):
        """
        Detects peaks and troughs using SciPy's `find_peaks` function.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        try:
            peaks = signal.find_peaks(s)[0]
            troughs = [
                np.argmin(s[peaks[i] : peaks[i + 1]]) + peaks[i]
                for i in range(len(peaks) - 1)
            ]
            return peaks, troughs
        except Exception as e:
            logging.error(f"SciPy peak/trough detection failed: {e}")
            return np.array([]), np.array([])

    def detect_peak_trough_billauer(self, s, delta=0.8):
        """
        Billauer's method for peak and trough detection, translated from MATLAB.

        Parameters
        ----------
        s : array_like
            Input signal.
        delta : float, optional
            Minimum difference required to consider a peak (default is 0.8).

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        try:
            maxtab, mintab = [], []
            mn, mx = np.Inf, -np.Inf
            look_for_max = True
            for i in range(len(s)):
                if s[i] > mx:
                    mx = s[i]
                if s[i] < mn:
                    mn = s[i]
                if look_for_max:
                    if s[i] < mx - delta:
                        maxtab.append(i)
                        mn = s[i]
                        look_for_max = False
                else:
                    if s[i] > mn + delta:
                        mintab.append(i)
                        mx = s[i]
                        look_for_max = True
            return np.array(maxtab), np.array(mintab)
        except Exception as e:
            logging.error(f"Billauer method-based detection failed: {e}")
            return np.array([]), np.array([])
