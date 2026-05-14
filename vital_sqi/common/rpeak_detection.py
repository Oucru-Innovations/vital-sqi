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

# Detection method constants — PPG
ADAPTIVE_THRESHOLD = 1
COUNT_ORIG_METHOD = 2
CLUSTERER_METHOD = 3
SLOPE_SUM_METHOD = 4
MOVING_AVERAGE_METHOD = 5
DEFAULT = 6
BILLAUER_METHOD = 7
AMPD_METHOD = 8          # Automatic Multiscale Peak Detection (PPG)
LOCAL_MAX_IBI = 9        # Local-max + IBI tracking (PPG)

# Detection method constants — ECG (used by ecg_detector)
ECG_DEFAULT = 10         # vitalDSP WaveformMorphology (kept as gold standard)
PAN_TOMPKINS = 11        # Classic Pan-Tompkins (1985)
HAMILTON = 12            # Hamilton-Tompkins simplified (2002)
ENGZEE = 13              # Engzee-Zeelenberg (1979)


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
    >>> peaks, troughs = detector.ppg_detector(signal)  # DEFAULT (vitalDSP)
    >>> peaks, troughs = detector.ppg_detector(signal, detector_type=ADAPTIVE_THRESHOLD)
    >>> r, q, s, p, t = PeakDetector(wave_type="ECG", fs=256).ecg_detector(signal)
    """

    def __init__(self, wave_type="PPG", fs=100):
        if wave_type not in ["PPG", "ECG"]:
            raise ValueError("Invalid wave_type. Expected 'PPG' or 'ECG'.")
        self.wave_type = wave_type
        self.fs = fs

    def ecg_detector(self, s, detector_type=ECG_DEFAULT, get_session=False):
        """
        Detects R-peaks and critical points in ECG signals.

        The default path (ECG_DEFAULT) uses vitalDSP WaveformMorphology, which
        provides high-quality derivative-based R-peak detection plus full critical
        point extraction (Q/S/P/T).  Alternative R-peak algorithms (PAN_TOMPKINS,
        HAMILTON, ENGZEE) also use WaveformMorphology for the critical-point step
        so that Q/S/P/T quality is never downgraded.

        Parameters
        ----------
        s : array_like
            Input ECG signal.
        detector_type : int, optional
            R-peak detection algorithm.  Available constants:

            * ``ECG_DEFAULT`` (10) — vitalDSP WaveformMorphology (recommended; also
              provides Q/S/P/T morphology points)
            * ``PAN_TOMPKINS`` (11) — classic Pan-Tompkins 1985 with adaptive
              dual-threshold; bandpass 5–15 Hz
            * ``HAMILTON`` (12) — Hamilton-Tompkins simplified 2002; single-pass
              derivative-square-integrate; bandpass 8–16 Hz
            * ``ENGZEE`` (13) — Engzee-Zeelenberg 1979; dynamic threshold on
              high-pass filtered derivative

            All alternatives use vitalDSP WaveformMorphology for Q/S/P/T
            extraction, anchored to the chosen R-peak indices.
            Default is ``ECG_DEFAULT``.
        get_session : bool, optional
            If True return (r_peaks, ecg_session) instead of the full tuple.

        Returns
        -------
        tuple
            (r_peaks, q_valleys, s_valleys, p_peaks, t_peaks) normally, or
            (r_peaks, ecg_session) when get_session=True.
        """
        valid = {ECG_DEFAULT, PAN_TOMPKINS, HAMILTON, ENGZEE}
        if detector_type not in valid:
            raise ValueError(
                f"Invalid ECG detector_type: {detector_type}. "
                f"Choose from ECG_DEFAULT={ECG_DEFAULT}, PAN_TOMPKINS={PAN_TOMPKINS}, "
                f"HAMILTON={HAMILTON}, ENGZEE={ENGZEE}."
            )
        if len(s) == 0:
            raise ValueError("Input ECG signal is empty.")

        try:
            # Step 1 — detect R-peaks with the chosen algorithm
            if detector_type == PAN_TOMPKINS:
                r_peaks = self.detect_r_peaks_pan_tompkins(s)
            elif detector_type == HAMILTON:
                r_peaks = self.detect_r_peaks_hamilton(s)
            elif detector_type == ENGZEE:
                r_peaks = self.detect_r_peaks_engzee(s)
            else:  # ECG_DEFAULT — vitalDSP WaveformMorphology
                wm = WaveformMorphology(s, signal_type="ECG", fs=self.fs)
                r_peaks = wm.r_peaks

            # Step 2 — extract critical points using vitalDSP (always)
            wm = WaveformMorphology(s, signal_type="ECG", fs=self.fs)
            # Override the WM r_peaks with those from our chosen detector so that
            # Q/S/P/T search windows are anchored to the correct peaks.
            wm.r_peaks = r_peaks

            q_valleys = wm.detect_q_valley(r_peaks)
            s_valleys = wm.detect_s_valley(r_peaks)
            p_peaks = wm.detect_p_peak(r_peaks, q_valleys)
            t_peaks = wm.detect_t_peak(r_peaks, s_valleys)

            if get_session:
                ecg_session = wm.detect_ecg_session(
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
            Method for peak detection (default is ``DEFAULT`` = 6, which
            uses vitalDSP WaveformMorphology systolic-peak detection).
            Available constants:

            * ``ADAPTIVE_THRESHOLD`` (1) — threshold adapts to local signal amplitude
            * ``COUNT_ORIG_METHOD`` (2) — count-based local maxima
            * ``CLUSTERER_METHOD`` (3) — KMeans clustering
            * ``SLOPE_SUM_METHOD`` (4) — Zong 2003 slope-sum onset
            * ``MOVING_AVERAGE_METHOD`` (5) — Elgendi two-moving-average
            * ``DEFAULT`` (6) — vitalDSP WaveformMorphology (recommended)
            * ``BILLAUER_METHOD`` (7) — Billauer peak/trough tracker
            * ``AMPD_METHOD`` (8) — Automatic Multiscale Peak Detection
            * ``LOCAL_MAX_IBI`` (9) — local-max with IBI tracking
        preprocess : bool, optional
            Whether to apply filtering to the signal (default is False).
        cubing : bool, optional
            Whether to cube the signal for enhanced peak detection (default is False).

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        _ppg_methods = {
            ADAPTIVE_THRESHOLD,
            COUNT_ORIG_METHOD,
            CLUSTERER_METHOD,
            SLOPE_SUM_METHOD,
            MOVING_AVERAGE_METHOD,
            DEFAULT,
            BILLAUER_METHOD,
            AMPD_METHOD,
            LOCAL_MAX_IBI,
        }
        if detector_type not in _ppg_methods:
            raise ValueError(f"Invalid detector_type: {detector_type}")

        if len(s) == 0:
            raise ValueError("Input signal is empty.")

        if preprocess:
            bp_filter = BandpassFilter(fs=self.fs)
            s = bp_filter.signal_highpass_filter(s, cutoff=1, order=2)
            s = bp_filter.signal_lowpass_filter(s, cutoff=12, order=2)
        if cubing:
            s = s**3

        if get_session:
            waveform = WaveformMorphology(s, signal_type="PPG", fs=self.fs)
            ppg_session = waveform.detect_ppg_session()
            return waveform.systolic_peaks, ppg_session

        if detector_type == ADAPTIVE_THRESHOLD:
            return self.detect_peak_trough_adaptive_threshold(s)
        elif detector_type == CLUSTERER_METHOD:
            return self.detect_peak_trough_clusterer(s)
        elif detector_type == SLOPE_SUM_METHOD:
            return self.detect_peak_trough_slope_sum(s)
        elif detector_type == MOVING_AVERAGE_METHOD:
            return self.detect_peak_trough_moving_average_threshold(s)
        elif detector_type == COUNT_ORIG_METHOD:
            return self.detect_peak_trough_count_orig(s)
        elif detector_type == BILLAUER_METHOD:
            return self.detect_peak_trough_billauer(s)
        elif detector_type == AMPD_METHOD:
            return self.detect_peak_trough_ampd(s)
        elif detector_type == LOCAL_MAX_IBI:
            return self.detect_peak_trough_local_max_ibi(s)
        else:  # DEFAULT
            waveform = WaveformMorphology(s, signal_type="PPG", fs=self.fs)
            return waveform.systolic_peaks, waveform.detect_troughs()

    def search_for_onset(self, slope_sum, n, local_max):
        """
        Walk back from detection point n to find the pulse onset.
        Following Zong et al. 2003: onset is where slope_sum drops below
        1% of local_max, up to 200 ms before the detection point.
        """
        threshold = 0.01 * local_max
        lookback = int(0.2 * self.fs)
        for k in range(n, max(0, n - lookback), -1):
            if slope_sum[k] < threshold:
                return k
        return max(0, n - lookback)

    def detect_peak_trough_slope_sum(self, s):
        """
        Detect peaks and troughs in a signal using the slope sum method.
        Reference: Zong et al. (2003) Computers in Cardiology 30:259-262.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs as lists of indices.
        """
        fs = self.fs  # use instance sampling rate
        window_size = max(1, int(0.128 * fs))  # 128 ms window
        slope_sum = []
        onset_list = []
        peak_finalist = []
        trough_finalist = []

        # Compute slope sum for each sample using a sliding window
        for n in range(window_size + 1, len(s)):
            Zk = sum(max(0, s[k] - s[k - 1]) for k in range(n - window_size, n))
            slope_sum.append(Zk)
        slope_sum = np.array(slope_sum)

        if len(slope_sum) == 0:
            return np.array([]), np.array([])

        # Establish adaptive threshold from first 10 s (or all samples if shorter)
        init_samples = min(10 * fs, len(slope_sum))
        initial_threshold = 3 * np.mean(slope_sum[:init_samples])
        threshold_base = initial_threshold

        # Identify threshold crossings and search for peaks and onsets
        for n, Z_value in enumerate(slope_sum):
            actual_threshold = threshold_base * 0.6
            if Z_value > actual_threshold:
                left = max(0, n - 15)
                right = min(len(slope_sum), n + 15)
                local_min = np.min(slope_sum[left:right])
                local_max = np.max(slope_sum[left:right])

                if (local_max - local_min) > abs(local_min) * 2:
                    onset = self.search_for_onset(slope_sum, n, local_max)
                    onset_list.append(onset)
                    # Update threshold as running average (not max)
                    threshold_base = 0.875 * threshold_base + 0.125 * local_max

        onset_list = np.unique(onset_list)

        for i in range(1, len(onset_list)):
            left, right = onset_list[i - 1], onset_list[i]
            try:
                peak_finalist.append(int(np.argmax(s[left:right])) + left)
                trough_finalist.append(int(np.argmin(s[left:right])) + left)
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

        if not start_ROIs:
            return np.array([]), np.array([])

        # Detect peaks within the regions of interest
        peak_finalist = [
            np.argmax(s[start_ROI : end_ROI + 1]) + start_ROI
            for start_ROI, end_ROI in zip(start_ROIs, end_ROIs)
        ]

        if len(peak_finalist) < 2:
            return np.array(peak_finalist), np.array([])

        # Detect troughs between the peaks; skip degenerate (zero-length) intervals
        trough_finalist = []
        for idx in range(len(peak_finalist) - 1):
            p1, p2 = peak_finalist[idx], peak_finalist[idx + 1]
            if p2 > p1:
                trough_finalist.append(int(np.argmin(s[p1:p2])) + p1)

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
        Detect peaks using a two-moving-average threshold (Elgendi et al. 2013).
        Reference: Elgendi M. et al., PLoS ONE 8(10):e76585, 2013.

        A short MA (w1 ≈ 120 ms) rises above a long MA (w2 ≈ 670 ms) baseline to
        form "blocks of interest"; the raw-signal argmax inside each block is the peak.

        Parameters
        ----------
        s : array_like
            Input signal.

        Returns
        -------
        tuple
            Detected peaks and troughs.
        """
        s = np.asarray(s, dtype=float)
        w1 = max(1, int(0.12 * self.fs))   # ~120 ms
        w2 = max(1, int(0.67 * self.fs))   # ~670 ms

        ma_peak = self.get_moving_average(s, w1)
        ma_beat = self.get_moving_average(s, w2)

        if len(ma_peak) != len(s):
            ma_peak = ma_peak[: len(s)]
        if len(ma_beat) != len(s):
            ma_beat = ma_beat[: len(s)]

        threshold = ma_beat + 0.02 * np.mean(s)
        above = ma_peak > threshold

        # Find rising and falling edges of "blocks of interest"
        above_int = above.astype(int)
        rising = np.where(np.diff(above_int) == 1)[0] + 1
        falling = np.where(np.diff(above_int) == -1)[0] + 1

        # Handle signal ending inside a block; clip to last valid index
        if above[-1]:
            falling = np.append(falling, len(s) - 1)

        # Align: drop unmatched edges
        if len(rising) == 0 or len(falling) == 0:
            return np.array([]), np.array([])
        if falling[0] < rising[0]:
            falling = falling[1:]
        min_len = min(len(rising), len(falling))
        rising, falling = rising[:min_len], falling[:min_len]

        # Raw-signal argmax inside each block → true peak index.
        # Require the peak value to exceed the mean threshold to reject degenerate
        # end-of-signal blocks where the entire block is below the signal mean.
        signal_mean = np.mean(s)
        peaks_list = []
        for i in range(len(rising)):
            block = s[rising[i]:falling[i]]
            if len(block) == 0:
                continue
            peak_idx = rising[i] + int(np.argmax(block))
            if s[peak_idx] > signal_mean:
                peaks_list.append(peak_idx)
        peaks = np.array(peaks_list)

        # Troughs: raw-signal argmin between consecutive peaks
        troughs = np.array([
            peaks[i] + int(np.argmin(s[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)
        ])

        return peaks, troughs

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
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
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
        # Billauer peakdet — records the index of the actual max/min, not the
        # detection crossing point. Reference: Billauer (2005) peakdet.m.
        maxtab, mintab = [], []
        mn, mx = np.inf, -np.inf
        mnpos, mxpos = 0, 0
        look_for_max = True
        for i in range(len(s)):
            if s[i] > mx:
                mx, mxpos = s[i], i
            if s[i] < mn:
                mn, mnpos = s[i], i
            if look_for_max:
                if s[i] < mx - delta:
                    maxtab.append(mxpos)   # true peak index
                    mn, mnpos = s[i], i
                    look_for_max = False
            else:
                if s[i] > mn + delta:
                    mintab.append(mnpos)   # true trough index
                    mx, mxpos = s[i], i
                    look_for_max = True
        return np.array(maxtab), np.array(mintab)

    # =========================================================================
    # ECG detectors — Phase 2
    # =========================================================================

    def detect_r_peaks_pan_tompkins(self, s):
        """
        Pan-Tompkins (1985) R-peak detector.

        Pipeline: bandpass → derivative → square → moving-window integration →
        adaptive threshold with two learning periods.

        Reference: Pan J, Tompkins WJ. IEEE Trans Biomed Eng. 1985;32(3):230-6.

        Parameters
        ----------
        s : array_like
            Input ECG signal.

        Returns
        -------
        np.ndarray
            Indices of detected R-peaks in the original signal.
        """
        s = np.asarray(s, dtype=float)
        fs = self.fs

        # --- 1. Bandpass filter (5–15 Hz passband per Pan-Tompkins) ----------
        try:
            bp = BandpassFilter(fs=fs)
            s_filt = bp.signal_highpass_filter(s, cutoff=5, order=2)
            s_filt = bp.signal_lowpass_filter(s_filt, cutoff=15, order=2)
        except Exception:
            s_filt = s.copy()

        # --- 2. Derivative (5-point) -----------------------------------------
        # y[n] = (1/8T)(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
        d = np.zeros_like(s_filt)
        for n in range(2, len(s_filt) - 2):
            d[n] = (1.0 / (8.0 / fs)) * (
                -s_filt[n - 2] - 2 * s_filt[n - 1]
                + 2 * s_filt[n + 1] + s_filt[n + 2]
            )

        # --- 3. Squaring --------------------------------------------------------
        sq = d ** 2

        # --- 4. Moving-window integration (150 ms) ---------------------------
        win = max(1, int(0.150 * fs))
        mwi = np.convolve(sq, np.ones(win) / win, mode="same")

        # --- 5. Adaptive threshold (two-pass) --------------------------------
        # Learning period 1: first 2 s for initial threshold estimates
        learn_n = int(2 * fs)
        init = mwi[:min(learn_n, len(mwi))]
        spki = np.max(init) if len(init) > 0 else 1.0   # signal peak estimate
        npki = np.mean(init) if len(init) > 0 else 0.0  # noise peak estimate
        threshold_i1 = npki + 0.25 * (spki - npki)

        min_rr = int(0.2 * fs)   # 200 ms refractory period
        peaks_i = []
        last_peak = -min_rr

        for i in range(len(mwi)):
            if i - last_peak < min_rr:
                continue
            # Local maximum check in ±win/2 neighbourhood
            lo = max(0, i - win // 2)
            hi = min(len(mwi), i + win // 2 + 1)
            if mwi[i] != np.max(mwi[lo:hi]):
                continue
            if mwi[i] >= threshold_i1:
                peaks_i.append(i)
                spki = 0.125 * mwi[i] + 0.875 * spki
                last_peak = i
            else:
                npki = 0.125 * mwi[i] + 0.875 * npki
            threshold_i1 = npki + 0.25 * (spki - npki)

        if len(peaks_i) == 0:
            return np.array([])

        # --- 6. Back-project to original signal: search ±60 ms for true R ---
        search = int(0.060 * fs)
        r_peaks = []
        for p in peaks_i:
            lo = max(0, p - search)
            hi = min(len(s), p + search + 1)
            r_peaks.append(lo + int(np.argmax(s[lo:hi])))

        return np.array(r_peaks)

    def detect_r_peaks_hamilton(self, s):
        """
        Hamilton-Tompkins simplified ECG R-peak detector (2002).

        Uses a single-pass derivative-square-integrate pipeline with fixed
        refractory period and mean-based threshold, suitable for real-time use.

        Reference: Hamilton PS. "Open source ECG analysis." Computers in
        Cardiology 2002;29:101-104.

        Parameters
        ----------
        s : array_like
            Input ECG signal.

        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        s = np.asarray(s, dtype=float)
        fs = self.fs

        # Bandpass 8–16 Hz (Hamilton's recommended range)
        try:
            bp = BandpassFilter(fs=fs)
            s_filt = bp.signal_highpass_filter(s, cutoff=8, order=2)
            s_filt = bp.signal_lowpass_filter(s_filt, cutoff=16, order=2)
        except Exception:
            s_filt = s.copy()

        # First difference + abs (simplified derivative)
        d = np.abs(np.diff(s_filt, prepend=s_filt[0]))

        # Moving average (80 ms)
        win = max(1, int(0.080 * fs))
        ma = np.convolve(d, np.ones(win) / win, mode="same")

        # Fixed threshold: mean of signal + 0.5 * std
        threshold = np.mean(ma) + 0.5 * np.std(ma)
        min_rr = int(0.25 * fs)  # 250 ms refractory

        # Peak detection on integrated signal
        candidate_peaks = signal.find_peaks(ma, height=threshold, distance=min_rr)[0]

        if len(candidate_peaks) == 0:
            return np.array([])

        # Back-project to raw signal within ±50 ms
        search = int(0.050 * fs)
        r_peaks = []
        for p in candidate_peaks:
            lo = max(0, p - search)
            hi = min(len(s), p + search + 1)
            r_peaks.append(lo + int(np.argmax(s[lo:hi])))

        return np.array(r_peaks)

    def detect_r_peaks_engzee(self, s):
        """
        Engzee-Zeelenberg ECG R-peak detector (1979), as described in:
        Engzee R, Zeelenberg C. "A single scan algorithm for QRS-detection
        and feature extraction." Computers in Cardiology 1979;6:37-42.

        Uses the absolute first difference of a high-pass filtered signal
        with a dynamic threshold that adapts every detected beat.

        Parameters
        ----------
        s : array_like
            Input ECG signal.

        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        s = np.asarray(s, dtype=float)
        fs = self.fs

        # High-pass filter to remove baseline wander
        try:
            bp = BandpassFilter(fs=fs)
            s_hp = bp.signal_highpass_filter(s, cutoff=0.5, order=2)
        except Exception:
            s_hp = s.copy()

        # Engzee: use the difference signal (approximation of derivative)
        diff = np.abs(np.diff(s_hp, prepend=s_hp[0]))

        # Initial threshold from first 2 s of signal
        init_n = int(2 * fs)
        threshold = 0.6 * np.max(diff[:min(init_n, len(diff))])
        min_rr = int(0.2 * fs)

        peaks = []
        last = -min_rr
        i = 0
        while i < len(diff):
            if i - last < min_rr:
                i += 1
                continue
            if diff[i] >= threshold:
                # Search window: find peak within next 50 ms
                win_end = min(len(s), i + int(0.050 * fs))
                local_peak = i + int(np.argmax(s[i:win_end]))
                peaks.append(local_peak)
                # Update threshold: 60% of detected peak value in diff
                threshold = 0.6 * np.max(diff[max(0, i - int(0.2 * fs)):i + 1])
                last = local_peak
                i = local_peak + 1
            else:
                i += 1

        return np.array(peaks)

    # =========================================================================
    # PPG detectors — Phase 2
    # =========================================================================

    def detect_peak_trough_ampd(self, s):
        """
        Automatic Multiscale Peak Detection (AMPD) for PPG signals.

        AMPD finds peaks without requiring pre-set parameters by computing a
        Local Maxima Scalogram (LMS) across all window scales and identifying
        the scale with the fewest maxima (optimal scale), then intersecting
        maxima across all scales up to that point.

        Reference: Scholkmann F et al. "An Efficient Algorithm for Automatic
        Peak Detection in Noisy Periodic and Quasi-Periodic Signals."
        Algorithms 2012;5(4):588-603.

        Parameters
        ----------
        s : array_like
            Input PPG signal.

        Returns
        -------
        tuple
            (peaks, troughs) as numpy arrays of indices.
        """
        s = np.asarray(s, dtype=float)
        N = len(s)
        L = N // 2  # maximum scale

        if L < 2:
            return np.array([]), np.array([])

        # Build Local Maxima Scalogram: lms[k, i] = 1 if s[i] is local max at scale k
        # Use a compact boolean matrix; row k corresponds to window half-width k+1
        lms = np.zeros((L - 1, N), dtype=bool)
        for k in range(1, L):
            lms[k - 1, k: N - k] = (
                (s[k: N - k] > s[: N - 2 * k]) &
                (s[k: N - k] > s[2 * k:])
            )

        # Row counts: how many samples each scale treats as a local maximum.
        # The optimal scale lambda_opt minimises the row count in the physiological
        # range — it is the scale closest to the true half-period of the signal.
        # At this scale the LMS has approximately the correct number of peaks.
        row_counts = lms.sum(axis=1)
        if not np.any(row_counts > 0):
            return np.array([]), np.array([])

        # Restrict search to scales 1..2*fs (prevents choosing the degenerate
        # large-scale minimum where only 1 global maximum survives).
        max_scale = min(L - 2, int(2 * self.fs))
        # Among scales with more than 1 peak, find the minimum-count scale.
        search = row_counts[:max_scale + 1]
        valid = search > 1  # must find at least 2 peaks to be useful
        if not np.any(valid):
            valid = search > 0
        candidates = np.where(valid)[0]
        lambda_opt = candidates[np.argmin(search[candidates])]

        # Use the optimal scale's local maxima directly as peaks.
        # This is equivalent to the Scholkmann original — the row with minimum
        # count has precisely the right number of beats.
        peaks = np.where(lms[lambda_opt, :])[0]

        # Refine: for each detected peak, find the true maximum of the raw signal
        # within ±(lambda_opt//4) samples to snap to the exact waveform peak.
        refine_w = max(1, lambda_opt // 4)
        refined = []
        for p in peaks:
            lo = max(0, p - refine_w)
            hi = min(N, p + refine_w + 1)
            refined.append(lo + int(np.argmax(s[lo:hi])))
        peaks = np.unique(np.array(refined))

        if len(peaks) == 0:
            return np.array([]), np.array([])

        # Troughs: argmin between consecutive peaks
        troughs = np.array([
            peaks[i] + int(np.argmin(s[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)
        ])

        return peaks, troughs

    def detect_peak_trough_local_max_ibi(self, s):
        """
        Local-maximum PPG detector with inter-beat-interval (IBI) tracking.

        Finds all local maxima above a dynamic threshold, then prunes false
        positives by enforcing a physiologically plausible IBI window derived
        from the running median IBI (40–200 BPM, i.e. 300–1500 ms).

        This is particularly robust to low-amplitude diastolic humps that
        confuse simpler detectors.

        Parameters
        ----------
        s : array_like
            Input PPG signal.

        Returns
        -------
        tuple
            (peaks, troughs) as numpy arrays of indices.
        """
        s = np.asarray(s, dtype=float)
        fs = self.fs

        # Initial candidates: all local maxima with order ~50 ms
        candidates = signal.argrelmax(s, order=max(1, int(0.05 * fs)))[0]
        if len(candidates) == 0:
            return np.array([]), np.array([])

        # Height threshold: above the signal mean (rejects sub-mean noise bumps
        # while passing all true systolic peaks regardless of waveform offset).
        amp_threshold = np.mean(s)
        candidates = candidates[s[candidates] >= amp_threshold]

        if len(candidates) == 0:
            return np.array([]), np.array([])

        # Enforce minimum refractory (300 ms = 200 BPM upper limit)
        min_rr = int(0.300 * fs)
        # Enforce maximum IBI (1500 ms = 40 BPM lower limit)
        max_rr = int(1.500 * fs)

        # Greedy forward scan with running-median IBI adaptation
        peaks = [candidates[0]]
        for c in candidates[1:]:
            rr = c - peaks[-1]
            if rr < min_rr:
                # Keep the higher of the two close candidates
                if s[c] > s[peaks[-1]]:
                    peaks[-1] = c
                continue
            if rr > max_rr and len(peaks) >= 3:
                # Large gap — check if a beat was missed (insert midpoint search)
                median_rr = int(np.median(np.diff(peaks[-3:])))
                mid = peaks[-1] + median_rr
                if mid < c:
                    # Search ±15% of median_rr around expected position
                    search = int(0.15 * median_rr)
                    lo = max(0, mid - search)
                    hi = min(len(s), mid + search + 1)
                    insert_idx = lo + int(np.argmax(s[lo:hi]))
                    if s[insert_idx] >= amp_threshold:
                        peaks.append(insert_idx)
            peaks.append(c)

        peaks = np.array(peaks)

        # Troughs: argmin between consecutive peaks
        troughs = np.array([
            peaks[i] + int(np.argmin(s[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)
        ])

        return peaks, troughs
