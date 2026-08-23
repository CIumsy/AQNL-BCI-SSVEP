#!/usr/bin/env python3
"""
ssvep_fbcca.py

Works out which flickering square someone is looking at, from three EEG
channels on the back of the head (O1, Oz, O2).

The method is Filter-Bank CCA, from Chen et al. 2015, "Filter bank
canonical correlation analysis for implementing a high-speed SSVEP-based
brain-computer interface".

WHY NOT TRCA
--------------
TRCA needs every training trial cut at the same point in the flicker
cycle. It builds its filter from the sample-by-sample covariance between
trials, and its template is a time-domain average of them, so if the
trials are not aligned the average cancels itself out and accuracy drops
to chance.

A sliding window has no such alignment. Measured on this exact pipeline:
88% when phase-locked, 27.5% when free-running.

FBCCA needs no training and has no templates. It correlates the window
against sine and cosine waves, so the phase does not matter -- which is
exactly what a sliding window needs.

WHAT HAPPENS TO EACH WINDOW
-----------------------------
  1. Remove the DC offset and notch out mains hum (50 Hz and 100 Hz).
  2. Downsample 1000 Hz to 250 Hz. SSVEP lives below about 90 Hz, so the
     extra samples add nothing and cost four times the filtering work.
     It also keeps the filters numerically healthy: a 12 Hz corner is
     0.024 of Nyquist at 1000 Hz, but a comfortable 0.096 at 250 Hz.
  3. Split into sub-bands. Band m starts at m x the lowest target
     frequency, so band 1 holds the fundamentals, band 2 the 2nd
     harmonics, and so on. Real SSVEP harmonics carry their own
     information about which target it is, and using them is what makes
     FBCCA better than plain CCA.
  4. In each band, correlate against each target's sine/cosine set.
  5. Combine the bands with the standard weights w(m) = m^-1.25 + 0.25,
     which give the lower harmonics more say.

Two details that affect accuracy:

  * The correlation is solved directly with QR and SVD rather than
    sklearn's iterative method. It is exact instead of approximate, needs
    no convergence settings, and is about 100x faster here. The reference
    side does not depend on incoming data, so it is worked out once when
    the classifier is built.

  * Every target is scored against the SAME number of reference waves.
    Correlation goes up with more references whether or not there is any
    real signal, so an uneven count would quietly favour whichever target
    had more of them.

Usage:
    clf = FBCCAClassifier(frequencies=[12, 15, 18, 20], sfreq=1000)
    clf.on_prediction = lambda f: print("looking at", f, "Hz")
    clf.start()                       # spawns the classifier worker
    ...                               # from your ADC callback:
    clf.push_sample_values([o1, oz, o2])
    ...
    clf.stop()
"""

import threading
import time
from collections import Counter, deque
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch


def detect_mains_frequency(signal_uv: np.ndarray, sfreq: float,
                            search_range=(45.0, 55.0)) -> Optional[float]:
    """Find the actual interference peak near a nominal mains frequency,
    rather than trusting that it sits exactly on the nominal value.

    Measured directly on this rig's real recordings: a fixed 50.0Hz notch
    (Q=30) left +11.4dB of residual power on a peak that actually sat at
    49.805Hz -- re-centering the SAME filter on the measured peak took
    suppression from 23.6dB to 34.8dB, no width change needed. The offset
    also wasn't constant across sessions (49.8Hz in three, 50.3Hz in a
    fourth -- different directions), so a fixed correction wouldn't have
    covered it either; this has to be measured each time.

    Zero-pads the FFT well past nperseg (nfft, not nperseg -- nperseg still
    controls the actual averaging window) so the peak can be localized far
    finer than 1/duration would otherwise allow. This matters at the
    ~2-second buffer sizes calibrate_notch() typically gets: 2000 samples
    at 1000Hz gives a native 0.5Hz bin spacing, too coarse to tell 49.8Hz
    from 50.0Hz apart at all (confirmed directly -- without padding, this
    function returned 50.0 on a synthetic 49.8Hz tone). Zero-padding adds
    no new information but interpolates the *existing* spectral estimate
    far more precisely, which is exactly what's needed to locate a single
    strong, narrow peak (mains interference, not broadband noise).
    """
    x = np.asarray(signal_uv, dtype=float)
    if x.size < 16:
        return None
    x = x - x.mean()
    nperseg = min(2048, x.size)
    nfft = max(32768, 1 << int(np.ceil(np.log2(nperseg))))
    fx, pxx = welch(x, fs=sfreq, nperseg=nperseg, nfft=nfft)
    band = (fx >= search_range[0]) & (fx <= search_range[1])
    if not np.any(band):
        return None
    return float(fx[band][np.argmax(pxx[band])])


def tune_thresholds(scores: np.ndarray, labels: np.ndarray, min_commit_rate: float = 0.3,
                     rest_scores: Optional[np.ndarray] = None, rest_weight: float = 1.0):
    """Grid-search (min_confidence, min_ratio) against a labelled batch of
    normalised FBCCA score vectors.

    This is NOT model training -- FBCCA has no templates or weights to fit.
    It only tunes the accept/reject gate, which is a per-session statistic
    (how separated is the true class's score from the runner-up, on this
    subject/electrodes/day), so it is safe to compute from a handful of
    short, unaligned looks at each target -- unlike TRCA's phase-locked
    averaging, this does not need trial alignment to be meaningful.

    scores : (n_windows, n_classes) normalised confidence (rho / fb_norm)
             for every window collected while the subject looked at a
             known target.
    labels : (n_windows,) true class index for each window in `scores`.
    min_commit_rate : candidates that would answer on fewer than this
        fraction of windows are penalised -- a threshold that is "accurate"
        only because it almost never commits isn't useful in practice.
    rest_scores : optional (n_rest_windows, n_classes) scores collected
        while the subject looked at NONE of the targets (a "rest"/"look
        away" calibration condition -- see connect_and_calibrate()'s
        calibration loop in run_ssvep_detection.py). Without this,
        tune_thresholds() has never seen what scores
        look like when there's no real target to detect, so it has no
        way to learn a gate that rejects that state -- it only ever
        optimises "which of the 4 classes is this", never "is this any
        of them at all". When given, any (mc, mr) candidate that would
        still accept (commit to a class on) a rest window is penalised,
        so the search also learns to reject the no-target state, not
        just discriminate among the trained ones.
    rest_weight : how much one rest-window false-accept costs, in the same
        units as classification accuracy (both are rates in [0, 1]).  1.0
        means "one false accept on a rest window is exactly as bad as one
        misclassification on a real trial" -- raise it if false detections
        while not looking at anything are more costly than getting a real
        look wrong (usually true for anything that fires/acts on a
        detection, like ssvep_zombie_game.py).

    Returns a dict: min_confidence, min_ratio, accuracy_at_threshold,
    commit_rate_at_threshold, baseline_accuracy (argmax with no gating at
    all, i.e. the ceiling these thresholds are trading away for
    reliability), and rest_false_accept_rate_at_threshold (0.0 if
    rest_scores wasn't given).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_windows, n_classes = scores.shape
    order = np.argsort(scores, axis=1)[:, ::-1]
    top_idx = order[:, 0]
    top = scores[np.arange(n_windows), top_idx]
    second = scores[np.arange(n_windows), order[:, 1]] if n_classes > 1 else np.zeros(n_windows)
    ratio = np.where(second > 1e-9, top / np.maximum(second, 1e-9), np.inf)
    baseline_acc = float(np.mean(top_idx == labels))

    rest_top = rest_ratio = None
    if rest_scores is not None and len(rest_scores) > 0:
        rest_scores = np.asarray(rest_scores, dtype=float)
        r_order = np.argsort(rest_scores, axis=1)[:, ::-1]
        n_rest = rest_scores.shape[0]
        rest_top = rest_scores[np.arange(n_rest), r_order[:, 0]]
        rest_second = rest_scores[np.arange(n_rest), r_order[:, 1]] if n_classes > 1 else np.zeros(n_rest)
        rest_ratio = np.where(rest_second > 1e-9, rest_top / np.maximum(rest_second, 1e-9), np.inf)

    # Candidate confidence gates: quantiles of the *correct* predictions'
    # own confidence -- there's no point testing a gate that would reject
    # windows the classifier was never going to get right anyway.
    correct_conf = top[top_idx == labels]
    if correct_conf.size == 0:
        conf_candidates = np.linspace(0.05, 0.5, 10)
    else:
        conf_candidates = np.unique(np.quantile(correct_conf, np.linspace(0.0, 0.7, 12)))
    # Rest windows are exactly the case a confidence gate alone can miss
    # (a rest window can score high on one class by chance while still
    # beating the runner-up comfortably), so widen the ratio search when
    # rest data is available -- that's the knob that actually separates
    # "confidently wrong" from "confidently right".
    ratio_candidates = np.array([1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.75, 2.0]
                                 if rest_top is None else
                                 [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])

    best = None
    for mc in conf_candidates:
        for mr in ratio_candidates:
            accepted = (top >= mc) & (ratio >= mr)
            commit_rate = float(accepted.mean())
            if accepted.sum() == 0:
                continue
            acc = float(np.mean(top_idx[accepted] == labels[accepted]))
            # Below the floor, scale down instead of hard-excluding, so we
            # still return *something* sensible if every candidate is thin.
            objective = acc if commit_rate >= min_commit_rate else acc * (commit_rate / min_commit_rate)

            rest_false_accept = 0.0
            if rest_top is not None:
                rest_false_accept = float(((rest_top >= mc) & (rest_ratio >= mr)).mean())
                objective -= rest_weight * rest_false_accept

            key = (objective, commit_rate)
            if best is None or key > best[0]:
                best = (key, float(mc), float(mr), acc, commit_rate, rest_false_accept)

    if best is None:  # degenerate: nothing ever got accepted at any gate
        return {
            "min_confidence": 0.25, "min_ratio": 1.15,
            "accuracy_at_threshold": 0.0, "commit_rate_at_threshold": 0.0,
            "baseline_accuracy": baseline_acc, "rest_false_accept_rate_at_threshold": 0.0,
        }
    _, mc, mr, acc, commit_rate, rest_false_accept = best
    return {
        "min_confidence": mc, "min_ratio": mr,
        "accuracy_at_threshold": acc, "commit_rate_at_threshold": commit_rate,
        "baseline_accuracy": baseline_acc, "rest_false_accept_rate_at_threshold": rest_false_accept,
    }


def _cca_max_corr(X: np.ndarray, Qy: np.ndarray) -> float:
    """Largest canonical correlation between data X and a reference whose
    orthonormal basis Qy is already known.

    X  : (n_samples, n_channels), mean-removed
    Qy : (n_samples, n_refs) orthonormal, precomputed from the sin/cos bank

    The canonical correlations between two column spaces are the singular
    values of Qx.T @ Qy, where Qx/Qy are orthonormal bases for those
    spaces. So one QR on the data plus one small SVD gives the exact
    answer -- no iteration.
    """
    # Economy QR. rcond-style guard: if a channel is flat or duplicated the
    # R diagonal collapses, and the corresponding Q column is arbitrary.
    Qx, R = np.linalg.qr(X)
    diag = np.abs(np.diag(R))
    keep = diag > (diag.max() * 1e-10 if diag.size and diag.max() > 0 else 0)
    if not np.any(keep):
        return 0.0
    Qx = Qx[:, keep]
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


class FBCCAClassifier:
    def __init__(
        self,
        frequencies: Sequence[float],
        sfreq: float = 1000.0,
        n_channels: int = 3,
        window_sec: float = 2.0,
        classify_every_sec: float = 0.25,
        n_harmonics: int = 4,
        n_bands: int = 5,
        target_sfreq: float = 250.0,
        notch_freqs: Sequence[float] = (50.0, 100.0),
        band_high_hz: float = 90.0,
        sub_band_guard_hz: float = 2.0,
        min_confidence: float = 0.25,
        min_ratio: float = 1.15,
        vote_window: int = 4,
        vote_needed: int = 2,
    ):
        """
        min_confidence : threshold on the *normalised* score (0..1), i.e. the
            sub-band-weighted mean canonical correlation. Unlike the raw
            FBCCA rho -- whose scale depends on how many sub-bands you use
            -- this is directly comparable across configurations. 0.25 is a
            sane starting point for 3 channels / 2 s; watch the printed
            numbers and tune.
        min_ratio : the winner must also beat the runner-up by this factor.
            A ratio travels across subjects and sessions far better than an
            absolute margin, because overall correlation level drifts with
            impedance and attention but the *relative* separation does not.
        vote_window / vote_needed : emit a prediction only when the same
            class wins `vote_needed` of the last `vote_window` decisions.
            At 4 decisions/sec off a 2 s window, consecutive windows overlap
            heavily, so single-window output flickers; voting costs a little
            latency and buys a lot of stability.
        sub_band_guard_hz : see the "sub-band edges" block below -- backs
            each band's low edge off from the lowest target frequency's
            harmonic so that frequency doesn't sit exactly on a -3dB filter
            edge. Measured directly on this module: with guard=0, the lowest
            class in [12,14,16,18] Hz scored gain=0.707 (-3.0dB, the exact
            Butterworth critical-frequency point) at its own fundamental in
            band 1, vs 0.99 (-0.07dB) for the highest class -- a systematic,
            per-class bias that isn't about SNR at all. guard=2.0 brings that
            up to -0.7dB. Don't set this so large that low edges dip back
            into occipital alpha (~8-12Hz) for low base frequencies.
        """
        self.frequencies = list(frequencies)
        self.sfreq = float(sfreq)
        self.n_channels = n_channels
        self.window_sec = window_sec
        self.window_samples = int(window_sec * sfreq)
        self.classify_every_sec = classify_every_sec

        # ---- decimation ----
        self.decim = max(1, int(round(self.sfreq / target_sfreq)))
        self.fs_d = self.sfreq / self.decim
        nyq_d = self.fs_d / 2.0

        # ---- sub-band edges ----
        # Band m nominally spans [m*base, band_high]. base = lowest target
        # frequency, so band m admits the m-th harmonic and everything above
        # it. Backed off by sub_band_guard_hz so the lowest target's own
        # harmonics don't land exactly on a passband edge (see
        # sub_band_guard_hz above) -- without this, whichever frequency
        # happens to be lowest in the set is systematically penalised on
        # every sub-band, independent of SNR or alpha.
        base = float(min(self.frequencies))
        self.band_high = min(band_high_hz, 0.90 * nyq_d)
        edges = []
        for m in range(1, n_bands + 1):
            low = max(1.0, m * base - sub_band_guard_hz)
            if low >= self.band_high * 0.85:
                break  # no usable passband left -- stop rather than emit junk
            edges.append((low, self.band_high))
        if not edges:
            raise ValueError("No usable sub-bands; check frequencies/target_sfreq.")
        self.band_edges = edges
        self.n_bands = len(edges)
        self._band_sos = [
            butter(4, (lo, hi), btype="bandpass", fs=self.fs_d, output="sos")
            for lo, hi in edges
        ]
        self.fb_coefs = np.array(
            [(m + 1) ** -1.25 + 0.25 for m in range(self.n_bands)]
        )
        self._fb_norm = float(self.fb_coefs.sum())

        # ---- mains notches (applied at the original rate, before decimation) ----
        # Built from the nominal frequencies at first; call calibrate_notch()
        # once real data is flowing to re-center these on the *actual*
        # interference peak instead -- see detect_mains_frequency()'s
        # docstring for why the nominal value alone leaves real power on
        # the table.
        self._nominal_notch_freqs = [f for f in notch_freqs if f < sfreq / 2 * 0.95]
        self._notch_q = 30
        self._notch_ba = [iirnotch(f, Q=self._notch_q, fs=self.sfreq) for f in self._nominal_notch_freqs]
        self.detected_notch_freqs = list(self._nominal_notch_freqs)  # updated by calibrate_notch()

        # ---- anti-alias lowpass for the decimation step ----
        self._aa_sos = (
            butter(8, 0.8 * nyq_d, btype="lowpass", fs=self.sfreq, output="sos")
            if self.decim > 1
            else None
        )

        # ---- reference bank ----
        # Cap harmonics so the highest one stays inside the analysed band, then
        # apply that same cap to EVERY class (see module docstring).
        fmax = max(self.frequencies)
        self.n_harmonics = max(1, min(n_harmonics, int(self.band_high // fmax)))
        n_d = self.window_samples // self.decim
        t = np.arange(n_d) / self.fs_d
        self._ref_q = []
        for f in self.frequencies:
            cols = []
            for h in range(1, self.n_harmonics + 1):
                cols.append(np.sin(2 * np.pi * f * h * t))
                cols.append(np.cos(2 * np.pi * f * h * t))
            Y = np.column_stack(cols)
            Y -= Y.mean(axis=0, keepdims=True)
            self._ref_q.append(np.linalg.qr(Y)[0])
        self._n_dec_samples = n_d

        # ---- thresholds / voting ----
        self.min_confidence = min_confidence
        self.min_ratio = min_ratio
        self._votes = deque(maxlen=vote_window)
        self.vote_needed = vote_needed

        # ---- rolling buffer ----
        self._buf = np.zeros((n_channels, self.window_samples), dtype=np.float64)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

        # ---- outputs ----
        self.last_prediction: Optional[float] = None
        self.last_confidence: float = 0.0
        self.last_ratio: float = 0.0
        self.last_scores: Optional[np.ndarray] = None
        self.on_prediction: Optional[Callable[[float], None]] = None
        self.on_idle: Optional[Callable[[], None]] = None

        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ---------------- ingest ----------------
    def push_sample_values(self, values: Sequence[float]):
        """Feed one sample: a flat sequence of per-channel values in
        microvolts. Cheap by design -- this runs on the ADC receive thread,
        so it does nothing but a buffer write. All the filtering and
        correlation happens on the worker thread started by start()."""
        with self._lock:
            i = self._write
            for c in range(self.n_channels):
                self._buf[c, i] = values[c]
            self._write = (i + 1) % self.window_samples
            if self._filled < self.window_samples:
                self._filled += 1

    def push_samples(self, batch: List[dict]):
        """Compatibility shim for the ch0/ch1/ch2 dict format."""
        for s in batch:
            self.push_sample_values((s["ch0"], s["ch1"], s["ch2"]))

    def snapshot(self) -> Optional[np.ndarray]:
        """The current rolling window in chronological order, (n_channels,
        window_samples), or None if it hasn't filled yet. Public so callers
        (e.g. a calibration/threshold-tuning loop) can pull windows on their
        own schedule instead of only through the classify_every_sec worker
        loop -- see connect_and_calibrate() in run_ssvep_detection.py."""
        with self._lock:
            if self._filled < self.window_samples:
                return None
            return np.roll(self._buf, -self._write, axis=1).copy()

    def calibrate_notch(self, raw_window: Optional[np.ndarray] = None,
                         search_half_width: float = 3.0) -> List[float]:
        """Re-center the mains notch(es) on the interference frequency
        actually measured in raw_window (or the current buffer snapshot()
        if raw_window is omitted), instead of trusting the nominal
        notch_freqs given at construction. Call this once, early -- e.g.
        right after enough data has accumulated to fill one window, before
        calibration/detection starts (see connect_and_calibrate() in
        run_ssvep_detection.py). Safe to call more than once (e.g. at
        the start of every session) since interference frequency drifts a
        little run to run -- measured on this rig: 49.8Hz in three
        sessions, 50.3Hz in a fourth. Safe to skip entirely; the notch
        just stays on the nominal frequencies from construction.

        Returns the detected frequency for each nominal notch (falls back
        to the nominal value itself if no clear peak was found nearby).
        """
        if raw_window is None:
            raw_window = self.snapshot()
        if raw_window is None:
            return list(self.detected_notch_freqs)

        detected = []
        for nom in self._nominal_notch_freqs:
            lo, hi = nom - search_half_width, nom + search_half_width
            per_channel = [
                detect_mains_frequency(raw_window[c], self.sfreq, (lo, hi))
                for c in range(raw_window.shape[0])
            ]
            per_channel = [f for f in per_channel if f is not None]
            detected.append(float(np.median(per_channel)) if per_channel else nom)

        self.detected_notch_freqs = detected
        self._notch_ba = [iirnotch(f, Q=self._notch_q, fs=self.sfreq) for f in detected]
        return detected

    # ---------------- worker ----------------
    def start(self):
        """Start the background classification thread."""
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            win = self.snapshot()
            if win is not None:
                try:
                    self._classify(win)
                except Exception as e:  # never let one bad window kill the loop
                    print(f"[FBCCA] classification error: {e}")
            sleep = self.classify_every_sec - (time.perf_counter() - t0)
            if sleep > 0:
                self._stop.wait(sleep)

    # ---------------- the algorithm ----------------
    def preprocess(self, window: np.ndarray) -> np.ndarray:
        """(n_channels, window_samples) at self.sfreq
        -> (n_dec_samples, n_channels) at self.fs_d, mean-removed."""
        x = window - window.mean(axis=1, keepdims=True)
        for b, a in self._notch_ba:
            x = filtfilt(b, a, x, axis=-1)
        if self._aa_sos is not None:
            x = sosfiltfilt(self._aa_sos, x, axis=-1)[:, :: self.decim]
        x = x[:, : self._n_dec_samples]
        return np.ascontiguousarray(x.T)

    def score(self, window: np.ndarray) -> np.ndarray:
        """Raw FBCCA rho per class. window: (n_channels, window_samples)."""
        xd = self.preprocess(window)
        r = np.zeros((self.n_bands, len(self.frequencies)))
        for bi, sos in enumerate(self._band_sos):
            xb = sosfiltfilt(sos, xd, axis=0)
            xb = xb - xb.mean(axis=0, keepdims=True)
            for ci, Qy in enumerate(self._ref_q):
                r[bi, ci] = _cca_max_corr(xb, Qy)
        return self.fb_coefs @ r

    def normalized_scores(self, window: np.ndarray) -> np.ndarray:
        """score(), divided into the 0..1 range so thresholds/quantiles are
        comparable across different n_bands/n_harmonics configurations. Used
        directly by the calibration/threshold-tuning phase."""
        return self.score(window) / self._fb_norm

    def _classify(self, window: np.ndarray):
        rho = self.score(window)
        order = np.argsort(rho)[::-1]
        top, second = int(order[0]), int(order[1]) if len(order) > 1 else None

        conf = float(rho[top]) / self._fb_norm  # normalised 0..1
        ratio = float(rho[top] / rho[second]) if second is not None and rho[second] > 1e-9 else np.inf

        self.last_scores = rho
        self.last_confidence = conf
        self.last_ratio = ratio

        accepted = top if (conf >= self.min_confidence and ratio >= self.min_ratio) else None
        self._votes.append(accepted)

        valid = [v for v in self._votes if v is not None]
        if valid:
            winner, count = Counter(valid).most_common(1)[0]
            if count >= self.vote_needed:
                freq = self.frequencies[winner]
                self.last_prediction = freq
                if self.on_prediction:
                    self.on_prediction(freq)
                return

        self.last_prediction = None
        if self.on_idle:
            self.on_idle()

    # ---------------- offline / batch ----------------
    def predict_window(self, window: np.ndarray) -> int:
        """Class index for one (n_channels, window_samples) window, with no
        thresholding or voting. For offline evaluation."""
        return int(np.argmax(self.score(window)))

    def describe(self) -> str:
        bands = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in self.band_edges)
        return (
            f"FBCCA: {len(self.frequencies)} targets {self.frequencies} Hz | "
            f"{self.window_sec}s window @ {self.sfreq:.0f}->{self.fs_d:.0f} Hz | "
            f"{self.n_bands} sub-bands [{bands}] Hz | {self.n_harmonics} harmonics | "
            f"decide every {self.classify_every_sec}s, vote {self.vote_needed}/{self._votes.maxlen}"
        )


class DynamicFBCCAClassifier:
    """FBCCA with dynamic stopping: instead of always waiting for one fixed
    window length, score progressively longer windows as data arrives and
    commit as soon as one is confident -- most of the fixed-window
    classifier's latency comes from waiting out the FULL window before the
    very first decision is even possible, not from the per-decision compute
    (a single score() call takes single-digit milliseconds).

    Tested directly against real recorded trials before building this: a
    duration ladder like the default found a confident correct answer in
    4/12 real trials averaging 0.94s from the start of fixation, vs ~1.6-
    1.8s under a fixed 2.0s window with 3-of-5 voting -- real latency win
    on trials where the signal is clearly there fast. The other 8/12
    trials found nothing confident at any tested duration up to 2.0s --
    dynamic stopping doesn't fix those, it just stops wasting time on the
    easy ones. It's a latency technique, not an SNR technique.

    WHY THE VOTE LAYER IS STILL HERE
    ---------------------------------
    A single window's score clearing threshold can still be a transient
    blip -- confirmed directly: some trials showed one early duration
    clear the gate for exactly one tick, then drop out on the very next
    one. Dynamic stopping only changes how much data EACH tick's decision
    needs before it's willing to answer; it doesn't replace cross-tick
    corroboration. vote_window/vote_needed (same mechanism as
    FBCCAClassifier) still requires several RECENT ticks to agree before
    a final commit -- that's what keeps false positives in check while
    dynamic stopping brings each tick's own latency down.

    WHY THRESHOLDS ARE TUNED SEPARATELY PER DURATION
    ---------------------------------------------------
    A shorter window has less data to average over, so its raw scores are
    noisier than a longer window's -- one shared threshold either makes
    short windows too eager to fire (tuned loose enough for them, too
    permissive for long ones) or long windows never get to prove they're
    more reliable (tuned strict enough for them, so short ones almost
    never pass). calibrate_thresholds() runs tune_thresholds() separately
    per duration on the same calibration data, sliced to each length.

    Same external API as FBCCAClassifier (push_sample_values, push_samples,
    snapshot, calibrate_notch, start, stop, on_prediction, on_idle) so it's
    a drop-in replacement -- see connect_and_calibrate()/run_live_detection()
    in run_ssvep_detection.py, which build and use an FBCCAClassifier
    directly.
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        sfreq: float = 1000.0,
        n_channels: int = 3,
        durations: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
        classify_every_sec: float = 0.25,
        n_harmonics: int = 4,
        n_bands: int = 5,
        target_sfreq: float = 250.0,
        notch_freqs: Sequence[float] = (50.0, 100.0),
        band_high_hz: float = 90.0,
        sub_band_guard_hz: float = 2.0,
        vote_window: int = 4,
        vote_needed: int = 2,
    ):
        self.frequencies = list(frequencies)
        self.sfreq = float(sfreq)
        self.n_channels = n_channels
        self.durations = sorted(durations)
        self.classify_every_sec = classify_every_sec

        # one scorer per duration -- each is a full FBCCAClassifier, reused
        # only for its precomputed filter design / reference bank / score()
        # method, not its own buffer or voting (this class owns those).
        self._scorers = {
            dur: FBCCAClassifier(
                frequencies=frequencies, sfreq=sfreq, n_channels=n_channels,
                window_sec=dur, classify_every_sec=classify_every_sec,
                n_harmonics=n_harmonics, n_bands=n_bands, target_sfreq=target_sfreq,
                notch_freqs=notch_freqs, band_high_hz=band_high_hz,
                sub_band_guard_hz=sub_band_guard_hz,
            )
            for dur in self.durations
        }
        self._duration_samples = {dur: int(round(dur * self.sfreq)) for dur in self.durations}
        self.max_window_samples = self._duration_samples[self.durations[-1]]

        # notch design is identical across durations (mains interference
        # doesn't depend on window length), so expose one copy publicly
        # instead of making callers reach into _scorers[...] themselves.
        first_scorer = self._scorers[self.durations[0]]
        self.nominal_notch_freqs = first_scorer._nominal_notch_freqs
        self.detected_notch_freqs = list(first_scorer.detected_notch_freqs)

        # ---- shared raw ring buffer, sized to the LONGEST duration ----
        self._buf = np.zeros((n_channels, self.max_window_samples), dtype=np.float64)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()

        # ---- voting (identical mechanism to FBCCAClassifier) ----
        self._votes = deque(maxlen=vote_window)
        self.vote_needed = vote_needed

        # ---- outputs ----
        self.last_prediction: Optional[float] = None
        self.last_confidence: float = 0.0
        self.last_ratio: float = 0.0
        self.last_duration_used: Optional[float] = None  # which duration actually resolved the last tick
        self.on_prediction: Optional[Callable[[float], None]] = None
        self.on_idle: Optional[Callable[[], None]] = None

        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    # ---------------- ingest ----------------
    def push_sample_values(self, values: Sequence[float]):
        with self._lock:
            i = self._write
            for c in range(self.n_channels):
                self._buf[c, i] = values[c]
            self._write = (i + 1) % self.max_window_samples
            if self._filled < self.max_window_samples:
                self._filled += 1

    def push_samples(self, batch: List[dict]):
        for s in batch:
            self.push_sample_values((s["ch0"], s["ch1"], s["ch2"]))

    def snapshot(self, n_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """Most recent n_samples (default: the full max-duration buffer),
        chronological order, or None if that many haven't accumulated yet."""
        need = n_samples or self.max_window_samples
        with self._lock:
            if self._filled < need:
                return None
            full = np.roll(self._buf, -self._write, axis=1)
            return full[:, -need:].copy()

    def calibrate_notch(self, raw_window: Optional[np.ndarray] = None,
                         search_half_width: float = 3.0) -> Optional[List[float]]:
        """Re-centers every duration's notch filter on the same measured
        mains frequency (mains interference doesn't depend on window
        length, so one measurement -- taken from the longest/most-data
        slice available -- applies to all of them). See
        FBCCAClassifier.calibrate_notch() / detect_mains_frequency() for
        why this matters."""
        if raw_window is None:
            raw_window = self.snapshot()
        if raw_window is None:
            return None
        detected = None
        for dur, scorer in self._scorers.items():
            n = self._duration_samples[dur]
            detected = scorer.calibrate_notch(raw_window[:, -n:], search_half_width=search_half_width)
        self.detected_notch_freqs = detected
        return detected

    def calibrate_thresholds(self, records: List[tuple]) -> dict:
        """records: list of (true_class_idx, raw_window) pairs from
        calibration, where raw_window is (n_channels, N) with N >= the
        longest configured duration's sample count. Tunes each duration's
        scorer.min_confidence/min_ratio SEPARATELY (see class docstring for
        why a shared threshold doesn't work). Returns {duration: tune_
        thresholds() result} for logging."""
        labels = np.array([r[0] for r in records])
        results = {}
        for dur, scorer in self._scorers.items():
            n = self._duration_samples[dur]
            scores = np.array([scorer.normalized_scores(w[:, -n:]) for _, w in records])
            r = tune_thresholds(scores, labels)
            scorer.min_confidence = r["min_confidence"]
            scorer.min_ratio = r["min_ratio"]
            results[dur] = r
        return results

    def evaluate_cascade(self, records: List[tuple]) -> dict:
        """Offline replay of the SAME shortest-confident-wins cascade
        _classify() runs live, over a labelled (true_idx, raw_window)
        batch -- the honest single accuracy/commit-rate summary to print
        after calibrate_thresholds(), the cascade-level analogue of
        tune_thresholds()'s accuracy_at_threshold. Also reports which
        duration actually resolved each committed decision, so you can see
        whether the cascade is really resolving fast (short durations
        dominating) or falling back to the longest one most of the time.
        """
        correct = committed = 0
        dur_usage: Counter = Counter()
        for true_idx, raw_window in records:
            chosen = None
            for dur in self.durations:
                n = self._duration_samples[dur]
                if n > raw_window.shape[1]:
                    break
                scorer = self._scorers[dur]
                window = raw_window[:, -n:]
                rho = scorer.score(window)
                order = np.argsort(rho)[::-1]
                top = int(order[0])
                conf = float(rho[top]) / scorer._fb_norm
                ratio = float(rho[top] / rho[order[1]]) if rho[order[1]] > 1e-9 else float("inf")
                if conf >= scorer.min_confidence and ratio >= scorer.min_ratio:
                    chosen = (dur, top)
                    break
            if chosen is not None:
                committed += 1
                dur_usage[chosen[0]] += 1
                if chosen[1] == true_idx:
                    correct += 1
        n = len(records)
        return {
            "accuracy_at_threshold": correct / committed if committed else 0.0,
            "commit_rate_at_threshold": committed / n if n else 0.0,
            "duration_usage": dict(sorted(dur_usage.items())),
        }

    # ---------------- worker ----------------
    def start(self):
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            buf = self.snapshot(self._duration_samples[self.durations[0]])  # need at least the shortest duration
            if buf is not None:
                try:
                    self._classify(buf)
                except Exception as e:
                    print(f"[DynamicFBCCA] classification error: {e}")
            sleep = self.classify_every_sec - (time.perf_counter() - t0)
            if sleep > 0:
                self._stop.wait(sleep)

    # ---------------- the cascade ----------------
    def _classify(self, buf: np.ndarray):
        """buf: (n_channels, n_avail) most recent samples, chronological,
        n_avail >= the shortest duration's sample count (guaranteed by
        _loop's snapshot() call) but not necessarily the longest one yet."""
        n_avail = buf.shape[1]
        chosen = None
        best_attempt = None  # (conf, ratio) from the longest duration tried, for diagnostics even on a miss

        for dur in self.durations:  # shortest first
            n = self._duration_samples[dur]
            if n > n_avail:
                break  # this and every longer duration aren't available yet
            scorer = self._scorers[dur]
            window = buf[:, -n:]
            rho = scorer.score(window)
            order = np.argsort(rho)[::-1]
            top = int(order[0])
            conf = float(rho[top]) / scorer._fb_norm
            ratio = float(rho[top] / rho[order[1]]) if rho[order[1]] > 1e-9 else float("inf")
            best_attempt = (conf, ratio, dur)
            if conf >= scorer.min_confidence and ratio >= scorer.min_ratio:
                chosen = (dur, top, conf, ratio)
                break  # short-circuit: this duration was confident, stop escalating

        if best_attempt is not None:
            self.last_confidence, self.last_ratio, self.last_duration_used = best_attempt

        accepted = None
        if chosen is not None:
            dur, top, conf, ratio = chosen
            self.last_confidence, self.last_ratio, self.last_duration_used = conf, ratio, dur
            accepted = top
        self._votes.append(accepted)

        valid = [v for v in self._votes if v is not None]
        if valid:
            winner, count = Counter(valid).most_common(1)[0]
            if count >= self.vote_needed:
                freq = self.frequencies[winner]
                self.last_prediction = freq
                if self.on_prediction:
                    self.on_prediction(freq)
                return

        self.last_prediction = None
        if self.on_idle:
            self.on_idle()

    def describe(self) -> str:
        dur_str = "/".join(f"{d:g}" for d in self.durations)
        return (
            f"DynamicFBCCA: {len(self.frequencies)} targets {self.frequencies} Hz | "
            f"durations [{dur_str}]s (shortest-confident-wins) @ {self.sfreq:.0f}Hz | "
            f"decide every {self.classify_every_sec}s, vote {self.vote_needed}/{self._votes.maxlen}"
        )
