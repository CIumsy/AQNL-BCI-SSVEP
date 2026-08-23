#!/usr/bin/env python3
"""
run_ssvep_detection.py

Reads EEG from the ADS1299 board, calibrates, and decodes which target
the subject is looking at. This file has no display of its own.

Three ways to run it:
  1. On its own (python run_ssvep_detection.py) -- terminal only, no
     flickering squares at all. Needs fbcca_thresholds.json to already
     exist. Copy that file over from a machine that ran a full
     calibration. There is no network link between machines here on
     purpose; it is just a file you copy once.
  2. Imported by main.py, which adds the Tk display for Windows.
  3. Imported by run_uno_q.py, which runs the display or the game in a
     separate process and this pipeline in the parent.

WHAT IS RUNNING RIGHT NOW
---------------------------
The plain FBCCAClassifier at 7, 13, 15 and 17 Hz, with a fixed 2.0 s
window and a 2-of-4 vote. This is the setup confirmed working on real
hardware.

ssvep_fbcca.py also contains DynamicFBCCAClassifier, which can answer
early when it is already sure. It has been checked against synthetic data
and against replayed recordings, but never on live hardware. Do not
switch to it without testing it live first.

WHY NOTHING IS TRAINED HERE
-----------------------------
The original plan used TRCA. TRCA averages calibration trials together to
build a template per target, and that only works if every trial is cut at
the same point in the flicker cycle.

This pipeline has no such alignment: trials are cut by a timer on one
thread while the flicker runs off its own clock on another. We checked
the trained model directly, and none of its four templates peaked at its
own frequency. Isolating just that one variable: 88% accuracy when
phase-locked, 27.5% -- chance -- when free-running.

FBCCA has no templates and nothing to train. It correlates each window
against plain sine and cosine waves, which works no matter what the
phase is. What is called "calibration" below only tunes two accept/reject
thresholds from score statistics, and reports an honest accuracy estimate
before you trust it.

WHY 7 / 13 / 15 / 17 Hz
-------------------------
Confirmed working on real hardware.

We checked harmonics up to the 4th and flagged any gap under 2 Hz. One
turned up: 13 Hz's 4th harmonic (52 Hz) sits 1 Hz from 17 Hz's 3rd
harmonic (51 Hz). It is real, but both are high harmonics, which carry
little weight in the fusion, so it is not a clash between fundamentals.

7 Hz sits about 2.8 Hz from this subject's alpha peak (~9.8 Hz). Closer
than ideal, but workable. 12 Hz was tried first and never worked once --
0 of 3 real trials ever locked on -- because it was only 2.2 Hz away.

If you change the frequencies, run the same two checks (harmonic
collisions and distance from alpha) before trusting the result.

A FILTER BUG THAT PENALISED ONE TARGET
----------------------------------------
The sub-band edges in ssvep_fbcca.py used to be `low = m * base`, where
base is the lowest target frequency. That puts the lowest target exactly
on the filter's -3 dB corner, every time, by construction. Measured gain
was 0.7071, i.e. exactly -3.01 dB.

So the lowest frequency was always handicapped, whatever you chose.
Fixed with sub_band_guard_hz, which backs the edge off by 2 Hz.

THE MAINS NOTCH IS MEASURED, NOT ASSUMED
------------------------------------------
A fixed 50.0 Hz notch left 11.4 dB of mains hum still in the signal,
because the real peak on this rig drifts between 49.8 and 50.3 Hz from
session to session. Phase 0 below measures the actual peak from live data
each run and moves the notch onto it. See calibrate_notch() in
ssvep_fbcca.py.

CALIBRATING "NOT LOOKING AT ANYTHING"
---------------------------------------
tune_thresholds() used to only ever see windows where the subject WAS
looking at a target. It had no idea what the scores look like when the
subject looks away, so it could not learn to reject that -- and in live
use it kept firing detections at nothing.

Calibration now mixes in CAL_REST_TRIALS "look at the centre" trials, and
tune_thresholds() penalises any threshold pair that would still accept a
rest window. On synthetic data shaped like ours: 55% of "not looking"
windows were misclassified without rest trials, 11% with them. The cost
is a lower commit rate on real trials, not lower accuracy.

WHAT WE WOULD DO NEXT: A TRCA-R HYBRID
----------------------------------------
Not implemented. Notes for whoever picks it up.

"Single-trial TRCA" is not a thing mathematically. The covariance matrix
is built by summing over pairs of DIFFERENT trials, so with one trial the
loop never runs and the matrix is exactly zero. We confirmed this: the
eigenvalues come back as 0.0.

TRCA-R (Wong et al. 2020) works around it by padding the single real
trial with synthetic sine/cosine trials. To get there we need two things:

  1. A clock locked to the stimulus. Pass the sample number through
     on_adc_sample and line up wall-clock time with sample number once at
     connect time, instead of letting two clocks run independently.
  2. Real labelled recordings to tune the FBCCA/TRCA-R blend against. A
     synthetic simulator cannot show TRCA's real per-subject advantage.

Calibration already saves raw labelled windows for exactly this purpose
(see SESSION_LOG_OUT). They are not phase-locked yet, but they are real
recordings of this subject, ready to build against.

Usage (standalone, console-only):
    python run_ssvep_detection.py
"""

import json
import os
import random
import sys
import threading
import time

import numpy as np

from ads1299_stream import SAMPLING_RATE, run_stream
from ssvep_fbcca import FBCCAClassifier, tune_thresholds

# ────────────────────────────────────────────────────────────────────────
# EDIT ME: which 3 of the board's 16 channels (numbered 0-15) your
# electrodes are plugged into, in the order O1, Oz, O2. Match your wiring.
CHANNEL_INDICES = [0, 1, 2]

# EDIT ME -- THE ONLY PLACE TARGET FREQUENCIES ARE SET. Everything else in
# this file and ssvep_fbcca.py reads from this constant (or from values
# derived from it, e.g. sub-band edges scale with min(this)) -- change the
# count or the values here and the FBCCA reference bank, calibration
# trial list, threshold cache, AND every display/game that imports
# FREQUENCIES from this module (main.py, run_uno_q.py) all
# follow automatically -- this is the only place it's set. CONFIRMED
# WORKING on real hardware -- see the module docstring's TARGET
# FREQUENCIES section before changing this.
FREQUENCIES = [7, 13, 15, 17]

# EDIT ME: FBCCA analysis window. Longer = more accurate but slower to
# react. 2.0s is the confirmed-working value -- see the module docstring.
WINDOW_SEC = 2.0

# EDIT ME: FBCCA sin/cos harmonics per reference (spec calls for 3-4 to
# exploit clean harmonic structure on integer-divisor frequencies).
N_HARMONICS = 4

# EDIT ME: vote_window/vote_needed for the live-detection smoothing layer.
# 4/2 is the confirmed-working value -- measured directly against a real
# recorded session (simulating the exact same vote logic on the true
# per-window score sequence): 5/3 locked onto the correct target in 6/12
# trials, "on" 16% of trial time, avg 1.83s to first commit. 4/2 raised
# that to 8/12 trials, "on" 24% of trial time, same underlying per-window
# accuracy (this only changes how many consecutive agreeing windows are
# required, not the accept/reject gate itself).
VOTE_WINDOW = 4
VOTE_NEEDED = 2

# EDIT ME: calibration is now threshold-tuning + a live accuracy estimate,
# NOT model training, so it can be short. 3 trials/class x ~4.5s each.
CAL_TRIALS_PER_FREQ = 3
CAL_TRIAL_SEC = WINDOW_SEC + 2.5  # long enough to get several sliding windows/trial

# EDIT ME: "look away from every target" trials, interleaved into the SAME
# calibration pass (not a separate phase) -- see the module docstring's
# REST-CONDITION CALIBRATION section for why these exist and what they
# measurably fix. Costs roughly CAL_REST_TRIALS * CAL_TRIAL_SEC extra
# seconds in Phase 1 -- 3 trials adds about 20s.
CAL_REST_TRIALS = 3
REST = -1  # sentinel trial_order value: "look away from all targets"

# Skip Phase 1 and go straight to live detection on the last saved
# thresholds. NOT set automatically anywhere -- skipping calibration is
# always an explicit, opt-in choice made at the SPACE prompt (see
# prompt_skip_calibration()), because stale thresholds fail silently:
# you get plausible-looking but wrong detections rather than an error.
SKIP_CALIBRATION = False
THRESH_OUT = "fbcca_thresholds.json"  # tuned thresholds, reloaded next run

# EDIT ME: how long the skip prompt counts down before starting calibration.
SKIP_PROMPT_TIMEOUT_SEC = 5.0

# EDIT ME: a saved fbcca_thresholds.json older than this is not even
# offered as a skip option. Electrode contact changes over tens of
# minutes as gel dries and you move, so old thresholds are worse than no
# shortcut at all. Set to None to always offer the skip.
THRESH_MAX_AGE_SEC = 30 * 60

# Every calibration window, raw and unfiltered in microvolts, is saved
# here with its true label. This costs nothing extra -- calibration
# already has the data.
#
# These windows are NOT phase-locked to the display. They are cut on
# timer sleeps while the flicker runs off its own clock, the same as live
# detection. That is fine for FBCCA, which does not care about phase, but
# it means you cannot train TRCA on them as-is: TRCA needs every sample
# cut at the same offset from the start of the flicker cycle, which this
# pipeline does not do yet (see the TRCA-R section at the top of this
# file).
#
# They are still useful as ground truth for tuning, and as real recordings
# to design that clock-sync step against. One .npz file per run.
SESSION_LOG_OUT = "sessions"  # directory; filename gets a timestamp
# ────────────────────────────────────────────────────────────────────────

# The board sends 24-bit signed counts, not volts, and they can run into
# the millions. Numbers that large break the correlation maths, so convert
# to microvolts first.
#
# VREF=4.5 and GAIN=24 match the register values written in
# ads1299_stream.py (0x03 0xEC and CHANNEL_VAL=0x60). If you change those,
# change these two as well.
VREF = 4.5
GAIN = 24
LSB_UV = (2 * VREF) / GAIN / (2 ** 24) * 1e6  # microvolts per raw ADC count


def to_uv(raw_code):
    return raw_code * LSB_UV


def _await_space(timeout_sec):
    """Count down on screen and return True only if SPACE is pressed
    before time runs out.

    Any other key stops the wait straight away and returns False -- once
    the user has answered there is no point counting.

    Reads a single keypress without waiting for Enter, using msvcrt on
    Windows and cbreak mode elsewhere. If input is piped or redirected
    there is no keyboard to read, so it says so and calibrates."""
    if not sys.stdin.isatty():
        print("  (stdin isn't a terminal -- can't read a keypress here, running calibration)")
        return False

    def _render(secs_left):
        print(f"\r  >>> Press SPACE to SKIP calibration -- starting in {secs_left}s ...  ",
              end="", flush=True)

    def _finish(msg):
        print(f"\r  {msg}{' ' * 30}")

    end = time.time() + timeout_sec
    shown = None
    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                remaining = end - time.time()
                if remaining <= 0:
                    break
                secs = int(remaining) + 1
                if secs != shown:
                    _render(secs)
                    shown = secs
                if msvcrt.kbhit():
                    pressed = msvcrt.getch()
                    if pressed == b" ":
                        _finish("SPACE pressed -- skipping calibration, reusing saved thresholds.")
                        return True
                    _finish("Key other than SPACE -- running calibration.")
                    return False
                time.sleep(0.02)
        else:
            import select
            import termios
            import tty
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)  # single keypress, no Enter required
                while True:
                    remaining = end - time.time()
                    if remaining <= 0:
                        break
                    secs = int(remaining) + 1
                    if secs != shown:
                        _render(secs)
                        shown = secs
                    if select.select([sys.stdin], [], [], 0.02)[0]:
                        pressed = sys.stdin.read(1)
                        if pressed == " ":
                            _finish("SPACE pressed -- skipping calibration, reusing saved thresholds.")
                            return True
                        _finish("Key other than SPACE -- running calibration.")
                        return False
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    except Exception as e:
        print(f"\r  (couldn't read a keypress here: {e} -- running calibration){' ' * 20}")
        return False

    _finish("No key pressed -- running calibration.")
    return False


def prompt_skip_calibration(timeout_sec=None, max_age_sec=None):
    """Ask whether to reuse saved thresholds instead of calibrating, and
    return True only on an explicit SPACE press. Never skips on its own:
    a timeout, any other key, a missing thresholds file, or a stale one
    all mean "calibrate". Shared by this module's own main() and by
    run_uno_q.py so both behave identically.
    """
    timeout_sec = SKIP_PROMPT_TIMEOUT_SEC if timeout_sec is None else timeout_sec
    max_age_sec = THRESH_MAX_AGE_SEC if max_age_sec is None else max_age_sec

    if not os.path.exists(THRESH_OUT):
        print(f"\nNo saved {THRESH_OUT} to reuse -- running calibration.")
        return False
    age_sec = time.time() - os.path.getmtime(THRESH_OUT)
    if max_age_sec is not None and age_sec > max_age_sec:
        print(f"\nSaved thresholds are {age_sec / 60:.0f} min old (> "
              f"{max_age_sec // 60:.0f} min) -- too stale to trust (electrode contact and "
              f"impedance drift over a session), so the skip isn't offered. "
              f"Running calibration.")
        return False
    print(f"\nFound saved thresholds ({THRESH_OUT}) from {age_sec / 60:.0f} min ago.")
    return _await_space(timeout_sec)


def connect_and_calibrate(display=None):
    """Connects to the ADS1299, runs mains-notch calibration (Phase 0)
    and threshold calibration (Phase 1), and returns (clf, stream_thread)
    -- pass both to run_live_detection() to continue Phase 2 on the SAME
    connection. Returns (None, None) if the connection never came up.

    display may be None (headless/console-only -- see this module's
    docstring, use case 1) or any object with a set_highlight(freq_or_None)
    method (e.g. run_ssvep_display.py's SSVEPDisplay) -- every call is
    guarded so this works either way.
    """
    clf = FBCCAClassifier(frequencies=FREQUENCIES, sfreq=SAMPLING_RATE, window_sec=WINDOW_SEC,
                           n_harmonics=N_HARMONICS, vote_window=VOTE_WINDOW, vote_needed=VOTE_NEEDED)
    first_sample_event = threading.Event()

    def on_adc_sample(channel_data):
        if not first_sample_event.is_set():
            first_sample_event.set()
        # Always feed the classifier's buffer, not just during calibrate/
        # detect -- calibrate_notch() below needs a full window as soon as
        # possible after connecting, and there's no reason to gate this
        # (push_sample_values() is cheap, no classification work happens
        # until start() is called in Phase 2).
        clf.push_sample_values([to_uv(channel_data[ch]) for ch in CHANNEL_INDICES])

    stream_error = {"exc": None}

    def run_stream_guarded():
        try:
            run_stream(on_adc_sample=on_adc_sample)
        except BaseException as e:
            # BaseException, not Exception: ads1299_stream.parse_adc_payload
            # raises SystemExit on blank-data frames, which is a
            # BaseException and would otherwise slip past this guard and
            # kill the thread silently (stream_error staying None, so the
            # rest of the script would just see "connection dropped (None)"
            # with no explanation of why).
            stream_error["exc"] = e

    stream_thread = threading.Thread(target=run_stream_guarded, daemon=True)
    stream_thread.start()
    print("Connecting + configuring ADS1299 (watch for 'Setup done -- streaming...' above)...")
    if not first_sample_event.wait(timeout=15):
        print("No data received within 15s -- check the device/connection.", file=sys.stderr)
        if display is not None:
            display.stop()
        return None, None
    print("Data flowing.")
    time.sleep(0.5)

    # ---------------- Phase 0: mains notch calibration ----------------
    # Wait for one full window (~WINDOW_SEC) to accumulate, then re-center
    # the notch filter(s) on the ACTUAL interference frequency instead of
    # trusting the nominal 50/60Hz assumption. Measured directly on this
    # rig: a fixed 50.0Hz notch left +11.4dB of residual power on a peak
    # that actually sat at 49.805Hz; re-centering the same filter on the
    # measured peak took suppression from 23.6dB to 34.8dB. The offset also
    # wasn't constant run to run (49.8Hz in three sessions, 50.3Hz in a
    # fourth), so this has to be measured fresh each time, not hardcoded.
    notch_wait_start = time.time()
    while clf.snapshot() is None and time.time() - notch_wait_start < WINDOW_SEC + 3.0:
        time.sleep(0.1)
    win = clf.snapshot()
    if win is not None:
        detected = clf.calibrate_notch(win)
        print("Notch calibration: " + ", ".join(
            f"nominal {nom:.1f}Hz -> measured {det:.2f}Hz"
            for nom, det in zip(clf._nominal_notch_freqs, detected)))
    else:
        print("[WARN] Not enough data yet to calibrate the notch filter -- "
              "keeping nominal frequencies.", file=sys.stderr)

    # ---------------- Phase 1: threshold calibration ----------------
    # NOT model training -- FBCCA has no templates. This only tunes the
    # accept/reject gate (min_confidence / min_ratio) from score statistics
    # collected while you look at each target (and, now, while looking at
    # the center -- see REST-CONDITION CALIBRATION in the module
    # docstring), and reports an honest accuracy estimate before going live.
    if SKIP_CALIBRATION:
        _load_thresholds(clf)
        print(f"\n[Calibration skipped] Using thresholds: "
              f"min_confidence={clf.min_confidence:.3f}, min_ratio={clf.min_ratio:.3f}")
    else:
        print("\n=== THRESHOLD CALIBRATION ===")
        print(f"{clf.describe()}\n")
        trial_order = (list(range(len(FREQUENCIES))) * CAL_TRIALS_PER_FREQ) + ([REST] * CAL_REST_TRIALS)
        random.shuffle(trial_order)

        records_scores, records_labels = [], []
        rest_records_scores = []
        # Raw (unfiltered) windows + labels, saved alongside the tuned
        # thresholds purely as a byproduct of trials you're already running
        # for FBCCA -- no extra hardware time. FBCCA doesn't need these to
        # be phase-locked, so they're not; that also means they are NOT
        # directly usable for TRCA/TRCA-R training as-is (see SESSION_LOG_OUT
        # note in the module docstring) -- but they're real recordings of
        # this subject/session's SSVEP response, which is exactly the
        # ingredient a synthetic simulation can't provide when it comes time
        # to design and tune a TRCA-based hybrid.
        raw_windows, raw_labels = [], []
        for i, true_idx in enumerate(trial_order):
            is_rest = (true_idx == REST)
            freq = None if is_rest else FREQUENCIES[true_idx]
            if stream_error["exc"] is not None or not stream_thread.is_alive():
                print(f"\n[ABORTED] Connection dropped ({stream_error['exc']}). "
                      f"Stopping calibration early -- reconnect and re-run.", file=sys.stderr)
                break

            if is_rest:
                print(f"\n[Trial {i + 1}/{len(trial_order)}] Look at the CENTER "
                      f"(away from every flashing target).")
            else:
                print(f"\n[Trial {i + 1}/{len(trial_order)}] Look at the {freq} Hz target.")
            if display is not None:
                display.set_highlight(freq)  # None for rest trials -- already the "no highlight" state
            for c in (2, 1):
                print(f"  starting in {c}...")
                time.sleep(1.0)

            trial_start = time.time()
            n_before = len(records_scores) + len(rest_records_scores)
            while time.time() - trial_start < CAL_TRIAL_SEC:
                time.sleep(clf.classify_every_sec)
                win = clf.snapshot()
                if win is None:
                    continue  # buffer not full yet (first trial only)
                scores = clf.normalized_scores(win)
                if is_rest:
                    rest_records_scores.append(scores)
                else:
                    records_scores.append(scores)
                    records_labels.append(true_idx)
                raw_windows.append(win)
                raw_labels.append(REST if is_rest else true_idx)
            if display is not None:
                display.set_highlight(None)
            print(f"  captured {len(records_scores) + len(rest_records_scores) - n_before} scored windows.")

        _save_session_log(raw_windows, raw_labels)

        if len(records_scores) < 2 * len(FREQUENCIES):
            print("\n[WARN] Too few calibration windows captured -- falling back to "
                  "default thresholds instead of tuning on thin data.", file=sys.stderr)
            _load_thresholds(clf)
        else:
            scores_arr = np.array(records_scores)
            labels_arr = np.array(records_labels)
            rest_arr = np.array(rest_records_scores) if rest_records_scores else None
            result = tune_thresholds(scores_arr, labels_arr, rest_scores=rest_arr)
            clf.min_confidence = result["min_confidence"]
            clf.min_ratio = result["min_ratio"]
            print(f"\n  Baseline accuracy (best-score guess, no gate): "
                  f"{100 * result['baseline_accuracy']:.1f}%")
            print(f"  Tuned thresholds -> min_confidence={clf.min_confidence:.3f}, "
                  f"min_ratio={clf.min_ratio:.3f}")
            print(f"  Estimated live accuracy at these thresholds: "
                  f"{100 * result['accuracy_at_threshold']:.1f}%  "
                  f"(commits on {100 * result['commit_rate_at_threshold']:.0f}% of windows; "
                  f"the rest correctly abstain rather than guess)")
            if rest_arr is not None:
                print(f"  False-detection rate while looking at the CENTER: "
                      f"{100 * result['rest_false_accept_rate_at_threshold']:.1f}% of "
                      f"{len(rest_records_scores)} rest windows "
                      f"(this is the number that answers 'does it fire when I'm not looking at anything')")
            if result["accuracy_at_threshold"] < 0.75:
                print("  [NOTE] That's below the ~90% target on your own calibration data -- "
                      "check electrode contact/impedance, sit still, and make sure the "
                      "target frequencies clear your alpha band before trusting live output.")
            _save_thresholds(clf, result)

    return clf, stream_thread


def run_live_detection(clf, stream_thread, display=None):
    """Phase 2 -- live detection on an already-calibrated clf/connection
    (from connect_and_calibrate()). Blocks until the connection drops or
    display.stop() ends up being called some other way (e.g. the window
    closing sets a flag stream_thread's loop can't see, but display.stop()
    itself is what actually tears things down on the display side).

    display may be None (console-only -- predictions/idle state still
    print) or any object with set_highlight(freq_or_None)/stop() methods.
    """
    def on_prediction(freq):
        print(f"[SSVEP] Detected target: {freq} Hz  "
              f"(confidence={clf.last_confidence:.2f}, ratio={clf.last_ratio:.2f})")
        if display is not None:
            display.set_highlight(freq)

    last_idle_print = [0.0]

    def on_idle():
        now = time.time()
        if now - last_idle_print[0] > 2.0:
            print(f"[SSVEP] (not confidently looking at anything -- "
                  f"confidence={clf.last_confidence:.2f}, ratio={clf.last_ratio:.2f})")
            last_idle_print[0] = now
        if display is not None:
            display.set_highlight(None)

    clf.on_prediction = on_prediction
    clf.on_idle = on_idle

    print("\n=== LIVE DETECTION ===")
    print(f"{clf.describe()}")
    print("Stare at a target to detect it. Close the window (or Ctrl+C here if headless) to stop.\n")
    clf.start()
    try:
        while stream_thread.is_alive():
            time.sleep(0.5)
    finally:
        clf.stop()
    print("\nConnection ended -- stopping.")
    if display is not None:
        display.stop()


def _save_session_log(raw_windows, raw_labels):
    if not raw_windows:
        return
    try:
        os.makedirs(SESSION_LOG_OUT, exist_ok=True)
        path = os.path.join(SESSION_LOG_OUT, f"session_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        np.savez_compressed(
            path,
            windows=np.stack(raw_windows, axis=0),   # (n_windows, n_channels, window_samples), raw uV
            labels=np.array(raw_labels),              # (n_windows,) int index into `frequencies`
            frequencies=np.array(FREQUENCIES),
            sfreq=SAMPLING_RATE,
            channel_indices=np.array(CHANNEL_INDICES),
        )
        print(f"  Logged {len(raw_windows)} raw calibration windows to {path} "
              f"(not phase-locked -- see SESSION_LOG_OUT note at the top of this file).")
    except OSError as e:
        print(f"[WARN] Could not write session log: {e}", file=sys.stderr)


def _save_thresholds(clf: FBCCAClassifier, result: dict):
    try:
        with open(THRESH_OUT, "w") as f:
            json.dump({
                "frequencies": FREQUENCIES,
                "min_confidence": clf.min_confidence,
                "min_ratio": clf.min_ratio,
                "accuracy_at_threshold": result["accuracy_at_threshold"],
                "commit_rate_at_threshold": result["commit_rate_at_threshold"],
                "rest_false_accept_rate_at_threshold": result.get("rest_false_accept_rate_at_threshold", 0.0),
            }, f, indent=2)
    except OSError as e:
        print(f"[WARN] Could not save thresholds to {THRESH_OUT}: {e}", file=sys.stderr)


def _load_thresholds(clf: FBCCAClassifier):
    """Best-effort: apply previously tuned thresholds if they exist and
    match the current frequency set, else leave the classifier's built-in
    defaults in place."""
    if not os.path.exists(THRESH_OUT):
        return
    try:
        with open(THRESH_OUT) as f:
            saved = json.load(f)
        if saved.get("frequencies") == FREQUENCIES:
            clf.min_confidence = saved["min_confidence"]
            clf.min_ratio = saved["min_ratio"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass  # fall back to defaults silently -- this is a convenience cache, not required


def main():
    """Standalone, console-only: this script drives no display of its own,
    so it needs fbcca_thresholds.json to already exist (copy it from a
    machine that ran the full calibration in main.py). See this module's
    docstring, use case 1.

    Reusing those thresholds is still an explicit choice at the SPACE
    prompt, not automatic. Declining it re-runs calibration console-guided:
    the per-trial instructions ("Look at the 13 Hz target" / "Look at the
    CENTER") still print here, so it works if run_ssvep_display.py is
    showing the targets on another screen -- there's just no green
    highlight to point at the right square for you."""
    if not os.path.exists(THRESH_OUT):
        print(f"[ERROR] {THRESH_OUT} not found -- this script only runs detection, it "
              f"never shows a display to calibrate against. Run main.py once (anywhere) "
              f"to produce {THRESH_OUT}, then copy that file next to this script before "
              f"running it here.", file=sys.stderr)
        sys.exit(1)
    print(f"Detection-only mode -- frequencies {FREQUENCIES}, thresholds file {THRESH_OUT}.")
    global SKIP_CALIBRATION
    SKIP_CALIBRATION = prompt_skip_calibration()
    clf, stream_thread = connect_and_calibrate(display=None)
    if clf is None:
        return
    run_live_detection(clf, stream_thread, display=None)


if __name__ == "__main__":
    main()
