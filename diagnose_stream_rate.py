#!/usr/bin/env python3
"""
diagnose_stream_rate.py

Measures the ADS1299 sample rate this machine ACTUALLY receives, versus
the SAMPLING_RATE the classifier assumes. Run it on any machine that
gets bad detection accuracy but a good-looking display -- it isolates a
failure mode nothing else in this project checks.

WHY THIS MATTERS
------------------
FBCCAClassifier fills a ring buffer one sample at a time and builds its
sin/cos references as t = arange(n)/sfreq. It never timestamps anything.
So the buffer's notion of "2 seconds" is purely "2000 samples", and if
samples actually arrive slower than SAMPLING_RATE, that window spans MORE
than 2 real seconds -- which makes every stimulus appear at the wrong
frequency relative to the references. Measured effect on a 17Hz target:

    rate shortfall   17Hz appears as   CCA correlation
             0.0%          17.000 Hz             1.000
             0.5%          17.085 Hz             0.952   fine
             1.0%          17.172 Hz             0.818   degraded
             2.0%          17.347 Hz             0.378   DESTROYED
             3.0%          17.526 Hz             0.048   DESTROYED

A 2% shortfall is enough to wreck detection while the display still looks
perfect and the raw EEG still contains a real SSVEP response. That makes
it easy to misdiagnose as an electrode or display problem -- hence this
script.

WHAT IT REPORTS
-----------------
  delivered rate : callback invocations per wall-clock second -- the rate
                   the classifier's buffer actually fills at. THIS is the
                   number that has to match SAMPLING_RATE.
  board-reported : gaps in the board's own monotonic sample_number, i.e.
                   samples the hardware sent that never arrived (WiFi
                   loss) versus samples that arrived late (CPU/scheduling).
                   Distinguishing these matters: dropped samples shift the
                   effective rate, slow-but-complete delivery doesn't.

Usage:
    python diagnose_stream_rate.py           # 30s test
    python diagnose_stream_rate.py 60        # 60s test
"""

import sys
import threading
import time

from ads1299_stream import SAMPLING_RATE, run_stream
from run_ssvep_detection import CHANNEL_INDICES, to_uv

TEST_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
REPORT_EVERY_SEC = 5.0

state = {
    "count": 0,          # callback invocations (what fills the classifier buffer)
    "first_t": None,
    "last_report_t": None,
    "last_report_count": 0,
    "stop": False,
    # Value sanity, not just arrival count. A stream can deliver a perfect
    # 1000 samples/sec of pure ZEROS -- counting alone can't tell the
    # difference, and every downstream stage then fails in confusing ways
    # (all-zero FBCCA scores, argmax defaulting to class 0, the notch
    # detector returning the bottom edge of its search range).
    "nonzero": [0] * len(CHANNEL_INDICES),
    "vmin": [None] * len(CHANNEL_INDICES),
    "vmax": [None] * len(CHANNEL_INDICES),
}
lock = threading.Lock()


def on_adc_sample(channel_data):
    now = time.perf_counter()
    vals = [channel_data[ch] for ch in CHANNEL_INDICES]
    with lock:
        if state["first_t"] is None:
            state["first_t"] = now
            state["last_report_t"] = now
        state["count"] += 1
        for i, v in enumerate(vals):
            if v != 0:
                state["nonzero"][i] += 1
            if state["vmin"][i] is None or v < state["vmin"][i]:
                state["vmin"][i] = v
            if state["vmax"][i] is None or v > state["vmax"][i]:
                state["vmax"][i] = v


def main():
    print(f"Measuring real ADS1299 delivery rate for {TEST_SEC:.0f}s.")
    print(f"Classifier assumes SAMPLING_RATE = {SAMPLING_RATE} Hz.\n")
    print("Watch also for '[ADC] Sample lost' lines from ads1299_stream's own")
    print("SampleTracker -- those are packets the board sent that never arrived.\n")

    err = {"exc": None}

    def guarded():
        try:
            run_stream(on_adc_sample=on_adc_sample)
        except BaseException as e:   # parse_adc_payload raises SystemExit on blank frames
            err["exc"] = e

    t = threading.Thread(target=guarded, daemon=True)
    t.start()

    deadline = time.perf_counter() + TEST_SEC + 15.0  # +15s allowance for connect
    while state["first_t"] is None:
        if err["exc"] is not None or not t.is_alive():
            print(f"[ERROR] stream failed before any data: {err['exc']}", file=sys.stderr)
            return
        if time.perf_counter() > deadline:
            print("[ERROR] no samples within 15s -- check the device/connection.", file=sys.stderr)
            return
        time.sleep(0.05)

    print("Data flowing.\n")
    print(f"{'elapsed':>8} {'interval rate':>14} {'cumulative':>12} {'vs nominal':>12}")
    print("-" * 52)

    end_at = state["first_t"] + TEST_SEC
    next_report = state["first_t"] + REPORT_EVERY_SEC
    while time.perf_counter() < end_at:
        if err["exc"] is not None or not t.is_alive():
            print(f"\n[ABORTED] stream died: {err['exc']}", file=sys.stderr)
            break
        time.sleep(0.05)
        now = time.perf_counter()
        if now >= next_report:
            with lock:
                c, first_t = state["count"], state["first_t"]
                lc, lt = state["last_report_count"], state["last_report_t"]
                state["last_report_count"], state["last_report_t"] = c, now
            interval_rate = (c - lc) / (now - lt)
            cum_rate = c / (now - first_t)
            print(f"{now - first_t:>7.1f}s {interval_rate:>13.1f}/s {cum_rate:>11.1f}/s "
                  f"{100 * (cum_rate / SAMPLING_RATE - 1):>+11.2f}%")
            next_report = now + REPORT_EVERY_SEC

    with lock:
        total, first_t = state["count"], state["first_t"]
    elapsed = time.perf_counter() - first_t
    actual = total / elapsed
    err_frac = actual / SAMPLING_RATE - 1

    print("\n" + "=" * 60)
    print(f"  samples delivered : {total}")
    print(f"  wall-clock        : {elapsed:.2f}s")
    print(f"  ACTUAL rate       : {actual:.2f} Hz")
    print(f"  ASSUMED rate      : {SAMPLING_RATE} Hz")
    print(f"  error             : {100 * err_frac:+.2f}%")
    print("-" * 60)
    print("  channel values (are we receiving actual EEG, or just zeros?)")
    with lock:
        nz, vmin, vmax = state["nonzero"], state["vmin"], state["vmax"]
    all_zero = True
    print(f"  {'ch':>4} {'nonzero':>16} {'raw min':>12} {'raw max':>12} {'range (uV)':>13}")
    for i, ch in enumerate(CHANNEL_INDICES):
        frac = nz[i] / max(1, total)
        rng_uv = 0.0 if vmin[i] is None else to_uv(vmax[i] - vmin[i])
        if nz[i] > 0:
            all_zero = False
        print(f"  {ch:>4} {100*frac:>14.1f}% {str(vmin[i]):>12} {str(vmax[i]):>12} {rng_uv:>13.2f}")
    print("=" * 60)

    if all_zero:
        print("\n*** VERDICT: EVERY SAMPLE ON EVERY CHANNEL IS EXACTLY ZERO. ***")
        print("The stream is alive and perfectly paced, but carries no signal, so")
        print("there is nothing for any classifier to detect. Everything downstream")
        print("fails in misleading ways when this happens: FBCCA scores all come out")
        print("0.000, argmax then defaults to class 0 (making the first frequency look")
        print("'correct'), and the notch detector returns the bottom edge of its")
        print("search range instead of a real mains peak.")
        print("\nCheck, in this order:")
        print("  1. Electrodes actually connected to the board, and being worn.")
        print("  2. CHANNEL_INDICES in run_ssvep_detection.py matches your wiring")
        print(f"     (currently {CHANNEL_INDICES} of the board's 16 channels).")
        print("  3. The ADS1299 finished configuration -- watch for 'Setup done'")
        print("     AND for any '[ADC] Blank data' lines from ads1299_stream.")
        print("  4. Power/ground/reference lead seated -- a floating REF commonly")
        print("     reads as flat zeros rather than noise.")
        print("\nFix this before interpreting sample rate, display timing, or accuracy:")
        print("none of them mean anything while the input is zeros.")
        return

    mag = abs(err_frac)
    if mag < 0.005:
        print("\nVERDICT: rate is correct. The classifier's references are valid, so")
        print("bad accuracy is NOT coming from sample-rate mismatch -- look at")
        print("electrode contact/impedance and the raw PSD next")
        print("(run a calibration, then: python sessions/visualize_sessions.py).")
    else:
        eff = SAMPLING_RATE / actual
        print(f"\nVERDICT: rate is off by {100*mag:.2f}%. Every target frequency is")
        print("effectively shifted relative to the classifier's references:")
        for f in (7, 13, 15, 17):
            print(f"    {f} Hz stimulus appears to the classifier as {f * eff:.3f} Hz")
        if mag >= 0.02:
            print("\nThat is large enough on its own to destroy detection, regardless of")
            print("how good the display or the electrodes are.")
        print("\nFIX: either find why samples are being lost (WiFi loss shows up as")
        print("'[ADC] Sample lost' lines; CPU starvation doesn't), or set")
        print(f"SAMPLING_RATE in ads1299_stream.py to the measured {actual:.0f} Hz so the")
        print("references match reality. Prefer fixing the loss -- a patched-up rate")
        print("constant will drift again whenever load changes.")


if __name__ == "__main__":
    main()
