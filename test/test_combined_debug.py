#!/usr/bin/env python3
"""
test/test_combined_debug.py

THE pre-electrode check. Runs the exact parent/child architecture
run_uno_q.py uses -- SDL display in a child process, ADS1299 stream in
the parent -- and reports, once per second, whether BOTH are healthy at
the same time. No electrodes, no calibration, no EEG needed: this
measures sample throughput and frame pacing only, which don't depend on
signal quality at all.

This is the combination that matters. Each half was already fine alone;
what broke was running them together in ONE process, where SDL's
renderer.present() blocks ~8.3ms per frame holding the GIL and starves
the websocket thread. Measured on the UNO Q:

                              sample intake   display frame drops
    detection alone              1001 Hz              --
    display alone                  --                0.51%
    BOTH, one process             428 Hz  (-57%)     5.55%
    BOTH, separate processes     1001 Hz  (+0.10%)   0.17%

So this script exists to confirm you're getting the last row, not the
third one, before you bother putting the cap on.

PASS CRITERIA (printed as a verdict at the end):
    sample rate within +/-0.5% of SAMPLING_RATE
        Anything worse shifts every stimulus relative to the classifier's
        references -- 1% costs ~18% correlation at 17Hz, 2% destroys it.
    display frame drops under 1%
        Occasional drops are harmless (phase catches up); sustained ones
        mean the display is being starved again.

Usage:
    python3 test/test_combined_debug.py            # 30s
    python3 test/test_combined_debug.py 60         # 60s
"""

import multiprocessing as mp
import os
import sys
import time

# Run from the Uno-Q folder OR from inside test/ -- resolve either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ads1299_stream import SAMPLING_RATE, run_stream          # noqa: E402
from config import (ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG,             # noqa: E402
                     VIEWING_DISTANCE_CM)
from run_ssvep_detection import FREQUENCIES                          # noqa: E402

TEST_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

RATE_TOL = 0.005      # +/-0.5%
DROP_TOL = 0.01       # 1% of frames


def _display_child(cmd_q, stat_q, freqs):
    """Child process: the real SDL display, reporting its own frame stats
    back through stat_q so the parent can print one combined view."""
    sys.path.insert(0, _ROOT)
    from run_ssvep_display_sdl import SSVEPDisplay

    class Reporting(SSVEPDisplay):
        """Same display, but pushes (frames, missed, median_dt) upstream
        each second instead of only printing locally."""

        def _drain_commands(self):
            super()._drain_commands()
            now = time.perf_counter()
            last = getattr(self, "_dbg_last", None)
            if last is None:
                self._dbg_last = now
                self._dbg_frames = 0
                self._dbg_missed_at = 0
                return
            self._dbg_frames += 1
            if now - last >= 1.0:
                try:
                    # total_missed is maintained by the base class's present
                    # loop as an instance attribute for exactly this purpose.
                    stat_q.put_nowait(("disp", self._dbg_frames,
                                        getattr(self, "total_missed", 0)))
                except Exception:
                    pass
                self._dbg_frames = 0
                self._dbg_last = now

    disp = Reporting(frequencies=freqs, layout=LAYOUT,
                      eccentricity_deg=ECCENTRICITY_DEG,
                      stimulus_size_deg=STIMULUS_SIZE_DEG,
                      viewing_distance_cm=VIEWING_DISTANCE_CM,
                      command_queue=cmd_q)
    disp.set_message("timing test -- no electrodes needed")
    disp.set_status("watch the squares: flicker should look smooth and steady")
    disp.run_mainloop()


def main():
    print("=" * 68)
    print(" UNO Q PRE-ELECTRODE CHECK -- display child + stream parent")
    print(f" {TEST_SEC:.0f}s | assumed SAMPLING_RATE = {SAMPLING_RATE} Hz | "
          f"targets {FREQUENCIES}")
    print("=" * 68)
    print("\nWhile this runs, LOOK AT THE SCREEN: the four squares should flicker")
    print("smoothly with no visible stutter or tearing. The numbers below tell you")
    print("whether the sample stream survives at the same time.\n")

    ctx = mp.get_context("spawn")
    cmd_q, stat_q = ctx.Queue(), ctx.Queue()
    child = ctx.Process(target=_display_child, args=(cmd_q, stat_q, FREQUENCIES),
                         daemon=True)
    child.start()
    print(f"[parent] display child pid {child.pid}; giving it 4s to measure refresh...")
    time.sleep(4.0)
    if not child.is_alive():
        print("\n[FAIL] display child died during startup -- scroll up for its error.",
              file=sys.stderr)
        return

    count = {"n": 0}

    def on_adc_sample(channel_data):
        count["n"] += 1

    err = {"exc": None}

    def guarded():
        try:
            run_stream(on_adc_sample=on_adc_sample)
        except BaseException as e:
            err["exc"] = e

    import threading
    t = threading.Thread(target=guarded, daemon=True)
    t.start()

    t_wait = time.perf_counter()
    while count["n"] == 0:
        if err["exc"] is not None or not t.is_alive():
            print(f"\n[FAIL] stream died before any data: {err['exc']}", file=sys.stderr)
            cmd_q.put(("quit",))
            return
        if time.perf_counter() - t_wait > 20:
            print("\n[FAIL] no samples within 20s -- check the device/connection.",
                  file=sys.stderr)
            cmd_q.put(("quit",))
            return
        time.sleep(0.05)

    print("[parent] data flowing.\n")
    print(f"{'t':>5} {'sample rate':>13} {'vs nominal':>11} {'display fps':>12} {'missed':>8}")
    print("-" * 55)

    start = time.perf_counter()
    last_t, last_n = start, count["n"]
    rate_hist, fps_hist = [], []
    disp_frames = disp_missed = 0
    last_missed = 0
    highlight_i = 0
    next_hl = start + 3.0

    while time.perf_counter() - start < TEST_SEC:
        time.sleep(1.0)
        now = time.perf_counter()
        n = count["n"]
        rate = (n - last_n) / (now - last_t)
        last_t, last_n = now, n
        rate_hist.append(rate)

        # drain whatever the child reported this second
        fps = None
        while True:
            try:
                kind, frames, total_missed = stat_q.get_nowait()
            except Exception:
                break
            if kind == "disp":
                fps = frames
                disp_frames += frames
                disp_missed = total_missed
        if fps is not None:
            fps_hist.append(fps)

        # exercise the command path too, so we know IPC works under load
        if now >= next_hl:
            cmd_q.put(("highlight", FREQUENCIES[highlight_i % len(FREQUENCIES)]))
            highlight_i += 1
            next_hl = now + 3.0

        d_missed = disp_missed - last_missed
        last_missed = disp_missed
        print(f"{now-start:>4.0f}s {rate:>12.1f}/s {100*(rate/SAMPLING_RATE-1):>+10.2f}% "
              f"{(str(fps) if fps is not None else '-'):>12} {d_missed:>8}")

        if not child.is_alive():
            print("\n[FAIL] display child exited mid-test.", file=sys.stderr)
            break
        if err["exc"] is not None:
            print(f"\n[FAIL] stream died mid-test: {err['exc']}", file=sys.stderr)
            break

    cmd_q.put(("quit",))
    child.join(timeout=6)
    if child.is_alive():
        child.terminate()

    # ---------------- verdict ----------------
    print("\n" + "=" * 68)
    if not rate_hist:
        print(" no data collected")
        return
    mean_rate = sum(rate_hist) / len(rate_hist)
    rate_err = mean_rate / SAMPLING_RATE - 1
    rate_ok = abs(rate_err) <= RATE_TOL
    print(f"  sample rate   : {mean_rate:.1f} Hz  ({100*rate_err:+.2f}%)   "
          f"{'PASS' if rate_ok else 'FAIL'}  (tolerance +/-{100*RATE_TOL:.1f}%)")

    if fps_hist:
        mean_fps = sum(fps_hist) / len(fps_hist)
        est_frames = max(1, disp_frames)
        drop_frac = disp_missed / (est_frames + disp_missed)
        drops_ok = drop_frac <= DROP_TOL
        print(f"  display       : {mean_fps:.1f} fps, {disp_missed} missed / "
              f"~{est_frames} frames = {100*drop_frac:.2f}%   "
              f"{'PASS' if drops_ok else 'FAIL'}  (tolerance {100*DROP_TOL:.0f}%)")
    else:
        drops_ok = False
        print("  display       : no stats received from the child   FAIL")
    print("=" * 68)

    if rate_ok and drops_ok:
        print("\nBoth halves are healthy running together. The parent/child split is")
        print("working -- go ahead and connect the electrodes, then run:")
        print("    python3 run_uno_q.py")
    else:
        if not rate_ok:
            eff = SAMPLING_RATE / mean_rate
            print("\nSample intake is being starved. Every stimulus would be shifted:")
            for f in FREQUENCIES:
                print(f"    {f} Hz would look like {f*eff:.2f} Hz to the classifier")
            print("Do NOT calibrate until this passes -- the thresholds would be garbage.")
        if not drops_ok:
            print("\nThe display is dropping too many frames; the flicker the subject")
            print("sees is not the flicker the classifier assumes.")
        print("\nIf either failed, close other apps (browser, ollama) and re-run;")
        print("if it persists, send this output.")


if __name__ == "__main__":
    main()
