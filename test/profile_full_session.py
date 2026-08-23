#!/usr/bin/env python3
"""
test/profile_full_session.py

The REAL run_uno_q session -- real ADS1299 stream, real calibration, real
detection, real game -- with per-second performance logging added. This is
not a dry run: put the electrodes on and it behaves exactly like
`python3 run_uno_q.py`, just noisier.

It exists because the two halves can only starve each other while both are
actually working, and the numbers that matter are on opposite sides of a
process boundary:

    parent process : samples/sec arriving from the ADS1299
    child process  : fps, frame interval, render time, dropped frames

Both are printed on ONE line per second so you can see them interact. That
is the whole point -- the failure this catches is display and stream
degrading each other, which is invisible if you measure them separately.

WHAT TO LOOK FOR
------------------
  sps        samples/sec reaching the classifier's buffer. Must sit within
             ~0.5% of SAMPLING_RATE. FBCCA never timestamps anything, so a
             shortfall silently shifts every stimulus frequency: 1% off
             costs ~18% correlation at 17Hz, 2% destroys it.
  fps        frames presented per second by the game/display child. Should
             be near the panel rate (~120 on the UNO Q).
  med/max    frame interval ms. med ~8.3ms at 120Hz; max is the worst stall.
  render     frame build time before waiting for vblank -- the headroom
             number against the 8.3ms budget.
  drops      intervals that spanned more than one refresh.

Usage:
    python3 test/profile_full_session.py            # detection only
    python3 test/profile_full_session.py --game     # calibration then the game
"""

import multiprocessing as mp
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run_ssvep_detection as det                                      # noqa: E402
from ads1299_stream import SAMPLING_RATE                               # noqa: E402
from config import (ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG,       # noqa: E402
                     VIEWING_DISTANCE_CM)
from run_uno_q import RemoteDisplay                                    # noqa: E402


def _debug_child(cmd_q, stat_q, freqs, layout, ecc, size, dist, game_mode, difficulty):
    """Same as run_uno_q._display_process, plus the stats hook wired to
    stat_q. Kept in step with that function deliberately -- if you change
    the real child, change this too."""
    sys.path.insert(0, _ROOT)
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        for var in ("SDL_VIDEODRIVER", "SSVEP_VIDEO_DRIVER"):
            if os.environ.get(var, "").lower() == "kmsdrm":
                os.environ.pop(var, None)

    def push(s):
        try:
            stat_q.put_nowait(s)
        except Exception:
            pass

    if game_mode:
        from ssvep_zombie_game import ZombieGame
        game = ZombieGame(frequencies=freqs, layout=layout, eccentricity_deg=ecc,
                           stimulus_size_deg=size, viewing_distance_cm=dist,
                           difficulty=difficulty, enable_keyboard_debug_gaze=False,
                           fullscreen=True,
                           # Kept identical to run_uno_q.py so this measures
                           # the real thing, including the env override.
                           render_scale=float(os.environ.get("SSVEP_RENDER_SCALE", 1.0)))
        game.set_armed(False)
        game.stats_cb = push

        def pump():
            while True:
                try:
                    cmd = cmd_q.get()
                except Exception:
                    return
                if not cmd:
                    continue
                kind = cmd[0]
                arg = cmd[1] if len(cmd) > 1 else None
                if kind == "quit":
                    game.stop()
                    return
                if kind == "highlight":
                    game.set_highlight(arg)
                elif kind == "message":
                    game.set_message(arg)
                elif kind == "arm":
                    game.set_armed(bool(arg))

        threading.Thread(target=pump, daemon=True).start()
        game.run_mainloop()
    else:
        # The plain SDL display has no stats_cb; it prints its own
        # "[SSVEP timing]" line every 5s, which covers the same ground.
        from run_ssvep_display_sdl import SSVEPDisplay
        SSVEPDisplay(frequencies=freqs, layout=layout, eccentricity_deg=ecc,
                      stimulus_size_deg=size, viewing_distance_cm=dist,
                      command_queue=cmd_q).run_mainloop()


def main():
    game_mode = "--game" in sys.argv
    freqs = det.FREQUENCIES

    print("=" * 84)
    print(" run_uno_q DEBUG -- real hardware, real calibration, real detection")
    print(f" mode = {'GAME' if game_mode else 'detection'} | "
          f"SAMPLING_RATE = {SAMPLING_RATE} Hz | targets {freqs}")
    print("=" * 84)

    ctx = mp.get_context("spawn")
    cmd_q, stat_q = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_debug_child,
                        args=(cmd_q, stat_q, freqs, LAYOUT, ECCENTRICITY_DEG,
                              STIMULUS_SIZE_DEG, VIEWING_DISTANCE_CM,
                              game_mode, "easy"),
                        daemon=True)
    proc.start()
    print(f"\n[parent] display child pid {proc.pid}")
    display = RemoteDisplay(cmd_q, proc)
    time.sleep(2.5)
    if not proc.is_alive():
        print("\n*** child died at startup -- traceback above ***", file=sys.stderr)
        return

    # ---- count samples as they reach the classifier ----
    # to_uv() is the last thing every sample passes through before it
    # lands in the FBCCA ring buffer, so patching it counts exactly what
    # the classifier actually receives -- not what the socket delivered.
    # It fires once per CHANNEL, so divide by the channel count.
    _orig_to_uv = det.to_uv
    n_ch = len(det.CHANNEL_INDICES)
    conversions = {"n": 0}

    def counting_to_uv(raw):
        conversions["n"] += 1
        return _orig_to_uv(raw)

    det.to_uv = counting_to_uv
    monitor_stop = threading.Event()

    def monitor():
        """One line per second combining BOTH processes: samples/sec from
        the parent, frame stats from the child. Seeing them together is
        the point -- each looks fine alone even when they're starving
        each other."""
        last = 0
        last_t = time.perf_counter()
        hdr = 0
        worst = {"max_ms": 0.0, "drops": 0, "frames": 0.0, "sps_err": 0.0}
        while not monitor_stop.is_set():
            time.sleep(1.0)
            # Divide by the REAL elapsed interval, not an assumed 1.0s.
            # time.sleep() overshoots slightly, so assuming exactly one
            # second made the rate occasionally read ~275/s and trip a
            # bogus "SPS OFF" warning when nothing was actually wrong.
            now_t = time.perf_counter()
            elapsed = max(1e-6, now_t - last_t)
            last_t = now_t
            n = conversions["n"] // n_ch
            sps = int(round((n - last) / elapsed))
            last = n
            gs = None
            while True:                       # keep only the newest report
                try:
                    gs = stat_q.get_nowait()
                except Exception:
                    break
            if hdr % 20 == 0:
                print(f"\n{'sps':>6} {'err':>7} | {'fps':>7} {'med':>7} {'max':>7} "
                      f"{'render':>7} {'drops':>6} {'zomb':>5} {'armed':>6}")
                print("-" * 76)
            hdr += 1
            err = 100.0 * (sps / SAMPLING_RATE - 1) if SAMPLING_RATE else 0.0
            line = f"{sps:6d} {err:+6.2f}%"
            if gs:
                line += (f" | {gs['fps']:7.1f} {gs['median_ms']:7.2f} {gs['max_ms']:7.2f} "
                         f"{gs['render_ms']:7.2f} {gs['drops']:6d} "
                         f"{gs.get('zombies', 0):5d} {str(gs.get('armed', '-')):>6}")
                worst["max_ms"] = max(worst["max_ms"], gs["max_ms"])
                worst["drops"] += gs["drops"]
                worst["frames"] += gs["fps"]
            else:
                line += " |   (no child stats this second)"
            if sps > 0 and abs(err) > 0.5:
                line += "  <- SPS OFF, references invalid"
                worst["sps_err"] = max(worst["sps_err"], abs(err))
            print(line)
        print("\n" + "=" * 76)
        print(" SUMMARY")
        print("=" * 76)
        if worst["frames"]:
            print(f"  worst frame interval : {worst['max_ms']:.1f} ms")
            print(f"  dropped frames       : {worst['drops']} of ~{worst['frames']:.0f} "
                  f"({100*worst['drops']/max(1.0, worst['frames']):.2f}%)")
        if worst["sps_err"]:
            print(f"  WORST sps error      : {worst['sps_err']:.2f}%  -- calibration from")
            print( "                         this run should NOT be trusted")
        else:
            print("  sample rate stayed within 0.5% the whole run")

    threading.Thread(target=monitor, daemon=True).start()

    try:
        display.set_message("Press SPACE to skip calibration")
        print("\n>>> Press SPACE at the prompt to skip calibration.\n")
        det.SKIP_CALIBRATION = det.prompt_skip_calibration()

        t0 = time.time()
        display.set_message("Live" if det.SKIP_CALIBRATION else "Calibrating")
        clf, stream_thread = det.connect_and_calibrate(display)
        if clf is None:
            return
        print(f"\n[timing] connect + calibration took {time.time()-t0:.1f}s")

        display.set_message("Live")
        display.arm(True)
        if game_mode:
            print("\n=== LIVE GAMEPLAY (armed) ===")
        det.run_live_detection(clf, stream_thread, display)
    except KeyboardInterrupt:
        print("\n[profile_full_session] interrupted.")
    finally:
        monitor_stop.set()
        det.to_uv = _orig_to_uv
        display.stop()
        proc.join(timeout=6)
        if proc.is_alive():
            proc.terminate()
        time.sleep(1.2)      # let the monitor print its summary
        print("[profile_full_session] done.")


if __name__ == "__main__":
    main()
