#!/usr/bin/env python3
"""
test/dryrun_no_electrodes.py

Instrumented dry run of `main.py --game`. Walks the EXACT sequence a
real session goes through, using the real parent/child split and the real
ZombieGame, but with fake gaze so it needs no cap or band -- and by default
no hardware at all.

It answers the four things that are hard to eyeball while recording:

  1. Does the game actually open FULLSCREEN and stay alive?
  2. Is the frame budget being met now that the background is cached?
     Fullscreen scanlines used to cost 37% of an 8.33ms frame; caching
     brought that to ~11%. The game prints its own
     "[ZombieGame] present rate ..." line at startup -- that's the check.
  3. Do zombies REALLY stay frozen during calibration? The child starts
     disarmed; this counts actual spawns and asserts zero until armed.
  4. Do the three banners appear on screen, in order?
        "Press SPACE to skip calibration" -> "Calibrating" -> "Live"

Phases (watch the screen alongside the console):
    A   5s  disarmed, skip-prompt banner        -- expect NO zombies
    B  14s  disarmed, "Calibrating", gaze       -- expect NO zombies, even
            cycling like real trials               though a full 10s zombie
                                                   run would have finished
    C  15s  ARMED, "Live", gaze cycling         -- expect zombies to spawn

Usage:
    python3 test/dryrun_no_electrodes.py            # no hardware needed
    python3 test/dryrun_no_electrodes.py --stream   # also connect the ADS1299
                                               # and report samples/sec, so
                                               # you can see display and
                                               # stream coexisting
"""

import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PHASE_A_SEC = 5.0
PHASE_B_SEC = 14.0        # deliberately longer than ZOMBIE_TRAVEL_SEC (10s)
PHASE_C_SEC = 15.0


def _child(cmd_q, stat_q, freqs, layout, ecc, size, dist):
    """The real game child, plus a per-second stats report. Mirrors
    main._display_process's env handling so the SDL backend choice is
    identical to a real run."""
    sys.path.insert(0, _ROOT)
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        for var in ("SDL_VIDEODRIVER", "SSVEP_VIDEO_DRIVER"):
            if os.environ.get(var, "").lower() == "kmsdrm":
                os.environ.pop(var, None)

    import threading

    from ssvep_zombie_game import ZombieGame

    class Reporting(ZombieGame):
        """Counts real spawns by wrapping the spawn call itself, so the
        'no zombies during calibration' claim is measured rather than
        inferred from what happens to be on screen."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.spawns = 0

        def _maybe_spawn(self, active, now):
            before = len(active)
            super()._maybe_spawn(active, now)
            self.spawns += max(0, len(active) - before)

    game = Reporting(frequencies=freqs, layout=layout, eccentricity_deg=ecc,
                      stimulus_size_deg=size, viewing_distance_cm=dist,
                      difficulty="easy", enable_keyboard_debug_gaze=False,
                      fullscreen=True)
    game.set_armed(False)

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

    def reporter():
        last = 0
        while True:
            time.sleep(1.0)
            try:
                stat_q.put_nowait((game.spawns - last, game.spawns,
                                    bool(game._armed)))
            except Exception:
                return
            last = game.spawns

    threading.Thread(target=pump, daemon=True).start()
    threading.Thread(target=reporter, daemon=True).start()
    game.run_mainloop()


def main():
    use_stream = "--stream" in sys.argv
    from config import (ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG,
                         VIEWING_DISTANCE_CM)
    from run_ssvep_detection import FREQUENCIES
    from main import RemoteDisplay

    print("=" * 70)
    print(" DEBUG DRY RUN of main.py --game   (no cap or band needed)")
    print(f" hardware stream: {'ON (--stream)' if use_stream else 'OFF'}")
    print("=" * 70)

    ctx = mp.get_context("spawn")
    cmd_q, stat_q = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_child,
                        args=(cmd_q, stat_q, FREQUENCIES, LAYOUT, ECCENTRICITY_DEG,
                              STIMULUS_SIZE_DEG, VIEWING_DISTANCE_CM),
                        daemon=True)
    proc.start()
    print(f"\n[parent] game child pid {proc.pid}; giving it 5s to open...")
    time.sleep(5.0)
    if not proc.is_alive():
        print("\n*** FAIL: the game child died at startup -- its traceback is "
              "above this line. ***", file=sys.stderr)
        return
    print("[parent] child alive -- the game should be FULLSCREEN now.\n")

    display = RemoteDisplay(cmd_q, proc)
    counts = {"n": 0}
    if use_stream:
        import threading

        from ads1299_stream import run_stream

        def on_sample(_channel_data):
            counts["n"] += 1

        def guarded():
            try:
                run_stream(on_adc_sample=on_sample)
            except BaseException as exc:
                print(f"[stream] died: {exc}", file=sys.stderr)

        threading.Thread(target=guarded, daemon=True).start()
        print("[parent] ADS1299 stream starting...\n")

    spawned = {"A": 0, "B": 0, "C": 0}

    def run_phase(name, secs, banner, armed, gaze):
        display.set_message(banner)
        if armed is not None:
            display.arm(armed)
        print(f"--- PHASE {name}: {secs:.0f}s | banner={banner!r} | armed={armed} ---")
        while True:                      # drop stats left over from the last phase
            try:
                stat_q.get_nowait()
            except Exception:
                break
        end = time.time() + secs
        last_n = counts["n"]
        i = 0
        while time.time() < end:
            if gaze:
                display.set_highlight(FREQUENCIES[i % len(FREQUENCIES)])
            time.sleep(0.5)
            if gaze:
                display.set_highlight(None)
            time.sleep(0.5)
            i += 1
            if not proc.is_alive():
                print(f"*** FAIL: child died during phase {name} ***", file=sys.stderr)
                return False
            line = f"    t+{i:>2}s"
            while True:
                try:
                    new_sp, total_sp, armed_now = stat_q.get_nowait()
                except Exception:
                    break
                spawned[name] += new_sp
                line += f" | spawns +{new_sp} (total {total_sp}) | armed={armed_now}"
            if use_stream:
                rate = counts["n"] - last_n
                last_n = counts["n"]
                line += f" | {rate:>4}/s samples"
            print(line)
        return True

    ok = True
    try:
        ok &= run_phase("A", PHASE_A_SEC, "Press SPACE to skip calibration", False, False)
        ok &= run_phase("B", PHASE_B_SEC, "Calibrating", False, True)
        ok &= run_phase("C", PHASE_C_SEC, "Live", True, True)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        display.stop()
        proc.join(timeout=8)
        if proc.is_alive():
            proc.terminate()

    a_ok, b_ok, c_ok = spawned["A"] == 0, spawned["B"] == 0, spawned["C"] > 0
    print("\n" + "=" * 70)
    print(" RESULT")
    print("=" * 70)
    print(f"  A  disarmed {PHASE_A_SEC:.0f}s   spawns={spawned['A']:<4} "
          + ("PASS (frozen)" if a_ok else "FAIL - spawned while disarmed"))
    print(f"  B  disarmed {PHASE_B_SEC:.0f}s  spawns={spawned['B']:<4} "
          + ("PASS (frozen past a full 10s zombie run)" if b_ok
             else "FAIL - spawned during calibration"))
    print(f"  C  ARMED    {PHASE_C_SEC:.0f}s  spawns={spawned['C']:<4} "
          + ("PASS (play started)" if c_ok else "FAIL - nothing spawned after arming"))
    print()
    if ok and a_ok and b_ok and c_ok:
        print("  ALL PASS -- calibration won't be disturbed by zombies, and play")
        print("  starts the moment calibration ends. Safe to record.")
    else:
        print("  SOMETHING FAILED -- send this output before recording.")
    print("\n  Check by eye too: was it fullscreen? did the banner go")
    print("  A -> B -> C? was the flicker smooth? And scroll up for the")
    print("  '[ZombieGame] present rate ...' line -- that is the fps/vsync check.")


if __name__ == "__main__":
    main()
