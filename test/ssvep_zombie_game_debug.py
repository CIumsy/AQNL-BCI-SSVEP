#!/usr/bin/env python3
"""
test/ssvep_zombie_game_debug.py

The REAL zombie game, unmodified, with per-second performance logging
switched on. Not a simulation and not a copy -- it imports ZombieGame and
sets its `stats_cb` hook, so whatever you measure here is exactly what
the real game does.

Use this to check the game's own performance in isolation, before adding
the detection process on top (test/profile_full_session.py does both together).

WHAT THE COLUMNS MEAN
-----------------------
  fps        frames actually presented per second. With vsync active this
             should sit at the panel rate (~120 on the UNO Q at
             1680x1050). Well below it means the loop can't keep up.
  med / max  frame interval, milliseconds. med should be ~1000/refresh
             (8.3ms at 120Hz). max shows the worst stall in that second.
  render     time spent building the frame BEFORE waiting for vblank --
             i.e. the actual cost of the game's drawing. This is the
             headroom number: at 120Hz the budget is 8.33ms, so render
             should be a small fraction of it. Fullscreen scanlines once
             cost 3.11ms here (37% of budget) before the background was
             cached; it should now be around 1ms.
  drops      frame intervals long enough to have spanned more than one
             panel refresh. Occasional single drops are harmless (the
             stimulus phase catches up); a steady stream is not.
  zomb       zombies alive, so you can correlate cost with load.

Keyboard: hold 1/2/3/4 to fake gaze at a target, E/M/H difficulty,
ESC quits. This is the standalone game, so it is armed from the start.

Usage:
    python3 test/ssvep_zombie_game_debug.py            # fullscreen, like the real thing
    python3 test/ssvep_zombie_game_debug.py --window   # windowed, easier to read the log
    python3 test/ssvep_zombie_game_debug.py --window 45   # ...and stop after 45s
"""

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG,      # noqa: E402
                     VIEWING_DISTANCE_CM)
from run_ssvep_detection import FREQUENCIES                            # noqa: E402
from ssvep_zombie_game import ZombieGame                               # noqa: E402


def main():
    windowed = "--window" in sys.argv
    secs = None
    for a in sys.argv[1:]:
        try:
            secs = float(a)
        except ValueError:
            pass

    print("=" * 78)
    print(" ZOMBIE GAME -- REAL, with performance logging")
    print(f" {'windowed' if windowed else 'FULLSCREEN'}"
          + (f" | auto-stop after {secs:.0f}s" if secs else " | ESC to quit"))
    print("=" * 78)
    print(" hold 1/2/3/4 = fake gaze on a target, E/M/H = difficulty, ESC = quit\n")

    game = ZombieGame(frequencies=FREQUENCIES, layout=LAYOUT,
                       eccentricity_deg=ECCENTRICITY_DEG,
                       stimulus_size_deg=STIMULUS_SIZE_DEG,
                       viewing_distance_cm=VIEWING_DISTANCE_CM,
                       difficulty="easy", enable_keyboard_debug_gaze=True,
                       fullscreen=not windowed)

    hdr = {"n": 0}
    worst = {"max_ms": 0.0, "render_ms": 0.0, "drops": 0, "frames": 0, "fps_min": 1e9}

    def on_stats(s):
        if hdr["n"] % 20 == 0:
            print(f"{'fps':>7} {'med ms':>7} {'max ms':>7} {'render':>7} "
                  f"{'budget':>7} {'drops':>6} {'zomb':>5} {'lives':>6} {'score':>6}")
            print("-" * 68)
        hdr["n"] += 1
        budget = 100.0 * s["render_ms"] / (1000.0 / s["refresh"]) if s["refresh"] else 0.0
        flag = ""
        if s["drops"] > 3:
            flag += "  <- dropping frames"
        if budget > 60:
            flag += "  <- render near budget"
        print(f"{s['fps']:7.1f} {s['median_ms']:7.2f} {s['max_ms']:7.2f} "
              f"{s['render_ms']:7.2f} {budget:6.1f}% {s['drops']:6d} "
              f"{s['zombies']:5d} {s['lives']:6d} {s['score']:6d}{flag}")
        worst["max_ms"] = max(worst["max_ms"], s["max_ms"])
        worst["render_ms"] = max(worst["render_ms"], s["render_ms"])
        worst["drops"] += s["drops"]
        worst["frames"] += s["fps"]
        worst["fps_min"] = min(worst["fps_min"], s["fps"])

    game.stats_cb = on_stats

    if secs:
        t = threading.Timer(secs, game.stop)
        t.daemon = True
        t.start()

    game.run_mainloop()

    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    if worst["frames"]:
        drop_pct = 100.0 * worst["drops"] / max(1.0, worst["frames"])
        print(f"  lowest fps in any second : {worst['fps_min']:.1f}")
        print(f"  worst frame interval     : {worst['max_ms']:.2f} ms")
        print(f"  worst render time        : {worst['render_ms']:.2f} ms")
        print(f"  total dropped frames     : {worst['drops']}  ({drop_pct:.2f}% of frames)")
        print()
        print("  Healthy on the UNO Q looks like: fps near 120, med ~8.3ms,")
        print("  render well under 8.3ms, drops under ~1% of frames.")
    else:
        print("  no stats collected -- did the window open?")


if __name__ == "__main__":
    main()
