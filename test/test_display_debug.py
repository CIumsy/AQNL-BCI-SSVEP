#!/usr/bin/env python3
"""
test/test_display_debug.py

Display ONLY, with verbose per-second timing. No hardware, no electrodes,
no stream -- so it gives you the display's best-case baseline in
isolation. Run it before test_combined_debug.py: if the numbers here are
good but degrade once the stream is added, that's process contention; if
they're bad here too, it's the display/panel itself and the stream is
innocent.

Reference numbers measured on the UNO Q at 1680x1050 @ 119.99Hz:
    measured refresh   119.885 Hz  (-875 ppm from nominal -- good)
    frame drops        0.17%       (~1 drop per 600 frames)
Anything close to that is healthy. Warning signs:
    - "VSync verification FAILED"  -> vsync isn't active; the flicker will
      tear and the stimulus won't match what the classifier assumes
    - drops climbing above ~1%     -> something else is competing for the
      GPU or CPU (browser, ollama, compositor)
    - measured refresh far from the panel's rated figure -> wrong mode set

It also cycles the green highlight and the on-screen text so you can
confirm those render correctly before relying on them for calibration.

Usage:
    python3 test/test_display_debug.py           # 30s
    python3 test/test_display_debug.py 60        # 60s
"""

import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG,     # noqa: E402
                     VIEWING_DISTANCE_CM)
from run_ssvep_detection import FREQUENCIES                           # noqa: E402
from run_ssvep_display_sdl import SSVEPDisplay                        # noqa: E402

TEST_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


def main():
    print("=" * 68)
    print(" DISPLAY-ONLY TIMING CHECK (no hardware needed)")
    print(f" {TEST_SEC:.0f}s | targets {FREQUENCIES}")
    print("=" * 68)
    print("\nLOOK AT THE SCREEN. You should see:")
    print("  - four squares flickering smoothly, no stutter, no tearing")
    print("  - a green outline stepping between them every 3s")
    print("  - centred text changing as it steps")
    print("\nThe display prints its own '[SSVEP timing]' line every 5s.\n")

    disp = SSVEPDisplay(frequencies=FREQUENCIES, layout=LAYOUT,
                         eccentricity_deg=ECCENTRICITY_DEG,
                         stimulus_size_deg=STIMULUS_SIZE_DEG,
                         viewing_distance_cm=VIEWING_DISTANCE_CM)

    def driver():
        """Exercise highlight + text from another thread, exactly the way
        the detection side does, then stop and print the verdict."""
        time.sleep(4.0)         # let it measure refresh first
        end = time.time() + TEST_SEC
        i = 0
        while time.time() < end:
            f = FREQUENCIES[i % len(FREQUENCIES)]
            disp.set_highlight(f)
            disp.set_message(f"Look at the {f:g} Hz target")
            disp.set_status(f"display-only test -- {int(end - time.time())}s left")
            i += 1
            time.sleep(3.0)
        disp.set_highlight(None)
        disp.set_message("")

        missed = getattr(disp, "total_missed", 0)
        frames = getattr(disp, "frames_presented", 0)
        rate = getattr(disp, "refresh_hz", None)
        print("\n" + "=" * 68)
        if frames:
            frac = missed / (frames + missed)
            print(f"  measured refresh : {rate:.3f} Hz" if rate else "  refresh: unknown")
            print(f"  frames presented : {frames}")
            print(f"  frames missed    : {missed}  ({100*frac:.2f}%)")
            print(f"  verdict          : {'PASS' if frac <= 0.01 else 'FAIL'} "
                  f"(want under 1%; UNO Q reference was 0.17%)")
        else:
            print("  no frames presented -- the display never got running")
        print("=" * 68)
        disp.stop()

    threading.Thread(target=driver, daemon=True).start()
    disp.run_mainloop()


if __name__ == "__main__":
    main()
