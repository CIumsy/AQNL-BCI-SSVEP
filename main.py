#!/usr/bin/env python3
"""
main.py

The one command that runs everything on the Arduino UNO Q. One command
for you, but two separate OS processes underneath -- and that split is
the whole point of this file.

WHY TWO PROCESSES
-------------------
We measured this; it was not a style choice. Running the display and the
EEG pipeline in a single Python process does not work on this board.

Every frame, SDL waits about 8.3 ms for the screen's next refresh, and it
holds Python's global interpreter lock the whole time. That leaves no
room for the thread receiving EEG over the network. Measured on the
UNO Q, back when the nEXG ran at 1000 Hz (see SAMPLING_RATE in
ads1299_stream.py -- since lowered to 250 Hz for other reasons; the GIL
contention below starves the receive thread by the same proportion
regardless of the sample rate):

                              EEG samples in   frames dropped
    decoding alone               1001 Hz              --
    display alone                  --                0.51%
    both, one process             428 Hz  (-57%)     5.55%
    both, separate processes     1001 Hz  (+0.10%)   0.17%

One process broke a third thing too: the display measured its own refresh
rate as 117.70 Hz when it was really 119.885 Hz. Stimulus phase is
computed from that rate, so every frequency we emitted was about 1.9%
wrong.

Separate processes get separate interpreter locks, and the other CPU
cores were idle anyway, so splitting fixes all three problems at once.
This is why the repository root's main.py -- the Windows entry point,
single-process and correct there -- must not be used on this board.

The display owns its own process and its own screen. It shows the green
highlight when a target is detected, one fixed "Calibrating"/"Live"
banner, and the game. Per-trial instructions stay in the terminal on
purpose: text that rewrites itself every few seconds pulls the subject's
eye away from the square they are supposed to be staring at.

    main.py  (you run this)
       |
       +-- child process : SDL display -- flicker + green highlight + one
       |                                  static banner, or the zombie game
       |        ^ multiprocessing.Queue (commands)
       |        |
       +-- parent process: ADS1299 stream + FBCCA -- sends "highlight" and
                                                     the "Calibrating"/"Live" banner

Usage:
    python3 main.py            # calibration + live detection
    python3 main.py --game     # calibration, then the zombie game
"""

import multiprocessing as mp
import os
import sys
import time

import run_ssvep_detection as det
from config import ECCENTRICITY_DEG, LAYOUT, STIMULUS_SIZE_DEG, VIEWING_DISTANCE_CM


# ---------------------------------------------------------------- display side
def _display_process(cmd_q, freqs, layout, ecc, size, dist, game_mode, difficulty):
    """Runs in the CHILD process and owns the screen for its whole life.

    pygame and SDL are imported here and nowhere else. The parent must
    never touch them, so no graphics state can be inherited across the
    process boundary."""
    if game_mode:
        # On a desktop, let SDL use X11. Drop any leftover kmsdrm
        # setting: the desktop already owns the display hardware, so
        # kmsdrm cannot start and the game window never appears.
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            if os.environ.get("SDL_VIDEODRIVER", "").lower() == "kmsdrm":
                os.environ.pop("SDL_VIDEODRIVER", None)
            if os.environ.get("SSVEP_VIDEO_DRIVER", "").lower() == "kmsdrm":
                os.environ.pop("SSVEP_VIDEO_DRIVER", None)

        from ssvep_zombie_game import ZombieGame
        game = ZombieGame(frequencies=freqs, layout=layout, eccentricity_deg=ecc,
                           stimulus_size_deg=size, viewing_distance_cm=dist,
                           difficulty=difficulty, enable_keyboard_debug_gaze=False,
                           fullscreen=True,
                           # Full resolution. The game draws on the GPU now,
                           # using the same API as the plain display, which
                           # holds 120 Hz at full resolution on this board.
                           # That removed the 11.5 ms per frame that used to
                           # drag it down to 62.9 Hz. SSVEP_RENDER_SCALE=0.5
                           # is still here as an escape hatch if something
                           # makes it heavy again: it trades sharpness for
                           # speed, and does not change how big the targets
                           # physically are.
                           render_scale=float(os.environ.get("SSVEP_RENDER_SCALE", 1.0)))
        # Start frozen. The squares still flicker so calibration can
        # run, but no zombies appear until the parent arms the game.
        game.set_armed(False)

        # Read the queue on a helper thread. SDL requires the game loop
        # to stay on this process's main thread, and the game has no way
        # to poll a queue by itself.
        import threading

        def pump():
            while True:
                try:
                    cmd = cmd_q.get()
                except Exception:
                    return
                if not cmd:
                    continue
                if cmd[0] == "quit":
                    game.stop()
                    return
                if cmd[0] == "message":
                    game.set_message(cmd[1] if len(cmd) > 1 else None)
                if cmd[0] == "arm":
                    game.set_armed(bool(cmd[1]) if len(cmd) > 1 else True)
                if cmd[0] == "highlight":
                    game.set_highlight(cmd[1] if len(cmd) > 1 else None)

        threading.Thread(target=pump, daemon=True).start()
        game.run_mainloop()
    else:
        from run_ssvep_display_sdl import SSVEPDisplay
        SSVEPDisplay(frequencies=freqs, layout=layout, eccentricity_deg=ecc,
                      stimulus_size_deg=size, viewing_distance_cm=dist,
                      command_queue=cmd_q).run_mainloop()


class RemoteDisplay:
    """A stand-in for the display, used by the decoding side.

    It offers the same set_highlight()/stop() methods as every other
    display backend, so connect_and_calibrate() and run_live_detection()
    in run_ssvep_detection.py drive it without changes -- they never find
    out they are talking to another process.

    set_message() sets the one fixed banner. set_status() is accepted for
    compatibility but is no longer drawn."""

    def __init__(self, cmd_q, proc):
        self._q = cmd_q
        self._proc = proc

    def _send(self, *cmd):
        try:
            self._q.put_nowait(cmd)
        except Exception:
            pass    # display already exited; let decoding finish cleanly

    def set_highlight(self, freq):
        self._send("highlight", freq)

    def set_message(self, text):
        self._send("message", text)

    def set_status(self, text):
        self._send("status", text)

    def arm(self, on=True):
        """Game mode only: lets zombies start spawning. The plain
        display ignores this, since it has nothing to arm."""
        self._send("arm", on)

    def stop(self):
        self._send("quit")

    def alive(self):
        return self._proc.is_alive()


def main():
    game_mode = "--game" in sys.argv
    freqs = det.FREQUENCIES

    # Use 'spawn' rather than Linux's default 'fork'. The child starts
    # up SDL and needs a clean interpreter to do it. Forking a parent that
    # has already loaded graphics libraries or started threads is a
    # well-known way to get deadlocks and duplicated state.
    ctx = mp.get_context("spawn")
    cmd_q = ctx.Queue()
    proc = ctx.Process(
        target=_display_process,
        args=(cmd_q, freqs, LAYOUT, ECCENTRICITY_DEG, STIMULUS_SIZE_DEG,
              VIEWING_DISTANCE_CM, game_mode, "easy"),
        daemon=True,
    )
    proc.start()
    print(f"[main] display process started (pid {proc.pid}), "
          f"mode = {'GAME' if game_mode else 'detection'}")

    display = RemoteDisplay(cmd_q, proc)
    time.sleep(2.5)   # give the child time to measure the refresh rate first

    try:
        # Put the skip prompt on screen too. During the countdown the
        # subject is looking at the monitor, not the terminal.
        display.set_message("Press SPACE to skip calibration")
        print("\n>>> Press SPACE at the first prompt to skip calibration "
              "and reuse recent thresholds.\n")
        det.SKIP_CALIBRATION = det.prompt_skip_calibration()

        # Only two banners for the whole run. Per-trial text was removed
        # on purpose: a caption that changes every few seconds pulls the
        # eye off the square it is supposed to be staring at. The green
        # outline shows which target to look at, and the terminal still
        # prints the full prompt for each trial.
        display.set_message("Live" if det.SKIP_CALIBRATION else "Calibrating")
        clf, stream_thread = det.connect_and_calibrate(display)
        if clf is None:
            return
        display.set_message("Live")
        # Calibration is finished, so now the zombies may start.
        display.arm(True)
        if game_mode:
            print("\n=== LIVE GAMEPLAY ===")
        det.run_live_detection(clf, stream_thread, display)
    except KeyboardInterrupt:
        print("\n[main] interrupted.")
    finally:
        display.stop()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
        print("[main] done.")


if __name__ == "__main__":
    main()
