#!/usr/bin/env python3
"""
run_ssvep_display_sdl.py

Draws the flickering squares on Linux and the Arduino UNO Q, using the
GPU and waiting for the screen's own refresh between frames.
run_ssvep_display.py picks this backend automatically on Linux. On
Windows it keeps the Tk backend, which already works well there.

WHY IT EXISTS
---------------
We measured this rather than assuming it.

The Tk backend redraws on a millisecond timer that has nothing to do with
when the screen actually refreshes. On the UNO Q it ran at 227 draws per
second into a 119.99 Hz panel. Two things went wrong: the image tore,
because a square could change halfway down the screen and the eye then
sees a half-bright frame, and each on/off edge landed a frame early or
late at random.

This backend draws exactly once per screen refresh and works out the
flicker phase from the frame number, so the screen's own clock becomes
the stimulus clock.

Confirmed on the UNO Q: 8.3072 ms between frames against the panel's
8.3340 ms -- 27 microseconds apart, so genuinely locked. 0.51% of frames
were dropped, about 1.2 per 2-second window, and a drop advances the
frame counter by however many refreshes were actually missed, so it never
builds up into drift.

To be honest about the size of the win: simulating both waveforms against
the classifier's own scoring, the fundamental only gains 0.3-1.5% in
amplitude and 1.3-2.6x in phase stability over the Tk backend. The real
benefit is that the tearing is gone, which the simulation cannot model.
Expect a modest accuracy gain, not a dramatic one.

WE MEASURE THE REFRESH RATE INSTEAD OF TRUSTING IT
----------------------------------------------------
The flicker phase is the frame number divided by the refresh rate, so if
that rate is wrong, every frequency we emit is wrong by the same
proportion.

On the UNO Q the system reports 119.99 Hz but the panel actually presents
at about 120.38 Hz. That 0.3% error turns a 17 Hz target into 17.055 Hz
and costs roughly 2% correlation over a 2-second window -- about as much
as locking to the refresh just gained us.

So this file measures the real rate during a warm-up before the stimulus
starts, and uses that. Only pass refresh_hz to the constructor if you
have a better number than the screen reports.

Public API is identical to run_ssvep_display.py's SSVEPDisplay --
set_highlight(freq) / stop() / run_mainloop() -- so it is a drop-in
substitute anywhere a `display` is expected (the root's main.py, this
folder's main.py, run_ssvep_detection.py's helpers).

For the cleanest presentation on XFCE/X11, disable the compositor:
    xfconf-query -c xfwm4 -p /general/use_compositing -s false

Usage (standalone -- just the flashing targets, no highlight, no hardware):
    python3 run_ssvep_display_sdl.py
ESC or closing the window quits.
"""

import re
import statistics
import subprocess
import threading
import time

try:
    import pygame
    from pygame._sdl2.video import Window, Renderer, Texture
except ImportError as exc:
    raise SystemExit(
        "pygame 2.x with SDL2 support is required.\n"
        "Install it with:\n"
        "  python3 -m pip install --upgrade pygame==2.6.1"
    ) from exc

from game_lib.ssvep_geometry import (compute_layout, outward_label_pos,
                                      print_layout_report)

# Last-resort fallback if the mode can't be read from xrandr. The nominal
# rate is ONLY used to size the warm-up and to sanity-check the measured
# rate -- never to compute stimulus phase, which always uses the measured
# value (see the MEASURED REFRESH note above).
#
# This used to be a hand-edited constant, which is a trap: it was left at
# 59.99 after a 60Hz experiment, and the next run at 119.99Hz then failed
# vsync verification ("expected 60, got 120") even though the display was
# working perfectly. Reading the active mode from xrandr instead means
# changing resolution or refresh can't silently invalidate the check.
FALLBACK_REFRESH_HZ = 119.99

# Reject a measured rate outside this band around the nominal as implausible
# (vsync not actually active, or the desktop is at a different mode).
VSYNC_TOLERANCE = 0.15

# Warm-up before measuring: the first frames after window creation are not
# representative. Measured directly on the UNO Q -- the first 40 samples
# gave 115.9Hz while steady state was 120.38Hz, so measuring too early
# would have baked in a 3.7% frequency error.
WARMUP_FRAMES = 60
MEASURE_FRAMES = 180

TIMING_REPORT_SECONDS = 5.0

_FALLBACK_MM_W, _FALLBACK_MM_H = 531.4, 298.9


def _query_refresh_hz():
    """Refresh rate of the mode that is CURRENTLY active, from xrandr.

    xrandr marks the active mode with '*', e.g.
        1680x1050    119.99*+   59.95
    so the starred number is the panel's real nominal rate. Returns it, or
    FALLBACK_REFRESH_HZ if xrandr isn't usable."""
    try:
        out = subprocess.check_output(
            ["xrandr", "--current"], text=True,
            stderr=subprocess.DEVNULL, timeout=2.0,
        )
        for line in out.splitlines():
            if "*" not in line:
                continue
            for tok in line.split():
                if "*" in tok:
                    try:
                        hz = float(tok.replace("*", "").replace("+", ""))
                    except ValueError:
                        continue
                    if 20.0 <= hz <= 400.0:
                        return hz
    except Exception:
        pass
    print(f"[SSVEPDisplay-SDL] could not read the active mode's refresh from xrandr; "
          f"assuming {FALLBACK_REFRESH_HZ:.2f} Hz for the sanity check only "
          f"(stimulus phase still uses the MEASURED rate).")
    return FALLBACK_REFRESH_HZ


def _query_monitor_mm():
    """Physical size of the active X11 output, from xrandr. Returns
    (width_mm, height_mm), or a 24in 16:9 fallback if unavailable."""
    try:
        out = subprocess.check_output(
            ["xrandr", "--current"], text=True,
            stderr=subprocess.DEVNULL, timeout=2.0,
        )
        for line in out.splitlines():
            if " connected " not in f" {line} ":
                continue
            if not re.search(r"\d+x\d+\+\d+\+\d+", line):
                continue  # connected but not currently driving a mode
            match = re.search(r"(\d+)\s*mm\s+x\s+(\d+)\s*mm", line)
            if match:
                mm_w, mm_h = float(match.group(1)), float(match.group(2))
                if 50 <= mm_w <= 3000 and 50 <= mm_h <= 3000:
                    return mm_w, mm_h
    except Exception:
        pass
    print(f"[SSVEPDisplay-SDL] WARNING: could not get physical monitor size from xrandr; "
          f"using fallback {_FALLBACK_MM_W:.0f}x{_FALLBACK_MM_H:.0f}mm. Visual-angle "
          f"sizing will be wrong if that's not close to your real panel.")
    return _FALLBACK_MM_W, _FALLBACK_MM_H


class SSVEPDisplay:
    def __init__(self, frequencies, layout="cross", eccentricity_deg=7.0,
                 stimulus_size_deg=3.5, viewing_distance_cm=60.0, margin_px=70,
                 display_index=0, refresh_hz=None, command_queue=None):
        """refresh_hz: leave None to measure the panel's real rate at
        startup (recommended -- see the module docstring). Pass a number
        only to override that measurement.

        command_queue: optional multiprocessing.Queue. When given, the
        render loop drains it every frame and applies commands, which is
        how the detection process drives this display when the two run as
        separate processes (see main.py, and the GIL note in this
        module's docstring for why they must be separate). Commands are
        ("highlight", freq_or_None) and ("quit",). "message"/"status" are
        still accepted but draw nothing (see set_message). Leaving it None keeps the
        plain in-process set_highlight() behaviour unchanged.
        """
        if layout not in ("cross", "grid"):
            raise ValueError('layout must be "cross" or "grid"')
        if not frequencies:
            raise ValueError("frequencies cannot be empty")

        self.frequencies = [float(f) for f in frequencies]
        self.layout = layout
        self.eccentricity_deg = float(eccentricity_deg)
        self.stimulus_size_deg = float(stimulus_size_deg)
        self.viewing_distance_cm = float(viewing_distance_cm)
        self.margin_px = int(margin_px)
        self.display_index = int(display_index)
        self.refresh_hz = None if refresh_hz is None else float(refresh_hz)
        self._refresh_forced = refresh_hz is not None
        self.command_queue = command_queue

        self.width = self.height = None
        self._highlight = None
        self._message = None      # single static banner, e.g. "Calibrating"
        self._text_cache = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # ---------------- thread-safe control API ----------------
    def set_highlight(self, freq):
        """Thread-safe -- call from the detection worker thread. Same
        contract as run_ssvep_display.py's SSVEPDisplay.set_highlight()."""
        with self._lock:
            self._highlight = None if freq is None else float(freq)

    def set_message(self, text):
        """One short, STATIC banner along the top -- "Calibrating", then
        "Live". Deliberately not per-trial text: something that rewrites
        itself every few seconds pulls the eye away from the target it is
        supposed to be fixating. Which target is active is shown by the
        green outline, not by words. Kept dim and at the top edge, well
        clear of the fovea while the subject looks at a peripheral square."""
        with self._lock:
            self._message = None if text is None else str(text)

    def set_status(self, text):
        """Accepted and ignored -- there is only the one top banner now,
        set via set_message(). Kept so existing callers don't break."""

    def stop(self):
        """Thread-safe -- call from any thread to end run_mainloop()."""
        self._stop_event.set()

    # ---------------- command intake (cross-process) ----------------
    def _drain_commands(self):
        """Apply everything queued by the detection process. Non-blocking:
        this runs inside the vsync-locked render loop, so it must never
        wait -- a blocking get() here would cost dropped frames."""
        if self.command_queue is None:
            return
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except Exception:
                return          # Empty, or the queue's other end went away
            if not cmd:
                continue
            kind = cmd[0]
            arg = cmd[1] if len(cmd) > 1 else None
            if kind == "highlight":
                self.set_highlight(arg)
            elif kind == "message":
                self.set_message(arg)
            elif kind == "status":
                pass    # accepted for compatibility; only the top banner is drawn
            elif kind == "quit":
                self._stop_event.set()
                return

    # ---------------- helpers ----------------
    def _pump_events(self):
        """Returns False if the user asked to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _measure_refresh(self, renderer):
        """Present a static blank screen and time the VBlank-locked
        presents, to learn the panel's true rate before any stimulus
        starts. Discards WARMUP_FRAMES first -- early presents are not
        representative (see WARMUP_FRAMES). Returns Hz, or None if the
        user quit during the warm-up."""
        # Nominal comes from the mode xrandr says is ACTIVE, so changing
        # resolution/refresh can't leave a stale constant behind.
        nominal = _query_refresh_hz()
        print(f"[SSVEPDisplay-SDL] active mode reports {nominal:.2f} Hz; measuring the "
              f"real rate ({WARMUP_FRAMES} warm-up + {MEASURE_FRAMES} timed frames, "
              f"~{(WARMUP_FRAMES + MEASURE_FRAMES) / nominal:.1f}s)...")
        intervals = []
        prev = None
        for i in range(WARMUP_FRAMES + MEASURE_FRAMES):
            if not self._pump_events() or self._stop_event.is_set():
                return None
            renderer.draw_color = (0, 0, 0, 255)
            renderer.clear()
            renderer.present()
            now = time.perf_counter()
            if prev is not None and i >= WARMUP_FRAMES:
                dt = now - prev
                if 0.001 <= dt <= 0.250:   # ignore absurd gaps (scheduler hiccups)
                    intervals.append(dt)
            prev = now

        if len(intervals) < 30:
            print(f"[SSVEPDisplay-SDL] WARNING: only {len(intervals)} usable timing samples; "
                  f"falling back to the mode's nominal {nominal:.2f}Hz.")
            return nominal

        median_dt = statistics.median(intervals)
        measured = 1.0 / median_dt
        rel_err = abs(measured - nominal) / nominal
        print(f"[SSVEPDisplay-SDL] measured {measured:.3f} Hz "
              f"(median {median_dt * 1000:.4f} ms over {len(intervals)} frames), "
              f"nominal {nominal:.2f} Hz -> {1e6 * (measured / nominal - 1):+.0f} ppm")

        if rel_err > VSYNC_TOLERANCE:
            raise RuntimeError(
                f"\nVSync verification FAILED.\n"
                f"The active mode reports {nominal:.2f}Hz but presents are landing at "
                f"{measured:.2f}Hz ({median_dt * 1000:.3f}ms).\n"
                f"If the measured rate looks like your real panel rate, the mode xrandr "
                f"reports is stale -- re-check with 'xrandr | grep \\*'. Otherwise vsync "
                f"probably isn't active: disable the XFCE compositor and make sure this "
                f"is running fullscreen."
            )
        print("[SSVEPDisplay-SDL] VSync verification OK -- using the MEASURED rate for "
              "stimulus phase.")
        return measured

    def _build_targets(self, renderer, screen_w, screen_h):
        mm_w, mm_h = _query_monitor_mm()
        # Per-axis density, NOT the average: when the panel runs at a
        # non-native aspect these differ a lot (48% on the UNO Q at
        # 1680x1050), and averaging draws pixel-squares that are physically
        # rectangular. See compute_layout()'s docstring.
        px_per_mm = (screen_w / mm_w, screen_h / mm_h)
        n = len(self.frequencies)
        # Shared with ssvep_zombie_game.py so target positions stay
        # consistent with whatever the subject calibrated on.
        geo = compute_layout(n, self.eccentricity_deg, self.stimulus_size_deg,
                              self.viewing_distance_cm, px_per_mm,
                              screen_w, screen_h, self.layout, self.margin_px)
        print(f"[SSVEPDisplay-SDL] {screen_w}x{screen_h}px, {mm_w:.0f}x{mm_h:.0f}mm")
        print_layout_report(geo, n, self.viewing_distance_cm, px_per_mm,
                             tag="[SSVEPDisplay-SDL]")

        cx, cy = screen_w / 2.0, screen_h / 2.0
        font = pygame.font.Font(None, 28)
        self._font_banner = pygame.font.Font(None, 34)
        squares = []
        for (dx, dy), freq in zip(geo["offsets"], self.frequencies):
            sw, sh = int(round(geo["size_px_w"])), int(round(geo["size_px_h"]))
            rect = pygame.Rect(int(round(cx + dx - sw / 2.0)),
                                int(round(cy + dy - sh / 2.0)), sw, sh)
            text = f"{int(freq)} Hz" if float(freq).is_integer() else f"{freq:g} Hz"
            surf = font.render(text, True, (255, 255, 255))
            lw, lh = surf.get_width(), surf.get_height()
            # Caption on the OUTER edge -- the side facing away from centre.
            # Keeps it consistent with the game (where the inner edge is
            # occupied by the gun) and puts the text further from the fovea
            # while the subject fixates a peripheral target.
            lcx, lcy = outward_label_pos(cx, cy, dx, dy, sw, sh, lw, lh)
            squares.append({
                "freq": freq,
                "rect": rect,
                "label_texture": Texture.from_surface(renderer, surf),
                "label_rect": pygame.Rect(int(lcx - lw / 2), int(lcy - lh / 2), lw, lh),
            })
        return squares

    def _text_texture(self, renderer, text, colour):
        """Cached text->Texture. The banner changes at most twice a run, so
        rebuilding the texture every frame inside the vsync-locked loop
        would be pure waste."""
        key = (text, colour)
        hit = self._text_cache.get(key)
        if hit is None:
            surf = self._font_banner.render(text, True, colour)
            hit = (Texture.from_surface(renderer, surf),
                   surf.get_width(), surf.get_height())
            if len(self._text_cache) > 16:
                self._text_cache.clear()
            self._text_cache[key] = hit
        return hit

    @staticmethod
    def _draw_outline(renderer, rect, color, width=6):
        renderer.draw_color = color
        for inset in range(width):
            renderer.draw_rect(pygame.Rect(
                rect.x + inset, rect.y + inset,
                max(1, rect.width - 2 * inset), max(1, rect.height - 2 * inset)))

    def _draw_frame(self, renderer, squares, frame_index, highlight, message=None):
        """Draw the state for one physical refresh. Phase comes from
        frame_index / refresh_hz -- the panel's own clock -- not from
        wall-clock time, so the emitted waveform is identical every cycle
        instead of jittering with whenever the draw code happened to run.

        Draws the targets, their "N Hz" captions, and one static banner
        along the top edge (see set_message). Nothing else -- which target
        is active is shown by the green outline, not by text."""
        t = frame_index / self.refresh_hz
        renderer.draw_color = (0, 0, 0, 255)
        renderer.clear()
        for sq in squares:
            on = ((t * sq["freq"]) % 1.0) < 0.5
            renderer.draw_color = (255, 255, 255, 255) if on else (0, 0, 0, 255)
            renderer.fill_rect(sq["rect"])
            lit = highlight is not None and abs(sq["freq"] - highlight) < 1e-9
            self._draw_outline(renderer, sq["rect"],
                                (0, 255, 0, 255) if lit else (50, 50, 50, 255), width=6)
            sq["label_texture"].draw(dstrect=sq["label_rect"])

        if message:
            tex, tw, th = self._text_texture(renderer, message, (170, 170, 180))
            tex.draw(dstrect=pygame.Rect(int(self.width / 2 - tw / 2), 18, tw, th))

    def run_mainloop(self):
        """Builds the fullscreen window and runs the VSync-locked
        presentation loop. BLOCKS until stop()/ESC/window close. Must be
        called from the main thread (SDL video requirement)."""
        pygame.init()
        pygame.font.init()
        window = None
        try:
            sizes = pygame.display.get_desktop_sizes()
            if not sizes:
                raise RuntimeError("SDL could not find a desktop display.")
            if not (0 <= self.display_index < len(sizes)):
                raise ValueError(f"display_index={self.display_index} but SDL sees "
                                  f"{len(sizes)} display(s).")
            screen_w, screen_h = sizes[self.display_index]
            self.width, self.height = int(screen_w), int(screen_h)

            window = Window("SSVEP targets", size=(self.width, self.height))
            window.borderless = True
            window.set_fullscreen(desktop=True)
            try:
                renderer = Renderer(window, accelerated=True, vsync=True)
            except Exception as exc:
                raise RuntimeError(
                    "Could not create a hardware-accelerated VSync SDL2 renderer. "
                    "Not falling back to an unsynchronised path -- that's what this "
                    "backend exists to avoid."
                ) from exc
            pygame.mouse.set_visible(False)

            # Measure BEFORE building/showing the stimulus, so phase is
            # computed from the panel's real rate from frame 0 onward.
            if self._refresh_forced:
                print(f"[SSVEPDisplay-SDL] using caller-supplied refresh "
                      f"{self.refresh_hz:.3f} Hz (measurement skipped)")
            else:
                measured = self._measure_refresh(renderer)
                if measured is None:
                    return  # user quit during warm-up
                self.refresh_hz = measured
            frame_period = 1.0 / self.refresh_hz

            squares = self._build_targets(renderer, self.width, self.height)
            print("[SSVEPDisplay-SDL] stimulus running. Press ESC to quit.")

            # Exposed as instance attributes (not just locals) so a caller or
            # subclass can read live timing health -- e.g. the pre-electrode
            # check in test/ reports these back to the parent process.
            self.total_missed = 0
            self.frames_presented = 0
            frame_index = 0
            last_present = None
            intervals = []
            total_missed = 0
            next_report = time.perf_counter() + TIMING_REPORT_SECONDS

            while not self._stop_event.is_set():
                if not self._pump_events():
                    break
                self._drain_commands()      # non-blocking; see _drain_commands
                if self._stop_event.is_set():
                    break
                with self._lock:
                    highlight = self._highlight
                    message = self._message

                self._draw_frame(renderer, squares, frame_index, highlight, message)
                renderer.present()          # blocks until VBlank
                now = time.perf_counter()

                if last_present is None:
                    last_present = now
                    frame_index = 1
                    continue
                dt = now - last_present
                last_present = now
                if 0.001 <= dt <= 0.250:
                    intervals.append(dt)
                    if len(intervals) > 600:
                        del intervals[:100]

                # Advance by the number of refreshes that ACTUALLY elapsed, so
                # a dropped frame is a one-off glitch rather than permanent
                # phase drift.
                elapsed = max(1, int(round(dt / frame_period)))
                total_missed += elapsed - 1
                frame_index += elapsed
                self.total_missed = total_missed
                self.frames_presented += 1

                if now >= next_report and len(intervals) >= 20:
                    recent = intervals[-240:]
                    med = statistics.median(recent)
                    drops = sum(max(0, int(round(x / frame_period)) - 1) for x in recent)
                    print(f"[SSVEP timing] {1.0 / med:7.3f} Hz | {med * 1000:7.4f} ms/frame | "
                          f"recent missed: {drops} | total missed: {total_missed}")
                    next_report = now + TIMING_REPORT_SECONDS
        finally:
            pygame.mouse.set_visible(True)
            if window is not None:
                try:
                    window.destroy()
                except Exception:
                    pass
            pygame.quit()


if __name__ == "__main__":
    # Standalone: just the flashing targets to look at. No highlight is ever
    # set (no green outline), since nothing is detecting here.
    SSVEPDisplay(frequencies=[7, 13, 15, 17], layout="cross",
                  eccentricity_deg=7.0, stimulus_size_deg=3.5,
                  viewing_distance_cm=60.0).run_mainloop()
