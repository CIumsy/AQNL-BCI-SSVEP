#!/usr/bin/env python3
"""
ssvep_zombie_game.py

A retro "defend the targets" game you play by looking at things.

Zombies appear in the middle of the screen and walk outwards towards one
of the flickering squares. Stare at the square a zombie is heading for
and its turret shoots it. A zombie that reaches its square costs you a
life.

The squares are the same 7/13/15/17 Hz targets, in the same positions, as
the plain display. Both files get their layout from
game_lib/ssvep_geometry.py, so the targets are pixel-identical to
whatever the subject calibrated on -- otherwise the calibration would not
transfer.

RULES
-------
- Zombies take ZOMBIE_TRAVEL_SEC (10s) to walk from center to target.
- Difficulty sets the max number of zombies concurrently alive, and when
  the next one spawns (both selectable live, bottom-left UI or E/M/H):
    easy   : cap 1  -- a new zombie spawns as soon as the screen is empty.
    medium : cap 2  -- 2nd zombie spawns once the oldest active zombie
             reaches 80% of its path.
    hard   : cap 3  -- 2nd zombie spawns at the oldest active zombie's
             50% mark, 3rd at its 90% mark.
  This is evaluated continuously (not just once per wave), so play
  continues indefinitely: whenever the active count is below the cap and
  the oldest active zombie has crossed the next threshold, another spawns.
- Each zombie is assigned a target square not already targeted by another
  active zombie (there are always more squares than the hardest cap, so
  this never blocks).
- Losing all lives ends the run; score = zombies killed.

WHY ZOMBIES HAVE HEALTH BARS
------------------------------
Detection is not instant. The classifier only answers once several
windows agree, and stays quiet the rest of the time.

Rather than fight that delay, the game uses it as the aiming mechanic:
each zombie has a health bar that drains while your gaze is detected on
its square, and it dies after KILL_GAZE_SEC of gaze in total.

Damage never heals. That is deliberate. The classifier going quiet is it
working correctly, not you making a mistake, so a bar that refilled would
punish the system for behaving as designed and make kills feel impossible
whenever detection stuttered. Every window it is sure about is progress
you keep.

Only one zombie takes damage at a time, even on medium and hard. Choosing
which one to shoot first is part of the game, not a limitation.

REAL BCI INTEGRATION
----------------------
Same set_highlight(freq_or_None) contract as run_ssvep_display.py's
SSVEPDisplay -- ZombieGame is a drop-in substitute for it, which is what
lets run_ssvep_detection.py's run_live_detection() drive either one
unchanged. See run_uno_q.py for the real, working wire-up (calibrates
first, then arms the game, all on one hardware connection). Minimal
pattern:

    game = ZombieGame(frequencies=[7, 13, 15, 17], layout="cross",
                       eccentricity_deg=7.0, stimulus_size_deg=3.5,
                       viewing_distance_cm=60.0,
                       enable_keyboard_debug_gaze=False)   # real runs: BCI only

    clf.on_prediction = lambda freq: game.set_highlight(freq)
    clf.on_idle = lambda: game.set_highlight(None)
    clf.start()

    game.run_mainloop()      # BLOCKS on the main thread until quit/closed

STANDALONE (KEYBOARD) TESTING -- no EEG hardware needed
-----------------------------------------------------------
    python ssvep_zombie_game.py
Press and hold 1/2/3/4 to simulate gazing at that target (numbered
left-to-right, top-to-bottom in spawn order == self.frequencies order);
release to simulate looking away. Press E/M/H to change difficulty,
click the buttons bottom-left, or override SSVEP_VIDEO_DRIVER (e.g.
SSVEP_VIDEO_DRIVER=dummy for a headless smoke test). Esc / close window
to quit.
enable_keyboard_debug_gaze defaults to True so this always works out of
the box; explicitly set it False when wiring up a real BCI session so a
stray keypress can't be mistaken for a detection.
"""

import math
import os
import random
import sys
import threading
import time

from game_lib.ssvep_geometry import (compute_layout, outward_label_pos,
                                      print_layout_report)

_IS_LINUX = sys.platform.startswith("linux")

# Total seconds from spawn to reaching the target square. The spawn-blink
# phase below happens INSIDE this budget, so "spawn to reach" really is
# this number regardless of how long the blink lasts.
ZOMBIE_TRAVEL_SEC = 10.0

# On spawn the zombie sits at the centre and GROWS from nothing to full
# size before it starts moving -- a telegraph so the player sees where the
# next one came from instead of it appearing mid-flight.
#
# This was originally an on/off blink (3 blinks over 1.2s). That had to
# go: a 2.5Hz square wave carries strong odd harmonics, and 3x/5x/7x land
# on 7.5, 12.5 and 17.5Hz -- colliding with three of the four targets. A
# smooth scale-in conveys the same thing with no periodic luminance change
# at all, which is the rule for every non-target element on screen.
SPAWN_TELEGRAPH_SEC = 1.2

# Once a zombie reaches its square it stops being shootable, moves INSIDE
# the square and detonates. Purely a death animation -- the life is
# deducted the moment it arrives, not when the animation ends, so the
# player can't be saved by a late shot during it.
EXPLODE_SEC = 0.55

# Muzzle flash at the gun while it fires -- the bright root of the beam
# defined just below.
#
# The flame is a CONSTANT size and colour. It simply appears while the gun
# is firing and vanishes when it stops -- one state change, not a
# repeating one. An earlier version jittered its size every frame; even
# though that jitter was meant to be incoherent, it still put changing
# luminance next to a target. The only thing on this screen allowed to
# change periodically is the four stimulus squares.
MUZZLE_PX = 11

# The flame BEAM that runs from the muzzle to the zombie being shot. Same
# rule as the muzzle flash and for the same reason: it is one baked sprite
# with a fixed red -> orange -> yellow ramp across its thickness, constant
# along its length, and it does not animate. It only appears while that
# gun is firing and vanishes when it stops. Nothing about it repeats, so
# it contributes no frequency of its own to the occipital signal.
BEAM_TH = 15           # beam thickness in px (the full flame envelope)
BEAM_STRIP_W = 8       # baked strip width; stretched to the shot's length

# Total seconds of accumulated gaze needed to kill a zombie. Damage is
# CUMULATIVE and PERMANENT -- health never regenerates when you look away
# (there is deliberately no "unlock"/decay constant here any more). Real
# FBCCA output is intermittent by design: it commits only when the vote
# window agrees and abstains otherwise, so a decaying bar would punish the
# classifier's normal behaviour and make kills feel impossible whenever
# detection stuttered. With permanent damage, every confidently-detected
# window is progress you keep.
KILL_GAZE_SEC = 0.66   # was 1.5; reduced to 44% -- kills were taking too long.
# The health BAR is unchanged in size and still spans a full 1.0 -> 0.0;
# only how fast it drains changed, so it reads exactly as before.
STARTING_LIVES = 5
SHOT_FLASH_SEC = 0.15

DIFFICULTIES = {
    "easy":   {"cap": 1, "thresholds": []},
    "medium": {"cap": 2, "thresholds": [0.8]},
    "hard":   {"cap": 3, "thresholds": [0.5, 0.9]},
}
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DEBUG_KEY_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}  # pygame K_1..K_4 -> target index, set below

# 8x8 pixel-art zombie, scaled up at draw time -- 'G'=body green, 'D'=dark
# shading, 'w'=eye, ' '=transparent. No external art assets.
ZOMBIE_SPRITE = [
    "  DGGD  ",
    " DGGGGD ",
    " GwGGwG ",
    " GGGGGG ",
    " DGGGGD ",
    "GGGGGGGG",
    "G  GG  G",
    "G  GG  G",
]
_ZOMBIE_COLORS = {"G": (60, 160, 70), "D": (30, 100, 40), "w": (230, 230, 230)}


class Zombie:
    """Lifecycle: blink (stationary at centre) -> move -> either killed by
    gaze, or arrive and explode. Only the "move" phase is shootable."""

    __slots__ = ("target_freq", "spawn_time", "health", "square_idx", "exploded_at")

    def __init__(self, target_freq, square_idx, spawn_time):
        self.target_freq = target_freq
        self.square_idx = square_idx
        self.spawn_time = spawn_time
        self.health = 1.0  # 1.0 = full, drains toward 0.0 under gaze; never regenerates
        self.exploded_at = None   # set when it reaches the square

    def age(self, now):
        return now - self.spawn_time

    def progress(self, now):
        """Overall 0..1 from spawn to arrival, INCLUDING the blink phase.
        The difficulty spawn thresholds are expressed against this, so
        "2nd zombie at 80%" means 80% of the full 10s, not of the travel
        leg alone."""
        return min(1.0, self.age(now) / ZOMBIE_TRAVEL_SEC)

    def is_spawning(self, now):
        return self.age(now) < SPAWN_TELEGRAPH_SEC

    def travel_frac(self, now):
        """0..1 position along the path from the screen centre to the
        target's MUZZLE (see the square's "path_end"). Stays at 0 for the
        whole blink phase, then covers the distance in the remaining time.

        1.0 means "arrived at the gun", not "at the centre of the square".
        Clamped, so a zombie can never walk past its own turret and force
        the gun to shoot backwards at it."""
        t = self.age(now) - SPAWN_TELEGRAPH_SEC
        span = max(1e-6, ZOMBIE_TRAVEL_SEC - SPAWN_TELEGRAPH_SEC)
        return max(0.0, min(1.0, t / span))

    def shootable(self, now):
        """Only while actually approaching. Not during the spawn blink,
        and not once it has reached the gun -- from there it goes inside
        the square and detonates, and no amount of staring saves the
        life."""
        return (self.exploded_at is None
                and not self.is_spawning(now)
                and self.travel_frac(now) < 1.0)

    def spawn_scale(self, now):
        """0.15 -> 1.0 sprite scale across the telegraph, then 1.0. Smooth
        and monotonic on purpose -- no flicker, so it adds nothing
        periodic to the visual field (see SPAWN_TELEGRAPH_SEC)."""
        if not self.is_spawning(now):
            return 1.0
        return 0.15 + 0.85 * (self.age(now) / SPAWN_TELEGRAPH_SEC)


class ZombieGame:
    # margin_px MUST match run_ssvep_display.py's SSVEPDisplay default (70).
    # It isn't just padding: compute_layout() folds it into the shrink-to-fit
    # solve, so a different margin can produce a different scale factor on
    # screens where the ideal layout doesn't fit -- which would move the
    # target squares and break transfer from the calibration display the
    # subject actually trained on. This used to be 90 here vs 70 there, which
    # silently diverged whenever avail*0.08 landed between the two.
    def __init__(self, frequencies, layout="cross", eccentricity_deg=7.0,
                 stimulus_size_deg=3.5, viewing_distance_cm=60.0, margin_px=70,
                 max_fps_fallback=120, difficulty="easy",
                 enable_keyboard_debug_gaze=True, fullscreen=True,
                 render_scale=1.0):
        if layout not in ("cross", "grid"):
            raise ValueError('layout must be "cross" or "grid"')
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTIES)}")
        self.frequencies = list(frequencies)
        self.layout = layout
        self.eccentricity_deg = eccentricity_deg
        self.stimulus_size_deg = stimulus_size_deg
        self.viewing_distance_cm = viewing_distance_cm
        self.margin_px = margin_px
        self.max_fps_fallback = max_fps_fallback
        self.enable_keyboard_debug_gaze = enable_keyboard_debug_gaze
        self.fullscreen = fullscreen
        # Render at a FRACTION of the panel resolution and let the GPU
        # scale the result up (pygame.SCALED). Physical geometry is
        # unaffected -- the layout is computed from the logical size
        # against the same physical millimetres, so a 3.5deg target is
        # still 3.5deg on the retina; there are simply fewer pixels in it.
        #
        # This exists because the game is drawn through the software
        # surface path (pygame.display.set_mode), where BOTH the drawing
        # and flip()'s texture upload cost scale with pixel count. Measured
        # on the UNO Q at 1680x1050 fullscreen: 11.5ms render against an
        # 8.33ms budget, so vsync halved and the game presented at 62.9Hz
        # instead of 120Hz. That is not just a smoothness problem -- at
        # 63Hz the 17Hz target gets 1.85 frames per half-cycle (broken),
        # and 15/17Hz third harmonics alias onto each other's targets,
        # which is why calibration came out at chance. Quartering the
        # pixels brings the render back inside budget and restores 120Hz.
        self.render_scale = float(render_scale)
        # Disarmed = squares flicker, but NOTHING spawns. run_uno_q holds
        # the game here while threshold calibration runs, so zombies don't
        # crawl across the screen while the subject is trying to fixate a
        # prompted target. Standalone play arms immediately.
        self._armed = True
        self._banner = None
        # Optional instrumentation hook. Set to a callable and run_mainloop()
        # invokes it about once a second with a dict of real frame timings
        # (see the call site). Left None for normal play, where it costs
        # nothing but one `is not None` test per frame -- the debug runners
        # in test/ are the only things that set it.
        self.stats_cb = None

        self._lock_state = threading.Lock()
        self._gaze = None
        self._difficulty = difficulty
        self._stop_event = threading.Event()

        self.on_life_lost = None   # optional callable(lives_remaining)
        self.on_game_over = None   # optional callable(score)
        self.on_kill = None        # optional callable(score)

    # ---------------- thread-safe external API ----------------
    def set_highlight(self, freq):
        """Thread-safe -- call from the classifier's on_prediction(freq) /
        on_idle() (pass None) callbacks. Same name/contract as
        run_ssvep_display.py's SSVEPDisplay.set_highlight(), so this
        class is a drop-in substitute for it wherever a `display`
        argument with that interface is expected (e.g.
        run_ssvep_detection.py's run_live_detection())."""
        with self._lock_state:
            self._gaze = freq

    def set_armed(self, armed):
        """Thread-safe. False = freeze spawning and clear the field (used
        during calibration); True = normal play."""
        with self._lock_state:
            self._armed = bool(armed)

    def set_message(self, text):
        """Thread-safe. One short STATIC line along the top, e.g.
        "Calibrating". Kept static on purpose -- text that rewrites itself
        pulls the eye off the target being fixated."""
        with self._lock_state:
            self._banner = None if text is None else str(text)

    def set_difficulty(self, difficulty):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTIES)}")
        with self._lock_state:
            self._difficulty = difficulty

    def stop(self):
        """Thread-safe -- call from any thread to end run_mainloop()."""
        self._stop_event.set()

    # ---------------- game logic (main thread only) ----------------
    def _maybe_spawn(self, active, now):
        cfg = DIFFICULTIES[self._difficulty]
        cap = cfg["cap"]
        thresholds = cfg["thresholds"]
        while len(active) < cap:
            if active:
                oldest = min(active, key=lambda z: z.spawn_time)
                need_idx = len(active) - 1
                if need_idx >= len(thresholds) or oldest.progress(now) < thresholds[need_idx]:
                    break
            taken = {z.square_idx for z in active}
            free = [i for i in range(len(self.frequencies)) if i not in taken]
            if not free:
                break
            idx = random.choice(free)
            active.append(Zombie(self.frequencies[idx], idx, now))

    def run_mainloop(self):
        """Builds the display and runs the game loop. BLOCKS until stop()
        is called or the window/display is closed. Must be called from
        your program's main thread -- SDL's video calls are only safe
        there, same constraint Tk has (see run_ssvep_display.py)."""
        import pygame
        # Hardware-accelerated renderer, same API run_ssvep_display_sdl.py
        # uses. Imported here rather than at module scope so this file
        # still imports on a machine without pygame installed.
        from pygame._sdl2.video import Renderer, Texture, Window

        # Let SDL select X11 under the UNO Q XFCE desktop.  kmsdrm remains
        # available only as an explicit override for a bare-TTY session.
        driver = os.environ.get("SSVEP_VIDEO_DRIVER")
        if driver:
            os.environ["SDL_VIDEODRIVER"] = driver

        pygame.init()
        # Fullscreen is independent of the SDL backend: under XFCE this now
        # uses X11 fullscreen instead of requiring the broken kmsdrm path.
        # Keep the dummy backend windowed for automated/headless tests.
        fullscreen = self.fullscreen and driver != "dummy"

        from game_lib.screen_utils import _physical_size_mm, _forced_resolution, _FALLBACK_MM_W, _FALLBACK_MM_H
        forced = _forced_resolution()
        if fullscreen and forced:
            work_w_px, work_h_px = forced
            print(f"[ZombieGame] SSVEP_FORCE_RESOLUTION set -- requesting "
                  f"{forced[0]}x{forced[1]} explicitly instead of the current mode.")
        else:
            info = pygame.display.Info()
            work_w_px, work_h_px = info.current_w, info.current_h
        if not fullscreen:
            work_h_px = max(1, work_h_px - 80)

        # Remember the true panel size BEFORE any downscale, then compute
        # all geometry in LOGICAL pixels. px_per_mm therefore shrinks with
        # render_scale, which is exactly right: fewer logical pixels per
        # millimetre means the same PHYSICAL size once SDL scales the
        # surface back up to the panel. Getting this order wrong would
        # draw targets render_scale-times too large on screen.
        panel_w_px, panel_h_px = work_w_px, work_h_px
        scaled_render = fullscreen and 0.1 < self.render_scale < 0.999
        if scaled_render:
            work_w_px = max(320, int(work_w_px * self.render_scale))
            work_h_px = max(240, int(work_h_px * self.render_scale))

        mm = _physical_size_mm()
        if mm is None:
            print(f"[ZombieGame] WARNING: could not detect physical screen size -- "
                  f"falling back to an assumed 24in 16:9 panel. Target positions may "
                  f"not exactly match a real calibration run on this machine.")
            mm = (_FALLBACK_MM_W, _FALLBACK_MM_H)
        # Per-axis density, not the average -- on a panel driven at a
        # non-native aspect these differ enough (48% on the UNO Q at
        # 1680x1050) that averaging makes the targets physically
        # rectangular. See compute_layout()'s docstring.
        px_per_mm = (work_w_px / mm[0], work_h_px / mm[1])

        n = len(self.frequencies)
        geo = compute_layout(n, self.eccentricity_deg, self.stimulus_size_deg,
                              self.viewing_distance_cm, px_per_mm,
                              work_w_px, work_h_px, self.layout, self.margin_px)
        print_layout_report(geo, n, self.viewing_distance_cm, px_per_mm, tag="[ZombieGame]")

        if not fullscreen:
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        # ---- hardware-accelerated renderer ----
        # The game used to draw through pygame.display.set_mode(), the
        # SOFTWARE surface path: every rect/sprite is composited on the CPU
        # and flip() then uploads the whole surface to the GPU each frame.
        # Both costs scale with pixel count, and at 1680x1050 that measured
        # 11.5ms/frame on the UNO Q -- over the 8.33ms budget, so vsync
        # halved and the game ran at 62.9Hz. That is not merely less smooth:
        # at 63Hz the 17Hz target gets 1.85 frames per half-cycle (broken)
        # and the 15/17Hz third harmonics alias onto each other's targets,
        # which is why calibration came out at chance.
        #
        # run_ssvep_display_sdl.py holds a steady 120Hz on that same board
        # because it uses this accelerated Renderer instead, so the game now
        # uses it too. Everything below draws via GPU primitives or
        # pre-uploaded Textures; nothing is composited on the CPU per frame.
        win_w, win_h = (panel_w_px, panel_h_px) if fullscreen else (geo["width"], geo["height"])
        window = Window("SSVEP Zombie Defense", size=(win_w, win_h))
        if fullscreen:
            window.borderless = True
            window.set_fullscreen(desktop=True)
        try:
            renderer = Renderer(window, accelerated=True, vsync=True)
        except Exception as exc:
            # Fall back rather than abort -- headless/dummy backends and
            # machines without acceleration must still be able to run the
            # game. But say so loudly: without GPU acceleration the frame
            # budget is what forced 62.9Hz on the UNO Q, and at 63Hz the
            # 17Hz target is unusable. Never let this pass silently.
            print(f"[ZombieGame] WARNING: no accelerated VSync renderer ({exc}). "
                  f"Falling back to an unaccelerated one -- frame timing will be "
                  f"worse and the stimulus may not hold the panel rate. Check the "
                  f"'present rate' line below before trusting any session.",
                  file=sys.stderr)
            try:
                renderer = Renderer(window, vsync=True)
            except Exception:
                renderer = Renderer(window)

        # Render at a smaller LOGICAL size when asked; SDL scales to the
        # panel on the GPU for free. Geometry was already computed in these
        # logical pixels, so physical target size is unchanged.
        if scaled_render:
            renderer.logical_size = (work_w_px, work_h_px)
            print(f"[ZombieGame] render_scale={self.render_scale:g}: logical "
                  f"{work_w_px}x{work_h_px} -> panel {panel_w_px}x{panel_h_px}")

        print(f"[ZombieGame] video driver: {pygame.display.get_driver()}  |  "
              f"{win_w}x{win_h}"
              f"{'  (fullscreen, accelerated)' if fullscreen else '  (windowed, accelerated)'}")
        if self.enable_keyboard_debug_gaze:
            print("[ZombieGame] keyboard debug-gaze is ON (hold 1-4 to simulate looking "
                  "at a target). Pass enable_keyboard_debug_gaze=False for real BCI runs.")

        w, h = (work_w_px, work_h_px) if scaled_render else (win_w, win_h)
        # Dead centre -- NOT nudged up for the bottom UI, because the UI now
        # sits in the empty bottom margin band instead of in space taken from
        # the playfield. Any offset here would shift every target away from
        # where run_ssvep_display.py put it during calibration.
        cx, cy = w / 2.0, h / 2.0
        # font_sm is now only the difficulty buttons; the "N Hz" captions
        # moved up to font_md so they stay readable from the 60cm viewing
        # distance the layout assumes. Caption size feeds back into
        # geometry -- outward_label_pos() pushes each caption clear of its
        # square by the caption's own height, so a bigger font sits
        # further out. It still lands inside the layout's margin band, but
        # that is the thing to re-check if these grow much beyond this.
        font_sm = pygame.font.SysFont(None, 24)
        font_md = pygame.font.SysFont(None, 30)
        font_lg = pygame.font.SysFont(None, 64)

        # ---- sprites pre-uploaded to the GPU, once ----
        # Every sprite is still drawn by the same _draw_* helpers as before,
        # just onto a small Surface at startup instead of onto the screen
        # every frame. The result becomes a Texture and costs one GPU blit
        # per frame thereafter. This is where most of the CPU saving comes
        # from: the zombie alone was 44 pygame.draw.rect calls PER ZOMBIE
        # PER FRAME, and five hearts another 135.
        def _tex(size_wh, drawfn):
            surf = pygame.Surface(size_wh, pygame.SRCALPHA)
            drawfn(surf)
            return Texture.from_surface(renderer, surf)

        ZS = 28
        tex_zombie = _tex((ZS, ZS), lambda sf: _draw_zombie(pygame, sf, ZS / 2, ZS / 2, size=ZS))
        tex_heart = _tex((24, 20), lambda sf: _draw_heart(pygame, sf, 12, 10))

        # One shared beam strip for all four guns: it is stretched to the
        # shot's length and rotated to the gun's direction at draw time, so
        # orientation and distance cost nothing extra. Alpha blending is
        # required -- the ramp fades out at the beam's edges so it reads as
        # flame rather than as a hard-edged bar.
        tex_beam = _tex((BEAM_STRIP_W, BEAM_TH),
                         lambda sf: _draw_beam_strip(pygame, sf, BEAM_TH, BEAM_STRIP_W))
        tex_beam.blend_mode = pygame.BLENDMODE_BLEND

        # Explosion is animated, so bake a handful of frames rather than
        # drawing circles on the CPU each time.
        EXPL_N, EXPL_PX = 8, 96
        tex_expl = [_tex((EXPL_PX, EXPL_PX),
                          lambda sf, k=k: _draw_explosion(pygame, sf, EXPL_PX / 2,
                                                           EXPL_PX / 2,
                                                           (k + 0.5) / EXPL_N, 1.0))
                    for k in range(EXPL_N)]

        # Scanlines as one full-screen texture: a single GPU blit replaces
        # the 262 CPU line draws that fullscreen used to cost.
        def _scan(sf):
            for yy in range(0, h, 4):
                pygame.draw.line(sf, (0, 0, 0, 90), (0, yy), (w, yy), 1)
        tex_scan = _tex((w, h), _scan)
        tex_scan.blend_mode = pygame.BLENDMODE_BLEND

        # Cached text -> Texture. Strings here change at most a few times a
        # second (score, banner), so re-uploading every frame would be waste.
        _text_cache = {}

        def _text(msg, font, colour):
            key = (msg, id(font), colour)
            hit = _text_cache.get(key)
            if hit is None:
                surf = font.render(msg, True, colour)
                hit = (Texture.from_surface(renderer, surf),
                       surf.get_width(), surf.get_height())
                if len(_text_cache) > 64:
                    _text_cache.clear()
                _text_cache[key] = hit
            return hit

        # 1x1 black texture stretched for the game-over dim -- the Renderer
        # has no alpha-fill primitive.
        _dim = pygame.Surface((1, 1), pygame.SRCALPHA)
        _dim.fill((0, 0, 0, 180))
        tex_dim = Texture.from_surface(renderer, _dim)
        tex_dim.blend_mode = pygame.BLENDMODE_BLEND

        squares = []
        for i, ((dx, dy), freq) in enumerate(zip(geo["offsets"], self.frequencies)):
            rect = pygame.Rect(0, 0, int(round(geo["size_px_w"])),
                                int(round(geo["size_px_h"])))
            rect.center = (cx + dx, cy + dy)
            # Aim vector points from this square back at the centre, where
            # zombies spawn -- used to place the gun on the square's INNER
            # edge and to orient its barrel. Works for both "cross" and
            # "grid" layouts because it's derived from the offset, not
            # hardcoded per compass direction.
            dist = math.hypot(dx, dy) or 1.0
            ux, uy = -dx / dist, -dy / dist
            reach = max(rect.width, rect.height) / 2.0 + 6
            label = font_md.render(f"{freq:g} Hz", True, (230, 230, 230))
            label_center = outward_label_pos(
                cx, cy, dx, dy, rect.width, rect.height,
                label.get_width(), label.get_height(),
            )
            # Per-target textures: the Hz caption, and the gun/muzzle at
            # this target's orientation. Baked once; the gun and flame are
            # polygons, which the Renderer API has no primitive for anyway.
            lab_tex = Texture.from_surface(renderer, label)
            lab_rect = pygame.Rect(0, 0, label.get_width(), label.get_height())
            lab_rect.center = (int(label_center[0]), int(label_center[1]))

            GP = 44                      # canvas big enough for barrel + flame
            gun_xy = (cx + dx + ux * reach, cy + dy + uy * reach)
            gun_tex = _tex((GP, GP),
                            lambda sf, u=(ux, uy): _draw_gun(pygame, sf, (GP / 2, GP / 2), u))
            flash_tex = _tex((GP, GP),
                              lambda sf, u=(ux, uy): _draw_muzzle_flash(
                                  pygame, sf, (GP / 2, GP / 2), u, 0))
            gun_rect = pygame.Rect(0, 0, GP, GP)
            gun_rect.center = (int(gun_xy[0]), int(gun_xy[1]))

            # Where the zombie's WALK ends -- the muzzle, not the square's
            # centre. travel_frac() == 1.0 puts it here, which is what
            # stops a zombie ever getting behind its own gun. It used to
            # walk all the way to the centre of the square, straight past
            # and through the turret; for that last stretch the beam had
            # to be aimed BACKWARDS to reach it, and the zombie was still
            # taking damage from a gun it had already walked behind.
            #
            # Ending the path here also restores the intended sequence:
            # reach the box -> no longer shootable -> go in -> explode.
            # The detonation is still drawn at the square's centre, and
            # the burst is wider than this last gap, so it engulfs the
            # spot the zombie vanished from rather than jumping away from
            # it. Backed off by half a zombie plus the beam's own origin
            # offset so the sprite's leading edge just meets the muzzle
            # and the beam still has positive length on the final frame.
            approach = min(reach + MUZZLE_PX * 0.6 + ZS * 0.5, dist * 0.85)
            path_end = (cx + dx + ux * approach, cy + dy + uy * approach)

            squares.append({"idx": i, "freq": freq, "rect": rect,
                             "center": (cx + dx, cy + dy),
                             "path_end": path_end,
                             "label_tex": lab_tex, "label_rect": lab_rect,
                             "gun": gun_xy, "aim": (ux, uy),
                             "gun_tex": gun_tex, "flash_tex": flash_tex,
                             "gun_rect": gun_rect,
                             })

        key_to_index = {getattr(pygame, f"K_{k}"): v for k, v in DEBUG_KEY_TO_INDEX.items()
                         if v < len(squares)}

        # ---- bottom-left difficulty buttons ----
        # These sit INSIDE the layout's bottom margin band, which is empty
        # by construction: the targets lie on a circle inscribed in this
        # window, so the lowest square's bottom edge is exactly margin_px
        # above the window bottom, full width. Compact enough (26px tall) to
        # clear that band without touching the playfield, so the window
        # needs no extra height for them. Width still adapts to the
        # left-of-centre space so they stay inside the band on a small
        # screen, where compute_layout() shrinks it to avail*0.08.
        btn_h, btn_gap, btn_x0 = 26, 6, 12
        avail_w = max(90.0, cx - btn_x0 - 12)
        btn_w = int(min(76, (avail_w - btn_gap * (len(DIFFICULTY_ORDER) - 1)) / len(DIFFICULTY_ORDER)))
        buttons = []
        for i, name in enumerate(DIFFICULTY_ORDER):
            rect = pygame.Rect(btn_x0 + i * (btn_w + btn_gap), h - btn_h - 10, btn_w, btn_h)
            buttons.append({"name": name, "rect": rect})

        # ---- frame-locked stimulus timing ----
        # The flicker phase is driven by a FRAME COUNTER divided by the
        # panel's measured refresh rate, not by wall-clock time. Same fix
        # as run_ssvep_display_sdl.py: with vsync active, pygame's flip()
        # blocks until VBlank, so the frame counter IS the display clock
        # and the emitted waveform is identical every cycle. Driving phase
        # from perf_counter() instead re-introduces exactly the jitter
        # vsync just removed, because which refresh an edge lands on then
        # depends on when the Python loop happened to run.
        #
        # measured_refresh is learned below rather than assumed: the UNO Q
        # reports 119.99Hz via xrandr but actually presents at ~120.38Hz,
        # and phase = frames/rate turns that 3234ppm error into a real
        # frequency offset (17Hz emitted as 17.055Hz, ~2% CCA correlation
        # lost over a 2s window).
        clock = pygame.time.Clock()
        start_time = time.perf_counter()
        last_frame_time = start_time
        frame_index = 0
        measured_refresh = float(self.max_fps_fallback)
        refresh_locked = False
        warmup_intervals = []
        vsync_active = False
        # Tracked separately from last_frame_time (which the top of the loop
        # reassigns for the gameplay dt): this one must measure present ->
        # present, i.e. the true frame period, not per-frame render duration.
        last_present = None
        _dbg_t0, _dbg_frames, _dbg_render_ms, _dbg_iv = time.perf_counter(), 0, 0.0, []
        active_zombies = []
        lives = STARTING_LIVES
        score = 0
        explosions = []    # (x, y, start_time, scale) -- kill bursts
        firing_at = {}     # (re-set each frame in the loop below)
        debug_key_gaze = None
        game_over = False

        running = True
        while running and not self._stop_event.is_set():
            now = time.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_e:
                        self.set_difficulty("easy")
                    elif event.key == pygame.K_m:
                        self.set_difficulty("medium")
                    elif event.key == pygame.K_h:
                        self.set_difficulty("hard")
                    elif self.enable_keyboard_debug_gaze and event.key in key_to_index:
                        debug_key_gaze = key_to_index[event.key]
                elif event.type == pygame.KEYUP:
                    if self.enable_keyboard_debug_gaze and event.key in key_to_index:
                        if debug_key_gaze == key_to_index[event.key]:
                            debug_key_gaze = None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for b in buttons:
                        if b["rect"].collidepoint(event.pos):
                            self.set_difficulty(b["name"])
                    if game_over:
                        # click anywhere to restart
                        active_zombies = []
                        lives = STARTING_LIVES
                        score = 0
                        explosions = []
                        game_over = False
                        start_time = time.perf_counter()

            with self._lock_state:
                gaze = self._gaze
                difficulty = self._difficulty
                armed = self._armed
                banner = self._banner
            if self.enable_keyboard_debug_gaze and debug_key_gaze is not None:
                gaze = self.frequencies[debug_key_gaze]

            # Reset every frame, OUTSIDE the game_over guard: if it were only
            # cleared inside, a game-over would freeze with whatever guns were
            # firing on the final frame still flashing.
            # square index -> the (x, y) its beam has to reach this frame.
            # Carries the hit point, not just a flag, because the beam is
            # drawn from the gun to wherever the zombie has walked to.
            firing_at = {}

            if not armed:
                # Calibration in progress: keep the squares flickering (the
                # subject is fixating them) but run no simulation at all,
                # and drop anything already on the field so play starts
                # clean the moment it arms.
                active_zombies = []
                explosions = []
            elif not game_over:
                self._maybe_spawn(active_zombies, now)

                survivors = []
                for z in active_zombies:
                    # ---- already detonating inside the square ----
                    # The life was taken on arrival, so this is animation
                    # only; it just has to finish before the zombie is gone.
                    if z.exploded_at is not None:
                        if now - z.exploded_at < EXPLODE_SEC:
                            survivors.append(z)
                        continue

                    # ---- arrived: stops being shootable, goes in and blows up ----
                    if z.travel_frac(now) >= 1.0:
                        z.exploded_at = now
                        lives -= 1
                        if self.on_life_lost:
                            self.on_life_lost(lives)
                        if lives <= 0:
                            game_over = True
                            if self.on_game_over:
                                self.on_game_over(score)
                        survivors.append(z)
                        continue

                    # ---- travelling: the only phase that takes damage ----
                    # shootable() excludes the spawn blink as well, so a
                    # zombie can't be killed before it has even set off.
                    # No else-branch: health never regenerates, so
                    # intermittent detection still makes progress (see
                    # KILL_GAZE_SEC).
                    if z.shootable(now) and gaze == z.target_freq:
                        z.health = max(0.0, z.health - dt / KILL_GAZE_SEC)
                        tx, ty = squares[z.square_idx]["path_end"]
                        zf = z.travel_frac(now)
                        firing_at[z.square_idx] = (cx + (tx - cx) * zf,
                                                    cy + (ty - cy) * zf)
                    if z.health <= 0.0:
                        score += 1
                        tx, ty = squares[z.square_idx]["path_end"]
                        f = z.travel_frac(now)
                        explosions.append((cx + (tx - cx) * f, cy + (ty - cy) * f, now, 0.7))
                        if self.on_kill:
                            self.on_kill(score)
                        continue  # killed -- removed
                    survivors.append(z)
                active_zombies = survivors

            explosions = [e for e in explosions if now - e[2] < EXPLODE_SEC]

            # ---------------- render (GPU) ----------------
            # Every call below is either a Renderer primitive or a blit of a
            # texture uploaded once at startup. Nothing is composited on the
            # CPU, and there is no per-frame surface upload -- that is the
            # difference between 63Hz and 120Hz on the UNO Q.
            renderer.draw_color = (8, 8, 12, 255)
            renderer.clear()
            tex_scan.draw(dstrect=pygame.Rect(0, 0, w, h))

            elapsed = frame_index / measured_refresh
            for sq in squares:
                phase = (elapsed * sq["freq"]) % 1.0
                renderer.draw_color = ((235, 235, 235, 255) if phase < 0.5
                                        else (20, 20, 20, 255))
                renderer.fill_rect(sq["rect"])
                renderer.draw_color = ((0, 255, 0, 255) if sq["freq"] == gaze
                                        else (60, 60, 60, 255))
                for inset in range(6):                     # 6px outline
                    r = sq["rect"]
                    renderer.draw_rect(pygame.Rect(
                        r.x + inset, r.y + inset,
                        max(1, r.width - 2 * inset), max(1, r.height - 2 * inset)))
                sq["label_tex"].draw(dstrect=sq["label_rect"])

                # Flame beam, gun, then muzzle flash -- in that order so the
                # barrel covers the beam's squared-off start and the flash
                # sits on top of both, which is what makes the beam look
                # like it leaves the muzzle instead of floating beside it.
                # Drawn before the zombie pass below, so the zombie sprite
                # is over the far end and the beam reads as hitting it.
                hit = firing_at.get(sq["idx"])
                if hit is not None:
                    gx, gy = sq["gun"]
                    ux, uy = sq["aim"]
                    sx, sy = gx + ux * MUZZLE_PX * 0.6, gy + uy * MUZZLE_PX * 0.6
                    hx, hy = hit[0] - sx, hit[1] - sy
                    length = math.hypot(hx, hy)
                    if length > 4.0:
                        # Rotate about the strip's left-middle -- i.e. the
                        # muzzle -- so the beam pivots on the gun and its
                        # far end lands on the zombie. atan2 in screen
                        # coords (y down) already matches SDL's clockwise
                        # rotation, so no sign flip is needed.
                        tex_beam.draw(
                            dstrect=pygame.Rect(int(sx), int(sy - BEAM_TH / 2),
                                                 int(length), BEAM_TH),
                            angle=math.degrees(math.atan2(hy, hx)),
                            origin=(0, BEAM_TH / 2))

                sq["gun_tex"].draw(dstrect=sq["gun_rect"])
                if hit is not None:
                    sq["flash_tex"].draw(dstrect=sq["gun_rect"])

            for z in active_zombies:
                # Arrived: it has gone INSIDE the square and is detonating,
                # so the burst is drawn at the square's centre -- past the
                # muzzle where the walk ended. No zombie sprite and no
                # health bar; it cannot be shot any more.
                if z.exploded_at is not None:
                    ex_x, ex_y = squares[z.square_idx]["center"]
                    k = min(EXPL_N - 1,
                            int((now - z.exploded_at) / EXPLODE_SEC * EXPL_N))
                    tex_expl[k].draw(dstrect=pygame.Rect(
                        int(ex_x - EXPL_PX / 2), int(ex_y - EXPL_PX / 2),
                        EXPL_PX, EXPL_PX))
                    continue

                # Still walking: the path runs centre -> muzzle, so the
                # zombie is always in FRONT of the gun that is shooting it.
                tx, ty = squares[z.square_idx]["path_end"]
                f = z.travel_frac(now)
                zx, zy = cx + (tx - cx) * f, cy + (ty - cy) * f

                # Spawn telegraph: grows at the centre, then sets off. The
                # scale is applied to the destination rect, so the GPU does
                # the resampling instead of re-rasterising the sprite.
                zs = ZS * (z.spawn_scale(now) if z.is_spawning(now) else 1.0)
                tex_zombie.draw(dstrect=pygame.Rect(
                    int(zx - zs / 2), int(zy - zs / 2), int(zs), int(zs)))
                if z.is_spawning(now):
                    continue

                # Health bar: drawn full at spawn and shrinking as damaged,
                # so remaining health reads at a glance.
                hp = max(0.0, min(1.0, z.health))
                bx, by, bw, bh = int(zx - 15), int(zy - 26), 30, 5
                renderer.draw_color = (40, 40, 40, 255)
                renderer.fill_rect(pygame.Rect(bx, by, bw, bh))
                renderer.draw_color = (int(220 * (1.0 - hp)) + 35,
                                        int(200 * hp) + 40, 50, 255)
                renderer.fill_rect(pygame.Rect(bx, by, int(bw * hp), bh))
                renderer.draw_color = (200, 200, 200, 255)
                renderer.draw_rect(pygame.Rect(bx, by, bw, bh))

            for ex, ey, t0, sc in explosions:
                k = min(EXPL_N - 1, int((now - t0) / EXPLODE_SEC * EXPL_N))
                side = int(EXPL_PX * sc)
                tex_expl[k].draw(dstrect=pygame.Rect(
                    int(ex - side / 2), int(ey - side / 2), side, side))

            if banner:
                bt, bw_, bh_ = _text(banner, font_md, (175, 175, 185))
                bt.draw(dstrect=pygame.Rect(int(cx - bw_ / 2), 14, bw_, bh_))

            # ---- HUD ----
            for i in range(max(0, lives)):
                tex_heart.draw(dstrect=pygame.Rect(24 + i * 22 - 12, 22 - 10, 24, 20))
            st, sw_, sh_ = _text(f"Score: {score}", font_md, (230, 230, 230))
            st.draw(dstrect=pygame.Rect(w - sw_ - 16, 12, sw_, sh_))

            for b in buttons:
                active = b["name"] == difficulty
                renderer.draw_color = ((70, 130, 70, 255) if active
                                        else (50, 50, 55, 255))
                renderer.fill_rect(b["rect"])
                renderer.draw_color = (255, 255, 255, 255)
                renderer.draw_rect(b["rect"])
                bt2, bw2, bh2 = _text(b["name"].upper(), font_sm, (255, 255, 255))
                bt2.draw(dstrect=pygame.Rect(
                    int(b["rect"].centerx - bw2 / 2),
                    int(b["rect"].centery - bh2 / 2), bw2, bh2))

            if game_over:
                # Alpha dim over the whole screen. Renderer has no alpha
                # fill, so blend a 1x1 texture stretched to fit.
                tex_dim.draw(dstrect=pygame.Rect(0, 0, w, h))
                gt, gw_, gh_ = _text("GAME OVER", font_lg, (255, 60, 60))
                gt.draw(dstrect=pygame.Rect(int(cx - gw_ / 2), int(cy - 20 - gh_ / 2),
                                             gw_, gh_))
                ct, cw_, ch_ = _text(f"Score: {score}  -  click to restart",
                                      font_md, (230, 230, 230))
                ct.draw(dstrect=pygame.Rect(int(cx - cw_ / 2), int(cy + 30 - ch_ / 2),
                                             cw_, ch_))

            render_ms = (time.perf_counter() - now) * 1000.0   # work before present
            renderer.present()      # blocks until VBlank when vsync is active
            present_time = time.perf_counter()

            if self.stats_cb is not None:
                _dbg_frames += 1
                _dbg_render_ms += render_ms
                if last_present is not None:
                    iv = (present_time - last_present) * 1000.0
                    if 0.1 <= iv <= 250.0:
                        _dbg_iv.append(iv)
                if present_time - _dbg_t0 >= 1.0:
                    span = present_time - _dbg_t0
                    ivs = sorted(_dbg_iv)
                    med = ivs[len(ivs) // 2] if ivs else 0.0
                    # A "drop" is an interval long enough to have spanned
                    # more than one panel refresh.
                    fp = 1000.0 / measured_refresh if measured_refresh else 0.0
                    drops = sum(max(0, int(round(x / fp)) - 1) for x in ivs) if fp else 0
                    try:
                        self.stats_cb({
                            "fps": _dbg_frames / span,
                            "median_ms": med,
                            "max_ms": max(ivs) if ivs else 0.0,
                            "render_ms": _dbg_render_ms / max(1, _dbg_frames),
                            "drops": drops,
                            "refresh": measured_refresh,
                            "vsync": vsync_active,
                            "zombies": len(active_zombies),
                            "armed": armed,
                            "lives": lives,
                            "score": score,
                        })
                    except Exception:
                        pass
                    _dbg_t0, _dbg_frames, _dbg_render_ms = present_time, 0, 0.0
                    _dbg_iv = []

            # ---- learn the real refresh rate, then advance the frame clock ----
            interval = None if last_present is None else present_time - last_present
            last_present = present_time
            if interval is None:
                frame_index += 1
            elif not refresh_locked:
                # Discard the first frames -- early presents aren't
                # representative (measured on the UNO Q: the first 40 samples
                # read 115.9Hz while steady state was 120.38Hz).
                if len(warmup_intervals) < 240:
                    if 0.001 <= interval <= 0.250:
                        warmup_intervals.append(interval)
                    frame_index += 1
                else:
                    timed = sorted(warmup_intervals[60:])
                    median_dt = timed[len(timed) // 2]
                    candidate = 1.0 / median_dt
                    # Only trust it if it looks like a real panel rate. If
                    # vsync silently isn't active, flip() returns immediately
                    # and this would read as hundreds of Hz -- in that case
                    # keep max_fps_fallback and let clock.tick() pace us.
                    vsync_active = 20.0 <= candidate <= 400.0 and median_dt > 0.002
                    if vsync_active:
                        measured_refresh = candidate
                    print(f"[ZombieGame] present rate {candidate:.2f} Hz "
                          f"({median_dt * 1000:.3f} ms/frame) -- "
                          + ("vsync active, driving stimulus phase off the frame clock."
                             if vsync_active else
                             f"does NOT look vsync-paced; falling back to "
                             f"clock.tick({self.max_fps_fallback}) and treating "
                             f"{measured_refresh:.0f}Hz as the stimulus clock."))
                    refresh_locked = True
                    frame_index += 1
            else:
                # Advance by the number of refreshes that ACTUALLY elapsed, so
                # a dropped frame is a one-off glitch instead of permanent
                # phase drift.
                if vsync_active and 0.001 <= interval <= 0.250:
                    frame_index += max(1, int(round(interval * measured_refresh)))
                else:
                    frame_index += 1

            # Do NOT pace during the warm-up: clock.tick() would throttle the
            # loop to max_fps_fallback, and the measurement above would then
            # read back its own pacing and conclude vsync was working even
            # when flip() is returning instantly. Unpaced, a real vsync shows
            # up as the panel rate and a missing one as implausibly high fps,
            # which is exactly what the check needs to distinguish. After the
            # rate is locked, pace only when vsync ISN'T doing it for us --
            # calling tick() on top of working vsync just adds sleep jitter.
            if refresh_locked and not vsync_active:
                clock.tick(self.max_fps_fallback)

        pygame.quit()


def _draw_zombie(pygame, screen, cx, cy, size=28):
    cell = size / len(ZOMBIE_SPRITE)
    x0 = cx - size / 2
    y0 = cy - size / 2
    for r, row in enumerate(ZOMBIE_SPRITE):
        for c, ch in enumerate(row):
            if ch == " ":
                continue
            col = _ZOMBIE_COLORS[ch]
            pygame.draw.rect(screen, col, (x0 + c * cell, y0 + r * cell, cell + 1, cell + 1))


# 7x6 pixel-art heart for the lives readout -- same no-external-assets
# approach as the zombie sprite, and far more readable at this size than
# a glyph from a font.
HEART_SPRITE = [
    " XX XX ",
    "XXXXXXX",
    "XXXXXXX",
    " XXXXX ",
    "  XXX  ",
    "   X   ",
]


def _draw_heart(pygame, screen, cx, cy, cell=3):
    w = len(HEART_SPRITE[0]) * cell
    h = len(HEART_SPRITE) * cell
    x0, y0 = cx - w / 2, cy - h / 2
    for r, row in enumerate(HEART_SPRITE):
        for c, ch in enumerate(row):
            if ch == "X":
                pygame.draw.rect(screen, (220, 60, 70),
                                  (x0 + c * cell, y0 + r * cell, cell, cell))


def _draw_gun(pygame, screen, gun_xy, aim, size=9):
    """Small turret on a target's inner edge, barrel pointing at the centre.
    Static grey by design -- it sits right beside a flickering square, so
    anything animated or bright here would add luminance change the eye
    picks up along with the stimulus."""
    gx, gy = gun_xy
    ux, uy = aim
    px, py = -uy, ux                       # perpendicular, for the body
    body = [(gx + px * size * 0.5, gy + py * size * 0.5),
            (gx - px * size * 0.5, gy - py * size * 0.5),
            (gx - px * size * 0.5 + ux * size * 0.5, gy - py * size * 0.5 + uy * size * 0.5),
            (gx + px * size * 0.5 + ux * size * 0.5, gy + py * size * 0.5 + uy * size * 0.5)]
    pygame.draw.polygon(screen, (120, 125, 135), body)
    pygame.draw.line(screen, (150, 155, 165), (gx, gy),
                      (gx + ux * size * 1.1, gy + uy * size * 1.1), 3)


def _draw_beam_strip(pygame, screen, thickness, length_px):
    """One baked slice of the flame beam: a horizontal strip whose colour
    ramps red -> orange -> yellow-white from its edges to its core, and
    which is IDENTICAL all the way along its length.

    Being uniform along the length is what makes this cheap. The strip is
    baked once at startup and then stretched to whatever distance the shot
    covers and rotated to whatever direction the gun points, so a beam of
    any length and angle costs exactly one GPU blit -- no per-frame
    polygon building, and nothing added to the frame budget that the
    120Hz stimulus needs.

    It is also deliberately static. A scrolling or pulsing beam would put
    a second periodic luminance change on screen right where the subject
    is fixating, and FBCCA cannot tell that apart from the stimulus it is
    trying to lock onto. The squares are the only thing here allowed to
    change periodically."""
    bands = (   # (outer edge as a fraction of half-thickness, rgb, alpha)
        (0.22, (255, 247, 210), 255),      # white-hot core
        (0.44, (255, 206, 74), 245),       # yellow
        (0.68, (252, 134, 30), 225),       # orange
        (0.86, (214, 56, 20), 185),        # red
        (1.00, (140, 26, 12), 110),        # dark red falloff at the edge
    )
    half = thickness / 2.0
    for y in range(thickness):
        d = abs((y + 0.5) - half) / half
        for edge, rgb, alpha in bands:
            if d <= edge:
                pygame.draw.line(screen, (rgb[0], rgb[1], rgb[2], alpha),
                                  (0, y), (length_px, y))
                break


def _draw_muzzle_flash(pygame, screen, gun_xy, aim, seed):
    """Short flame off the barrel while this gun is firing -- the bright
    root of the beam, sitting over the point the beam is drawn from.

    Constant size and colour -- see MUZZLE_PX. `seed` is accepted so the
    call site doesn't have to change if an animated variant is ever
    wanted, but nothing here varies frame to frame."""
    gx, gy = gun_xy
    ux, uy = aim
    px, py = -uy, ux
    length = MUZZLE_PX
    half = length * 0.45
    tip = (gx + ux * length, gy + uy * length)
    base_a = (gx + px * half, gy + py * half)
    base_b = (gx - px * half, gy - py * half)
    pygame.draw.polygon(screen, (255, 170, 40), [base_a, tip, base_b])
    inner = length * 0.55
    pygame.draw.polygon(screen, (255, 235, 170), [
        (gx + px * half * 0.5, gy + py * half * 0.5),
        (gx + ux * inner, gy + uy * inner),
        (gx - px * half * 0.5, gy - py * half * 0.5)])


def _draw_explosion(pygame, screen, cx, cy, frac, scale=1.0):
    """Expanding ring + shards. frac is 0..1 through EXPLODE_SEC."""
    frac = max(0.0, min(1.0, frac))
    r = (6 + 26 * frac) * scale
    for radius, colour, width in ((r, (255, 190, 60), 3),
                                   (r * 0.65, (255, 120, 40), 4)):
        if radius >= 2:
            pygame.draw.circle(screen, colour, (int(cx), int(cy)), int(radius), width)
    if frac < 0.75:                       # shards only early in the burst
        for k in range(8):
            ang = k * (math.pi / 4)
            d = r * 1.15
            sx, sy = cx + math.cos(ang) * d, cy + math.sin(ang) * d
            pygame.draw.rect(screen, (255, 225, 150),
                              (sx - 2 * scale, sy - 2 * scale, 4 * scale, 4 * scale))


def _draw_health_bar(pygame, screen, cx, cy, health, width=30, height=5):
    """Remaining health, so the filled portion SHRINKS as the zombie takes
    damage (it starts full at spawn). Green -> red as it drops, so how close
    a zombie is to dying reads at a glance without measuring the bar."""
    health = max(0.0, min(1.0, health))
    x0 = cx - width / 2
    pygame.draw.rect(screen, (40, 40, 40), (x0, cy, width, height))
    colour = (int(220 * (1.0 - health)) + 35, int(200 * health) + 40, 50)
    pygame.draw.rect(screen, colour, (x0, cy, width * health, height))
    pygame.draw.rect(screen, (200, 200, 200), (x0, cy, width, height), width=1)


if __name__ == "__main__":
    headless = os.environ.get("SSVEP_VIDEO_DRIVER") == "dummy"
    game = ZombieGame(frequencies=[7, 13, 15, 17], layout="cross",
                       eccentricity_deg=7.0, stimulus_size_deg=3.5,
                       viewing_distance_cm=60.0,
                       render_scale=float(os.environ.get("SSVEP_RENDER_SCALE", 1.0)))
    if headless:
        t = threading.Timer(10.0, game.stop)
        t.daemon = True
        t.start()
    game.run_mainloop()
