#!/usr/bin/env python3
"""
game_lib/screen_utils.py

Two small helpers the game needs to size its window properly:

  * the screen's real physical size in millimetres, so the squares can be
    positioned by visual angle rather than by arbitrary pixel counts
    (see ssvep_geometry.py for what that is used for), and
  * an optional resolution override.

Deliberately plain: no pygame, no SDL, and no imports from the rest of
the project, so this is safe to import from anywhere.

These were originally part of a pygame display backend written while
trying to get accurate frame timing on the UNO Q through kmsdrm. That
approach was dropped -- it never reliably beat the normal desktop path on
that hardware -- but these two functions are still how the game sizes
itself, so they stayed.
"""

import ctypes
import os
import sys

_IS_LINUX = sys.platform.startswith("linux")
_IS_WINDOWS = sys.platform == "win32"

# Same 24in 16:9 fallback run_ssvep_display_sdl.py uses if physical-size
# detection comes back implausible.
_FALLBACK_MM_W, _FALLBACK_MM_H = 531.4, 298.9


def _edid_physical_size_mm():
    """Linux only: parse /sys/class/drm/*/edid for the first connector
    that has a plausible one. Base-EDID bytes 21/22 are the max
    horizontal/vertical image size in whole cm (0 = "unknown", skipped).
    Returns None if nothing usable was found -- caller falls back."""
    if not _IS_LINUX:
        return None
    base = "/sys/class/drm"
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return None
    for name in names:
        edid_path = os.path.join(base, name, "edid")
        try:
            with open(edid_path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        if len(data) < 23:
            continue
        w_cm, h_cm = data[21], data[22]
        if w_cm > 0 and h_cm > 0:
            return (w_cm * 10.0, h_cm * 10.0)
    return None


def _windows_physical_size_mm():
    if not _IS_WINDOWS:
        return None
    try:
        HORZSIZE, VERTSIZE = 4, 6
        hdc = ctypes.windll.user32.GetDC(0)
        try:
            w_mm = ctypes.windll.gdi32.GetDeviceCaps(hdc, HORZSIZE)
            h_mm = ctypes.windll.gdi32.GetDeviceCaps(hdc, VERTSIZE)
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
        if w_mm > 0 and h_mm > 0:
            return (float(w_mm), float(h_mm))
    except Exception:
        pass
    return None


def _physical_size_mm():
    return _edid_physical_size_mm() or _windows_physical_size_mm()


def _forced_resolution():
    """SSVEP_FORCE_RESOLUTION=1680x1050 -- request that exact resolution
    for the game's fullscreen window instead of the display's current/
    default mode. Optional; unset by default, harmless no-op unless you
    explicitly set the env var."""
    val = os.environ.get("SSVEP_FORCE_RESOLUTION")
    if not val:
        return None
    try:
        w, h = val.lower().split("x")
        return (int(w), int(h))
    except ValueError:
        print(f"SSVEP_FORCE_RESOLUTION='{val}' isn't 'WIDTHxHEIGHT' -- ignoring.", file=sys.stderr)
        return None
