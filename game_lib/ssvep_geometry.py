#!/usr/bin/env python3
"""
ssvep_geometry.py

Works out where the squares go on screen and how big they are, in
pixels, from sizes given in degrees of visual angle.

Every display backend uses this same file -- the Tk display, the SDL
display, and the game. That matters: the squares must land on exactly the
same pixels no matter which one is drawing them. If the game worked out
its own slightly different layout, a calibration done on the plain
display would no longer apply to it.

The maths assumes a flat screen at right angles to where you are looking.
Something that takes up angle theta at distance d is 2*d*tan(theta/2)
across, and something sitting e degrees off to the side is d*tan(e) away
from the centre. These are exact for a flat monitor, not small-angle
approximations.
"""

import math


def pairwise_separation_deg(n, eccentricity_deg):
    """Exact angular separation between adjacent targets (both at the
    same eccentricity, azimuths differing by 360/n), via the dot product
    of their eye-direction unit vectors."""
    e = math.radians(eccentricity_deg)
    d = math.radians(360.0 / n)
    cos_sep = math.sin(e) ** 2 * math.cos(d) + math.cos(e) ** 2
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def target_offsets_px(n, radius_px, layout="cross"):
    """(dx, dy) pixel offsets from the layout center for n targets, evenly
    spaced on a circle -- azimuth 0 = up, going clockwise. "cross" starts
    at up (compass points); "grid" starts 45deg off (diagonals) -- same
    circle, same spacing, just rotated."""
    start_deg = -90.0 if layout == "cross" else -90.0 + (180.0 / n)
    offsets = []
    for i in range(n):
        az = math.radians(start_deg + i * (360.0 / n))
        offsets.append((radius_px * math.cos(az), radius_px * math.sin(az)))
    return offsets


def compute_layout(n, eccentricity_deg, stimulus_size_deg, viewing_distance_cm,
                    px_per_mm, work_w_px, work_h_px, layout="cross", margin_px=70):
    """Everything a display backend needs to compute before it draws the
    targets, factored out so every rendering backend gets pixel-identical
    target positions for the same physical parameters.

    px_per_mm may be a single number (square pixels) OR an (x, y) pair for
    displays whose horizontal and vertical pixel density differ. That
    happens whenever the panel is driven at a non-native aspect ratio, and
    it is NOT a rounding detail: measured on the UNO Q at 1680x1050 on a
    690x291mm ultrawide, density is 2.435 px/mm horizontally vs 3.608
    vertically -- 48% apart. Feeding the average (the old behaviour) drew
    targets that were square in PIXELS but 4.34deg x 2.93deg on the
    subject's retina, at 8.66deg/5.87deg eccentricity instead of the
    requested 3.5deg square at 7deg -- with the vertical size below the
    >=3deg spec. Scaling each axis by its own density fixes that, so the
    same configuration produces the same PHYSICAL stimulus on any panel.

    Returns a dict:
      width, height   : ideal bounding-box size in px -- NOT necessarily
                         square when pixels aren't. What a WINDOWED
                         backend should size its window to; a fullscreen
                         backend centres this content in the real screen.
      radius_px_x/_y  : target-circle radius in px, per axis
      size_px_w/_h    : target size in px, per axis (a physical square)
      radius_px, size_px : mean-axis values, kept for reporting and for
                         callers that just want one number
      offsets         : list of (dx, dy) px offsets from the layout
                         center, one per target, in the same order the
                         frequencies were given in
      sep_deg         : actual nearest-neighbor angular separation
      ok              : True if sep_deg >= 5.0 (SSVEP spec minimum)
      scale_applied   : None, or the shrink factor if the ideal layout
                         didn't fit work_w_px x work_h_px
      actual_ecc_deg, actual_size_deg : populated only when scaled down,
                         otherwise equal to the requested values
    """
    try:
        ppmm_x, ppmm_y = px_per_mm
    except TypeError:
        ppmm_x = ppmm_y = px_per_mm

    dist_mm = viewing_distance_cm * 10.0
    # Work in PHYSICAL millimetres first, then convert per axis -- that's
    # what keeps the stimulus a true square on non-square pixels.
    radius_mm = dist_mm * math.tan(math.radians(eccentricity_deg))
    size_mm = 2 * dist_mm * math.tan(math.radians(stimulus_size_deg / 2))
    sep_deg = pairwise_separation_deg(n, eccentricity_deg)
    ok = sep_deg >= 5.0

    scale = 1.0
    actual_ecc, actual_size = eccentricity_deg, stimulus_size_deg
    scale_applied = None
    eff_margin = margin_px

    def extents(sc, marg):
        rx, ry = radius_mm * sc * ppmm_x, radius_mm * sc * ppmm_y
        sw, sh = size_mm * sc * ppmm_x, size_mm * sc * ppmm_y
        return rx, ry, sw, sh, 2 * (rx + sw / 2 + marg), 2 * (ry + sh / 2 + marg)

    rx, ry, sw, sh, need_w, need_h = extents(1.0, margin_px)
    avail_w, avail_h = work_w_px - 20, work_h_px - 20
    if (need_w > avail_w or need_h > avail_h) and avail_w > 0 and avail_h > 0:
        # margin_px is fixed padding, not part of the visual-angle geometry,
        # so solve for the scale that lands inside the budget once the
        # margin is subtracted -- a naive avail/needed factor overshoots.
        eff_margin = min(margin_px, max(1.0, min(avail_w, avail_h) * 0.08))
        core_w, core_h = 2 * (radius_mm * ppmm_x) + size_mm * ppmm_x, \
                          2 * (radius_mm * ppmm_y) + size_mm * ppmm_y
        s_w = (avail_w - 2 * eff_margin) / core_w if core_w > 0 else 1.0
        s_h = (avail_h - 2 * eff_margin) / core_h if core_h > 0 else 1.0
        # One scale for BOTH axes -- scaling them independently would
        # distort the stimulus back into a rectangle, which is the bug
        # this function exists to avoid.
        scale = max(0.05, min(s_w, s_h))
        rx, ry, sw, sh, need_w, need_h = extents(scale, eff_margin)
        actual_ecc = math.degrees(math.atan(radius_mm * scale / dist_mm))
        actual_size = 2 * math.degrees(math.atan(size_mm * scale / 2 / dist_mm))
        scale_applied = scale

    offsets = []
    for dx, dy in target_offsets_px(n, 1.0, layout):   # unit circle, then per-axis
        offsets.append((dx * rx, dy * ry))

    return {
        "width": int(round(need_w)), "height": int(round(need_h)),
        "radius_px_x": rx, "radius_px_y": ry,
        "size_px_w": sw, "size_px_h": sh,
        "radius_px": (rx + ry) / 2.0, "size_px": (sw + sh) / 2.0,
        "offsets": offsets, "sep_deg": sep_deg, "ok": ok,
        "scale_applied": scale_applied,
        "actual_ecc_deg": actual_ecc, "actual_size_deg": actual_size,
    }


def print_layout_report(info, n, viewing_distance_cm, px_per_mm, tag="[layout]"):
    """The startup diagnostic block every display backend prints,
    factored out so every backend reports identically."""
    try:
        ppx, ppy = px_per_mm
    except TypeError:
        ppx = ppy = px_per_mm
    dens = (f"{ppx:.2f} px/mm" if abs(ppy - ppx) < 1e-6 else
            f"{ppx:.2f}x{ppy:.2f} px/mm, {100*abs(ppy-ppx)/ppx:.0f}% non-square")
    print(f"{tag} {n} targets @ {viewing_distance_cm:.0f}cm viewing distance ({dens}):")
    if abs(info["size_px_w"] - info["size_px_h"]) < 0.5:
        print(f"    radius {info['radius_px']:.0f}px, stimulus {info['size_px_w']:.0f}px square")
    else:
        # Deliberately different pixel counts per axis -- that is what makes
        # the target a PHYSICAL square on a non-square-pixel panel.
        print(f"    radius {info['radius_px_x']:.0f}x{info['radius_px_y']:.0f}px, "
              f"stimulus {info['size_px_w']:.0f}x{info['size_px_h']:.0f}px "
              f"(physically square, corrected for non-square pixels)")
    print(f"    nearest-neighbor separation {info['sep_deg']:.1f} deg (spec wants >=5deg) -- "
          f"{'OK' if info['ok'] else 'WARNING: BELOW 5deg -- increase eccentricity or verify viewing distance'}")
    if info["scale_applied"] is not None:
        print(f"    WARNING: ideal layout exceeded usable screen area -- scaled down by "
              f"{info['scale_applied']:.2f}x to fit. Actual on-screen eccentricity is now "
              f"{info['actual_ecc_deg']:.1f} deg, stimulus size {info['actual_size_deg']:.1f} deg.")


def outward_label_pos(cx, cy, dx, dy, box_w, box_h, lab_w, lab_h, gap=8.0):
    """Centre point for a target's "N Hz" caption, placed on the OUTER
    edge -- the side facing away from the layout centre. The inner edge is
    reserved (the game puts its turret there), and a caption simply drawn
    below every square lands on top of the top target's gun.

    Returns (x, y) for the label's CENTRE.

    The distance is solved rather than guessed. Scaling the clearance by
    the unit direction component looks right but under-shoots on
    diagonals: at (0.707, -0.707) it only pushes 71% of the needed
    distance, which measured as all four captions overlapping their
    squares in "grid" layout while "cross" looked fine. Instead, find the
    travel that separates the two rectangles on EITHER axis and take the
    smaller -- correct for axis-aligned and diagonal placements alike.
    """
    dist = math.hypot(dx, dy)
    if dist < 1e-9:                       # single target dead centre
        return cx, cy + box_h / 2.0 + gap + lab_h / 2.0
    ox, oy = dx / dist, dy / dist
    cands = []
    if abs(ox) > 1e-9:
        cands.append((box_w / 2.0 + lab_w / 2.0 + gap) / abs(ox))
    if abs(oy) > 1e-9:
        cands.append((box_h / 2.0 + lab_h / 2.0 + gap) / abs(oy))
    t = min(cands)
    return cx + dx + ox * t, cy + dy + oy * t
