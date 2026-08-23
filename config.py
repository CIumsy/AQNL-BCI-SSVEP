#!/usr/bin/env python3
"""
config.py

Where the flickering squares go on screen, and how big they are.

Every entry point reads these same values: main.py on Windows, this
folder's own main.py on the Arduino UNO Q, and the zombie game. They live in one
file because both machines must agree exactly. If they differ even
slightly, the thresholds you calibrated on one machine are no longer
valid on the other.

The target frequencies are NOT here. They are in run_ssvep_detection.py,
because the classifier builds its filter bank and reference waves from
them.
"""

# EDIT ME: "cross" puts the squares up/right/down/left. "grid" puts them
# on the diagonals (a 2x2). Same circle and same spacing either way -- it
# is only a rotation. See run_ssvep_display.py for the picture.
LAYOUT = "cross"

# EDIT ME: how far your eyes are from the screen, in cm.
# MEASURE THIS. Do not guess. Every angle below is turned into pixels
# using this number, so if it is wrong, all of them are wrong and nothing
# will warn you.
VIEWING_DISTANCE_CM = 60.0

# EDIT ME: how far each square sits from the centre, in degrees of visual
# angle. The usable range is 4-10 degrees, where vision is still sharp.
# 7 is the middle of that range and puts neighbouring squares about 9.9
# degrees apart -- comfortably over the 5 degrees they need to stay
# separable. The real numbers are printed at startup so you can check
# them for your actual viewing distance.
ECCENTRICITY_DEG = 7.0

# EDIT ME: how big each square looks to your eye, in degrees. It needs to
# be at least 3 degrees to drive a strong enough response; 3.5 leaves a
# little margin. It is drawn as a true PHYSICAL square even on screens
# whose pixels are not square -- see compute_layout() in
# game_lib/ssvep_geometry.py.
STIMULUS_SIZE_DEG = 3.5
