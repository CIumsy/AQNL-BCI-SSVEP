#!/usr/bin/env python3
"""
visualize_sessions.py

Offline viewer for the calibration session logs run_ssvep_detection.py
writes to sessions/*.npz (raw uV windows + true labels, one file per run --
see SESSION_LOG_OUT in that script). Not part of the real-time pipeline;
run this by hand whenever you want to look at a recording. Lives inside
sessions/ itself, alongside the .npz files it reads.

For every session file, produces two PNGs in sessions/plots/:
  <name>_traces.png  -- one representative 2s raw waveform per target
                         frequency, per channel (O1/Oz/O2). Sanity check:
                         does anything look like a plausible EEG signal at
                         all, or is a channel flat/railed/pure drift?
  <name>_psd.png      -- average power spectrum per channel, one line per
                         target frequency, with dashed reference lines at
                         each frequency (and its 2nd harmonic). This is the
                         direct visual version of the "does the calibration
                         data actually contain the target frequency"
                         check -- a genuine SSVEP shows as a line PEAKING
                         at its own dashed reference line; if the lines are
                         flat/interchangeable, there's nothing to detect
                         regardless of what the classifier does with it.

If 2+ session files are present, also produces noise_trend.png: per-channel
AC noise (post-DC-removal std) across sessions in chronological order --
the same comparison that surfaced the 3x noise increase across your first
three sessions, as a picture instead of printed numbers.

Usage (run from anywhere -- paths are relative to this script, not cwd):
    python sessions/visualize_sessions.py          # all sessions/*.npz
    python sessions/visualize_sessions.py foo.npz  # just one
"""

import glob
import sys

import matplotlib
matplotlib.use("Agg")  # headless -- this script only ever writes PNGs
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

# ---- validated categorical palette (dataviz skill reference instance) ----
# Fixed assignment per entity so color means the same thing in every plot in
# this script: slots 1-3 for channel identity (traces/noise-trend), slots
# 1-4 for frequency-class identity (PSD) -- both draw from the same ordered
# list, each in its own context, per the "adjacent pairlist" validation
# (this order clears every adjacent-CVD gate for lines/bars up to 8 slots).
_PAL = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
}
CHANNEL_COLOR = {"O1": _PAL["blue"], "Oz": _PAL["orange"], "O2": _PAL["aqua"]}
CLASS_COLORS = [_PAL["blue"], _PAL["orange"], _PAL["aqua"], _PAL["yellow"]]  # by frequency rank

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

CHAN_NAMES = ["O1", "Oz", "O2"]


def plot_traces(sess, out_path):
    w, labels, freqs, sfreq = sess["windows"], sess["labels"], sess["frequencies"], float(sess["sfreq"])
    n_classes = len(freqs)
    t = np.arange(w.shape[2]) / sfreq

    fig, axes = plt.subplots(3, n_classes, figsize=(3.2 * n_classes, 6.0), sharex=True)
    for ci, f in enumerate(freqs):
        idx = np.where(labels == ci)[0]
        win = w[idx[len(idx) // 2]] if len(idx) else None  # a representative window, mid-trial
        for chi, name in enumerate(CHAN_NAMES):
            ax = axes[chi, ci]
            if win is not None:
                sig = win[chi] - win[chi].mean()
                ax.plot(t, sig, color=INK, linewidth=0.8)
            if chi == 0:
                ax.set_title(f"{f} Hz", color=INK, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{name}\nµV", fontsize=9)
            if chi == 2:
                ax.set_xlabel("s", fontsize=9)
    fig.suptitle("Representative raw window per target (DC-removed)", color=INK, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_psd(sess, out_path):
    w, labels, freqs, sfreq = sess["windows"], sess["labels"], sess["frequencies"], float(sess["sfreq"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for chi, name in enumerate(CHAN_NAMES):
        ax = axes[chi]
        for ci, f in enumerate(freqs):
            idx = np.where(labels == ci)[0]
            if len(idx) == 0:
                continue
            pxx_list = []
            for i in idx:
                sig = w[i, chi] - w[i, chi].mean()
                fx, pxx = welch(sig, fs=sfreq, nperseg=min(1024, len(sig)))
                pxx_list.append(pxx)
            pxx_mean = np.mean(pxx_list, axis=0)
            band = (fx >= 3) & (fx <= 45)
            ax.plot(fx[band], 10 * np.log10(pxx_mean[band] + 1e-12),
                    color=CLASS_COLORS[ci % len(CLASS_COLORS)], linewidth=1.6, label=f"{f} Hz")
            ax.axvline(f, color=CLASS_COLORS[ci % len(CLASS_COLORS)], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(name, color=INK)
        ax.set_xlabel("Hz")
        if chi == 0:
            ax.set_ylabel("power (dB)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Average power spectrum per channel, by target frequency\n"
                  "(dashed line = that class's own target -- a real SSVEP peaks AT its own line)",
                  color=INK, y=1.06, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_noise_trend(sessions_meta, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(sessions_meta))
    for chi, name in enumerate(CHAN_NAMES):
        ys = [m["ac_std"][chi] for m in sessions_meta]
        ax.plot(xs, ys, marker="o", markersize=6, color=CHANNEL_COLOR[name], linewidth=2, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in sessions_meta], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AC noise, std (µV)")
    ax.set_title("Per-channel noise floor across sessions, chronological", color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    # Location-independent (not cwd-dependent): this script lives inside
    # sessions/ alongside the .npz files it reads, so default paths are
    # relative to ITS OWN location, not wherever it happens to be launched
    # from -- `python visualize_sessions.py` and `python
    # sessions/visualize_sessions.py` (from the project root) both work.
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "*.npz")))
    if not paths:
        print(f"No session files found (looked in {here}). Pass a path explicitly.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(here, "plots")
    os.makedirs(out_dir, exist_ok=True)

    noise_meta = []
    for path in paths:
        d = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"{path}: {d['windows'].shape[0]} windows, {len(d['frequencies'])} classes @ {int(d['sfreq'])}Hz")

        plot_traces(d, f"{out_dir}/{name}_traces.png")
        plot_psd(d, f"{out_dir}/{name}_psd.png")
        print(f"  wrote {out_dir}/{name}_traces.png, {out_dir}/{name}_psd.png")

        w = d["windows"]
        w_ac = w - w.mean(axis=2, keepdims=True)
        noise_meta.append({"label": name.replace("session_", ""), "ac_std": w_ac.std(axis=(0, 2))})

    if len(noise_meta) >= 2:
        plot_noise_trend(noise_meta, f"{out_dir}/noise_trend.png")
        print(f"  wrote {out_dir}/noise_trend.png")


if __name__ == "__main__":
    main()
