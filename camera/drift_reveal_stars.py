#!/usr/bin/env python3
"""Measure star drift between successive full-resolution frames.

Takes one NEF every INTERVAL seconds (default 20), detects stars, and reports
how the field has drifted relative to the first frame -- direction as a clock
position (12:00 = up in the displayed image, 3:00 = right, 6:00 = down,
9:00 = left) and rate in pixels per minute.

No live-view or other persistent camera-mode changes are made, so Ctrl-C
leaves the camera in exactly the state it was found.

Usage
-----
    drift_reveal_stars.py [--exposure SEC] [--iso N] [--interval SEC]
    drift_reveal_stars.py --sim "frames/*.NEF"   # offline test

Matching algorithm: for every (new_star, ref_star) pair the displacement
new - ref is computed and cast as a vote in a 2-D histogram.  The peak bin is
the consensus drift.  After the first frame the search is narrowed to a window
around the previously measured drift, which prevents false peaks when the
drift per interval is small.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from find_it import capture_one, decode_raw, detect_stars  # noqa: E402

# ---- tunables ---------------------------------------------------------------
DEFAULT_INTERVAL_S = 20.0
VOTE_BIN_PX   = 4.0    # histogram bin size for the offset vote (px)
PRIOR_RADIUS  = 60.0   # search window around previous drift after frame 0 (px)
MIN_VOTES     = 4      # minimum peak votes to trust a match
MIN_STARS     = 5      # minimum detected stars needed to attempt matching


# =============================================================================
# Optional center crop  (applied after decode, before detection)
# =============================================================================

def center_crop(rgb16: np.ndarray, size: int) -> np.ndarray:
    """Return a (size × size) center crop of rgb16 (HxWx3)."""
    h, w = rgb16.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return rgb16[y0:y0 + size, x0:x0 + size]


# =============================================================================
# Drift estimation by pairwise offset voting
# =============================================================================

def estimate_drift(
    ref_stars: list[dict],
    new_stars: list[dict],
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, int] | None:
    """Return (drift_xy, n_votes) or None if matching fails.

    drift_xy = (dx, dy): a star at (x, y) in the reference frame appears at
    (x+dx, y+dy) in the new frame.  n_votes is the histogram peak height,
    i.e. how many (new, ref) star pairs agreed on this displacement.
    """
    if len(ref_stars) < MIN_STARS or len(new_stars) < MIN_STARS:
        return None

    ref_xy = np.array([[s["x"], s["y"]] for s in ref_stars], float)
    new_xy = np.array([[s["x"], s["y"]] for s in new_stars], float)

    # All pairwise offsets: new[j] - ref[i], shape (N_new * N_ref, 2)
    offsets = (new_xy[:, None, :] - ref_xy[None, :, :]).reshape(-1, 2)

    # Narrow the search to a window around the prior (all offsets if no prior)
    if prior is not None:
        mask = np.linalg.norm(offsets - prior, axis=1) <= PRIOR_RADIUS
        offsets = offsets[mask]
        if len(offsets) < MIN_VOTES:
            return None

    dx, dy = offsets[:, 0], offsets[:, 1]
    pad = VOTE_BIN_PX

    nx = max(3, int(math.ceil((dx.max() - dx.min()) / VOTE_BIN_PX)) + 1)
    ny = max(3, int(math.ceil((dy.max() - dy.min()) / VOTE_BIN_PX)) + 1)
    H, xe, ye = np.histogram2d(
        dx, dy, bins=[nx, ny],
        range=[[dx.min() - pad, dx.max() + pad],
               [dy.min() - pad, dy.max() + pad]])

    pi, pj = np.unravel_index(H.argmax(), H.shape)
    votes = int(H[pi, pj])
    if votes < MIN_VOTES:
        return None

    # Refine: mean position of all offsets that fell in the winning bin
    in_peak = ((dx >= xe[pi]) & (dx < xe[pi + 1]) &
               (dy >= ye[pj]) & (dy < ye[pj + 1]))
    drift = np.array([dx[in_peak].mean(), dy[in_peak].mean()])
    return drift, votes


# =============================================================================
# Clock-dial direction
# =============================================================================

def clock_direction(dx: float, dy: float) -> str:
    """Express a pixel displacement as a clock position string.

    12:00 = up (-dy in image coords), 3:00 = right (+dx),
    6:00 = down (+dy),              9:00 = left (-dx).
    """
    angle_deg = math.degrees(math.atan2(dx, -dy)) % 360.0
    hour_f = angle_deg / 30.0        # 360 deg / 12 h = 30 deg per h
    h = int(hour_f) % 12 or 12       # map 0 → 12
    m = int(round((hour_f % 1) * 60)) % 60
    return f"{h}:{m:02d}"


# =============================================================================
# Helpers
# =============================================================================

def _sleep_to_next(t_start: float, interval_s: float, offline: bool) -> None:
    """Sleep for however long is left of the current interval."""
    if offline:
        return
    remaining = interval_s - (time.monotonic() - t_start)
    if remaining > 0.1:
        print(f"  [sleep {remaining:.1f} s]")
        time.sleep(remaining)


# =============================================================================
# CLI + main loop
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exposure", type=float, default=None,
                   help="shutter in seconds (default: camera's current setting)")
    p.add_argument("--iso", default=None,
                   help="ISO to set before shooting (optional)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help=f"target seconds between shots (default {DEFAULT_INTERVAL_S})")
    p.add_argument("--max-stars", type=int, default=80,
                   help="max stars to detect per frame (default 80)")
    p.add_argument("--crop", type=int, default=None, metavar="PX",
                   help="use only a centered square crop of this side (px) for "
                        "detection; speeds up detect_stars ~6x on a 2000 px crop "
                        "without affecting drift accuracy (default: full image)")
    p.add_argument("--sim", metavar="GLOB",
                   help='offline test with existing files, e.g. --sim "frames/*.NEF"')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    offline = bool(args.sim)

    print("drift_reveal_stars — Ctrl-C to stop\n")

    sim_files: list[str] = []
    if offline:
        sim_files = sorted(glob.glob(args.sim))
        if not sim_files:
            sys.exit(f"[sim] no files matched: {args.sim!r}")
        print(f"[sim] {len(sim_files)} file(s): "
              + ", ".join(os.path.basename(f) for f in sim_files) + "\n")

    ref_stars: list[dict] | None = None
    t_ref: float | None = None
    prev_drift: np.ndarray | None = None
    frame_n = 0

    try:
        while True:
            t_loop_start = time.monotonic()

            # -- acquire frame ------------------------------------------------
            if offline:
                if frame_n >= len(sim_files):
                    print("[sim] no more files — done.")
                    break
                path = sim_files[frame_n]
                label = os.path.basename(path)
                print(f"[frame {frame_n}] {label}", end="  ", flush=True)
                with open(path, "rb") as fh:
                    raw = fh.read()
                t_now = t_loop_start
                name = label
            else:
                print(f"[frame {frame_n}] shooting …", end="  ", flush=True)
                raw, name = capture_one(args.exposure, args.iso)
                t_now = time.monotonic()

            # -- decode + detect ----------------------------------------------
            print("decoding …", end="  ", flush=True)
            rgb16 = decode_raw(raw, name)
            if args.crop:
                rgb16 = center_crop(rgb16, args.crop)
            stars = detect_stars(rgb16, args.max_stars)
            print(f"{len(stars)} stars detected")

            # -- first frame: set reference -----------------------------------
            if ref_stars is None:
                ref_stars = stars
                t_ref = t_now
                frame_n += 1
                crop_note = f", cropped to {args.crop}×{args.crop} px" if args.crop else ""
                print(f"  reference frame set ({len(ref_stars)} stars{crop_note})\n")
                _sleep_to_next(t_loop_start, args.interval, offline)
                continue

            # -- subsequent frames: match and report --------------------------
            elapsed_s = t_now - t_ref
            elapsed_min = elapsed_s / 60.0

            if len(stars) < MIN_STARS:
                print(f"  only {len(stars)} stars — skipping (clouds / blur?)\n")
                frame_n += 1
                _sleep_to_next(t_loop_start, args.interval, offline)
                continue

            result = estimate_drift(ref_stars, stars, prior=prev_drift)
            frame_n += 1

            if result is None:
                print(f"  match failed  "
                      f"(ref {len(ref_stars)} stars, new {len(stars)} stars, "
                      f"elapsed {elapsed_s:.0f} s)\n")
            else:
                drift, votes = result
                dx, dy = float(drift[0]), float(drift[1])
                magnitude_px = math.hypot(dx, dy)
                rate_px_min = magnitude_px / elapsed_min if elapsed_min > 0 else 0.0
                direction = clock_direction(dx, dy)
                prev_drift = drift

                print(f"  t+{elapsed_s:5.0f} s  |  "
                      f"total drift  dx={dx:+.1f} dy={dy:+.1f} px  |  "
                      f"{direction}  @  {rate_px_min:.2f} px/min  "
                      f"({votes} votes)\n")

            _sleep_to_next(t_loop_start, args.interval, offline)

    except KeyboardInterrupt:
        print("\n[stop] Ctrl-C — no camera settings were changed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
