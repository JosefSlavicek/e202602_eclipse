#!/usr/bin/env python
"""Standalone preview of v6/plan0.md's static-atmosphere correction, no pipeline run.

pipeline.py now fits and applies this correction automatically, from its own run's
calibration-capable frames (eclipse_v6.atmosphere.calibrate_from_source), for any
--input-mode. This script exists to inspect that fit on its own -- to see the per-frame
solve quality before committing to a full pipeline run -- not because a separate build
step is required any more.

Two modes:

  real (default) -- point at a directory of real 2026 NEF frames. Thin wrapper around the
      same calibrate_from_source() pipeline.py calls.

  synthetic-test -- the data_atmosphere/ closed-loop path this script started as: real
      detection on the 2024 dev set's actual JPEG pixels, using the already-known confirmed
      2024 stars and the cached dev_solution() geometry as a shortcut a first-time 2026 run
      won't have. Useful for re-validating round 4's pipeline.py comparison, not for 2026
      data -- and unlike real mode, NOT what pipeline.py does internally, since pipeline.py
      has no such precomputed shortcut for any input.

    python v6/build_atmosphere_correction.py --nef-dir /path/to/2026/NEFs

CPU only (no CUDA needed). real mode additionally needs internet (a live Gaia query).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "find_stars"))

import starlib as S  # noqa: E402
from eclipse_v6 import atmosphere as A  # noqa: E402
from eclipse_v6.inputs import NefSource  # noqa: E402
from eclipse_v6.stage0 import get_image_infos, get_info_from_exif  # noqa: E402

DEFAULT_NEF_DIR = os.path.expanduser("~/e202602_eclipse/data_2026")
DEFAULT_ATMOSPHERE_DIR = os.path.expanduser("~/e202602_eclipse/data_atmosphere")
DEFAULT_OUT = os.path.join(HERE, "atmosphere_trend.pkl")


# --------------------------------------------------------------------------------------- #
# real mode -- exactly what pipeline.py does internally, standalone
# --------------------------------------------------------------------------------------- #
def real_trend(nef_dir, exp_min, min_snr, min_stars):
    source = NefSource(nef_dir)
    image_infos = get_image_infos(source)
    device = torch.device("cpu")
    return A.calibrate_from_source(image_infos, device, exp_min=exp_min,
                                   peak_thresh=min_snr, min_stars=min_stars)


# --------------------------------------------------------------------------------------- #
# synthetic-test mode (data_atmosphere/, the 2024 dev set's already-known geometry) --
# NOT what pipeline.py does; a validation-only shortcut kept for re-running round 4.
# --------------------------------------------------------------------------------------- #
def detect_confirmed_stars(g, snr_map, cs, search_radius=80, half=6, min_snr=6.0):
    """For each confirmed star's ORIGINAL (undistorted) position, search a window generous
    enough to catch the known worst-case compression, take the strongest SNR peak in that
    window, refine with a sub-pixel Gaussian fit. Returns arrays for whichever stars
    cleared min_snr -- most won't, at low exposure; that's expected, per round 2."""
    det_x, det_y, ra_ok, dec_ok, snr_ok = [], [], [], [], []
    h, w = snr_map.shape
    for x0, y0, ra, dec in zip(cs["x"], cs["y"], cs["ra"], cs["dec"]):
        yi0, xi0 = int(round(y0)), int(round(x0))
        y_lo, y_hi = max(0, yi0 - search_radius), min(h, yi0 + search_radius + 1)
        x_lo, x_hi = max(0, xi0 - search_radius), min(w, xi0 + search_radius + 1)
        patch = snr_map[y_lo:y_hi, x_lo:x_hi]
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        if patch[py, px] < min_snr:
            continue
        f = S.gauss2d_fit(g, x_lo + px, y_lo + py, half=half)
        if f is None:
            continue
        det_x.append(f["x"]); det_y.append(f["y"])
        ra_ok.append(ra); dec_ok.append(dec); snr_ok.append(float(patch[py, px]))
    return (np.array(det_x), np.array(det_y), np.array(ra_ok), np.array(dec_ok),
            np.array(snr_ok))


def synthetic_test_trend(atmosphere_dir, exp_min, min_snr, min_stars):
    """make_data_atmosphere.py's own EXIF-timestamped files, matched via the ALREADY known
    2024 dev_solution()/confirmed_stars() -- a shortcut real mode doesn't get."""
    with open(os.path.join(atmosphere_dir, "atmosphere.json")) as fh:
        hdr = json.load(fh)
    calib_files = sorted(f["file"] for f in hdr["frames"] if f["exp"] >= exp_min)
    paths = [os.path.join(atmosphere_dir, f) for f in calib_files]
    times = np.array([get_info_from_exif(p)[1] for p in paths], dtype=float)
    print(f"{len(calib_files)} calibration-capable frames (exp >= {exp_min} s), "
         f"spanning {times.max() - times.min():.1f} s")

    D = S.dev_solution()
    cs = S.confirmed_stars()
    moon, info, scale0 = D["moon"], D["info"], D["scale0"]
    p_base = D["sol"]["p"]

    calib = []
    for f, t in zip(calib_files, times):
        g = S.load_linear(os.path.join(atmosphere_dir, f))
        snr, _bg = S.dogsnr(g)
        det_x, det_y, ra_ok, dec_ok, snr_ok = detect_confirmed_stars(
            g, snr, cs, min_snr=min_snr)
        if len(det_x) < min_stars:
            print(f"  {f}: only {len(det_x)} real detections (need {min_stars}) -- skip")
            continue
        res = S.solve_plate(det_x, det_y, ra_ok, dec_ok, moon["ra"], moon["dec"],
                            info["cx"], info["cy"], scale0, use_distortion=True,
                            use_atmosphere=True, p0=p_base)
        ok = np.isfinite(res["rms"]) and res["nmatch"] >= max(min_stars, int(0.6 * len(det_x)))
        print(f"  {f}: {len(det_x)} real detections (median SNR {np.median(snr_ok):.1f}), "
             f"nmatch={res['nmatch']}, rms={res['rms']:.3f} px -> "
             f"{'used' if ok else 'REJECTED'}")
        if ok:
            calib.append((float(t), res["p"]))
    print(f"\n{len(calib)} calibration frames usable")
    if len(calib) < 4:
        sys.exit("too few usable calibration frames to fit a trend (need >= 4)")
    grid = A.reference_grid(info["cx"], info["cy"], 0.6 * info["cx"], 0.6 * info["cy"], n=9)
    return A.fit_trend([t for t, _ in calib], [p for _, p in calib], p_base, grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["real", "synthetic-test"], default="real")
    ap.add_argument("--nef-dir", default=DEFAULT_NEF_DIR,
                    help="real mode: directory of 2026 .NEF calibration frames")
    ap.add_argument("--atmosphere-dir", default=DEFAULT_ATMOSPHERE_DIR,
                    help="synthetic-test mode: data_atmosphere/ (needs atmosphere.json)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--exp-min", type=float, default=0.25,
                    help="calibration-capable exposure floor, matches check_06/round 2")
    ap.add_argument("--min-snr", type=float, default=None,
                    help="peak-detection SNR floor. Default 4.0 (real) or 6.0 "
                         "(synthetic-test)")
    ap.add_argument("--min-stars", type=int, default=6,
                    help="a calibration frame needs at least this many matched stars to "
                         "be kept")
    args = ap.parse_args()

    if args.mode == "real":
        min_snr = 4.0 if args.min_snr is None else args.min_snr
        trend = real_trend(args.nef_dir, args.exp_min, min_snr, args.min_stars)
    else:
        min_snr = 6.0 if args.min_snr is None else args.min_snr
        trend = synthetic_test_trend(args.atmosphere_dir, args.exp_min, min_snr,
                                     args.min_stars)

    with open(args.out, "wb") as fh:
        pickle.dump(trend, fh)
    print(f"wrote {args.out} (t_mean={trend.t_mean:.2f})")


if __name__ == "__main__":
    main()
