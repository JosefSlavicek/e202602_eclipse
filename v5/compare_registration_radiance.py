#!/usr/bin/env python3
"""Does the calibrated photometry move the cross-exposure registration?

Stage 2 aligns each consecutive exposure pair by pre-scaling the longer frame with a
fitted per-pair exponent (`gamma`), then minimising an L1 residual on Fourier-cleaned
images.  That residual is NOT scale-invariant, and the saturation mask handed to the
grid search is itself derived from the same fitted exponent:

    apriori_valid = (img0 * ((t1 / t0) ** (1.0 / gamma)) <= 0.9)      # stage2.py

The calibration that runs *after* stage 2 knows the real answer: `f` inverts the stored
value to accumulated light and `c_k` corrects the shutter time.  This script asks, without
re-running the pipeline, whether feeding that back into registration would move anything.

Three registrations per pair, all through the same `grid_search_registration`, differing
only in the images handed to it:

  control  stored values, scaled by the pickled `gamma` — reproduces stage 2 exactly.
           Its delta against `v5-stage2.pkl` must be ~0; that is what makes the other two
           deltas trustworthy rather than an artefact of this harness's conventions.
  linear   physical radiance from `source.load_radiance`, no scaling at all (radiance is
           light per unit time — the same quantity in both frames, so the exposure ratio
           is already gone).  Isolates the effect of a correct scale while keeping the
           current residual's weighting.
  log      log radiance.  An exposure ratio becomes an additive constant, which the
           existing `remove_lowfeq` strips, so the objective is exposure-invariant by
           construction.  This is the reformulation that would let registration stop
           needing a photometric estimate at all.  Note it also reweights the residual:
           in log space the faint outer streamers count as much as the inner corona.

Cost: two passes over every frame (stored-value averages for the control, radiance
averages for the other two) plus 3 grid searches per pair, versus stage 2's 2.  Use
`--limit-pairs 2` for a quick trial first, and `--space log --no-control` once the
control has been shown to reproduce.

    python v5/compare_registration_radiance.py --workdir /home/slavik/tmp/eclipse_v5

Reads `v5-stage1.pkl`, `v5-stage2.pkl`, `v5-calib.pkl` from the workdir. Writes nothing
except the JSON report (`--out`, default `<workdir>/v5-regcheck.json`).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v5 import calib as ca                                        # noqa: E402
from eclipse_v5 import stage2 as s2                                       # noqa: E402
from eclipse_v5.device import configure_cuda_visible_devices, require_cuda  # noqa: E402
from eclipse_v5.inputs import attach_source, make_source                  # noqa: E402
from eclipse_v5.merge import average_exposure_radiance                    # noqa: E402
from eclipse_v5.utils import (                                            # noqa: E402
    compute_weighted_average,
    grid_search_registration,
    moon_median,
)

# --- constants ------------------------------------------------------------------------
COVER_THRESH = 0.5           # same as merge.MIN_STACK_COVER: half a frame's worth of valid
                             # contributions. Below it the pixel is saturated, off-frame or
                             # moon, and neither its value nor its neighbours' can be trusted.
FLOOR_QUANTILE = 0.05        # the log floor, taken on the longer exposure (it measures the
                             # faint end best). Shared by both images of a pair so the
                             # additive offset between them survives untouched.
QUANTILE_SUBSAMPLE = 1 << 22  # torch.quantile refuses very large inputs; stride down to this
ROT_EVAL_RADIUS_FACTOR = 3.0  # report a rotation delta as the arc it moves at 3 moon radii,
                              # i.e. out where the corona structure actually lives
SIGNIFICANT_PX = 0.1         # grid_search_registration's own resolution floor: below this it
                             # does not refine further, so a smaller delta means nothing
CONTROL_TOLERANCE_PX = 1e-3  # the control re-runs a deterministic search on identical inputs


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=Path("/home/slavik/tmp/eclipse_v5"),
                    help="holds v5-stage1.pkl, v5-stage2.pkl, v5-calib.pkl")
    ap.add_argument("--input-mode", choices=["jpg", "atmosphere", "nef", "inject"], default="jpg")
    ap.add_argument("--jpg-dir", type=Path,
                    default=Path("/home/slavik/e202602_eclipse/data"))
    ap.add_argument("--atmosphere-dir", type=Path,
                    default=Path("/home/slavik/e202602_eclipse/data_atmosphere"))
    ap.add_argument("--nef-dir", type=Path, default=Path("/home/slavik/tmp/eclipse_fake_imgs"))
    ap.add_argument("--space", choices=["linear", "log", "both"], default="both",
                    help="which radiance-space registration(s) to run against the control")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the stage-2 reproduction. Only once it has been seen to pass: "
                         "without it a nonzero delta cannot be told from a harness bug.")
    ap.add_argument("--limit-pairs", type=int, default=0,
                    help="only the first N pairs (0 = all). Use for a cheap first look.")
    ap.add_argument("--include-uncalibrated", action="store_true",
                    help="also compare pairs involving the two shortest exposures, which the "
                         "calibration drops. Their c_k falls back to 1.0, so the comparison "
                         "there tests the response curve only, not the shutter correction.")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON report path; default <workdir>/v5-regcheck.json")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
#  Per-exposure averages                                                      #
# --------------------------------------------------------------------------- #
def average_gray(group, opt_results, exp, device):
    """Stage 2's own per-exposure average of STORED values, for the control."""
    abs_xy = torch.from_numpy(opt_results[exp]["abs_xy"]).to(device)
    abs_angle_t = torch.from_numpy(opt_results[exp]["abs_angle_t"]).to(device)
    avg_img, _, _ = compute_weighted_average(
        group, abs_xy, abs_angle_t, device, keep_warped=False
    )
    return avg_img


def average_radiance(group, opt_results, exp, source, device):
    """Per-exposure average in physical brightness, via the calibrated response."""
    abs_xy = torch.from_numpy(opt_results[exp]["abs_xy"]).to(device)
    abs_angle_t = torch.from_numpy(opt_results[exp]["abs_angle_t"]).to(device)
    Lbar, _var, covered, _moon_out = average_exposure_radiance(
        group, abs_xy, abs_angle_t, source, device
    )
    return Lbar, covered


class AverageCache:
    """Rolling cache over consecutive pairs: each exposure is built once, used twice.

    Tensors are held on the CPU (three full-res float32 arrays per exposure is ~290 MB at
    4000x6000) and moved to the GPU inside the pair loop.
    """

    def __init__(self, exposure_groups, opt_results, source, device, want_gray, want_radiance):
        self.groups = exposure_groups
        self.opt_results = opt_results
        self.source = source
        self.device = device
        self.want_gray = want_gray
        self.want_radiance = want_radiance
        self._cache = {}

    def get(self, exp):
        if exp not in self._cache:
            group = self.groups[exp]
            entry = {}
            if self.want_gray:
                entry["gray"] = average_gray(group, self.opt_results, exp, self.device).cpu()
            if self.want_radiance:
                L, covered = average_radiance(
                    group, self.opt_results, exp, self.source, self.device
                )
                entry["L"] = L.cpu()
                entry["covered"] = covered.cpu()
                del L, covered
            self._cache[exp] = entry
            torch.cuda.empty_cache()
        return self._cache[exp]

    def drop_all_but(self, keep):
        for exp in [e for e in self._cache if e not in keep]:
            del self._cache[exp]


# --------------------------------------------------------------------------- #
#  Radiance -> registration input                                             #
# --------------------------------------------------------------------------- #
def _low_quantile(x, valid, q):
    """q-quantile of `x` over `valid`, on a strided subsample (torch.quantile caps size)."""
    v = x[valid]
    assert v.numel() > 0, "no valid pixels to take a quantile over"
    if v.numel() > QUANTILE_SUBSAMPLE:
        v = v[:: (v.numel() // QUANTILE_SUBSAMPLE) + 1]
    return float(torch.quantile(v.float(), q))


def radiance_pair_images(L0, cov0, L1, cov1, space):
    """Both frames of a pair as one comparable image each, plus the target's valid mask.

    Radiance is already the same physical quantity in both frames, so nothing is scaled
    here — that is the entire point.  Uncovered pixels (off-frame, moon, saturated) are
    held at the floor rather than zeroed: a zero is a hard edge that `remove_lowfeq`
    would smear into every angular frequency, while a constant is exactly what the
    low-frequency strip is there to remove.  Saturated cores need no special handling
    beyond that; `clean_polar_fft`'s antiprot clip already flattens the top of the range.
    """
    # A pixel no frame covered has Lbar = 0/EPS, and a group of one bad frame can leave a
    # non-finite behind; either would poison the quantile and then the whole FFT row.
    valid0 = (cov0 >= COVER_THRESH) & torch.isfinite(L0) & (L0 > 0)
    valid1 = (cov1 >= COVER_THRESH) & torch.isfinite(L1) & (L1 > 0)
    floor = _low_quantile(L1, valid1, FLOOR_QUANTILE)
    assert floor > 0, floor

    def prep(L, valid):
        x = torch.where(valid, L, torch.full_like(L, floor)).clamp(min=floor)
        if space == "log":
            return torch.log(x) - math.log(floor)          # >= 0, common offset kept
        return x / floor                                   # common factor: argmin unchanged

    # The saturation mask stage 2 approximated by predicting img1's clipping from img0.
    # Here both exposures state it directly. valid1 is used un-warped: the pair is within
    # a few tens of pixels of alignment already, and the mask edge sits in the saturated
    # core where nothing is being matched anyway.
    apriori_valid = (valid0 & valid1).to(torch.float32)
    return prep(L0, valid0), prep(L1, valid1), apriori_valid


# --------------------------------------------------------------------------- #
#  Comparison                                                                 #
# --------------------------------------------------------------------------- #
def displacement(a, b, r_eval):
    """(translation px, rotation px at r_eval, total px) between two (si, sj, rot) triples."""
    d_i = a[0] - b[0]
    d_j = a[1] - b[1]
    d_rot_deg = a[2] - b[2]
    d_trans = math.hypot(d_i, d_j)
    d_rot_px = abs(math.radians(d_rot_deg)) * r_eval
    return d_trans, d_rot_px, d_trans + d_rot_px


def register_stored_values(img0, img1, gamma, t0, t1, moon0, moon1, device):
    """Stage 2's final registration step, verbatim, on the pickled gamma."""
    scale = (t0 / t1) ** (1.0 / gamma)
    img1_scaled = (img1 * scale).clamp(0.0, 1.0)
    return s2.register_cross_exposure(img0, img1_scaled, moon0, moon1, gamma, t0, t1, device)


def register_radiance(x0, x1, apriori_valid, moon0, moon1, device):
    """Same grid search, same starting bracket, on radiance images and an honest mask."""
    initial_shift_half = 2.0 * (5.0 * (2.0 + 2.0) + 0.0 + 3.0)   # as stage2.register_cross_exposure
    return grid_search_registration(
        x0, x1, moon0, moon1, initial_shift_half, device, apriori_valid=apriori_valid
    )


def main():
    args = _parse_args()
    workdir = args.workdir
    out_path = args.out or (workdir / "v5-regcheck.json")

    source = make_source(
        args.input_mode, jpg_dir=args.jpg_dir, nef_dir=args.nef_dir,
        atmosphere_dir=args.atmosphere_dir,
    )
    device = torch.device("cuda")

    with open(workdir / "v5-stage1.pkl", "rb") as fd:
        exposure_groups = pickle.load(fd)
        _reg = pickle.load(fd)
        opt_results = pickle.load(fd)
    with open(workdir / "v5-stage2.pkl", "rb") as fd:
        cross_reg = pickle.load(fd)
        gamma_by_pair = pickle.load(fd)
    calib = ca.load(workdir / "v5-calib.pkl")

    attach_source(exposure_groups, source)
    source.set_calibration(calib)
    print(f"Input mode: {source.kind} (is_linear={source.is_linear})")
    print(f"Loaded {len(exposure_groups)} exposure groups, {len(cross_reg)} registered pairs, "
          f"calibration over {len(calib.exposures)} exposures "
          f"(corrections {calib.corrections.min():.4f}..{calib.corrections.max():.4f})")

    calibrated = {float(t) for t in calib.exposures}
    exposure_times_sorted = sorted(exposure_groups.keys())
    moon_by_exp = {exp: moon_median(exposure_groups[exp]) for exp in exposure_times_sorted}

    spaces = ["linear", "log"] if args.space == "both" else [args.space]
    want_control = not args.no_control
    cache = AverageCache(
        exposure_groups, opt_results, source, device,
        want_gray=want_control, want_radiance=True,
    )

    pairs = []
    for idx in range(len(exposure_times_sorted) - 1):
        t0 = exposure_times_sorted[idx]
        t1 = exposure_times_sorted[idx + 1]
        if (t0, t1) not in cross_reg:
            continue                       # stage 2 skipped it (no poses, or a 1-frame group)
        if not args.include_uncalibrated and not (t0 in calibrated and t1 in calibrated):
            continue
        pairs.append((t0, t1))
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    assert pairs, "no comparable pairs; --include-uncalibrated, or check the workdir"
    print(f"Comparing {len(pairs)} pairs, spaces={spaces}, control={'on' if want_control else 'off'}")

    rows = []
    for t0, t1 in tqdm.tqdm(pairs, desc="Pairs"):
        moon0 = moon_by_exp[t0]
        moon1 = moon_by_exp[t1]
        r_eval = ROT_EVAL_RADIUS_FACTOR * moon0[2]
        stored = cross_reg[(t0, t1)]
        gamma = gamma_by_pair[(t0, t1)]
        e0 = cache.get(t0)
        e1 = cache.get(t1)
        row = {
            "t0": t0, "t1": t1, "gamma": gamma, "r_eval_px": r_eval,
            "stored": list(stored),
            "c0": source.exposure_correction(exposure_groups[t0][0]),
            "c1": source.exposure_correction(exposure_groups[t1][0]),
        }

        if want_control:
            img0 = e0["gray"].to(device)
            img1 = e1["gray"].to(device)
            got = register_stored_values(img0, img1, gamma, t0, t1, moon0, moon1, device)
            del img0, img1
            torch.cuda.empty_cache()
            d = displacement(got, stored, r_eval)
            row["control"] = {"result": list(got), "d_trans_px": d[0],
                              "d_rot_px": d[1], "d_total_px": d[2]}

        for space in spaces:
            L0 = e0["L"].to(device)
            c0 = e0["covered"].to(device)
            L1 = e1["L"].to(device)
            c1 = e1["covered"].to(device)
            x0, x1, apriori_valid = radiance_pair_images(L0, c0, L1, c1, space)
            del L0, L1, c0, c1
            torch.cuda.empty_cache()
            got = register_radiance(x0, x1, apriori_valid, moon0, moon1, device)
            del x0, x1, apriori_valid
            torch.cuda.empty_cache()
            d = displacement(got, stored, r_eval)
            row[space] = {"result": list(got), "d_trans_px": d[0],
                          "d_rot_px": d[1], "d_total_px": d[2]}

        rows.append(row)
        cache.drop_all_but({t1})

    # --- report ---------------------------------------------------------------------
    modes = (["control"] if want_control else []) + spaces
    print("\n" + "=" * 100)
    print("Displacement of the re-registered transform from the stage-2 one, in pixels.")
    print(f"d_total = |translation| + rotation arc at r={ROT_EVAL_RADIUS_FACTOR:g} moon radii.")
    print("=" * 100)
    header = f"{'t0':>9} {'t1':>9} {'gamma':>7} {'c0':>7} {'c1':>7}"
    for m in modes:
        header += f" | {m + ' dxy':>11} {m + ' drot':>11} {m + ' tot':>10}"
    print(header)
    for row in rows:
        line = (f"{row['t0']:9.5f} {row['t1']:9.5f} {row['gamma']:7.4f} "
                f"{row['c0']:7.4f} {row['c1']:7.4f}")
        for m in modes:
            r = row[m]
            line += f" | {r['d_trans_px']:11.3f} {r['d_rot_px']:11.3f} {r['d_total_px']:10.3f}"
        print(line)

    print("-" * 100)
    summary = {}
    for m in modes:
        tot = np.array([row[m]["d_total_px"] for row in rows])
        summary[m] = {
            "median_px": float(np.median(tot)),
            "max_px": float(tot.max()),
            "n_over_threshold": int((tot > SIGNIFICANT_PX).sum()),
            "n_pairs": len(rows),
        }
        print(f"{m:>8}: median {summary[m]['median_px']:.3f} px, max {summary[m]['max_px']:.3f} px, "
              f"{summary[m]['n_over_threshold']}/{len(rows)} pairs over {SIGNIFICANT_PX} px")

    print("-" * 100)
    if want_control:
        worst = summary["control"]["max_px"]
        if worst > CONTROL_TOLERANCE_PX:
            print(f"CONTROL FAILED: reproducing stage 2 on its own inputs moved by {worst:.4f} px "
                  f"(> {CONTROL_TOLERANCE_PX}). The harness disagrees with stage 2 about frames, "
                  f"moons or pair order — the radiance deltas below mean nothing until this is 0.")
        else:
            print(f"Control reproduces stage 2 to {worst:.2e} px. The radiance deltas are real.")
    else:
        print("Control skipped — these deltas are only meaningful if it has passed before.")

    for space in spaces:
        s = summary[space]
        if s["max_px"] <= SIGNIFICANT_PX:
            print(f"{space}: max {s['max_px']:.3f} px, at or below the grid search's own "
                  f"{SIGNIFICANT_PX} px resolution. Refeeding the calibration would change nothing.")
        else:
            print(f"{space}: {s['n_over_threshold']}/{s['n_pairs']} pairs move more than "
                  f"{SIGNIFICANT_PX} px (max {s['max_px']:.3f} px). Worth wiring the calibration "
                  f"back into registration.")

    with open(out_path, "w") as fd:
        json.dump({"pairs": rows, "summary": summary,
                   "config": {"space": args.space, "control": want_control,
                              "cover_thresh": COVER_THRESH,
                              "rot_eval_radius_factor": ROT_EVAL_RADIUS_FACTOR,
                              "workdir": str(workdir), "input_mode": args.input_mode}},
                  fd, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    configure_cuda_visible_devices()
    require_cuda()
    main()
