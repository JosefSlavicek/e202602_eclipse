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

    python v5/compare_registration_radiance.py --workdir /home/slavik/tmp/eclipse_v6

Reads `v5-stage1.pkl`, `v5-stage2.pkl`, `v5-calib.pkl` from the workdir. Writes nothing
except the JSON report (`--out`, default `<workdir>/v5-regcheck.json`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6 import calib as ca                                        # noqa: E402
from eclipse_v6 import stage2 as s2                                       # noqa: E402
from eclipse_v6.device import configure_cuda_visible_devices, require_cuda  # noqa: E402
from eclipse_v6.inputs import attach_source, make_source                  # noqa: E402
from eclipse_v6.reregister import (                                       # noqa: E402
    ROT_EVAL_RADIUS_FACTOR,
    average_radiance,
    displacement,
    radiance_pair_images,
    register_pair,
)
from eclipse_v6.utils import compute_weighted_average, moon_median        # noqa: E402

# --- constants ------------------------------------------------------------------------
# The radiance path itself lives in `eclipse_v6.reregister`, which is what the pipeline runs;
# this script only adds the control and the comparison so the two cannot drift apart.
SIGNIFICANT_PX = 0.1         # grid_search_registration's own resolution floor: below this it
                             # does not refine further, so a smaller delta means nothing
CONTROL_TOLERANCE_PX = 1e-3  # the control re-runs a deterministic search on identical inputs


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=Path("/home/slavik/tmp/eclipse_v6"),
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
#  Comparison                                                                 #
# --------------------------------------------------------------------------- #
def register_stored_values(img0, img1, gamma, t0, t1, moon0, moon1, device):
    """Stage 2's final registration step, verbatim, on the pickled gamma."""
    scale = (t0 / t1) ** (1.0 / gamma)
    img1_scaled = (img1 * scale).clamp(0.0, 1.0)
    return s2.register_cross_exposure(img0, img1_scaled, moon0, moon1, gamma, t0, t1, device)


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
            got = register_pair(x0, x1, apriori_valid, moon0, moon1, device)
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
