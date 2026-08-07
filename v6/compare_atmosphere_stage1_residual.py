#!/usr/bin/env python
"""v6/plan0.md round 4, step 3: compare two stage-1 runs' rigid-pose-fit residual.

Run pipeline.py twice on data_atmosphere/ -- once uncorrected (baseline), once with
--atmosphere-correction-pkl -- each with --stop-after-stage 1, then point this at the two
workdirs. Recomputes stage1.py's own loss (translation + rotation residual of the fitted
global rigid pose against every pairwise Fourier registration) directly from the saved
v5-stage1.pkl, per exposure group -- exactly the "does it help v5" question plan0.md's
round 4 asks. No CUDA needed here, this only reads pickles.

    python v6/compare_atmosphere_stage1_residual.py \\
        --baseline  /home/slavik/tmp/eclipse_v6_atmosphere_baseline \\
        --corrected /home/slavik/tmp/eclipse_v6_atmosphere_corrected
"""
from __future__ import annotations

import argparse
import itertools
import math
import pickle
from pathlib import Path

import numpy as np


def load_stage1(pkl_path):
    with open(pkl_path, "rb") as fh:
        exposure_groups = pickle.load(fh)
        reg = pickle.load(fh)
        opt_results = pickle.load(fh)
    return exposure_groups, reg, opt_results


def residual_for_exposure(exposure_time, n, reg, opt):
    """Mirrors stage1.optimize_group_poses's loss_fn, evaluated at the saved solution."""
    abs_xy = opt["abs_xy"]
    abs_angle = opt["abs_angle_t"]
    pairs = [(i, j) for i, j in itertools.permutations(range(n), 2)
            if (exposure_time, i, j) in reg]
    if not pairs:
        return None
    d_shift, d_rot = [], []
    for i, j in pairs:
        rs_i, rs_j, rrot_deg = reg[(exposure_time, i, j)]
        x_i, y_i = abs_xy[i]
        x_j, y_j = abs_xy[j]
        th_i, th_j = float(abs_angle[i]), float(abs_angle[j])
        dx, dy = x_j - x_i, y_j - y_i
        ci, si = math.cos(-th_i), math.sin(-th_i)
        impl_i = ci * dx - si * dy
        impl_j = si * dx + ci * dy
        impl_rot = th_i - th_j
        d_shift.append(math.hypot(impl_i - rs_i, impl_j - rs_j))
        d_rot.append(abs(math.degrees(impl_rot) - rrot_deg))
    return dict(n_pairs=len(pairs), shift_mean=float(np.mean(d_shift)),
               shift_max=float(np.max(d_shift)), rot_mean_deg=float(np.mean(d_rot)),
               rot_max_deg=float(np.max(d_rot)))


def summarize(pkl_path):
    exposure_groups, reg, opt_results = load_stage1(pkl_path)
    rows = {}
    for exposure_time, group in exposure_groups.items():
        n = len(group)
        if exposure_time not in opt_results or n < 2:
            continue
        r = residual_for_exposure(exposure_time, n, reg, opt_results[exposure_time])
        if r is not None:
            rows[exposure_time] = r
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path, required=True,
                    help="workdir of the run WITHOUT --atmosphere-correction-pkl")
    ap.add_argument("--corrected", type=Path, required=True,
                    help="workdir of the run WITH --atmosphere-correction-pkl")
    args = ap.parse_args()

    base = summarize(args.baseline / "v5-stage1.pkl")
    corr = summarize(args.corrected / "v5-stage1.pkl")

    print(f"{'exposure':>10} {'n_pairs':>8} | {'shift_mean':>11} {'shift_max':>10} "
         f"{'rot_mean':>9} {'rot_max':>8} | {'shift_mean':>11} {'shift_max':>10} "
         f"{'rot_mean':>9} {'rot_max':>8}")
    print(f"{'':>10} {'':>8} | {'--- baseline (px / deg) ---':^43} | "
         f"{'--- corrected (px / deg) ---':^43}")
    all_exp = sorted(set(base) | set(corr))
    for e in all_exp:
        b, c = base.get(e), corr.get(e)
        bs = (f"{b['shift_mean']:11.4f} {b['shift_max']:10.4f} "
             f"{b['rot_mean_deg']:9.5f} {b['rot_max_deg']:8.5f}") if b else " " * 43
        cs = (f"{c['shift_mean']:11.4f} {c['shift_max']:10.4f} "
             f"{c['rot_mean_deg']:9.5f} {c['rot_max_deg']:8.5f}") if c else " " * 43
        npairs = (b or c)["n_pairs"]
        print(f"{e:10.5f} {npairs:8d} | {bs} | {cs}")

    common = [e for e in all_exp if e in base and e in corr]
    if common:
        b_mean = float(np.mean([base[e]["shift_mean"] for e in common]))
        c_mean = float(np.mean([corr[e]["shift_mean"] for e in common]))
        b_max = float(np.max([base[e]["shift_max"] for e in common]))
        c_max = float(np.max([corr[e]["shift_max"] for e in common]))
        print(f"\nmean of per-group shift_mean: baseline {b_mean:.4f} px, "
             f"corrected {c_mean:.4f} px  ({'DOWN' if c_mean < b_mean else 'UP'} "
             f"{abs(c_mean - b_mean) / b_mean:.1%})")
        print(f"worst shift_max over all groups: baseline {b_max:.4f} px, "
             f"corrected {c_max:.4f} px  ({'DOWN' if c_max < b_max else 'UP'} "
             f"{abs(c_max - b_max) / b_max:.1%})")
        print("\nPlan0.md's claim is that this should go DOWN when the v6 correction "
             "removes the compression a rigid model can't represent. If it doesn't, "
             "stop and find out why before touching real 2026 data.")


if __name__ == "__main__":
    main()
