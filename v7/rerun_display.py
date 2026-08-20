#!/usr/bin/env python3
"""Re-run the display chain from a previous run's saved composite, into a fresh output dir.

    python v7/rerun_display.py --in-dir /home/slavik/tmp/eclipse_v7_rere \
                               --out-dir /home/slavik/tmp/eclipse_v7_limbgrow

`--in-dir` is read-only; every artifact goes to `--out-dir`, so the previous run's PNGs are never
clobbered and the two can be diffed afterwards.

The cut is `v7-stage3_composite.npy` — the output of `crop_and_save_composite`, physical brightness
with `NO_DATA` still in the moon. Everything before it (ingest, moon detection, registration,
calibration, re-registration, the merge) is skipped, along with its need for frames, `v7-calib.pkl`
and the stage pickles. What runs is exactly:

    radial_normalize_display  ->  fft_unsharp_and_save  ->  rgb_vignette_and_radial_pickle

Use `pipeline.py --start-stage 4` instead when the composite itself has to be rebuilt (that redoes
the merge, which needs the frames and the calibration).

One deviation from the in-line pipeline: `crop_and_save_composite` seeds `find_moon` with the
reference exposure's median moon shifted by the crop offset, and neither the moon table nor the
crop bounds are saved. The seed here is the centroid of the blanked region — the moon itself, so a
closer seed than the original — or `--seed-moon i,j` to pin it. `find_moon` refines it three times
either way. Its triplet sampling is unseeded in the pipeline; `--seed` makes reruns reproducible.

`--no-grow` reproduces the pre-change behaviour (circle as fitted, `display` not blanked), so
running twice into two output dirs gives a before/after pair. The script prints how many pixels
that baseline is off by, since it is exact only when the fitted circle stays inside the blanked
region — normally it does, and the printed count is then 0.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v7").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v7 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from eclipse_v7 import stage3 as S3
from eclipse_v7.device import configure_cuda_visible_devices, require_cuda
from eclipse_v7.merge import NO_DATA


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="previous run's output dir, holding v7-stage3_composite.npy (read-only)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="this run's output dir; created if missing, must differ from --in-dir")
    ap.add_argument("--no-grow", action="store_true",
                    help="pre-change baseline: do not grow the moon circle over the blanked "
                         "region and do not blank `display` before the sharpen")
    ap.add_argument("--stride", type=int, default=None,
                    help=f"patch stride for the FFT sharpen; production is {S3.PATCH_STRIDE}. "
                         f"Cost scales as 1/stride^2 — 16 or 32 for a quick look.")
    ap.add_argument("--patch-side", type=int, default=None,
                    help=f"FFT patch side; production is {S3.PATCH_SIDE}")
    ap.add_argument("--seed-moon", type=str, default=None,
                    help="override the find_moon seed as 'i,j' in composite-crop coordinates")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds find_moon's triplet sampling so reruns are reproducible")
    ap.add_argument("--copy-composite", action="store_true",
                    help="also copy the composite (and variance) .npy into --out-dir, making it "
                         "a self-contained run dir at the cost of ~190 MB")
    return ap.parse_args()


def load_composite_into_ctx(ctx: S3.Stage3Context, in_dir: Path, seed_moon=None) -> None:
    """Rebuild the state `crop_and_save_composite` would have left on `ctx`.

    Sets what the display chain reads: `radiance_crop` (as saved, `NO_DATA` intact),
    `no_data_crop`, `composite_crop` (NO_DATA filled to 0 and display-normalised, exactly as the
    original hand-off did), the crop shape, and the moon seed.
    """
    path = in_dir / "v7-stage3_composite.npy"
    assert path.exists(), f"{path} not found — --in-dir must be a run that completed stage 3"
    radiance = np.load(path)
    assert radiance.ndim == 2, radiance.shape
    ctx.radiance_crop = radiance
    ctx.H_crop, ctx.W_crop = radiance.shape

    no_data = radiance <= NO_DATA
    ctx.no_data_crop = no_data if no_data.any() else None
    n_nd = int(no_data.sum())
    if ctx.no_data_crop is None:
        print(f"NOTE: {path} has no NO_DATA pixels — a --merge legacy composite. There is no "
              f"blanked region, so the moon circle cannot be grown over one.")

    var_path = in_dir / "v7-stage3_variance.npy"
    if var_path.exists():
        ctx.variance_crop = np.load(var_path)

    # Same two calls, same order, as the hand-off in crop_and_save_composite.
    composite_crop = S3.fill_no_data(radiance, ctx.no_data_crop)
    ctx.composite_crop = S3.composite_for_display(composite_crop, no_data=ctx.no_data_crop)

    if seed_moon is not None:
        mi, mj = seed_moon
        origin = "--seed-moon"
    else:
        assert ctx.no_data_crop is not None, (
            "no blanked region to take a moon seed from; pass --seed-moon i,j"
        )
        ii, jj = np.nonzero(no_data)
        mi, mj = float(ii.mean()), float(jj.mean())
        origin = "centroid of the blanked region"
    ctx.mi_crop, ctx.mj_crop = float(mi), float(mj)
    print(f"Loaded {path}: {radiance.shape}, {n_nd:,} blanked px"
          + (f", variance from {var_path.name}" if ctx.variance_crop is not None else ""))
    print(f"find_moon seed ({mi:.2f}, {mj:.2f}) from {origin}")


def _report_baseline_fidelity(ctx: S3.Stage3Context, no_data) -> None:
    """With --no-grow, `display` is still blanked over the fitted disk; say how much that differs.

    The pre-change code blanked `display` not at all, but the tone map and the p3 stretch send
    every zero of the composite to zero anyway, so the two agree wherever the fitted disk lies
    inside the blanked region. Only `fitted disk AND measured` differs, and that is usually empty.
    """
    if no_data is None or ctx.moon_mask is None:
        return
    n = int(np.sum(ctx.moon_mask & ~no_data))
    if n == 0:
        print("--no-grow baseline is exact: the fitted disk lies inside the blanked region")
    else:
        print(f"--no-grow baseline differs from the pre-change code on {n:,} px "
              f"(measured corona inside the fitted disk, blanked here, kept there)")


if __name__ == "__main__":
    configure_cuda_visible_devices()
    require_cuda()
    args = _parse_args()

    in_dir, out_dir = args.in_dir.resolve(), args.out_dir.resolve()
    assert in_dir != out_dir, "--out-dir must differ from --in-dir, or the rerun overwrites it"
    assert in_dir.is_dir(), f"{in_dir} is not a directory"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stride is not None:
        print(f"PATCH_STRIDE {S3.PATCH_STRIDE} -> {args.stride}")
        S3.PATCH_STRIDE = args.stride
    if args.patch_side is not None:
        print(f"PATCH_SIDE {S3.PATCH_SIDE} -> {args.patch_side}")
        S3.PATCH_SIDE = args.patch_side

    print(f"In:  {in_dir} (read-only)")
    print(f"Out: {out_dir}")

    ctx = S3.Stage3Context(workdir=out_dir, device=torch.device("cuda"))
    seed_moon = None
    if args.seed_moon is not None:
        si, sj = args.seed_moon.split(",")
        seed_moon = (float(si), float(sj))
    load_composite_into_ctx(ctx, in_dir, seed_moon)

    no_data_saved = ctx.no_data_crop
    if args.no_grow:
        # Hiding the mask is what disables both halves of the change: no growth, and the
        # containment assert has nothing to check. The blanking that remains uses the fitted
        # radius; _report_baseline_fidelity quantifies what that costs.
        print("Baseline mode: --no-grow (moon circle as fitted, display not grown)")
        ctx.no_data_crop = None

    print("\n=== Radial normalize ===")
    S3.radial_normalize_display(ctx)
    print(f"moon centre ({ctx.mi_crop:.2f}, {ctx.mj_crop:.2f}), fitted r {ctx.moon_r_fitted:.2f}, "
          f"r in use {ctx.moon_r:.2f}, disk {int(ctx.moon_mask.sum()):,} px")
    if args.no_grow:
        _report_baseline_fidelity(ctx, no_data_saved)
    elif no_data_saved is not None:
        ii = np.arange(ctx.H_crop, dtype=np.float32).reshape(-1, 1)
        jj = np.arange(ctx.W_crop, dtype=np.float32).reshape(1, -1)
        mask_fit = ((ii - ctx.mi_crop) ** 2 + (jj - ctx.mj_crop) ** 2) <= (ctx.moon_r_fitted**2)
        print(f"growth covered {int(np.sum(no_data_saved & ~mask_fit)):,} blanked px the fitted "
              f"circle left outside it (unprotected step edges); gave up "
              f"{int(np.sum(ctx.moon_mask & ~no_data_saved)):,} px of measured corona")

    print("\n=== FFT unsharp ===")
    n_patches = (
        len(range(0, ctx.H_crop - S3.PATCH_SIDE + 1, S3.PATCH_STRIDE))
        * len(range(0, ctx.W_crop - S3.PATCH_SIDE + 1, S3.PATCH_STRIDE))
    )
    print(f"{n_patches:,} patch positions x {len(S3.UNSHARP_GAUSSIAN_SIGMAS)} sigmas "
          f"= {n_patches * len(S3.UNSHARP_GAUSSIAN_SIGMAS):,} patch FFTs (stride {S3.PATCH_STRIDE})")
    S3.fft_unsharp_and_save(ctx)

    print("\n=== RGB + vignette ===")
    S3.rgb_vignette_and_radial_pickle(ctx)

    if args.copy_composite:
        for name in ("v7-stage3_composite.npy", "v7-stage3_variance.npy"):
            if (in_dir / name).exists():
                shutil.copy2(in_dir / name, out_dir / name)
                print(f"Copied {name}")

    meta = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "grow_moon_over_no_data": not args.no_grow,
        "patch_side": S3.PATCH_SIDE,
        "patch_stride": S3.PATCH_STRIDE,
        "seed": args.seed,
        "moon_centre_ij": [ctx.mi_crop, ctx.mj_crop],
        "moon_r_fitted": ctx.moon_r_fitted,
        "moon_r_used": ctx.moon_r,
    }
    with open(out_dir / "v7-rerun-display.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nDone. Outputs in {out_dir} (provenance in v7-rerun-display.json)")
