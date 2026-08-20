#!/usr/bin/env python3
"""Finish a stage-3 run that crashed after `crop_and_save_composite` completed, but before
`radial_normalize_display`, re-cropped to a new `crop_and_save_composite` margin.

Rationale: `crop_and_save_composite`'s margin only trims a fixed number of pixels off each
edge of the mutual-coverage bounding box; that box itself does not depend on the margin. So
going from margin 24 (what actually ran and got saved to disk) to margin 32 (the new value in
stage3.py) is exactly "trim OLD_MARGIN..NEW_MARGIN off each edge of the already-saved crop" --
no need to rerun the GPU merge (`merge_to_composite`) or the ~50 min registration stages to
reflect it. `ctx.r_lo/r_hi/c_lo/c_hi` from the crashed run were never persisted anywhere and
are not read by anything downstream of `crop_and_save_composite`, so they don't need to be
reconstructed -- only `mi_crop`/`mj_crop` (the moon center in crop-local coordinates) do, and
those are re-estimated from the blanked-moon centroid of the already-saved composite (then
`find_moon`, called inside `radial_normalize_display`, refines that guess anyway).

Usage: edit OLD_MARGIN/NEW_MARGIN below to match, then run with the GPU available.
"""
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from eclipse_v7.device import configure_cuda_visible_devices, require_cuda
from eclipse_v7.merge import NO_DATA
from eclipse_v7.stage3 import (
    Stage3Context,
    composite_for_display,
    fft_unsharp_and_save,
    fill_no_data,
    load_inputs,
    radial_normalize_display,
    rgb_vignette_and_radial_pickle,
)

WORKDIR = Path("/home/slavik/tmp/eclipse_v7_my_raws_weight_by_time")
OLD_MARGIN = 24   # margin crop_and_save_composite actually used when this run's .npy files were saved
NEW_MARGIN = 32   # margin now in stage3.py's crop_and_save_composite
DELTA = NEW_MARGIN - OLD_MARGIN
assert DELTA > 0, "NEW_MARGIN must be larger than OLD_MARGIN for a pure re-trim"


def _moon_centroid(composite_old: np.ndarray) -> tuple[float, float]:
    """Centroid of the largest NO_DATA blob in the old crop -- the moon disk, not a stray gap."""
    blanked = composite_old == NO_DATA
    labels, n = ndimage.label(blanked)
    assert n >= 1, "no NO_DATA region found in the saved composite -- can't locate the moon"
    sizes = ndimage.sum(blanked, labels, index=range(1, n + 1))
    biggest = 1 + int(np.argmax(sizes))
    mi, mj = ndimage.center_of_mass(blanked, labels, biggest)
    return float(mi), float(mj)


def main():
    configure_cuda_visible_devices()
    require_cuda()

    composite_old = np.load(WORKDIR / "v7-stage3_composite.npy")
    variance_old = np.load(WORKDIR / "v7-stage3_variance.npy")
    weights_old = np.load(WORKDIR / "v7-stage3_weights.npy")
    weight_times = np.load(WORKDIR / "v7-stage3_weights_exposures.npy")

    mi_old, mj_old = _moon_centroid(composite_old)
    print(f"Old crop moon centroid (initial guess only): ({mi_old:.1f}, {mj_old:.1f})")

    sl = slice(DELTA, -DELTA)
    radiance_crop = composite_old[sl, sl].copy()
    variance_crop = variance_old[sl, sl].copy()
    exposure_weights_crop = weights_old[:, sl, sl].copy()
    no_data_crop = radiance_crop == NO_DATA
    mi_crop_guess = mi_old - DELTA
    mj_crop_guess = mj_old - DELTA
    print(f"New crop shape {radiance_crop.shape}, moon guess ({mi_crop_guess:.1f}, {mj_crop_guess:.1f})")

    ctx = Stage3Context(workdir=WORKDIR)
    load_inputs(ctx, use_refined_registration=True)   # cheap: pickle loads only

    ctx.radiance_crop = radiance_crop
    ctx.variance_crop = variance_crop
    ctx.no_data_crop = no_data_crop
    ctx.exposure_weights_crop = exposure_weights_crop
    ctx.exposure_weight_times = weight_times
    ctx.H_crop, ctx.W_crop = radiance_crop.shape
    ctx.mi_crop, ctx.mj_crop = mi_crop_guess, mj_crop_guess
    ctx.moon_r = float(ctx.moon_ref[2])
    ctx.composite_crop = composite_for_display(
        fill_no_data(radiance_crop, no_data_crop), no_data=no_data_crop
    )

    # Re-save the crop artifacts so the on-disk files match the new margin too.
    np.save(WORKDIR / "v7-stage3_composite.npy", radiance_crop)
    np.save(WORKDIR / "v7-stage3_variance.npy", variance_crop)
    np.save(WORKDIR / "v7-stage3_weights.npy", exposure_weights_crop)
    np.save(WORKDIR / "v7-stage3_weights_exposures.npy", weight_times)
    print(f"Re-saved crop .npy files at margin={NEW_MARGIN} (was {OLD_MARGIN})")

    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)
    print("Done.")


if __name__ == "__main__":
    main()
