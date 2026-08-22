#!/usr/bin/env python3
"""Full eclipse image processing pipeline."""

import os
import sys
from pathlib import Path


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v8").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v8 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import math
import pickle

import numpy as np
import torch
from eclipse_v8.device import configure_cuda_visible_devices, require_cuda, seed_everything
from eclipse_v8 import calib as ca
from eclipse_v8 import darkcal as dc
from eclipse_v8 import flatcal as fc
from eclipse_v8 import rawprep as rp
from eclipse_v8 import reregister as rr
from eclipse_v8 import stage0 as s0
from eclipse_v8 import stage1 as s1
from eclipse_v8 import stage2 as s2
from eclipse_v8.inputs import make_source, attach_source
from eclipse_v8.merge import merge_to_composite
from eclipse_v8.stage3 import (
    Stage3Context,
    crop_and_save_composite,
    fft_unsharp_and_save,
    load_calibration,
    load_inputs,
    radial_normalize_display,
    rgb_vignette_and_radial_pickle,
)

SEED = 24601  # fixed, so reruns give the same result


def _parse_args():
    ap = argparse.ArgumentParser(description="Eclipse v8 pipeline")
    ap.add_argument("--nef-dir", type=Path,
                    default=Path(os.environ.get("ECLIPSE_V8_NEF_DIR", "/home/slavik/e202602_eclipse/my_raws")))
    ap.add_argument("--workdir", type=Path, default=None,
                    help="Default /home/slavik/tmp/eclipse_v8_run, or $ECLIPSE_V8_WORKDIR.")
    ap.add_argument("--exposure-group-subsample", type=int,
                    default=int(os.environ.get("ECLIPSE_V8_EXPOSURE_GROUP_SUBSAMPLE", "1")),
                    help="Keep ~1/N of the exposure groups (1=all, 2=half, ...); shortest and "
                         "longest are always kept. For faster iteration, not a free speedup.")
    ap.add_argument("--start-stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="Resume from an earlier run's saved files: 3 skips straight to "
                         "calibration, 4 to stage 3.")
    return ap.parse_args()


if __name__ == "__main__":
    configure_cuda_visible_devices()
    require_cuda()

    args = _parse_args()
    seed_everything(SEED)
    if args.workdir is not None:
        WORKDIR = args.workdir
    elif "ECLIPSE_V8_WORKDIR" in os.environ:
        WORKDIR = Path(os.environ["ECLIPSE_V8_WORKDIR"])
    else:
        WORKDIR = Path("/home/slavik/tmp/eclipse_v8_run")
    WORKDIR.mkdir(parents=True, exist_ok=True)

    PK_STAGE0 = WORKDIR / "v8-stage0.pkl"
    PK_STAGE1 = WORKDIR / "v8-stage1.pkl"
    PK_STAGE2 = WORKDIR / "v8-stage2.pkl"
    PK_STAGE2R = WORKDIR / "v8-stage2r.pkl"   # cross-exposure alignment, redone after calibration
    PK_CALIB = WORKDIR / "v8-calib.pkl"
    PK_DARK = WORKDIR / "v8-dark.pkl"
    PK_FLAT = WORKDIR / "v8-flat.pkl"
    RAWPREP_DIR = WORKDIR / "rawprep"      # per-frame overburn mask + weight, straight off the raw
    CORRECTED_DIR = WORKDIR / "corrected"  # per-frame after dark/flat correction

    source = make_source(nef_dir=args.nef_dir)
    print(f"Input mode: {source.kind}")
    print(f"Workdir:    {WORKDIR}")

    # --- Raw prep ---
    # We decode every raw frame once. From that raw decode we measure which pixels are
    # overburned (saturated) and how much weight the frame gets in the merge. Both need
    # the untouched sensor reading, so this has to happen before dark/flat correction.
    print("\n=== Raw prep: overburn mask + merge weight, from the raw decode ===")
    light_infos = source.scan()
    rp.run(light_infos, cache_dir=RAWPREP_DIR)

    # --- Dark model ---
    # Doesn't need a shutter-time correction (see darkcal.py), so it can run before we've
    # even looked at the light frames.
    print("\n=== Dark model: per-pixel bias + rate from the darks/ bracket ===")
    dark_dir = args.nef_dir / "darks"
    dark_model = dc.run(dark_dir, out_pkl=PK_DARK)

    # --- Flat field ---
    # Corrects vignetting and per-pixel sensor quirks. We only fit it if a flats/ folder exists.
    flat_dir = args.nef_dir / "flats"
    flat_model = None
    if flat_dir.is_dir():
        print("\n=== Flat model: per-pixel correction from the flats/ bracket ===")
        flat_model = fc.run(flat_dir, dark_model=dark_model, out_pkl=PK_FLAT)

    # --- Apply dark/flat ---
    print("\n=== Apply dark/flat correction, save the corrected frames ===")
    rp.apply_corrections(
        light_infos, cache_dir=RAWPREP_DIR, corrected_dir=CORRECTED_DIR,
        dark_model=dark_model, flat_model=flat_model,
    )
    source.set_cache_dirs(raw_cache_dir=RAWPREP_DIR, corrected_dir=CORRECTED_DIR)

    device = torch.device("cuda")

    # --- Stage 0 ---
    if args.start_stage <= 0:
        print("=== Stage 0: ingest, moon detection, intra-exposure registration ===")

        image_infos = s0.get_image_infos(source)
        print(f"Found {len(image_infos)} images, first: {image_infos[0].path}")

        exposure_groups = s0.group_by_exposure(image_infos)
        exposure_groups = s0.subsample_exposure_groups(exposure_groups, args.exposure_group_subsample)
        image_infos = [ii for group in exposure_groups.values() for ii in group]

        s0.detect_moons(image_infos)
        print("Exposure groups after moon detection:")
        s0.print_exposure_groups_stats(exposure_groups)

        print("Pruning failed moon estimations")
        s0.prune_moon_info_for_radius_outliers(exposure_groups)

        interp = s0.interpolate_missing_moons(image_infos, exposure_groups)
        s0.set_moon_position_std(image_infos, exposure_groups, interp)

        reg = s0.register_intra_exposure_pairs(exposure_groups)
        s0.save_pickle(exposure_groups, reg, PK_STAGE0)
        print(f"Stage 0 done → {PK_STAGE0}")

    # --- Stage 1 ---
    if args.start_stage <= 1:
        print("\n=== Stage 1: prune stacks, global pose fit per exposure ===")

        exposure_groups, reg = s1.load(PK_STAGE0)
        attach_source(exposure_groups, source)
        s1.prune_groups(exposure_groups, reg)

        opt_results = s1.optimize_poses_and_debug(
            exposure_groups, reg, device, debug_img_dir=WORKDIR
        )

        s1.save_pickle(PK_STAGE1, exposure_groups, reg, opt_results)
        print(f"Stage 1 done → {PK_STAGE1}")

    # --- Stage 2 ---
    if args.start_stage <= 2:
        print("\n=== Stage 2: full-res stack means, cross-exposure chain ===")

        exposure_groups, _reg, opt_results = s2.load(PK_STAGE1)
        attach_source(exposure_groups, source)
        moon_by_exp, exposure_times_sorted = s2.moon_median_table(exposure_groups)

        avg_images = s2.fullsize_averages(
            exposure_groups, exposure_times_sorted, opt_results, device
        )

        pairs_results = s2.cross_exposure_consecutive_pairs(
            exposure_times_sorted, avg_images, moon_by_exp, device, pair_gif_dir=WORKDIR,
            is_linear=source.is_linear,
        )

        s2.save_pickle(PK_STAGE2, pairs_results)
        print(f"Stage 2 done → {PK_STAGE2}")

    # --- Calibration + re-registration ---
    # Stage 2's per-pair exponents only ever aligned the frames -- from here on the actual
    # brightness comes from this calibration, not from them.
    if args.start_stage <= 3:
        print("\n=== Calibration: recover the camera's response curve + shutter corrections ===")

        exposure_groups, _reg, opt_results = s2.load(PK_STAGE1)
        attach_source(exposure_groups, source)
        with open(PK_STAGE2, "rb") as fd:
            cross_reg = pickle.load(fd)

        # Same choice as stage 3: drop the two shortest exposures, reference is the third-shortest.
        exposure_times_sorted = sorted(exposure_groups.keys())[2:]
        calib_result = ca.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg,
            exposure_times_sorted[0], device,
            n_exposure_refine=ca.N_EXPOSURE_REFINE, out_pkl=PK_CALIB,
        )
        print(f"Calibration done → {PK_CALIB}")

        # Stage 2 aligned the bracket before we knew the true response curve. Redo that
        # alignment on calibrated brightness, then refit the calibration on the result --
        # one round of this, deliberately, not a loop to convergence.
        print("\n=== Re-registration: redo the cross-exposure alignment using calibrated brightness ===")
        source.set_calibration(calib_result)
        cross_reg_refined, _rows = rr.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg, source, device,
            space="log", out_pkl=PK_STAGE2R,
        )

        print("\n=== Recalibration: refit the response on the refined alignment ===")
        ln_c_before = calib_result.ln_c.copy()
        calib_result = ca.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg_refined,
            exposure_times_sorted[0], device,
            n_exposure_refine=ca.N_EXPOSURE_REFINE, out_pkl=PK_CALIB,
        )
        # If this is already tiny, one round of re-registration was enough.
        d_ln_c = float(np.max(np.abs(calib_result.ln_c - ln_c_before)))
        print(f"Recalibration moved the worst shutter correction by "
              f"{math.expm1(d_ln_c) * 100:+.2f}%")
        print(f"Recalibration done → {PK_CALIB}")

    # --- Stage 3 ---
    print("\n=== Stage 3: reference merge, radial tone, FFT sharpen, RGB ===")

    ctx = Stage3Context(workdir=WORKDIR)
    load_inputs(ctx, use_refined_registration=True)
    attach_source(ctx.exposure_groups, source)
    print("Merge: per-frame physical brightness, weighted by measurement confidence")
    load_calibration(ctx, source, PK_CALIB)
    merge_to_composite(ctx, source)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)

    print("\nPipeline complete. Outputs in", WORKDIR)
