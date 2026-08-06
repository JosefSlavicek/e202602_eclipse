#!/usr/bin/env python3
"""Full eclipse image processing pipeline (non-interactive version of pipeline.ipynb)."""

import os
import sys
from pathlib import Path


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v5").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v5 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
import pickle

import numpy as np
import torch
from eclipse_v5.device import configure_cuda_visible_devices, require_cuda
from eclipse_v5 import calib as ca
from eclipse_v5 import reregister as rr
from eclipse_v5 import stage0 as s0
from eclipse_v5 import stage1 as s1
from eclipse_v5 import stage2 as s2
from eclipse_v5.inputs import make_source, attach_source
from eclipse_v5.merge import merge_to_composite
from eclipse_v5.stage3 import (
    Stage3Context,
    build_per_exposure_averages,
    crop_and_save_composite,
    fft_unsharp_and_save,
    load_calibration,
    load_inputs,
    radial_normalize_display,
    rgb_vignette_and_radial_pickle,
    warp_merge_to_composite,
)


def _parse_args():
    ap = argparse.ArgumentParser(description="Eclipse v5 pipeline")
    ap.add_argument("--input-mode", choices=["jpg", "atmosphere", "nef", "inject"],
                    default=os.environ.get("ECLIPSE_V5_INPUT_MODE", "jpg"))
    ap.add_argument("--jpg-dir", type=Path,
                    default=Path(os.environ.get("EDA00_DATA_ROOT", "/home/slavik/e202602_eclipse/data")))
    ap.add_argument("--atmosphere-dir", type=Path,
                    default=Path(os.environ.get("ECLIPSE_V5_ATMOSPHERE_DIR",
                                                "/home/slavik/e202602_eclipse/data_atmosphere")),
                    help="For --input-mode atmosphere: the data_atmosphere/ directory "
                         "built by make_data_atmosphere.py (must contain atmosphere.json).")
    ap.add_argument("--nef-dir", type=Path,
                    default=Path(os.environ.get("ECLIPSE_V5_NEF_DIR", "/home/slavik/tmp/eclipse_fake_imgs")))
    ap.add_argument("--workdir", type=Path, default=None,
                    help="Default /home/slavik/tmp/eclipse_v5_run, or $ECLIPSE_V5_WORKDIR. "
                         "In --input-mode atmosphere the default gains an _atmosphere "
                         "suffix so a comparison run cannot clobber the baseline; pass "
                         "--workdir or set the env var to override.")
    ap.add_argument("--exposure-group-subsample", type=int,
                    default=int(os.environ.get("ECLIPSE_V5_EXPOSURE_GROUP_SUBSAMPLE", "1")),
                    help="Keep ~1/N of the exposure groups (1=all, 2=~half, 3=~third, ...); "
                         "shortest+longest always kept. Runtime knob for faster iteration, "
                         "NOT a same-result speedup (widens cross-exposure gaps).")
    ap.add_argument("--merge", choices=["radiometric", "legacy"], default="radiometric",
                    help="radiometric: convert every frame to physical brightness through the "
                         "fitted response and combine by 1/sigma^2. legacy: v2's per-pair "
                         "exponent chain and bell weight, kept so the two can be compared "
                         "without a checkout.")
    ap.add_argument("--n-exposure-refine", type=int, default=ca.N_EXPOSURE_REFINE,
                    help="Rounds of alternation refining the shutter-time corrections. "
                         "0 is the honest nominal-times baseline.")
    ap.add_argument("--refine-registration", choices=["on", "off"], default="on",
                    help="After the first calibration, redo the cross-exposure registration "
                         "on calibrated radiance and refit the calibration on the result "
                         "(one alternation step). off reproduces stage 2's gamma-scaled "
                         "alignment. Ignored with --merge legacy, which has no calibration.")
    ap.add_argument("--refine-space", choices=["log", "linear"], default="log",
                    help="Image space for the refined registration. log makes the objective "
                         "invariant to the exposure ratio; linear only fixes the scale.")
    ap.add_argument("--start-stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help="Resume from an existing run's pickles: 3 skips straight to the "
                         "calibration (registration is ~50 min), 4 to stage 3.")
    return ap.parse_args()


if __name__ == "__main__":
    configure_cuda_visible_devices()
    require_cuda()

    args = _parse_args()
    if args.workdir is not None:
        WORKDIR = args.workdir
    elif "ECLIPSE_V5_WORKDIR" in os.environ:
        WORKDIR = Path(os.environ["ECLIPSE_V5_WORKDIR"])
    else:
        WORKDIR = Path("/home/slavik/tmp/eclipse_v5_run"
                       + ("_atmosphere" if args.input_mode == "atmosphere" else ""))
    WORKDIR.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args.input_mode, jpg_dir=args.jpg_dir, nef_dir=args.nef_dir,
        atmosphere_dir=args.atmosphere_dir,
    )
    print(f"Input mode: {source.kind} (is_linear={source.is_linear})")
    print(f"Workdir:    {WORKDIR}")
    if source.kind == "atmosphere":
        # Log and archive what this run consumed: the frames look like mode-1 JPGs,
        # so without this the outputs would be indistinguishable from a baseline run.
        print(f"Atmosphere: {source.data_root}")
        print(source.describe())
        with open(WORKDIR / "v5-input-atmosphere.json", "w") as fh:
            json.dump(source.meta, fh, indent=2)

    PK_STAGE0 = WORKDIR / "v5-stage0.pkl"
    PK_STAGE1 = WORKDIR / "v5-stage1.pkl"
    PK_STAGE2 = WORKDIR / "v5-stage2.pkl"
    PK_STAGE2R = WORKDIR / "v5-stage2r.pkl"   # cross_reg redone on calibrated radiance
    PK_CALIB = WORKDIR / "v5-calib.pkl"

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

    # --- Calibration ---
    #
    # Between stage 2 and stage 3: it needs registration (corresponding pixels across
    # exposures), and the merge needs it. Stage 2's per-pair exponents are used for
    # ALIGNMENT ONLY from here on — Fourier alignment cares about structure, not absolute
    # scale — and no longer determine any brightness. With --refine-registration on (the
    # default) they stop determining even the alignment: the block below redoes it on
    # calibrated radiance, leaving the exponents as nothing but a bootstrap for this fit.

    if args.start_stage <= 3 and args.merge != "legacy":
        print("\n=== Calibration: recover the response curve and the shutter corrections ===")

        exposure_groups, _reg, opt_results = s2.load(PK_STAGE1)
        attach_source(exposure_groups, source)
        with open(PK_STAGE2, "rb") as fd:
            cross_reg = pickle.load(fd)

        # Same exposure selection as stage 3: the two shortest are dropped (inherited from
        # v2, kept for comparability), so the reference is the third-shortest.
        exposure_times_sorted = sorted(exposure_groups.keys())[2:]
        calib_result = ca.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg,
            exposure_times_sorted[0], device,
            n_exposure_refine=args.n_exposure_refine, out_pkl=PK_CALIB,
        )
        print(f"Calibration done → {PK_CALIB}")

    # --- Re-registration + recalibration (one alternation step) ---
    #
    # Stage 2 aligned the bracket before anything was calibrated, using a fitted exponent per
    # pair and a saturation mask derived from it. Now that the response curve and the shutter
    # corrections are known, redo that alignment on physical radiance — and then refit the
    # calibration, which consumes `cross_reg` to find corresponding pixels, so that the two
    # agree. One step only: see `--refine-registration off` to skip it entirely.

    if args.start_stage <= 3 and args.merge != "legacy" and args.refine_registration == "on":
        print("\n=== Re-registration: redo the cross-exposure alignment on calibrated radiance ===")

        source.set_calibration(calib_result)
        cross_reg_refined, _rows = rr.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg, source, device,
            space=args.refine_space, out_pkl=PK_STAGE2R,
        )

        print("\n=== Recalibration: refit the response on the refined alignment ===")
        ln_c_before = calib_result.ln_c.copy()
        calib_result = ca.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg_refined,
            exposure_times_sorted[0], device,
            n_exposure_refine=args.n_exposure_refine, out_pkl=PK_CALIB,
        )
        # How much the alternation actually changed: if this is already tiny, one step was
        # enough and a second would buy nothing.
        d_ln_c = float(np.max(np.abs(calib_result.ln_c - ln_c_before)))
        print(f"Alternation: max |change in ln c| = {d_ln_c:.5f} "
              f"({math.expm1(d_ln_c) * 100:+.2f}% on the worst shutter correction)")
        print(f"Recalibration done → {PK_CALIB}")

    # --- Stage 3 ---

    print("\n=== Stage 3: reference merge, radial tone, FFT sharpen, RGB ===")

    ctx = Stage3Context(workdir=WORKDIR)
    load_inputs(
        ctx,
        use_refined_registration=(args.merge != "legacy" and args.refine_registration == "on"),
    )
    attach_source(ctx.exposure_groups, source)
    if args.merge == "legacy":
        print("Merge: LEGACY (v2 per-pair exponent chain + bell weight on encoded values)")
        build_per_exposure_averages(ctx)
        warp_merge_to_composite(ctx)
    else:
        print("Merge: radiometric (per-frame physical brightness, inverse-variance weights)")
        load_calibration(ctx, source, PK_CALIB)
        merge_to_composite(ctx, source)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)

    print("\nPipeline complete. Outputs in", WORKDIR)
