#!/usr/bin/env python3
"""Full eclipse image processing pipeline (non-interactive version of pipeline.ipynb)."""

import os
import sys
from pathlib import Path


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v2").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v2 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import torch
from eclipse_v2.device import configure_cuda_visible_devices, require_cuda
from eclipse_v2 import stage0 as s0
from eclipse_v2 import stage1 as s1
from eclipse_v2 import stage2 as s2
from eclipse_v2.inputs import make_source, attach_source
from eclipse_v2.stage3 import (
    Stage3Context,
    build_per_exposure_averages,
    crop_and_save_composite,
    fft_unsharp_and_save,
    load_inputs,
    radial_normalize_display,
    rgb_vignette_and_radial_pickle,
    warp_merge_to_composite,
)


def _parse_args():
    ap = argparse.ArgumentParser(description="Eclipse v2 pipeline")
    ap.add_argument("--input-mode", choices=["jpg", "nef", "inject"],
                    default=os.environ.get("ECLIPSE_V2_INPUT_MODE", "jpg"))
    ap.add_argument("--jpg-dir", type=Path,
                    default=Path(os.environ.get("EDA00_DATA_ROOT", "/home/slavik/e202602_eclipse/data")))
    ap.add_argument("--nef-dir", type=Path,
                    default=Path(os.environ.get("ECLIPSE_V2_NEF_DIR", "/home/slavik/tmp/eclipse_fake_imgs")))
    ap.add_argument("--workdir", type=Path,
                    default=Path(os.environ.get("ECLIPSE_V2_WORKDIR", "/home/slavik/tmp/eclipse_v2_run")))
    return ap.parse_args()


if __name__ == "__main__":
    configure_cuda_visible_devices()
    require_cuda()

    args = _parse_args()
    WORKDIR = args.workdir
    WORKDIR.mkdir(parents=True, exist_ok=True)

    source = make_source(
        args.input_mode, jpg_dir=args.jpg_dir, nef_dir=args.nef_dir
    )
    print(f"Input mode: {source.kind} (is_linear={source.is_linear})")

    PK_STAGE0 = WORKDIR / "v2-stage0.pkl"
    PK_STAGE1 = WORKDIR / "v2-stage1.pkl"
    PK_STAGE2 = WORKDIR / "v2-stage2.pkl"

    device = torch.device("cuda")

    # --- Stage 0 ---

    print("=== Stage 0: ingest, moon detection, intra-exposure registration ===")

    image_infos = s0.get_image_infos(source)
    print(f"Found {len(image_infos)} images, first: {image_infos[0].path}")

    exposure_groups = s0.group_by_exposure(image_infos)

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

    # --- Stage 3 ---

    print("\n=== Stage 3: reference merge, radial tone, FFT sharpen, RGB ===")

    ctx = Stage3Context(workdir=WORKDIR)
    load_inputs(ctx)
    attach_source(ctx.exposure_groups, source)
    build_per_exposure_averages(ctx)
    warp_merge_to_composite(ctx)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)

    print("\nPipeline complete. Outputs in", WORKDIR)
