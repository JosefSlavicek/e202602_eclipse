#!/usr/bin/env python3
"""Full eclipse image processing pipeline (non-interactive version of pipeline.ipynb)."""

import os
import sys
from pathlib import Path


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v1").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v1 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from eclipse_v1.device import configure_cuda_visible_devices, require_cuda
from eclipse_v1 import stage0 as s0
from eclipse_v1 import stage1 as s1
from eclipse_v1 import stage2 as s2
from eclipse_v1.stage3 import (
    Stage3Context,
    stage3_build_per_exposure_averages,
    stage3_crop_and_save_composite,
    stage3_fft_unsharp_and_save,
    stage3_load_inputs,
    stage3_radial_normalize_display,
    stage3_rgb_vignette_and_radial_pickle,
    stage3_warp_merge_to_composite,
)

configure_cuda_visible_devices()
require_cuda()

DATA_ROOT = Path(os.environ.get("EDA00_DATA_ROOT", "/home/slavik/e202602_eclipse/data"))
WORKDIR = Path(os.environ.get("ECLIPSE_V1_WORKDIR", "/home/slavik/tmp/eclipse_v1_run"))
WORKDIR.mkdir(parents=True, exist_ok=True)

PK_EDA00 = WORKDIR / "v1-eda00.pkl"
PK_EDA02 = WORKDIR / "v1-eda02.pkl"
PK_EDA03 = WORKDIR / "v1-eda03.pkl"

device = torch.device("cuda")


# --- Stage 0 ---

print("=== Stage 0: ingest, moon detection, intra-exposure registration ===")

image_infos = s0.get_image_infos(DATA_ROOT)
print(f"Found {len(image_infos)} images, first: {image_infos[0].path}")

exposure_groups = s0.stage0_group_by_exposure(image_infos)

s0.stage0_detect_moons(image_infos)
print("Exposure groups after moon detection:")
s0.stage0_print_exposure_groups_stats(exposure_groups)

print("Pruning failed moon estimations")
s0.stage0_prune_moon_info_for_radius_outliers(exposure_groups)

interp = s0.stage0_interpolate_missing_moons(image_infos, exposure_groups)
s0.stage0_set_moon_position_std(image_infos, exposure_groups, interp)

reg = s0.stage0_register_intra_exposure_pairs(exposure_groups)
s0.stage0_save_pickle(exposure_groups, reg, PK_EDA00)
print(f"Stage 0 done → {PK_EDA00}")


# --- Stage 1 ---

print("\n=== Stage 1: prune stacks, global pose fit per exposure ===")

exposure_groups, reg = s1.stage1_load(PK_EDA00)
s1.stage1_prune_groups(exposure_groups, reg)

opt_results = s1.stage1_optimize_poses_and_debug(
    exposure_groups, reg, device, debug_img_dir=WORKDIR
)

s1.stage1_save_pickle(PK_EDA02, exposure_groups, reg, opt_results)
print(f"Stage 1 done → {PK_EDA02}")


# --- Stage 2 ---

print("\n=== Stage 2: full-res stack means, cross-exposure chain ===")

exposure_groups, _reg, opt_results = s2.stage2_load(PK_EDA02)
moon_by_exp, exposure_times_sorted = s2.stage2_moon_median_table(exposure_groups)

avg_images = s2.stage2_fullsize_averages(
    exposure_groups, exposure_times_sorted, opt_results, device
)

pairs_results = s2.stage2_cross_exposure_consecutive_pairs(
    exposure_times_sorted, avg_images, moon_by_exp, device, pair_gif_dir=WORKDIR
)

s2.stage2_save_pickle(PK_EDA03, pairs_results)
print(f"Stage 2 done → {PK_EDA03}")


# --- Stage 3 ---

print("\n=== Stage 3: reference merge, radial tone, FFT sharpen, RGB ===")

ctx = Stage3Context(workdir=WORKDIR)
stage3_load_inputs(ctx)
stage3_build_per_exposure_averages(ctx)
stage3_warp_merge_to_composite(ctx)
stage3_crop_and_save_composite(ctx)
stage3_radial_normalize_display(ctx)
stage3_fft_unsharp_and_save(ctx)
stage3_rgb_vignette_and_radial_pickle(ctx)

print("\nPipeline complete. Outputs in", WORKDIR)
