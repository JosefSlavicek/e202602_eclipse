"""
Eclipse stack pipeline (v0).

Shows what each stage does (loops, merges, patch grids, where FFTs run). Heavy math
stays in eclipse_v0/*.py; this script is the control flow and parameters.

Artifacts use the v0- prefix under WORKDIR.

One-shot alternative (same math, fewer visible steps): s0.run_stage0(DATA_ROOT, PK_EDA00),
s2.run_stage2(...), s3.run_stage3(...), run_stage5(WORKDIR).
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


def v0_package_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "eclipse_v0").is_dir():
        return cwd
    if (cwd / "v0" / "eclipse_v0").is_dir():
        return cwd / "v0"
    raise RuntimeError(
        "Could not find eclipse_v0: set cwd to the `v0` directory or the repo root."
    )


def main() -> None:
    V0_ROOT = v0_package_root()
    if str(V0_ROOT) not in sys.path:
        sys.path.insert(0, str(V0_ROOT))

    # GPU selection — same defaults as legacy eda00.py. Override in the shell, e.g.
    #   CUDA_VISIBLE_DEVICES=0 python pipeline.py
    from eclipse_v0.device import configure_cuda_visible_devices, require_cuda

    configure_cuda_visible_devices()
    require_cuda()

    import torch
    import numpy as np
    from eclipse_v0 import stage0 as s0
    from eclipse_v0 import stage2 as s2
    from eclipse_v0 import stage3 as s3
    from eclipse_v0.stage5 import (
        STAGE5_FFT_INWARD_MEDIAN_SPAN,
        STAGE5_PATCH_SIDE,
        STAGE5_PATCH_STRIDE,
        STAGE5_UNSHARP_GAUSSIAN_SIGMAS,
        STAGE5_UNSHARP_WEIGHTS,
        run_stage5,
        stage5_count_fft_patch_placements,
    )

    DATA_ROOT = Path(os.environ.get("EDA00_DATA_ROOT", "/home/slavik/e202602_eclipse/data"))
    WORKDIR = Path(os.environ.get("ECLIPSE_V0_WORKDIR", "/home/slavik/tmp/eclipse_v0_run"))
    WORKDIR.mkdir(parents=True, exist_ok=True)

    PK_EDA00 = WORKDIR / "v0-eda00.pkl"
    PK_EDA02 = WORKDIR / "v0-eda02.pkl"
    PK_EDA03 = WORKDIR / "v0-eda03.pkl"

    # --- Stage 0 — ingest, moon, intra-exposure registration ---
    # 0a — Scan disk → ImageInfo list
    image_infos = s0.get_image_infos(DATA_ROOT)
    print(len(image_infos), image_infos[0].path.name)

    # 0b — Moon detection (loop over every frame)
    s0.stage0_detect_moons(image_infos)

    # 0c — Group by exposure; drop radius outliers
    exposure_groups = s0.stage0_group_by_exposure(image_infos)
    s0.stage0_prune_radius_outliers(exposure_groups)

    # 0d — Interpolate missing positions; set position uncertainty
    interp = s0.stage0_interpolate_missing_moons(image_infos, exposure_groups)
    s0.stage0_set_moon_position_std(image_infos, exposure_groups, interp)

    # 0e — Intra-exposure registration: all ordered pairs per group (slow)
    reg = s0.stage0_register_intra_exposure_pairs(exposure_groups)
    s0.stage0_save_pickle(exposure_groups, reg, PK_EDA00)

    # --- Stage 2 — prune stacks, global pose fit per exposure ---
    exposure_groups, reg = s2.stage2_load(PK_EDA00)
    s2.stage2_prune_groups(exposure_groups, reg)

    device = torch.device("cuda")
    opt_results = s2.stage2_optimize_poses_and_debug(
        exposure_groups, reg, device, debug_img_dir=WORKDIR
    )
    s2.stage2_save_pickle(PK_EDA02, exposure_groups, reg, opt_results)

    # --- Stage 3 — full-res stack mean per exposure, then cross-exposure chain ---
    exposure_groups, _reg, opt_results = s3.stage3_load(PK_EDA02)
    moon_by_exp, exposure_times_sorted = s3.stage3_moon_median_table(exposure_groups)

    avg_images = s3.stage3_fullsize_averages(
        exposure_groups, exposure_times_sorted, opt_results, device
    )

    pairs_results = s3.stage3_cross_exposure_consecutive_pairs(
        exposure_times_sorted, avg_images, moon_by_exp, device, pair_gif_dir=WORKDIR
    )
    s3.stage3_save_pickle(PK_EDA03, pairs_results)

    # --- Stage 5 — reference merge, radial tone, patch FFT sharpen, RGB ---
    with open(PK_EDA02, "rb") as fd:
        _eg = pickle.load(fd)
    _sample = next(iter(_eg.values()))[0]
    H0, W0 = _sample.height, _sample.width
    nr, nc, n_tot = stage5_count_fft_patch_placements(H0, W0)
    print(
        "FFT sharpen patch scan (on radial output, size ≈ crop — often smaller than full frame):\n"
        f"  full-frame geometry if we ran on {H0}x{W0}: grid {nr}x{nc} origins = {n_tot} patches per σ\n"
        f"  patch={STAGE5_PATCH_SIDE}px stride={STAGE5_PATCH_STRIDE}\n"
        f"  Gaussian σ for residual blur: {STAGE5_UNSHARP_GAUSSIAN_SIGMAS}\n"
        f"  combine weights: {STAGE5_UNSHARP_WEIGHTS}\n"
        f"  polar median span on |F| (library): {STAGE5_FFT_INWARD_MEDIAN_SPAN}"
    )

    run_stage5(WORKDIR)

    composite = np.load(WORKDIR / "v0-eda05_composite.npy")
    h, w = composite.shape
    nr, nc, nt = stage5_count_fft_patch_placements(h, w)
    print(f"Composite crop {h}x{w}: FFT sharpen uses {nr}x{nc} = {nt} patch starts per σ")

    # Quick look (optional) — same as notebook
    import matplotlib.pyplot as plt
    from PIL import Image

    rgb = np.asarray(Image.open(WORKDIR / "v0-eda05_rgb_rescaled.png")) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(composite, cmap="gray")
    axes[0].set_title("composite (float crop)")
    axes[1].imshow(rgb)
    axes[1].set_title("RGB export")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
