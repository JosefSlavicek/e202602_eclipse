"""Stage 2: per-exposure averages + cross-exposure registration and gamma."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image

import eclipse_v1.stage0  # noqa: F401
from eclipse_v1.utils import (
    load_grayscale,
    apply_transform_single,
    compute_weighted_average,
    moon_median,
    grid_search_registration,
)


def make_mask_for_gamma(H, W, moon_center, moon_radius, device):
    mi, mj, r = moon_center[0], moon_center[1], moon_radius
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    dist = torch.sqrt((ii - mi) ** 2 + (jj - mj) ** 2)
    return (dist > r + 1.0).to(torch.float32)


def mae_out_of_moon(img0, img1_scaled, mask0, mask1):
    valid = (mask0 > 0.5) & (mask1 > 0.5) & (img0 >= 0.01) & (img0 <= 0.45)
    if valid.sum() == 0:
        return float("inf")
    return (torch.abs(img0 - img1_scaled).to(img0.device) * valid).sum().item() / valid.sum().item()


def estimate_gamma(img0, img1, moon0, moon1, t0, t1, device, step_min=0.001, transform=None):
    H, W = img0.shape
    mask0 = make_mask_for_gamma(H, W, (moon0[0], moon0[1]), moon0[2], device)
    mask1_img = make_mask_for_gamma(H, W, (moon1[0], moon1[1]), moon1[2], device)
    if isinstance(img0, np.ndarray):
        img0 = torch.from_numpy(img0).to(device=device, dtype=torch.float32)
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).to(device=device, dtype=torch.float32)
    if transform is not None:
        shift_i, shift_j, rotation = transform
        mask1 = apply_transform_single(mask1_img, shift_i, shift_j, rotation, device)
    else:
        mask1 = mask1_img
    scale_ratio = t0 / t1
    center = 2.5
    step = 2.0
    best_gamma = center
    while step >= step_min:
        g_vals = [center - step, center, center + step]
        g_vals = [max(0.01, g) for g in g_vals]
        maes = []
        for g in g_vals:
            factor = (scale_ratio) ** (1.0 / g)
            img1_scaled = (img1 * factor).clamp(0.0, 1.0)
            if transform is not None:
                img1_scaled = apply_transform_single(img1_scaled, shift_i, shift_j, rotation, device)
            m = mae_out_of_moon(img0, img1_scaled, mask0, mask1)
            maes.append(m)
        idx_best = int(np.argmin(maes))
        best_gamma = g_vals[idx_best]
        if idx_best == 0:
            center = g_vals[0]
        elif idx_best == 2:
            center = g_vals[2]
        else:
            center = g_vals[1]
            step = step / 2.0
    return best_gamma


def register_cross_exposure(img0, img1_scaled, moon0, moon1, gamma, t0, t1, device):
    if isinstance(img0, np.ndarray):
        img0 = torch.from_numpy(img0).to(device=device, dtype=torch.float32)
    if isinstance(img1_scaled, np.ndarray):
        img1_scaled = torch.from_numpy(img1_scaled).to(device=device, dtype=torch.float32)
    g0 = img0.clone()
    g1 = img1_scaled.clone()
    apriori_valid = (g0 * ((t1 / t0) ** (1.0 / gamma)) <= 0.9).to(torch.float32)
    initial_shift_half = 2.0 * (5.0 * (2.0 + 2.0) + 0.0 + 3.0)
    return grid_search_registration(g0, g1, moon0, moon1, initial_shift_half, device, apriori_valid=apriori_valid)


def load(stage1_pkl: Path):
    stage1_pkl = Path(stage1_pkl)
    with open(stage1_pkl, "rb") as fd:
        exposure_groups = pickle.load(fd)
        reg = pickle.load(fd)
        opt_results = pickle.load(fd)
    return exposure_groups, reg, opt_results


def moon_median_table(exposure_groups: dict) -> dict:
    moon_by_exp = {}
    for exp in exposure_groups:
        moon_by_exp[exp] = moon_median(exposure_groups[exp])
    exposure_times_sorted = sorted(exposure_groups.keys())
    print(f"Loaded {len(exposure_times_sorted)} exposure groups; moon_by_exp computed.")
    return moon_by_exp, exposure_times_sorted


def fullsize_averages(
    exposure_groups: dict,
    exposure_times_sorted: list,
    opt_results: dict,
    device: torch.device,
) -> dict:
    """Stack+mask+warp each raw frame to global coords per exposure; mean -> full-res tensor."""
    avg_images = {}
    for exp in tqdm.tqdm(exposure_times_sorted, desc="Averaged images"):
        group = exposure_groups[exp]
        if exp not in opt_results or len(group) < 2:
            continue
        abs_xy = torch.from_numpy(opt_results[exp]["abs_xy"]).to(device)
        abs_angle_t = torch.from_numpy(opt_results[exp]["abs_angle_t"]).to(device)
        avg_images[exp], _, _ = compute_weighted_average(group, abs_xy, abs_angle_t, device)

    print(f"Built {len(avg_images)} averaged images (full size).")
    return avg_images


def cross_exposure_consecutive_pairs(
    exposure_times_sorted: list,
    avg_images: dict,
    moon_by_exp: dict,
    device: torch.device,
    pair_gif_dir: Path,
) -> list:
    """For each (t0,t1) consecutive in time: estimate gamma, register longer to shorter, refine; GIF."""
    pairs_results = []
    for idx in range(len(exposure_times_sorted) - 1):
        t0 = exposure_times_sorted[idx]
        t1 = exposure_times_sorted[idx + 1]
        if t0 not in avg_images or t1 not in avg_images:
            continue
        img0 = avg_images[t0]
        img1 = avg_images[t1]
        moon0 = moon_by_exp[t0]
        moon1 = moon_by_exp[t1]

        gamma1 = estimate_gamma(img0, img1, moon0, moon1, t0, t1, device)
        scale1 = (t0 / t1) ** (1.0 / gamma1)
        img1_scaled1 = (img1 * scale1).clamp(0.0, 1.0)
        shift_i, shift_j, rotation = register_cross_exposure(img0, img1_scaled1, moon0, moon1, gamma1, t0, t1, device)

        gamma2 = estimate_gamma(img0, img1, moon0, moon1, t0, t1, device, transform=(shift_i, shift_j, rotation))
        scale2 = (t0 / t1) ** (1.0 / gamma2)
        img1_scaled2 = (img1 * scale2).clamp(0.0, 1.0)
        shift_i, shift_j, rotation = register_cross_exposure(img0, img1_scaled2, moon0, moon1, gamma2, t0, t1, device)

        pairs_results.append((t0, t1, gamma2, shift_i, shift_j, rotation))

        mi, mj, r = moon0[0], moon0[1], moon0[2]
        if t0 < 0.004 - 1e-6:
            half = 1.2 * r
        elif t0 < 0.5 - 1e-6:
            half = 3.0 * r
        else:
            half = 6.0 * r
        H, W = img0.shape
        i_lo = max(0, int(mi - half))
        i_hi = min(H, int(mi + half))
        j_lo = max(0, int(mj - half))
        j_hi = min(W, int(mj + half))

        img1_aligned = apply_transform_single(img1_scaled2, shift_i, shift_j, rotation, device)
        crop0 = (img0[i_lo:i_hi, j_lo:j_hi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        crop1 = (img1_aligned[i_lo:i_hi, j_lo:j_hi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pair_gif_dir.mkdir(parents=True, exist_ok=True)
        frame0 = Image.fromarray(np.stack([crop0, crop0, crop0], axis=-1))
        frame1 = Image.fromarray(np.stack([crop1, crop1, crop1], axis=-1))
        gif_path = pair_gif_dir / f"v1-stage2_pair_{t0:.5f}_{t1:.5f}_gamma{gamma2:.4f}.gif"
        frame0.save(gif_path, save_all=True, append_images=[frame1], duration=500, loop=0)

        print(f"t0={t0:.5f} t1={t1:.5f} gamma={gamma2:.4f} -> {gif_path}")

    print(f"Processed {len(pairs_results)} consecutive pairs.")
    return pairs_results


def save_pickle(out_pkl: Path, pairs_results: list) -> Path:
    cross_reg = {(t0, t1): (shift_i, shift_j, rotation) for (t0, t1, _, shift_i, shift_j, rotation) in pairs_results}
    gamma_by_pair = {(t0, t1): gamma for (t0, t1, gamma, _, _, _) in pairs_results}
    out_pkl = Path(out_pkl)
    with open(out_pkl, "wb") as fd:
        pickle.dump(cross_reg, fd)
        pickle.dump(gamma_by_pair, fd)
    print(f"Saved {out_pkl} (cross_reg, gamma_by_pair).")
    return out_pkl


def run(stage1_pkl: Path, out_pkl: Path, pair_gif_dir: Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage 2.")
    device = torch.device("cuda")

    exposure_groups, _reg, opt_results = load(stage1_pkl)
    moon_by_exp, exposure_times_sorted = moon_median_table(exposure_groups)
    avg_images = fullsize_averages(exposure_groups, exposure_times_sorted, opt_results, device)
    pairs_results = cross_exposure_consecutive_pairs(
        exposure_times_sorted, avg_images, moon_by_exp, device, pair_gif_dir
    )
    return save_pickle(out_pkl, pairs_results)
