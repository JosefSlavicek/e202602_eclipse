"""Stage 2: per-exposure averages + cross-exposure registration and gamma (eda03)."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image

import eclipse_v0.stage0  # noqa: F401
from eclipse_v0.stage0 import (
    fill_bottom,
    fill_moon,
    gaussian_blur,
    image_to_polars,
    polar_to_cartesian,
    remove_lowfeq,
    transform_moon_center_batched,
)


def moon_median(group):
    centers_i = [ii.moon[0] for ii in group]
    centers_j = [ii.moon[1] for ii in group]
    radii = [ii.moon[2] for ii in group]
    return (float(np.median(centers_i)), float(np.median(centers_j)), float(np.median(radii)))


def load_grayscale(ii, device):
    with Image.open(ii.path) as img:
        arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def apply_transform_single(img, shift_i, shift_j, angle_deg, device):
    H, W = img.shape
    ci, cj = H / 2.0, W / 2.0
    angle_rad = math.radians(-angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    di = ii - ci - shift_i
    dj = jj - cj - shift_j
    i_src = di * cos_a + dj * sin_a + ci
    j_src = -di * sin_a + dj * cos_a + cj
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    grid = torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(0).squeeze(0)


def compute_weighted_average(group, abs_xy, abs_angle_t, device, epsilon=1e-6):
    n = len(group)
    sum_img = None
    sum_mask = None
    for j in range(n):
        img_j = load_grayscale(group[j], device)
        mj_i, mj_j, r_j = group[j].moon[0], group[j].moon[1], group[j].moon[2]
        H, W = img_j.shape
        ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
        jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
        dist = torch.sqrt((ii - mj_i) ** 2 + (jj - mj_j) ** 2)
        mask_j = (dist > r_j + 2.0).to(torch.float32)
        x_j = float(abs_xy[j, 0])
        y_j = float(abs_xy[j, 1])
        theta_j_deg = -math.degrees(float(abs_angle_t[j]))
        w_img = apply_transform_single(img_j, x_j, y_j, theta_j_deg, device)
        w_mask = apply_transform_single(mask_j, x_j, y_j, theta_j_deg, device)
        if sum_img is None:
            sum_img = w_img * w_mask
            sum_mask = w_mask.clone()
        else:
            sum_img = sum_img + w_img * w_mask
            sum_mask = sum_mask + w_mask
    avg_img = sum_img / (sum_mask + epsilon)
    return avg_img


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


def discrepancy_batched_fourier3_apriori(
    target,
    warped,
    moon_center_target,
    moon_radius_target,
    moon_centers_warped,
    moon_radius_warped,
    batch,
    blur_sigma,
    best_setup,
    apriori_valid,
):
    list_of_all = [(target, moon_center_target, moon_radius_target)]
    for i in range(len(moon_centers_warped)):
        list_of_all.append((warped[i], moon_centers_warped[i], moon_radius_warped))
    list_of_all_processed = []
    antiprotuberance_threshold = None
    for img, moon_center, moon_radius in list_of_all:
        radius_max = min(
            moon_center[0],
            img.shape[0] - moon_center[0],
            moon_center[1],
            img.shape[1] - moon_center[1],
        )
        assert 32 < moon_radius < radius_max - 32
        polar_img, _ = image_to_polars(img, moon_center, moon_radius, radius_max)
        if antiprotuberance_threshold is None:
            maxidx = polar_img.sum(dim=1).argmax()
            maxrow = polar_img[maxidx, :]
            maxrow = maxrow[maxrow > 0]
            antiprotuberance_threshold = maxrow.quantile(0.9)
        polar_img[polar_img > antiprotuberance_threshold] = antiprotuberance_threshold
        polar_img = fill_bottom(polar_img, 4)
        polar_img = remove_lowfeq(polar_img, 16)
        img = polar_to_cartesian(polar_img, moon_center, moon_radius, radius_max, (img.shape[0], img.shape[1]))
        if blur_sigma > 0:
            img = gaussian_blur(img, blur_sigma)
        mask = torch.ones_like(polar_img)
        mask = polar_to_cartesian(mask, moon_center, moon_radius, radius_max, (img.shape[0], img.shape[1]))
        mask = fill_moon(mask, moon_center, moon_radius, 0.0)
        list_of_all_processed.append((img, mask))
    target_img, target_mask = list_of_all_processed.pop(0)
    target_mask = target_mask * apriori_valid.to(target_mask.device)
    if best_setup is not None:
        bs_shift_i, bs_shift_j, bs_angle, bs_img, bs_mask = best_setup
        batch.append((bs_shift_i, bs_shift_j, bs_angle))
        list_of_all_processed.append((bs_img, bs_mask))
    assert len(list_of_all_processed) == len(batch)
    while len(list_of_all_processed) >= 2:
        shift_i_0, shift_j_0, angle_0 = batch.pop()
        shift_i_1, shift_j_1, angle_1 = batch.pop()
        warped_img_0, warped_mask_0 = list_of_all_processed.pop()
        warped_img_1, warped_mask_1 = list_of_all_processed.pop()
        mask = target_mask * warped_mask_0 * warped_mask_1
        diff_0 = ((torch.abs(target_img - warped_img_0) * mask).sum() / (mask.sum() + 1e-9)).item()
        diff_1 = ((torch.abs(target_img - warped_img_1) * mask).sum() / (mask.sum() + 1e-9)).item()
        if diff_0 < diff_1:
            batch.append((shift_i_0, shift_j_0, angle_0))
            list_of_all_processed.append((warped_img_0, warped_mask_0))
        else:
            batch.append((shift_i_1, shift_j_1, angle_1))
            list_of_all_processed.append((warped_img_1, warped_mask_1))
    shift_i, shift_j, angle = batch.pop()
    warped_img, warped_mask = list_of_all_processed.pop()
    return shift_i, shift_j, angle, warped_img, warped_mask


def register_cross_exposure(img0, img1_scaled, moon0, moon1, gamma, t0, t1, device):
    H, W = img0.shape
    if isinstance(img0, np.ndarray):
        img0 = torch.from_numpy(img0).to(device=device, dtype=torch.float32)
    if isinstance(img1_scaled, np.ndarray):
        img1_scaled = torch.from_numpy(img1_scaled).to(device=device, dtype=torch.float32)
    g0 = img0.clone()
    g1 = img1_scaled.clone()
    r0, r1 = moon0[2], moon1[2]
    apriori_valid = (g0 * ((t1 / t0) ** (1.0 / gamma)) <= 0.9).to(torch.float32)
    initial_shift_half = 2.0 * (5.0 * (2.0 + 2.0) + 0.0 + 3.0)
    ci, cj = H / 2.0, W / 2.0
    r_border = max(H, W)
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W).unsqueeze(0)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W).unsqueeze(0)

    def apply_transform_batched(img, shift_i_t, shift_j_t, cos_a_t, sin_a_t):
        di = ii - ci - shift_i_t
        dj = jj - cj - shift_j_t
        i_src = di * cos_a_t + dj * sin_a_t + ci
        j_src = -di * sin_a_t + dj * cos_a_t + cj
        j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
        i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
        grid = torch.stack([j_norm, i_norm], dim=-1)
        img_4d = img.unsqueeze(0).unsqueeze(1).expand(grid.shape[0], 1, H, W)
        return torch.nn.functional.grid_sample(
            img_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(1)

    best_shift_i, best_shift_j, best_angle = 0.0, 0.0, 0.0
    step_shift = initial_shift_half / 2.0
    step_angle = 5.0
    refine_shift = refine_angle = True
    best_setup = None
    while refine_shift or refine_angle:
        if step_shift < 0.1:
            refine_shift = False
        step_angle_in_px = step_angle * r_border * (math.pi / 180.0)
        if step_angle_in_px < 0.1:
            refine_angle = False
        blur_sigma = 0.0 if (step_shift < 2 and step_angle_in_px < 2) else min(8, max(step_angle_in_px, step_shift))
        shift_i_vals = [best_shift_i] if not refine_shift else [best_shift_i + step_shift * (k - 2) for k in range(5)]
        shift_j_vals = [best_shift_j] if not refine_shift else [best_shift_j + step_shift * (k - 2) for k in range(5)]
        angle_vals = [best_angle] if not refine_angle else [best_angle + step_angle * (k - 2) for k in range(5)]
        triples = [(si, sj, a) for si in shift_i_vals for sj in shift_j_vals for a in angle_vals]
        GRID_BATCH_SIZE = 8
        for start in range(0, len(triples), GRID_BATCH_SIZE):
            batch = triples[start : start + GRID_BATCH_SIZE]
            N = len(batch)
            shift_i_t = torch.tensor([t[0] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            shift_j_t = torch.tensor([t[1] for t in batch], device=device, dtype=torch.float32).view(N, 1, 1)
            cos_a_t = torch.cos(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            sin_a_t = torch.sin(
                torch.tensor([math.radians(-t[2]) for t in batch], device=device, dtype=torch.float32)
            ).view(N, 1, 1)
            warped = apply_transform_batched(g1.clone(), shift_i_t, shift_j_t, cos_a_t, sin_a_t)
            moon_centers_warped = transform_moon_center_batched(
                moon1[0], moon1[1], ci, cj, shift_i_t, shift_j_t, cos_a_t, sin_a_t
            )
            best_setup = discrepancy_batched_fourier3_apriori(
                g0.clone(),
                warped,
                moon0[:2],
                r0,
                moon_centers_warped,
                r1,
                list(batch),
                blur_sigma,
                best_setup,
                apriori_valid,
            )
        best_shift_i, best_shift_j, best_angle, _, _ = best_setup
        is_corner = len(shift_i_vals) > 2 and (
            best_shift_i in [shift_i_vals[0], shift_i_vals[-1]] or best_shift_j in [shift_j_vals[0], shift_j_vals[-1]]
        )
        step_shift = step_shift / 2.0 if (refine_shift and not is_corner) else step_shift
        is_corner = len(angle_vals) > 2 and best_angle in [angle_vals[0], angle_vals[-1]]
        step_angle = step_angle / 2.0 if (refine_angle and not is_corner) else step_angle
    return (float(best_shift_i), float(best_shift_j), float(best_angle))


def stage2_load(eda02_pkl: Path):
    eda02_pkl = Path(eda02_pkl)
    with open(eda02_pkl, "rb") as fd:
        exposure_groups = pickle.load(fd)
        reg = pickle.load(fd)
        opt_results = pickle.load(fd)
    return exposure_groups, reg, opt_results


def stage2_moon_median_table(exposure_groups: dict) -> dict:
    moon_by_exp = {}
    for exp in exposure_groups:
        moon_by_exp[exp] = moon_median(exposure_groups[exp])
    exposure_times_sorted = sorted(exposure_groups.keys())
    print(f"Loaded {len(exposure_times_sorted)} exposure groups; moon_by_exp computed.")
    return moon_by_exp, exposure_times_sorted


def stage2_fullsize_averages(
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
        avg_images[exp] = compute_weighted_average(group, abs_xy, abs_angle_t, device)

    print(f"Built {len(avg_images)} averaged images (full size).")
    return avg_images


def stage2_cross_exposure_consecutive_pairs(
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
        gif_path = pair_gif_dir / f"v0-eda03_pair_{t0:.5f}_{t1:.5f}_gamma{gamma2:.4f}.gif"
        frame0.save(gif_path, save_all=True, append_images=[frame1], duration=500, loop=0)

        print(f"t0={t0:.5f} t1={t1:.5f} gamma={gamma2:.4f} -> {gif_path}")

    print(f"Processed {len(pairs_results)} consecutive pairs.")
    return pairs_results


def stage2_save_pickle(out_pkl: Path, pairs_results: list) -> Path:
    cross_reg = {(t0, t1): (shift_i, shift_j, rotation) for (t0, t1, _, shift_i, shift_j, rotation) in pairs_results}
    gamma_by_pair = {(t0, t1): gamma for (t0, t1, gamma, _, _, _) in pairs_results}
    out_pkl = Path(out_pkl)
    with open(out_pkl, "wb") as fd:
        pickle.dump(cross_reg, fd)
        pickle.dump(gamma_by_pair, fd)
    print(f"Saved {out_pkl} (cross_reg, gamma_by_pair).")
    return out_pkl


def run_stage2(eda02_pkl: Path, out_pkl: Path, pair_gif_dir: Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage 2.")
    device = torch.device("cuda")

    exposure_groups, _reg, opt_results = stage2_load(eda02_pkl)
    moon_by_exp, exposure_times_sorted = stage2_moon_median_table(exposure_groups)
    avg_images = stage2_fullsize_averages(exposure_groups, exposure_times_sorted, opt_results, device)
    pairs_results = stage2_cross_exposure_consecutive_pairs(
        exposure_times_sorted, avg_images, moon_by_exp, device, pair_gif_dir
    )
    return stage2_save_pickle(out_pkl, pairs_results)
