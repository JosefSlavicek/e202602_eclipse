"""Stage 1: brightness + triplet pruning, per-exposure pose optimization (eda02)."""

from __future__ import annotations

import itertools
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import tqdm

import eclipse_v0.stage0  # noqa: F401 — pickle loads ImageInfo from this module

from PIL import Image


def compose_transforms(s1_i, s1_j, rot1_deg, s2_i, s2_j, rot2_deg):
    """Compose transform (s1, rot1) then (s2, rot2). Returns (s_i, s_j, rot_deg)."""
    theta1_rad = math.radians(rot1_deg)
    cos1, sin1 = math.cos(theta1_rad), math.sin(theta1_rad)
    s_rot_i = cos1 * s2_i - sin1 * s2_j
    s_rot_j = sin1 * s2_i + cos1 * s2_j
    return (s1_i + s_rot_i, s1_j + s_rot_j, rot1_deg + rot2_deg)


def prune_brightness_outliers(exposure_groups, reg, std_reduction_factor=3.0):
    for exp_key in sorted(exposure_groups.keys()):
        group = exposure_groups[exp_key]
        if len(group) <= 4:
            continue
        while True:
            if len(group) <= 4:
                break
            brightnesses = np.array([ii.avg_brightness for ii in group])
            mean_b = float(np.mean(brightnesses))
            std_b = float(np.std(brightnesses))
            if std_b <= 0:
                break
            k = int(np.argmax(np.abs(brightnesses - mean_b)))
            mask = np.ones(len(group), dtype=bool)
            mask[k] = False
            new_std = float(np.std(brightnesses[mask]))
            if new_std > std_b / std_reduction_factor:
                break
            removed_ii = group[k]
            group.pop(k)
            to_drop = [key for key in reg if key[0] == exp_key and (key[1] == k or key[2] == k)]
            for key in to_drop:
                del reg[key]
            reg_exp_old = {(i, j): reg[(exp_key, i, j)] for (e, i, j) in list(reg.keys()) if e == exp_key}
            for (i, j) in list(reg_exp_old.keys()):
                del reg[(exp_key, i, j)]
            for (i, j), v in reg_exp_old.items():
                if i == k or j == k:
                    continue
                i_new = i if i < k else i - 1
                j_new = j if j < k else j - 1
                reg[(exp_key, i_new, j_new)] = v
            print(
                f"prune_brightness_outliers: exp={exp_key:.5f} removed image (idx {k}) {removed_ii.path.name} "
                f"brightness={removed_ii.avg_brightness:.6f} group_std change when removed: {std_b:.6f} -> {new_std:.6f}"
            )


def prune_failed_registrations(exposure_groups, reg, threshold):
    exposure_times = sorted(list({e for (e, i, j) in reg}))

    for exp_key in exposure_times:
        group = exposure_groups.get(exp_key)
        if group is None:
            continue
        while True:
            reg_exp = {(i, j): v for (e, i, j), v in reg.items() if e == exp_key}
            indices = sorted(list({i for (i, j) in reg_exp}.union({j for (i, j) in reg_exp})))
            bad_triplets = []

            for a, b, c in itertools.permutations(indices, 3):
                if (a, b) not in reg_exp or (b, c) not in reg_exp or (a, c) not in reg_exp:
                    continue
                rab = reg_exp[(a, b)]
                rbc = reg_exp[(b, c)]
                rac = reg_exp[(a, c)]
                composed = compose_transforms(rab[0], rab[1], rab[2], rbc[0], rbc[1], rbc[2])
                score = max(abs(composed[0] - rac[0]), abs(composed[1] - rac[1]), abs(composed[2] - rac[2]))
                if score > threshold:
                    bad_triplets.append((a, b, c, score))

            if not bad_triplets:
                break

            counts = Counter()
            for (a, b, c, _) in bad_triplets:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            k = counts.most_common(1)[0][0]
            removed_ii = group[k]
            group.pop(k)
            to_drop = [key for key in reg if key[0] == exp_key and (key[1] == k or key[2] == k)]
            for key in to_drop:
                del reg[key]
            reg_exp_old = {(i, j): reg[(exp_key, i, j)] for (e, i, j) in list(reg.keys()) if e == exp_key}
            for (i, j) in list(reg_exp_old.keys()):
                del reg[(exp_key, i, j)]
            for (i, j), v in reg_exp_old.items():
                if i == k or j == k:
                    continue
                i_new = i if i < k else i - 1
                j_new = j if j < k else j - 1
                reg[(exp_key, i_new, j_new)] = v
            print(f"prune_failed_registrations: exp={exp_key:.5f} removed image (idx {k}) {removed_ii.path.name}")


def run_group(exposure_time, n, reg_exp, device, n_iter=100_000, peak_lr=1e-3, warmup_frac=0.1):
    best_ij = None
    best_mag = float("inf")
    for (i, j) in reg_exp:
        s_i, s_j, _ = reg_exp[(i, j)]
        mag = math.sqrt(s_i**2 + s_j**2)
        if mag < best_mag:
            best_mag = mag
            best_ij = (i, j)
    i0, j0 = best_ij
    s_i, s_j, r = reg_exp[(i0, j0)]
    placed = {i0, j0}
    abs_x = [0.0] * n
    abs_y = [0.0] * n
    abs_angle = [0.0] * n
    abs_x[i0], abs_y[i0], abs_angle[i0] = 0.0, 0.0, 0.0
    abs_x[j0], abs_y[j0], abs_angle[j0] = s_i, s_j, math.radians(-r)

    while len(placed) < n:
        best_k = None
        best_ref = None
        best_mag = float("inf")
        for k in range(n):
            if k in placed:
                continue
            for ref in placed:
                if (ref, k) not in reg_exp:
                    continue
                s_i, s_j, _ = reg_exp[(ref, k)]
                mag = math.sqrt(s_i**2 + s_j**2)
                if mag < best_mag:
                    best_mag = mag
                    best_k = k
                    best_ref = ref
        if best_k is None:
            break
        ref, k = best_ref, best_k
        s_i, s_j, r_deg = reg_exp[(ref, k)]
        r_rad = math.radians(r_deg)
        theta_ref = abs_angle[ref]
        dx = math.cos(theta_ref) * s_i - math.sin(theta_ref) * s_j
        dy = math.sin(theta_ref) * s_i + math.cos(theta_ref) * s_j
        theta_k = theta_ref - r_rad
        abs_x[k] = abs_x[ref] + dx
        abs_y[k] = abs_y[ref] + dy
        abs_angle[k] = theta_k
        placed.add(k)

    abs_xy = torch.tensor([[abs_x[i], abs_y[i]] for i in range(n)], dtype=torch.float32, device=device, requires_grad=True)
    abs_angle_t = torch.tensor([abs_angle[i] for i in range(n)], dtype=torch.float32, device=device, requires_grad=True)

    pairs = list(reg_exp.keys())
    reg_shift_i = torch.tensor([reg_exp[(i, j)][0] for (i, j) in pairs], dtype=torch.float32, device=device)
    reg_shift_j = torch.tensor([reg_exp[(i, j)][1] for (i, j) in pairs], dtype=torch.float32, device=device)
    reg_rot = torch.tensor([math.radians(reg_exp[(i, j)][2]) for (i, j) in pairs], dtype=torch.float32, device=device)
    idx_i = torch.tensor([i for (i, j) in pairs], dtype=torch.long, device=device)
    idx_j = torch.tensor([j for (i, j) in pairs], dtype=torch.long, device=device)

    def loss_fn():
        x_i = abs_xy[idx_i, 0]
        y_i = abs_xy[idx_i, 1]
        x_j = abs_xy[idx_j, 0]
        y_j = abs_xy[idx_j, 1]
        theta_i = abs_angle_t[idx_i]
        theta_j = abs_angle_t[idx_j]
        dx = x_j - x_i
        dy = y_j - y_i
        ci = torch.cos(-theta_i)
        si = torch.sin(-theta_i)
        impl_shift_i = ci * dx - si * dy
        impl_shift_j = si * dx + ci * dy
        impl_rot = theta_i - theta_j
        loss_shift = ((impl_shift_i - reg_shift_i) ** 2 + (impl_shift_j - reg_shift_j) ** 2).sum()
        loss_rot = ((impl_rot - reg_rot) ** 2).sum()
        return loss_shift, loss_rot

    def lr_schedule(step, n_steps):
        if warmup_frac > 0 and step < n_steps * warmup_frac:
            return peak_lr * (step / (n_steps * warmup_frac))
        progress = (step - n_steps * warmup_frac) / max(1, n_steps * (1 - warmup_frac))
        return 0.5 * peak_lr * (1 + math.cos(math.pi * min(1.0, progress)))

    n_phase = 50_000
    with torch.no_grad():
        ls0, lr0 = loss_fn()
    print(f"\n  exp={exposure_time}: initial loss_shift={ls0.item():.6f} loss_rot={lr0.item():.6f}")

    if lr0 > 0.0:
        opt_rot = torch.optim.Adam([abs_angle_t], lr=peak_lr)
        for step in tqdm.tqdm(range(n_phase), desc="angles"):
            opt_rot.zero_grad()
            for g in opt_rot.param_groups:
                g["lr"] = lr_schedule(step, n_phase)
            _, loss_rot = loss_fn()
            loss_rot.backward()
            opt_rot.step()

        with torch.no_grad():
            ls0, lr0 = loss_fn()
        print(f"  exp={exposure_time}: phase1 loss_shift={ls0.item():.6f} loss_rot={lr0.item():.6f}")

    abs_angle_t.requires_grad_(False)
    opt_shift = torch.optim.Adam([abs_xy], lr=peak_lr)
    for step in tqdm.tqdm(range(n_phase), desc="shifts"):
        opt_shift.zero_grad()
        for g in opt_shift.param_groups:
            g["lr"] = lr_schedule(step, n_phase)
        loss_shift, _ = loss_fn()
        loss_shift.backward()
        opt_shift.step()

    with torch.no_grad():
        ls1, lr1 = loss_fn()
    print(f"  exp={exposure_time}: final   loss_shift={ls1.item():.6f} loss_rot={lr1.item():.6f}")
    return ls1.item(), lr1.item(), abs_xy.detach(), abs_angle_t.detach()


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
    warped_list = []
    for j in range(n):
        img_j = load_grayscale(group[j], device)
        mj_i, mj_j, r_j = group[j].moon[0], group[j].moon[1], group[j].moon[2]
        H, W = img_j.shape
        ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
        jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
        dist = torch.sqrt((ii - mj_i) ** 2 + (jj - mj_j) ** 2)
        mask_j = (dist > r_j + 2.0).to(torch.float32)
        x_j = abs_xy[j, 0].item()
        y_j = abs_xy[j, 1].item()
        theta_j_deg = -math.degrees(abs_angle_t[j].item())
        w_img = apply_transform_single(img_j, x_j, y_j, theta_j_deg, device)
        w_mask = apply_transform_single(mask_j, x_j, y_j, theta_j_deg, device)
        warped_list.append(w_img)
        if sum_img is None:
            sum_img = w_img * w_mask
            sum_mask = w_mask.clone()
        else:
            sum_img = sum_img + w_img * w_mask
            sum_mask = sum_mask + w_mask
    avg_img = sum_img / (sum_mask + epsilon)
    return avg_img, warped_list


def stage1_load(eda00_pkl: Path):
    """Load stage-0 pickle: exposure_groups, pairwise reg (mutable reg dict for pruning)."""
    with open(eda00_pkl, "rb") as fd:
        exposure_groups = pickle.load(fd)
        reg = pickle.load(fd)
    return exposure_groups, reg


def stage1_prune_groups(exposure_groups, reg) -> None:
    """Brightness outlier drop (large groups), then triplet-consistency pruning (in-place)."""
    prune_brightness_outliers(exposure_groups, reg, std_reduction_factor=3.0)
    prune_failed_registrations(exposure_groups, reg, 1.0)


def stage1_optimize_poses_and_debug(
    exposure_groups,
    reg,
    device: torch.device,
    debug_img_dir: Path,
    peak_lr: float = 1e-3,
    warmup_frac: float = 0.1,
) -> dict:
    """Per exposure: Adam on global poses to match all pairwise regs; write crop/GIF debug."""
    n_iter = 100_000
    opt_results = {}
    for exposure_time in sorted(exposure_groups.keys()):
        group = list(exposure_groups[exposure_time])
        n = len(group)
        if n < 2:
            continue
        reg_exp = {
            (i, j): reg[(exposure_time, i, j)]
            for (i, j) in itertools.permutations(range(n), 2)
            if (exposure_time, i, j) in reg
        }
        if not reg_exp:
            continue
        _, _, abs_xy, abs_angle_t = run_group(
            exposure_time, n, reg_exp, device, n_iter=n_iter, peak_lr=peak_lr, warmup_frac=warmup_frac
        )
        opt_results[exposure_time] = {
            "abs_xy": abs_xy.detach().cpu().numpy().astype(np.float64),
            "abs_angle_t": abs_angle_t.detach().cpu().numpy().astype(np.float64),
        }

        idx_show = random.randint(0, n - 1)
        ii_show = group[idx_show]
        mi, mj, r = ii_show.moon[0], ii_show.moon[1], ii_show.moon[2]
        if exposure_time < 0.004 - 1.0e-6:
            half = 1.2 * r
        elif exposure_time < 0.5 - 1.0e-6:
            half = 3.0 * r
        else:
            half = 6.0 * r
        H, W = ii_show.height, ii_show.width
        i_lo = max(0, int(mi - half))
        i_hi = min(H, int(mi + half))
        j_lo = max(0, int(mj - half))
        j_hi = min(W, int(mj + half))
        debug_img_dir.mkdir(parents=True, exist_ok=True)
        img_single = load_grayscale(ii_show, device)
        avg_img, warped_list = compute_weighted_average(group, abs_xy, abs_angle_t, device)
        crop_single = (img_single[i_lo:i_hi, j_lo:j_hi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        crop_avg = (avg_img[i_lo:i_hi, j_lo:j_hi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(crop_single).save(debug_img_dir / f"v0-eda02_debugimg_{exposure_time:.6f}_random.png")
        Image.fromarray(crop_avg).save(debug_img_dir / f"v0-eda02_debugimg_{exposure_time:.6f}_average.png")
        frames = []
        for w in warped_list:
            c = (w[i_lo:i_hi, j_lo:j_hi].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(c))
        frames[0].save(
            debug_img_dir / f"v0-eda02_debugimg_{exposure_time:.6f}_anim.gif",
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0,
        )

    return opt_results


def stage1_save_pickle(out_pkl: Path, exposure_groups, reg: dict, opt_results: dict) -> Path:
    reg_plain = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in reg.items()}
    out_pkl = Path(out_pkl)
    with open(out_pkl, "wb") as fd:
        pickle.dump(exposure_groups, fd)
        pickle.dump(reg_plain, fd)
        pickle.dump(opt_results, fd)
    print(f"Saved {out_pkl} (exposure_groups, reg, opt_results: abs_xy + abs_angle_t per exposure)")
    return out_pkl


def run_stage1(eda00_pkl: Path, out_pkl: Path, debug_img_dir: Path) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage 1.")
    device = torch.device("cuda")

    exposure_groups, reg = stage1_load(eda00_pkl)
    stage1_prune_groups(exposure_groups, reg)
    opt_results = stage1_optimize_poses_and_debug(exposure_groups, reg, device, debug_img_dir)
    return stage1_save_pickle(out_pkl, exposure_groups, reg, opt_results)
