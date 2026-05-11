"""Stage 3: composite merge, radial processing, sharpen, RGB."""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter

import eclipse_v1.stage0  # noqa: F401
from eclipse_v1.stage0 import find_moon
from eclipse_v1.coords import cartesian_to_polar, polar_to_cartesian
from eclipse_v1.utils import load_grayscale, apply_transform_single, compute_weighted_average, moon_median

# --- Parameters for the sliding-window FFT sharpen (visible from notebooks) ---
PATCH_SIDE = 256
PATCH_STRIDE = 8
FFT_INWARD_MEDIAN_SPAN = 32
UNSHARP_GAUSSIAN_SIGMAS = (2.0, 4.0, 8.0)
UNSHARP_WEIGHTS = (8.0, 8.0, 0.5)

MERGE_WEIGHT_SIGMA = 0.2
VALID_THRESH = 0.9999
RGB_DIM_QUOTIENTS = (0.25, 0.28, 0.37)


@dataclass
class Stage3Context:
    """Mutable state passed through stepped stage-3 functions (notebooks) or `run` (one-shot)."""

    workdir: Path                                              # root output directory; set at construction
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))  # CUDA device; set at construction
    exposure_groups: Any = None                                # exposure_time -> [ImageInfo]; loaded by load_inputs
    reg: Any = None                                            # pairwise intra-exposure registration dict from stage0; loaded by load_inputs
    opt_results: Any = None                                    # per-exposure pose optimization (abs_xy, abs_angle_t) from stage1; loaded by load_inputs
    cross_reg: Any = None                                      # (t0,t1) -> (shift_i, shift_j, rotation) cross-exposure reg from stage2; loaded by load_inputs
    gamma_by_pair: Any = None                                  # (t0,t1) -> gamma brightness scaling from stage2; loaded by load_inputs
    exposure_times_sorted: list[Any] = field(default_factory=list)  # sorted exposure times (shortest two dropped); set by load_inputs
    t_ref: Any = None                                          # reference exposure time (shortest kept); set by load_inputs
    moon_ref: tuple[float, float, float] | None = None         # (i, j, radius) median moon in reference exposure; set by load_inputs
    avg_images: dict[Any, torch.Tensor] = field(default_factory=dict)  # exp -> full-res per-exposure average tensor; set by build_per_exposure_averages
    avg_masks: dict[Any, torch.Tensor] = field(default_factory=dict)   # exp -> valid-pixel mask tensor; set by build_per_exposure_averages
    H_ref: int = 0                                             # pixel height of the reference image; set by build_per_exposure_averages
    W_ref: int = 0                                             # pixel width of the reference image; set by build_per_exposure_averages
    composite: Optional[np.ndarray] = None                     # weighted merge of all exposures in ref coords (float64); set by warp_merge_to_composite, freed after crop_and_save_composite
    valid_all: Optional[np.ndarray] = None                     # minimum valid coverage map across exposures; set by warp_merge_to_composite
    r_lo: int = 0                                              # top crop row (mutual-coverage boundary + 16 px margin); set by crop_and_save_composite
    r_hi: int = 0                                              # bottom crop row; set by crop_and_save_composite
    c_lo: int = 0                                              # left crop column; set by crop_and_save_composite
    c_hi: int = 0                                              # right crop column; set by crop_and_save_composite
    composite_crop: Optional[np.ndarray] = None                # composite sliced to crop bounds (float64); set by crop_and_save_composite
    mi_crop: float = 0.0                                       # moon center row in crop coordinates; set by crop_and_save_composite, refined by radial_normalize_display
    mj_crop: float = 0.0                                       # moon center column in crop coordinates; set by crop_and_save_composite, refined by radial_normalize_display
    moon_r: float = 0.0                                        # moon radius in pixels; set by crop_and_save_composite, refined by radial_normalize_display
    moon_mask: Optional[np.ndarray] = None                     # boolean mask of pixels inside the moon disk; set by radial_normalize_display
    H_crop: int = 0                                            # pixel height of composite_crop; set by crop_and_save_composite
    W_crop: int = 0                                            # pixel width of composite_crop; set by crop_and_save_composite
    display: Optional[np.ndarray] = None                       # radially tone-mapped grayscale image in [0,1]; set by radial_normalize_display
    sharpened_fft_diff: Optional[np.ndarray] = None            # display + FFT-smoothed unsharp signal, moon zeroed; set by fft_unsharp_and_save
    display_rgb: Optional[np.ndarray] = None                   # RGB image with vignette, ready for export; set by rgb_vignette_and_radial_pickle


def _ref_to_source_grid(H, W, chain_tuples, device):
    ci, cj = H / 2.0, W / 2.0
    ii = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    jj = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    i_cur = ii.clone()
    j_cur = jj.clone()
    for (shift_i, shift_j, rotation_deg) in chain_tuples:
        angle_rad = math.radians(-rotation_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        di = i_cur - ci - shift_i
        dj = j_cur - cj - shift_j
        i_cur = di * cos_a + dj * sin_a + ci
        j_cur = -di * sin_a + dj * cos_a + cj
    return i_cur, j_cur


def _grid_to_normalized_grid(i_src, j_src, H, W):
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    return torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)


def _warp_to_ref(img, chain_tuples, H_ref, W_ref, device):
    i_src, j_src = _ref_to_source_grid(H_ref, W_ref, chain_tuples, device)
    grid = _grid_to_normalized_grid(i_src, j_src, img.shape[0], img.shape[1])
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(0).squeeze(0)


def _scale_to_ref(exposure_times_sorted, gamma_by_pair, k):
    if k == 0:
        return 1.0
    scale = 1.0
    for i in range(k):
        t0, t1 = exposure_times_sorted[i], exposure_times_sorted[i + 1]
        if (t0, t1) not in gamma_by_pair:
            return None
        g = gamma_by_pair[(t0, t1)]
        scale *= (t0 / t1) ** (1.0 / g)
    return scale


def _weight_from_value(y, weight_sigma: float = MERGE_WEIGHT_SIGMA):
    y = np.clip(y, 0.0, 1.0).astype(np.float64)
    return np.exp(-((y - 0.5) / weight_sigma) ** 2)


def _dist_to_corners(ci, cj, H, W):
    corners = [(0, 0), (H - 1, 0), (0, W - 1), (H - 1, W - 1)]
    return max(math.sqrt((i - ci) ** 2 + (j - cj) ** 2) for i, j in corners)


def _vertical_gaussian_blur(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.0) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    ax = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-(ax**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1, 1)
    x = x.unsqueeze(0).unsqueeze(0)
    y = F.conv2d(x, kernel.to(x.dtype).to(x.device), padding=(kernel_size // 2, 0))
    return y.squeeze(0).squeeze(0)


# --- FFT / unsharp helpers -------------------------------------------------


def _polar_fft_geometry(a: int):
    r_max = (a / 2.0) * math.sqrt(2.0)
    Nr = max(1, math.ceil(2.0 * r_max))
    Ntheta = max(1, math.ceil(2.0 * math.pi * r_max))
    return r_max, Nr, Ntheta


def _amp_cart_to_polar_bilinear(amp, a, r_max, Nr, Ntheta, ci, cj):
    device, dtype = amp.device, amp.dtype
    ir = torch.arange(Nr, device=device, dtype=dtype).view(Nr, 1).expand(Nr, Ntheta)
    itv = torch.arange(Ntheta, device=device, dtype=dtype).view(1, Ntheta).expand(Nr, Ntheta)
    denom_r = max(Nr - 1, 1)
    r = ir * (r_max / denom_r)
    theta = (itv + 0.5) * (math.pi / Ntheta)
    dx = r * torch.cos(theta)
    dy = r * torch.sin(theta)
    row = ci + dy
    col = cj + dx
    amp4 = amp.unsqueeze(0).unsqueeze(0)
    H = W = a
    x = 2.0 * col / max(W - 1, 1) - 1.0
    y = 2.0 * row / max(H - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1).unsqueeze(0)
    P = F.grid_sample(amp4, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    P = P.squeeze(0).squeeze(0)
    dc = amp[int(round(float(ci))), int(round(float(cj)))]
    P[0, :].fill_(float(dc))
    return P


def _polar_add_inward_median(P, inward_median_span: int):
    Nr, Ntheta = P.shape
    k = inward_median_span
    if k < 1:
        return P
    if Nr == 0:
        return P
    if Nr <= k:
        base = torch.median(P, dim=0).values
        return P + base.unsqueeze(0).expand_as(P)
    m = torch.zeros_like(P)
    m[:k, :] = torch.median(P[0:k, :], dim=0).values.unsqueeze(0).expand(k, -1)
    unf = P.unfold(0, k, 1)
    take = Nr - k
    m[k:, :] = torch.median(unf[:take, :, :], dim=2).values
    return P + m


def _amp_polar_to_cart_bilinear(P, a, r_max, Nr, Ntheta, ci, cj):
    device, dtype = P.device, P.dtype
    ii = torch.arange(a, device=device, dtype=dtype).view(a, 1).expand(a, a)
    jj = torch.arange(a, device=device, dtype=dtype).view(1, a).expand(a, a)
    dx = jj - cj
    dy = ii - ci
    neg = dy < 0
    dx = torch.where(neg, -dx, dx)
    dy = torch.where(neg, -dy, dy)
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
    r = torch.clamp(r, max=r_max)
    pdc = torch.mean(P[0, :]).to(dtype)
    near = r < 1e-9
    denom_r = max(Nr - 1, 1)
    ir_float = (r / r_max) * denom_r
    it_float = theta * (Ntheta / math.pi) - 0.5
    x = 2.0 * it_float / max(Ntheta - 1, 1) - 1.0
    y = 2.0 * ir_float / denom_r - 1.0
    P4 = P.unsqueeze(0).unsqueeze(0)
    grid = torch.stack([x, y], dim=-1).unsqueeze(0)
    out = F.grid_sample(P4, grid, mode="bilinear", padding_mode="border", align_corners=True)
    out = out.squeeze(0).squeeze(0)
    out = torch.where(near, pdc, out)
    return out


def _fft_amp_polar_median_roundtrip(amp: torch.Tensor, inward_median_span: int) -> torch.Tensor:
    a = amp.shape[0]
    assert amp.shape == (a, a), amp.shape
    ci = (a - 1) / 2.0
    cj = (a - 1) / 2.0
    r_max, Nr, Ntheta = _polar_fft_geometry(a)
    P = _amp_cart_to_polar_bilinear(amp, a, r_max, Nr, Ntheta, ci, cj)
    P = _polar_add_inward_median(P, inward_median_span)
    return _amp_polar_to_cart_bilinear(P, a, r_max, Nr, Ntheta, ci, cj)


def _fftshift_complex_polar_preprocess(Fs: torch.Tensor, inward_median_span: int) -> torch.Tensor:
    amp = torch.abs(Fs)
    amp_mod = _fft_amp_polar_median_roundtrip(amp, inward_median_span)
    ph = torch.angle(Fs)
    rd = Fs.real.dtype
    out_r = (amp_mod * torch.cos(ph)).to(rd)
    out_i = (amp_mod * torch.sin(ph)).to(rd)
    return torch.complex(out_r, out_i)


def _blur_dist_to_corners(ci, cj, H, W):
    corners = [(0, 0), (H - 1, 0), (0, W - 1), (H - 1, W - 1)]
    return max(math.hypot(i - ci, j - cj) for i, j in corners)


def _patch_window_flat_circle(a, dtype=np.float64):
    ci = (a - 1) / 2.0
    cj = (a - 1) / 2.0
    ii = np.arange(a, dtype=dtype).reshape(-1, 1)
    jj = np.arange(a, dtype=dtype).reshape(1, -1)
    r = np.hypot(ii - ci, jj - cj)
    R_flat = 0.667 * a / 2.0
    R_max = float(np.hypot(ci, cj))
    denom = max(R_max - R_flat, 1e-12)
    t = np.clip((r - R_flat) / denom, 0.0, 1.0)
    w = np.where(r <= R_flat, 1.0, 0.5 * (1.0 + np.cos(np.pi * t)))
    w = np.where(r >= R_max, 0.0, w)
    return w.astype(np.float32, copy=False)


def _fft_bw_energy_protect_mask(Fs_bw, energy_frac=0.9):
    a = Fs_bw.shape[0]
    assert Fs_bw.shape == (a, a), Fs_bw.shape
    power = (torch.abs(Fs_bw) ** 2).reshape(-1)
    total = power.sum()
    dc_flat = (a // 2) * a + (a // 2)
    mask = torch.zeros(a * a, device=Fs_bw.device, dtype=torch.bool)
    mask[dc_flat] = True
    if float(total) < 1e-30:
        return mask.view(a, a)
    target = float(energy_frac) * total
    order = torch.argsort(power, descending=True)
    csum = torch.cumsum(power[order], dim=0)
    hit = csum >= target
    if bool(hit.any()):
        k = int(torch.argmax(hit.to(torch.int8)).item()) + 1
    else:
        k = power.numel()
    mask[order[:k]] = True
    mask[dc_flat] = True
    return mask.view(a, a)


def _make_fft_mask2_percentile_for_tile(
    Fs, ci, cj, moon_i, moon_j, moon_r, epsilon=1e-12, p_near=90.0, p_far=99.8, exp_k=2.0
):
    R = max(float(moon_r), 1e-12)
    d = math.hypot(float(ci) - float(moon_i), float(cj) - float(moon_j))
    rho = d / R
    if rho <= 1.0:
        percentile = p_near
    elif rho >= 7.0:
        percentile = p_far
    else:
        t = (7.0 - rho) / 6.0
        v0 = 100.0 - p_far
        v1 = 100.0 - p_near
        v = v0 * (v1 / v0) ** t
        percentile = 100.0 - v
    aa = Fs.shape[0]
    assert Fs.shape == (aa, aa), Fs.shape
    dtype = Fs.real.dtype
    amplitude = torch.abs(Fs)
    th = torch.quantile(amplitude.reshape(-1), percentile / 100.0)
    newampl = (amplitude - th).clamp(min=0)
    q = newampl / (amplitude + epsilon)
    return q.to(dtype=dtype)


def _radial_blend_fancy_plain(fancy_np, plain_np, mi, mj, moon_r, blur_sigma):
    H, W = plain_np.shape
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    r = np.hypot(ii - float(mi), jj - float(mj))
    r0 = float(moon_r) + float(blur_sigma) * 3.0
    r1 = float(moon_r) + float(blur_sigma) * 4.0
    span = max(r1 - r0, 1e-20)
    w_fancy = np.clip((r1 - r) / span, 0.0, 1.0).astype(np.float32)
    return w_fancy * fancy_np.astype(np.float32) + (1.0 - w_fancy) * plain_np.astype(np.float32)


def _periodic_gaussian_kernel_1d(n, sigma, device, dtype):
    n = int(n)
    if n <= 1 or sigma < 1e-8:
        return None
    idx = torch.arange(0, n, device=device, dtype=dtype)
    d = torch.minimum(idx, n - idx)
    g = torch.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)
    g = g / g.sum().clamp(min=1e-20)
    return g


def _fancy_polar_blur_cartesian(display_np, blur_sigma, mi_crop, mj_crop, moon_r, device, dtype=torch.float32):
    H, W = display_np.shape
    center = (float(mi_crop), float(mj_crop))
    radius_min = 0.0
    radius_max = _blur_dist_to_corners(mi_crop, mj_crop, H, W)
    n_r = max(int(math.ceil(2 * (radius_max - radius_min))) + 1, 2)
    n_theta = max(int(math.ceil(4 * math.pi * radius_max)) + 1, 2)
    dr = (radius_max - radius_min) / max(n_r - 1, 1)
    i_cont = (radius_max - float(moon_r + 4)) / max(dr, 1e-20)
    sigma_polar = float(blur_sigma) / max(dr, 1e-20)

    img = torch.from_numpy(display_np.astype(np.float32, copy=False)).to(device=device, dtype=dtype)
    polar, _ = cartesian_to_polar(img, center, radius_min, radius_max, n_r, n_theta)
    polar_h = polar.clone()

    i_band_lo = max(0, int(math.floor(i_cont - 3.0 * sigma_polar)))
    i_band_hi = min(n_r - 1, int(math.ceil(i_cont + sigma_polar)))
    denom_r = max(n_r - 1, 1)
    if n_theta > 1:
        inv_twopi = (n_theta - 1) / (2.0 * math.pi)
        for i in tqdm.tqdm(
            range(i_band_lo, i_band_hi + 1),
            desc=f"fancy polar θ-blur σ={blur_sigma}",
            leave=False,
        ):
            r_i = radius_max - i * (radius_max - radius_min) / denom_r
            sigma_h = float(blur_sigma) * inv_twopi / max(float(r_i), 1e-6)
            g = _periodic_gaussian_kernel_1d(n_theta, sigma_h, device, dtype)
            row = polar[i]
            if g is None:
                polar_h[i] = row
            else:
                Xf = torch.fft.rfft(row)
                Gf = torch.fft.rfft(g)
                polar_h[i] = torch.fft.irfft(Xf * Gf, n=n_theta)

    i_lim = int(math.ceil(i_cont))
    i_lim_b = int(min(i_lim, int(n_r)))
    if i_lim_b > 0:
        j_idx = torch.arange(0, int(n_r), device=device, dtype=dtype)
        r_j = radius_max - j_idx * (radius_max - radius_min) / denom_r
        src_ok = r_j >= float(moon_r)
        i_rows = torch.arange(0, i_lim_b, device=device, dtype=dtype).view(-1, 1)
        j_cols = j_idx.view(1, -1)
        sp_t = torch.tensor(sigma_polar, device=device, dtype=dtype)
        delta = j_cols - i_rows
        W_vert = torch.exp(-0.5 * (delta / sp_t.clamp(min=1e-20)) ** 2)
        W_vert = W_vert * src_ok.to(dtype).view(1, -1)
        row_sum = W_vert.sum(dim=1, keepdim=True)
        W_vert = W_vert / row_sum.clamp(min=1e-20)
        bad = (row_sum.squeeze(1) < 1e-20) | (sp_t < 1e-8)
        if bool(bad.any().item()):
            bi = torch.nonzero(bad, as_tuple=False).squeeze(-1)
            W_vert[bi, :] = 0
            W_vert[bi, bi] = 1.0
        polar_v = polar_h.clone()
        polar_v[:i_lim_b, :] = W_vert @ polar_h
    else:
        polar_v = polar_h

    cart = polar_to_cartesian(polar_v, center, radius_min, radius_max, H, W)
    return cart.detach().cpu().numpy()


def _sliding_diff_smooth_for_sigma(
    display,
    display_for_blur,
    blur_sigma,
    inward_median_span,
    mi_crop,
    mj_crop,
    moon_r,
    dev,
    a,
    stride,
    area_tol=0.05,
    protect_energy_frac=0.9999,
):
    blurred_plain = gaussian_filter(display_for_blur, sigma=blur_sigma, mode="nearest")
    fancy_cart = _fancy_polar_blur_cartesian(
        display_for_blur, blur_sigma, mi_crop, mj_crop, moon_r, dev, torch.float32
    )
    blurred = _radial_blend_fancy_plain(
        fancy_cart, blurred_plain, mi_crop, mj_crop, moon_r, blur_sigma
    )
    diff_np = (display.astype(np.float64) - blurred.astype(np.float64)).astype(np.float32)
    H_d, W_d = display.shape
    assert H_d >= a and W_d >= a, (H_d, W_d, a)
    diff_t = torch.from_numpy(diff_np).to(device=dev, dtype=torch.float32)
    win_np = _patch_window_flat_circle(a)
    win = torch.from_numpy(win_np).to(dev)
    ii = torch.arange(H_d, device=dev, dtype=torch.float32).view(-1, 1).expand(H_d, W_d)
    jj = torch.arange(W_d, device=dev, dtype=torch.float32).view(1, -1).expand(H_d, W_d)
    dist_sq = (ii - float(mi_crop)) ** 2 + (jj - float(mj_crop)) ** 2
    bw_t = (dist_sq > float(moon_r) ** 2).to(torch.float32)
    acc = torch.zeros(H_d, W_d, device=dev, dtype=torch.float32)
    wsum = torch.zeros(H_d, W_d, device=dev, dtype=torch.float32)
    row_starts = list(range(0, H_d - a + 1, stride))
    col_starts = list(range(0, W_d - a + 1, stride))
    for r0 in tqdm.tqdm(row_starts, desc=f"FFT-smooth diff sigma={blur_sigma}"):
        for c0 in col_starts:
            patch = diff_t[r0 : r0 + a, c0 : c0 + a]
            p_w = patch * win
            bw_patch = bw_t[r0 : r0 + a, c0 : c0 + a]
            n_in = int((bw_patch < 0.5).sum().item())
            n_out = a * a - n_in
            tot = a * a
            f0 = n_in / tot
            f1 = n_out / tot
            tc_i = r0 + 0.5 * (a - 1)
            tc_j = c0 + 0.5 * (a - 1)
            if f0 == 0 or f1 == 0 or min(f0, f1) < area_tol:
                F_fft = torch.fft.fft2(p_w)
                Fs = torch.fft.fftshift(F_fft)
                Fs_proc = _fftshift_complex_polar_preprocess(Fs, inward_median_span)
                G = _make_fft_mask2_percentile_for_tile(Fs_proc, tc_i, tc_j, mi_crop, mj_crop, moon_r)
                Fm = torch.fft.ifftshift(Fs_proc * G)
                recon = torch.fft.ifft2(Fm).real
            elif False and min(f0, f1) < area_tol:
                continue
            else:
                bw_w = bw_patch * win
                Fs_bw = torch.fft.fftshift(torch.fft.fft2(bw_w))
                protect = _fft_bw_energy_protect_mask(Fs_bw, protect_energy_frac)
                F_fft = torch.fft.fft2(p_w)
                Fs = torch.fft.fftshift(F_fft)
                Fs_save = Fs.clone()
                Fs_work = Fs.clone()
                Fs_work[protect] = 0
                Fs_proc = _fftshift_complex_polar_preprocess(Fs_work, inward_median_span)
                G = _make_fft_mask2_percentile_for_tile(Fs_proc, tc_i, tc_j, mi_crop, mj_crop, moon_r)
                Fs_out = Fs_proc * G
                Fs_out[protect] = Fs_save[protect]
                Fm = torch.fft.ifftshift(Fs_out)
                recon = torch.fft.ifft2(Fm).real
            acc[r0 : r0 + a, c0 : c0 + a] += recon * win
            wsum[r0 : r0 + a, c0 : c0 + a] += win
    return (acc / wsum.clamp(min=1e-9)).cpu().numpy()


def load_inputs(ctx: Stage3Context) -> None:
    """Load stage1/stage2 pickles; set exposure list, reference exposure, and moon prior."""
    with open(ctx.workdir / "v1-stage1.pkl", "rb") as fd:
        ctx.exposure_groups = pickle.load(fd)
        ctx.reg = pickle.load(fd)
        ctx.opt_results = pickle.load(fd)
    with open(ctx.workdir / "v1-stage2.pkl", "rb") as fd:
        ctx.cross_reg = pickle.load(fd)
        ctx.gamma_by_pair = pickle.load(fd)
    ctx.device = torch.device("cuda")
    ctx.exposure_times_sorted = sorted(ctx.exposure_groups.keys())[2:]
    ctx.t_ref = ctx.exposure_times_sorted[0]
    ctx.moon_ref = moon_median(ctx.exposure_groups[ctx.t_ref])
    print(f"Reference exposure t_ref={ctx.t_ref}, moon_ref (i,j,r)={ctx.moon_ref}")
    print(f"cross_reg pairs: {len(ctx.cross_reg)}, gamma_by_pair: {len(ctx.gamma_by_pair)}")


def build_per_exposure_averages(ctx: Stage3Context) -> None:
    """GPU stack mean per exposure (moon blackened, warped with stage-1 poses)."""
    device = ctx.device
    ctx.avg_images.clear()
    ctx.avg_masks.clear()
    for exp in tqdm.tqdm(ctx.exposure_times_sorted, desc="Averaged images"):
        group = ctx.exposure_groups[exp]
        if exp not in ctx.opt_results or len(group) < 2:
            continue
        abs_xy = torch.from_numpy(ctx.opt_results[exp]["abs_xy"]).to(device)
        abs_angle_t = torch.from_numpy(ctx.opt_results[exp]["abs_angle_t"]).to(device)
        ctx.avg_images[exp], ctx.avg_masks[exp], _ = compute_weighted_average(group, abs_xy, abs_angle_t, device)
    print(f"Built {len(ctx.avg_images)} averaged images.")
    ctx.H_ref, ctx.W_ref = next(iter(ctx.avg_images.values())).shape
    print(f"Reference shape H={ctx.H_ref}, W={ctx.W_ref}")


def warp_merge_to_composite(ctx: Stage3Context) -> None:
    """Chain cross_reg, gamma scaling, warp to ref grid; weighted merge → full composite + valid mask."""
    device = ctx.device
    exposure_times_sorted = ctx.exposure_times_sorted
    t_ref = ctx.t_ref
    avg_images = ctx.avg_images
    avg_masks = ctx.avg_masks
    cross_reg = ctx.cross_reg
    gamma_by_pair = ctx.gamma_by_pair

    exposures_with_chain = [t_ref]
    chain_tuples_by_exp = {t_ref: []}
    for k in range(1, len(exposure_times_sorted)):
        t_k = exposure_times_sorted[k]
        if t_k not in avg_images:
            continue
        chain = []
        valid = True
        for i in range(k):
            t0, t1 = exposure_times_sorted[i], exposure_times_sorted[i + 1]
            if (t0, t1) not in cross_reg:
                valid = False
                break
            chain.append(cross_reg[(t0, t1)])
        if not valid:
            continue
        chain_tuples_by_exp[t_k] = chain
        exposures_with_chain.append(t_k)

    print(f"Exposures in chain: {len(exposures_with_chain)}")
    H_ref, W_ref = ctx.H_ref, ctx.W_ref
    sum_val = np.zeros((H_ref, W_ref), dtype=np.float64)
    sum_weight = np.zeros((H_ref, W_ref), dtype=np.float64)
    valid_all = None
    mask_0 = avg_masks[min(avg_masks.keys())] > 0.99
    for t_k_index, t_k in tqdm.tqdm(enumerate(exposures_with_chain), desc="Warp and merge"):
        img_k = avg_images[t_k]
        mask_k = avg_masks[t_k]
        chain = chain_tuples_by_exp[t_k]
        k_idx = exposure_times_sorted.index(t_k)
        scale_k = _scale_to_ref(exposure_times_sorted, gamma_by_pair, k_idx)
        if scale_k is None:
            continue

        orig = img_k.cpu().numpy().astype(np.float64)
        orig_clip = np.clip(orig, 0.0, 1.0)
        value_scaled = np.clip(orig * scale_k, 0.0, 1.0)
        weight_img = _weight_from_value(orig_clip)
        if t_k_index == 0:
            weight_img[orig > 0.5] = 1.0
        if k_idx > 0:
            weight_img *= (mask_k > 0.99).detach().cpu().numpy()

        val_t = torch.from_numpy(value_scaled.astype(np.float32)).to(device)
        w_t = torch.from_numpy(weight_img.astype(np.float32)).to(device)
        valid_t = torch.ones_like(img_k, device=device, dtype=torch.float32)
        warped_val = (_warp_to_ref(val_t, chain, H_ref, W_ref, device) * mask_0).cpu().numpy().astype(np.float64)
        warped_w = _warp_to_ref(w_t, chain, H_ref, W_ref, device).cpu().numpy().astype(np.float64)
        warped_valid = _warp_to_ref(valid_t, chain, H_ref, W_ref, device).cpu().numpy()

        use = warped_valid >= 0.5
        sum_val += np.where(use, warped_val * warped_w, 0.0)
        sum_weight += np.where(use, warped_w, 0.0)

        if valid_all is None:
            valid_all = warped_valid.copy()
        else:
            valid_all = np.minimum(valid_all, warped_valid)

    denom = np.maximum(sum_weight, 1e-20)
    ctx.composite = (sum_val / denom).astype(np.float64)
    ctx.valid_all = valid_all
    ctx.avg_images.clear()
    ctx.avg_masks.clear()
    print(f"Composite shape {ctx.composite.shape}, dtype {ctx.composite.dtype}")


def crop_and_save_composite(ctx: Stage3Context) -> None:
    """Crop to mutual coverage; save float composite + preview; set crop geometry and initial moon (pre-find_moon)."""
    assert ctx.composite is not None and ctx.valid_all is not None and ctx.moon_ref is not None
    out_dir = ctx.workdir
    composite = ctx.composite
    valid_all = ctx.valid_all
    mi, mj, moon_r0 = ctx.moon_ref

    all_valid_mask = valid_all >= VALID_THRESH
    rows = np.any(all_valid_mask, axis=1)
    cols = np.any(all_valid_mask, axis=0)
    r_lo, r_hi = np.where(rows)[0][[0, -1]]
    c_lo, c_hi = np.where(cols)[0][[0, -1]]
    r_lo += 16
    r_hi -= 16
    c_lo += 16
    c_hi -= 16
    ctx.r_lo, ctx.r_hi, ctx.c_lo, ctx.c_hi = r_lo, r_hi, c_lo, c_hi
    composite_crop = composite[r_lo : r_hi + 1, c_lo : c_hi + 1].copy()
    ctx.composite = None

    mi_crop = mi - r_lo
    mj_crop = mj - c_lo
    ctx.H_crop, ctx.W_crop = composite_crop.shape
    print(f"Crop bounds rows [{r_lo},{r_hi}], cols [{c_lo},{c_hi}]; shape {composite_crop.shape}")

    ii = np.arange(ctx.H_crop, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(ctx.W_crop, dtype=np.float64).reshape(1, -1)
    dist_sq = (ii - mi_crop) ** 2 + (jj - mj_crop) ** 2
    moon_mask_preview = dist_sq <= (moon_r0**2)

    np.save(out_dir / "v1-stage3_composite.npy", composite_crop)
    v_min = np.percentile(
        composite_crop[~moon_mask_preview] if np.any(~moon_mask_preview) else composite_crop, 1
    )
    v_max = np.percentile(
        composite_crop[~moon_mask_preview] if np.any(~moon_mask_preview) else composite_crop, 99
    )
    preview = np.clip((composite_crop - v_min) / (v_max - v_min + 1e-9), 0, 1)
    Image.fromarray((preview * 255).clip(0, 255).astype(np.uint8)).save(
        out_dir / "v1-stage3_composite_preview.png"
    )
    print(f"Saved {out_dir / 'v1-stage3_composite.npy'} (float64), {out_dir / 'v1-stage3_composite_preview.png'}")

    ctx.composite_crop = composite_crop
    ctx.mi_crop = float(mi_crop)
    ctx.mj_crop = float(mj_crop)
    ctx.moon_r = float(moon_r0)


def _polar_transform_and_extrapolate(img, center, radius_min, radius_max, n_r, n_theta):
    """Polar-transform img; fill invalid pixels via linear fit from the innermost valid rows."""
    dtype = img.dtype
    polar_img, valid = cartesian_to_polar(img, center, radius_min, radius_max, n_r, n_theta, mask_margin=2)
    valid = valid.to(dtype=dtype)
    polar_img = polar_img.clone()
    valid = valid.clone()

    row_all = valid.bool().all(dim=1)
    if not bool(row_all.any().item()):
        raise AssertionError("polar extrapolation: no fully valid row")
    i_first = int(torch.argmax(row_all.to(torch.int64)).item())
    if not bool(row_all[i_first].item()):
        raise AssertionError("polar extrapolation: no fully valid row (argmax)")
    if not bool(valid[i_first:, :].bool().all().item()):
        raise AssertionError(
            "polar extrapolation: all rows from first full-valid row downward must be fully valid"
        )
    if not ((n_r - 1) - i_first > 100):
        raise AssertionError(
            "polar extrapolation: first full-valid row must be strictly >100 rows above bottom (index n_r-1)"
        )
    slab = polar_img[i_first : i_first + 100, :]
    mean_100 = slab.mean(dim=1, keepdim=True)
    std_100 = slab.std(dim=1, correction=0, keepdim=True)
    if bool((std_100.squeeze(1) < 1e-8).any().item()):
        raise AssertionError("polar extrapolation: template row std < 1e-8")
    slab = (slab - mean_100) / std_100
    frow = slab.mean(dim=0)

    valid_for_mean = valid.clone()
    m = valid_for_mean.bool()
    n_per = m.sum(dim=1).to(dtype)
    for i in tqdm.tqdm(range(i_first)):
        if n_per[i] < 64:
            continue
        sub_row = polar_img[i, :][m[i, :]]
        sub_frow = frow[m[i, :]]
        a = sub_frow.unsqueeze(1)
        a = torch.cat([a, torch.ones_like(a)], dim=1)
        b = sub_row.unsqueeze(1)
        sol = torch.linalg.lstsq(a, b, driver="gels").solution
        row = sol[0, 0] * frow + sol[1, 0]
        polar_img[i, :][~m[i, :]] = row[~m[i, :]]
        valid_for_mean[i, :] = 1
    return polar_img, valid_for_mean, valid


def _radial_tone_map(img, polar_img, valid_for_mean, center, radius_min, radius_max, n_r, n_theta):
    """Sliding-window polar mean → piecewise-linear tone map; average over two angular window sizes."""
    device = img.device
    dtype = img.dtype
    H_crop, W_crop = img.shape[:2]
    display_ts = []
    for row_fraction in [0.15, 1.0]:
        n_cols_use = max(1, int(n_theta * row_fraction))
        half_window = n_cols_use // 2
        polar_ext = torch.cat([polar_img, polar_img, polar_img], dim=1)
        valid_ext = torch.cat([valid_for_mean, valid_for_mean, valid_for_mean], dim=1)
        valid_ext[valid_ext > 0.5] = 1.0
        valid_ext[valid_ext < 0.51] = 0.0
        row_val_ext = (polar_ext * valid_ext).to(dtype)
        row_cnt_ext = valid_ext.to(dtype)
        cs_val = torch.cumsum(
            torch.cat([torch.zeros(n_r, 1, device=device, dtype=dtype), row_val_ext], dim=1), dim=1
        )
        cs_cnt = torch.cumsum(
            torch.cat([torch.zeros(n_r, 1, device=device, dtype=dtype), row_cnt_ext], dim=1), dim=1
        )
        start = (n_theta - half_window + torch.arange(n_theta, device=device)).long()
        sum_v = cs_val[:, start + n_cols_use] - cs_val[:, start]
        sum_n = (cs_cnt[:, start + n_cols_use] - cs_cnt[:, start]).clamp(min=1e-20)
        mean_polar_2d = sum_v / sum_n
        assert torch.all(torch.isfinite(mean_polar_2d))
        argmax = mean_polar_2d.argmax(dim=0)
        max_val = mean_polar_2d.max(dim=0).values
        mask = torch.arange(mean_polar_2d.size(0), device=mean_polar_2d.device).unsqueeze(1) > argmax
        mean_polar_2d[mask] = max_val.unsqueeze(0).expand_as(mean_polar_2d)[mask]

        mean_at = polar_to_cartesian(mean_polar_2d, center, radius_min, radius_max, H_crop, W_crop)
        valid_mask = torch.isfinite(mean_at) & (mean_at > 0)
        display_t = torch.zeros_like(img, device=device, dtype=dtype)
        v = img[valid_mask]
        m_ = mean_at[valid_mask]
        mask1 = (v > m_ / 2) & (v <= m_)
        mask2 = (v > m_) & (v <= 2 * m_)
        mask3 = v > 2 * m_
        display_t[valid_mask] = torch.where(mask1, 0.4 * (v - m_ / 2) / (m_ / 2).clamp(min=1e-9), torch.zeros_like(v))
        display_t[valid_mask] = torch.where(
            mask2, 0.4 + 0.6 * (v - m_) / m_.clamp(min=1e-9), display_t[valid_mask]
        )
        display_t[valid_mask] = torch.where(mask3, torch.ones_like(v), display_t[valid_mask])
        display_ts.append(display_t)
    return torch.stack(display_ts).mean(dim=0)


def _percentile_stretch(display_t, valid, center, radius_min, radius_max, n_r, n_theta):
    """Per-row radial p3 quantile in polar space → stretch [p3, 1] → [0, 1]."""
    H_crop, W_crop = display_t.shape[:2]
    polar_display, _ = cartesian_to_polar(display_t, center, radius_min, radius_max, n_r, n_theta, mask_margin=2)
    blur = torchvision.transforms.GaussianBlur(kernel_size=13, sigma=5)
    valid_blur = blur(valid.float().unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0) > 0.9
    polar_display[~valid_blur] = 1.0
    q = torch.linspace(
        start=0.03, end=0.0001, steps=polar_display.shape[0], device=polar_display.device, dtype=polar_display.dtype
    )
    p3_row = polar_display.quantile(q=q, dim=1).diag()
    p3_smooth = _vertical_gaussian_blur(p3_row.unsqueeze(1), kernel_size=133, sigma=33).squeeze(1)
    p3_polar_2d = p3_smooth.unsqueeze(1).expand(n_r, n_theta)
    p3_at = torch.nan_to_num(
        polar_to_cartesian(p3_polar_2d, center, radius_min, radius_max, H_crop, W_crop), nan=0.0
    )
    span = (1.0 - p3_at).clamp(min=1e-9)
    return ((display_t - p3_at) / span).clamp(0.0, 1.0)


def radial_normalize_display(ctx: Stage3Context) -> None:
    """Refine moon; polar radial tone + p3 stretch → ctx.display (grayscale [0,1])."""
    assert ctx.composite_crop is not None
    device = ctx.device
    composite_crop = ctx.composite_crop
    H_crop, W_crop = ctx.H_crop, ctx.W_crop
    mi_crop, mj_crop = ctx.mi_crop, ctx.mj_crop

    img_rgb = torch.from_numpy(composite_crop).to(device=device, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 3)
    mi_crop, mj_crop, moon_r = find_moon(img_rgb, float(mi_crop), float(mj_crop))
    ctx.mi_crop, ctx.mj_crop, ctx.moon_r = float(mi_crop), float(mj_crop), float(moon_r)

    ii = np.arange(H_crop, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(W_crop, dtype=np.float64).reshape(1, -1)
    dist_sq = (ii - mi_crop) ** 2 + (jj - mj_crop) ** 2
    ctx.moon_mask = dist_sq <= (moon_r**2)

    img = torch.from_numpy(composite_crop).to(device=device, dtype=torch.float64)
    center = (float(mi_crop), float(mj_crop))
    radius_min = 0.0
    radius_max = _dist_to_corners(mi_crop, mj_crop, H_crop, W_crop)
    n_r = max(int(math.ceil(2 * (radius_max - radius_min))) + 1, 2)
    n_theta = max(int(math.ceil(4 * math.pi * radius_max)) + 1, 2)

    polar_img, valid_for_mean, valid = _polar_transform_and_extrapolate(img, center, radius_min, radius_max, n_r, n_theta)
    display_t = _radial_tone_map(img, polar_img, valid_for_mean, center, radius_min, radius_max, n_r, n_theta)
    display_t = _percentile_stretch(display_t, valid, center, radius_min, radius_max, n_r, n_theta)

    ctx.display = display_t.cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(ctx.display, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Radial normalize (polar, torch): mean/2→0.4, 2×mean→1; then (p3,1)→(0,1)")
    plt.tight_layout()
    plt.close(fig)
    Image.fromarray((ctx.display * 255).clip(0, 255).astype(np.uint8)).save(
        ctx.workdir / "v1-stage3_radial_normalize.png"
    )
    print(f"Saved {ctx.workdir / 'v1-stage3_radial_normalize.png'}")


def fft_unsharp_and_save(ctx: Stage3Context) -> None:
    """Sliding-patch FFT-smoothed unsharp (three σ); writes sharpened PNG and debug figure."""
    assert ctx.display is not None and ctx.moon_mask is not None
    inward_median_span = FFT_INWARD_MEDIAN_SPAN
    strength_r2, strength_r4, strength_r8 = UNSHARP_WEIGHTS
    a = PATCH_SIDE
    stride = PATCH_STRIDE
    dev = ctx.device
    display = ctx.display
    mi_crop, mj_crop, moon_r = ctx.mi_crop, ctx.mj_crop, ctx.moon_r

    display_for_blur = display

    diff_smooth_r2 = _sliding_diff_smooth_for_sigma(
        display_for_blur, display_for_blur, UNSHARP_GAUSSIAN_SIGMAS[0], inward_median_span, mi_crop, mj_crop, moon_r, dev, a, stride
    )
    diff_smooth_r4 = _sliding_diff_smooth_for_sigma(
        display_for_blur, display_for_blur, UNSHARP_GAUSSIAN_SIGMAS[1], inward_median_span, mi_crop, mj_crop, moon_r, dev, a, stride
    )
    diff_smooth_r8 = _sliding_diff_smooth_for_sigma(
        display_for_blur, display_for_blur, UNSHARP_GAUSSIAN_SIGMAS[2], inward_median_span, mi_crop, mj_crop, moon_r, dev, a, stride
    )

    combined_diff = strength_r2 * diff_smooth_r2 + strength_r4 * diff_smooth_r4 + strength_r8 * diff_smooth_r8
    sharpened = display + combined_diff
    sharpened = np.clip(sharpened, 0.0, 1.0)
    sharpened[ctx.moon_mask] = 0.0
    ctx.sharpened_fft_diff = sharpened

    out_png = ctx.workdir / "v1-stage3_radial_normalize_sharpen_fft_smoothed_diff.png"
    Image.fromarray((sharpened * 255).clip(0, 255).astype(np.uint8)).save(out_png)
    vlim_ds = float(np.percentile(np.abs(combined_diff), 99.0))
    vlim_ds = max(vlim_ds, 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(combined_diff, cmap="gray", vmin=-vlim_ds, vmax=vlim_ds, interpolation="nearest")
    axes[0].set_title(
        f"Combined FFT-smoothed diff: {strength_r2}*D(s=2)+{strength_r4}*D(s=4)+{strength_r8}*D(s=8)"
    )
    axes[1].imshow(sharpened, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("stage3 unsharp: display + combined diff")
    plt.tight_layout()
    plt.close(fig)
    print(f"Saved {out_png}")


def rgb_vignette_and_radial_pickle(ctx: Stage3Context) -> None:
    """RGB + vignette PNG and v1-stage3_radial.pkl sidecar."""
    assert ctx.sharpened_fft_diff is not None and ctx.moon_mask is not None #and ctx.p3_at is not None
    gray = np.clip(ctx.sharpened_fft_diff.astype(np.float64), 0.0, 1.0)
    R = gray
    G = 0.16 + 0.84 * gray
    B = 0.30 + 0.70 * gray
    display_rgb = np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0)

    H, W = display_rgb.shape[0], display_rgb.shape[1]
    ci, cj = float(ctx.mi_crop), float(ctx.mj_crop)
    sigma = max(
        np.hypot(ci - 0, cj - 0),
        np.hypot(ci - (H - 1), cj - 0),
        np.hypot(ci - 0, cj - (W - 1)),
        np.hypot(ci - (H - 1), cj - (W - 1)),
    )
    sigma = max(float(sigma), 1e-12)
    ii = np.arange(H, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(W, dtype=np.float64).reshape(1, -1)
    r = np.sqrt((ii - ci) ** 2 + (jj - cj) ** 2)
    g = np.exp(-0.5 * (r / sigma) ** 2)
    exp_half = np.exp(-0.5)
    qs = [(1.0 - dq) / (1.0 - exp_half) for dq in RGB_DIM_QUOTIENTS]
    scale = np.stack([1.0 - q * (1.0 - g) for q in qs], axis=-1)
    display_rgb = display_rgb * scale
    display_rgb = np.clip(display_rgb, 0.0, 1.0)
    ctx.display_rgb = display_rgb

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(display_rgb)
    ax.set_title("RGB: R as-is; G (0,1)→(0.08,1); B (0,1)→(0.1,1)")
    plt.tight_layout()
    plt.close(fig)

    out_png = ctx.workdir / "v1-stage3_rgb_rescaled.png"
    Image.fromarray((display_rgb * 255).round().clip(0, 255).astype(np.uint8)).save(out_png)
    print(f"Saved {out_png}")


def run(workdir: Path) -> None:
    """Run full stage 3 pipeline; writes v1-stage3_* artifacts under workdir."""
    ctx = Stage3Context(workdir=workdir)
    load_inputs(ctx)
    build_per_exposure_averages(ctx)
    warp_merge_to_composite(ctx)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)
