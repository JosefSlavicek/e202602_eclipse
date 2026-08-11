"""Stage 3: composite merge, radial processing, sharpen, RGB."""
from __future__ import annotations

import json
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

import eclipse_v6.stage0  # noqa: F401
from eclipse_v6.stage0 import find_moon
from eclipse_v6.coords import cartesian_to_polar, polar_to_cartesian
from eclipse_v6.merge import NO_DATA
from eclipse_v6.utils import load_grayscale, apply_transform_single, moon_median
from eclipse_v6.warp import (
    grid_to_normalized_grid as _grid_to_normalized_grid,
    ref_to_source_grid as _ref_to_source_grid,
)

# --- Parameters for the sliding-window FFT sharpen (visible from notebooks) ---
PATCH_SIDE = 256
PATCH_STRIDE = 8
FFT_INWARD_MEDIAN_SPAN = 32
UNSHARP_GAUSSIAN_SIGMAS = (2.0, 4.0, 8.0)
UNSHARP_WEIGHTS = (8.0, 8.0, 0.5)

VALID_THRESH = 0.9999
RGB_DIM_QUOTIENTS = (0.25, 0.28, 0.37)

# Cap on how far `_grow_moon_over_no_data` may push the fitted radius. The growth should be a
# few px (the blanked region is the union of disks spanning 5.94 px of detected radius); more
# than this means the no-data mask holds something that is not the moon.
MOON_GROW_MAX_PX = 16.0

# The composite is physical brightness now, while every constant downstream
# (`_radial_tone_map` breakpoints, `_percentile_stretch`, UNSHARP_WEIGHTS, RGB_DIM_QUOTIENTS)
# was tuned against v2's encoded composite. This is the one knob provided instead of
# retuning all of them; everything downstream works on ratios to a local mean or per-radius
# quantiles, so the normalisation is cosmetic and only the exponent matters.
#
# For the JPEG data the correct value is 1.0 — no compression. Do NOT derive it from first
# principles ("v2's composite was JPEG-compressed, so imitate that" gives ~1.63 and ruins the
# image): v2 scaled exposures by (t_ref/t_k)^(1/g) with g ~ 1.1, which is nearly the plain
# exposure ratio, so its composite was already nearly proportional to brightness. Measured
# spread (p99/p50) was 167 for v2 against 233 uncompressed, i.e. an exponent of 1.065.
# `report_display_exponent.py` re-measures it.
DISPLAY_GAMMA = 1.0
DISPLAY_NORM_PERCENTILE = 99.9


@dataclass
class Stage3Context:
    """Mutable state passed through stepped stage-3 functions (notebooks) or `run` (one-shot)."""

    workdir: Path                                              # root output directory; set at construction
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))  # CUDA device; set at construction
    exposure_groups: Any = None                                # exposure_time -> [ImageInfo]; loaded by load_inputs
    reg: Any = None                                            # pairwise intra-exposure registration dict from stage0; loaded by load_inputs
    opt_results: Any = None                                    # per-exposure pose optimization (abs_xy, abs_angle_t) from stage1; loaded by load_inputs
    cross_reg: Any = None                                      # (t0,t1) -> (shift_i, shift_j, rotation) cross-exposure reg from stage2; loaded by load_inputs
    refined_registration: bool = False                         # True when cross_reg came from v6-stage2r.pkl (redone on calibrated radiance); set by load_inputs
    exposure_times_sorted: list[Any] = field(default_factory=list)  # sorted exposure times (shortest two dropped); set by load_inputs
    t_ref: Any = None                                          # reference exposure time (shortest kept); set by load_inputs
    moon_ref: tuple[float, float, float] | None = None         # (i, j, radius) median moon in reference exposure; set by load_inputs
    H_ref: int = 0                                             # pixel height of the reference image; set by merge.merge_to_composite
    W_ref: int = 0                                             # pixel width of the reference image; set by merge.merge_to_composite
    composite: Optional[np.ndarray] = None                     # weighted merge of all exposures in ref coords, physical brightness with NO_DATA holes (merge.merge_to_composite); freed after crop_and_save_composite
    composite_variance: Optional[np.ndarray] = None            # per-pixel variance of `composite`, inf at NO_DATA; set by merge.merge_to_composite only
    no_data_mask: Optional[np.ndarray] = None                  # pixels no exposure could measure (moon disk + saturated-everywhere); set by merge.merge_to_composite
    calib: Any = None                                          # CalibResult driving the radiometry; set by load_calibration
    valid_all: Optional[np.ndarray] = None                     # minimum valid coverage map across exposures; set by the merge
    r_lo: int = 0                                              # top crop row (mutual-coverage boundary + 16 px margin); set by crop_and_save_composite
    r_hi: int = 0                                              # bottom crop row; set by crop_and_save_composite
    c_lo: int = 0                                              # left crop column; set by crop_and_save_composite
    c_hi: int = 0                                              # right crop column; set by crop_and_save_composite
    composite_crop: Optional[np.ndarray] = None                # display-ready composite sliced to crop bounds (NO_DATA filled, display scaling applied); set by crop_and_save_composite
    radiance_crop: Optional[np.ndarray] = None                 # physical brightness sliced to crop bounds, NO_DATA preserved; set by crop_and_save_composite
    variance_crop: Optional[np.ndarray] = None                 # variance sliced to crop bounds; set by crop_and_save_composite
    no_data_crop: Optional[np.ndarray] = None                  # no_data_mask sliced to crop bounds; set by crop_and_save_composite
    mi_crop: float = 0.0                                       # moon center row in crop coordinates; set by crop_and_save_composite, refined by radial_normalize_display
    mj_crop: float = 0.0                                       # moon center column in crop coordinates; set by crop_and_save_composite, refined by radial_normalize_display
    moon_r: float = 0.0                                        # moon radius in pixels, grown to cover the blanked region; set by crop_and_save_composite, refined and grown by radial_normalize_display
    moon_r_fitted: float = 0.0                                 # radius find_moon actually fitted, before the growth; set by radial_normalize_display
    moon_mask: Optional[np.ndarray] = None                     # boolean mask of the grown moon disk — exactly the region blanked in `display` and the complement of the FFT protection indicator; set by radial_normalize_display
    H_crop: int = 0                                            # pixel height of composite_crop; set by crop_and_save_composite
    W_crop: int = 0                                            # pixel width of composite_crop; set by crop_and_save_composite
    display: Optional[np.ndarray] = None                       # radially tone-mapped grayscale image in [0,1]; set by radial_normalize_display
    sharpened_fft_diff: Optional[np.ndarray] = None            # display + FFT-smoothed unsharp signal, moon zeroed; set by fft_unsharp_and_save
    display_rgb: Optional[np.ndarray] = None                   # RGB image with vignette, ready for export; set by rgb_vignette_and_radial_pickle


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


def _patch_window_flat_circle(a, dtype=np.float32):
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
    ii = np.arange(H, dtype=np.float32)[:, None]
    jj = np.arange(W, dtype=np.float32)[None, :]
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
    del polar  # only polar_h is read from here on

    i_lim = int(math.ceil(i_cont))
    i_lim_b = int(min(i_lim, int(n_r)))
    if i_lim_b > 0:
        j_idx = torch.arange(0, int(n_r), device=device, dtype=dtype)
        r_j = radius_max - j_idx * (radius_max - radius_min) / denom_r
        src_ok_row = (r_j >= float(moon_r)).to(dtype).view(1, -1)
        j_cols = j_idx.view(1, -1)
        sp_t = torch.tensor(sigma_polar, device=device, dtype=dtype)
        polar_v = polar_h.clone()
        # Chunk over output rows: build only this chunk's [chunk, n_r] weight
        # slice and matmul it, instead of materializing the full [i_lim_b, n_r]
        # all-pairs W_vert. Weight values are identical to the unchunked build
        # (elementwise + per-row reductions); only the matmul's fp32 reduction
        # order may differ, well below the uint8 output quantization.
        row_chunk = 4096
        for s in range(0, i_lim_b, row_chunk):
            e = min(s + row_chunk, i_lim_b)
            i_rows = torch.arange(s, e, device=device, dtype=dtype).view(-1, 1)
            delta = j_cols - i_rows
            W_chunk = torch.exp(-0.5 * (delta / sp_t.clamp(min=1e-20)) ** 2)
            W_chunk = W_chunk * src_ok_row
            row_sum = W_chunk.sum(dim=1, keepdim=True)
            W_chunk = W_chunk / row_sum.clamp(min=1e-20)
            bad = (row_sum.squeeze(1) < 1e-20) | (sp_t < 1e-8)
            if bool(bad.any().item()):
                bi = torch.nonzero(bad, as_tuple=False).squeeze(-1)
                W_chunk[bi, :] = 0
                W_chunk[bi, s + bi] = 1.0  # diagonal: global col == global row
            polar_v[s:e, :] = W_chunk @ polar_h
            del W_chunk, delta, row_sum
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
    diff_np = (display.astype(np.float32) - blurred.astype(np.float32))
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


def load_inputs(ctx: Stage3Context, use_refined_registration: bool = True) -> None:
    """Load stage1/stage2 pickles; set exposure list, reference exposure, and moon prior.

    `use_refined_registration` swaps in `v6-stage2r.pkl` — the cross-exposure transforms
    redone on calibrated radiance — when the pipeline produced one and `--refine-registration`
    didn't turn it off. Falls back to stage 2's gamma-scaled alignment otherwise.
    """
    with open(ctx.workdir / "v6-stage1.pkl", "rb") as fd:
        ctx.exposure_groups = pickle.load(fd)
        ctx.reg = pickle.load(fd)
        ctx.opt_results = pickle.load(fd)
    with open(ctx.workdir / "v6-stage2.pkl", "rb") as fd:
        ctx.cross_reg = pickle.load(fd)
        # gamma_by_pair, also in this pickle, is stage 2's own alignment bootstrap — not read
        # from here on.
    refined_pkl = ctx.workdir / "v6-stage2r.pkl"
    if use_refined_registration:
        if refined_pkl.exists():
            from eclipse_v6 import reregister as RR

            ctx.cross_reg, _rows = RR.load(refined_pkl)
            ctx.refined_registration = True
            print(f"Using refined cross-exposure registration from {refined_pkl}")
        else:
            print(f"NOTE: {refined_pkl} not found — falling back to stage 2's gamma-scaled "
                  f"alignment. Re-run from --start-stage 3 to produce it.")
    ctx.device = torch.device("cuda")
    # Inherited from v2: the two shortest exposures are dropped, so 15 of 17 are used and
    # t_ref = 0.001. No reason was ever recorded; kept for comparability with v2's numbers.
    ctx.exposure_times_sorted = sorted(ctx.exposure_groups.keys())[2:]
    ctx.t_ref = ctx.exposure_times_sorted[0]
    ctx.moon_ref = moon_median(ctx.exposure_groups[ctx.t_ref])
    print(f"Reference exposure t_ref={ctx.t_ref}, moon_ref (i,j,r)={ctx.moon_ref}")
    print(f"cross_reg pairs: {len(ctx.cross_reg)}")


def fill_no_data(composite, no_data=None, fill: float = 0.0):
    """Replace the merge's NO_DATA sentinel with `fill` (0.0, matching v2's blanked moon).

    Returns the input unchanged when there is nothing to fill.
    """
    if no_data is None:
        no_data = composite <= NO_DATA
        if not no_data.any():
            return composite
    elif not no_data.any():
        return composite
    out = composite.copy()
    out[no_data] = fill
    return out


def composite_for_display(composite, gamma: float = None, no_data=None):
    """Normalise brightness for the display chain. `gamma == 1.0` is an exact no-op.

    One knob instead of retuning `_radial_tone_map`, `_percentile_stretch`, UNSHARP_WEIGHTS
    and RGB_DIM_QUOTIENTS, all of which were fitted against v2's composite. See DISPLAY_GAMMA
    for why the right value on the JPEG data is 1.0 and why deriving it from first principles
    is wrong.
    """
    gamma = DISPLAY_GAMMA if gamma is None else float(gamma)
    if gamma == 1.0:
        return composite                       # exact: not even the normalisation is applied
    sample = composite if no_data is None else composite[~no_data]
    ref = float(np.percentile(sample, DISPLAY_NORM_PERCENTILE))
    assert ref > 0, ref
    return (np.clip(composite, 0.0, None) / ref) ** (1.0 / gamma)


def brightness_spread(composite, no_data=None, moon_mask=None) -> float:
    """p99 / p50 of the brightness — the single number the display exponent is fitted against."""
    sel = np.ones(composite.shape, dtype=bool)
    if no_data is not None:
        sel &= ~no_data
    if moon_mask is not None:
        sel &= ~moon_mask
    sel &= composite > 0
    vals = composite[sel]
    assert vals.size, "no positive, measured pixel to take a spread over"
    p50, p99 = np.percentile(vals, [50.0, 99.0])
    return float(p99 / max(p50, 1e-20))


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
    radiance_crop = composite[r_lo : r_hi + 1, c_lo : c_hi + 1].copy()
    ctx.composite = None
    if ctx.composite_variance is not None:
        ctx.variance_crop = ctx.composite_variance[r_lo : r_hi + 1, c_lo : c_hi + 1].copy()
        ctx.composite_variance = None
    if ctx.no_data_mask is not None:
        ctx.no_data_crop = ctx.no_data_mask[r_lo : r_hi + 1, c_lo : c_hi + 1].copy()
        ctx.no_data_mask = None

    mi_crop = mi - r_lo
    mj_crop = mj - c_lo
    ctx.H_crop, ctx.W_crop = radiance_crop.shape
    print(f"Crop bounds rows [{r_lo},{r_hi}], cols [{c_lo},{c_hi}]; shape {radiance_crop.shape}")

    ii = np.arange(ctx.H_crop, dtype=np.float32).reshape(-1, 1)
    jj = np.arange(ctx.W_crop, dtype=np.float32).reshape(1, -1)
    dist_sq = (ii - mi_crop) ** 2 + (jj - mj_crop) ** 2
    moon_mask_preview = dist_sq <= (moon_r0**2)

    ctx.radiance_crop = radiance_crop
    np.save(out_dir / "v6-stage3_composite.npy", radiance_crop)
    saved = [out_dir / "v6-stage3_composite.npy"]
    if ctx.variance_crop is not None:
        np.save(out_dir / "v6-stage3_variance.npy", ctx.variance_crop)
        saved.append(out_dir / "v6-stage3_variance.npy")

    # Hand-off to the display chain. Everything downstream — find_moon, the polar
    # extrapolation, the p3 stretch, and the limb-protection machinery in the FFT unsharp —
    # was written against v2's composite, which held 0.0 inside the moon because every
    # exposure was multiplied by `mask_0`. The radiometric merge writes NO_DATA (-1.0) there
    # instead, so fill it back to 0.0 before anything reads percentiles or hunts for the
    # disk; leaving a negative sentinel in would skew every quantile and confuse `find_moon`.
    composite_crop = fill_no_data(radiance_crop, ctx.no_data_crop)
    composite_crop = composite_for_display(composite_crop, no_data=ctx.no_data_crop)

    valid_preview = ~moon_mask_preview
    if ctx.no_data_crop is not None:
        valid_preview = valid_preview & ~ctx.no_data_crop
    sample = composite_crop[valid_preview] if np.any(valid_preview) else composite_crop
    v_min = np.percentile(sample, 1)
    v_max = np.percentile(sample, 99)
    preview = np.clip((composite_crop - v_min) / (v_max - v_min + 1e-9), 0, 1)
    Image.fromarray((preview * 255).clip(0, 255).astype(np.uint8)).save(
        out_dir / "v6-stage3_composite_preview.png"
    )
    print("Saved " + ", ".join(str(p) for p in saved)
          + f", {out_dir / 'v6-stage3_composite_preview.png'}")

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


def _grow_moon_over_no_data(dist_sq, no_data, moon_r_fitted):
    """Grow the fitted moon circle until it covers every blanked pixel. Returns the radius.

    `find_moon` fits a circle to the boundary of the region the merge blanked, but that region
    is a pixel map, not a circle: the union of ~15 per-frame disks whose centres differ by
    registration residuals and whose radii span 5.94 px, bilinear-resampled into reference
    coordinates and thresholded. Parts of it stick out past the fitted circle.

    Everything downstream models the black area as exactly `dist^2 <= moon_r^2` — the FFT limb
    protection builds its indicator from the complement of that predicate, the anisotropic blur
    drops source radii below `moon_r`, and the final blackening uses the disk itself. A pixel
    that is blank but outside the circle is therefore an unmodelled step edge in the one place
    the protection exists to guard, and its zeros leak into the blur. Growing the radius to
    cover them (and blanking the annulus this gains, in `radial_normalize_display`) makes the
    three agree on one pixel set.

    Assumes the blanked region is the moon disk alone. It can also hold pixels no exposure
    could measure for other reasons — every exposure saturated or masked — and a single one of
    those elsewhere in the frame would inflate the radius without bound, so the growth is
    capped and exceeding the cap is fatal rather than silently clamped.
    """
    if no_data is None or not no_data.any():
        return float(moon_r_fitted)
    # +1e-3 px: `dist_sq` is float32 and the mask is rebuilt as `dist_sq <= moon_r**2`, so a
    # bare sqrt can land one ULP low and drop the very pixel that set the radius.
    r_needed = math.sqrt(float(dist_sq[no_data].max())) + 1e-3
    grown = max(float(moon_r_fitted), r_needed)
    growth = grown - float(moon_r_fitted)
    if growth > MOON_GROW_MAX_PX:  # a gap that is not the moon; see the docstring
        raise AssertionError(
            f"blanked region reaches {r_needed:.2f} px from the moon centre, {growth:.2f} px "
            f"past the fitted radius {moon_r_fitted:.2f} (cap {MOON_GROW_MAX_PX} px). The "
            f"no-data mask is assumed to be the moon disk alone; this looks like a coverage "
            f"gap elsewhere in the frame — check the 'coverage gaps outside moon' count "
            f"printed by merge_to_composite."
        )
    if growth > 0:
        print(f"Moon circle grown {growth:.2f} px to cover the blanked region: "
              f"r {moon_r_fitted:.2f} -> {grown:.2f}")
    else:
        print(f"Moon circle r={moon_r_fitted:.2f} already covers the blanked region "
              f"(reaches {r_needed:.2f} px); not grown")
    return grown


def _radial_tone_map(img, polar_img, valid_for_mean, center, radius_min, radius_max, n_r, n_theta):
    """Sliding-window polar mean → piecewise-linear tone map; average over two angular window sizes."""
    device = img.device
    dtype = img.dtype
    H_crop, W_crop = img.shape[:2]
    display_ts = []
    valid_bin = (valid_for_mean > 0.5).to(dtype)
    for row_fraction in [0.15, 1.0]:
        n_cols_use = max(1, int(n_theta * row_fraction))
        if n_cols_use >= n_theta:
            # Full-row circular window: mean is constant along theta — skip padding/pooling.
            val_sum = (polar_img * valid_bin).sum(dim=1, keepdim=True)
            cnt_sum = valid_bin.sum(dim=1, keepdim=True).clamp(min=1e-20)
            mean_polar_2d = (val_sum / cnt_sum).expand(n_r, n_theta).contiguous()
        else:
            half_window = n_cols_use // 2
            right_pad = n_cols_use - half_window - 1
            pv_ext = F.pad((polar_img * valid_bin).unsqueeze(1), (half_window, right_pad), mode='circular')
            mean_v = F.avg_pool1d(pv_ext, kernel_size=n_cols_use, stride=1).squeeze(1)
            del pv_ext
            vb_ext = F.pad(valid_bin.unsqueeze(1), (half_window, right_pad), mode='circular')
            mean_n = F.avg_pool1d(vb_ext, kernel_size=n_cols_use, stride=1).squeeze(1).clamp(min=1e-20)
            del vb_ext
            mean_polar_2d = mean_v / mean_n
            del mean_v, mean_n
        assert torch.all(torch.isfinite(mean_polar_2d))
        argmax = mean_polar_2d.argmax(dim=0)
        max_val = mean_polar_2d.max(dim=0).values
        mask = torch.arange(mean_polar_2d.size(0), device=mean_polar_2d.device).unsqueeze(1) > argmax
        mean_polar_2d[mask] = max_val.unsqueeze(0).expand_as(mean_polar_2d)[mask]

        mean_at = polar_to_cartesian(mean_polar_2d, center, radius_min, radius_max, H_crop, W_crop)
        del mean_polar_2d
        valid_mask = torch.isfinite(mean_at) & (mean_at > 0)
        display_t = torch.zeros_like(img, device=device, dtype=dtype)
        v = img[valid_mask]
        m_ = mean_at[valid_mask]
        del mean_at
        mask1 = (v > m_ / 2) & (v <= m_)
        mask2 = (v > m_) & (v <= 2 * m_)
        mask3 = v > 2 * m_
        display_t[valid_mask] = torch.where(mask1, 0.4 * (v - m_ / 2) / (m_ / 2).clamp(min=1e-9), torch.zeros_like(v))
        display_t[valid_mask] = torch.where(
            mask2, 0.4 + 0.6 * (v - m_) / m_.clamp(min=1e-9), display_t[valid_mask]
        )
        display_t[valid_mask] = torch.where(mask3, torch.ones_like(v), display_t[valid_mask])
        display_ts.append(display_t)
        torch.cuda.empty_cache()
    del valid_bin
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
    # Equivalent to polar_display.quantile(q=q, dim=1).diag() (linear interp, the
    # torch.quantile default), but avoids materializing the [n_r, n_r] all-pairs
    # matrix that .diag() immediately discards.
    #
    # sort() over the full [n_r, n_theta] tensor also allocates an int64 index
    # tensor (2x the float32 payload) that we discard, so at full-sensor (raw)
    # resolution the peak is ~3x polar_display and OOMs. Chunk over rows to bound
    # the sort's working set; results are identical to a single full sort.
    n_cols = polar_display.shape[1]
    n_rows = polar_display.shape[0]
    pos = q * (n_cols - 1)
    lo = pos.floor().long().clamp(max=n_cols - 1)
    up = (lo + 1).clamp(max=n_cols - 1)
    frac = pos - lo.to(pos.dtype)
    p3_row = torch.empty(n_rows, device=polar_display.device, dtype=polar_display.dtype)
    row_chunk = 4096
    for s in range(0, n_rows, row_chunk):
        e = min(s + row_chunk, n_rows)
        sorted_row, _ = polar_display[s:e].sort(dim=1)
        ri = torch.arange(e - s, device=sorted_row.device)
        v_lo = sorted_row[ri, lo[s:e]]
        v_up = sorted_row[ri, up[s:e]]
        p3_row[s:e] = v_lo + frac[s:e] * (v_up - v_lo)
        del sorted_row, v_lo, v_up
    p3_smooth = _vertical_gaussian_blur(p3_row.unsqueeze(1), kernel_size=133, sigma=33).squeeze(1)
    p3_polar_2d = p3_smooth.unsqueeze(1).expand(n_r, n_theta)
    p3_at = torch.nan_to_num(
        polar_to_cartesian(p3_polar_2d, center, radius_min, radius_max, H_crop, W_crop), nan=0.0
    )
    span = (1.0 - p3_at).clamp(min=1e-9)
    return ((display_t - p3_at) / span).clamp(0.0, 1.0)


def radial_normalize_display(ctx: Stage3Context) -> None:
    """Refine and grow moon; polar radial tone + p3 stretch → ctx.display (grayscale [0,1], moon blanked).

    The growth (`_grow_moon_over_no_data`) affects nothing in the normalisation itself — the
    polar pole is the moon *centre*, `radius_min` is 0, and the extrapolation keys off validity
    rather than radius, so `moon_r` is read here only to build `ctx.moon_mask`. It matters to
    the sharpener, which is why the disk is blanked in `ctx.display` on the way out.
    """
    assert ctx.composite_crop is not None
    device = ctx.device
    composite_crop = ctx.composite_crop
    H_crop, W_crop = ctx.H_crop, ctx.W_crop
    mi_crop, mj_crop = ctx.mi_crop, ctx.mj_crop

    img_rgb = torch.from_numpy(composite_crop).to(device=device, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 3)
    mi_crop, mj_crop, moon_r = find_moon(img_rgb, float(mi_crop), float(mj_crop))
    ctx.mi_crop, ctx.mj_crop = float(mi_crop), float(mj_crop)
    ctx.moon_r_fitted = float(moon_r)

    ii = np.arange(H_crop, dtype=np.float32).reshape(-1, 1)
    jj = np.arange(W_crop, dtype=np.float32).reshape(1, -1)
    dist_sq = (ii - mi_crop) ** 2 + (jj - mj_crop) ** 2
    moon_r = _grow_moon_over_no_data(dist_sq, ctx.no_data_crop, moon_r)
    ctx.moon_r = float(moon_r)
    ctx.moon_mask = dist_sq <= (moon_r**2)
    if ctx.no_data_crop is not None:
        # The point of the growth: the disk the sharpener models must contain every blanked
        # pixel, or the ones left out are unprotected step edges right at the limb.
        assert not np.any(ctx.no_data_crop & ~ctx.moon_mask), (
            f"{int(np.sum(ctx.no_data_crop & ~ctx.moon_mask))} blanked pixels outside the "
            f"grown moon disk r={moon_r:.4f}"
        )

    img = torch.from_numpy(composite_crop).to(device=device, dtype=torch.float32)
    center = (float(mi_crop), float(mj_crop))
    radius_min = 0.0
    radius_max = _dist_to_corners(mi_crop, mj_crop, H_crop, W_crop)
    n_r = max(int(math.ceil(2 * (radius_max - radius_min))) + 1, 2)
    n_theta = max(int(math.ceil(4 * math.pi * radius_max)) + 1, 2)

    polar_img, valid_for_mean, valid = _polar_transform_and_extrapolate(img, center, radius_min, radius_max, n_r, n_theta)
    display_t = _radial_tone_map(img, polar_img, valid_for_mean, center, radius_min, radius_max, n_r, n_theta)
    del polar_img, valid_for_mean
    torch.cuda.empty_cache()
    display_t = _percentile_stretch(display_t, valid, center, radius_min, radius_max, n_r, n_theta)

    ctx.display = display_t.cpu().numpy()
    # Blank the grown disk in the one array the sharpener reads, so the black region it sees is
    # exactly the circle its limb protection models — the same predicate, quantization included.
    # `composite_crop` is deliberately not touched: nothing reads it after this function, and
    # when the merge left no holes to fill it is the *same object* as `radiance_crop`, the
    # physical brightness already written to v6-stage3_composite.npy.
    ctx.display[ctx.moon_mask] = 0.0
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(ctx.display, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Radial normalize (polar, torch): mean/2→0.4, 2×mean→1; then (p3,1)→(0,1)")
    plt.tight_layout()
    plt.close(fig)
    Image.fromarray((ctx.display * 255).clip(0, 255).astype(np.uint8)).save(
        ctx.workdir / "v6-stage3_radial_normalize.png"
    )
    print(f"Saved {ctx.workdir / 'v6-stage3_radial_normalize.png'}")

    # Full-precision state for `fft_unsharp_and_save`/`rgb_vignette_and_radial_pickle`, so a
    # later run can resume right before the sharpen without redoing the merge. The PNG above
    # is 8-bit and lossy; `ctx.moon_mask` and the crop-space moon geometry aren't saved
    # anywhere else.
    np.save(ctx.workdir / "v6-stage3_display.npy", ctx.display)
    np.save(ctx.workdir / "v6-stage3_moon_mask.npy", ctx.moon_mask)
    with open(ctx.workdir / "v6-stage3_radial_state.json", "w") as fd:
        json.dump({
            "mi_crop": ctx.mi_crop, "mj_crop": ctx.mj_crop,
            "moon_r": ctx.moon_r, "moon_r_fitted": ctx.moon_r_fitted,
        }, fd, indent=2)
    print(f"Saved {ctx.workdir / 'v6-stage3_display.npy'}, "
          f"{ctx.workdir / 'v6-stage3_moon_mask.npy'}, "
          f"{ctx.workdir / 'v6-stage3_radial_state.json'}")


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
    torch.cuda.empty_cache()
    diff_smooth_r4 = _sliding_diff_smooth_for_sigma(
        display_for_blur, display_for_blur, UNSHARP_GAUSSIAN_SIGMAS[1], inward_median_span, mi_crop, mj_crop, moon_r, dev, a, stride
    )
    torch.cuda.empty_cache()
    diff_smooth_r8 = _sliding_diff_smooth_for_sigma(
        display_for_blur, display_for_blur, UNSHARP_GAUSSIAN_SIGMAS[2], inward_median_span, mi_crop, mj_crop, moon_r, dev, a, stride
    )

    combined_diff = strength_r2 * diff_smooth_r2 + strength_r4 * diff_smooth_r4 + strength_r8 * diff_smooth_r8
    sharpened = display + combined_diff
    sharpened = np.clip(sharpened, 0.0, 1.0)
    # `display` is already blanked here; this removes what the unsharp put back inside the disk
    # (the protected limb coefficients reconstruct the step at gain 1, then get multiplied by
    # the UNSHARP_WEIGHTS).
    sharpened[ctx.moon_mask] = 0.0
    ctx.sharpened_fft_diff = sharpened

    out_png = ctx.workdir / "v6-stage3_radial_normalize_sharpen_fft_smoothed_diff.png"
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
    """RGB + vignette PNG and v6-stage3_radial.pkl sidecar."""
    assert ctx.sharpened_fft_diff is not None and ctx.moon_mask is not None #and ctx.p3_at is not None
    gray = np.clip(ctx.sharpened_fft_diff.astype(np.float32), 0.0, 1.0)
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
    ii = np.arange(H, dtype=np.float32).reshape(-1, 1)
    jj = np.arange(W, dtype=np.float32).reshape(1, -1)
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

    out_png = ctx.workdir / "v6-stage3_rgb_rescaled.png"
    Image.fromarray((display_rgb * 255).round().clip(0, 255).astype(np.uint8)).save(out_png)
    print(f"Saved {out_png}")


def load_calibration(ctx: Stage3Context, source=None, path: Path = None):
    """Attach the response calibration to `ctx` and, if given, to the frame source."""
    from eclipse_v6 import calib as CA

    path = (ctx.workdir / "v6-calib.pkl") if path is None else Path(path)
    ctx.calib = CA.load(path)
    print(f"Loaded calibration {path}: {len(ctx.calib.exposures)} exposures, "
          f"corrections {ctx.calib.corrections.min():.4f}..{ctx.calib.corrections.max():.4f}")
    if source is not None:
        source.set_calibration(ctx.calib)
    return ctx.calib


def run(workdir: Path, source=None) -> None:
    """Run full stage 3 pipeline; writes v6-stage3_* artifacts under workdir.

    Needs `source` for the radiometric merge (it reads frames through `load_radiance`) and a
    `v6-calib.pkl` alongside the stage pickles.
    """
    from eclipse_v6.merge import merge_to_composite

    ctx = Stage3Context(workdir=workdir)
    load_inputs(ctx)
    assert source is not None, "the radiometric merge needs the frame source"
    load_calibration(ctx, source)
    merge_to_composite(ctx, source)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)
