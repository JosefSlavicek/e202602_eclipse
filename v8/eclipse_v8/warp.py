"""Reference-grid warping and the cross-exposure transform chain.

Shared by the merge (`merge.py`) and the response-curve calibration (`calib.py`), so
there's only one copy of the chain composition order and the sign convention of
`rotation_deg` -- the only thing tying an exposure's pixels to the reference exposure's
pixels. Two divergent copies would silently disagree.
"""
from __future__ import annotations

import math

import torch


def ref_to_source_grid(H, W, chain_tuples, device):
    """Sampling coordinates in the source image for every pixel of the (H, W) reference grid."""
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


def grid_to_normalized_grid(i_src, j_src, H, W):
    j_norm = 2.0 * j_src / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_src)
    i_norm = 2.0 * i_src / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_src)
    return torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)


def warp_to_ref(img, chain_tuples, H_ref, W_ref, device):
    """Resample `img` onto the reference grid through `chain_tuples` (bilinear, zero-padded)."""
    i_src, j_src = ref_to_source_grid(H_ref, W_ref, chain_tuples, device)
    grid = grid_to_normalized_grid(i_src, j_src, img.shape[0], img.shape[1])
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.squeeze(0).squeeze(0)


def sample_at_ref_points(img, chain_tuples, i_ref, j_ref, H_ref, W_ref, device):
    """Bilinear value of `img` at the given reference-grid points only.

    Same map as `warp_to_ref`, evaluated at an (N,) scatter of reference pixels instead of
    the whole raster — used by the calibration, which wants a few thousand pixels and would
    otherwise pay for a full-resolution warp per exposure just to throw it away.
    """
    assert i_ref.ndim == 1 and j_ref.shape == i_ref.shape, (i_ref.shape, j_ref.shape)
    ci, cj = H_ref / 2.0, W_ref / 2.0
    i_cur = i_ref.to(device=device, dtype=torch.float32).clone()
    j_cur = j_ref.to(device=device, dtype=torch.float32).clone()
    for (shift_i, shift_j, rotation_deg) in chain_tuples:
        angle_rad = math.radians(-rotation_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        di = i_cur - ci - shift_i
        dj = j_cur - cj - shift_j
        i_cur = di * cos_a + dj * sin_a + ci
        j_cur = -di * sin_a + dj * cos_a + cj
    H, W = img.shape
    j_norm = 2.0 * j_cur / (W - 1) - 1.0 if W > 1 else torch.zeros_like(j_cur)
    i_norm = 2.0 * i_cur / (H - 1) - 1.0 if H > 1 else torch.zeros_like(i_cur)
    grid = torch.stack([j_norm, i_norm], dim=-1).view(1, 1, -1, 2)
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0).unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return out.view(-1)


def build_chains(exposure_times_sorted, available_exposures, cross_reg, t_ref):
    """Per-exposure transform chain onto the reference grid.

    `available_exposures` is the set/dict of exposures that actually have an averaged image.
    An exposure is usable only if every consecutive link between it and `t_ref` is present in
    `cross_reg`; a missing link truncates that exposure (and only it) out of the run, exactly
    as the v2 merge did.

    Returns `(exposures_with_chain, chain_tuples_by_exp)` with `t_ref` first and its chain empty.
    """
    assert exposure_times_sorted[0] == t_ref, (exposure_times_sorted[0], t_ref)
    exposures_with_chain = [t_ref]
    chain_tuples_by_exp = {t_ref: []}
    for k in range(1, len(exposure_times_sorted)):
        t_k = exposure_times_sorted[k]
        if t_k not in available_exposures:
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
    return exposures_with_chain, chain_tuples_by_exp
