"""Combine the bracket in physical brightness, weighted by a per-exposure brightness window.

Replaces `stage3.warp_merge_to_composite`, which rescaled *encoded* values down a chain of
fitted per-pair exponents and averaged them under a hand-shaped bell weight on the *stored*
value. Here every exposure is converted to brightness independently through the calibrated
response (`source.load_radiance`), and each exposure group's cross-exposure weight is a
smooth window over its own brightness `Lbar`, clamped to `[0, 1/t_eff]`: full trust in the
middle of that range, tapering to 0 near both ends. Long exposures still dominate the faint
outer corona and short exposures the bright inner corona, because each exposure's window is
only ever full-weight for the brightness range that exposure resolves well — but the weight
curve itself is a fixed hand-shaped window (`_window`, `MERGE_WINDOW_PLATEAU`), not fitted or
derived from a noise model.

Two things are deliberately kept from v2:

  * **Common moon blanking.** The detected moon radius drifts 5.94 px across the bracket
    (316.03 px at 1/4000 s down to 310.19 px at 2 s) because glare in long exposures makes
    the edge-finder place the limb further in. Every exposure is blanked to the union of the
    disks — v2 achieved this by multiplying everything by the shortest exposure's mask.
    Without it, the annulus between the smallest and largest disk is populated only by
    saturated long exposures.
  * **One exposure at a time.** 15 exposures x 2 arrays at 4000x6000 float32 is ~2.9 GB on
    the GPU. Build, warp, accumulate, free.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import tqdm

from eclipse_v6.utils import apply_transform_single
from eclipse_v6.warp import build_chains, warp_to_ref

NO_DATA = -1.0               # composite sentinel: no exposure could see this pixel
MOON_MARGIN_PX = 2.0         # same limb margin as utils.compute_weighted_average
MIN_STACK_COVER = 0.5        # a merged pixel needs half a frame's worth of valid contributions
REF_COVER_THRESH = 0.999     # inside the reference-grid footprint of this exposure
MOON_BLANK_THRESH = 0.999    # below this, some exposure called the pixel "moon"
MERGE_WINDOW_PLATEAU = 0.5   # fraction of [0, 1/t_eff] held at full weight; the outer
                              # quarters taper smoothly to 0 (Tukey-style) instead of a hard
                              # cutoff, so no fixed-brightness threshold shows up as a ring
MERGE_WINDOW_EPS = 1.0e-4    # weight floor added to every _window output so it is never
                              # exactly 0 at/beyond lo or hi, only ever small
EPS = 1e-20


def _moon_out_mask(ii, H, W, device):
    """1.0 outside this frame's moon disk (plus a 2 px margin), 0.0 inside."""
    mi, mj, r = ii.moon[0], ii.moon[1], ii.moon[2]
    rows = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    dist = torch.sqrt((rows - mi) ** 2 + (cols - mj) ** 2)
    return (dist > r + MOON_MARGIN_PX).to(torch.float32)


def _window(x: torch.Tensor, lo: float, hi: float, plateau: float = MERGE_WINDOW_PLATEAU) -> torch.Tensor:
    """Continuous weight over [lo, hi]: 1.0 on the middle `plateau` fraction, cosine taper
    down to (but never below) `MERGE_WINDOW_EPS` at lo/hi and beyond — the merge weight
    itself, not a noise model. The epsilon floor keeps `sum_w` from landing on exactly 0.0
    at a pixel purely because every exposure's Lbar happened to clamp to lo or hi there.
    """
    edge = (1.0 - plateau) / 2.0
    u = (x - lo) / (hi - lo)
    ramp = 0.5 * (1.0 - torch.cos(math.pi * (u / edge).clamp(0.0, 1.0)))
    fall = 0.5 * (1.0 - torch.cos(math.pi * ((1.0 - u) / edge).clamp(0.0, 1.0)))
    w = torch.minimum(ramp, fall)
    return torch.where((u > 0.0) & (u < 1.0), w, torch.zeros_like(w)) + MERGE_WINDOW_EPS


def average_exposure_radiance(group, abs_xy, abs_angle_t, source, device):
    """Stack one exposure group in brightness. Returns (Lbar, Vbar, covered, moon_out).

    Converting each frame and then averaging the brightnesses — rather than averaging the
    stored values and converting once, as v2 did — matters because `f` is curved: the
    average of encoded values is not the encoding of the average.

        Lbar = sum(m*L) / sum(m)          Vbar = sum(m^2*var) / sum(m)^2

    which is the ordinary variance of a weighted mean for *any* nonnegative weights `m`, not
    only 0/1 ones — so a group with more frames, or with frames trusted more (source.valid's
    continuous taper), earns proportionally more weight with nothing extra to say about it.

    `m` (the trust weight, `source.valid`) and `m_geom` (the coverage mask, `source.usable`)
    are deliberately kept separate. If `covered` were built from the same taper that weights
    `Lbar`, a pixel every frame genuinely covers but that merely sits in the taper's rolloff
    would read as "under-covered" and get killed by `MIN_STACK_COVER` in merge_to_composite —
    reintroducing, at the taper's midpoint, exactly the sharp edge the taper exists to remove.
    `covered` must answer "did enough frames see this pixel at all", independent of how much
    any of them is trusted.

    `moon_out` is returned separately from `covered` because `covered` also excludes
    unusable (saturated) pixels, and the merge needs the moon geometry on its own.
    """
    n = len(group)
    assert n >= 1, group
    sum_Lm = sum_varm2 = sum_m = sum_mgeom = sum_moon = None
    for j in range(n):
        L, var, valid, usable = source.load_radiance(group[j], device)
        H, W = L.shape
        moon_out = _moon_out_mask(group[j], H, W, device)
        m = valid.to(torch.float32) * moon_out
        m_geom = usable.to(torch.float32) * moon_out

        x_j = float(abs_xy[j, 0])
        y_j = float(abs_xy[j, 1])
        theta_j_deg = -math.degrees(float(abs_angle_t[j]))
        w_Lm = apply_transform_single(L * m, x_j, y_j, theta_j_deg, device)
        w_varm2 = apply_transform_single(var * m * m, x_j, y_j, theta_j_deg, device)
        w_m = apply_transform_single(m, x_j, y_j, theta_j_deg, device)
        w_mgeom = apply_transform_single(m_geom, x_j, y_j, theta_j_deg, device)
        w_moon = apply_transform_single(moon_out, x_j, y_j, theta_j_deg, device)
        del L, var, valid, usable, m, m_geom, moon_out

        if sum_Lm is None:
            sum_Lm, sum_varm2, sum_m, sum_mgeom, sum_moon = w_Lm, w_varm2, w_m, w_mgeom, w_moon
        else:
            sum_Lm += w_Lm
            sum_varm2 += w_varm2
            sum_m += w_m
            sum_mgeom += w_mgeom
            sum_moon += w_moon
            del w_Lm, w_varm2, w_m, w_mgeom, w_moon

    denom = sum_m.clamp(min=EPS)
    Lbar = sum_Lm / denom
    Vbar = sum_varm2 / (denom * denom)
    covered = sum_mgeom / float(n)
    moon_out = sum_moon / float(n)
    del sum_Lm, sum_varm2, sum_m, sum_mgeom, sum_moon
    return Lbar, Vbar, covered, moon_out


def merge_to_composite(ctx, source) -> None:
    """Window-weighted merge of every exposure onto the reference grid.

    Sets `ctx.composite` (brightness, `NO_DATA` where nothing could see the pixel),
    `ctx.composite_variance` (`1/sum(w)`, `inf` at `NO_DATA` — a nominal effective-variance
    proxy under the window weights below, not a physically derived uncertainty), `ctx.valid_all`
    (mutual geometric coverage, as v2), `ctx.no_data_mask`, and `ctx.exposure_weights` /
    `ctx.exposure_weight_times` — every individual exposure's own weight map in ref coords,
    before summing, for later inspection of which exposure dominated where. That stack is
    one full-resolution array per exposure (tens of exposures), so it is the single largest
    thing this function computes; nothing here needs it, it exists purely to be saved.
    """
    device = ctx.device
    available = {
        exp for exp in ctx.exposure_times_sorted
        if exp in ctx.opt_results and len(ctx.exposure_groups.get(exp, ())) >= 2
    }
    exposures_with_chain, chains = build_chains(
        ctx.exposure_times_sorted, available, ctx.cross_reg, ctx.t_ref
    )
    ref_group = ctx.exposure_groups[ctx.t_ref]
    H_ref, W_ref = int(ref_group[0].height), int(ref_group[0].width)
    ctx.H_ref, ctx.W_ref = H_ref, W_ref
    n_exp = len(exposures_with_chain)
    print(f"Exposures in chain: {n_exp} of {len(ctx.exposure_times_sorted)}")

    sum_wL = np.zeros((H_ref, W_ref), dtype=np.float64)
    sum_w = np.zeros((H_ref, W_ref), dtype=np.float64)
    valid_all = None
    common_moon_out = None
    exposure_weights = np.zeros((n_exp, H_ref, W_ref), dtype=np.float32)

    for k, exp in enumerate(tqdm.tqdm(exposures_with_chain, desc="Merge (radiance)")):
        group = ctx.exposure_groups[exp]
        abs_xy = torch.from_numpy(ctx.opt_results[exp]["abs_xy"]).to(device)
        abs_angle_t = torch.from_numpy(ctx.opt_results[exp]["abs_angle_t"]).to(device)
        Lbar, Vbar, covered, moon_out = average_exposure_radiance(
            group, abs_xy, abs_angle_t, source, device
        )
        del Vbar  # no longer the weight source; see merge_to_composite's docstring

        # Weights are formed here, in this exposure's own frame, and warped already
        # multiplied in. Warping w and Lbar separately would let bilinear interpolation pair
        # a trusted pixel's value with an untrusted neighbour's weight.
        t_eff = source.effective_exposure(group[0])
        hi = 1.0 / t_eff
        ok = (covered >= MIN_STACK_COVER) & torch.isfinite(Lbar)
        L_clamped = Lbar.clamp(0.0, hi)
        w = torch.where(ok, _window(L_clamped, 0.0, hi), torch.zeros_like(Lbar))
        # Zero the values too, not just the weights: an uncovered pixel's Lbar can be inf or
        # nan, and 0 * inf is nan, which would then spread through the warp's interpolation.
        L_ok = torch.where(ok, Lbar, torch.zeros_like(Lbar))
        chain = chains[exp]
        wL_ref = warp_to_ref(w * L_ok, chain, H_ref, W_ref, device)
        w_ref = warp_to_ref(w, chain, H_ref, W_ref, device)
        cover = warp_to_ref(
            torch.ones((Lbar.shape[0], Lbar.shape[1]), device=device, dtype=torch.float32),
            chain, H_ref, W_ref, device,
        )
        moon_ref = warp_to_ref(moon_out, chain, H_ref, W_ref, device)
        del Lbar, covered, moon_out, ok, w, L_ok, L_clamped

        inside = (cover >= REF_COVER_THRESH).cpu().numpy()
        w_ref_np = np.where(inside, w_ref.cpu().numpy(), 0.0)
        sum_wL += np.where(inside, wL_ref.double().cpu().numpy(), 0.0)
        sum_w += w_ref_np
        exposure_weights[k] = w_ref_np.astype(np.float32)
        cover_np = cover.cpu().numpy()
        moon_np = moon_ref.cpu().numpy()
        valid_all = cover_np if valid_all is None else np.minimum(valid_all, cover_np)
        common_moon_out = moon_np if common_moon_out is None else np.minimum(common_moon_out, moon_np)
        del wL_ref, w_ref, cover, moon_ref, w_ref_np
        torch.cuda.empty_cache()

    # Blank the union of the moon disks — equivalently the largest, which is what v2's
    # `mask_0` achieved by multiplying every exposure by the shortest one's coverage. Applied
    # to every exposure's own weight map too, so it stays consistent with what actually fed
    # the composite: a slice of exposure_weights reads exactly like that exposure's
    # contribution to composite_variance, not its pre-blanking value.
    moon_blank = common_moon_out < MOON_BLANK_THRESH
    sum_w[moon_blank] = 0.0
    sum_wL[moon_blank] = 0.0
    exposure_weights[:, moon_blank] = 0.0

    have = sum_w > 0
    radiance = np.where(have, sum_wL / np.maximum(sum_w, EPS), NO_DATA).astype(np.float32)
    variance = np.where(have, 1.0 / np.maximum(sum_w, EPS), np.inf).astype(np.float32)

    ctx.composite = radiance
    ctx.composite_variance = variance
    ctx.valid_all = valid_all
    ctx.no_data_mask = ~have
    ctx.exposure_weights = exposure_weights
    ctx.exposure_weight_times = np.asarray(exposures_with_chain, dtype=np.float64)

    # Report the two kinds of missing pixel separately: lumping them together hides the
    # informative one. Moon is expected; a gap outside it means every exposure saturated.
    region = valid_all >= REF_COVER_THRESH
    n_region = int(region.sum())
    n_moon = int((moon_blank & region).sum())
    n_gap = int((~have & ~moon_blank & region).sum())
    print(f"Composite shape {radiance.shape}, dtype {radiance.dtype}")
    print(f"  masked as moon             {n_moon:>12,}  ({n_moon / max(n_region, 1):.2%} of covered)")
    print(f"  coverage gaps outside moon {n_gap:>12,}  ({n_gap / max(n_region, 1):.2%} of covered)")
    if n_gap:
        print("  (a gap outside the moon means every exposure was saturated or masked there)")
    lo, hi = np.percentile(radiance[have], [50.0, 99.0])
    print(f"  brightness p50 {lo:.6g}, p99 {hi:.6g}, spread p99/p50 {hi / max(lo, EPS):.1f}")
    torch.cuda.empty_cache()
