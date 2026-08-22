"""Combine the bracket into one physical-brightness image, weighted by measurement confidence.

Replaces `stage3.warp_merge_to_composite`, which averaged encoded pixel values scaled by a
chain of fitted exponents, under a hand-shaped bell weight. Here every exposure is
converted to real brightness independently (`source.load_radiance`), and each frame's own
merge weight is measured once, at ingestion, straight off its raw decode -- before
dark/flat correction, before averaging, before anything else touches it (`rawprep.py`).
That weight map rides through the same warps as the image data, so it stays pixel-aligned
with it and needs no clamping of its own -- it was never a brightness value to begin with,
just a window over how trustworthy the raw reading was. Long exposures still dominate the
faint outer corona and short exposures the bright inner corona, because each frame's
window is only full-weight where that exposure reads the scene well.

We then scale each group's weight by its total integration time (`t_eff * frame count`)
before summing across exposures. Since every exposure is estimating the same true
brightness at a given pixel, and shot noise scales roughly as brightness / integration
time, this makes the weight track the *inverse-variance* more closely than the raw window
shape alone would. It's still an approximation, not a fitted noise model: things like
read noise don't scale this way.

Two things kept from v2:

  * Common moon blanking. The detected moon radius drifts 5.94 px across the bracket
    (316.03 px at 1/4000 s down to 310.19 px at 2 s) because glare in long exposures makes
    the edge-finder place the limb further out. So every exposure is blanked to the union
    of all the moon disks -- otherwise the ring between the smallest and largest disk
    would be populated only by saturated long exposures.
  * One exposure at a time. All the exposures at once would be ~2.9 GB of GPU memory (15
    exposures x 2 full-res arrays); instead we build, warp, accumulate, and free each one
    before moving to the next.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import tqdm

from eclipse_v8.utils import apply_transform_single
from eclipse_v8.warp import build_chains, warp_to_ref

NO_DATA = -1.0               # composite sentinel: no exposure could see this pixel
MOON_MARGIN_PX = 2.0         # same limb margin as utils.compute_weighted_average
MIN_STACK_COVER = 0.5        # a merged pixel needs half a frame's worth of valid contributions
REF_COVER_THRESH = 0.999     # inside the reference-grid footprint of this exposure
MOON_BLANK_THRESH = 0.999    # below this, some exposure called the pixel "moon"
EPS = 1e-20


def _moon_out_mask(ii, H, W, device):
    """1.0 outside this frame's moon disk (plus a 2 px margin), 0.0 inside."""
    mi, mj, r = ii.moon[0], ii.moon[1], ii.moon[2]
    rows = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    dist = torch.sqrt((rows - mi) ** 2 + (cols - mj) ** 2)
    return (dist > r + MOON_MARGIN_PX).to(torch.float32)


def average_exposure_radiance(group, abs_xy, abs_angle_t, source, device):
    """Stack one exposure group in physical brightness. Returns (Lbar, covered, moon_out, group_weight).

    We convert each frame to brightness first and average those, rather than averaging the
    raw stored values and converting once -- the two aren't the same, since the response
    curve isn't a straight line.

    `Lbar` is a plain mean over the frames that actually cover a pixel (not saturated, not
    on the moon); `covered` is what fraction of the group's frames did. `group_weight` is
    the group's merge weight, averaged the same way -- it comes from `rawprep.py`'s
    per-frame measurement, not from `Lbar` itself, and just rides along through the same
    warp so it stays pixel-aligned with the data. `moon_out` is kept separate from
    `covered` since the merge needs the moon's shape on its own, not lumped in with
    saturation.
    """
    n = len(group)
    assert n >= 1, group
    sum_Lm = sum_m = sum_moon = sum_wm = None
    for j in range(n):
        L, _var, usable = source.load_radiance(group[j], device)
        weight = source.load_weight(group[j], device)
        H, W = L.shape
        moon_out = _moon_out_mask(group[j], H, W, device)
        m = usable.to(torch.float32) * moon_out

        x_j = float(abs_xy[j, 0])
        y_j = float(abs_xy[j, 1])
        theta_j_deg = -math.degrees(float(abs_angle_t[j]))
        w_Lm = apply_transform_single(L * m, x_j, y_j, theta_j_deg, device)
        w_m = apply_transform_single(m, x_j, y_j, theta_j_deg, device)
        w_moon = apply_transform_single(moon_out, x_j, y_j, theta_j_deg, device)
        w_wm = apply_transform_single(weight * m, x_j, y_j, theta_j_deg, device)
        del L, _var, usable, weight, m, moon_out

        if sum_Lm is None:
            sum_Lm, sum_m, sum_moon, sum_wm = w_Lm, w_m, w_moon, w_wm
        else:
            sum_Lm += w_Lm
            sum_m += w_m
            sum_moon += w_moon
            sum_wm += w_wm
            del w_Lm, w_m, w_moon, w_wm

    denom = sum_m.clamp(min=EPS)
    Lbar = sum_Lm / denom
    covered = sum_m / float(n)
    moon_out = sum_moon / float(n)
    group_weight = sum_wm / denom
    del sum_Lm, sum_m, sum_moon, sum_wm
    return Lbar, covered, moon_out, group_weight


def merge_to_composite(ctx, source) -> None:
    """Merge every exposure onto the reference grid, weighted by confidence.

    Sets `ctx.composite` (brightness, `NO_DATA` where nothing could see the pixel) and
    `ctx.composite_variance` (`1/sum(w)`, a rough effective-variance estimate, not a truly
    measured one). Also saves each exposure's own weight map before summing
    (`ctx.exposure_weights`/`ctx.exposure_weight_times`), purely so it can be inspected
    later -- nothing downstream needs it, and it's the single largest thing this function
    computes.
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
        Lbar, covered, moon_out, group_weight = average_exposure_radiance(
            group, abs_xy, abs_angle_t, source, device
        )

        # Weights are formed here, in this exposure's own frame, and warped already
        # multiplied in. Warping w and Lbar separately would let bilinear interpolation pair
        # a trusted pixel's value with an untrusted neighbour's weight.
        t_eff = source.effective_exposure(group[0])
        ok = (covered >= MIN_STACK_COVER) & torch.isfinite(Lbar)
        w = torch.where(ok, group_weight, torch.zeros_like(group_weight))
        w = w * (t_eff * len(group))
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
        del Lbar, covered, moon_out, group_weight, ok, w, L_ok

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
