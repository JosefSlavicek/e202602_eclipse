"""Combine the bracket in physical brightness, weighted by each estimate's own uncertainty.

Replaces v2's merge (v6's `stage3.warp_merge_to_composite`, since removed), which rescaled
*encoded* values down a chain of fitted per-pair exponents and averaged them under a
hand-shaped bell weight.  Here every
exposure is converted to brightness independently through the calibrated response
(`source.load_radiance`) and combined with `w = 1/sigma^2`, so nothing accumulates along a
ladder and no weight is guessed.

Three behaviours that used to be hand-tuned now fall out of the formula: long exposures
dominate the faint outer corona, short exposures dominate the bright inner corona, and
readings from the parts of the response curve we know least well are discounted.

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

import json
import math
from pathlib import Path

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
EPS = 1e-20


def _moon_out_mask(ii, H, W, device):
    """1.0 outside this frame's moon disk (plus a 2 px margin), 0.0 inside."""
    mi, mj, r = ii.moon[0], ii.moon[1], ii.moon[2]
    rows = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1).expand(H, W)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, -1).expand(H, W)
    dist = torch.sqrt((rows - mi) ** 2 + (cols - mj) ** 2)
    return (dist > r + MOON_MARGIN_PX).to(torch.float32)


def average_exposure_radiance(group, abs_xy, abs_angle_t, source, device, keep_frames=False):
    """Stack one exposure group in brightness. Returns (Lbar, Vbar, covered, moon_out, frame_records).

    Converting each frame and then averaging the brightnesses — rather than averaging the
    stored values and converting once, as v2 did — matters because `f` is curved: the
    average of encoded values is not the encoding of the average.

        Lbar = sum(m*L) / sum(m)          Vbar = sum(m^2*var) / sum(m)^2

    which is the ordinary variance of a weighted mean, so a group with more frames earns
    proportionally more weight in the merge with nothing extra to say about it.

    `moon_out` is returned separately from `covered` because `covered` also excludes
    saturated pixels, and the merge needs the moon geometry on its own.

    `keep_frames=True` additionally returns, per frame, `(img_info, w_m, w_Lm)` — the same
    intra-exposure-warped mask and mask*radiance this loop already computes on the way to
    `sum_m`/`sum_Lm`, just not thrown away. `merge_to_composite` uses this to let a frame
    dump reconstruct the composite as a flat weighted sum over individual frames without
    ever holding more than one exposure group's worth of them at once.
    """
    n = len(group)
    assert n >= 1, group
    sum_Lm = sum_varm2 = sum_m = sum_moon = None
    frame_records = []
    for j in range(n):
        L, var, valid = source.load_radiance(group[j], device)
        H, W = L.shape
        moon_out = _moon_out_mask(group[j], H, W, device)
        m = valid.to(torch.float32) * moon_out

        x_j = float(abs_xy[j, 0])
        y_j = float(abs_xy[j, 1])
        theta_j_deg = -math.degrees(float(abs_angle_t[j]))
        w_Lm = apply_transform_single(L * m, x_j, y_j, theta_j_deg, device)
        w_varm2 = apply_transform_single(var * m * m, x_j, y_j, theta_j_deg, device)
        w_m = apply_transform_single(m, x_j, y_j, theta_j_deg, device)
        w_moon = apply_transform_single(moon_out, x_j, y_j, theta_j_deg, device)
        del L, var, valid, m, moon_out

        if keep_frames:
            # Clone: `sum_Lm`/`sum_m` alias these exact tensors on the first iteration
            # (see below) and are mutated in place on every later one via `+=`.
            frame_records.append((group[j], w_m.clone(), w_Lm.clone()))

        if sum_Lm is None:
            sum_Lm, sum_varm2, sum_m, sum_moon = w_Lm, w_varm2, w_m, w_moon
        else:
            sum_Lm += w_Lm
            sum_varm2 += w_varm2
            sum_m += w_m
            sum_moon += w_moon
            if not keep_frames:
                del w_Lm, w_varm2, w_m, w_moon
            else:
                del w_varm2

    denom = sum_m.clamp(min=EPS)
    Lbar = sum_Lm / denom
    Vbar = sum_varm2 / (denom * denom)
    covered = sum_m / float(n)
    moon_out = sum_moon / float(n)
    del sum_Lm, sum_varm2, sum_m, sum_moon
    return Lbar, Vbar, covered, moon_out, frame_records


def merge_to_composite(ctx, source, frame_dump_dir: Path | None = None) -> None:
    """Inverse-variance merge of every exposure onto the reference grid.

    Sets `ctx.composite` (brightness, `NO_DATA` where nothing could see the pixel),
    `ctx.composite_variance` (its uncertainty squared, `inf` at `NO_DATA`), `ctx.valid_all`
    (mutual geometric coverage, as v2) and `ctx.no_data_mask`.

    `frame_dump_dir`, if given, additionally streams every individual frame's contribution
    to disk: for each frame, `weight` and `wval` rasters (both `(H_ref, W_ref)` float32, on
    the *uncropped* reference grid) such that

        composite == sum(wval over every frame) / sum(weight over every frame)

    at every pixel not blanked by the moon (see `moon_blank.npy`, written once at the end
    alongside `sum_w.npy` — both needed to finish the reconstruction; a frame's `weight` is
    not itself normalised to sum to 1, since the normaliser isn't known until every exposure
    has been visited). `manifest.json` lists one entry per dumped frame. Nothing beyond one
    exposure group's frames is ever held at once — same discipline as the merge itself.
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
    print(f"Exposures in chain: {len(exposures_with_chain)} of {len(ctx.exposure_times_sorted)}")

    sum_wL = np.zeros((H_ref, W_ref), dtype=np.float64)
    sum_w = np.zeros((H_ref, W_ref), dtype=np.float64)
    valid_all = None
    common_moon_out = None
    manifest = [] if frame_dump_dir is not None else None
    if frame_dump_dir is not None:
        frame_dump_dir = Path(frame_dump_dir)
        frame_dump_dir.mkdir(parents=True, exist_ok=True)

    for exp in tqdm.tqdm(exposures_with_chain, desc="Merge (radiance)"):
        group = ctx.exposure_groups[exp]
        abs_xy = torch.from_numpy(ctx.opt_results[exp]["abs_xy"]).to(device)
        abs_angle_t = torch.from_numpy(ctx.opt_results[exp]["abs_angle_t"]).to(device)
        n = len(group)
        Lbar, Vbar, covered, moon_out, frame_records = average_exposure_radiance(
            group, abs_xy, abs_angle_t, source, device, keep_frames=frame_dump_dir is not None
        )

        # Weights are formed here, in this exposure's own frame, and warped already
        # multiplied in. Warping w and Lbar separately would let bilinear interpolation pair
        # a trusted pixel's value with an untrusted neighbour's weight.
        ok = (covered >= MIN_STACK_COVER) & torch.isfinite(Vbar) & (Vbar > 0) & torch.isfinite(Lbar)
        w = torch.where(ok, 1.0 / Vbar.clamp(min=EPS), torch.zeros_like(Vbar))
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

        inside = (cover >= REF_COVER_THRESH).cpu().numpy()

        if frame_records:
            # Same decomposition `Lbar = sum_j(w_m_j * L_j) / sum_j(w_m_j)` that produced
            # Lbar/Vbar in the first place, just not collapsed across j before multiplying
            # by this exposure's own inverse-variance weight `w` and warping to the ref
            # grid. Summing every frame's (weight, wval) reproduces (w_ref, wL_ref) exactly.
            denom = (covered * float(n)).clamp(min=EPS)  # == sum_m from average_exposure_radiance
            for img_info, w_m, w_Lm in frame_records:
                frame_weight = w_m / denom * w
                frame_wval = w_Lm / denom * w
                weight_ref = warp_to_ref(frame_weight, chain, H_ref, W_ref, device)
                wval_ref = warp_to_ref(frame_wval, chain, H_ref, W_ref, device)
                weight_np = np.where(inside, weight_ref.cpu().numpy(), 0.0).astype(np.float32)
                wval_np = np.where(inside, wval_ref.cpu().numpy(), 0.0).astype(np.float32)
                del frame_weight, frame_wval, weight_ref, wval_ref, w_m, w_Lm

                stem = f"{exp:.6f}__{Path(img_info.path).stem}"
                out_path = frame_dump_dir / f"{stem}.npz"
                np.savez(out_path, weight=weight_np, wval=wval_np)
                manifest.append({
                    "exposure": exp, "path": str(img_info.path), "npz": out_path.name,
                })
            del frame_records
            torch.cuda.empty_cache()

        sum_wL += np.where(inside, wL_ref.double().cpu().numpy(), 0.0)
        sum_w += np.where(inside, w_ref.double().cpu().numpy(), 0.0)
        cover_np = cover.cpu().numpy()
        moon_np = moon_ref.cpu().numpy()
        valid_all = cover_np if valid_all is None else np.minimum(valid_all, cover_np)
        common_moon_out = moon_np if common_moon_out is None else np.minimum(common_moon_out, moon_np)
        del Lbar, Vbar, covered, moon_out, ok, w, L_ok, wL_ref, w_ref, cover, moon_ref
        torch.cuda.empty_cache()

    # Blank the union of the moon disks — equivalently the largest, which is what v2's
    # `mask_0` achieved by multiplying every exposure by the shortest one's coverage.
    moon_blank = common_moon_out < MOON_BLANK_THRESH
    sum_w[moon_blank] = 0.0
    sum_wL[moon_blank] = 0.0

    if frame_dump_dir is not None:
        np.save(frame_dump_dir / "sum_w.npy", sum_w.astype(np.float32))
        np.save(frame_dump_dir / "moon_blank.npy", moon_blank)
        with open(frame_dump_dir / "manifest.json", "w") as fd:
            json.dump(manifest, fd, indent=2)
        print(f"Frame dump: {len(manifest)} frames -> {frame_dump_dir}")

    have = sum_w > 0
    radiance = np.where(have, sum_wL / np.maximum(sum_w, EPS), NO_DATA).astype(np.float32)
    variance = np.where(have, 1.0 / np.maximum(sum_w, EPS), np.inf).astype(np.float32)

    ctx.composite = radiance
    ctx.composite_variance = variance
    ctx.valid_all = valid_all
    ctx.no_data_mask = ~have

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


def reconstruct_from_frame_dump(dump_dir) -> np.ndarray:
    """Rebuild the composite as a flat weighted sum over individual frames.

    The original goal this and `merge_to_composite`'s `frame_dump_dir` exist for: for each
    individual frame, be able to get its rectangle contributing to the composite, its pixel
    values expressed in the same units the composite holds, and its pixel weights — so that
    reconstructing the composite is as easy as a weighted sum.

    That per-frame data (`weight`, `wval`, both full `(H_ref, W_ref)` rasters, zero outside
    the frame's own footprint) is what `merge_to_composite(..., frame_dump_dir=...)` already
    writes to `dump_dir`, one `.npz` per frame, listed in `manifest.json`. This function is
    only the second half — the weighted sum itself:

        composite = sum(wval over every frame) / sum(weight over every frame)

    with `moon_blank.npy` zeroing both sums first (that mask isn't known per frame — only
    once every exposure has been visited — so it can't be baked into the per-frame files).
    `sum_w.npy` is the merge's own copy of the normaliser, kept here only as a cross-check
    against the sum recomputed from the per-frame files.

    Returns the reconstructed composite: physical brightness, `NO_DATA` where nothing could
    see the pixel, on the *uncropped* reference grid — bit-for-bit what `ctx.composite` held
    right after `merge_to_composite` ran (before `crop_and_save_composite` sliced it down).
    """
    dump_dir = Path(dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    assert manifest, f"no frames in {dump_dir / 'manifest.json'}"
    sum_w_saved = np.load(dump_dir / "sum_w.npy").astype(np.float64)
    moon_blank = np.load(dump_dir / "moon_blank.npy")

    sum_weight = np.zeros_like(sum_w_saved)
    sum_wval = np.zeros_like(sum_w_saved)
    for entry in manifest:
        data = np.load(dump_dir / entry["npz"])
        sum_weight += data["weight"].astype(np.float64)
        sum_wval += data["wval"].astype(np.float64)

    sum_weight[moon_blank] = 0.0
    sum_wval[moon_blank] = 0.0

    mismatch = np.abs(sum_weight - sum_w_saved)
    tol = 1e-4 * np.maximum(sum_w_saved, EPS)
    assert np.all(mismatch <= tol + 1e-6), (
        f"per-frame weights don't add up to sum_w.npy: max mismatch {mismatch.max():.6g} "
        f"— the frame dump in {dump_dir} may be from a different/incomplete run"
    )

    have = sum_weight > 0
    return np.where(have, sum_wval / np.maximum(sum_weight, EPS), NO_DATA).astype(np.float32)
