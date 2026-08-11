#!/usr/bin/env python3
"""Redo the tail of the v6 pipeline from a finished run's frame dump, with a slot to deform
every individual frame before it's summed into the composite.

    frames (edited) -> composite -> crop -> radial-normalize -> sharpen -> RGB

Needs an old workdir that already has `frame_dump/` (the default since `pipeline.py` was
changed to dump frames), and writes a fresh run into a new workdir. No source images and no
GPU merge are needed: everything downstream of the dump is either pure per-pixel arithmetic
on the dumped rasters or the existing display chain, unchanged.

The one slot meant to be edited is `deformator` below.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from scipy.ndimage import map_coordinates


def _find_package_root() -> Path:
    cwd = Path(__file__).parent.resolve()
    if (cwd / "eclipse_v6").is_dir():
        return cwd
    raise RuntimeError("Could not find eclipse_v6 package alongside this script.")


ROOT = _find_package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6.device import configure_cuda_visible_devices, require_cuda  # noqa: E402
from eclipse_v6.merge import EPS, NO_DATA  # noqa: E402
from eclipse_v6.stage3 import (  # noqa: E402
    Stage3Context,
    crop_and_save_composite,
    fft_unsharp_and_save,
    load_inputs,
    radial_normalize_display,
    rgb_vignette_and_radial_pickle,
)

DOWNSCALE = 4  # default block-reduce factor for the stacks handed to `deformator`


# --- The slot: replace this with the real thing ------------------------------------------

def deformator(weight: np.ndarray, wval: np.ndarray, meta: list[dict], scale: int) -> np.ndarray:
    """Decide a per-frame displacement field from every frame's downscaled data at once.

    `weight`, `wval` — float32 arrays `(N, h, w)`: every frame's block-averaged weight and
    weight*value, same order as `meta`. `meta[k]` is `{"exposure": float, "path": str}` for
    frame `k`. `scale` is the block-reduce factor actually used (nominally `DOWNSCALE`, but
    read at runtime rather than assumed).

    Must return a float32 array `(N, h, w, 2)`: `field[k, i, j, 0]` = dy, `[..., 1]` = dx, in
    *downscaled*-pixel units. Backward-mapping convention — the caller reads
    `output[k, i, j] = input[k, i + dy, j + dx]` (bilinear) at full resolution, after
    upsampling this field and scaling dy/dx up by `scale`.

    The one hard rule: it must never sample outside the full-res image — the caller asserts
    this per frame and aborts with the offending frame's name rather than silently clamping.

    Placeholder: identity (no deformation).
    """
    n, h, w = weight.shape
    return np.zeros((n, h, w, 2), dtype=np.float32)


# --- Plumbing: frame dump -> deformed composite -----------------------------------------

def _reduced_shape(H: int, W: int, scale: int) -> tuple[int, int]:
    return -(-H // scale), -(-W // scale)


def _block_reduce_mean(a: np.ndarray, scale: int) -> np.ndarray:
    """Average `a` (2D) over non-overlapping `scale`x`scale` blocks; a ragged last block is
    padded by edge-replication rather than dropped."""
    H, W = a.shape
    h, w = _reduced_shape(H, W, scale)
    pad_h, pad_w = h * scale - H, w * scale - W
    if pad_h or pad_w:
        a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="edge")
    return a.reshape(h, scale, w, scale).mean(axis=(1, 3))


def _load_manifest(dump_dir: Path) -> list[dict]:
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    assert manifest, f"no frames in {dump_dir / 'manifest.json'}"
    return manifest


def _full_res_shape(dump_dir: Path) -> tuple[int, int]:
    return np.load(dump_dir / "sum_w.npy", mmap_mode="r").shape


def _build_downscaled_stacks(dump_dir: Path, manifest: list[dict], scale: int):
    H, W = _full_res_shape(dump_dir)
    h, w = _reduced_shape(H, W, scale)
    n = len(manifest)
    weight_small = np.empty((n, h, w), dtype=np.float32)
    wval_small = np.empty((n, h, w), dtype=np.float32)
    meta = []
    for k, entry in enumerate(tqdm.tqdm(manifest, desc="Downscaling frames")):
        data = np.load(dump_dir / entry["npz"])
        weight_small[k] = _block_reduce_mean(data["weight"], scale)
        wval_small[k] = _block_reduce_mean(data["wval"], scale)
        meta.append({"exposure": entry["exposure"], "path": entry["path"]})
    return weight_small, wval_small, meta


def _upsample_field(field_small: np.ndarray, H: int, W: int, scale: int):
    """Bilinear-upsample a `(h, w, 2)` field (downscaled-pixel units) to full-res `(H, W)`
    dy/dx (full-res-pixel units)."""
    h, w, _ = field_small.shape
    ii = np.clip((np.arange(H, dtype=np.float32) + 0.5) / scale - 0.5, 0, h - 1)
    jj = np.clip((np.arange(W, dtype=np.float32) + 0.5) / scale - 0.5, 0, w - 1)
    grid_i, grid_j = np.meshgrid(ii, jj, indexing="ij")
    dy = map_coordinates(field_small[..., 0], [grid_i, grid_j], order=1, mode="nearest")
    dx = map_coordinates(field_small[..., 1], [grid_i, grid_j], order=1, mode="nearest")
    return dy.astype(np.float32) * scale, dx.astype(np.float32) * scale


def _assert_in_bounds(sample_i: np.ndarray, sample_j: np.ndarray, H: int, W: int, label: str) -> None:
    lo_i, hi_i = float(sample_i.min()), float(sample_i.max())
    lo_j, hi_j = float(sample_j.min()), float(sample_j.max())
    assert 0.0 <= lo_i and hi_i <= H - 1 and 0.0 <= lo_j and hi_j <= W - 1, (
        f"deformator field for frame {label!r} samples outside the {H}x{W} image: "
        f"rows [{lo_i:.2f}, {hi_i:.2f}], cols [{lo_j:.2f}, {hi_j:.2f}]"
    )


def _apply_field(a: np.ndarray, sample_i: np.ndarray, sample_j: np.ndarray) -> np.ndarray:
    """output[i, j] = a[sample_i[i, j], sample_j[i, j]], bilinear."""
    return map_coordinates(a, [sample_i, sample_j], order=1, mode="constant", cval=0.0)


def _reconstruct_deformed_composite(dump_dir, manifest, field, scale, H, W):
    """Second pass over the dump: warp each frame's full-res (weight, wval) with its
    (upsampled) field and accumulate — same running-sum shape as `merge_to_composite` and
    `reconstruct_from_frame_dump`, just with a per-frame warp folded in before the sum.

    Also returns `have` (sum_weight > 0): this doubles as both the no-data mask and the crop
    boundary — no separate mutual-coverage recompute needed, since a bounding box of "any
    exposure has data here" already stops at the same outer edge a purely geometric coverage
    map would (the moon is a hole nowhere near that edge, so it never enters into it).
    """
    rows, cols = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    sum_weight = np.zeros((H, W), dtype=np.float64)
    sum_wval = np.zeros((H, W), dtype=np.float64)
    for k, entry in enumerate(tqdm.tqdm(manifest, desc="Warping + accumulating")):
        data = np.load(dump_dir / entry["npz"])
        dy, dx = _upsample_field(field[k], H, W, scale)
        sample_i, sample_j = rows + dy, cols + dx
        _assert_in_bounds(sample_i, sample_j, H, W, f"{entry['exposure']:.6f} {entry['path']}")
        sum_weight += _apply_field(data["weight"].astype(np.float32), sample_i, sample_j)
        sum_wval += _apply_field(data["wval"].astype(np.float32), sample_i, sample_j)
    have = sum_weight > 0
    composite = np.where(have, sum_wval / np.maximum(sum_weight, EPS), NO_DATA).astype(np.float32)
    return composite, have


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old_workdir", type=Path, help="Finished v6 run with frame_dump/ present.")
    ap.add_argument("new_workdir", type=Path, help="Must not already exist.")
    ap.add_argument("--downscale", type=int, default=DOWNSCALE,
                    help=f"Block-reduce factor for the stacks handed to deformator() (default {DOWNSCALE}).")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.new_workdir.exists():
        sys.exit(f"Refusing to run: {args.new_workdir} already exists.")

    dump_dir = args.old_workdir / "frame_dump"
    if not (dump_dir / "manifest.json").exists():
        sys.exit(f"No frame dump at {dump_dir} — re-run pipeline.py on {args.old_workdir} "
                 f"(or with --frame-dump on) first.")

    configure_cuda_visible_devices()
    require_cuda()
    device = torch.device("cuda")

    args.new_workdir.mkdir(parents=True)

    print("=== Loading run state (pickles only, no source images) ===")
    ctx = Stage3Context(workdir=args.old_workdir, device=device)
    load_inputs(ctx)

    manifest = _load_manifest(dump_dir)
    H, W = _full_res_shape(dump_dir)
    ref_group = ctx.exposure_groups[ctx.t_ref]
    assert (H, W) == (int(ref_group[0].height), int(ref_group[0].width)), (
        f"frame dump shape {(H, W)} doesn't match {args.old_workdir}'s reference exposure "
        f"shape {(int(ref_group[0].height), int(ref_group[0].width))} — mismatched run?"
    )
    print(f"{len(manifest)} frames, full res {H}x{W}")

    print("=== Downscaling every frame for deformator() ===")
    weight_small, wval_small, meta = _build_downscaled_stacks(dump_dir, manifest, args.downscale)
    print(f"Stacks: {weight_small.shape}, {2 * weight_small.nbytes / 1e9:.2f} GB total")

    field = deformator(weight_small, wval_small, meta, args.downscale)
    assert field.shape == (*weight_small.shape, 2), (field.shape, weight_small.shape)
    del weight_small, wval_small

    print("=== Warping full-res frames and re-merging ===")
    composite, have = _reconstruct_deformed_composite(
        dump_dir, manifest, field, args.downscale, H, W
    )

    ctx.workdir = args.new_workdir
    ctx.composite = composite
    ctx.composite_variance = None
    ctx.no_data_mask = ~have
    ctx.valid_all = have.astype(np.float32)  # doubles as the crop-boundary map, see above

    print("\n=== Stage 3 tail: crop, radial-normalize, sharpen, RGB ===")
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)

    print("\nDone. Outputs in", args.new_workdir)
