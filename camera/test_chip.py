#!/usr/bin/env python3
"""Per-pixel gradient-magnitude statistics over a random subset of .NEF chip-state frames.

Two-pass, fully iterative (never more than one decoded image in RAM/VRAM):

  Pass 0: over the N sampled frames, average each photosite's |value - same-color
          neighborhood median| on the RAW Bayer array (no demosaic), then report how many
          pixels exceed RESIDUAL_THRESHOLD and print the 10 largest mean-residual values.
          The flagged photosites form a defect mask reused to repair every frame in
          Pass 1/2: before demosaic, each bad photosite is overwritten with its same-color
          neighborhood median, so the defect never propagates through interpolation.
  Pass 1: load N randomly chosen .NEF files, decode each to an RGB float32 [0,1] array,
          move to GPU, collapse to luminance, compute the gradient magnitude
          sqrt(d/dx^2 + d/dy^2), crop the 1px boundary, and record the per-image MAX.
  Select: keep the int(f*N) images with the LOWEST max gradient.
  Pass 2: recompute the cropped gradient magnitude for each kept image and aggregate
          into per-pixel mean and std (running sum / sum-of-squares / count in float64).

Run with the e202602_eclipse conda env python (torch + CUDA), and rawpy must be importable:
    /home/slavik/usr/anaconda3/envs/e202602_eclipse/bin/python camera/test_chip.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import rawpy
import torch
import torch.nn.functional as F_nn
from PIL import Image

# ---- hardcoded parameters -------------------------------------------------
DATA_DIR = Path("/home/slavik/tmp/chipstate")
N = 174               # number of .NEF files to sample
F = 0.8              # fraction kept for pass 2 (lowest max-gradient frames)
SEED = 42            # set to None for non-reproducible sampling
MEDIAN_WINDOW = 5    # pass 0: same-color neighborhood window (odd; center excluded)
RESIDUAL_THRESHOLD = 100.0  # pass 0: raw-code units; flag pixels whose mean |residual| exceeds this
OUT_DIR = Path(__file__).resolve().parent
MEAN_PATH = OUT_DIR / "test_chip_grad_mean.npy"
STD_PATH = OUT_DIR / "test_chip_grad_std.npy"
PNG_PATH = OUT_DIR / "test_chip_grad_mean_minus_std.png"

# Luminance weights (Rec. 601).
LUMA = (0.299, 0.587, 0.114)


def load_nef_rgb(
    path: Path,
    device: torch.device | None = None,
    wrong_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Decode a .NEF to an RGB float32 array in [0, 1].

    no_auto_bright keeps the per-image scaling consistent across frames (required
    for meaningful per-pixel aggregation); 16-bit output is normalized by 65535.

    user_flip=0 disables the EXIF/maker-note orientation that postprocess() would
    otherwise apply (default user_flip=-1), so every frame stays in the camera's
    native sensor orientation regardless of how the camera was held. This keeps the
    RGB output aligned photosite-for-photosite with raw_image_visible (Pass 0), which
    is essential because the per-pixel aggregation maps fixed chip defects.

    If `wrong_mask` (a bool (H, W) tensor matching raw_image_visible) is given, the
    flagged photosites are repaired on the RAW Bayer array BEFORE demosaic, so the
    defect never propagates through interpolation or the gradient.
    """
    with rawpy.imread(str(path)) as raw:
        if wrong_mask is not None:
            repair_raw_visible(raw, wrong_mask, device)
        rgb16 = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            user_flip=0,  # no EXIF rotation: keep native sensor orientation, aligned to raw_image_visible
        )
    return rgb16.astype(np.float32) / 65535.0


def load_nef_raw(path: Path) -> np.ndarray:
    """Decode a .NEF to its visible RAW Bayer array (float32, raw code units, no demosaic).

    Each value is one photosite's reading, so a defect is independent of its neighbors
    (no demosaic interpolation mixing them). Copied because the rawpy view is freed on exit.
    """
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image_visible.astype(np.float32).copy()


def local_median_excl_center(plane: torch.Tensor, k: int) -> torch.Tensor:
    """Per-pixel median of the k*k neighborhood EXCLUDING the center pixel.

    `plane` is a single-color 2D sub-plane; reflect padding handles the borders so a
    defect on its own does not bias the reference value drawn from its neighbors.
    """
    pad = k // 2
    p = F_nn.pad(plane[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
    windows = p.unfold(0, k, 1).unfold(1, k, 1).reshape(plane.shape[0], plane.shape[1], k * k)
    center = (k * k) // 2
    neigh = torch.cat([windows[..., :center], windows[..., center + 1:]], dim=-1)
    return neigh.median(dim=-1).values


def abs_residual_raw(path: Path, device: torch.device) -> torch.Tensor:
    """RAW Bayer map of |pixel - same-color-neighborhood-median|, shape (H, W), on `device`.

    Split into the 4 color sub-planes by row/col parity (each parity is one CFA color),
    so the local median is always computed over same-color neighbors.
    """
    raw = torch.from_numpy(load_nef_raw(path)).to(device)    # (H, W) raw codes, float32
    res = torch.empty_like(raw)
    for oy in (0, 1):
        for ox in (0, 1):
            plane = raw[oy::2, ox::2]
            bg = local_median_excl_center(plane, MEDIAN_WINDOW)
            res[oy::2, ox::2] = (plane - bg).abs()
    return res


def repair_raw_visible(raw, wrong_mask: torch.Tensor, device: torch.device) -> None:
    """In-place repair of flagged photosites in `raw.raw_image_visible` (pre-demosaic).

    Each defect is replaced by the median of its same-color neighborhood (the very value
    Pass 0 measures against), computed per CFA sub-plane so only same-color photosites
    contribute. raw_image_visible is a writable view into the raw buffer, so the repaired
    values flow into the subsequent postprocess()/demosaic. Only flagged positions are
    written back; every other photosite stays byte-identical to the original decode.
    """
    vis = raw.raw_image_visible                          # (H, W) integer view into raw buffer
    if tuple(vis.shape) != tuple(wrong_mask.shape):
        raise SystemExit(
            f"raw_image_visible shape {tuple(vis.shape)} != mask shape "
            f"{tuple(wrong_mask.shape)}; cannot repair."
        )
    t = torch.from_numpy(vis.astype(np.float32)).to(device)
    for oy in (0, 1):
        for ox in (0, 1):
            sub_mask = wrong_mask[oy::2, ox::2]
            if not bool(sub_mask.any()):
                continue
            plane = t[oy::2, ox::2]                       # strided view of one CFA color
            med = local_median_excl_center(plane, MEDIAN_WINDOW)  # full-plane median first ...
            plane[sub_mask] = med[sub_mask]               # ... then overwrite only defects
    mask_np = wrong_mask.cpu().numpy()
    repaired = t.round().clamp_min(0.0).cpu().numpy().astype(vis.dtype)
    vis[mask_np] = repaired[mask_np]                      # write back repaired photosites only


def grad_magnitude(
    path: Path, device: torch.device, wrong_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Return the boundary-cropped luminance gradient magnitude, shape (H-2, W-2), on `device`."""
    rgb = load_nef_rgb(path, device, wrong_mask)
    t = torch.from_numpy(rgb).to(device)                 # (H, W, 3) float32 on GPU
    lum = LUMA[0] * t[..., 0] + LUMA[1] * t[..., 1] + LUMA[2] * t[..., 2]  # (H, W)
    gy, gx = torch.gradient(lum, dim=(0, 1))             # central diff interior
    mag = torch.sqrt(gx * gx + gy * gy)
    return mag[1:-1, 1:-1].contiguous()                  # drop 1px boundary -> valid values


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required but not available.")
    device = torch.device("cuda")

    nef_files = sorted(DATA_DIR.glob("*.NEF"))
    if len(nef_files) < N:
        raise SystemExit(f"Need >= {N} .NEF files in {DATA_DIR}, found {len(nef_files)}.")

    rng = random.Random(SEED)
    sample = rng.sample(nef_files, N)

    # ---- Pass 0: wrong-pixel detection (neighborhood abs-residual on RAW Bayer) ----
    # Per frame, |pixel - same-color-neighborhood-median|; averaged over the N frames.
    # A fixed defect is anomalous in every frame -> high mean residual; a transient
    # (star/edge/cosmic-ray) hits one frame and is washed out by the average.
    res_sum: torch.Tensor | None = None
    for i, path in enumerate(sample, 1):
        res = abs_residual_raw(path, device).to(torch.float64)
        if res_sum is None:
            res_sum = torch.zeros_like(res)
        elif res.shape != res_sum.shape:
            raise SystemExit(
                f"Frame {path.name} has raw shape {tuple(res.shape)} != "
                f"{tuple(res_sum.shape)}; wrong-pixel detection needs identical dimensions."
            )
        res_sum += res
        print(f"[pass0 {i}/{N}] {path.name}  accumulated")
        del res
    mean_res = res_sum / N                                  # (H, W) mean |residual| in raw codes
    wrong_mask = mean_res > RESIDUAL_THRESHOLD
    n_wrong = int(wrong_mask.sum().item())
    n_total = wrong_mask.numel()
    top10 = torch.topk(mean_res.flatten(), 10).values.tolist()
    print(
        f"\nWrong pixels (mean |residual| > {RESIDUAL_THRESHOLD}): {n_wrong} / {n_total} "
        f"({100.0 * n_wrong / n_total:.4f}%)"
    )
    print("Top 10 mean |residual| values: " + ", ".join(f"{v:.2f}" for v in top10) + "\n")
    # Keep wrong_mask: it repairs the flagged photosites (same-color neighborhood median)
    # on the RAW Bayer array before demosaic in every subsequent frame load.
    del res_sum, mean_res
    torch.cuda.empty_cache()

    # ---- Pass 1: per-image max gradient ----
    max_grads: list[float] = []
    for i, path in enumerate(sample, 1):
        mag = grad_magnitude(path, device, wrong_mask)
        max_grads.append(float(mag.max().item()))
        print(f"[pass1 {i}/{N}] {path.name}  max_grad={max_grads[-1]:.6f}")
        del mag
    torch.cuda.empty_cache()

    # ---- Select int(f*N) frames with the LOWEST max gradient -----------
    n_keep = int(F * N)
    order = sorted(range(N), key=lambda k: max_grads[k])
    kept = [sample[k] for k in order[:n_keep]]
    print(f"\nKeeping {n_keep} lowest-max-gradient frames:")
    for path in kept:
        print(f"  {path.name}")
    print()

    # ---- Pass 2: per-pixel mean / std (running float64 accumulators) ---
    acc_sum: torch.Tensor | None = None
    acc_sumsq: torch.Tensor | None = None
    count = 0
    for i, path in enumerate(kept, 1):
        mag = grad_magnitude(path, device, wrong_mask).to(torch.float64)
        if acc_sum is None:
            acc_sum = torch.zeros_like(mag)
            acc_sumsq = torch.zeros_like(mag)
        elif mag.shape != acc_sum.shape:
            raise SystemExit(
                f"Frame {path.name} has gradient shape {tuple(mag.shape)} != "
                f"{tuple(acc_sum.shape)}; per-pixel aggregation needs identical dimensions."
            )
        acc_sum += mag
        acc_sumsq += mag * mag
        count += 1
        print(f"[pass2 {i}/{n_keep}] {path.name}  accumulated")
        del mag

    mean = acc_sum / count
    var = (acc_sumsq - count * mean * mean) / max(count - 1, 1)  # ddof=1
    std = torch.sqrt(var.clamp_min(0.0))

    mean_np = mean.to(torch.float32).cpu().numpy()
    std_np = std.to(torch.float32).cpu().numpy()
    np.save(MEAN_PATH, mean_np)
    np.save(STD_PATH, std_np)
    print(f"\nSaved mean -> {MEAN_PATH}  shape={mean_np.shape} dtype={mean_np.dtype}")
    print(f"Saved std  -> {STD_PATH}  shape={std_np.shape} dtype={std_np.dtype}")

    # ---- PNG: clip(mean - std, 0) scaled to [0, 255] -------------------
    assert np.all(mean_np >= 0), "mean has negative pixels; expected all >= 0"
    diff = np.clip(mean_np - std_np, 0.0, None)
    peak = float(diff.max())
    scaled = diff / peak * 255.0 if peak > 0 else diff
    img = scaled.astype(np.uint8)
    Image.fromarray(img, mode="L").save(PNG_PATH)
    print(f"Saved png  -> {PNG_PATH}  shape={img.shape} dtype={img.dtype} (peak={peak:.6f})")


if __name__ == "__main__":
    main()
