"""Per-frame overburn mask + merge-intensity window, measured once from unmodified raw.

Computed at ingestion — before dark subtraction, before flat-field division, before anything
else touches the frame — directly from the decoded raw counts. Saturation is a property of
the physical sensor's raw ADC reading; deciding it here, rather than after dark/flat
correction, means the answer can't be nudged across the threshold by those later corrections.

Two products per light frame, cached to disk and never recomputed downstream:

  overburn  (H, W) bool   — decoded >= RAW_OVERBURN_HI. The only hard exclusion: a saturated
                            pixel's true value is lost, so it is dropped from the within-
                            exposure plain average outright, not merely down-weighted.
  weight    (H, W) float  — a continuous window over the same decoded value (see `window`):
                            full trust in the middle of the range, tapering to (not below) a
                            small floor near 0 and RAW_OVERBURN_HI. This — not anything
                            derived from calibrated radiance — is the merge weight. It is
                            carried through both the intra-exposure and cross-exposure warps
                            alongside the image data (see merge.average_exposure_radiance),
                            so it never needs its own [0, 1/t_eff]-style clamp: it was never
                            in radiance units to begin with.

`run` persists (decoded, overburn, weight) per frame so the decode happens exactly once;
`apply_corrections` reads the cached `decoded` back to bake in dark/flat correction (producing
the values load_gray/load_radiance actually read), so the NEF is never decoded twice.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tqdm

from eclipse_v7.inputs import _decode_nef_linear

RAW_OVERBURN_HI = 0.99       # decoded raw fraction at/above which a pixel is saturated
WINDOW_LO = 0.0
WINDOW_PLATEAU = 0.5         # fraction of [lo, hi] held at full weight; the outer quarters
                              # taper smoothly (Tukey-style) instead of a hard cutoff
WINDOW_EPS = 1.0e-4          # weight floor added to every window() output so it is never
                              # exactly 0 at/beyond lo or hi, only ever small


def window(x: np.ndarray, lo: float, hi: float, plateau: float = WINDOW_PLATEAU) -> np.ndarray:
    """Continuous weight over [lo, hi]: 1.0 on the middle `plateau` fraction, cosine taper
    down to (but never below) `WINDOW_EPS` at lo/hi and beyond. Plain numpy — this runs once
    per raw frame on the CPU decode, never on the GPU.
    """
    edge = (1.0 - plateau) / 2.0
    u = (x - lo) / (hi - lo)
    ramp = 0.5 * (1.0 - np.cos(math.pi * np.clip(u / edge, 0.0, 1.0)))
    fall = 0.5 * (1.0 - np.cos(math.pi * np.clip((1.0 - u) / edge, 0.0, 1.0)))
    w = np.minimum(ramp, fall)
    return np.where((u > 0.0) & (u < 1.0), w, 0.0) + WINDOW_EPS


def decode_and_measure(path: Path):
    """(decoded, overburn, weight) for one light NEF, straight off the unmodified raw."""
    decoded = _decode_nef_linear(path)
    overburn = decoded >= RAW_OVERBURN_HI
    weight = window(decoded, WINDOW_LO, RAW_OVERBURN_HI).astype(np.float32)
    return decoded, overburn, weight


def _cache_path(cache_dir: Path, ii) -> Path:
    return Path(cache_dir) / f"{ii.path.stem}.npz"


def _corrected_path(corrected_dir: Path, ii) -> Path:
    return Path(corrected_dir) / f"{ii.path.stem}.npy"


def run(light_infos, cache_dir: Path) -> None:
    """Decode + measure every light frame not already cached. Idempotent."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [ii for ii in light_infos if not _cache_path(cache_dir, ii).is_file()]
    print(f"rawprep: {len(light_infos) - len(todo)} frame(s) already cached, "
          f"{len(todo)} to decode")
    for ii in tqdm.tqdm(todo, desc="rawprep decode"):
        decoded, overburn, weight = decode_and_measure(ii.path)
        np.savez(
            _cache_path(cache_dir, ii),
            decoded=decoded.astype(np.float32), overburn=overburn, weight=weight,
        )


def apply_corrections(light_infos, cache_dir: Path, corrected_dir: Path, *,
                       dark_model=None, flat_model=None) -> None:
    """Bake dark/flat correction into each cached raw decode, save the corrected array.

    Skips frames already corrected, so resuming a run does not re-touch every file. Reads
    `decoded` back from the rawprep cache (never re-decodes the NEF).
    """
    corrected_dir = Path(corrected_dir)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    todo = [ii for ii in light_infos if not _corrected_path(corrected_dir, ii).is_file()]
    print(f"rawprep: {len(light_infos) - len(todo)} frame(s) already corrected, "
          f"{len(todo)} to process"
          + (" (dark)" if dark_model is not None else "")
          + (" (flat)" if flat_model is not None else ""))
    for ii in tqdm.tqdm(todo, desc="rawprep correct"):
        with np.load(_cache_path(cache_dir, ii)) as z:
            corrected = z["decoded"].astype(np.float64)
        if dark_model is not None:
            corrected = corrected - (
                dark_model.bias.astype(np.float64)
                + dark_model.rate.astype(np.float64) * float(ii.exposure_time)
            )
        if flat_model is not None:
            corrected = flat_model.apply_masked(corrected)
        np.save(_corrected_path(corrected_dir, ii), corrected.astype(np.float32))


def load_measurements(cache_dir: Path, ii):
    """(overburn, weight) numpy arrays for one frame, from the rawprep cache."""
    with np.load(_cache_path(cache_dir, ii)) as z:
        return z["overburn"], z["weight"]


def load_corrected(corrected_dir: Path, ii) -> np.ndarray:
    """The dark/flat-corrected decoded array for one frame."""
    return np.load(_corrected_path(corrected_dir, ii))
