#!/usr/bin/env python3
"""End-to-end smoke test of calibration -> load_radiance -> merge, on a small synthetic scene.

Fast (a few seconds on one GPU, ~500x700 frames) and deliberately not about accuracy: it
exercises the plumbing that the synthetic curve test cannot — sample gathering through the
registration chain, the source's radiance contract, the inverse-variance merge, the moon
blanking and the NO_DATA sentinel — on data whose right answer is known exactly.

The planted scene is a corona-like 1/r^3 falloff around a moon disk, encoded through a
non-power-law response, so the merge's output should reproduce it up to a global scale (the
calibration's gauge fixes brightness only up to a constant factor).

Run directly (`python v6/test_merge_smoke.py`) or under pytest. Needs CUDA.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6 import calib as CA                       # noqa: E402
from eclipse_v6.inputs import FrameSource                # noqa: E402
from eclipse_v6.merge import NO_DATA, merge_to_composite  # noqa: E402
from eclipse_v6.stage0 import ImageInfo                  # noqa: E402
from eclipse_v6.stage3 import Stage3Context              # noqa: E402

H, W = 500, 700
MOON_I, MOON_J, MOON_R = 250.0, 350.0, 60.0
EXPOSURES = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064]
FRAMES_PER_EXPOSURE = 2
S_CURVE_BETA = 0.5


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def planted_f(v):
    u = srgb_to_linear(np.asarray(v, dtype=np.float64))
    return (1.0 - S_CURVE_BETA) * u + S_CURVE_BETA * (3.0 * u ** 2 - 2.0 * u ** 3)


_GRID = np.linspace(0.0, 1.0, 100001)
_F = planted_f(_GRID)


def encode(a):
    return np.interp(np.clip(a, 0.0, 1.0), _F, _GRID)


def true_radiance():
    """A corona: bright at the limb, falling as 1/r^3, plus a mild angular ripple."""
    ii = np.arange(H, dtype=np.float64)[:, None]
    jj = np.arange(W, dtype=np.float64)[None, :]
    r = np.hypot(ii - MOON_I, jj - MOON_J)
    th = np.arctan2(ii - MOON_I, jj - MOON_J)
    rn = np.maximum(r / MOON_R, 1.0)
    return 60.0 * rn ** -3.0 * (1.0 + 0.25 * np.cos(3.0 * th))


class SyntheticSource(FrameSource):
    """A JPEG-like source: 8-bit stored values from a known response, with a known moon."""

    kind = "synthetic"
    is_linear = False
    value_sigma = 1.0 / 255.0

    def __init__(self):
        self.L = true_radiance()
        self.ln_c_true = np.array([0.03, -0.02, 0.05, -0.04, 0.01, -0.01, -0.02])

    def scan(self):
        raise NotImplementedError("synthetic frames are built by make_groups()")

    def load_gray(self, ii, device):
        t_eff = ii.exposure_time * math.exp(self.ln_c_true[ii.link])
        v = encode(self.L * t_eff)
        rng = np.random.default_rng(abs(hash(str(ii.path))) % (2 ** 32))
        v = np.clip(v + rng.normal(0.0, 1.0 / 255.0, v.shape), 0.0, 1.0)
        v = np.round(v * 255.0) / 255.0            # 8-bit quantization, as a real JPEG
        rows = np.arange(H, dtype=np.float64)[:, None]
        cols = np.arange(W, dtype=np.float64)[None, :]
        v[np.hypot(rows - MOON_I, cols - MOON_J) <= MOON_R] = 0.0
        return torch.from_numpy(v.astype(np.float32)).to(device=device, dtype=torch.float32)

    def load_radiance(self, ii, device):
        # Same contract as JpgSource, reusing its implementation through the base class LUTs.
        from eclipse_v6.inputs import JpgSource
        return JpgSource.load_radiance(self, ii, device)


def make_groups(source):
    """Exposure groups, poses and cross-exposure registration for a perfectly aligned scene."""
    groups, opt_results = {}, {}
    for k, t in enumerate(EXPOSURES):
        group = []
        for j in range(FRAMES_PER_EXPOSURE):
            info = ImageInfo(
                path=Path(f"synthetic_{k}_{j}.jpg"), width=W, height=H,
                avg_brightness=0.1, timestamp=float(k * 10 + j), exposure_time=float(t),
                moon=(MOON_I, MOON_J, MOON_R),
            )
            info.source = source
            info.link = k                     # which planted shutter error this frame carries
            group.append(info)
        groups[float(t)] = group
        opt_results[float(t)] = {
            "abs_xy": np.zeros((FRAMES_PER_EXPOSURE, 2), dtype=np.float32),
            "abs_angle_t": np.zeros(FRAMES_PER_EXPOSURE, dtype=np.float32),
        }
    cross_reg = {
        (EXPOSURES[i], EXPOSURES[i + 1]): (0.0, 0.0, 0.0) for i in range(len(EXPOSURES) - 1)
    }
    return groups, opt_results, cross_reg


def test_merge_smoke():
    assert torch.cuda.is_available(), "this smoke test needs CUDA"
    device = torch.device("cuda")
    source = SyntheticSource()
    groups, opt_results, cross_reg = make_groups(source)
    exposure_times_sorted = sorted(groups.keys())
    t_ref = exposure_times_sorted[0]

    V, valid, exposures, sample_ij = CA.gather_samples(
        groups, opt_results, exposure_times_sorted, cross_reg, t_ref, device,
        n_radial_bins=24, samples_per_bin=120,
    )
    assert V.shape[0] == len(EXPOSURES), V.shape
    assert valid.any(), "no sample was valid in any exposure"

    result = CA.calibrate(V, valid, exposures, verbose=False)
    CA.print_report(result)
    assert result.istop != 3, result.istop

    source.set_calibration(result)
    ctx = Stage3Context(workdir=Path("/tmp"), device=device)
    ctx.exposure_groups = groups
    ctx.opt_results = opt_results
    ctx.cross_reg = cross_reg
    ctx.exposure_times_sorted = exposure_times_sorted
    ctx.t_ref = t_ref
    merge_to_composite(ctx, source)

    comp = ctx.composite
    assert comp.shape == (H, W), comp.shape
    assert ctx.composite_variance.shape == (H, W)

    # The moon disk must be blanked, and nothing outside it should be missing.
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    inside_moon = np.hypot(rows - MOON_I, cols - MOON_J) <= MOON_R - 3
    assert np.all(comp[inside_moon] == NO_DATA), int((comp[inside_moon] != NO_DATA).sum())

    L_true = true_radiance()
    # Compare where the bracket can actually see: away from the limb, inside the frame.
    r = np.hypot(rows - MOON_I, cols - MOON_J)
    band = (r > MOON_R + 12) & (r < 220) & (comp > 0)
    ratio = comp[band] / L_true[band]
    scale = float(np.median(ratio))
    rel = np.abs(ratio / scale - 1.0)
    print(f"  merged/planted: global scale {scale:.4g}, "
          f"median deviation {np.median(rel):.3%}, p95 {np.percentile(rel, 95):.3%}")
    # Outside the blanked disk (moon radius + the 2 px limb margin, plus a pixel of warp
    # slack) nothing should be missing. Measuring from MOON_R itself would count the margin
    # annulus as a gap and read as a failure that is not one.
    outside_blank = r > MOON_R + 4
    n_gap = int(((comp <= NO_DATA) & outside_blank & (r < 200)).sum())
    print(f"  coverage gaps outside the blanked disk, r<200: {n_gap}")
    assert n_gap == 0, n_gap

    assert band.sum() > 10000, int(band.sum())
    assert np.median(rel) < 0.05, float(np.median(rel))
    assert np.percentile(rel, 95) < 0.20, float(np.percentile(rel, 95))


if __name__ == "__main__":
    test_merge_smoke()
    print("OK")
