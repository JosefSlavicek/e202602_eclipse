#!/usr/bin/env python
"""Round 5 of v6/plan0.md: the differential per-channel dispersion measurement.

Pure synthetic Gaussian stars (no GPU, ~1 s): renders a star at a slightly different
sub-pixel position per channel, checks dispersion_offset() recovers the KNOWN (mean-
removed) per-channel offset, then checks fit_dispersion_trend() recovers a known linear
drift across several synthetic calibration frames -- same shape as test_atmosphere_trend.py,
just for the per-channel scalar case instead of the spatial grid.

Run directly (`python v6/test_dispersion.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "find_stars"))

from eclipse_v6 import atmosphere as A  # noqa: E402

SIZE = 41
CENTER = 20.0
SIGMA = 1.6
AMP = 500.0
BG = 20.0


def render_star(dx, dy, size=SIZE, rng=None):
    """A single Gaussian star at (CENTER+dx, CENTER+dy), plus a little read noise."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = BG + AMP * np.exp(-0.5 * (((xx - CENTER - dx) / SIGMA) ** 2
                                    + ((yy - CENTER - dy) / SIGMA) ** 2))
    if rng is not None:
        img = img + rng.normal(0.0, 1.5, img.shape)
    return img


def test_dispersion_offset_recovers_known_shift():
    # R shifted +0.30 px in x, B shifted -0.25 px in x (opposite signs, like real
    # dispersion straddling G); small y shifts too, to check both axes at once.
    true_shift = {"R": (0.30, 0.05), "G": (0.0, 0.0), "B": (-0.25, -0.04)}
    rgb = np.stack([render_star(*true_shift[name]) for name in A.CHANNEL_NAMES], axis=-1)

    off = A.dispersion_offset(rgb, CENTER, CENTER, half=8)
    assert off is not None
    mean_dx = sum(v[0] for v in true_shift.values()) / 3.0
    mean_dy = sum(v[1] for v in true_shift.values()) / 3.0
    for name in A.CHANNEL_NAMES:
        want_x = true_shift[name][0] - mean_dx
        want_y = true_shift[name][1] - mean_dy
        got_x, got_y = off[name]
        print(f"{name}: want ({want_x:+.3f}, {want_y:+.3f})  got ({got_x:+.3f}, {got_y:+.3f})")
        assert abs(got_x - want_x) < 0.01, (name, got_x, want_x)
        assert abs(got_y - want_y) < 0.01, (name, got_y, want_y)


def test_dispersion_trend_recovers_linear_drift():
    rng = np.random.default_rng(20260812)
    slope_true = {"R": (0.004, -0.001), "G": (0.0, 0.0), "B": (-0.003, 0.0015)}
    times = np.linspace(-50.0, 50.0, 6)
    offsets, ts = [], []
    for t in times:
        shift = {name: (slope_true[name][0] * t, slope_true[name][1] * t)
                 for name in A.CHANNEL_NAMES}
        rgb = np.stack([render_star(*shift[name], rng=rng) for name in A.CHANNEL_NAMES], axis=-1)
        off = A.dispersion_offset(rgb, CENTER, CENTER, half=8)
        if off is not None:
            offsets.append(off)
            ts.append(t)
    trend = A.fit_dispersion_trend(ts, offsets)
    print(f"fitted t_mean = {trend.t_mean:+.3f}")
    for name in A.CHANNEL_NAMES:
        sx, sy = trend.slope[name]
        tx, ty = slope_true[name]
        print(f"{name}: slope want ({tx:+.5f}, {ty:+.5f})  got ({sx:+.5f}, {sy:+.5f})")
        assert abs(sx - tx) < 0.002, (name, sx, tx)
        assert abs(sy - ty) < 0.002, (name, sy, ty)

    # apply_dispersion sign convention: a channel whose fitted centroid DRIFTED by (dx,dy)
    # from t_mean to t must land back at the reference centroid after correction.
    t_probe = 30.0
    shift = {name: (slope_true[name][0] * t_probe, slope_true[name][1] * t_probe)
            for name in A.CHANNEL_NAMES}
    rgb = np.stack([render_star(*shift[name]) for name in A.CHANNEL_NAMES], axis=-1)
    fixed = A.apply_dispersion(rgb, t_probe, trend)
    off_before = A.dispersion_offset(rgb, CENTER, CENTER, half=8)
    off_after = A.dispersion_offset(fixed, CENTER, CENTER, half=8)
    before_spread = max(abs(v[0]) for v in off_before.values())
    after_spread = max(abs(v[0]) for v in off_after.values())
    print(f"channel spread before {before_spread:.3f} px, after correction {after_spread:.4f} px")
    # Not exact -- trend.slope itself carries the fit's small noise-driven error (visible
    # above, a few 1e-4 px/s), which scales with t_probe like anything multiplied by dt
    # would (same reasoning as test_atmosphere_trend.py's far-future check). The point of
    # this assertion is "collapsed to near nothing", not "exactly zero".
    assert after_spread < 0.02


if __name__ == "__main__":
    test_dispersion_offset_recovers_known_shift()
    test_dispersion_trend_recovers_linear_drift()
    print("OK")
