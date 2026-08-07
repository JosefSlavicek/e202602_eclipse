#!/usr/bin/env python
"""Round 3 of v6/plan0.md: the trend fit and inverse-warp machinery in eclipse_v6.atmosphere.

Pure synthetic, no images, no GPU, ~1 s. Plants several "calibration frames" spread over a
totality-length window with qx,qy drifting *linearly* in time (same shape as the real
atmosphere signal is expected to have), fits each frame independently with the round-1/2
machinery, runs fit_trend, and checks the recovered Trend against the truth two ways:

1. analytically -- project()'s qx,qy branch is `x += qx * u(x)**2` with u independent of
   qx, so d(x)/dt = u**2 * dqx/dt *exactly*, no finite-differencing needed;
2. structurally -- to_reference inverts to_observed, and sample_indices is to_observed
   evaluated on the pixel grid (the same two checks the deleted v1 atmosphere.py test used,
   now against the fitted-from-data Trend rather than a hand-built one).

Run directly (`python v6/test_atmosphere_trend.py`) or under pytest.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "find_stars"))

import starlib as S  # noqa: E402
from eclipse_v6 import atmosphere as A  # noqa: E402

SEED = 20260812
N_STARS = 50
N_CALIB = 6
TOTALITY_S = 101.6

RA0, DEC0 = 142.105, 14.907
SCALE0 = 3.061
CX0, CY0 = 3024.0, 2012.0
ROT_TRUE, PARITY_TRUE = 15.0, 1
K1_TRUE, K2_TRUE = 0.02, -0.01
QX0, QX_SLOPE = 4.0, 0.03      # px, px/s -- the atmosphere term drifting over totality
QY0, QY_SLOPE = -3.0, -0.02


def p_base():
    return np.concatenate([S.affine_from(SCALE0, ROT_TRUE, PARITY_TRUE, CX0, CY0),
                           [K1_TRUE, K2_TRUE]])


def make_calibration_set(seed=SEED):
    rng = np.random.default_rng(seed)
    dec = DEC0 + rng.uniform(-1.7, 1.7, N_STARS)
    ra = RA0 + rng.uniform(-2.6, 2.6, N_STARS) / math.cos(math.radians(DEC0))
    xi, eta = S.gnomonic(ra, dec, RA0, DEC0)
    times = np.linspace(-0.5 * TOTALITY_S, 0.5 * TOTALITY_S, N_CALIB)
    pb = p_base()
    calib = []
    for t in times:
        qx, qy = QX0 + QX_SLOPE * t, QY0 + QY_SLOPE * t
        p_true = np.concatenate([pb, [qx, qy]])
        det_x, det_y = S.project(p_true, xi, eta)
        res = A.fit_calibration_frame(det_x, det_y, ra, dec, RA0, DEC0, CX0, CY0, SCALE0, pb)
        assert res["nmatch"] == N_STARS and res["rms"] < 1e-6, (res["nmatch"], res["rms"])
        calib.append((t, res["p"]))
    return calib, pb


def test_trend_matches_the_analytic_slope():
    calib, pb = make_calibration_set()
    times = [t for t, _ in calib]
    ps = [p for _, p in calib]
    grid = A.reference_grid(CX0, CY0, 2800.0, 1900.0, n=11)
    trend = A.fit_trend(times, ps, pb, grid)

    print(f"fitted t_mean = {trend.t_mean:+.3f} (true 0.000)")
    assert abs(trend.t_mean) < 1e-6, trend.t_mean

    # Analytic truth: x_final = x_pre + qx * u**2 with u = (x_pre - DIST_C[0])/DIST_NORM
    # independent of qx (see project()), so dx/dt = u**2 * QX_SLOPE exactly. Pick the probe
    # points as (xi, eta) directly -- x_pre, y_pre = project(pb, xi, eta) needs no
    # inversion, unlike going the other way from an arbitrary pixel grid.
    dec_p = DEC0 + np.linspace(-1.6, 1.6, 25)
    ra_p = RA0 + np.linspace(-2.5, 2.5, 25) / math.cos(math.radians(DEC0))
    xi_p, eta_p = S.gnomonic(ra_p, dec_p, RA0, DEC0)
    x_pre, y_pre = S.project(pb, xi_p, eta_p)
    ux = (x_pre - S.DIST_C[0]) / S.DIST_NORM
    uy = (y_pre - S.DIST_C[1]) / S.DIST_NORM
    slope_x_true = ux ** 2 * QX_SLOPE
    slope_y_true = uy ** 2 * QY_SLOPE

    dx1, dy1 = trend.correction(1.0, x_pre, y_pre)  # correction() = slope*(t-t_mean); t_mean=0
    err_x = np.abs(dx1 - slope_x_true)
    err_y = np.abs(dy1 - slope_y_true)
    print(f"slope error: x max {err_x.max():.4f} px/s, y max {err_y.max():.4f} px/s")
    assert err_x.max() < 0.01, err_x.max()
    assert err_y.max() < 0.01, err_y.max()

    # And at a time well past the calibration window -- v6 will apply this to short,
    # star-free frames that may fall outside [t_first, t_last]. The per-second error above
    # is the grid's own discretization bias, not a growing extrapolation error -- it scales
    # with dt like anything multiplied by dt would, so the bound scales too.
    t_future = 0.7 * TOTALITY_S
    dxf, dyf = trend.correction(t_future, x_pre, y_pre)
    print(f"at t={t_future:.1f}s: x max {np.abs(dxf - ux**2*QX_SLOPE*t_future).max():.4f} px, "
          f"y max {np.abs(dyf - uy**2*QY_SLOPE*t_future).max():.4f} px")
    assert np.abs(dxf - ux ** 2 * QX_SLOPE * t_future).max() < 0.05
    assert np.abs(dyf - uy ** 2 * QY_SLOPE * t_future).max() < 0.05


def test_inverse_and_sampling_are_consistent():
    calib, pb = make_calibration_set()
    grid = A.reference_grid(CX0, CY0, 2800.0, 1900.0, n=11)
    trend = A.fit_trend([t for t, _ in calib], [p for _, p in calib], pb, grid)

    x = CX0 + np.linspace(-2500.0, 2500.0, 31)
    y = CY0 + np.linspace(-1700.0, 1700.0, 31)
    for t in (-40.0, 0.0, 55.0):
        ox, oy = trend.to_observed(t, x, y)
        rx, ry = trend.to_reference(t, ox, oy)
        assert np.abs(rx - x).max() < 1e-6 and np.abs(ry - y).max() < 1e-6, t

    cols, rows = np.meshgrid(np.arange(200, dtype=float), np.arange(150, dtype=float))
    sx, sy = trend.sample_indices(20.0, 150, 200)
    ox2, oy2 = trend.to_observed(20.0, cols, rows)
    assert np.array_equal(sx, ox2) and np.array_equal(sy, oy2)


if __name__ == "__main__":
    test_trend_matches_the_analytic_slope()
    test_inverse_and_sampling_are_consistent()
    print("OK")
