#!/usr/bin/env python3
"""Synthetic check of the response-curve recovery. No GPU, ~10 s.

Plants a known response across the real 17-exposure ladder with 1/255 noise and per-exposure
scale errors, then asserts the solver recovers both.  This is the first verification step of
PHOTOMETRY_SPEC.md §3.7 and it is worth keeping cheap: every trap in §3.5 that was hit on
real data shows up here first, and here it costs seconds instead of a pipeline run.

Run directly (`python v5/test_calib_synthetic.py`) or under pytest.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v5 import calib as CA  # noqa: E402

# The real bracket, 1/4000 s to 2 s.
EXPOSURES = np.array([
    0.00025, 0.0005, 0.001, 0.0015625, 0.002, 0.004, 0.008, 0.01666666667, 0.025,
    0.03333333333, 0.05, 0.06666666667, 0.125, 0.25, 0.5, 1.0, 2.0,
])

N_SAMPLES = 3000
VALUE_NOISE = 1.0 / 255.0
EXPOSURE_ERROR_SIGMA = 0.05
S_CURVE_BETA = 0.5
SEED = 7


def srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def planted_f(v, beta: float = S_CURVE_BETA):
    """The planted response: sRGB decode followed by a smoothstep S-curve.

    Monotone over [0,1] and maps 0->0, 1->1 — trap 3 in the spec: a planted curve that
    leaves [0,1] clips at the ends, is not invertible, and makes the whole test vacuous.
    Deliberately *not* a power law, since a power law is exactly what the old per-pair
    exponent assumed and what this method exists to stop assuming.
    """
    u = srgb_to_linear(v)
    return (1.0 - beta) * u + beta * (3.0 * u ** 2 - 2.0 * u ** 3)


_INV_GRID = np.linspace(0.0, 1.0, 200001)
_INV_F = planted_f(_INV_GRID)


def planted_f_inv(a):
    """Stored value that would produce accumulated light `a` (the camera's encoding)."""
    return np.interp(np.asarray(a, dtype=np.float64), _INV_F, _INV_GRID)


def make_synthetic(seed: int = SEED):
    rng = np.random.default_rng(seed)
    # Brightnesses spanning what the bracket can hold: every pixel is well exposed somewhere.
    E = np.exp(rng.uniform(math.log(0.2), math.log(4000.0), N_SAMPLES))
    ln_c_true = rng.normal(0.0, EXPOSURE_ERROR_SIGMA, EXPOSURES.size)
    ln_c_true -= ln_c_true.mean()          # only relative corrections are identifiable
    t_eff = EXPOSURES * np.exp(ln_c_true)

    A = np.clip(E[None, :] * t_eff[:, None], 0.0, 1.0)   # accumulated light, sensor-clipped
    V = planted_f_inv(A)
    V = np.clip(V + rng.normal(0.0, VALUE_NOISE, V.shape), 0.0, 1.0)
    valid = np.ones_like(V, dtype=bool)
    return V, valid, ln_c_true, E


def curve_error(result, lo=None, hi=None):
    """|f_rec/f_true − 1| over the fitted range, anchored at v = 0.5.

    Anchoring mid-range, never at v=1: only [FIT_VALUE_LO, FIT_VALUE_HI] is constrained, so
    normalising at the extrapolated top smears extrapolation error over everything else and
    made a 0.3%-accurate fit look 5% wrong (trap 2).
    """
    lo = CA.FIT_VALUE_LO if lo is None else lo
    hi = CA.FIT_VALUE_HI if hi is None else hi
    v, f_rec = CA.response_lut(result)
    f_true = planted_f(v)
    anchor = np.exp(result.ln_f(0.5)) / planted_f(0.5)
    rel = np.full_like(v, np.nan)
    nz = f_true > 0                       # f_true(0) == 0 exactly; no relative error there
    rel[nz] = np.abs((f_rec[nz] / anchor) / f_true[nz] - 1.0)
    sel = (v >= lo) & (v <= hi) & nz
    return rel, v, sel


def report_bands(result):
    rel, v, ok = curve_error(result)
    print("  recovered-curve error by stored-value band (this is what sets FIT_VALUE_LO):")
    for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.95), (0.95, 0.98)]:
        sel = (v >= lo) & (v < hi) & np.isfinite(rel)
        print(f"    v in [{lo:.2f},{hi:.2f})   {np.median(rel[sel]):7.2%} median, "
              f"{rel[sel].max():7.2%} max")


def report_pair_multiplier(result):
    """What a single 2x exposure step really does to a stored value, across the range.

    v2 fitted one number per adjacent pair. If the response were a power law this ratio would
    be constant; it is not, which is the direct demonstration that no single exponent can
    describe two frames.
    """
    v, f = CA.response_lut(result)
    print("  implied stored-value multiplier for one 2x exposure step:")
    ratios = []
    for v0 in [0.12, 0.20, 0.30, 0.40, 0.50, 0.60]:
        f0 = np.interp(v0, v, f)
        v1 = np.interp(2.0 * f0, f, v)
        ratios.append(v1 / v0)
        print(f"    v0 = {v0:.2f} -> v1 = {v1:.4f}   (x{v1 / v0:.4f})")
    spread = max(ratios) / min(ratios) - 1.0
    print(f"    spread across the range: {spread:.1%}  (a power law would give 0%)")


def test_synthetic_response_recovery():
    V, valid, ln_c_true, _E = make_synthetic()

    print(f"synthetic: {EXPOSURES.size} exposures x {N_SAMPLES} pixels, "
          f"planted correction spread {np.exp(ln_c_true).max() / np.exp(ln_c_true).min() - 1:.1%}")

    nominal = CA.calibrate(V, valid, EXPOSURES, n_exposure_refine=0, verbose=False)
    refined = CA.calibrate(V, valid, EXPOSURES, n_exposure_refine=CA.N_EXPOSURE_REFINE,
                           verbose=False)

    rel_r, _, sel = curve_error(refined)
    rel_n, _, _ = curve_error(nominal)
    med_refined = float(np.median(rel_r[sel]))
    med_nominal = float(np.median(rel_n[sel]))

    ln_c_rec = refined.ln_c - refined.ln_c.mean()
    corr_err = float(np.max(np.abs(np.exp(ln_c_rec - ln_c_true) - 1.0)))
    mono = CA.monotonicity_report(refined)

    print(f"  lsqr istop                      {refined.istop}")
    print(f"  median curve error (fit range)  {med_refined:.2%}   (nominal times: {med_nominal:.2%})")
    print(f"  max exposure-correction error   {corr_err:.2%}")
    print(f"  non-monotone steps in fit range {mono['n_non_monotone_in_fit_range']}")
    report_bands(refined)
    report_pair_multiplier(refined)

    assert refined.istop != 3, refined.istop
    assert med_refined < 0.02, med_refined
    assert corr_err < 0.04, corr_err
    assert mono["n_non_monotone_in_fit_range"] == 0, mono
    # Refinement must not make the curve worse than the honest nominal-times fit.
    assert med_refined <= med_nominal + 1e-3, (med_refined, med_nominal)


if __name__ == "__main__":
    test_synthetic_response_recovery()
    print("OK")
