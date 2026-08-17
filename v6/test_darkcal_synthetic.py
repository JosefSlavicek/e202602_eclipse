#!/usr/bin/env python3
"""Synthetic check of the dark-current model recovery. No GPU, no files, ~1 s.

Plants a known per-pixel bias/rate map (with a few hot pixels) across a 39-exposure ladder
like the real darks/ bracket, with per-exposure shutter-time errors like the real camera's
(darkcal.py deliberately never tries to correct for these — see its module docstring for
why matched-by-label subtraction doesn't need to), and asserts
`fit_dark_model_from_stack` recovers bias/rate to noise-level accuracy anyway.

Run directly (`python v6/test_darkcal_synthetic.py`) or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6 import darkcal as DC  # noqa: E402

H, W = 48, 32
N_PER_EXPOSURE = 3
READ_NOISE_SIGMA = 2.0 / 16383.0          # ~READ_SIGMA in inputs.py
TIMING_ERROR_SIGMA = 0.08                  # real (unmodelled) shutter-time error, same order
                                            # as calib.py's fitted corrections; not corrected
                                            # for here — see darkcal.py's module docstring
SEED = 11

# Same spread as the real darks/ bracket (1/8000 s .. 0.8 s), 39 settings.
EXPOSURES = np.exp(np.linspace(np.log(1.0 / 8000.0), np.log(0.8), 39))


def make_synthetic(seed: int = SEED):
    rng = np.random.default_rng(seed)

    bias_true = rng.uniform(0.0005, 0.004, size=(H, W)).astype(np.float64)
    rate_true = rng.uniform(0.0002, 0.003, size=(H, W)).astype(np.float64)
    # A handful of hot pixels: rate an order of magnitude above the bulk.
    hot_ij = [(3, 5), (10, 20), (30, 7)]
    for i, j in hot_ij:
        rate_true[i, j] = rng.uniform(0.05, 0.2)

    # The shutter's *true* duration differs from what EXIF reports, by a per-exposure
    # factor — same phenomenon calib.py corrects for on the light bracket. darkcal.py does
    # not try to correct for it: dark frames are only ever matched to lights by the same
    # reported label, so the true duration never needs to be known here.
    ln_err_true = rng.normal(0.0, TIMING_ERROR_SIGMA, EXPOSURES.size)
    true_duration = EXPOSURES * np.exp(ln_err_true)

    times_reported = np.repeat(EXPOSURES, N_PER_EXPOSURE)      # what the fit is given
    times_true = np.repeat(true_duration, N_PER_EXPOSURE)      # what actually happened

    frames = []
    for t in times_true:
        y = bias_true + rate_true * t + rng.normal(0.0, READ_NOISE_SIGMA, size=(H, W))
        frames.append(y.astype(np.float32))

    return bias_true, rate_true, hot_ij, times_reported, frames


def test_synthetic_dark_recovery():
    bias_true, rate_true, hot_ij, times_reported, frames = make_synthetic()

    print(f"synthetic: {EXPOSURES.size} exposures x {N_PER_EXPOSURE} frames = {len(frames)}")

    bias, rate, rms = DC.fit_dark_model_from_stack(times_reported, frames)

    bias_err = float(np.max(np.abs(bias - bias_true)))
    rate_err = float(np.max(np.abs(rate - rate_true)))
    print(f"  max bias err {bias_err:.5f}, max rate err {rate_err:.5f}, residual rms {rms:.6f}")

    for i, j in hot_ij:
        print(f"  hot pixel ({i},{j}): true {rate_true[i, j]:.4f}, recovered {rate[i, j]:.4f}")
        assert abs(rate[i, j] - rate_true[i, j]) < 0.015, (i, j, rate[i, j], rate_true[i, j])

    assert bias_err < 5 * READ_NOISE_SIGMA, bias_err
    assert rate_err < 0.015, rate_err
    assert rms < 3 * READ_NOISE_SIGMA, rms


def _expect_assertion(fn, needle: str):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), e
        return
    raise AssertionError(f"expected an AssertionError containing {needle!r}, got none")


def test_min_frames_and_degeneracy_assertions():
    rng = np.random.default_rng(0)
    frame = rng.normal(0, 0.01, size=(8, 8)).astype(np.float32)

    _expect_assertion(
        lambda: DC.fit_dark_model_from_stack([1.0, 2.0], [frame, frame]), "need >=")
    _expect_assertion(
        lambda: DC.fit_dark_model_from_stack([1.0, 1.0, 1.0], [frame, frame, frame]),
        "same exposure time")


if __name__ == "__main__":
    test_synthetic_dark_recovery()
    test_min_frames_and_degeneracy_assertions()
    print("OK")
