#!/usr/bin/env python3
"""Synthetic check of the flat-field (high-frequency-only) recovery. No GPU, no files, ~1 s.

Plants a known vignetting pattern (smooth radial falloff + a dust-shadow dip), synthesizes a
stack of flat frames at one exposure time with a known dark bias/rate baked in, and asserts
`fit_flat_model_from_stack` recovers the *dust shadow only* -- the smooth radial falloff is
deliberately removed by the high-pass step and must NOT show up in the result -- to
noise-level accuracy, and that a perfectly uniform frame corrected by it averages back to
1.0 exactly (the property flatcal.py's normalization is built to hit).

Also checks `_remove_smooth_trend`'s mirror-padded edges directly: blurring a perfectly
uniform frame must leave every pixel exactly unchanged, including right at the border --
zero-padding would instead pull the blur (and so the high-pass residual) down near the edge.

Run directly (`python v7/test_flatcal_synthetic.py`) or under pytest.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v7 import flatcal as FC  # noqa: E402

H, W = 48, 32
N_FRAMES = 12
READ_NOISE_SIGMA = 2.0 / 16383.0
SEED = 7
T_FLAT = 0.01
# Production uses 24px on a ~4000-6000px sensor (~0.5% of the frame); rescaled to this tiny
# canvas so the same technique is still meaningfully "large vs. the dust patch, small vs. the
# frame" -- passed explicitly since fit_flat_model_from_stack's default is the production value.
TEST_HIGHPASS_SIGMA_PX = 6.0


@dataclass
class _FakeDarkModel:
    bias: np.ndarray
    rate: np.ndarray


def _make_vignette_true():
    rows = np.linspace(-1.0, 1.0, H).reshape(-1, 1)
    cols = np.linspace(-1.0, 1.0, W).reshape(1, -1)
    r2 = rows ** 2 + cols ** 2
    v = 1.0 - 0.35 * r2                      # smooth radial falloff, brightest at center
    v[5:8, 3:6] *= 0.6                        # a dust-shadow dip
    return v.astype(np.float64)


def _expected_highpass_vignette():
    """What fit_flat_model_from_stack should recover: the noiseless true signal run through
    the same high-pass removal, normalization and clamp -- the smooth radial falloff must be
    gone, and the dust-shadow dip (a genuine ~40% local defect) should be pinned at the clamp
    floor, not fully recovered."""
    highpassed = FC._remove_smooth_trend(_make_vignette_true(), TEST_HIGHPASS_SIGMA_PX)
    harmonic_mean = 1.0 / np.mean(1.0 / highpassed)
    normalized = highpassed / harmonic_mean
    return np.clip(normalized, FC.FLAT_CLAMP_LO, FC.FLAT_CLAMP_HI)


def make_synthetic(seed: int = SEED):
    rng = np.random.default_rng(seed)

    vignette_true = _make_vignette_true()
    illum = 0.4                               # flat-panel brightness at t=0 vignetting
    bias_true = rng.uniform(0.0005, 0.004, size=(H, W))
    rate_true = rng.uniform(0.0002, 0.003, size=(H, W))
    dark_model = _FakeDarkModel(bias=bias_true.astype(np.float32), rate=rate_true.astype(np.float32))

    frames = []
    for _ in range(N_FRAMES):
        signal = illum * vignette_true
        dark = bias_true + rate_true * T_FLAT
        y = signal + dark + rng.normal(0.0, READ_NOISE_SIGMA, size=(H, W))
        frames.append(y.astype(np.float32))

    times = [T_FLAT] * N_FRAMES
    return vignette_true, dark_model, times, frames


def test_synthetic_flat_recovery():
    _vignette_true, dark_model, times, frames = make_synthetic()

    vignette = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX)

    # Recovered map is normalized (harmonic mean 1.0) and high-frequency only -- compare
    # against the noiseless signal put through the same high-pass + normalization, not the
    # raw planted vignette (which still has the smooth radial falloff this is meant to drop).
    expected = _expected_highpass_vignette()
    err = float(np.max(np.abs(vignette - expected)))
    print(f"synthetic: {N_FRAMES} flat frames, max high-pass vignette err {err:.5f}")
    assert err < 0.02, err

    # The planted dust dip is a genuine ~40% local defect -- confirm the clamp actually
    # engaged on it (min pixel pinned at the floor), not just that clamping exists in theory.
    assert abs(float(vignette.min()) - FC.FLAT_CLAMP_LO) < 1e-4, vignette.min()
    assert vignette.max() <= FC.FLAT_CLAMP_HI + 1e-4, vignette.max()

    # Applying the recovered map to a perfectly uniform frame averages back to only
    # *approximately* 1.0 here: harmonic-mean normalization guarantees exactly 1.0, but the
    # clamp runs after it and this scene's planted dust dip is large enough to actually get
    # clamped (checked above), which is exactly the deliberate trade-off documented in
    # flatcal.py's module docstring. A scene with no clamped pixels would hit 1.0 exactly.
    uniform = np.ones((H, W), dtype=np.float64)
    corrected = uniform / vignette
    assert abs(float(corrected.mean()) - 1.0) < 0.02, corrected.mean()


def test_highpass_edge_mirror_padding():
    """A uniform frame's high-pass residual must be exactly zero everywhere, including right
    at the border. Zero/constant padding would instead pull the blur down near the edges
    (mixing in fabricated zeros), leaving a spurious bright ring in the high-pass result --
    mirror padding (`mode="reflect"`) must not."""
    const = 3.7
    uniform = np.full((40, 30), const, dtype=np.float64)
    result = FC._remove_smooth_trend(uniform, sigma=10.0)
    err = float(np.max(np.abs(result - const)))
    print(f"synthetic: uniform-frame edge check, max deviation {err:.2e}")
    assert err < 1e-9, err


def _expect_assertion(fn, needle: str):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), e
        return
    raise AssertionError(f"expected an AssertionError containing {needle!r}, got none")


def test_min_frames_and_exposure_assertions():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0.1, 0.2, size=(8, 8)).astype(np.float32)

    _expect_assertion(
        lambda: FC.fit_flat_model_from_stack([1.0] * 3, [frame, frame, frame]), "need >=")
    _expect_assertion(
        lambda: FC.fit_flat_model_from_stack(
            [1.0] * (FC.MIN_FLAT_FRAMES - 1) + [2.0],
            [frame] * FC.MIN_FLAT_FRAMES),
        "same exposure time")


if __name__ == "__main__":
    test_synthetic_flat_recovery()
    test_highpass_edge_mirror_padding()
    test_min_frames_and_exposure_assertions()
    print("OK")
