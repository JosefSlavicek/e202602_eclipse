#!/usr/bin/env python3
"""Synthetic check of the brightness-dependent flat-field model. No GPU, no files, ~1 s.

Plants a known optical setup: a smooth radial vignetting shape (removed by the high-pass
step, must not show up in the result) times a fixed 3x3 dust-shadow multiplier, imaged at
several exposure times so the panel brightness genuinely varies. Checks:

  - the fitted per-pixel (a, b) line matches an independently-computed ground truth (built
    by running the same high-pass removal on the noiseless planted signal, then fitting each
    pixel's line with `np.polyfit` -- a different numerical method than the production
    closed-form solve, so this isn't just testing the code against itself);
  - a frame with too many saturated pixels is dropped from its group, and does not change
    that group's fitted values;
  - a whole exposure-time group with too few surviving frames is dropped, reported as
    unused, and does not influence the fit;
  - with only one qualifying group, the model falls back to the old brightness-independent
    behaviour (b = 0 everywhere);
  - the high-pass step's mirror-padded edges don't bias a uniform frame (unchanged from
    before -- `_remove_smooth_trend` itself did not change in this rework).

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
READ_NOISE_SIGMA = 2.0 / 16383.0
SEED = 7
# Production uses 24px on a ~4000-6000px sensor (~0.5% of the frame); rescaled to this tiny
# canvas so the same technique is still meaningfully "large vs. the dust patch, small vs. the
# frame" -- passed explicitly since fit_flat_model_from_stack's default is the production value.
TEST_HIGHPASS_SIGMA_PX = 6.0

EXPOSURES = [0.004, 0.008, 0.016, 0.032]     # 4 "good" exposure groups
ILLUM_RATE = 15.0                            # panel brightness = ILLUM_RATE * exposure_time
N_FRAMES_GOOD = 18                           # > MIN_FRAMES_PER_GROUP (5)
T_THIN = 0.002                               # a 5th exposure with too few frames to qualify
N_FRAMES_THIN = 3                            # < MIN_FRAMES_PER_GROUP (5)
PATCH_ROWS, PATCH_COLS = slice(5, 8), slice(3, 6)   # the planted 3x3 dust-shadow defect
PATCH_MULT = 0.6


@dataclass
class _FakeDarkModel:
    bias: np.ndarray
    rate: np.ndarray


def _make_radial_true():
    rows = np.linspace(-1.0, 1.0, H).reshape(-1, 1)
    cols = np.linspace(-1.0, 1.0, W).reshape(1, -1)
    r2 = rows ** 2 + cols ** 2
    return (1.0 - 0.35 * r2).astype(np.float64)   # smooth radial falloff, brightest at center


def _true_signal(t: float) -> np.ndarray:
    """Noiseless dark-subtracted true signal at exposure `t`: radial vignetting x a fixed
    3x3 multiplicative dust shadow, scaled by panel brightness at that exposure."""
    mult = np.ones((H, W), dtype=np.float64)
    mult[PATCH_ROWS, PATCH_COLS] = PATCH_MULT
    return ILLUM_RATE * t * _make_radial_true() * mult


def _make_dark_model(rng):
    bias_true = rng.uniform(0.0005, 0.004, size=(H, W))
    rate_true = rng.uniform(0.0002, 0.003, size=(H, W))
    return _FakeDarkModel(bias=bias_true.astype(np.float32), rate=rate_true.astype(np.float32))


def _synthesize_group(t: float, n_frames: int, dark_model, rng, *, overburn_frame: bool = False):
    """`n_frames` noisy frames at exposure `t` (raw, i.e. dark still included, as
    fit_flat_model_from_stack expects), plus one extra badly-saturated frame appended if
    `overburn_frame` -- appended, not substituted, so the RNG draws for the first `n_frames`
    are identical either way and the only difference is that one extra frame."""
    dark = dark_model.bias.astype(np.float64) + dark_model.rate.astype(np.float64) * t
    signal = _true_signal(t)
    frames = []
    for i in range(n_frames):
        y = signal + dark + rng.normal(0.0, READ_NOISE_SIGMA, size=(H, W))
        frames.append(y.astype(np.float32))
    if overburn_frame:
        bad = np.full((H, W), 0.999, dtype=np.float32)
        bad[:5, :5] = 0.5   # leave a corner unsaturated -- irrelevant, just realism
        frames.append(bad)
    return frames


def _expected_ab():
    """Independent ground truth: run the noiseless true signal through the same high-pass
    removal, then fit each pixel's line with np.polyfit (not the production closed-form
    solve), then apply the same harmonic-mean normalization."""
    brights, coeffs = [], []
    for t in EXPOSURES:
        true_signal = _true_signal(t)
        brights.append(true_signal)
        coeffs.append(FC._remove_smooth_trend(true_signal, TEST_HIGHPASS_SIGMA_PX))

    a = np.zeros((H, W), dtype=np.float64)
    b = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            x = np.array([br[i, j] for br in brights])
            y = np.array([co[i, j] for co in coeffs])
            slope, intercept = np.polyfit(x, y, 1)
            a[i, j], b[i, j] = intercept, slope

    # Same anchor as production: normalize at each pixel's own mean brightness across the
    # groups, not at brightness 0 -- the intercept alone can be arbitrarily close to zero
    # (it is here: these lines pass almost exactly through the origin), which would blow up
    # a harmonic mean taken directly on `a`.
    ref_brightness = np.mean(np.stack(brights, axis=0), axis=0)
    predicted_at_ref = a + b * ref_brightness
    harmonic_mean = 1.0 / np.mean(1.0 / predicted_at_ref)
    return a / harmonic_mean, b / harmonic_mean


def _make_full_dataset(rng, overburn_frame: bool = False):
    dark_model = _make_dark_model(rng)
    times, frames = [], []
    for t in EXPOSURES:
        for f in _synthesize_group(t, N_FRAMES_GOOD, dark_model, rng,
                                    overburn_frame=(overburn_frame and t == EXPOSURES[0])):
            times.append(t)
            frames.append(f)
    for f in _synthesize_group(T_THIN, N_FRAMES_THIN, dark_model, rng):
        times.append(T_THIN)
        frames.append(f)
    return times, frames, dark_model


def test_multi_group_recovery():
    rng = np.random.default_rng(SEED)
    times, frames, dark_model = _make_full_dataset(rng)

    a, b, used_exposures, n_frames, group_report = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX)

    # The thin group must be reported but excluded; the four good ones must be used.
    by_t = {g["t"]: g for g in group_report}
    assert by_t[round(T_THIN, 9)]["used"] is False, by_t[round(T_THIN, 9)]
    assert by_t[round(T_THIN, 9)]["n_valid"] == N_FRAMES_THIN
    for t in EXPOSURES:
        assert by_t[round(t, 9)]["used"] is True, by_t[round(t, 9)]
        assert by_t[round(t, 9)]["n_valid"] == N_FRAMES_GOOD
    assert list(used_exposures) == sorted(EXPOSURES)
    assert n_frames == N_FRAMES_GOOD * len(EXPOSURES)

    a_expected, b_expected = _expected_ab()
    err_a = float(np.max(np.abs(a - a_expected)))
    err_b = float(np.max(np.abs(b - b_expected)))
    print(f"synthetic: max |a - expected| {err_a:.5f}, max |b - expected| {err_b:.5f}")
    assert err_a < 0.02, err_a
    assert err_b < 0.02, err_b

    # The planted dust shadow is a genuine defect at typical calibration brightness -- the
    # patch should show up as a real (nonzero) slope, not get fit away to ~0.
    assert np.abs(b[PATCH_ROWS, PATCH_COLS]).mean() > 0.05, b[PATCH_ROWS, PATCH_COLS]

    # Evaluating far outside the flats' own brightness range must still land in the clamp.
    lo_val = np.zeros((H, W), dtype=np.float64)
    hi_val = np.full((H, W), 10.0, dtype=np.float64)
    coeff_lo = FC.FlatModel(a=a, b=b, exposures=used_exposures, n_frames=n_frames).evaluate(lo_val)
    coeff_hi = FC.FlatModel(a=a, b=b, exposures=used_exposures, n_frames=n_frames).evaluate(hi_val)
    assert coeff_lo.min() >= FC.FLAT_CLAMP_LO - 1e-6 and coeff_lo.max() <= FC.FLAT_CLAMP_HI + 1e-6
    assert coeff_hi.min() >= FC.FLAT_CLAMP_LO - 1e-6 and coeff_hi.max() <= FC.FLAT_CLAMP_HI + 1e-6


def test_overburn_frame_excluded():
    """A badly saturated frame injected into a group must be dropped, leaving that group's
    fit identical to the same group without it (same seed, so the only difference is the
    injected frame)."""
    rng_a = np.random.default_rng(123)
    times_a, frames_a, dark_a = _make_full_dataset(rng_a, overburn_frame=False)
    rng_b = np.random.default_rng(123)
    times_b, frames_b, dark_b = _make_full_dataset(rng_b, overburn_frame=True)

    a1, b1, _e1, n1, report1 = FC.fit_flat_model_from_stack(
        times_a, frames_a, dark_model=dark_a, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX)
    a2, b2, _e2, n2, report2 = FC.fit_flat_model_from_stack(
        times_b, frames_b, dark_model=dark_b, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX)

    g1 = {g["t"]: g for g in report1}[round(EXPOSURES[0], 9)]
    g2 = {g["t"]: g for g in report2}[round(EXPOSURES[0], 9)]
    assert g2["n_total"] == g1["n_total"] + 1, (g1, g2)
    assert g2["n_valid"] == g1["n_valid"], (g1, g2)
    assert n1 == n2
    assert np.max(np.abs(a1 - a2)) < 1e-9, np.max(np.abs(a1 - a2))
    assert np.max(np.abs(b1 - b2)) < 1e-9, np.max(np.abs(b1 - b2))
    print("synthetic: overburn frame correctly excluded, fit unchanged")


def test_single_group_fallback():
    """With only one qualifying exposure group, the model must reduce to the old
    brightness-independent behaviour: b = 0 everywhere."""
    rng = np.random.default_rng(99)
    dark_model = _make_dark_model(rng)
    t = EXPOSURES[0]
    frames = _synthesize_group(t, N_FRAMES_GOOD, dark_model, rng)

    a, b, used_exposures, n_frames, group_report = FC.fit_flat_model_from_stack(
        [t] * N_FRAMES_GOOD, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX)

    assert np.all(b == 0.0), float(np.max(np.abs(b)))
    assert list(used_exposures) == [t]
    assert n_frames == N_FRAMES_GOOD
    print(f"synthetic: single-group fallback OK, a mean {a.mean():.4f}")


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


def test_no_qualifying_groups_fails_loudly():
    rng = np.random.default_rng(0)
    dark_model = _make_dark_model(rng)
    frames = _synthesize_group(EXPOSURES[0], N_FRAMES_THIN, dark_model, rng)
    _expect_assertion(
        lambda: FC.fit_flat_model_from_stack(
            [EXPOSURES[0]] * N_FRAMES_THIN, frames, dark_model=dark_model,
            highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX),
        "no exposure-time group")


if __name__ == "__main__":
    test_multi_group_recovery()
    test_overburn_frame_excluded()
    test_single_group_fallback()
    test_highpass_edge_mirror_padding()
    test_no_qualifying_groups_fails_loudly()
    print("OK")
