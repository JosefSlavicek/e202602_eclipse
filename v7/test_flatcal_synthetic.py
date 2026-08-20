#!/usr/bin/env python3
"""Synthetic check of the streamed, per-frame, quadratic flat-field model. No GPU, no files,
~1 s.

Plants a known optical setup: a smooth radial vignetting shape (removed by the masked blur,
must not show up in the result) plus a 3x3 patch with a genuine `corr = K_LIN*smooth +
K_QUAD*smooth**2` defect, imaged at several exposure times so brightness genuinely varies, one
noisy frame at a time (not averaged into a group first, unlike the old model). Checks:

  - the fitted per-pixel (a, b, c) quadratic matches an independently-computed ground truth
    (masked-blur the noiseless planted signal, then fit each pixel with `np.polyfit` -- a
    different numerical method than the production closed-form solve, so this isn't just
    testing the code against itself), and the planted patch shows up as real nonzero b/c while
    unaffected corners stay near zero;
  - a pixel forced saturated in one extra frame gets zero weight there and its own fit is
    completely unaffected, while an unrelated pixel in that *same* frame still picks up that
    frame's data -- the whole frame is never dropped, unlike the old grouped model;
  - too narrow a brightness bracket (single exposure time) falls back to a = b = c = 0 with
    `reliable` all False;
  - `_masked_blur` excludes an overburn neighbor from the blur it feeds to everyone else, and
    its mirror-padded edges don't bias a uniform frame;
  - `FlatModel.evaluate` is additive (a=b=c=0 is the exact identity) and clamps to
    `+/- FLAT_CORR_CLAMP`.

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
# Production uses 24px on a ~4000-6000px sensor; rescaled to this tiny canvas so the same
# technique is still meaningfully "large vs. the dust patch, small vs. the frame" -- passed
# explicitly since fit_flat_model_from_stack's default is the production value.
TEST_HIGHPASS_SIGMA_PX = 6.0
# Production's MIN_RELIABLE_RANGE (0.5) assumes a full-scale [0, 1] sensor bracket; this tiny
# canvas's own brightness range is much smaller, so it's rescaled down for the same reason.
TEST_MIN_RELIABLE_RANGE = 0.1

EXPOSURES = [0.004, 0.008, 0.016, 0.032]     # brightness genuinely varies across these
ILLUM_RATE = 15.0                            # panel brightness = ILLUM_RATE * exposure_time
N_FRAMES_PER_EXPOSURE = 18
PATCH_ROWS, PATCH_COLS = slice(5, 8), slice(3, 6)   # the planted 3x3 defect
K_LIN = 0.15                                 # planted linear defect coefficient
K_QUAD = 0.20                                # planted quadratic defect coefficient


@dataclass
class _FakeDarkModel:
    bias: np.ndarray
    rate: np.ndarray


def _make_radial_true():
    rows = np.linspace(-1.0, 1.0, H).reshape(-1, 1)
    cols = np.linspace(-1.0, 1.0, W).reshape(1, -1)
    r2 = rows ** 2 + cols ** 2
    return (1.0 - 0.35 * r2).astype(np.float64)   # smooth radial falloff, brightest at center


def _true_signal(t: float, with_patch: bool = True) -> np.ndarray:
    """Noiseless dark-subtracted true signal at exposure `t`: smooth radial vignetting
    everywhere, plus (if `with_patch`) a planted 3x3 patch with `corr = K_LIN*smooth +
    K_QUAD*smooth**2`. `with_patch=False` isolates everything the smooth field alone
    contributes to the fit -- including the blur's own small bias on a *curved* smooth field,
    which is not the planted defect and must not be mistaken for it (see
    test_recovery_linear_and_quadratic_defect)."""
    smooth = ILLUM_RATE * t * _make_radial_true()
    if not with_patch:
        return smooth
    y = smooth.copy()
    patch_smooth = smooth[PATCH_ROWS, PATCH_COLS]
    y[PATCH_ROWS, PATCH_COLS] = (
        patch_smooth - (K_LIN * patch_smooth + K_QUAD * patch_smooth ** 2))
    return y


def _make_dark_model(rng):
    bias_true = rng.uniform(0.0005, 0.004, size=(H, W))
    rate_true = rng.uniform(0.0002, 0.003, size=(H, W))
    return _FakeDarkModel(bias=bias_true.astype(np.float32), rate=rate_true.astype(np.float32))


def _synthesize_frames(times, dark_model, rng, with_patch: bool = True):
    """One noisy raw frame (dark still included, as fit_flat_model_from_stack expects) per
    entry of `times`."""
    frames = []
    for t in times:
        dark = dark_model.bias.astype(np.float64) + dark_model.rate.astype(np.float64) * t
        y = (_true_signal(t, with_patch) + dark
             + rng.normal(0.0, READ_NOISE_SIGMA, size=(H, W)))
        frames.append(y.astype(np.float32))
    return frames


def _make_bracket(rng, n_per_exposure=N_FRAMES_PER_EXPOSURE, with_patch: bool = True):
    dark_model = _make_dark_model(rng)
    times = [t for t in EXPOSURES for _ in range(n_per_exposure)]
    frames = _synthesize_frames(times, dark_model, rng, with_patch)
    return times, frames, dark_model


def _expected_abc(with_patch: bool = True):
    """Independent ground truth: run the noiseless true signal through the same masked blur,
    then fit each pixel's quadratic with np.polyfit (not the production closed-form solve)."""
    xs, corrs = [], []
    for t in EXPOSURES:
        true_signal = _true_signal(t, with_patch)
        valid = np.ones_like(true_signal, dtype=bool)
        blurred = FC._masked_blur(true_signal, valid, TEST_HIGHPASS_SIGMA_PX)
        xs.append(true_signal)
        corrs.append(blurred - true_signal)

    a = np.zeros((H, W)); b = np.zeros((H, W)); c = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            x = np.array([xx[i, j] for xx in xs])
            y = np.array([cc[i, j] for cc in corrs])
            c2, c1, c0 = np.polyfit(x, y, 2)
            a[i, j], b[i, j], c[i, j] = c0, c1, c2
    return a, b, c


def test_recovery_linear_and_quadratic_defect():
    rng = np.random.default_rng(SEED)
    times, frames, dark_model = _make_bracket(rng)

    a, b, c, used_exposures, n_frames, _report, reliable = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX, min_reliable_range=TEST_MIN_RELIABLE_RANGE)

    assert list(used_exposures) == sorted(EXPOSURES)
    assert n_frames == N_FRAMES_PER_EXPOSURE * len(EXPOSURES)
    assert bool(reliable.all()), "every pixel should have plenty of dynamic range on this bracket"

    a_exp, b_exp, c_exp = _expected_abc()
    err_a = float(np.max(np.abs(a - a_exp)))
    err_b = float(np.max(np.abs(b - b_exp)))
    err_c = float(np.max(np.abs(c - c_exp)))
    print(f"synthetic: max |a-exp| {err_a:.4f}, |b-exp| {err_b:.4f}, |c-exp| {err_c:.4f}")
    assert err_a < 0.02, err_a
    assert err_b < 0.1, err_b
    assert err_c < 0.3, err_c

    # Isolate the planted defect from the smooth field's own contribution to the fit (a
    # Gaussian blur of a *curved* smooth field isn't exactly the identity -- negligible at
    # production's sigma/sensor-size ratio, but not at this tiny canvas's) by rerunning with
    # the patch switched off, same dark model and noise draws. The patch's own value change
    # also leaks a little into the blur of *nearby* pixels (real, expected -- the blur kernel
    # has some reach), so only a region genuinely far from the patch is checked for a clean
    # match; the patch region itself should differ a lot.
    rng_clean = np.random.default_rng(SEED)
    times_c, frames_c, dark_c = _make_bracket(rng_clean, with_patch=False)
    a_c, b_c, c_c, *_ = FC.fit_flat_model_from_stack(
        times_c, frames_c, dark_model=dark_c,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX, min_reliable_range=TEST_MIN_RELIABLE_RANGE)

    far = np.s_[25:35, 15:25]   # well outside the blur kernel's reach from the patch
    assert np.max(np.abs((a - a_c)[far])) < 1e-5
    assert np.max(np.abs((b - b_c)[far])) < 1e-5
    assert np.max(np.abs((c - c_c)[far])) < 1e-5

    d_b = np.abs((b - b_c)[PATCH_ROWS, PATCH_COLS]).mean()
    d_c = np.abs((c - c_c)[PATCH_ROWS, PATCH_COLS]).mean()
    print(f"synthetic: planted-defect-only |b| mean {d_b:.4f}, |c| mean {d_c:.4f}")
    assert d_b > 0.05, d_b
    assert d_c > 0.05, d_c


def test_overburn_pixel_excluded_others_unaffected():
    """A pixel forced saturated in one extra frame gets zero weight there, so that extra
    frame's data must not move its own fit at all relative to a baseline that never saw the
    extra frame -- while a different, unrelated (non-saturated) pixel in that same frame does
    pick up the extra data point. Whole frames are never dropped, unlike the old grouped
    model."""
    rng = np.random.default_rng(123)
    times0, frames0, dark_model = _make_bracket(rng)

    a0, b0, c0, _e0, _n0, _r0, _rel0 = FC.fit_flat_model_from_stack(
        times0, frames0, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX, min_reliable_range=TEST_MIN_RELIABLE_RANGE)

    t_extra = EXPOSURES[0]
    dark = dark_model.bias.astype(np.float64) + dark_model.rate.astype(np.float64) * t_extra
    extra_bad = (_true_signal(t_extra) + dark).astype(np.float32)
    bad_pixel = (10, 10)
    far_pixel = (30, 20)
    extra_bad[bad_pixel] = 1.5   # far above OVERBURN_HI

    times1 = list(times0) + [t_extra]
    frames1 = list(frames0) + [extra_bad]
    a1, b1, c1, _e1, n1, _r1, _rel1 = FC.fit_flat_model_from_stack(
        times1, frames1, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX, min_reliable_range=TEST_MIN_RELIABLE_RANGE)

    assert n1 == len(times0) + 1
    # bad_pixel's own fit is untouched -- the extra frame's data point there is zero-weighted,
    # contributing exactly nothing to its accumulated sums.
    assert abs(a1[bad_pixel] - a0[bad_pixel]) < 1e-9, (a1[bad_pixel], a0[bad_pixel])
    assert abs(b1[bad_pixel] - b0[bad_pixel]) < 1e-9, (b1[bad_pixel], b0[bad_pixel])
    assert abs(c1[bad_pixel] - c0[bad_pixel]) < 1e-9, (c1[bad_pixel], c0[bad_pixel])
    # far_pixel, not saturated in that same frame, does pick up the extra data point -- the
    # whole frame was not dropped just because one other pixel in it saturated.
    assert (a1[far_pixel] != a0[far_pixel] or b1[far_pixel] != b0[far_pixel]
            or c1[far_pixel] != c0[far_pixel])
    print("synthetic: overburn pixel excluded per-pixel, rest of that frame still used")


def test_reliability_fallback_on_narrow_bracket():
    """A single exposure time gives zero dynamic range -- every pixel must fall back to
    a = b = c = 0 with reliable = False."""
    rng = np.random.default_rng(99)
    dark_model = _make_dark_model(rng)
    t = EXPOSURES[0]
    times = [t] * N_FRAMES_PER_EXPOSURE
    frames = _synthesize_frames(times, dark_model, rng)

    a, b, c, used_exposures, n_frames, _report, reliable = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX, min_reliable_range=TEST_MIN_RELIABLE_RANGE)

    assert not reliable.any(), int(reliable.sum())
    assert np.all(a == 0.0) and np.all(b == 0.0) and np.all(c == 0.0)
    assert list(used_exposures) == [t]
    assert n_frames == N_FRAMES_PER_EXPOSURE
    print("synthetic: narrow-bracket fallback OK, a=b=c=0 everywhere")


def test_masked_blur_excludes_overburn_neighbors():
    """A single very bright (overburn-flagged) outlier pixel must not pull the blur at its
    neighbors toward it once masked -- and measurably would, if not masked."""
    rng = np.random.default_rng(1)
    y = 0.3 + 0.01 * rng.standard_normal((40, 40))
    y_with_outlier = y.copy()
    y_with_outlier[20, 20] = 5.0   # way above any real signal
    valid = np.ones_like(y, dtype=bool)
    valid[20, 20] = False

    blurred_masked = FC._masked_blur(y_with_outlier, valid, sigma=6.0)
    blurred_unmasked = FC._masked_blur(y_with_outlier, np.ones_like(valid), sigma=6.0)
    blurred_clean = FC._masked_blur(y, np.ones_like(valid), sigma=6.0)

    near = (21, 22)
    # Masked blur near the outlier should track the outlier-free blur closely...
    assert abs(blurred_masked[near] - blurred_clean[near]) < 0.01
    # ...while the unmasked blur is measurably pulled up by the outlier at the same spot.
    assert blurred_unmasked[near] - blurred_clean[near] > 0.01
    print("synthetic: masked blur excludes overburn neighbor, unmasked blur is biased by it")


def test_highpass_edge_mirror_padding():
    """A uniform frame's masked blur must equal that constant everywhere, including right at
    the border -- zero/constant padding would instead pull the blur down near the edges."""
    const = 3.7
    uniform = np.full((40, 30), const, dtype=np.float64)
    result = FC._masked_blur(uniform, np.ones_like(uniform, dtype=bool), sigma=10.0)
    err = float(np.max(np.abs(result - const)))
    print(f"synthetic: uniform-frame edge check, max deviation {err:.2e}")
    assert err < 1e-9, err


def test_evaluate_additive_identity_and_clamp():
    a = np.array([[0.0, 0.2]])
    b = np.array([[0.0, 0.0]])
    c = np.array([[0.0, 0.0]])
    reliable = np.array([[False, True]])
    model = FC.FlatModel(a=a, b=b, c=c, reliable=reliable,
                          exposures=np.array([0.01]), n_frames=10)
    value = np.array([[0.5, 0.5]])
    corr = model.evaluate(value)
    assert corr[0, 0] == 0.0                          # a=b=c=0 -> exact identity when added
    assert corr[0, 1] == FC.FLAT_CORR_CLAMP            # 0.2 clamped down to the cap
    print("synthetic: evaluate() additive identity and clamp OK")


def _expect_assertion(fn, needle: str):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), e
        return
    raise AssertionError(f"expected an AssertionError containing {needle!r}, got none")


def test_too_few_frames_fails_loudly():
    rng = np.random.default_rng(0)
    dark_model = _make_dark_model(rng)
    times = [EXPOSURES[0], EXPOSURES[0]]
    frames = _synthesize_frames(times, dark_model, rng)
    _expect_assertion(
        lambda: FC.fit_flat_model_from_stack(
            times, frames, dark_model=dark_model, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX),
        "need >=")


if __name__ == "__main__":
    test_recovery_linear_and_quadratic_defect()
    test_overburn_pixel_excluded_others_unaffected()
    test_reliability_fallback_on_narrow_bracket()
    test_masked_blur_excludes_overburn_neighbors()
    test_highpass_edge_mirror_padding()
    test_evaluate_additive_identity_and_clamp()
    test_too_few_frames_fails_loudly()
    print("OK")
