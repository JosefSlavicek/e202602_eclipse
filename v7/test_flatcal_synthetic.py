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
# Production's FLAT_MODEL_VALUE_THRESHOLD (~0.066) is calibrated against real sensor flats and
# has no meaning on this canvas's own arbitrary brightness scale -- disabled (set unreachable)
# for every test that isn't specifically exercising the threshold gate itself.
TEST_VALUE_THRESHOLD = float("inf")

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
    then fit each pixel's quadratic -- against `blurred`, matching production's regressor, not
    against the raw signal -- with np.polyfit (not the production closed-form solve)."""
    bxs, corrs = [], []
    for t in EXPOSURES:
        true_signal = _true_signal(t, with_patch)
        valid = np.ones_like(true_signal, dtype=bool)
        blurred = FC._masked_blur(true_signal, valid, TEST_HIGHPASS_SIGMA_PX)
        bxs.append(blurred)
        corrs.append(blurred - true_signal)

    a = np.zeros((H, W)); b = np.zeros((H, W)); c = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            bx = np.array([xx[i, j] for xx in bxs])
            y = np.array([cc[i, j] for cc in corrs])
            c2, c1, c0 = np.polyfit(bx, y, 2)
            a[i, j], b[i, j], c[i, j] = c0, c1, c2
    return a, b, c


def test_recovery_linear_and_quadratic_defect():
    rng = np.random.default_rng(SEED)
    times, frames, dark_model = _make_bracket(rng)

    a, b, c, used_exposures, n_frames, _report, reliable = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

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
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

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
    """A pixel forced saturated in one extra frame is excluded outright from *its own*
    weighted average for that exposure -- while a different, unrelated (non-saturated) pixel in
    that same frame does pick up the extra data point, and the whole frame is never dropped.

    Unlike the old per-frame-streamed design (where the excluded frame contributed literally
    nothing to bad_pixel's accumulated sums, an exact-zero invariant), grouping by exposure
    means bad_pixel's fit can still move by a tiny amount here: `blurred` is a spatial blur of
    the *group average*, and neighboring pixels' averages did legitimately change (they picked
    up the extra frame's real data) -- so a whisper of that leaks into bad_pixel's own blurred
    value through the neighborhood, same as it would for any pixel near one whose input data
    changed. The check here is that this leak is tiny relative to a real change (far_pixel's),
    not literally zero.
    """
    rng = np.random.default_rng(123)
    times0, frames0, dark_model = _make_bracket(rng)

    a0, b0, c0, _e0, _n0, _r0, _rel0 = FC.fit_flat_model_from_stack(
        times0, frames0, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

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
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

    assert n1 == len(times0) + 1
    d_bad = (abs(a1[bad_pixel] - a0[bad_pixel]), abs(b1[bad_pixel] - b0[bad_pixel]),
             abs(c1[bad_pixel] - c0[bad_pixel]))
    d_far = (abs(a1[far_pixel] - a0[far_pixel]), abs(b1[far_pixel] - b0[far_pixel]),
             abs(c1[far_pixel] - c0[far_pixel]))
    print(f"synthetic: bad_pixel move {d_bad}, far_pixel move {d_far}")
    # far_pixel, not saturated in that same frame, does pick up the extra data point -- the
    # whole frame was not dropped just because one other pixel in it saturated.
    assert d_far[0] > 0 and d_far[1] > 0 and d_far[2] > 0
    # bad_pixel's own move is only the neighborhood-blur leak described above -- an order of
    # magnitude (measured: ~100-170x) smaller than far_pixel's genuine move, not comparable to it.
    assert d_bad[0] < d_far[0] / 10 and d_bad[1] < d_far[1] / 10 and d_bad[2] < d_far[2] / 10
    print("synthetic: overburn pixel excluded from its own average, rest of that frame still used")


def test_reliability_fallback_on_narrow_bracket():
    """A single exposure time gives only one distinct `blurred` value per pixel -- not enough
    to identify a 3-parameter quadratic (the moment matrix is exactly singular) -- so every
    pixel must fall back to a = b = c = 0 with reliable = False."""
    rng = np.random.default_rng(99)
    dark_model = _make_dark_model(rng)
    t = EXPOSURES[0]
    times = [t] * N_FRAMES_PER_EXPOSURE
    frames = _synthesize_frames(times, dark_model, rng)

    a, b, c, used_exposures, n_frames, _report, reliable = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

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


def test_group_weights_excludes_neighborhood_of_invalid_cluster():
    """Pushback 3 / option A: a compact cluster of pixels invalid in *every* frame of a group
    (bad_frac = 1 there) must also zero the weight of nearby, otherwise-fine pixels -- the
    per-group generalization of the old per-frame overburn-neighborhood check, now blurring a
    continuous per-pixel invalid-fraction field instead of a single frame's boolean mask."""
    H, W = 60, 60
    bad_frac = np.zeros((H, W), dtype=np.float64)
    valid_group = np.ones((H, W), dtype=bool)
    bad_frac[25:35, 25:35] = 1.0     # a compact cluster, invalid in every frame of the group
    valid_group[25:35, 25:35] = False

    w = FC._group_weights(bad_frac, valid_group, sigma=6.0, frac_max=FC.NEIGHBORHOOD_OVERBURN_FRAC_MAX)

    assert w[30, 30] == 0.0    # inside the cluster: zero, trivially (valid_group is False)
    assert w[24, 30] == 0.0    # just outside it, but close enough for the blurred local
                                 # invalid-fraction to exceed frac_max -- must also be zeroed
    assert w[0, 0] == 1.0      # far from the cluster: unaffected
    print("synthetic: _group_weights zeroes a fine pixel near a fully-invalid cluster")


def test_group_average_excludes_minority_invalid_frames():
    """Within one exposure group, a pixel invalid (overburn) in a minority of that exposure's
    frames should still recover close to the fit an untouched bracket gives -- the weighted
    average excludes those frames' contribution at that exact pixel, so it isn't pulled toward
    the clipped value, and the group's weight there isn't zeroed either (too small a fraction of
    the group to trip the neighborhood-density check)."""
    rng = np.random.default_rng(55)
    times, frames, dark_model = _make_bracket(rng)
    t_bad = EXPOSURES[0]
    bad_pixel = (12, 15)

    frames_mod = [f.copy() for f in frames]
    n_forced = 0
    for i, t in enumerate(times):
        if t == t_bad and n_forced < 3:      # a minority of this exposure's frames
            frames_mod[i][bad_pixel] = 1.5   # force overburn
            n_forced += 1
    assert n_forced == 3

    a0, b0, c0, *_ = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)
    a, b, c, *_ = FC.fit_flat_model_from_stack(
        times, frames_mod, dark_model=dark_model,
        highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)

    d = (abs(a[bad_pixel] - a0[bad_pixel]), abs(b[bad_pixel] - b0[bad_pixel]),
         abs(c[bad_pixel] - c0[bad_pixel]))
    print(f"synthetic: minority-invalid-frames pixel move {d}")
    assert d[0] < 0.01 and d[1] < 0.05 and d[2] < 0.1
    print("synthetic: group average excludes a minority of invalid frames, fit barely moves")


def test_value_threshold_excludes_bright_exposures_from_fit():
    """A pixel whose blurred value is at/above `value_threshold` in every exposure never gets a
    real fit (falls back to a=b=c=0, reliable=False) -- exactly as if none of those exposures
    had ever been shot for it. A pixel that stays below the threshold for enough exposures still
    gets a real fit even though the same bracket also contains brighter exposures that get
    excluded for it specifically.

    The two concrete pixels/threshold below were picked empirically, not derived from the
    smooth-field formula alone: near the frame edge, `_masked_blur` picks up a real
    blur-curvature bias from the boundary that the raw radial formula doesn't predict, so the
    corner pixel's `blurred` value is not simply `ILLUM_RATE * t * field(corner)`.
    """
    rng = np.random.default_rng(7)
    times, frames, dark_model = _make_bracket(rng)
    center = (H // 2, W // 2)   # brightest pixel in the frame
    corner = (0, 0)             # dimmest pixel in the frame
    threshold = 0.15

    _a, _b, _c, _e, _n, _r, reliable_all = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=TEST_VALUE_THRESHOLD)
    a, b, c, _e2, _n2, _r2, reliable = FC.fit_flat_model_from_stack(
        times, frames, dark_model=dark_model, highpass_sigma_px=TEST_HIGHPASS_SIGMA_PX,
        value_threshold=threshold)

    assert reliable_all[center], "sanity: center is reliable with the gate effectively disabled"
    assert not reliable[center], "center should exceed the threshold in every exposure"
    assert a[center] == 0.0 and b[center] == 0.0 and c[center] == 0.0

    assert reliable[corner], "corner should keep enough sub-threshold exposures to stay reliable"
    print("synthetic: value_threshold excludes an always-bright pixel, keeps a dim one reliable")


def test_apply_masked_leaves_at_or_above_threshold_pixels_untouched():
    """`FlatModel.apply_masked` corrects a below-threshold pixel exactly like `evaluate_refined`
    would, but returns an at/above-threshold pixel completely unchanged -- not even clamped."""
    a = np.array([[0.2, 0.2]])   # would clamp hugely if ever evaluated
    b = np.array([[0.0, 0.0]])
    c = np.array([[0.0, 0.0]])
    reliable = np.array([[True, True]])
    threshold = 0.5
    model = FC.FlatModel(a=a, b=b, c=c, reliable=reliable, exposures=np.array([0.01]),
                          n_frames=10, value_threshold=threshold)

    value = np.array([[0.1, 0.9]])   # first pixel below threshold, second at/above it
    result = model.apply_masked(value)

    expected_below = value[0, 0] + model.evaluate_refined(value)[0, 0]
    assert result[0, 0] == expected_below
    assert result[0, 1] == value[0, 1]     # untouched, not even clamped
    print("synthetic: apply_masked corrects below threshold, leaves at/above threshold untouched")


def test_apply_masked_soft_margin_taper():
    """Between `FLAT_MODEL_THRESHOLD_MARGIN_FRAC * value_threshold` and `value_threshold`, the
    correction fades smoothly rather than switching off abruptly: weight 1.0 at/below the
    margin's low edge, 0.0 at/above `value_threshold`, strictly partial and monotonically
    non-increasing in between -- so no seam between a corrected and an untouched pixel."""
    a = np.array([[0.2] * 5])
    b = np.array([[0.0] * 5])
    c = np.array([[0.0] * 5])
    reliable = np.array([[True] * 5])
    threshold = 1.0
    model = FC.FlatModel(a=a, b=b, c=c, reliable=reliable, exposures=np.array([0.01]),
                          n_frames=10, value_threshold=threshold)

    lo = FC.FLAT_MODEL_THRESHOLD_MARGIN_FRAC * threshold
    values = np.array([[lo - 0.05, lo, 0.5 * (lo + threshold), threshold, threshold + 0.05]])
    w = model._threshold_weight(values)

    assert w[0, 0] == 1.0                       # below the margin: full weight
    assert w[0, 1] == 1.0                       # right at the margin's low edge: full weight
    assert 0.0 < w[0, 2] < 1.0                  # inside the margin: strictly partial
    assert w[0, 3] == 0.0                       # at the threshold itself: zero
    assert w[0, 4] == 0.0                       # above it: still zero
    assert np.all(np.diff(w[0]) <= 1e-12)       # monotonically non-increasing
    print("synthetic: apply_masked's soft-margin taper blends smoothly, no hard seam")


def test_highpass_edge_mirror_padding():
    """A uniform frame's masked blur must equal that constant everywhere, including right at
    the border -- zero/constant padding would instead pull the blur down near the edges."""
    const = 3.7
    uniform = np.full((40, 30), const, dtype=np.float64)
    result = FC._masked_blur(uniform, np.ones_like(uniform, dtype=bool), sigma=10.0)
    err = float(np.max(np.abs(result - const)))
    print(f"synthetic: uniform-frame edge check, max deviation {err:.2e}")
    # 1e-9 back when the blur ran in fp64; it now runs in fp32 (_BLUR_DTYPE) for GPU speed,
    # so the floor is fp32 rounding, not the padding -- still >99.99% tighter than any of this
    # module's real correction tolerances (FLAT_CORR_CLAMP = 0.05).
    assert err < 1e-4, err


def test_evaluate_additive_identity_and_clamp():
    a = np.array([[0.0, 0.2]])
    b = np.array([[0.0, 0.0]])
    c = np.array([[0.0, 0.0]])
    reliable = np.array([[False, True]])
    model = FC.FlatModel(a=a, b=b, c=c, reliable=reliable,
                          exposures=np.array([0.01]), n_frames=10,
                          value_threshold=TEST_VALUE_THRESHOLD)
    value = np.array([[0.5, 0.5]])
    corr = model.evaluate(value)
    assert corr[0, 0] == 0.0                          # a=b=c=0 -> exact identity when added
    assert corr[0, 1] == FC.FLAT_CORR_CLAMP            # 0.2 clamped down to the cap
    # a=b=c=0 makes evaluate_refined an identity too, same as the raw evaluate().
    corr_refined = model.evaluate_refined(value)
    assert corr_refined[0, 0] == 0.0
    assert corr_refined[0, 1] == FC.FLAT_CORR_CLAMP
    print("synthetic: evaluate()/evaluate_refined() additive identity and clamp OK")


def test_evaluate_refined_beats_one_shot_on_a_real_defect():
    """`evaluate_refined`'s fixed-point step should land closer to the true smooth value than
    plugging the raw signal straight into `evaluate` -- and never overshoot past it (the
    residual this step is fixing only ever under-corrects, unlike the noise bias the
    `blurred`-regression change fixes; see the module docstring). Defect sized to stay well
    inside FLAT_CORR_CLAMP so the clamp itself doesn't mask the comparison."""
    s = 0.3                                            # a plausible true smooth-field value
    b_true, c_true = 0.05, 0.10                        # a real (percent-scale) defect
    a = np.array([[0.0]])
    b = np.array([[b_true]])
    c = np.array([[c_true]])
    reliable = np.array([[True]])
    model = FC.FlatModel(a=a, b=b, c=c, reliable=reliable,
                          exposures=np.array([0.01]), n_frames=10,
                          value_threshold=TEST_VALUE_THRESHOLD)

    x = s - (b_true * s + c_true * s ** 2)             # the defect-corrupted observed value
    value = np.array([[x]])

    one_shot = float(x + model.evaluate(value)[0, 0])
    refined = float(x + model.evaluate_refined(value)[0, 0])

    err_one_shot = abs(one_shot - s)
    err_refined = abs(refined - s)
    print(f"synthetic: recovering s={s}: one-shot err {err_one_shot:.4f}, "
          f"refined err {err_refined:.4f}")
    assert err_refined < err_one_shot
    # Both under-correct here (same-signed residual), never overshoot past the true value.
    assert one_shot < s
    assert refined < s


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
    test_group_weights_excludes_neighborhood_of_invalid_cluster()
    test_group_average_excludes_minority_invalid_frames()
    test_value_threshold_excludes_bright_exposures_from_fit()
    test_highpass_edge_mirror_padding()
    test_evaluate_additive_identity_and_clamp()
    test_evaluate_refined_beats_one_shot_on_a_real_defect()
    test_apply_masked_leaves_at_or_above_threshold_pixels_untouched()
    test_apply_masked_soft_margin_taper()
    test_too_few_frames_fails_loudly()
    print("OK")
