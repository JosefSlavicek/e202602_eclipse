"""Per-pixel flat field, fit from per-exposure-time averages of the flat bracket: each distinct
exposure time contributes one `(blurred, corr, w)` data point per pixel to a per-pixel weighted
least-squares fit of `corr = a + b*blurred + c*blurred**2`, where `blurred` is that exposure's
local (Gaussian-blurred) smooth trend and `corr = blurred - x` is how far the exposure's own
(weighted-average, dark-subtracted) value `x` deviates from it -- dust shadows, pixel-to-pixel
sensitivity, not the smooth optical vignetting, which is what the blur estimates and what
stays out of `corr`.

**Regressor is `blurred`, not `x` -- deliberately.** An earlier version of this fit used `x`
itself (this frame's own noisy pixel value) as the regressor, since that is what
`FlatModel.evaluate` ultimately has to run on for a light frame. That choice silently injects
this frame's own per-pixel noise into both sides of the regression at once (`x` directly, and
`corr = blurred - x` through the `-x` term), which biases the fitted slope toward `-1` by an
amount that *grows*, not shrinks, with more frames, and can flip its sign entirely when that
noise is large relative to the pixel's real frame-to-frame brightness swing -- see
`FlatModel.evaluate_refined` below for how the corrected pixel value is recovered from `x`
without needing `x` in the fit itself. `blurred` is averaged over thousands of *other* pixels
(`FLAT_HIGHPASS_SIGMA_PX`-wide), so it carries essentially none of this one pixel's own noise;
regressing against it instead removes that shared-noise term, and as a side effect also
recovers the true physical defect strength (`b`/`c` stop being distorted by the fact that `x`
is itself already the defect-corrupted value being explained).

**Grouped by exposure time, deliberately.** An earlier version of this module streamed every
individual flat frame straight into the regression, one data point per *frame*, specifically to
avoid an older grouped design where one badly-saturated region in one frame would taint that
whole exposure's average for every pixel. This version groups again -- for the compute win (one
`sigma=64` blur per *exposure time* instead of per frame, easily an order of magnitude fewer
blurs on a real bracket) and the lower-noise `x` it gives the fit -- but fixes the original
reason grouping was dropped by making the per-exposure average itself a weighted mean that
excludes invalid pixels per frame (see "Overburn" below), so a saturated region in one frame
still can't taint that pixel's average, or any other pixel's, in that exposure.

**Overburn is pixel-level, checked before averaging.** Saturation is checked on the raw decode,
before dark subtraction (same `OVERBURN_HI` convention as `rawprep.py`), before anything else
runs on a frame. Within one exposure's frames, a pixel's weighted average excludes every frame
where that exact pixel was saturated -- the "outright drop it, don't down-weight it" rule this
module has always applied to overburn, now applied per pixel during averaging rather than per
whole frame. This also tracks, per pixel, `bad_frac`: the fraction of that exposure's frames
which were invalid there. `bad_frac` feeds two things, the per-exposure generalization of what
the old per-frame design did with a single frame's overburn mask: (1) a per-pixel weight `w` for
this exposure (`_group_weights`) -- zero wherever a pixel had zero valid frames at all, and also
zero wherever the Gaussian-weighted (same sigma as the blur) local density of `bad_frac` exceeds
`NEIGHBORHOOD_OVERBURN_FRAC_MAX`, since a blur computed near a cluster of mostly-invalid pixels
is itself biased even at a pixel that was fine in every one of its own frames; (2) the blur
itself (`_masked_blur`), which excludes zero-valid-frame pixels from its convolution outright
(a normalized masked blur, `conv(y*valid)/conv(valid)`) rather than letting an undefined average
quietly pull down neighboring pixels' blur.

**Per-pixel regression.** For a given pixel, every exposure time contributes one `(blurred,
corr, w)`: `blurred` is that exposure's own local smooth-trend estimate at this pixel, and `corr
= blurred - x` is how far the exposure's weighted-average value fell from it. Every exposure
time pools into one weighted least-squares solve per pixel (closed form, no `(N, H, W)` stack of
individual frames ever materializes -- only one accumulator pair per exposure group, plus the
group's own running weighted-average accumulators while its frames stream through) for `corr =
a + b*blurred + c*blurred**2`.

**Applying the fit to `x`, not `blurred`, needs a correction of its own.** `FlatModel.evaluate`
only ever has a light frame's own `x` to work with -- there is no independent `blurred` for a
light frame's signal (computing one would mean blurring every full-res light frame at
`FLAT_HIGHPASS_SIGMA_PX` too, just to throw the blur away and keep the correction). Plugging
`x` directly into `a + b*x + c*x**2` in place of `blurred` is a first-order approximation that
under-corrects by an amount that grows with the defect's own size (still same-signed, unlike
the noise bias above -- it never flips) -- since `x` is already offset from `blurred` by the
very `corr` the model is trying to add back. `evaluate_refined` fixes this with one fixed-point
step: evaluate once at `x` to get a first-pass corrected estimate, then evaluate again at *that*
estimate (closer to the true `blurred` than `x` was) for the correction actually applied. See
`evaluate_refined`'s docstring for the numbers.

**Reliability.** A pixel's fit is trusted only if its 3x3 moment matrix is nonsingular --
otherwise `a = b = c = 0`, i.e. no correction, rather than solving an ill-posed system. With
one data point per exposure time, that needs at least 3 exposure groups with distinct `blurred`
values and nonzero weight at that pixel (fewer, and the matrix is exactly singular). There used
to also be a minimum-dynamic-range requirement (`MIN_RELIABLE_RANGE`) on top of this, dropped
because `FLAT_MODEL_VALUE_THRESHOLD` (below) already bounds every pixel's usable range from
above, to something a full-`[0, 1]`-bracket constant had no way to know about -- nonsingularity
is what's left to guard the 3-parameter solve itself. `reliable` on `FlatModel` records which
pixels got a real fit; nothing here checks *how narrow* that fit's own input range was, so a
fit from a very tight cluster of `blurred` values is trusted exactly like one from a wide
spread as long as the solve itself is well-posed.

**Value threshold: fit and apply only below it.** `FLAT_MODEL_VALUE_THRESHOLD` is measured
empirically (the 90th percentile of decoded pixel intensity across the flat bracket's own
0.02s-exposure frames -- see `FLAT_MODEL_VALUE_THRESHOLD`'s own comment for the exact
provenance), not derived from anything else in this module. It gates two different things,
each in the way that suits it:

  - At fit time, it is a hard cutoff on `blurred`, one more multiplicative factor on the
    per-exposure group weight `w` -- an exposure group is simply invisible to a pixel's
    regression wherever its own `blurred` is at or above the threshold, exactly as if that
    exposure had never been shot for that pixel. A hard cutoff is fine here: it only decides
    which *data* feeds the regression, not a value that ends up rendered.
  - At apply time (`FlatModel.apply_masked`), a hard cutoff would put a literal seam in the
    image -- neighboring pixels a hair apart in brightness, one corrected and one not. Instead
    the correction is blended out smoothly: full correction (weight 1.0) at/below
    `FLAT_MODEL_THRESHOLD_MARGIN_FRAC * value_threshold`, none at all (weight 0.0) at/above
    `value_threshold`, a raised-cosine taper (zero slope at both ends, so no kink either) over
    the margin between them. See `FlatModel._threshold_weight`.

Both still key off the same threshold on purpose: nothing above it ever influenced the fit, so
nothing above it should ever be treated as something the fit can trust a correction from.

**Additive, not multiplicative.** `FlatModel.evaluate_refined(value)` returns something in the
same units as `value` itself, clamped (per `evaluate` call) to `+/- FLAT_CORR_CLAMP`, and
`rawprep.apply_corrections` *adds* it to the light frame's own dark-subtracted value. This is a
departure from the old multiplicative-divide flat correction: `corr` is a residual in the same
units as a pixel value, so `a = b = c = 0` is exactly the identity (no correction), not a
divide-by-something-near-1. The clamp guards against extrapolation the same way the old
multiplicative clamp did -- light frames reach brightness levels the flats never sampled.

**GPU via torch.** Unlike `darkcal.py`'s cheap 2-parameter fit, this one's actual cost is the
`sigma=64` Gaussian blur (a few hundred taps wide) run two or three times per *exposure time*
over a full-res sensor image -- `scipy.ndimage.gaussian_filter` on CPU is the bottleneck.
Grouping by exposure (see above) is what keeps this to a handful of blurs total rather than one
per frame. The blur, the weighted-average accumulation, and the final per-pixel 3x3 solve all
run as `torch` tensor ops on GPU when one is visible (`_default_device`), CPU otherwise so the
small synthetic test stays GPU-free. `_masked_blur`/`_group_weights` keep their original plain-
numpy in/out signature (thin wrappers around the `_t`-suffixed tensor versions the fit calls
directly, so per-frame data never round-trips through numpy). `FlatModel` itself is still plain
numpy -- picklable, no torch -- only the fit's internals moved to GPU.

The blur itself runs in fp32 (`_BLUR_DTYPE`), not fp64: on this class of GPU, fp64 throughput is
throttled to roughly 1/100th of fp32 (measured on a full-res frame: 4.7s scipy CPU vs. 1.8s fp64
GPU vs. 0.02s fp32 GPU -- fp64 barely beats CPU, fp32 is what actually fixes the slowness), and a
smooth vignetting/dust estimate has no need of double precision. The streamed sums and the
closed-form 3x3 solve around it stay float64, since that part is cheap regardless of dtype and
it's what the fit's numerical stability actually depends on.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

OVERBURN_HI = 0.99                     # raw decoded fraction counting as saturated (matches
                                         # rawprep.py's RAW_OVERBURN_HI convention, duplicated
                                         # to avoid a cross-module import for one constant)
MIN_FLAT_FRAMES = 3                    # need at least this many flat frames to attempt a fit
                                         # at all (same spirit as darkcal.MIN_DARK_FRAMES)
FLAT_HIGHPASS_SIGMA_PX = 64.0          # Gaussian sigma separating vignetting (smooth, stays
                                         # out of corr via the blur) from dust/pixel-sensitivity
                                         # artefacts (kept, in corr); also the sigma used to
                                         # judge a pixel's local overburn density
NEIGHBORHOOD_OVERBURN_FRAC_MAX = 0.10  # a pixel's weight is zeroed in an exposure group if
                                         # more than this fraction of its Gaussian-weighted
                                         # neighborhood (same sigma as the blur) was, on
                                         # average, invalid across that group's frames --
                                         # protects against a blur that's already biased by a
                                         # nearby cluster of mostly-clipped pixels
FLAT_CORR_CLAMP = 0.05                 # evaluate() clamps the additive correction to +/- this,
                                         # in the same [0, 1] units as a dark-subtracted pixel
                                         # value -- guards against extrapolation when light
                                         # frames reach brightness levels the flats never sampled
FLAT_MODEL_VALUE_THRESHOLD = 0.066387  # 90th percentile of decoded (pre-dark-subtraction)
                                         # pixel intensity pooled across the flat bracket's 9
                                         # frames at t=0.02s, measured 2026-08-21 against
                                         # /home/slavik/e202602_eclipse/my_raws/flats. Data at
                                         # or above this is excluded from the fit (a
                                         # multiplicative gate on the per-exposure weight,
                                         # alongside overburn) and left uncorrected at
                                         # application time regardless of what the fit would
                                         # predict there -- see FlatModel.apply_masked and the
                                         # module docstring's "Value threshold" section.
FLAT_MODEL_THRESHOLD_MARGIN_FRAC = 0.9  # apply_masked's cosine taper: full correction at/below
                                         # this fraction of value_threshold, none at all
                                         # at/above value_threshold itself -- a hard cutoff
                                         # exactly at the fit's own cutoff would put a visible
                                         # seam in the image; this fades it out over the margin
                                         # instead. See FlatModel._threshold_weight.


def _default_device() -> torch.device:
    """CUDA if visible, CPU otherwise. Production always calls `device.require_cuda()` before
    reaching this module (see `pipeline.py`), so this only ever falls back to CPU for the small
    synthetic test, which is sized to run in ~1s either way."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reflect101_index(size: int, pad: int, device: torch.device, side: str) -> torch.Tensor:
    """Source index, for each of `pad` positions immediately outside `[0, size)` on `side`,
    that reproduces `scipy.ndimage`'s `mode="reflect"` (half-sample symmetric, edge value
    duplicated: `d c b a | a b c d | d c b a`) -- including `pad >= size`, needed only by one
    tiny unit-test canvas since production images are always far larger than the blur radius.
    Derivation: the padding is periodic with period `2*size`, alternating a flipped copy of the
    array and a plain copy as it moves away from the boundary."""
    k = torch.arange(1, pad + 1, device=device)
    period = 2 * size
    kk = (k - 1) % period + 1
    if side == "left":
        idx = torch.where(kk <= size, kk - 1, period - kk)
        return idx.flip(0)   # spatial order: farthest first .. nearest (position -1) last
    idx = torch.where(kk <= size, size - kk, kk - size - 1)
    return idx                # spatial order: nearest (position size) first .. farthest last


def _reflect_pad_1d(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Pad `x` along `dim` by `pad` on each side with `_reflect101_index`."""
    if pad == 0:
        return x
    size = x.shape[dim]
    left = x.index_select(dim, _reflect101_index(size, pad, x.device, "left"))
    right = x.index_select(dim, _reflect101_index(size, pad, x.device, "right"))
    return torch.cat([left, x, right], dim=dim)


def _gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype,
                         truncate: float = 4.0) -> torch.Tensor:
    """Same kernel `scipy.ndimage.gaussian_filter1d` builds by default: truncated at
    `truncate` (default 4) standard deviations, normalized to sum to 1."""
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


_BLUR_DTYPE = torch.float32   # the blur is a smooth vignetting/dust estimate, not a place that
                                # needs bit-exact fp64 -- and on this class of GPU fp64 throughput
                                # is ~100x weaker than fp32 (measured: a sigma=64 blur over a
                                # full-res frame, 4.7s scipy CPU vs 1.8s fp64 GPU vs 0.02s fp32
                                # GPU), so this is what actually fixes the reported slowness. The
                                # streamed sums/solve around it stay float64 for the closed-form
                                # fit's numerical stability; only this op runs narrower.


def _gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a 2-D tensor via two 1-D `conv2d` passes, reflect101-padded
    to match `scipy.ndimage.gaussian_filter(..., mode="reflect")` (see `_reflect_pad_1d`). Runs
    in `x`'s own dtype -- callers that want the fp32 fast path cast in first."""
    kernel = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    radius = kernel.shape[0] // 2
    xp = _reflect_pad_1d(x, radius, dim=1)
    xp = _reflect_pad_1d(xp, radius, dim=0)
    xp = xp.unsqueeze(0).unsqueeze(0)                      # (1, 1, H+2r, W+2r)
    x_h = F.conv2d(xp, kernel.view(1, 1, 1, -1))           # (1, 1, H+2r, W)
    x_v = F.conv2d(x_h, kernel.view(1, 1, -1, 1))          # (1, 1, H, W)
    return x_v.squeeze(0).squeeze(0)


def _masked_blur_t(y: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    """Tensor version of `_masked_blur`, operating in place on the GPU/CPU device `y` already
    lives on -- what the streamed fit loop calls directly, so per-frame data never round-trips
    through numpy. See `_masked_blur` for the semantics. The blur itself runs in `_BLUR_DTYPE`
    (fp32); the result is cast back to `y`'s own dtype (fp64 in the streamed fit)."""
    orig_dtype = y.dtype
    y32 = y.to(_BLUR_DTYPE)
    valid_f = valid.to(_BLUR_DTYPE)
    num = _gaussian_blur_2d(y32 * valid_f, sigma)
    den = _gaussian_blur_2d(valid_f, sigma)
    den_safe = torch.where(den > 1e-6, den, torch.ones_like(den))
    blurred = num / den_safe
    if torch.any(den <= 1e-6):
        blurred = torch.where(den > 1e-6, blurred, _gaussian_blur_2d(y32, sigma))
    return blurred.to(orig_dtype)


def _group_weights_t(
    bad_frac: torch.Tensor, valid_group: torch.Tensor, sigma: float, frac_max: float
) -> torch.Tensor:
    """Tensor version of `_group_weights` -- see there for the semantics. The blur runs in
    `_BLUR_DTYPE` (fp32); the returned weight is float64, matching the fit's sums."""
    local_frac = _gaussian_blur_2d(bad_frac.to(_BLUR_DTYPE), sigma).to(torch.float64)
    zero = local_frac.new_zeros(())
    one = local_frac.new_ones(())
    return torch.where(~valid_group | (local_frac > frac_max), zero, one)


def _group_average_t(
    raw_frames, t: float, dark_bias_t, dark_rate_t, overburn_hi: float, device: torch.device
):
    """Tensor version of `group_average` -- see there for the semantics. `raw_frames` is an
    iterable of (H, W) arrays, one exposure group's raw decoded frames (pre-dark-subtraction).
    Returns `(avg_x, valid_group, bad_frac, n_overburn_total)`, tensors on `device` plus a plain
    int, so the fit loop that calls this directly never round-trips through numpy."""
    sum_w = sum_wx = None
    n = 0
    n_overburn_total = 0
    for raw in raw_frames:
        x = torch.as_tensor(np.asarray(raw, dtype=np.float64), device=device)
        overburn = x >= overburn_hi            # raw decode, before dark subtraction
        n_overburn_total += int(overburn.sum().item())

        if dark_bias_t is not None:
            x = x - (dark_bias_t + dark_rate_t * float(t))

        valid = (~overburn).to(torch.float64)
        if sum_w is None:
            sum_w = torch.zeros_like(x)
            sum_wx = torch.zeros_like(x)
        sum_w += valid
        sum_wx += valid * x
        n += 1

    valid_group = sum_w > 0                          # >=1 valid frame at this pixel
    sum_w_safe = torch.where(valid_group, sum_w, torch.ones_like(sum_w))
    avg_x = torch.where(valid_group, sum_wx / sum_w_safe, torch.zeros_like(sum_wx))
    # Fraction of this group's frames that were invalid at each pixel -- the per-exposure
    # generalization of a single frame's boolean overburn mask (pushback 3 / option A: see
    # the module docstring's "Overburn is pixel-level" section).
    bad_frac = 1.0 - sum_w / n
    return avg_x, valid_group, bad_frac, n_overburn_total


def group_average(
    raw_frames, t: float, dark_model=None, overburn_hi: float = OVERBURN_HI,
    device: torch.device | None = None,
):
    """Per-pixel weighted mean of one exposure group's raw decoded flat frames
    (pre-dark-subtraction), excluding a frame's own overburn pixels outright from that pixel's
    mean -- the exact averaging `fit_flat_model_from_stack` uses per exposure group. Exposed
    (not `_`-prefixed) so callers outside this module -- e.g. a debug script visualizing what
    the fit actually saw -- compute the identical average rather than a plain, unweighted one
    that would silently disagree with it whenever a flat has any overburn at all.

    `dark_model` is anything with `.bias`/`.rate` (H, W) arrays, evaluated at `t`. Returns
    `(avg_x, valid_group, bad_frac)`, all plain numpy: `avg_x` is dark-subtracted (if
    `dark_model` given) and 0 wherever `valid_group` is False (no frame was ever valid there);
    `bad_frac` is the fraction of this group's frames invalid at each pixel (see
    `_group_weights`).
    """
    device = device or _default_device()
    dark_bias_t = dark_rate_t = None
    if dark_model is not None:
        dark_bias_t = torch.as_tensor(np.asarray(dark_model.bias, dtype=np.float64), device=device)
        dark_rate_t = torch.as_tensor(np.asarray(dark_model.rate, dtype=np.float64), device=device)
    avg_x, valid_group, bad_frac, _n_overburn = _group_average_t(
        raw_frames, t, dark_bias_t, dark_rate_t, overburn_hi, device)
    return avg_x.cpu().numpy(), valid_group.cpu().numpy(), bad_frac.cpu().numpy()


def _masked_blur(y: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of `y`, normalized so pixels flagged invalid (overburn -- clipped, not a
    real measurement) are excluded from every neighboring pixel's blur, rather than silently
    pulling it toward the clipped value. Mirror-pads at the border for the same reason as
    before: without it, blur near an edge would mix in fabricated zeros and bias the result
    there. Runs on GPU when one is visible (`_masked_blur_t`); this wrapper is plain numpy
    in/out for standalone/test use.

    Where a pixel has no valid neighbors at all within the kernel (only possible with a
    contiguous overburn region far larger than `sigma`), falls back to the unmasked blur there
    rather than dividing by ~0.
    """
    device = _default_device()
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), device=device)
    valid_t = torch.as_tensor(np.asarray(valid, dtype=bool), device=device)
    return _masked_blur_t(y_t, valid_t, sigma).cpu().numpy()


def _group_weights(
    bad_frac: np.ndarray, valid_group: np.ndarray, sigma: float, frac_max: float
) -> np.ndarray:
    """Per-pixel reliability weight for one exposure group: 0 wherever a pixel had zero valid
    (non-overburn) frames in this group at all (`~valid_group`), and 0 wherever the
    Gaussian-weighted (same sigma as the blur) local density of `bad_frac` -- the per-pixel
    fraction of this group's frames that were invalid -- exceeds `frac_max`. Protects against
    `corr` being computed from a blur that's already biased by a nearby cluster of
    mostly-invalid pixels, even at a pixel that was fine in every one of its own frames. Plain
    numpy in/out wrapper around `_group_weights_t`; see `_masked_blur` for why."""
    device = _default_device()
    bad_frac_t = torch.as_tensor(np.asarray(bad_frac, dtype=np.float64), device=device)
    valid_group_t = torch.as_tensor(np.asarray(valid_group, dtype=bool), device=device)
    return _group_weights_t(bad_frac_t, valid_group_t, sigma, frac_max).cpu().numpy()


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FlatModel:
    """Everything `rawprep.apply_corrections` needs. Plain numpy -- picklable, no torch.

    `a`/`b`/`c` were fit against `blurred` (see the module docstring), not against a light
    frame's own raw value -- so applying the correction to a light frame is `apply_masked`, not
    the raw `evaluate`. `a = b = c = 0` where `reliable` is False.
    """

    a: np.ndarray              # (H, W) float32, intercept
    b: np.ndarray              # (H, W) float32, linear coefficient
    c: np.ndarray              # (H, W) float32, quadratic coefficient
    reliable: np.ndarray       # (H, W) bool, whether this pixel had enough dynamic range to
                                 # trust the fit (False => a = b = c = 0)
    exposures: np.ndarray      # (n_exposures,) float64, distinct exposure times seen
    n_frames: int              # total flat frames streamed through the fit
    value_threshold: float     # data at/above this never entered the fit (see the module
                                 # docstring's "Value threshold" section) and is never
                                 # corrected by apply_masked either
    frame_report: list = field(default_factory=list)   # see print_report / fit_flat_model

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        """`a + b*value + c*value**2`, clamped to `+/- FLAT_CORR_CLAMP`. The fit's `value` axis
        is `blurred`, a smooth, near-noise-free estimate of the pixel's true brightness (see
        the module docstring) -- calling this directly on a light frame's own raw value is only
        a first-order approximation of the correction that value actually needs; use
        `apply_masked` for what `rawprep.apply_corrections` actually applies."""
        v = value.astype(np.float64) if isinstance(value, np.ndarray) else float(value)
        corr = (self.a.astype(np.float64) + self.b.astype(np.float64) * v
                + self.c.astype(np.float64) * v ** 2)
        return np.clip(corr, -FLAT_CORR_CLAMP, FLAT_CORR_CLAMP)

    def evaluate_refined(self, value: np.ndarray) -> np.ndarray:
        """The correction actually meant to be added to a light frame's dark-subtracted `value`.

        `a`/`b`/`c` are a function of `blurred`, which a light frame doesn't have (computing one
        would mean a full `FLAT_HIGHPASS_SIGMA_PX` blur of every light frame just to throw it
        away again). `evaluate(value)` -- plugging the raw value in where `blurred` belongs --
        under-corrects, by an amount that grows with the defect's own size: `value` is already
        offset from the true `blurred` by roughly the correction being solved for, so evaluating
        at `value` instead of at the (unknown) true `blurred` misses part of the curve.

        One fixed-point refinement step recovers most of that: evaluate once at `value` to get a
        first-pass corrected estimate, then evaluate again at *that* estimate, which sits closer
        to the true `blurred` than `value` did. On the module's own synthetic stress-test defect
        (15%/20% linear/quadratic, far larger than a real dust/sensitivity artifact), this took
        the residual error from ~8% (one-shot `evaluate(value)`) down to ~3% at the top of the
        bracket; for the percent-scale defects flats actually see, the remaining residual after
        this step should be negligible. Unlike the noise bias `blurred`-regression already fixes,
        this residual never changes sign -- it only ever under-corrects.
        """
        first_pass = value + self.evaluate(value)
        return self.evaluate(first_pass)

    def _threshold_weight(self, value: np.ndarray) -> np.ndarray:
        """1.0 at/below `FLAT_MODEL_THRESHOLD_MARGIN_FRAC * value_threshold`, 0.0 at/above
        `value_threshold`, a raised-cosine taper over the margin between -- zero slope at both
        ends, so `apply_masked`'s blend has no seam and no kink at either edge of the margin."""
        lo = FLAT_MODEL_THRESHOLD_MARGIN_FRAC * self.value_threshold
        hi = self.value_threshold
        v = value.astype(np.float64) if isinstance(value, np.ndarray) else float(value)
        u = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * u))

    def apply_masked(self, value: np.ndarray) -> np.ndarray:
        """The value to use in place of a light frame's dark-subtracted `value` -- what
        `rawprep.apply_corrections` actually calls.

        The fit never saw data at or above `value_threshold` (see the module docstring's
        "Value threshold" section), so evaluating the fitted curve there would be extrapolation
        onto brightness the flats never sampled, not correction. This always runs
        `evaluate_refined` over the whole array (simpler than branching per pixel, and the
        result is blended away where it's not wanted), then blends it against `value` itself by
        `_threshold_weight`: full correction well below the threshold, none at or above it, a
        smooth cosine taper over the margin between -- so two neighboring pixels a hair apart in
        brightness, one just under the threshold and one just over, don't come out of this with
        a visible seam between "corrected" and "untouched".
        """
        corrected = value + self.evaluate_refined(value)
        w = self._threshold_weight(value)
        return w * corrected + (1.0 - w) * value


# --------------------------------------------------------------------------- #
#  The fit itself — pure arrays, no file IO                                   #
# --------------------------------------------------------------------------- #
def fit_flat_model_from_stack(
    times, frames, dark_model=None, *,
    highpass_sigma_px: float = FLAT_HIGHPASS_SIGMA_PX,
    overburn_hi: float = OVERBURN_HI,
    neighborhood_overburn_frac_max: float = NEIGHBORHOOD_OVERBURN_FRAC_MAX,
    min_flat_frames: int = MIN_FLAT_FRAMES,
    value_threshold: float = FLAT_MODEL_VALUE_THRESHOLD,
    device: torch.device | None = None,
):
    """Average `frames` (one per entry of `times`) within each distinct exposure time, then run
    one per-pixel weighted least-squares fit of `corr = a + b*blurred + c*blurred**2` over those
    per-exposure averages.

    `frames` is any iterable of (H, W) arrays, raw decoded (pre-dark-subtraction), one per entry
    of `times`. `dark_model` is anything with `.bias`/`.rate` (H, W) arrays (a
    darkcal.DarkModel or a plain stand-in), evaluated at each frame's own exposure time. The
    keyword parameters are exposed only so a small synthetic test can rescale them to its own
    tiny canvas; production callers should leave them at the defaults. `device` defaults to
    `_default_device()` (GPU if visible); every array (weighted-average accumulators, blur,
    weights, the fit's own sums) lives on it, so nothing round-trips through numpy until the
    final result.

    `value_threshold` gates which exposure groups a pixel's regression ever sees: a group is
    invisible to a pixel wherever that group's own `blurred` is at or above the threshold (see
    the module docstring's "Value threshold" section) -- exactly as if that exposure had never
    been shot for that pixel.

    Returns `(a, b, c, used_exposures, n_frames, frame_report, reliable)`, all plain numpy.
    """
    device = device or _default_device()
    times = np.asarray(times, dtype=np.float64)
    frames = list(frames)
    n = len(frames)
    assert times.shape[0] == n, (times.shape, n, "times and frames length mismatch")
    assert n >= min_flat_frames, f"need >= {min_flat_frames} flat frames, got {n}"

    dark_bias_t = dark_rate_t = None
    if dark_model is not None:
        dark_bias_t = torch.as_tensor(np.asarray(dark_model.bias, dtype=np.float64), device=device)
        dark_rate_t = torch.as_tensor(np.asarray(dark_model.rate, dtype=np.float64), device=device)

    times_r = np.round(times, 9)
    group_idx: dict[float, list[int]] = {}
    for i, t in enumerate(times_r):
        group_idx.setdefault(float(t), []).append(i)

    S0 = S1 = S2 = S3 = S4 = None
    T0 = T1 = T2 = None
    frame_report = []

    for t in sorted(group_idx):
        idxs = group_idx[t]
        # Weighted average within this exposure: a frame's own overburn pixels are excluded
        # outright from the sum, not merely down-weighted, so one saturated region in one frame
        # can't pull that pixel's average toward the clipped value -- while every other pixel in
        # that same frame, and every other frame's contribution to *this* pixel, is unaffected.
        # Shared with `group_average` (the public numpy wrapper) so anything visualizing what
        # the fit saw -- e.g. a debug script -- computes the identical average, not a plain one.
        raw_frames = (frames[i] for i in tqdm.tqdm(idxs, desc=f"flat average t={t:.6f}"))
        avg_x, valid_group, bad_frac, n_overburn_total = _group_average_t(
            raw_frames, t, dark_bias_t, dark_rate_t, overburn_hi, device)

        blurred = _masked_blur_t(avg_x, valid_group, highpass_sigma_px)
        corr = blurred - avg_x
        w = _group_weights_t(bad_frac, valid_group, highpass_sigma_px, neighborhood_overburn_frac_max)
        # Value threshold: this exposure is invisible to a pixel's regression wherever its own
        # blurred value is at or above value_threshold -- see the module docstring's "Value
        # threshold" section.
        w = w * (blurred < value_threshold).to(torch.float64)

        if S0 is None:
            S0 = torch.zeros_like(avg_x); S1 = torch.zeros_like(avg_x); S2 = torch.zeros_like(avg_x)
            S3 = torch.zeros_like(avg_x); S4 = torch.zeros_like(avg_x)
            T0 = torch.zeros_like(avg_x); T1 = torch.zeros_like(avg_x); T2 = torch.zeros_like(avg_x)

        # Regressor is `blurred`, not `x` -- see the module docstring ("Regressor is `blurred`,
        # not `x`") for why: using `x` here would feed this exposure's own noise into both sides
        # of the fit at once.
        wb = w * blurred
        wb2 = wb * blurred
        S0 += w
        S1 += wb
        S2 += wb2
        S3 += wb2 * blurred
        S4 += wb2 * blurred * blurred
        T0 += w * corr
        T1 += wb * corr
        T2 += wb2 * corr

        frame_report.append({
            "t": float(t),
            "n_frames": len(idxs),
            "n_overburn": n_overburn_total,
            "mean_weight": float(w.mean().item()),
        })

    # Per-pixel weighted OLS for corr = a + b*blurred + c*blurred^2: a batched 3x3 closed-form
    # solve, one (symmetric) moment matrix and rhs vector per pixel -- same streamed-sums idea
    # darkcal.py uses for bias + rate*t, generalized to 3 parameters and a per-pixel regressor.
    # Runs as one batched torch.linalg solve over all H*W pixels at once.
    H, W_ = S0.shape
    M = torch.empty((H, W_, 3, 3), dtype=torch.float64, device=device)
    M[..., 0, 0] = S0; M[..., 0, 1] = S1; M[..., 0, 2] = S2
    M[..., 1, 0] = S1; M[..., 1, 1] = S2; M[..., 1, 2] = S3
    M[..., 2, 0] = S2; M[..., 2, 1] = S3; M[..., 2, 2] = S4
    V = torch.stack([T0, T1, T2], dim=-1)

    det = torch.linalg.det(M)
    nonsingular = det.abs() > 1e-12
    eye3 = torch.eye(3, dtype=torch.float64, device=device)
    M_safe = torch.where(nonsingular[..., None, None], M, eye3)
    coeffs = torch.linalg.solve(
        M_safe.reshape(-1, 3, 3), V.reshape(-1, 3, 1)
    ).reshape(H, W_, 3)
    a_fit, b_fit, c_fit = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]

    reliable = nonsingular
    zero = a_fit.new_zeros(())
    a = torch.where(reliable, a_fit, zero)
    b = torch.where(reliable, b_fit, zero)
    c = torch.where(reliable, c_fit, zero)

    assert bool(torch.all(torch.isfinite(a)) and torch.all(torch.isfinite(b))
                and torch.all(torch.isfinite(c))), (
        "flat model fit produced non-finite values -- check the flats/dark model")

    used_exposures = np.asarray(sorted(set(np.round(times, 9))), dtype=np.float64)
    return (
        a.to(torch.float32).cpu().numpy(), b.to(torch.float32).cpu().numpy(),
        c.to(torch.float32).cpu().numpy(), used_exposures, n, frame_report,
        reliable.cpu().numpy(),
    )


# --------------------------------------------------------------------------- #
#  Decode + orchestrate                                                       #
# --------------------------------------------------------------------------- #
def _flat_files(flat_dir) -> list:
    flat_dir = Path(flat_dir)
    return sorted(list(flat_dir.glob("*.NEF")) + list(flat_dir.glob("*.nef")))


def fit_flat_model(flat_dir, dark_model=None, device: torch.device | None = None) -> FlatModel:
    """Decode every flat NEF in `flat_dir` and fit the per-pixel brightness-dependent flat
    model, averaged within each distinct exposure time."""
    from eclipse_v7.inputs import _decode_nef_linear
    from eclipse_v7.stage0 import get_info_from_exif

    files = _flat_files(flat_dir)
    assert files, f"no flat NEFs found in {flat_dir}"

    times = [float(get_info_from_exif(f)[0]) for f in files]

    def _decoded_frames():
        for f in tqdm.tqdm(files, desc="flat decode"):
            yield _decode_nef_linear(f)

    a, b, c, used_exposures, n_frames, frame_report, reliable = fit_flat_model_from_stack(
        times, _decoded_frames(), dark_model=dark_model, device=device)
    return FlatModel(
        a=a, b=b, c=c, reliable=reliable, exposures=used_exposures, n_frames=n_frames,
        value_threshold=FLAT_MODEL_VALUE_THRESHOLD, frame_report=frame_report,
    )


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(model: FlatModel) -> None:
    print(f"flatcal: fit from {model.n_frames} flat frame(s) across "
          f"{len(model.exposures)} distinct exposure time(s) "
          f"({model.exposures.min():.6f} .. {model.exposures.max():.6f} s)")

    print(f"{'exposure (s)':>14} {'frames':>7} {'mean weight':>13}")
    for r in sorted(model.frame_report, key=lambda r: r["t"]):
        print(f"{r['t']:>14.6f} {r['n_frames']:>7d} {r['mean_weight']:>13.3%}")

    frac_reliable = float(model.reliable.mean())
    print(f"flatcal: {frac_reliable:.3%} of pixels have a reliable fit (nonsingular 3x3 solve, "
          f">=3 exposure groups with distinct blurred values below value_threshold); the rest "
          f"fall back to no correction (a=b=c=0)")
    print(f"flatcal: value_threshold = {model.value_threshold:.5f} -- data at/above this never "
          f"entered the fit; apply_masked fades the correction out over "
          f"[{FLAT_MODEL_THRESHOLD_MARGIN_FRAC * model.value_threshold:.5f}, "
          f"{model.value_threshold:.5f}] and never applies it at/above the threshold itself, "
          f"regardless of `reliable`")

    a, b, c = model.a, model.b, model.c
    print(f"flatcal: intercept  (a) mean {a.mean():+.5f}, min {a.min():+.5f}, max {a.max():+.5f}")
    print(f"flatcal: linear     (b) mean {b.mean():+.5f}, min {b.min():+.5f}, max {b.max():+.5f}")
    print(f"flatcal: quadratic  (c) mean {c.mean():+.5f}, min {c.min():+.5f}, max {c.max():+.5f}")

    # "bright" probes just below value_threshold, not 1.0 -- apply_masked never lets the fit
    # touch anything at or above it, so clamp stats there would describe a case that can't occur.
    for v, label in ((0.0, "dim"), (0.999 * model.value_threshold, "near threshold")):
        corr = model.evaluate_refined(np.full_like(a, v, dtype=np.float64))
        n_lo = int(np.sum(corr <= -FLAT_CORR_CLAMP + 1e-6))
        n_hi = int(np.sum(corr >= FLAT_CORR_CLAMP - 1e-6))
        print(f"flatcal: at value={v:.5f} ({label}): "
              f"{n_lo:,} pixel(s) clamped low, {n_hi:,} clamped high "
              f"({(n_lo + n_hi) / corr.size:.3%})")


def save(model: FlatModel, path: Path) -> Path:
    path = Path(path)
    with open(path, "wb") as fd:
        pickle.dump(model, fd)
    print(f"Saved {path} (flat model).")
    return path


def load(path: Path) -> FlatModel:
    with open(Path(path), "rb") as fd:
        return pickle.load(fd)


def run(flat_dir, dark_model=None, out_pkl: Path | None = None,
        device: torch.device | None = None) -> FlatModel:
    """Fit, report, optionally save."""
    model = fit_flat_model(flat_dir, dark_model=dark_model, device=device)
    print_report(model)
    if out_pkl is not None:
        save(model, out_pkl)
    return model
