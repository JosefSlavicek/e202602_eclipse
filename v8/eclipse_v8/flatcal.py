"""Per-pixel flat field: corrects dust shadows and pixel-to-pixel sensitivity, not the
smooth optical vignetting. `blurred` (below) captures the vignetting locally, but it's
only ever the baseline `corr` is measured against -- vignetting itself is never corrected
anywhere in this pipeline.

For each pixel we fit `corr = a + b*blurred + c*blurred**2`, where `blurred` is a smoothed
version of the flat frames at that pixel -- its local, dust-free trend -- and `corr` is how
far the pixel's real value falls from that trend. We fit against `blurred` rather than the
raw pixel value on purpose: the raw value carries this pixel's own noise, and using it on
both sides of the fit would bias the result. `blurred` is an average over thousands of
other pixels, so it's essentially noise-free.

Flats are grouped by exposure time before fitting. That's both faster (one blur per
exposure time instead of one per frame) and less noisy. To stop one overburned (saturated)
frame from ruining a whole group's average, each pixel's average within a group skips any
frame where that exact pixel was saturated.

The fit only ever sees `blurred`, but correcting a light frame later only has that frame's
own raw value to work with -- there's no independent `blurred` for it. Plugging the raw
value straight into the fit under-corrects a little, so `evaluate_refined` runs the
correction twice: once for a rough estimate, then again using that estimate in place of
the raw value, which lands much closer to the truth.

A pixel only gets a real correction if we saw it in at least 3 exposure groups at
different brightness levels; otherwise it's left alone (`a = b = c = 0`). The correction
is also only trusted below a measured brightness threshold -- above it we taper it off
smoothly rather than cut it off sharply, so there's no visible seam in the image.

The correction is additive (added to the frame's own value), not the old "divide by
something near 1" style, so no correction at all is simply zero.

Runs on GPU where available, since blurring a full-resolution frame this many times is
the expensive part.
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
FLAT_MODEL_VALUE_THRESHOLD = 0.08  # 90th percentile of decoded (pre-dark-subtraction)
                                         # pixel intensity pooled across the flat bracket's 9
                                         # frames at t=0.02s, measured 2026-08-21 against
                                         # /home/slavik/e202602_eclipse/my_raws/flats. Data at
                                         # or above this is excluded from the fit (a
                                         # multiplicative gate on the per-exposure weight,
                                         # alongside overburn) and left uncorrected at
                                         # application time regardless of what the fit would
                                         # predict there -- see FlatModel.apply_masked and the
                                         # module docstring's "Value threshold" section.
FLAT_MODEL_THRESHOLD_MARGIN_FRAC = 0.75  # apply_masked's cosine taper: full correction at/below
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
    """Source index for each of `pad` positions just outside `[0, size)`, matching
    `scipy.ndimage`'s `mode="reflect"` (edge value duplicated: `d c b a | a b c d | d c b a`)."""
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


_BLUR_DTYPE = torch.float32   # the blur is a smooth vignetting/dust estimate, doesn't need
                                # fp64 precision, and on this GPU fp64 is ~100x slower than
                                # fp32 (measured on a full-res frame: 4.7s scipy CPU vs 1.8s
                                # fp64 GPU vs 0.02s fp32 GPU). The fit's own sums stay float64
                                # for numerical stability; only the blur runs narrow.


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
    """Gaussian blur of `y` that ignores invalid (overburned) pixels: they don't pull down
    their neighbors' blur, and the border is mirror-padded so it isn't biased by fabricated
    zeros either. Falls back to the unmasked blur where a pixel has no valid neighbors at all."""
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
    """Per-pixel weight for one exposure group's blur: 0 where a pixel had no valid frames
    in this group at all, and 0 wherever too many of its neighbors were invalid too (a blur
    near a cluster of bad pixels is unreliable even where this one pixel was fine)."""
    local_frac = _gaussian_blur_2d(bad_frac.to(_BLUR_DTYPE), sigma).to(torch.float64)
    zero = local_frac.new_zeros(())
    one = local_frac.new_ones(())
    return torch.where(~valid_group | (local_frac > frac_max), zero, one)


def _group_average_t(
    raw_frames, t: float, dark_bias_t, dark_rate_t, overburn_hi: float, device: torch.device
):
    """Per-pixel weighted mean of one exposure group's raw flat frames, excluding each
    frame's own overburned pixels from that pixel's mean.

    Returns `(avg_x, valid_group, bad_frac, n_overburn_total)`: `avg_x` is dark-subtracted
    (if `dark_bias_t`/`dark_rate_t` are given) and 0 where `valid_group` is False; `bad_frac`
    is the fraction of this group's frames invalid at each pixel."""
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


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FlatModel:
    """Everything `rawprep.apply_corrections` needs. Plain numpy -- picklable, no torch.

    Use `apply_masked`, not `evaluate` directly, to correct a light frame (see the module
    docstring for why). `a = b = c = 0` where `reliable` is False.
    """

    a: np.ndarray              # (H, W) float32, intercept
    b: np.ndarray              # (H, W) float32, linear coefficient
    c: np.ndarray              # (H, W) float32, quadratic coefficient
    reliable: np.ndarray       # (H, W) bool, whether the fit's 3x3 solve was nonsingular
                                 # (False => a = b = c = 0). Doesn't check how *narrow* the
                                 # input range was -- a fit from 3 barely-different `blurred`
                                 # values is trusted the same as one from a wide spread.
    exposures: np.ndarray      # (n_exposures,) float64, distinct exposure times seen
    n_frames: int              # total flat frames streamed through the fit
    value_threshold: float     # data at/above this never entered the fit (see the module
                                 # docstring's "Value threshold" section) and is never
                                 # corrected by apply_masked either
    frame_report: list = field(default_factory=list)   # see print_report / fit_flat_model

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        """`a + b*value + c*value**2`, clamped to `+/- FLAT_CORR_CLAMP`.

        This is only a first-order approximation when called on a light frame's raw
        value -- use `apply_masked` for the real correction (see the module docstring)."""
        v = value.astype(np.float64) if isinstance(value, np.ndarray) else float(value)
        corr = (self.a.astype(np.float64) + self.b.astype(np.float64) * v
                + self.c.astype(np.float64) * v ** 2)
        return np.clip(corr, -FLAT_CORR_CLAMP, FLAT_CORR_CLAMP)

    def evaluate_refined(self, value: np.ndarray) -> np.ndarray:
        """The correction to actually add to a light frame's dark-subtracted `value`.

        `evaluate(value)` alone under-corrects a bit, since it's evaluated at the raw
        value instead of the true `blurred` it was fit against (see the module
        docstring). This applies it twice: once for a rough estimate, then again using
        that estimate in place of `value`, which lands much closer to the truth. On our
        synthetic stress-test defect (deliberately much larger than a real one) this
        brought the residual error from ~8% down to ~3%.
        """
        first_pass = value + self.evaluate(value)
        return self.evaluate(first_pass)

    def _threshold_weight(self, value: np.ndarray) -> np.ndarray:
        """1.0 well below `value_threshold`, 0.0 at/above it, a smooth taper in between --
        so `apply_masked`'s blend has no visible seam."""
        lo = FLAT_MODEL_THRESHOLD_MARGIN_FRAC * self.value_threshold
        hi = self.value_threshold
        v = value.astype(np.float64) if isinstance(value, np.ndarray) else float(value)
        u = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * u))

    def apply_masked(self, value: np.ndarray) -> np.ndarray:
        """What `rawprep.apply_corrections` actually calls to correct a light frame.

        Runs `evaluate_refined` everywhere, then blends it against the raw value using
        `_threshold_weight`: full correction well below the threshold, none above it, no
        visible seam at the boundary. We never trust the fit above `value_threshold` --
        it never saw data there (see the module docstring).
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
    """Average `frames` per exposure time, then fit one per-pixel `corr = a + b*blurred +
    c*blurred**2` over those averages (see the module docstring).

    `frames` is raw decoded, pre-dark-subtraction, one per entry of `times`. `dark_model`
    is anything with `.bias`/`.rate` (H, W) arrays. The keyword parameters exist mainly so
    the synthetic test can rescale them to a tiny canvas -- production callers should leave
    them at the defaults. Runs on `device` (GPU if visible) end to end.

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
    from eclipse_v8.inputs import _decode_nef_linear
    from eclipse_v8.stage0 import get_info_from_exif

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


def run(flat_dir, dark_model=None, out_pkl: Path | None = None,
        device: torch.device | None = None) -> FlatModel:
    """Fit, report, optionally save."""
    model = fit_flat_model(flat_dir, dark_model=dark_model, device=device)
    print_report(model)
    if out_pkl is not None:
        save(model, out_pkl)
    return model
