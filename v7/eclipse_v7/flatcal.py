"""Per-pixel flat field, streamed per flat frame (not grouped by exposure time): each frame
contributes one `(x, corr, w)` data point per pixel to a per-pixel weighted least-squares fit
of `corr = a + b*x + c*x**2`, where `x` is that pixel's own dark-subtracted brightness in that
frame and `corr` is how far the pixel deviates from its local (Gaussian-blurred) smooth trend
-- dust shadows, pixel-to-pixel sensitivity, not the smooth optical vignetting, which is what
the blur estimates and what stays out of `corr`.

**Per-frame processing, not per-group.** Earlier versions of this module grouped flats by
exposure time and fit across per-group averages. This version drops that entirely: every valid
flat frame streams through once and contributes directly to a per-pixel closed-form weighted
least-squares accumulation (the same technique `darkcal.py` uses for `bias + rate*t`,
generalized to a per-pixel x-value, three parameters instead of two, and a per-datapoint
weight). This lets one badly-saturated region in one frame get excluded without discarding
that frame's information for every other pixel, and lets brightness variation across a bracket
of many exposures feed the fit directly rather than being averaged away first.

**Overburn is pixel-level, not frame-level.** Saturation is checked on the raw decode, before
dark subtraction (same `OVERBURN_HI` convention as `rawprep.py`), before anything else runs on
a frame. It feeds two separate things: (1) a per-pixel weight `w` for this specific frame
(`_flat_weights`) -- zero at an overburn pixel itself, and also zero wherever the
Gaussian-weighted (same sigma as the blur) local density of overburn pixels exceeds
`NEIGHBORHOOD_OVERBURN_FRAC_MAX`, since a blur computed near enough overburn pixels is itself
biased even at a pixel that isn't saturated; (2) the blur itself (`_masked_blur`), which
excludes overburn pixels from its convolution outright (a normalized masked blur,
`conv(y*valid)/conv(valid)`) rather than letting their wrong, clipped raw value quietly pull
down neighboring pixels' blur.

**Per-pixel regression.** For a given pixel, every frame contributes one `(x, corr, w)`: `x`
is that frame's own dark-subtracted value at this pixel, and `corr` is computed from that same
value (`blur - x`) -- deliberately the same number on both sides. That shares each frame's own
noise between the regressor and the response with a fixed -1 coefficient, which biases `b`
(and `c`) toward more negative than the true effect; the bias does not shrink with more frames
(more data makes the estimate more *precisely* biased, not less biased) -- accepted here on the
judgment that a flat bracket's exposure-to-exposure brightness swing should dominate over any
one frame's own shot/read noise for most pixels. Worth checking for if a fitted `b`/`c` map
ever looks suspiciously systematic. All frames across every exposure time pool into one
weighted least-squares solve per pixel (streamed sums, closed form, no `(N, H, W)` stack ever
materializes) for `corr = a + b*x + c*x**2`.

**Reliability.** A pixel's fit is trusted only if the frames that weighted it (`w == 1`) span
more than `MIN_RELIABLE_RANGE` of dynamic range (assuming max pixel value 1.0) -- otherwise
`a = b = c = 0`, i.e. no correction, rather than trusting a fit extrapolated from a narrow
brightness range. `reliable` on `FlatModel` records which pixels got a real fit.

**Additive, not multiplicative.** `FlatModel.evaluate(value)` returns `a + b*value +
c*value**2`, clamped to `+/- FLAT_CORR_CLAMP`, and `rawprep.apply_corrections` *adds* it to the
light frame's own dark-subtracted value. This is a departure from the old multiplicative-divide
flat correction: `corr` is a residual in the same units as a pixel value, so `a = b = c = 0` is
exactly the identity (no correction), not a divide-by-something-near-1. The clamp guards against
extrapolation the same way the old multiplicative clamp did -- light frames reach brightness
levels the flats never sampled.

**GPU via torch.** Unlike `darkcal.py`'s cheap 2-parameter fit, this one's actual cost is the
`sigma=64` Gaussian blur (a few hundred taps wide) run two or three times per frame over a
full-res sensor image, streamed over every flat frame -- `scipy.ndimage.gaussian_filter` on CPU
is the bottleneck. The blur, the streamed sums, and the final per-pixel 3x3 solve all run as
`torch` tensor ops on GPU when one is visible (`_default_device`), CPU otherwise so the small
synthetic test stays GPU-free. `_masked_blur`/`_flat_weights` keep their original plain-numpy
in/out signature (thin wrappers around the `_t`-suffixed tensor versions the streamed loop calls
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
NEIGHBORHOOD_OVERBURN_FRAC_MAX = 0.10  # a pixel's weight is zeroed in a frame if more than
                                         # this fraction of its Gaussian-weighted neighborhood
                                         # (same sigma as the blur) is itself overburn --
                                         # protects against a blur that's already biased by
                                         # nearby clipped pixels
MIN_RELIABLE_RANGE = 0.5               # a pixel's fit is trusted only if the frames that
                                         # weighted it span more than this much dynamic range
                                         # (assuming max pixel value 1.0); otherwise falls back
                                         # to a = b = c = 0 (no correction)
FLAT_CORR_CLAMP = 0.05                 # evaluate() clamps the additive correction to +/- this,
                                         # in the same [0, 1] units as a dark-subtracted pixel
                                         # value -- guards against extrapolation when light
                                         # frames reach brightness levels the flats never sampled


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


def _flat_weights_t(overburn: torch.Tensor, sigma: float, frac_max: float) -> torch.Tensor:
    """Tensor version of `_flat_weights` -- see there for the semantics. The blur runs in
    `_BLUR_DTYPE` (fp32); the returned weight is float64, matching the streamed fit's sums."""
    local_frac = _gaussian_blur_2d(overburn.to(_BLUR_DTYPE), sigma).to(torch.float64)
    zero = local_frac.new_zeros(())
    one = local_frac.new_ones(())
    return torch.where(overburn | (local_frac > frac_max), zero, one)


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


def _flat_weights(overburn: np.ndarray, sigma: float, frac_max: float) -> np.ndarray:
    """Per-pixel reliability weight for one flat frame: 0 at an overburn pixel itself, and 0
    wherever the Gaussian-weighted (same sigma as the blur) local fraction of overburn pixels
    exceeds `frac_max` -- protects against `corr` being computed from a blur that's already
    biased by nearby saturated pixels, even at a pixel that isn't saturated itself. Plain numpy
    in/out wrapper around `_flat_weights_t`; see `_masked_blur` for why."""
    device = _default_device()
    overburn_t = torch.as_tensor(np.asarray(overburn, dtype=bool), device=device)
    return _flat_weights_t(overburn_t, sigma, frac_max).cpu().numpy()


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FlatModel:
    """Everything `rawprep.apply_corrections` needs. Plain numpy -- picklable, no torch.

    The correction depends on the light frame's own value: `evaluate(value)` computes
    `a + b*value + c*value**2`, clamped to `+/- FLAT_CORR_CLAMP`, meant to be *added* to the
    light frame's own dark-subtracted value. `a = b = c = 0` where `reliable` is False.
    """

    a: np.ndarray              # (H, W) float32, intercept
    b: np.ndarray              # (H, W) float32, linear coefficient
    c: np.ndarray              # (H, W) float32, quadratic coefficient
    reliable: np.ndarray       # (H, W) bool, whether this pixel had enough dynamic range to
                                 # trust the fit (False => a = b = c = 0)
    exposures: np.ndarray      # (n_exposures,) float64, distinct exposure times seen
    n_frames: int              # total flat frames streamed through the fit
    frame_report: list = field(default_factory=list)   # see print_report / fit_flat_model

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        """`a + b*value + c*value**2`, clamped to `+/- FLAT_CORR_CLAMP`. `value` is the light
        frame's own dark-subtracted signal, same shape as `a`/`b`/`c`. Meant to be *added* to
        that value, not divided into it."""
        v = value.astype(np.float64) if isinstance(value, np.ndarray) else float(value)
        corr = (self.a.astype(np.float64) + self.b.astype(np.float64) * v
                + self.c.astype(np.float64) * v ** 2)
        return np.clip(corr, -FLAT_CORR_CLAMP, FLAT_CORR_CLAMP)


# --------------------------------------------------------------------------- #
#  The fit itself — pure arrays, no file IO                                   #
# --------------------------------------------------------------------------- #
def fit_flat_model_from_stack(
    times, frames, dark_model=None, *,
    highpass_sigma_px: float = FLAT_HIGHPASS_SIGMA_PX,
    overburn_hi: float = OVERBURN_HI,
    neighborhood_overburn_frac_max: float = NEIGHBORHOOD_OVERBURN_FRAC_MAX,
    min_reliable_range: float = MIN_RELIABLE_RANGE,
    min_flat_frames: int = MIN_FLAT_FRAMES,
    device: torch.device | None = None,
):
    """Stream `frames` (one per entry of `times`) through a per-pixel weighted least-squares
    fit of `corr = a + b*x + c*x**2`.

    `frames` is any iterable of (H, W) arrays, raw decoded (pre-dark-subtraction), one per
    entry of `times`. `dark_model` is anything with `.bias`/`.rate` (H, W) arrays (a
    darkcal.DarkModel or a plain stand-in), evaluated at each frame's own exposure time. The
    keyword parameters are exposed only so a small synthetic test can rescale them to its own
    tiny canvas; production callers should leave them at the defaults. `device` defaults to
    `_default_device()` (GPU if visible); every per-frame array (blur, weights, accumulated
    sums) lives on it, so nothing round-trips through numpy until the final result.

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

    S0 = S1 = S2 = S3 = S4 = None
    T0 = T1 = T2 = None
    min_x = max_x = None
    frame_report = []

    for t, raw in zip(times, tqdm.tqdm(frames, desc="flat fit")):
        x = torch.as_tensor(np.asarray(raw, dtype=np.float64), device=device)
        overburn = x >= overburn_hi            # raw decode, before dark subtraction

        if dark_bias_t is not None:
            x = x - (dark_bias_t + dark_rate_t * float(t))

        blurred = _masked_blur_t(x, ~overburn, highpass_sigma_px)
        corr = blurred - x
        w = _flat_weights_t(overburn, highpass_sigma_px, neighborhood_overburn_frac_max)

        if S0 is None:
            S0 = torch.zeros_like(x); S1 = torch.zeros_like(x); S2 = torch.zeros_like(x)
            S3 = torch.zeros_like(x); S4 = torch.zeros_like(x)
            T0 = torch.zeros_like(x); T1 = torch.zeros_like(x); T2 = torch.zeros_like(x)
            min_x = torch.full_like(x, float("inf"))
            max_x = torch.full_like(x, float("-inf"))

        wx = w * x
        wx2 = wx * x
        S0 += w
        S1 += wx
        S2 += wx2
        S3 += wx2 * x
        S4 += wx2 * x * x
        T0 += w * corr
        T1 += wx * corr
        T2 += wx2 * corr

        has_w = w > 0
        min_x = torch.where(has_w, torch.minimum(min_x, x), min_x)
        max_x = torch.where(has_w, torch.maximum(max_x, x), max_x)

        frame_report.append({
            "t": float(t),
            "n_overburn": int(overburn.sum().item()),
            "mean_weight": float(w.mean().item()),
        })

    # Per-pixel weighted OLS for corr = a + b*x + c*x^2: a batched 3x3 closed-form solve, one
    # (symmetric) moment matrix and rhs vector per pixel -- same streamed-sums idea darkcal.py
    # uses for bias + rate*t, generalized to 3 parameters and a per-pixel x. Runs as one
    # batched torch.linalg solve over all H*W pixels at once.
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

    reliable = nonsingular & ((max_x - min_x) > min_reliable_range)
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
    model, streaming frame by frame."""
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
        frame_report=frame_report,
    )


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(model: FlatModel) -> None:
    print(f"flatcal: fit from {model.n_frames} flat frame(s) across "
          f"{len(model.exposures)} distinct exposure time(s) "
          f"({model.exposures.min():.6f} .. {model.exposures.max():.6f} s)")

    by_t: dict[float, list] = {}
    for r in model.frame_report:
        by_t.setdefault(round(r["t"], 9), []).append(r)
    print(f"{'exposure (s)':>14} {'frames':>7} {'mean weight':>13}")
    for t in sorted(by_t):
        rows = by_t[t]
        mean_w = float(np.mean([r["mean_weight"] for r in rows]))
        print(f"{t:>14.6f} {len(rows):>7d} {mean_w:>13.3%}")

    frac_reliable = float(model.reliable.mean())
    print(f"flatcal: {frac_reliable:.3%} of pixels have a reliable fit (dynamic range > "
          f"{MIN_RELIABLE_RANGE}); the rest fall back to no correction (a=b=c=0)")

    a, b, c = model.a, model.b, model.c
    print(f"flatcal: intercept  (a) mean {a.mean():+.5f}, min {a.min():+.5f}, max {a.max():+.5f}")
    print(f"flatcal: linear     (b) mean {b.mean():+.5f}, min {b.min():+.5f}, max {b.max():+.5f}")
    print(f"flatcal: quadratic  (c) mean {c.mean():+.5f}, min {c.min():+.5f}, max {c.max():+.5f}")

    for v, label in ((0.0, "dim"), (1.0, "bright")):
        corr = model.evaluate(np.full_like(a, v, dtype=np.float64))
        n_lo = int(np.sum(corr <= -FLAT_CORR_CLAMP + 1e-6))
        n_hi = int(np.sum(corr >= FLAT_CORR_CLAMP - 1e-6))
        print(f"flatcal: at value={v:.2f} ({label}): "
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
