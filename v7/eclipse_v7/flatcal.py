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
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

OVERBURN_HI = 0.99                     # raw decoded fraction counting as saturated (matches
                                         # rawprep.py's RAW_OVERBURN_HI convention, duplicated
                                         # to avoid a cross-module import for one constant)
MIN_FLAT_FRAMES = 3                    # need at least this many flat frames to attempt a fit
                                         # at all (same spirit as darkcal.MIN_DARK_FRAMES)
FLAT_HIGHPASS_SIGMA_PX = 24.0          # Gaussian sigma separating vignetting (smooth, stays
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


def _masked_blur(y: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of `y`, normalized so pixels flagged invalid (overburn -- clipped, not a
    real measurement) are excluded from every neighboring pixel's blur, rather than silently
    pulling it toward the clipped value. `mode="reflect"` mirror-pads at the border for the
    same reason as before: without it, blur near an edge would mix in fabricated zeros and bias
    the result there.

    Where a pixel has no valid neighbors at all within the kernel (only possible with a
    contiguous overburn region far larger than `sigma`), falls back to the unmasked blur there
    rather than dividing by ~0.
    """
    valid_f = valid.astype(np.float64)
    num = gaussian_filter(y * valid_f, sigma=sigma, mode="reflect")
    den = gaussian_filter(valid_f, sigma=sigma, mode="reflect")
    den_safe = np.where(den > 1e-6, den, 1.0)
    blurred = num / den_safe
    if np.any(den <= 1e-6):
        blurred = np.where(den > 1e-6, blurred, gaussian_filter(y, sigma=sigma, mode="reflect"))
    return blurred


def _flat_weights(overburn: np.ndarray, sigma: float, frac_max: float) -> np.ndarray:
    """Per-pixel reliability weight for one flat frame: 0 at an overburn pixel itself, and 0
    wherever the Gaussian-weighted (same sigma as the blur) local fraction of overburn pixels
    exceeds `frac_max` -- protects against `corr` being computed from a blur that's already
    biased by nearby saturated pixels, even at a pixel that isn't saturated itself."""
    local_frac = gaussian_filter(overburn.astype(np.float64), sigma=sigma, mode="reflect")
    return np.where(overburn | (local_frac > frac_max), 0.0, 1.0)


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
):
    """Stream `frames` (one per entry of `times`) through a per-pixel weighted least-squares
    fit of `corr = a + b*x + c*x**2`.

    `frames` is any iterable of (H, W) arrays, raw decoded (pre-dark-subtraction), one per
    entry of `times`. `dark_model` is anything with `.bias`/`.rate` (H, W) arrays (a
    darkcal.DarkModel or a plain stand-in), evaluated at each frame's own exposure time. The
    keyword parameters are exposed only so a small synthetic test can rescale them to its own
    tiny canvas; production callers should leave them at the defaults.

    Returns `(a, b, c, used_exposures, n_frames, frame_report, reliable)`.
    """
    times = np.asarray(times, dtype=np.float64)
    frames = list(frames)
    n = len(frames)
    assert times.shape[0] == n, (times.shape, n, "times and frames length mismatch")
    assert n >= min_flat_frames, f"need >= {min_flat_frames} flat frames, got {n}"

    S0 = S1 = S2 = S3 = S4 = None
    T0 = T1 = T2 = None
    min_x = max_x = None
    frame_report = []

    for t, raw in zip(times, frames):
        raw = np.asarray(raw, dtype=np.float64)
        overburn = raw >= overburn_hi          # raw decode, before dark subtraction

        dark = 0.0
        if dark_model is not None:
            dark = (dark_model.bias.astype(np.float64)
                     + dark_model.rate.astype(np.float64) * float(t))
        x = raw - dark

        blurred = _masked_blur(x, ~overburn, highpass_sigma_px)
        corr = blurred - x
        w = _flat_weights(overburn, highpass_sigma_px, neighborhood_overburn_frac_max)

        if S0 is None:
            S0 = np.zeros_like(x); S1 = np.zeros_like(x); S2 = np.zeros_like(x)
            S3 = np.zeros_like(x); S4 = np.zeros_like(x)
            T0 = np.zeros_like(x); T1 = np.zeros_like(x); T2 = np.zeros_like(x)
            min_x = np.full_like(x, np.inf)
            max_x = np.full_like(x, -np.inf)

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
        min_x = np.where(has_w, np.minimum(min_x, x), min_x)
        max_x = np.where(has_w, np.maximum(max_x, x), max_x)

        frame_report.append({
            "t": float(t),
            "n_overburn": int(overburn.sum()),
            "mean_weight": float(w.mean()),
        })

    # Per-pixel weighted OLS for corr = a + b*x + c*x^2: a batched 3x3 closed-form solve, one
    # (symmetric) moment matrix and rhs vector per pixel -- same streamed-sums idea darkcal.py
    # uses for bias + rate*t, generalized to 3 parameters and a per-pixel x.
    H, W_ = S0.shape
    M = np.empty((H, W_, 3, 3), dtype=np.float64)
    M[..., 0, 0] = S0; M[..., 0, 1] = S1; M[..., 0, 2] = S2
    M[..., 1, 0] = S1; M[..., 1, 1] = S2; M[..., 1, 2] = S3
    M[..., 2, 0] = S2; M[..., 2, 1] = S3; M[..., 2, 2] = S4
    V = np.stack([T0, T1, T2], axis=-1)

    det = np.linalg.det(M)
    nonsingular = np.abs(det) > 1e-12
    M_safe = np.where(nonsingular[..., None, None], M, np.eye(3))
    # Explicit (N, 3, 3) / (N, 3, 1) batch shape -- np.linalg.solve's gufunc broadcasting is
    # ambiguous about which trailing axis of a 3-D rhs is the batch dimension vs. the vector,
    # so collapse (H, W) into one batch axis rather than rely on it being inferred.
    coeffs = np.linalg.solve(
        M_safe.reshape(-1, 3, 3), V.reshape(-1, 3, 1)
    ).reshape(H, W_, 3)
    a_fit, b_fit, c_fit = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2]

    reliable = nonsingular & ((max_x - min_x) > min_reliable_range)
    a = np.where(reliable, a_fit, 0.0)
    b = np.where(reliable, b_fit, 0.0)
    c = np.where(reliable, c_fit, 0.0)

    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c)), (
        "flat model fit produced non-finite values -- check the flats/dark model")

    used_exposures = np.asarray(sorted(set(np.round(times, 9))), dtype=np.float64)
    return (
        a.astype(np.float32), b.astype(np.float32), c.astype(np.float32),
        used_exposures, n, frame_report, reliable,
    )


# --------------------------------------------------------------------------- #
#  Decode + orchestrate                                                       #
# --------------------------------------------------------------------------- #
def _flat_files(flat_dir) -> list:
    flat_dir = Path(flat_dir)
    return sorted(list(flat_dir.glob("*.NEF")) + list(flat_dir.glob("*.nef")))


def fit_flat_model(flat_dir, dark_model=None) -> FlatModel:
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
        times, _decoded_frames(), dark_model=dark_model)
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


def run(flat_dir, dark_model=None, out_pkl: Path | None = None) -> FlatModel:
    """Fit, report, optionally save."""
    model = fit_flat_model(flat_dir, dark_model=dark_model)
    print_report(model)
    if out_pkl is not None:
        save(model, out_pkl)
    return model
