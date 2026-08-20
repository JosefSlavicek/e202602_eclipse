"""Per-pixel flat field: pixel-to-pixel sensitivity and dust-shadow irregularities on the
chip -- deliberately NOT the smooth optical vignetting -- measured from stacks of flat
frames spanning several exposure times, with the correction allowed to depend on the light
frame's own pixel value (a fixed multiplicative map is not enough: a defect's effect on the
recorded value need not be the same fraction of full well at every brightness).

**Grouped by exposure time.** Unlike a single-exposure flat, this needs brightness variation
to fit against, so flats/ is expected to span multiple exposure times. Frames are grouped by
their reported exposure time; a frame is dropped from its group if it has more than
`MAX_OVERBURN_PIXELS` pixels at/above `OVERBURN_HI` (raw decode, before dark subtraction --
same convention as rawprep.py), and a whole group is dropped if fewer than
`MIN_FRAMES_PER_GROUP` frames survive that filter -- too few frames to average down the
per-pixel noise is worse than not using that exposure at all.

**Per group, same high-frequency-only processing as before.** Each surviving frame is dark-
subtracted, then has its own heavily Gaussian-blurred version (`FLAT_HIGHPASS_SIGMA_PX`,
mirror-padded at the border -- see `_remove_smooth_trend`) subtracted from it, keeping only
the deviation from that local smooth trend. This throws away the optics' smooth radial
vignetting falloff (which lives entirely in that trend) and keeps only small-scale structure:
dust shadows, pixel-to-pixel sensitivity variation. The group's frames are then averaged into
one per-pixel "coefficient" and one per-pixel "brightness" (the plain dark-subtracted
average, *not* high-pass filtered -- brightness is how much light that pixel actually
received, the physically meaningful driver of any brightness-dependence).

**Across groups, a per-pixel line.** With >= `MIN_GROUPS_FOR_MODEL` qualifying groups, each
pixel gets its own least-squares fit `coefficient = a(x,y) + b(x,y) * brightness(x,y)` across
the groups it appears in -- the same closed-form streamed-sums OLS solve `darkcal.py` uses
for `bias + rate * t`, just with a per-pixel (not shared) x-value. The regression runs on
*unclipped* per-group coefficients: clipping first would flatten or bias the fitted slope for
any pixel that happened to hit a bound in one group. With fewer qualifying groups this
reduces to the old brightness-independent model: `b = 0` everywhere, `a` = that one group's
(or the mean of however many sub-threshold groups exist, in the degenerate case) coefficient.

**Normalization and clamp.** The fitted model is evaluated at each pixel's own *mean*
brightness across the groups that fed it (not at brightness 0 -- the intercept `a` alone is
an extrapolation to zero light, far outside anything the flats actually measured, and can
come out small or even negative for a pixel with a non-trivial slope), and that evaluated
field's *harmonic* mean is what `a` and `b` are jointly rescaled by, so a light frame at
roughly the calibration data's own brightness divides back to close to 1.0. The clamp to
`[FLAT_CLAMP_LO, FLAT_CLAMP_HI]` (0.95..1.05) is separate and applied only at evaluation time,
to the actual `a + b*value` a light frame produces -- never to `a` or `b` themselves, and
never to the per-group coefficients the regression is fit from. This guards against
extrapolation in the other direction: light frames span a much wider brightness range than
the flats do, so `a + b*value` at a light frame's actual value can land far outside what was
ever fit.

`FlatModel.evaluate(value)` is what `rawprep.apply_corrections` calls with the light frame's
own dark-subtracted value, dividing that clamped result into the corrected signal -- baked
into the same corrected cache `load_gray` reads, so registration and calibration see
flat-corrected values too, not just the final merge. The merge's per-frame weight is
unaffected by any of this: it is measured once from the raw decode, before any correction
runs (see `rawprep.py`).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

OVERBURN_HI = 0.99              # raw decoded fraction counting as saturated (rawprep.py's
                                  # RAW_OVERBURN_HI convention, duplicated to avoid a
                                  # cross-module import for one constant)
MAX_OVERBURN_PIXELS = 100       # drop a flat frame with more saturated pixels than this
MIN_FRAMES_PER_GROUP = 5        # drop a whole exposure-time group with fewer valid frames
MIN_GROUPS_FOR_MODEL = 2        # need at least this many qualifying groups to fit a per-pixel
                                  # line; fewer falls back to a single brightness-independent
                                  # map (b = 0 everywhere)
FLAT_HIGHPASS_SIGMA_PX = 24.0   # Gaussian sigma separating vignetting (smooth, removed)
                                  # from dust/pixel-sensitivity artefacts (kept)
FLAT_CLAMP_LO = 0.95            # a + b*value is clamped to this range at evaluation time: a
FLAT_CLAMP_HI = 1.05            # high-frequency-only correction has no business swinging far
                                  # from 1.0, so anything beyond this is more likely noise, a
                                  # genuine defect, or extrapolation than a real correction


def _remove_smooth_trend(y: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract the spatially-varying part of a heavy Gaussian blur of `y`, isolating
    high-frequency structure from the smooth (vignetting-scale) trend.

    `mode="reflect"` mirror-pads at the border (scipy: "the input is extended by reflecting
    about the edge of the last pixel") instead of the default zero-padding a plain
    convolution would use. Zero-padding would pull the blurred value down near every edge
    purely from mixing in fabricated zeros beyond the frame, which would then show up as a
    bogus bright ring in the high-pass result. Mirroring instead means every output pixel's
    blur is an average of real, reflected frame values -- never a fabricated one -- so the
    blur (and the high-pass residual) stays correct all the way to the border.

    Only the blur's *deviation* from its own mean is subtracted (`blurred - blurred.mean()`),
    not the blur itself, so the frame's overall level survives; only its spatial shape does
    not.
    """
    blurred = gaussian_filter(y, sigma=sigma, mode="reflect")
    return y - (blurred - blurred.mean())


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class FlatModel:
    """Everything `rawprep.apply_corrections` needs. Plain numpy -- picklable, no torch.

    The correction depends on the light frame's own value: `evaluate(value)` computes
    `a + b*value`, clamped to [FLAT_CLAMP_LO, FLAT_CLAMP_HI]. `b` is all zeros when too few
    exposure groups were available to fit a slope (see MIN_GROUPS_FOR_MODEL), which reduces
    exactly to the old brightness-independent model.
    """

    a: np.ndarray              # (H, W) float32, intercept (harmonic-mean normalized)
    b: np.ndarray              # (H, W) float32, slope per unit of dark-subtracted brightness
    exposures: np.ndarray      # (n_groups_used,) float64, exposure times that fed the fit
    n_frames: int              # total valid frames across every group used in the fit
    group_report: list = field(default_factory=list)   # see print_report / fit_flat_model

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        """`a + b*value`, clamped to [FLAT_CLAMP_LO, FLAT_CLAMP_HI]. `value` is the light
        frame's own dark-subtracted signal, same shape as `a`/`b`."""
        coeff = self.a.astype(np.float64) + self.b.astype(np.float64) * value
        return np.clip(coeff, FLAT_CLAMP_LO, FLAT_CLAMP_HI)


# --------------------------------------------------------------------------- #
#  The fit itself — pure arrays, no file IO                                   #
# --------------------------------------------------------------------------- #
def fit_flat_model_from_stack(
    times, frames, dark_model=None, *,
    highpass_sigma_px: float = FLAT_HIGHPASS_SIGMA_PX,
    overburn_hi: float = OVERBURN_HI,
    max_overburn_pixels: int = MAX_OVERBURN_PIXELS,
    min_frames_per_group: int = MIN_FRAMES_PER_GROUP,
    min_groups_for_model: int = MIN_GROUPS_FOR_MODEL,
):
    """Group `frames` by `times`, fit each qualifying group's per-pixel coefficient and
    brightness, then fit a per-pixel line across groups.

    `frames` is any iterable of (H, W) arrays, decoded or synthetic, one per entry of
    `times` -- unlike the old single-exposure version, `times` is expected to take on
    several distinct values. `dark_model` is anything with `.bias`/`.rate` (H, W) arrays (a
    darkcal.DarkModel or a plain stand-in), evaluated at each group's own exposure time.
    The keyword parameters are exposed only so a small synthetic test can rescale them to its
    own tiny canvas / frame counts; production callers should leave them at the defaults.

    Returns `(a, b, used_exposures, total_valid_frames, group_report)` — see `FlatModel` for
    the first four; `group_report` is a list of one dict per exposure time found (used or
    not), each with `t`, `n_total`, `n_valid`, `avg_value`, `used` — exactly what
    `print_report`'s summary table needs.
    """
    times = np.asarray(times, dtype=np.float64)
    frames = list(frames)
    n = len(frames)
    assert times.shape[0] == n, (times.shape, n, "times and frames length mismatch")
    assert n > 0, "no flat frames given"

    by_exposure: dict[float, list] = {}
    for t, frame in zip(times, frames):
        by_exposure.setdefault(round(float(t), 9), []).append(np.asarray(frame, dtype=np.float64))

    group_report = []
    fit_bright, fit_coeff, used_exposures = [], [], []

    for t in sorted(by_exposure):
        raw_frames = by_exposure[t]
        n_total = len(raw_frames)
        valid = [f for f in raw_frames if int(np.sum(f >= overburn_hi)) <= max_overburn_pixels]
        n_valid = len(valid)
        avg_value = float(np.mean([f.mean() for f in valid])) if valid else float("nan")
        used = n_valid >= min_frames_per_group
        group_report.append({
            "t": t, "n_total": n_total, "n_valid": n_valid,
            "avg_value": avg_value, "used": used,
        })
        if not used:
            continue

        dark = None
        if dark_model is not None:
            dark = (dark_model.bias.astype(np.float64) + dark_model.rate.astype(np.float64) * t)

        sum_raw = sum_hp = None
        for f in valid:
            y = f - dark if dark is not None else f
            sum_raw = y if sum_raw is None else sum_raw + y
            y_hp = _remove_smooth_trend(y, highpass_sigma_px)
            sum_hp = y_hp if sum_hp is None else sum_hp + y_hp
        fit_bright.append(sum_raw / n_valid)     # per-pixel brightness, dark-subtracted
        fit_coeff.append(sum_hp / n_valid)       # per-pixel coefficient, UNNORMALIZED
        used_exposures.append(t)

    n_groups = len(fit_bright)
    assert n_groups >= 1, (
        f"no exposure-time group had >= {min_frames_per_group} valid flat frames -- "
        f"cannot fit any flat model at all")
    total_valid_frames = sum(g["n_valid"] for g in group_report if g["used"])

    if n_groups >= min_groups_for_model:
        # Per-pixel OLS, closed-form (same shape as darkcal.fit_dark_model_from_stack's
        # bias + rate*t solve, except the x-value here is per-pixel, not shared).
        S0 = float(n_groups)
        S1 = S2 = Sy = Sty = None
        for bright, coeff in zip(fit_bright, fit_coeff):
            S1 = bright.copy() if S1 is None else S1 + bright
            S2 = bright ** 2 if S2 is None else S2 + bright ** 2
            Sy = coeff.copy() if Sy is None else Sy + coeff
            Sty = (bright * coeff) if Sty is None else Sty + bright * coeff
        det = S0 * S2 - S1 ** 2
        det_safe = np.where(np.abs(det) > 1e-12, det, 1e-12)
        a = (S2 * Sy - S1 * Sty) / det_safe
        b = (S0 * Sty - S1 * Sy) / det_safe
        ref_brightness = S1 / n_groups   # this pixel's own mean brightness across groups
    else:
        # Fallback: not enough qualifying groups to fit a slope. Reduces to the old
        # brightness-independent model (b = 0); with n_groups==1 `a` is exactly that one
        # group's unnormalized coefficient.
        a = np.mean(np.stack(fit_coeff, axis=0), axis=0)
        b = np.zeros_like(a)
        ref_brightness = np.mean(np.stack(fit_bright, axis=0), axis=0)

    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (
        "flat model fit produced non-finite values -- check the flats/dark model")

    # Normalize by the harmonic mean of the model evaluated at each pixel's OWN mean
    # brightness across the calibration data -- not at brightness 0. The intercept `a` alone
    # is an extrapolation to zero light, far outside anything the flats actually measured,
    # and can come out small or negative for a pixel with a non-trivial slope; anchoring the
    # normalization there is exactly the kind of extrapolation the final evaluation clamp
    # exists to guard against, and it shouldn't happen silently inside the fit itself too.
    predicted_at_ref = a + b * ref_brightness
    assert predicted_at_ref.min() > 0, (
        f"flat model evaluated at its own calibration brightness has non-positive pixels "
        f"(min {predicted_at_ref.min():.6g}) -- check the flats were bright enough / the "
        f"dark model")
    harmonic_mean = 1.0 / np.mean(1.0 / predicted_at_ref)
    a = a / harmonic_mean
    b = b / harmonic_mean

    return (
        a.astype(np.float32), b.astype(np.float32),
        np.asarray(used_exposures, dtype=np.float64), total_valid_frames, group_report,
    )


# --------------------------------------------------------------------------- #
#  Decode + orchestrate                                                       #
# --------------------------------------------------------------------------- #
def _flat_files(flat_dir) -> list:
    flat_dir = Path(flat_dir)
    return sorted(list(flat_dir.glob("*.NEF")) + list(flat_dir.glob("*.nef")))


def fit_flat_model(flat_dir, dark_model=None) -> FlatModel:
    """Decode every flat NEF in `flat_dir`, group by exposure time, and fit the per-pixel
    brightness-dependent flat model."""
    from eclipse_v7.inputs import _decode_nef_linear
    from eclipse_v7.stage0 import get_info_from_exif

    files = _flat_files(flat_dir)
    assert files, f"no flat NEFs found in {flat_dir}"

    times = [float(get_info_from_exif(f)[0]) for f in files]

    def _decoded_frames():
        for f in tqdm.tqdm(files, desc="flat decode"):
            yield _decode_nef_linear(f)

    a, b, used_exposures, n_frames, group_report = fit_flat_model_from_stack(
        times, _decoded_frames(), dark_model=dark_model)
    return FlatModel(
        a=a, b=b, exposures=used_exposures, n_frames=n_frames, group_report=group_report,
    )


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(model: FlatModel) -> None:
    print(f"flatcal: {len(model.group_report)} exposure time(s) found among the flats")
    print(f"{'exposure (s)':>14} {'valid images':>13} {'avg pixel value':>16}  used")
    for g in model.group_report:
        print(f"{g['t']:>14.6f} {g['n_valid']:>13d} {g['avg_value']:>16.4f}  "
              f"{'yes' if g['used'] else 'no'}")

    n_used_groups = int(len(model.exposures))
    print(f"flatcal: fit from {n_used_groups} exposure group(s), "
          f"{model.n_frames} valid frames total")
    if n_used_groups < MIN_GROUPS_FOR_MODEL:
        print(f"flatcal: fewer than {MIN_GROUPS_FOR_MODEL} qualifying groups -- fell back to "
              f"a single brightness-independent map (b = 0 everywhere)")

    a, b = model.a, model.b
    print(f"flatcal: intercept (a) mean {a.mean():.4f} (by construction ~1.0 after "
          f"normalizing), min {a.min():.4f}, max {a.max():.4f}")
    print(f"flatcal: slope (b) mean {b.mean():+.5f}, min {b.min():+.5f}, max {b.max():+.5f}")

    used = [g for g in model.group_report if g["used"]]
    if used:
        v_lo = min(g["avg_value"] for g in used)
        v_hi = max(g["avg_value"] for g in used)
        for v, label in ((v_lo, "dimmest"), (v_hi, "brightest")):
            coeff = model.evaluate(np.full_like(a, v, dtype=np.float64))
            n_lo = int(np.sum(coeff <= FLAT_CLAMP_LO + 1e-6))
            n_hi = int(np.sum(coeff >= FLAT_CLAMP_HI - 1e-6))
            print(f"flatcal: at value={v:.4f} ({label} calibration exposure): "
                  f"{n_lo:,} pixel(s) clamped low, {n_hi:,} clamped high "
                  f"({(n_lo + n_hi) / coeff.size:.3%})")


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
