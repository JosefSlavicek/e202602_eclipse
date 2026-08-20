"""Per-pixel flat field: pixel-to-pixel sensitivity and dust-shadow irregularities on the
chip -- deliberately NOT the smooth optical vignetting -- measured from a stack of flat
frames all shot at one exposure time and averaged together.

Every flat frame must share the same reported exposure time -- mixing exposures would fold
each exposure's own dark-current level and any small nonlinearity in on top of the
pixel-to-pixel pattern this is trying to isolate, so it is checked, not assumed. The frames
are dark-subtracted first (same per-pixel bias+rate model as the lights, evaluated at the
flats' own exposure time -- see darkcal.py) so a uniform offset in the flat frames does not
leak into what is supposed to be a purely multiplicative map.

**High-frequency only.** After dark subtraction, each frame has its own heavily
Gaussian-blurred version (`FLAT_HIGHPASS_SIGMA_PX`, mirror-padded at the border -- see
`_remove_smooth_trend`) subtracted from it, keeping only the deviation from that local
smooth trend. This deliberately throws away the optics' smooth radial vignetting falloff
(which lives entirely in that trend) and keeps only small-scale structure: dust shadows,
pixel-to-pixel sensitivity variation. `corrected = raw / vignette` therefore no longer
corrects vignetting at all -- only these high-frequency artifacts.

The averaged, high-pass-filtered flat is then normalized so that applying it to a perfectly
uniform (all-1.0) frame gives back a frame whose average pixel is exactly 1.0. That target
is hit by dividing by the *harmonic* mean of the averaged flat, not the plain mean: for a
correction map `vignette` applied as `corrected = raw / vignette`, the average of
`1 / vignette` over all pixels is 1.0 exactly when `vignette` is scaled by the harmonic mean
of the raw average -- the plain mean only gets close, and only exactly matches when the flat
is already uniform (Jensen's inequality is against you otherwise).

Finally the normalized map is clamped to `[FLAT_CLAMP_LO, FLAT_CLAMP_HI]` (0.95..1.05): this
is a high-frequency-only correction now, so a pixel more than 5% off is far more likely to be
stack noise or a genuine defect than a real sensitivity difference worth fully correcting for.
The clamp runs *after* the harmonic-mean normalization, so if it actually clips any pixels the
"averages back to exactly 1.0" property above becomes only approximate -- see `print_report`
for how many pixels were affected.

Once normalized, `vignette` is what `rawprep.apply_corrections` divides the dark-subtracted
signal by (multiplicative-only, no offset), baked into the same corrected cache `load_gray`
reads — so registration and calibration see flat-corrected values too, not just the final
merge. The merge's per-frame weight is unaffected by this: it is measured once from the raw
decode, before any of this correction runs (see `rawprep.py`).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

MIN_FLAT_FRAMES = 10
FLAT_HIGHPASS_SIGMA_PX = 24.0   # Gaussian sigma separating vignetting (smooth, removed)
                                 # from dust/pixel-sensitivity artefacts (kept)
FLAT_CLAMP_LO = 0.95           # the normalized vignette map is clamped to this range: a
FLAT_CLAMP_HI = 1.05           # high-frequency-only correction has no business swinging far
                                 # from 1.0, so anything beyond this is more likely noise or
                                 # a genuine defect than a real sensitivity difference worth
                                 # fully correcting for


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
    """Everything `rawprep.apply_corrections` needs. Plain numpy -- picklable, no torch."""

    vignette: np.ndarray      # (H, W) float32, relative sensitivity; high-frequency only
                               # (smooth vignetting removed, see module docstring), clamped to
                               # [FLAT_CLAMP_LO, FLAT_CLAMP_HI]; applying it to a perfectly
                               # uniform frame gives an average pixel very close to 1.0
    exposure: float           # the single reported exposure time (s) every flat was shot at
    n_frames: int


# --------------------------------------------------------------------------- #
#  The fit itself — pure arrays, no file IO                                   #
# --------------------------------------------------------------------------- #
def fit_flat_model_from_stack(times, frames, dark_model=None,
                               highpass_sigma_px: float = FLAT_HIGHPASS_SIGMA_PX):
    """Average `frames` (all one exposure time `times[i]`, checked not assumed), dark-subtract
    each one, remove its smooth (vignetting-scale) trend, and normalize into a relative-
    sensitivity map.

    `frames` is any iterable of (H, W) arrays, decoded or synthetic, one per entry of
    `times`. `dark_model` is anything with `.bias`/`.rate` (H, W) arrays (a darkcal.DarkModel
    or a plain stand-in), evaluated at the flats' own exposure time. `highpass_sigma_px` is
    exposed only so a small synthetic test can rescale it to its own tiny canvas; production
    callers should leave it at the default. Returns the normalized `vignette` (H, W) float32
    array, clamped to `[FLAT_CLAMP_LO, FLAT_CLAMP_HI]` — see module docstring for why the
    harmonic mean, not the plain mean, is the right normalizer, why the smooth trend is
    removed at all, and why the result is clamped afterward.
    """
    times = np.asarray(times, dtype=np.float64)
    n = times.shape[0]
    assert n >= MIN_FLAT_FRAMES, f"need >= {MIN_FLAT_FRAMES} flat frames, got {n}"
    t0 = float(times[0])
    assert np.allclose(times, t0, rtol=1e-6, atol=1e-9), (
        f"flat frames are not all the same exposure time: {sorted(set(np.round(times, 9)))} "
        f"-- shoot the whole flats/ bracket at one exposure so averaging them together "
        f"doesn't mix in per-exposure dark current or nonlinearity")

    dark = None
    if dark_model is not None:
        dark = (dark_model.bias.astype(np.float64) + dark_model.rate.astype(np.float64) * t0)

    acc = None
    seen = 0
    for frame in frames:
        y = np.asarray(frame, dtype=np.float64)
        if dark is not None:
            y = y - dark
        y = _remove_smooth_trend(y, highpass_sigma_px)
        acc = y if acc is None else acc + y
        seen += 1
    assert seen == n, (seen, n, "times and frames length mismatch")
    mean_flat = acc / n

    assert mean_flat.min() > 0, (
        f"flat average has non-positive pixels (min {mean_flat.min():.6g}) after dark "
        f"subtraction -- check the flats were bright enough / the dark model")

    harmonic_mean = 1.0 / np.mean(1.0 / mean_flat)
    vignette = mean_flat / harmonic_mean
    vignette = np.clip(vignette, FLAT_CLAMP_LO, FLAT_CLAMP_HI)
    return vignette.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Decode + orchestrate                                                       #
# --------------------------------------------------------------------------- #
def _flat_files(flat_dir) -> list:
    flat_dir = Path(flat_dir)
    return sorted(list(flat_dir.glob("*.NEF")) + list(flat_dir.glob("*.nef")))


def fit_flat_model(flat_dir, dark_model=None) -> FlatModel:
    """Decode every flat NEF in `flat_dir` and fit the per-pixel vignette map."""
    from eclipse_v7.inputs import _decode_nef_linear
    from eclipse_v7.stage0 import get_info_from_exif

    files = _flat_files(flat_dir)
    assert files, f"no flat NEFs found in {flat_dir}"

    times = [float(get_info_from_exif(f)[0]) for f in files]

    def _decoded_frames():
        for f in tqdm.tqdm(files, desc="flat decode"):
            yield _decode_nef_linear(f)

    vignette = fit_flat_model_from_stack(times, _decoded_frames(), dark_model=dark_model)
    return FlatModel(vignette=vignette, exposure=float(times[0]), n_frames=len(files))


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(model: FlatModel) -> None:
    v = model.vignette
    print(f"flatcal: fit from {model.n_frames} frames at {model.exposure:.6f} s")
    print(f"flatcal: vignette mean {v.mean():.4f} (by construction ~1.0 after normalizing), "
          f"min {v.min():.4f}, max {v.max():.4f}")
    print(f"flatcal: strongest correction applied: "
          f"x{1.0 / max(v.min(), 1e-6):.3f} (dimmest pixel) .. "
          f"x{1.0 / max(v.max(), 1e-6):.3f} (brightest pixel)")
    n_lo = int(np.sum(v <= FLAT_CLAMP_LO + 1e-6))
    n_hi = int(np.sum(v >= FLAT_CLAMP_HI - 1e-6))
    print(f"flatcal: clamped to [{FLAT_CLAMP_LO}, {FLAT_CLAMP_HI}] -- "
          f"{n_lo:,} pixel(s) at the low bound, {n_hi:,} at the high bound "
          f"({(n_lo + n_hi) / v.size:.3%} of the frame)")


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
