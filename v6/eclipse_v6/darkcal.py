"""Per-pixel dark-current model: bias + rate against the camera's reported exposure time.

The darks/ bracket is shot at the same exposure times as the lights, several frames each.
Rather than average each exposure time's darks in isolation (matched-exposure subtraction,
noise reduced by 1/sqrt(frames-per-time)) this fits one affine model per pixel

    dark(t) = bias + rate * t

by ordinary least squares over *every* dark frame at once, `t` the plain EXIF-reported
exposure time. Pooling across all exposure times gives a much lower-noise estimate than
any single exposure time's few frames could supply on their own, at the cost of assuming
the model holds across the whole dark-shooting session (sensor temperature stable while
the darks were shot).

**No shutter-time correction is fitted or applied here, on purpose.** The camera's reported
time is known to be somewhat off, especially at short speeds — but this model is only ever
*applied* by subtracting `bias + rate * t` from a frame reporting that same `t`, and that
subtraction is fit and applied with the same EXIF label both times. Whatever the reported
time's absolute error is, it is the same error on a dark and a light shot at the same
labelled speed (same camera, same shutter), so it cancels out of the subtraction — this
model does not need to know the true exposure time to do its job. (True exposure time still
matters elsewhere, for converting a dark-subtracted light frame into physical brightness —
that is `calib.py`'s job, downstream of this one.) Trying to fit a per-exposure timing
correction here instead would trade this cancellation for a fit that is nearly uninformative
at short exposures (`rate * t` is tiny regardless of the timing error, so there is barely a
signal to fit the correction to) and risks corrupting the pooled bias/rate through exactly
those exposures.

Because there is no correction to fit, this can run before anything about the light bracket
is known, and its result — `NefSource.set_dark_model` — should be attached before stage 0,
so registration and the light calibration's own sample-gathering see the same
dark-subtracted values the final merge does, not just the merge.

The fit itself needs no iteration: `bias`/`rate` per pixel is one ordinary least-squares
solve with the same 1-D design matrix (the shared list of `t`) for every pixel, so it
reduces to accumulating three per-pixel sums (`sum(y)`, `sum(t*y)`, `sum(y^2)`) while
streaming the frames, then one closed-form 2x2 solve. No GPU needed; decoding the raw files
is the only real cost, and `fit_dark_model_from_stack` is separable from that, which is what
lets `test_darkcal_synthetic.py` exercise the numerics without any real files.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm

MIN_DARK_FRAMES = 3


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class DarkModel:
    """Everything `NefSource.set_dark_model` needs. Plain numpy — picklable, no torch."""

    bias: np.ndarray          # (H, W) float32, signal at t = 0
    rate: np.ndarray          # (H, W) float32, signal per second (per EXIF-reported second)
    exposures: np.ndarray     # (n_exp,) distinct reported exposure times the fit used
    n_frames: int
    residual_rms: float       # rms of (measured - bias - rate*t) over all fitted frames


# --------------------------------------------------------------------------- #
#  The fit itself — pure arrays, no file IO                                   #
# --------------------------------------------------------------------------- #
def fit_dark_model_from_stack(times, frames):
    """OLS `bias + rate * t` per pixel, streamed so the full (N, H, W) stack never exists.

    `frames` is any iterable of (H, W) arrays, decoded or synthetic, one per entry of
    `times`. Returns (bias, rate, residual_rms) — see `DarkModel` for shapes/meaning.
    """
    times = np.asarray(times, dtype=np.float64)
    n = times.shape[0]
    assert n >= MIN_DARK_FRAMES, f"need >= {MIN_DARK_FRAMES} dark frames, got {n}"
    assert len(set(np.round(times, 12))) >= 2, (
        "all dark frames have the same exposure time; rate is not identifiable")

    S0 = float(n)
    S1 = float(times.sum())
    S2 = float((times ** 2).sum())
    det = S0 * S2 - S1 * S1
    assert abs(det) > 1e-12, "exposure times are degenerate for a linear fit"

    Sy = Sty = Syy = None
    seen = 0
    for t, frame in zip(times, frames):
        y = np.asarray(frame, dtype=np.float64)
        if Sy is None:
            Sy = np.zeros_like(y)
            Sty = np.zeros_like(y)
            Syy = np.zeros_like(y)
        Sy += y
        Sty += t * y
        Syy += y * y
        seen += 1
    assert seen == n, (seen, n, "times and frames length mismatch")

    bias = (S2 * Sy - S1 * Sty) / det
    rate = (S0 * Sty - S1 * Sy) / det
    sse = np.maximum(Syy - bias * Sy - rate * Sty, 0.0)   # >= 0 up to float error
    dof = max(n - 2, 1)
    rms = float(np.sqrt(np.mean(sse / dof)))

    return bias.astype(np.float32), rate.astype(np.float32), rms


# --------------------------------------------------------------------------- #
#  Decode + orchestrate                                                       #
# --------------------------------------------------------------------------- #
def _dark_files(dark_dir) -> list:
    dark_dir = Path(dark_dir)
    return sorted(list(dark_dir.glob("*.NEF")) + list(dark_dir.glob("*.nef")))


def fit_dark_model(dark_dir) -> DarkModel:
    """Decode every dark NEF in `dark_dir` and fit the per-pixel bias/rate model."""
    from eclipse_v6.inputs import _decode_nef_linear
    from eclipse_v6.stage0 import get_info_from_exif

    files = _dark_files(dark_dir)
    assert files, f"no dark NEFs found in {dark_dir}"

    times = []
    for f in files:
        t_nom, _ = get_info_from_exif(f)
        times.append(float(t_nom))
    times = np.asarray(times, dtype=np.float64)

    def _decoded_frames():
        for f in tqdm.tqdm(files, desc="dark decode"):
            yield _decode_nef_linear(f)

    bias, rate, rms = fit_dark_model_from_stack(times, _decoded_frames())
    exposures = np.asarray(sorted(set(np.round(times, 9))), dtype=np.float64)
    return DarkModel(
        bias=bias, rate=rate, exposures=exposures, n_frames=len(files), residual_rms=rms,
    )


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(model: DarkModel) -> None:
    print(f"darkcal: fit from {model.n_frames} frames across {len(model.exposures)} "
          f"reported exposure times ({model.exposures.min():.6f} .. "
          f"{model.exposures.max():.6f} s)")
    print(f"darkcal: bias  mean {model.bias.mean():.5f}, p99 {np.percentile(model.bias, 99):.5f} "
          f"(decoded [0,1] units)")
    print(f"darkcal: rate  mean {model.rate.mean():.6f}/s, "
          f"p99 {np.percentile(model.rate, 99):.6f}/s (decoded [0,1] units per second)")
    hot = model.rate > (model.rate.mean() + 10.0 * model.rate.std())
    print(f"darkcal: {int(hot.sum())} pixels ({hot.mean():.4%}) at >10 sigma dark-current "
          f"rate (hot pixels)")
    print(f"darkcal: residual rms {model.residual_rms:.6f} (decoded [0,1] units, unexplained "
          f"by the linear fit — includes read/shot noise and any nonlinearity)")


def save(model: DarkModel, path: Path) -> Path:
    path = Path(path)
    with open(path, "wb") as fd:
        pickle.dump(model, fd)
    print(f"Saved {path} (dark model).")
    return path


def load(path: Path) -> DarkModel:
    with open(Path(path), "rb") as fd:
        return pickle.load(fd)


def run(dark_dir, out_pkl: Path | None = None) -> DarkModel:
    """Fit, report, optionally save."""
    model = fit_dark_model(dark_dir)
    print_report(model)
    if out_pkl is not None:
        save(model, out_pkl)
    return model
