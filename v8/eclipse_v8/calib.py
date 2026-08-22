"""Recover the camera response curve once for the whole bracket (Debevec & Malik).

Replaces v2's per-pair exponent chain.  The model is

    ln f(v_ik)  =  ln L_i  +  ln t_k  +  ln c_k

with `f` the (unknown) map from a stored value back to accumulated light, `L_i` the
brightness of sample pixel `i`, `t_k` the reported shutter time and `c_k` a correction to
it.  Linear in the unknowns, so a least-squares solve over a few thousand sample pixels
seen across ~15 exposures recovers `f` at 64 knots plus one `ln L` per pixel.

**`ln c_k` is not an unknown of that solve.**  `ln t_k + ln c_k` appears only as a sum, so
a free `c_k` makes the reported shutter times carry no information and the curve shape
trades against the exposure ratios at identical residual (Grossberg & Nayar's
response/exposure-ratio ambiguity).  `lsqr` does not fail loudly on it — it returns
`istop=3` and plausible-looking numbers with corrections wrong by >1000%.  The corrections
are instead refined by alternation (`_refine_step`), each subproblem well-posed.

The whole solve is numpy/scipy only: no torch, no GPU.  Only `gather_samples` needs the
GPU, and it is separable, which is what lets the synthetic test in
`v8/test_calib_synthetic.py` exercise the numerics in ~10 s on CPU.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import tqdm
from scipy.sparse.linalg import lsqr

from eclipse_v8.utils import compute_weighted_average, moon_median
from eclipse_v8.warp import build_chains, sample_at_ref_points, warp_to_ref

# --- constants; §3.4 step 1d of PHOTOMETRY_SPEC.md ------------------------------------
FIT_VALUE_LO = 0.10          # measured, not chosen: below ~0.10 an 8-bit JPEG carries no
FIT_VALUE_HI = 0.98          # usable brightness information (see the module docstring's spec)
N_KNOTS = 64
SMOOTHNESS_LAMBDA = 200.0
SMOOTHNESS_FLOOR = 0.05      # the `+0.05` in s(z): lets the prior govern the unobserved ends
GAUGE_WEIGHT = 1e4
N_RADIAL_BINS = 48
SAMPLES_PER_BIN = 220
MIN_EXPOSURES_PER_SAMPLE = 2
N_EXPOSURE_REFINE = 6
MAX_LN_CORRECTION = math.log(1.02)

MOON_SAMPLE_MARGIN_PX = 8.0  # samples must clear the reference moon by this much
SAMPLE_COVER_THRESH = 0.999  # a sample pixel must be inside every exposure's warped frame
SAMPLE_SEED = 20260806       # sampling is random but must be reproducible run to run

LSQR_ITER_LIM = 8000
LSQR_TOL = 1e-12

N_SIGMA_BANDS = 40           # value bands over which the calibration's own error is measured
MIN_SAMPLES_PER_SIGMA_BAND = 40
EXTRAP_SIGMA_FACTOR = 4.0    # outside the fitted range, do not pretend extrapolation is measurement
SIGMA_FLOOR = 0.002
DENSE_LUT_POINTS = 2048


# --------------------------------------------------------------------------- #
#  Weights                                                                    #
# --------------------------------------------------------------------------- #
def hat_weight(v, lo: float = FIT_VALUE_LO, hi: float = FIT_VALUE_HI):
    """Triangular hat over [lo, hi], peak 1 at the midpoint, exactly 0 outside.

    This selects which samples constrain the calibration — it is NOT the merge weight.
    Preferring well-exposed pixels when fitting a response curve is correct and standard;
    the thing this spec replaces is using a guessed bell as a *merge* weight.
    """
    v = np.asarray(v, dtype=np.float64)
    mid = 0.5 * (lo + hi)
    w = np.where(v <= mid, (v - lo) / (mid - lo), (hi - v) / (hi - mid))
    return np.clip(w, 0.0, None)


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class CalibResult:
    """Everything the merge and the diagnostics need. Plain numpy — picklable, no torch."""

    knots: np.ndarray                  # (K,) stored values of the curve knots, [0,1]
    g: np.ndarray                      # (K,) ln f at the knots
    exposures: np.ndarray              # (n_exp,) reported shutter times, ascending
    ln_c: np.ndarray                   # (n_exp,) log corrections to those shutter times
    lnE: np.ndarray                    # (n_kept,) log brightness per retained sample pixel
    V: np.ndarray                      # (n_exp, M) stored values gathered per sample
    valid: np.ndarray                  # (n_exp, M) sample was inside frame + outside moon
    used: np.ndarray                   # (n_exp, M) sample actually constrained the fit
    col: np.ndarray                    # (M,) index into lnE, or -1 if the sample was dropped
    resid: np.ndarray                  # (n_exp, M) ln-space residual where `used`, else nan
    istop: int                         # lsqr stopping condition; 3 means ill-conditioned
    fit_lo: float = FIT_VALUE_LO
    fit_hi: float = FIT_VALUE_HI
    n_exposure_refine: int = N_EXPOSURE_REFINE
    refine_history: list = field(default_factory=list)   # per-round max |step| on ln c
    sample_ij: np.ndarray | None = None                  # (M, 2) reference-grid coords, for plots
    notes: dict = field(default_factory=dict)

    @property
    def corrections(self) -> np.ndarray:
        """c_k — multiply the reported shutter time by this to get the effective one."""
        return np.exp(self.ln_c)

    def correction_by_exposure(self) -> dict:
        return {float(t): float(c) for t, c in zip(self.exposures, self.corrections)}

    def ln_f(self, v):
        """ln f at arbitrary stored values, by the same linear interpolation the fit used."""
        return np.interp(np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0), self.knots, self.g)


# --------------------------------------------------------------------------- #
#  1a. Gather samples                                                         #
# --------------------------------------------------------------------------- #
def _log_radial_sample_points(covered_np, moon_ref, n_bins, per_bin, seed):
    """Sample pixel coordinates stratified in log radius from the moon centre.

    Log radius, not area: the corona spans ~1e4 in brightness over the frame, so a pixel
    near the limb is only ever well exposed in the shortest frames and one at the corner
    only in the longest.  Uniform-over-area sampling piles almost every sample into one
    part of the response curve and leaves the rest unconstrained.
    """
    H, W = covered_np.shape
    mi, mj, moon_r = moon_ref
    r_in = float(moon_r) + MOON_SAMPLE_MARGIN_PX
    rows = np.any(covered_np, axis=1)
    cols = np.any(covered_np, axis=0)
    assert rows.any() and cols.any(), "no pixel is covered by every exposure"
    r_lo_i, r_hi_i = np.where(rows)[0][[0, -1]]
    c_lo_i, c_hi_i = np.where(cols)[0][[0, -1]]
    r_out = max(
        math.hypot(r_lo_i - mi, c_lo_i - mj), math.hypot(r_lo_i - mi, c_hi_i - mj),
        math.hypot(r_hi_i - mi, c_lo_i - mj), math.hypot(r_hi_i - mi, c_hi_i - mj),
    )
    assert r_out > r_in, (r_in, r_out)

    rng = np.random.default_rng(seed)
    edges = np.exp(np.linspace(math.log(r_in), math.log(r_out), n_bins + 1))
    ii_list, jj_list = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        got_i, got_j = [], []
        # Rejection sampling in (log r, theta): most draws land in the covered region, and
        # the few that do not are cheap to discard. 24 attempts per wanted sample is ample
        # for the inner bins and lets the outermost bins — mostly outside the frame — run
        # dry gracefully instead of looping forever.
        n_try = per_bin * 24
        r = np.exp(rng.uniform(math.log(lo), math.log(hi), n_try))
        th = rng.uniform(0.0, 2.0 * math.pi, n_try)
        pi = np.rint(mi + r * np.sin(th)).astype(np.int64)
        pj = np.rint(mj + r * np.cos(th)).astype(np.int64)
        ok = (pi >= 0) & (pi < H) & (pj >= 0) & (pj < W)
        pi, pj = pi[ok], pj[ok]
        if pi.size:
            ok2 = covered_np[pi, pj]
            pi, pj = pi[ok2], pj[ok2]
        if pi.size:
            keyed = np.unique(np.stack([pi, pj], axis=1), axis=0)
            take = min(per_bin, keyed.shape[0])
            sel = rng.choice(keyed.shape[0], size=take, replace=False)
            got_i, got_j = keyed[sel, 0], keyed[sel, 1]
        if len(got_i):
            ii_list.append(np.asarray(got_i))
            jj_list.append(np.asarray(got_j))
    assert ii_list, "log-radial sampling produced no points"
    return np.concatenate(ii_list), np.concatenate(jj_list)


def gather_samples(
    exposure_groups,
    opt_results,
    exposure_times_sorted,
    cross_reg,
    t_ref,
    device,
    *,
    n_radial_bins: int = N_RADIAL_BINS,
    samples_per_bin: int = SAMPLES_PER_BIN,
    seed: int = SAMPLE_SEED,
):
    """Stored values of the same sky pixels across every exposure.

    Runs after registration (it needs corresponding pixels), which looks circular because
    cross-exposure alignment used the fitted per-pair exponents.  It is not: those exponents
    are used *for alignment only* — Fourier alignment cares about structure, not absolute
    scale — and stop determining any brightness from here on.

    Returns `(V, valid, exposures, sample_ij)`:
        V         (n_exp, M) float64 stored values, encoded, in [0,1]
        valid     (n_exp, M) bool — inside that exposure's stack coverage and outside its moon
        exposures (n_exp,)   shutter times, ascending, that had a usable chain
        sample_ij (M, 2)     reference-grid coordinates of the samples
    """
    available = {
        exp for exp in exposure_times_sorted
        if exp in opt_results and len(exposure_groups.get(exp, ())) >= 2
    }
    exposures_with_chain, chains = build_chains(
        exposure_times_sorted, available, cross_reg, t_ref
    )
    assert len(exposures_with_chain) >= 3, exposures_with_chain

    ref_group = exposure_groups[t_ref]
    H_ref, W_ref = int(ref_group[0].height), int(ref_group[0].width)
    moon_ref = moon_median(ref_group)

    # Coverage common to all exposures, from a constant image per exposure. Geometry only —
    # no frame is decoded here, and per-sample stack coverage is checked again below.
    covered = None
    ones = torch.ones((H_ref, W_ref), device=device, dtype=torch.float32)
    for exp in tqdm.tqdm(exposures_with_chain, desc="calib coverage"):
        w = warp_to_ref(ones, chains[exp], H_ref, W_ref, device)
        covered = w if covered is None else torch.minimum(covered, w)
    covered_np = (covered >= SAMPLE_COVER_THRESH).cpu().numpy()
    del ones, covered
    torch.cuda.empty_cache()

    si, sj = _log_radial_sample_points(
        covered_np, moon_ref, n_radial_bins, samples_per_bin, seed
    )
    M = si.size
    print(f"calib: {M} sample pixels in {n_radial_bins} log-radial bins, "
          f"moon_r={moon_ref[2]:.2f}, covered fraction {covered_np.mean():.3f}")

    i_t = torch.from_numpy(si.astype(np.float32)).to(device)
    j_t = torch.from_numpy(sj.astype(np.float32)).to(device)

    n_exp = len(exposures_with_chain)
    V = np.zeros((n_exp, M), dtype=np.float64)
    valid = np.zeros((n_exp, M), dtype=bool)
    for k, exp in enumerate(tqdm.tqdm(exposures_with_chain, desc="calib sample")):
        group = exposure_groups[exp]
        abs_xy = torch.from_numpy(opt_results[exp]["abs_xy"]).to(device)
        abs_angle_t = torch.from_numpy(opt_results[exp]["abs_angle_t"]).to(device)
        # keep_warped=False: one exposure's worth of full-resolution frames is already
        # ~100 MB each, and nothing here wants the per-frame stack.
        avg_img, avg_mask, _ = compute_weighted_average(
            group, abs_xy, abs_angle_t, device, keep_warped=False
        )
        v_k = sample_at_ref_points(avg_img, chains[exp], i_t, j_t, H_ref, W_ref, device)
        m_k = sample_at_ref_points(avg_mask, chains[exp], i_t, j_t, H_ref, W_ref, device)
        V[k] = v_k.double().cpu().numpy()
        valid[k] = (m_k >= SAMPLE_COVER_THRESH).cpu().numpy()
        del avg_img, avg_mask, v_k, m_k
        torch.cuda.empty_cache()

    sample_ij = np.stack([si, sj], axis=1)
    return V, valid, np.asarray(exposures_with_chain, dtype=np.float64), sample_ij


# --------------------------------------------------------------------------- #
#  1b. Solve                                                                  #
# --------------------------------------------------------------------------- #
def solve_response(
    V,
    valid,
    ln_t_eff,
    *,
    n_knots: int = N_KNOTS,
    lam: float = SMOOTHNESS_LAMBDA,
    gauge_weight: float = GAUGE_WEIGHT,
    lo: float = FIT_VALUE_LO,
    hi: float = FIT_VALUE_HI,
    min_exposures: int = MIN_EXPOSURES_PER_SAMPLE,
    iter_lim: int = LSQR_ITER_LIM,
):
    """One least-squares solve for the curve knots and the per-pixel brightnesses.

    `ln_t_eff` is `ln t_k + ln c_k` — a *fixed* offset, never an unknown (see module docstring).

    Rows, one per usable (pixel, exposure):
        w · [ (1-a)·g[z] + a·g[z+1] − lnE_i ]  =  w · ln_t_eff_k
    plus a second-difference smoothness prior scaled by `s(z) = hat(knot z) + 0.05`, plus one
    gauge row killing the last global degeneracy (`g += c`, `lnE += c`).

    Grayscale values are the mean of three 8-bit channels, so they do not sit on the
    256-integer grid — hence the interpolation between bracketing knots rather than a
    lookup.  Still linear in the unknowns.
    """
    V = np.asarray(V, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    ln_t_eff = np.asarray(ln_t_eff, dtype=np.float64)
    n_exp, M = V.shape
    assert ln_t_eff.shape == (n_exp,), (ln_t_eff.shape, n_exp)
    K = int(n_knots)
    knots = np.linspace(0.0, 1.0, K)

    W = hat_weight(V, lo, hi) * valid
    used = W > 0
    # A pixel seen in one exposure constrains nothing: its lnE absorbs the whole row.
    counts = used.sum(axis=0)
    keep = counts >= min_exposures
    used = used & keep[None, :]
    col = np.full(M, -1, dtype=np.int64)
    n_kept = int(keep.sum())
    assert n_kept > 0, "no sample pixel is usable in >= 2 exposures"
    col[keep] = np.arange(n_kept)

    ke, me = np.nonzero(used)
    w = W[ke, me]
    v = np.clip(V[ke, me], 0.0, 1.0)
    pos = v * (K - 1)
    z = np.clip(np.floor(pos).astype(np.int64), 0, K - 2)
    alpha = pos - z

    n_data = ke.size
    assert n_data > 0, "no usable (pixel, exposure) rows"
    rows_d = np.arange(n_data)
    zz = np.arange(1, K - 1)
    s = lam * (hat_weight(knots[zz], lo, hi) + SMOOTHNESS_FLOOR)
    rows_s = n_data + np.arange(K - 2)
    row_gauge = n_data + (K - 2)

    rows = np.concatenate([
        rows_d, rows_d, rows_d,
        rows_s, rows_s, rows_s,
        np.array([row_gauge]),
    ])
    cols = np.concatenate([
        z, z + 1, K + col[me],
        zz - 1, zz, zz + 1,
        np.array([K // 2]),
    ])
    vals = np.concatenate([
        w * (1.0 - alpha), w * alpha, -w,
        s, -2.0 * s, s,
        np.array([float(gauge_weight)]),
    ])
    n_rows = row_gauge + 1
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, K + n_kept)).tocsr()

    b = np.zeros(n_rows, dtype=np.float64)
    b[:n_data] = w * ln_t_eff[ke]

    out = lsqr(A, b, atol=LSQR_TOL, btol=LSQR_TOL, iter_lim=iter_lim)
    x, istop = out[0], int(out[1])
    g = x[:K]
    lnE = x[K:]
    return g, lnE, istop, used, col, W


def _residual(V, g, knots, lnE, col, ln_t_eff, used):
    """resid = g(v) − lnE_i − (ln t + d)_k, nan where the sample did not constrain the fit."""
    resid = np.full(V.shape, np.nan, dtype=np.float64)
    ke, me = np.nonzero(used)
    gv = np.interp(np.clip(V[ke, me], 0.0, 1.0), knots, g)
    resid[ke, me] = gv - lnE[col[me]] - ln_t_eff[ke]
    return resid


# --------------------------------------------------------------------------- #
#  1c. Alternate for the exposure-time corrections                            #
# --------------------------------------------------------------------------- #
def calibrate(
    V,
    valid,
    exposures,
    *,
    n_exposure_refine: int = N_EXPOSURE_REFINE,
    n_knots: int = N_KNOTS,
    lam: float = SMOOTHNESS_LAMBDA,
    gauge_weight: float = GAUGE_WEIGHT,
    lo: float = FIT_VALUE_LO,
    hi: float = FIT_VALUE_HI,
    min_exposures: int = MIN_EXPOSURES_PER_SAMPLE,
    max_ln_correction: float = MAX_LN_CORRECTION,
    sample_ij=None,
    verbose: bool = True,
) -> CalibResult:
    """Fit the response curve, alternating with the per-exposure shutter corrections.

    Each subproblem is well-posed: the curve solve sees the exposure times as fixed, and the
    correction step is a plain weighted mean of the leftover discrepancy per exposure — a
    systematic offset for one exposure *is* its timing error.  `n_exposure_refine=0` is the
    honest nominal-times baseline.
    """
    V = np.asarray(V, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    exposures = np.asarray(exposures, dtype=np.float64)
    n_exp = V.shape[0]
    assert exposures.shape == (n_exp,), (exposures.shape, n_exp)
    knots = np.linspace(0.0, 1.0, int(n_knots))
    ln_t = np.log(exposures)
    d = np.zeros(n_exp, dtype=np.float64)
    history = []

    g = lnE = used = col = W = None
    istop = -1
    for it in range(int(n_exposure_refine) + 1):
        g, lnE, istop, used, col, W = solve_response(
            V, valid, ln_t + d, n_knots=n_knots, lam=lam, gauge_weight=gauge_weight,
            lo=lo, hi=hi, min_exposures=min_exposures,
        )
        assert istop != 3, (
            "lsqr reported istop=3 (ill-conditioned). The exposure corrections have almost "
            "certainly been folded into the linear system — they must be alternated, not "
            "solved jointly; see PHOTOMETRY_SPEC.md §3.5 trap 1."
        )
        resid = _residual(V, g, knots, lnE, col, ln_t + d, used)
        if it == int(n_exposure_refine):
            break
        num = np.nansum(np.where(used, W * np.nan_to_num(resid), 0.0), axis=1)
        den = np.where(used, W, 0.0).sum(axis=1)
        step = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
        d = d + step
        d = d - d.mean()
        d = np.clip(d, -max_ln_correction, max_ln_correction)
        history.append(float(np.max(np.abs(step))))
        if verbose:
            print(f"  refine {it + 1}: max|step| = {history[-1]:.5f}, "
                  f"corrections {np.exp(d).min():.4f}..{np.exp(d).max():.4f}")

    result = CalibResult(
        knots=knots, g=g, exposures=exposures, ln_c=d, lnE=lnE,
        V=V, valid=valid, used=used, col=col, resid=resid, istop=istop,
        fit_lo=lo, fit_hi=hi, n_exposure_refine=int(n_exposure_refine),
        refine_history=history,
        sample_ij=None if sample_ij is None else np.asarray(sample_ij),
    )
    n_pinned = int(np.sum(np.abs(d) >= max_ln_correction - 1e-9))
    result.notes["n_corrections_pinned_at_clamp"] = n_pinned
    return result


# --------------------------------------------------------------------------- #
#  1e. Derived outputs                                                        #
# --------------------------------------------------------------------------- #
def response_lut(result: CalibResult, n: int = DENSE_LUT_POINTS):
    """(v grid, f(v)) densely, from the knots. Exact — the fit is piecewise linear in ln f."""
    v = np.linspace(0.0, 1.0, int(n))
    return v, np.exp(result.ln_f(v))


def systematic_sigma(result: CalibResult, n_bands: int = N_SIGMA_BANDS):
    """(v grid, sigma_fraction) — the calibration's own error, measured, not assumed.

    Per value band, `sqrt(mean² + var)` of the ln-space residual: bias and scatter both
    count, and a ln-space residual *is* a fractional error in brightness.  Bands with too
    few samples are interpolated across; outside the fitted range the edge value is
    multiplied by `EXTRAP_SIGMA_FACTOR` rather than pretending extrapolation is measurement.
    """
    edges = np.linspace(0.0, 1.0, int(n_bands) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    used = result.used
    v = result.V[used]
    r = result.resid[used]
    sigma = np.full(int(n_bands), np.nan, dtype=np.float64)
    for b in range(int(n_bands)):
        # Half-open bands, closed at the very top so v == 1.0 lands somewhere.
        upper = (v <= edges[b + 1]) if b + 1 == n_bands else (v < edges[b + 1])
        sel = (v >= edges[b]) & upper
        if int(sel.sum()) < MIN_SAMPLES_PER_SIGMA_BAND:
            continue
        rb = r[sel]
        sigma[b] = math.sqrt(float(np.mean(rb) ** 2 + np.var(rb)))

    have = np.isfinite(sigma)
    assert have.any(), "no value band had enough samples to measure the calibration error"
    sigma[~have] = np.interp(centres[~have], centres[have], sigma[have])

    # Outside the fitted range nothing was measured; the curve there is extrapolation.
    lo_i, hi_i = centres < result.fit_lo, centres > result.fit_hi
    first, last = np.argmax(have), len(have) - 1 - np.argmax(have[::-1])
    sigma[lo_i] = sigma[first] * EXTRAP_SIGMA_FACTOR
    sigma[hi_i] = sigma[last] * EXTRAP_SIGMA_FACTOR
    sigma = np.maximum(sigma, SIGMA_FLOOR)

    # Pin the endpoints so an interpolating consumer never extrapolates off the ends.
    grid = np.concatenate([[0.0], centres, [1.0]])
    vals = np.concatenate([[sigma[0]], sigma, [sigma[-1]]])
    return grid, vals


def effective_gamma(result: CalibResult, lo: float = 0.25, hi: float = 0.85) -> float:
    """Log-log least-squares slope of f over [lo, hi]. Display scaling only, never radiometry.

    Deliberately not a secant through v=1: the extrapolated toe does not reach zero
    (f(0) ~ 0.1 on the JPEG data), which inflates a secant by more than 50%.
    """
    v, f = response_lut(result)
    sel = (v >= lo) & (v <= hi) & (f > 0)
    assert sel.sum() >= 8, (lo, hi, int(sel.sum()))
    x = np.log(v[sel])
    y = np.log(f[sel])
    return float(np.polyfit(x, y, 1)[0])


def monotonicity_report(result: CalibResult) -> dict:
    """Counts of non-monotone steps in the recovered curve, overall and inside the fit range."""
    dg = np.diff(result.g)
    k_lo = int(np.searchsorted(result.knots, result.fit_lo))
    k_hi = int(np.searchsorted(result.knots, result.fit_hi))
    inside = dg[max(k_lo - 1, 0):max(k_hi, 1)]
    return {
        "n_steps": int(dg.size),
        "n_non_monotone": int(np.sum(dg <= 0)),
        "n_non_monotone_in_fit_range": int(np.sum(inside <= 0)),
        "min_step": float(dg.min()),
    }


def residual_report(result: CalibResult, lo: float = 0.25, hi: float = 0.85) -> dict:
    """Overall and core-band residual statistics, as fractions of brightness."""
    used = result.used
    v = result.V[used]
    r = result.resid[used]
    core = (v >= lo) & (v <= hi)
    return {
        "n_used": int(used.sum()),
        "rms_all": float(np.sqrt(np.mean(r ** 2))),
        "bias_all": float(np.mean(r)),
        "rms_core": float(np.sqrt(np.mean(r[core] ** 2))) if core.any() else float("nan"),
        "bias_core": float(np.mean(r[core])) if core.any() else float("nan"),
        "core_lo": lo,
        "core_hi": hi,
    }


def local_exponent(result: CalibResult, v0: float) -> float:
    """d ln f / d ln v at one stored value — constant iff the response is a power law."""
    v, f = response_lut(result)
    sel = (v > max(v0 - 0.05, 1e-3)) & (v < min(v0 + 0.05, 1.0)) & (f > 0)
    assert sel.sum() >= 4, v0
    return float(np.polyfit(np.log(v[sel]), np.log(f[sel]), 1)[0])


# --------------------------------------------------------------------------- #
#  Reporting / persistence                                                    #
# --------------------------------------------------------------------------- #
def print_report(result: CalibResult) -> None:
    res = residual_report(result)
    mono = monotonicity_report(result)
    print(f"calib: {len(result.exposures)} exposures, {result.V.shape[1]} samples, "
          f"{res['n_used']} usable (pixel, exposure) rows, {result.lnE.size} pixels retained")
    print(f"calib: lsqr istop = {result.istop} (3 would mean ill-conditioned)")
    print(f"calib: monotonicity — {mono['n_non_monotone']} non-monotone steps "
          f"({mono['n_non_monotone_in_fit_range']} inside the fit range), "
          f"min step {mono['min_step']:+.5f}")
    print(f"calib: residual rms {res['rms_all']:.4%} overall, "
          f"{res['rms_core']:.4%} in the core band "
          f"v in [{res['core_lo']}, {res['core_hi']}] (bias {res['bias_core']:+.4%})")
    print(f"calib: effective gamma {effective_gamma(result):.3f}; local exponent "
          f"{local_exponent(result, 0.20):.2f} at v=0.20 -> "
          f"{local_exponent(result, 0.80):.2f} at v=0.80")
    print("calib: per-exposure corrections (reported shutter time x c_k):")
    for t, c in zip(result.exposures, result.corrections):
        pinned = " <-- AT CLAMP" if abs(math.log(c)) >= MAX_LN_CORRECTION - 1e-9 else ""
        print(f"        t = {t:<12.6f}  c = {c:.4f}{pinned}")
    if result.notes.get("n_corrections_pinned_at_clamp"):
        print("calib: WARNING — corrections pinned at the +-25% clamp; if istop is also 3 the "
              "alternation has been implemented as a joint solve (spec §3.5 trap 1).")
    grid, sigma = systematic_sigma(result)
    show = np.linspace(0, len(grid) - 1, 12).astype(int)
    print("calib: systematic sigma (calibration's own error, fraction of brightness):")
    print("        v      " + " ".join(f"{grid[i]:6.2f}" for i in show))
    print("        sigma  " + " ".join(f"{sigma[i] * 100:5.1f}%" for i in show))


def save(result: CalibResult, path: Path) -> Path:
    path = Path(path)
    with open(path, "wb") as fd:
        pickle.dump(result, fd)
    print(f"Saved {path} (calibration).")
    return path


def load(path: Path) -> CalibResult:
    with open(Path(path), "rb") as fd:
        return pickle.load(fd)


def run(
    exposure_groups,
    opt_results,
    exposure_times_sorted,
    cross_reg,
    t_ref,
    device,
    *,
    n_exposure_refine: int = N_EXPOSURE_REFINE,
    out_pkl: Path | None = None,
) -> CalibResult:
    """Gather samples on the GPU, fit, report, optionally save."""
    V, valid, exposures, sample_ij = gather_samples(
        exposure_groups, opt_results, exposure_times_sorted, cross_reg, t_ref, device
    )
    result = calibrate(
        V, valid, exposures, n_exposure_refine=n_exposure_refine, sample_ij=sample_ij
    )
    print_report(result)
    if out_pkl is not None:
        save(result, out_pkl)
    return result
