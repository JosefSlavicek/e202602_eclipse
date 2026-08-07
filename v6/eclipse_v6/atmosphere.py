"""v6: measure the static-atmosphere warp from stars, not from physics. See v6/plan0.md.

Reuses find_stars/starlib.py's ``project``/``solve_plate`` (affine + k1,k2 lens distortion
+ the qx,qy separable atmosphere terms) unmodified -- this module adds only what is
specific to v6:

1. fit one instance of that extended model per calibration frame (a bracket exposure long
   enough to reveal stars), warm-started from the shared base geometry so it is solving
   the easy "which way did it drift" problem, not blind rediscovery;
2. track how the fitted *map* -- not the raw coefficients, see "the degeneracy that
   doesn't need resolving" in plan0.md -- drifts linearly over the calibration frames;
3. apply the drift, evaluated at an arbitrary frame's own timestamp, as a resampling
   correction.

No Bennett/Atmos physics lives here; ``make_data_atmosphere.py``'s physics chain is used
only by the closed-loop validation, to build a scene with a *known* warp to check against.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))          # v6/eclipse_v6
ROOT = os.path.dirname(os.path.dirname(HERE))               # -> src/e202602_eclipse
_FIND_STARS = os.path.join(ROOT, "find_stars")
if _FIND_STARS not in sys.path:
    sys.path.insert(0, _FIND_STARS)

import starlib as S  # noqa: E402


def fit_calibration_frame(det_x, det_y, g_ra, g_dec, ra0, dec0, cx0, cy0, scale0, p_base):
    """One calibration frame: the extended solve, warm-started from the shared base
    geometry. Returns starlib.solve_plate's full result dict (``['p']`` is what matters
    here, len 10: affine(6) + k1,k2 + qx,qy)."""
    return S.solve_plate(det_x, det_y, g_ra, g_dec, ra0, dec0, cx0, cy0, scale0,
                         use_distortion=True, use_atmosphere=True, p0=p_base)


def reference_grid(cx, cy, half_x, half_y, n=9):
    """A regular n x n grid of REFERENCE pixel positions, centered on (cx, cy). Regular
    on purpose -- Trend interpolates it with a RegularGridInterpolator."""
    xs = cx + np.linspace(-half_x, half_x, n)
    ys = cy + np.linspace(-half_y, half_y, n)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, gx.ravel(), gy.ravel()


def grid_xi_eta(p_base, gx, gy):
    """Invert the AFFINE part of p_base: reference pixel -> (xi, eta). ``p_base``'s affine
    defines what "reference" (undistorted) means here, so every calibration frame's own
    fitted model can be evaluated at the same sky positions."""
    a, b, c, d, e, f = p_base[:6]
    det = a * d - b * c
    dx, dy = gx - e, gy - f
    xi = (d * dx - b * dy) / det
    eta = (-c * dx + a * dy) / det
    return xi, eta


class Trend:
    """Per-grid-point linear-in-time fit of the calibration frames' displacement map.

    Deliberately does not touch the raw per-frame coefficients (plan0.md: "the degeneracy
    that doesn't need resolving") -- it only ever asks "where would a star at this
    reference position actually land, in a frame taken at time t", fits a line to that per
    grid point across the calibration frames, and keeps only the SLOPE: the line's
    intercept is a component present identically in every calibration frame (lens
    distortion, the mean atmosphere state), which is registration-neutral and is dropped
    on purpose, the same way ``make_data_atmosphere.py``'s ``bulk_px`` removes its own
    mean -- v5 already absorbs a constant field shape as part of its own per-frame fit.
    """

    def __init__(self, xs, ys, t_mean, slope_x, slope_y):
        self.xs, self.ys = np.asarray(xs), np.asarray(ys)
        self.t_mean = float(t_mean)
        shape = (len(self.ys), len(self.xs))
        self._sx = RegularGridInterpolator((self.ys, self.xs), slope_x.reshape(shape),
                                           bounds_error=False, fill_value=None)
        self._sy = RegularGridInterpolator((self.ys, self.xs), slope_y.reshape(shape),
                                           bounds_error=False, fill_value=None)

    def _query(self, interp, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        pts = np.column_stack([np.broadcast_to(y, shape).ravel(),
                               np.broadcast_to(x, shape).ravel()])
        return interp(pts).reshape(shape)

    def correction(self, t, x, y):
        """(dx, dy): how far the real image has drifted from the t_mean reference
        geometry, at reference position (x, y)."""
        dt = t - self.t_mean
        return self._query(self._sx, x, y) * dt, self._query(self._sy, x, y) * dt

    def to_observed(self, t, x, y):
        """Reference (x, y) -> where that content actually is in a frame shot at time t."""
        dx, dy = self.correction(t, x, y)
        return np.asarray(x, dtype=float) + dx, np.asarray(y, dtype=float) + dy

    def to_reference(self, t, x, y, iters=3):
        """Inverse of to_observed. Fixed-point iteration: the correction is a few px on a
        ~6000 px frame and smooth, so this converges in 2-3 steps -- no Newton needed."""
        rx, ry = np.array(x, dtype=float), np.array(y, dtype=float)
        for _ in range(iters):
            dx, dy = self.correction(t, rx, ry)
            rx, ry = np.asarray(x, dtype=float) - dx, np.asarray(y, dtype=float) - dy
        return rx, ry

    def sample_indices(self, t, height, width):
        """For every REFERENCE (row, col) in an (height, width) frame, the (x, y) to
        sample from in the real, observed frame shot at time t. Literally to_observed
        evaluated on the pixel grid -- kept as its own method only so a caller doesn't
        have to build the meshgrid, and so the two can be checked against each other."""
        cols, rows = np.meshgrid(np.arange(width, dtype=float),
                                 np.arange(height, dtype=float))
        return self.to_observed(t, cols, rows)


def fit_trend(times, ps, p_base, grid):
    """times: one timestamp (s) per calibration frame. ps: that frame's fitted p (len 10,
    from fit_calibration_frame), same order. p_base: the shared affine+k1,k2 defining
    "reference". grid: reference_grid(...)'s return value.

    For each grid point, evaluates every calibration frame's own fitted model there (its
    predicted position for a star at that reference sky position), then does one closed-
    form OLS line fit vs. time per point -- no Bennett physics, no per-frame coefficient
    comparison, just "where did this point's star actually land, over time".
    """
    xs, ys, gx, gy = grid
    xi, eta = grid_xi_eta(p_base, gx, gy)
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        raise ValueError("fit_trend needs at least 2 calibration frames")
    obs = np.array([S.project(p, xi, eta) for p in ps])   # (n_frames, 2, n_grid)
    ox, oy = obs[:, 0, :], obs[:, 1, :]
    t_mean = float(times.mean())
    dt = times - t_mean
    denom = float(np.sum(dt * dt))
    if denom == 0.0:
        raise ValueError("fit_trend needs calibration frames spread over time, not one instant")
    slope_x = (dt[:, None] * ox).sum(axis=0) / denom
    slope_y = (dt[:, None] * oy).sum(axis=0) / denom
    return Trend(xs, ys, t_mean, slope_x, slope_y)


def dewarp(img, t, trend):
    """img: (H, W) or (H, W, C) array. Resample so its geometry matches trend.t_mean --
    undoes the drift, leaving whatever constant (lens + mean-atmosphere) shape was already
    shared by every calibration frame, same as the rest of v6's convention."""
    from scipy.ndimage import map_coordinates
    h, w = img.shape[:2]
    sx, sy = trend.sample_indices(t, h, w)
    coords = [sy, sx]
    if img.ndim == 2:
        return map_coordinates(img, coords, order=1, mode="nearest")
    return np.stack([map_coordinates(img[..., c], coords, order=1, mode="nearest")
                     for c in range(img.shape[2])], axis=-1)


# --------------------------------------------------------------------------------------- #
# dispersion (plan0.md round 5): a per-channel, per-frame SCALAR offset, not a spatial
# grid -- the physics this replaces (make_data_atmosphere.py's channel_terms) puts the
# field-varying part ("smear") at a few tenths of a px, well under the ~1 px constant
# term this measures, so a single (dx, dy) per channel per calibration frame is enough.
# --------------------------------------------------------------------------------------- #
CHANNEL_NAMES = ("R", "G", "B")


def _channel_fit_snr(img_c, x, y, half):
    """One channel's Gaussian fit at (x, y): (cx, cy, snr) or None. snr is the fit's own
    amplitude over its residual scatter (sqrt(2*cost/n_px)) -- self-calibrating per patch,
    no external noise model needed."""
    f = S.gauss2d_fit(img_c, x, y, half=half)
    if f is None:
        return None
    npix = (2 * half + 1) ** 2
    rms = math.sqrt(max(2.0 * f["cost"], 0.0) / npix)
    if rms <= 0:
        return None
    return f["x"], f["y"], f["amp"] / rms


def dispersion_offset(rgb, x, y, half=6, min_snr=3.0):
    """One star's per-channel (R,G,B) centroid at an ALREADY-KNOWN fused-image position
    (x, y), mean-removed -- "the mean chromatic shift is pointing, not dispersion", same
    convention make_data_atmosphere.py's channel_terms() uses. Differential on purpose
    (plan0.md "Dispersion: one wrinkle"): needs far less per-channel SNR than an
    independent per-channel detection would, since (x, y) is not being rediscovered here,
    only refined slightly per channel. rgb: (H, W, 3) array. None if any channel fails the
    basic sanity floor (``min_snr``, deliberately loose) -- real per-star SNR is often only
    a few, in which case no single star's fit should be trusted alone; that's what
    ``frame_dispersion`` is for, which pools many stars instead of discarding them.
    """
    fitted, snr = {}, {}
    for c, name in enumerate(CHANNEL_NAMES):
        r = _channel_fit_snr(rgb[..., c], x, y, half)
        if r is None or r[2] < min_snr:
            return None
        fitted[name], snr[name] = (r[0], r[1]), r[2]
    mx = sum(v[0] for v in fitted.values()) / 3.0
    my = sum(v[1] for v in fitted.values()) / 3.0
    return {name: (vx - mx, vy - my) for name, (vx, vy) in fitted.items()}


def frame_dispersion(rgb, xs, ys, half=6, min_snr=3.0):
    """Pool MANY stars into one per-frame, per-channel (dx, dy), inverse-variance weighted
    by each star's own fit SNR**2. This, not dispersion_offset() on a single bright star,
    is the primitive fit_dispersion_trend actually wants: real per-star per-channel SNR is
    typically single digits to a few tens even for the brightest confirmed stars (measured
    on data_atmosphere/), so trusting one star -- or an unweighted median across a few --
    both quietly regress toward zero offset instead of failing loudly. Pooling many
    marginal-SNR stars is the same reasoning solve_plate's geometric fit already relies on
    (many stars in one joint fit, not one star at a time). Returns (offsets, n_stars_used),
    or (None, n) if fewer than 5 stars clear even the basic sanity floor.
    """
    acc = {name: [0.0, 0.0, 0.0] for name in CHANNEL_NAMES}
    n = 0
    for x, y in zip(xs, ys):
        fitted, snr = {}, {}
        ok = True
        for c, name in enumerate(CHANNEL_NAMES):
            r = _channel_fit_snr(rgb[..., c], x, y, half)
            if r is None or r[2] < min_snr:
                ok = False
                break
            fitted[name], snr[name] = (r[0], r[1]), r[2]
        if not ok:
            continue
        mx = sum(v[0] for v in fitted.values()) / 3.0
        my = sum(v[1] for v in fitted.values()) / 3.0
        n += 1
        for name in CHANNEL_NAMES:
            w = snr[name] ** 2
            acc[name][0] += w * (fitted[name][0] - mx)
            acc[name][1] += w * (fitted[name][1] - my)
            acc[name][2] += w
    if n < 5:
        return None, n
    return ({name: (acc[name][0] / acc[name][2], acc[name][1] / acc[name][2])
            for name in CHANNEL_NAMES}, n)


class DispersionTrend:
    """Linear-in-time trend of dispersion_offset(), one scalar (dx, dy) slope per channel.
    Same mean-removed convention as Trend: ``offset(t, ...)`` is slope * (t - t_mean)."""

    def __init__(self, t_mean, slope):
        self.t_mean = float(t_mean)
        self.slope = slope    # {"R": (sx, sy), "G": ..., "B": ...}, px/s

    def offset(self, t, channel):
        sx, sy = self.slope[channel]
        dt = t - self.t_mean
        return sx * dt, sy * dt


def fit_dispersion_trend(times, offsets):
    """times: one per calibration frame. offsets: dispersion_offset()'s return value, same
    order, with any None entries already dropped by the caller."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        raise ValueError("fit_dispersion_trend needs at least 2 calibration frames")
    t_mean = float(times.mean())
    dt = times - t_mean
    denom = float(np.sum(dt * dt))
    if denom == 0.0:
        raise ValueError("fit_dispersion_trend needs calibration frames spread over time")
    slope = {}
    for name in CHANNEL_NAMES:
        dx = np.array([o[name][0] for o in offsets])
        dy = np.array([o[name][1] for o in offsets])
        slope[name] = (float(np.sum(dt * dx) / denom), float(np.sum(dt * dy) / denom))
    return DispersionTrend(t_mean, slope)


def apply_dispersion(rgb, t, trend):
    """Undo each channel's fitted dispersion drift at time t -- a plain whole-channel
    sub-pixel shift, not a spatial resample (order of operations step 6: applied
    post-demosaic, i.e. exactly the (H, W, 3) array this operates on)."""
    from scipy.ndimage import shift as ndi_shift
    out = np.empty_like(rgb)
    for c, name in enumerate(CHANNEL_NAMES):
        dx, dy = trend.offset(t, name)
        # A channel that DRIFTED by (dx, dy) needs shifting by (-dy, -dx) (row, col) to
        # undo it -- same sign convention as Trend/dewarp.
        out[..., c] = ndi_shift(rgb[..., c], (-dy, -dx), order=1, mode="nearest")
    return out


# --------------------------------------------------------------------------------------- #
# building the correction directly from a pipeline run's own frames -- no separate
# precompute script, no known-star shortcut. This is what pipeline.py calls by default.
# --------------------------------------------------------------------------------------- #
SITE_2026 = dict(lat=42.2323283, lon=-3.2077811, elev=2131.0)
DEFAULT_SCALE0 = 3.061          # "/px, Z6 400mm, 5.936 um pitch -- check_02_target_2026.py
DEFAULT_GAIA_RADIUS_DEG = 3.2
DEFAULT_GAIA_GMAX = 12.0


def _t_utc_from_unix(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return S.t_utc(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                   dt.second + dt.microsecond / 1e6)


def _exif_gps(path):
    """(lat, lon, elev) from this file's own EXIF GPS tags, or None if it has none (many
    cameras -- quite possibly the real 2026 one -- carry no GPS module at all)."""
    md = S.exiftool([str(path)], tags=["GPSLatitude", "GPSLongitude", "GPSAltitude"])
    if not md:
        return None
    lat, lon, elev = md[0].get("GPSLatitude"), md[0].get("GPSLongitude"), md[0].get("GPSAltitude")
    if lat is None or lon is None:
        return None
    return dict(lat=float(lat), lon=float(lon), elev=float(elev) if elev is not None else 0.0)


def calibrate_from_source(image_infos, device, exp_min=0.25, peak_thresh=4.0,
                          min_stars=6, peak_keep=500, scale0=DEFAULT_SCALE0,
                          site=None, grid_frac=0.6, verbose=True):
    """Build a Trend directly from a run's own frames -- works for any FrameSource (jpg,
    nef, atmosphere, ...) since it only ever calls ``ii.source.load_gray(ii, device)``, the
    one method every source implements, rather than anything format-specific.

    No prior plate solve exists for an arbitrary camera/run, so the first calibration-
    capable frame (exposure >= exp_min) gets a blind solve (starlib.dogsnr + find_peaks
    across the whole frame, matched against a fresh Gaia query) to bootstrap a base
    geometry; every frame after that warm-starts from it, same as round 2/3's validated
    approach. Tangent point: the Moon's real apparent position at the middle calibration
    frame's own EXIF timestamp, seen from ``site`` -- if not given, read from that same
    frame's own EXIF GPS tags, falling back to SITE_2026 if it has none. This is what makes
    data_atmosphere/ (2024 EXIF/GPS, copied verbatim) and real 2026 NEFs both resolve to
    the correct site automatically, with no mode-specific branch.

    Raises RuntimeError if the bootstrap never succeeds or too few frames end up usable --
    the caller (pipeline.py) is not meant to silently fall back to an uncorrected run.
    """
    calib_infos = sorted((ii for ii in image_infos if ii.exposure_time >= exp_min),
                         key=lambda ii: ii.timestamp)
    if verbose:
        print(f"  {len(calib_infos)}/{len(image_infos)} frames are calibration-capable "
             f"(exposure >= {exp_min} s)")
    if not calib_infos:
        raise RuntimeError("no calibration-capable frames for the atmosphere correction "
                           f"(need exposure >= {exp_min} s)")

    mid = calib_infos[len(calib_infos) // 2]
    if site is None:
        gps = _exif_gps(mid.path)
        site = gps if gps is not None else SITE_2026
        if verbose:
            print(f"  site: {'EXIF GPS' if gps is not None else 'SITE_2026 (no GPS in EXIF)'} "
                 f"-> lat={site['lat']:.4f} lon={site['lon']:.4f} elev={site['elev']:.0f} m")
    t_mid = _t_utc_from_unix(mid.timestamp)
    ra0, dec0, _dist, _alt, _az = S.apparent("moon", t_mid, site["lat"], site["lon"],
                                             site["elev"])
    g = S.gaia(ra0, dec0, DEFAULT_GAIA_RADIUS_DEG, DEFAULT_GAIA_GMAX,
              tag=f"gaia_{ra0:.4f}_{dec0:+.4f}_r{DEFAULT_GAIA_RADIUS_DEG:.2f}")
    if verbose:
        print(f"  tangent point (Moon @ {mid.path.name}): ra={ra0:.4f} dec={dec0:.4f}, "
             f"{len(g)} Gaia stars in range")

    cx0 = (calib_infos[0].width - 1) / 2.0
    cy0 = (calib_infos[0].height - 1) / 2.0

    p_base = None
    calib = []
    for ii in calib_infos:
        gimg = ii.source.load_gray(ii, device).detach().cpu().numpy().astype(np.float64)
        snr, _bg = S.dogsnr(gimg)
        xs, ys, vals = S.find_peaks(snr, thresh=peak_thresh)
        if len(xs) == 0:
            if verbose:
                print(f"  {ii.path.name}: no peaks above SNR {peak_thresh} -- skip")
            continue
        k = np.argsort(vals)[::-1][:peak_keep]
        det_x, det_y = xs[k].astype(float), ys[k].astype(float)

        if p_base is None:
            # Bootstrap: no prior solve exists yet, so this one frame pays for the blind
            # brute-force rotation/parity search every later frame skips via warm-start.
            res = S.solve_plate(det_x, det_y, g["ra"], g["dec"], ra0, dec0, cx0, cy0,
                                scale0, use_distortion=True, use_atmosphere=True,
                                rot_step=0.2)
            tag = "BOOTSTRAP "
        else:
            res = S.solve_plate(det_x, det_y, g["ra"], g["dec"], ra0, dec0, cx0, cy0,
                                scale0, use_distortion=True, use_atmosphere=True, p0=p_base)
            tag = ""
        ok = np.isfinite(res["rms"]) and res["nmatch"] >= min_stars
        if verbose:
            print(f"  {tag}{ii.path.name}: {len(det_x)} peaks, nmatch={res['nmatch']}, "
                 f"rms={res['rms']:.3f} px -> {'used' if ok else 'REJECTED'}")
        if ok:
            if p_base is None:
                p_base = res["p"]
            calib.append((float(ii.timestamp), res["p"]))

    if p_base is None:
        raise RuntimeError("atmosphere calibration bootstrap never succeeded -- no "
                           "calibration frame produced a usable blind solve")
    if len(calib) < 4:
        raise RuntimeError(f"only {len(calib)} usable calibration frames (need >= 4) for "
                           "the atmosphere trend fit")

    grid = reference_grid(cx0, cy0, grid_frac * cx0, grid_frac * cy0, n=9)
    return fit_trend([t for t, _ in calib], [p for _, p in calib], p_base, grid)
    return out
