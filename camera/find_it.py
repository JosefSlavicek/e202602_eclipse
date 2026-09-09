#!/usr/bin/env python3
"""Shoot one frame, find the stars in it, and point an arrow at M31.

This is a cousin of ``get_histogram.py`` and ``focus_hunt_stars.py``. It borrows
their pieces -- the gphoto2 single-shot capture, the linear rawpy decode, the
pure-linear (no stretch) display and the integer-zoom pan/zoom viewer -- but the
job is different:

  1. Take one frame with the camera's current settings (or a shutter you pass
     with ``--exposure``).  ``--sim FILE`` uses an existing raw file instead, so
     the whole pipeline can be exercised with no camera attached.

  2. Decode it linearly (rawpy, unit white balance, raw colour) and detect the
     stars: local-background subtraction, threshold, connected components,
     flux-weighted centroids.

  3. Blind-solve the pointing from those star positions by matching them against
     a local bright-star catalogue (HYG, mag <= 9.5, shipped as a small .npz).
     A long detected star pair is lined up with catalogue pairs of the same
     length (both orders, both parities) to seed a similarity transform; the
     best seed is confirmed and tightened with an iterative, reprojected
     least-squares WCS fit.  No internet, no external solver.

  4. Draw a green arrow (3 px) from the image centre towards the centre of M31.
     If M31 is off the frame the arrow stops at the image border.

What you must give it for the solve to work:
  * a Nikon Z 6 + 400 mm lens frame (the plate scale is hard-coded for that);
  * pointing somewhere within ``--search-deg`` (default 8 deg) of M31;
  * enough stars -- roughly 6 or more detected.  A very short/empty exposure
    will not solve; the script then just shows the frame with the detected
    stars circled and says so.  It will not draw a guessed arrow.

System clock accuracy does not matter: M31 is a fixed target and the solve
comes from the stars, not from time.

Run with the python that has gphoto2 + rawpy + numpy + scipy + Pillow
(the e202602_eclipse env).

Usage
-----
    camera/find_it.py [--exposure SEC] [--iso N] [--search-deg D]
    camera/find_it.py --sim path/to/frame.NEF [--no-gui]

    # one-off, online, to (re)build the star catalogue from a HYG csv:
    #   curl -LO https://github.com/astronexus/HYG-Database/raw/main/hyg/CURRENT/hygdata_v41.csv
    camera/find_it.py --build-catalog hygdata_v41.csv

The shipped catalogue (find_it_data/stars.npz, ~1.3 MB, HYG mag <= 9.5) is all
the script needs at run time; the csv is only for rebuilding it.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "find_it_data")
CATALOG_NPZ = os.path.join(DATA_DIR, "stars.npz")

# ---- rig / target constants -------------------------------------------------
M31_RA_DEG = 10.68470834      # NGC 224, J2000  (00h 42m 44.33s)
M31_DEC_DEG = 41.26875        #                 (+41d 16' 07.5")
M31_NAME = "M31 (Andromeda)"

SENSOR_W_MM = 35.9            # Nikon Z 6 imaging area width
DEFAULT_FOCAL_MM = 400.0     # NIKKOR Z 400mm f/4.5
NEF_WHITE = 65535.0          # rawpy output_bps=16 white point

# ---- solver tunables ------------------------------------------------------
CAT_MAG = 9.0           # catalogue depth used for blind matching
VERIFY_MAG = 9.5       # catalogue depth for the final all-star match
DET_TRI_N = 16        # brightest detected stars used to form pairs
PAIR_TRIES = 8       # how many of the longest detected pairs to try as a baseline
PAIR_LEN_TOL = 0.05  # catalogue pair length must be within +-5% of the detected one
HYP_KEEP = 8         # distinct pair hypotheses handed to the full fit
HYP_TOL_PX = 45.0    # star-match gate while scoring / seeding a hypothesis
                     #  (loose: the M31-tangent projection distorts a far field;
                     #   _refine_finalize is the real quality gate)
REFINE_TOL_PX = 3.0  # inlier gate once the affine is fitted
MIN_DETECT = 6       # need at least this many detected stars to try at all
MIN_INLIERS = 6      # and this many catalogue matches to accept a solve
MAX_RMS_PX = 2.0     # and a fit residual below this
SCALE_TOL = 0.08     # fitted scale must be within +-8% of nominal


# ==========================================================================
# capture  (condensed from andromeda.py)
# ==========================================================================
SHUTTERSPEED_PRESETS = [
    '0.0001s', '0.0002s', '0.0003s', '0.0004s', '0.0005s', '0.0006s', '0.0008s',
    '0.0010s', '0.0012s', '0.0015s', '0.0020s', '0.0025s', '0.0031s', '0.0040s',
    '0.0050s', '0.0062s', '0.0080s', '0.0100s', '0.0125s', '0.0166s', '0.0200s',
    '0.0250s', '0.0333s', '0.0400s', '0.0500s', '0.0666s', '0.0769s', '0.1000s',
    '0.1250s', '0.1666s', '0.2000s', '0.2500s', '0.3333s', '0.4000s', '0.5000s',
    '0.6250s', '0.7692s', '1.0000s', '1.3000s', '1.6000s', '2.0000s', '2.5000s',
    '3.0000s', '4.0000s', '5.0000s', '6.0000s', '8.0000s', '10.0000s', '13.0000s',
    '15.0000s', '20.0000s', '25.0000s', '30.0000s',
]
_PRESETS = sorted((float(s[:-1]), s) for s in SHUTTERSPEED_PRESETS)
MAX_TIMED_S = _PRESETS[-1][0]


def _nearest_shorter_preset(seconds: float):
    cand = [(s, n) for s, n in _PRESETS if s <= seconds + 1e-9]
    return cand[-1] if cand else _PRESETS[0]


def capture_one(exposure: float | None, iso: str | None):
    """Take one frame and return (raw_bytes, name).

    exposure None  -> leave the camera's shutter as it is.
    exposure <=30  -> nearest shorter hardware preset.
    exposure >30   -> bulb.
    """
    import gphoto2 as gp

    cam = gp.Camera()
    cam.init()
    try:
        cfg = cam.get_config()

        def _set(name, value):
            try:
                node = cfg.get_child_by_name(name)
            except gp.GPhoto2Error:
                print(f"[warn] camera has no setting '{name}', skipping")
                return
            node.set_value(value)
            cam.set_config(cfg)
            print(f"[cam] {name} = {value}")

        _set('capturetarget', 'Memory card')
        _set('capturemode', 'Single Shot')
        try:
            _set('longexpnr', 'Off')
        except Exception:
            pass
        if iso is not None:
            _set('iso', iso)

        use_bulb = exposure is not None and exposure > MAX_TIMED_S
        if exposure is not None and not use_bulb:
            sec, name = _nearest_shorter_preset(exposure)
            if abs(sec - exposure) > 1e-6:
                print(f"[cam] requested {exposure:.4f}s -> preset {name}")
            _set('shutterspeed', name)
        elif use_bulb:
            _set('shutterspeed', 'Bulb')

        cur = cam.get_single_config('shutterspeed').get_value()
        print(f"[cam] shooting one frame at shutter={cur}")

        if use_bulb:
            bulb = cfg.get_child_by_name('bulb')
            bulb.set_value(1)
            cam.set_config(cfg)
            time.sleep(exposure)
            bulb.set_value(0)
            cam.set_config(cfg)
            path = _wait_for_file(cam, gp)
        else:
            path = cam.capture(gp.GP_CAPTURE_IMAGE)

        folder, name = path.folder, path.name
        print(f"[cam] captured {folder}/{name}")
        cf = cam.file_get(folder, name, gp.GP_FILE_TYPE_NORMAL)
        data = memoryview(cf.get_data_and_size()).tobytes()
        return data, name
    finally:
        cam.exit()


def _wait_for_file(cam, gp, timeout_ms=20000):
    waited = 0
    while waited < timeout_ms:
        ev_type, ev_data = cam.wait_for_event(200)
        waited += 200
        if ev_type == gp.GP_EVENT_FILE_ADDED:
            return ev_data
    raise RuntimeError("timed out waiting for the bulb frame to be written")


# ==========================================================================
# decode  (from get_histogram.decode_nef)
# ==========================================================================
def decode_raw(raw: bytes, name: str) -> np.ndarray:
    """Decode one raw frame into a linear uint16 HxWx3 array."""
    try:
        import rawpy
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"need rawpy to decode {name}: {e}")
    try:
        with rawpy.imread(io.BytesIO(raw)) as r:
            rgb16 = r.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16,
                user_wb=[1.0, 1.0, 1.0, 1.0], user_flip=0,
                output_color=rawpy.ColorSpace.raw)
    except Exception as e:
        raise RuntimeError(f"{name}: rawpy could not decode this frame ({e})")
    return np.ascontiguousarray(rgb16)


def to_display_gray(rgb16: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Pure-linear 8-bit grey RGB for the viewer (no stretch, optional lin gain)."""
    y = rgb16.astype(np.float32).mean(axis=2) / NEF_WHITE
    if gain != 1.0:
        y = y * gain
    g = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


# ==========================================================================
# star detection
# ==========================================================================
def detect_stars(rgb16: np.ndarray, max_stars: int = 80) -> list[dict]:
    """Return a flux-sorted list of {x, y, flux, area, peak} for detected stars."""
    import scipy.ndimage as ndi

    gray = rgb16.astype(np.float32).mean(axis=2)
    h, w = gray.shape

    # coarse local background: median on a decimated grid, bilinearly upsampled
    f = 32
    small = gray[::f, ::f]
    bg_small = ndi.median_filter(small, size=5, mode="nearest")
    bg = ndi.zoom(bg_small, (h / bg_small.shape[0], w / bg_small.shape[1]),
                  order=1)[:h, :w]
    resid = gray - bg

    samp = resid[::7, ::7]
    med = float(np.median(samp))
    mad = float(np.median(np.abs(samp - med))) + 1e-6
    sigma = 1.4826 * mad
    thr = med + 6.0 * sigma

    mask = resid > thr
    lbl, n = ndi.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = np.bincount(lbl.ravel())[1:]
    flux = ndi.sum(resid, lbl, idx)
    com = ndi.center_of_mass(np.clip(resid, 0, None), lbl, idx)   # (y, x)
    boxes = ndi.find_objects(lbl)

    out = []
    for k in range(n):
        a = int(area[k])
        if a < 3 or a > 6000:
            continue
        sy, sx = boxes[k]
        bh, bw = sy.stop - sy.start, sx.stop - sx.start
        elong = max(bh, bw) / max(1, min(bh, bw))
        if elong > 3.0 and a > 25:          # streak / galaxy edge / gradient
            continue
        cy, cx = com[k]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        out.append(dict(x=float(cx), y=float(cy), flux=float(flux[k]),
                        area=a, peak=float(resid[sy, sx].max())))
    out.sort(key=lambda d: -d["flux"])
    return out[:max_stars]


# ==========================================================================
# spherical / tangent-plane helpers  (all angles in radians unless _deg)
# ==========================================================================
def angsep_deg(ra1, dec1, ra0_deg, dec0_deg):
    a = np.radians(ra1)
    d = np.radians(dec1)
    a0 = math.radians(ra0_deg)
    d0 = math.radians(dec0_deg)
    c = (np.sin(d) * math.sin(d0) +
         np.cos(d) * math.cos(d0) * np.cos(a - a0))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def radec_to_tan(ra, dec, ra0, dec0):
    """Gnomonic. Returns xi (East), eta (North) in radians."""
    d = ra - ra0
    sd0, cd0 = math.sin(dec0), math.cos(dec0)
    sd, cd = np.sin(dec), np.cos(dec)
    cosd = np.cos(d)
    cosc = sd0 * sd + cd0 * cd * cosd
    xi = cd * np.sin(d) / cosc
    eta = (cd0 * sd - sd0 * cd * cosd) / cosc
    return xi, eta


def tan_to_radec(xi, eta, ra0, dec0):
    xi = np.asarray(xi, float)
    eta = np.asarray(eta, float)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    sc, cc = np.sin(c), np.cos(c)
    sd0, cd0 = math.sin(dec0), math.cos(dec0)
    small = rho < 1e-12
    rho_safe = np.where(small, 1.0, rho)
    dec = np.arcsin(cc * sd0 + eta * sc * cd0 / rho_safe)
    ra = ra0 + np.arctan2(xi * sc,
                          rho_safe * cd0 * cc - eta * sd0 * sc)
    dec = np.where(small, dec0, dec)
    ra = np.where(small, ra0, ra)
    return ra, dec


def _fit_affine(src, dst):
    """Least-squares  dst = src @ A.T + t .  Returns (A 2x2, t 2)."""
    n = len(src)
    X = np.hstack([src, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
    return sol[:2].T, sol[2]


# ==========================================================================
# blind solve
# ==========================================================================
def load_catalog(path: str = CATALOG_NPZ):
    if not os.path.exists(path):
        raise RuntimeError(
            f"star catalogue not found: {path}\n"
            f"build it once (online) with:  {sys.argv[0]} "
            f"--build-catalog path/to/hygdata_v41.csv")
    d = np.load(path, allow_pickle=True)
    return d["ra_deg"], d["dec_deg"], d["mag"], d["name"]


def nominal_arcsec_px(decoded_w: int, focal_mm: float) -> float:
    fov_deg = math.degrees(2.0 * math.atan((SENSOR_W_MM / 2.0) / focal_mm))
    return fov_deg * 3600.0 / decoded_w


def _radec_xyz(ra_deg, dec_deg):
    r = np.radians(np.atleast_1d(ra_deg))
    d = np.radians(np.atleast_1d(dec_deg))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def _sep_arcsec(u, v):
    """Angular separation (arcsec) between rows of unit-vector arrays u, v."""
    d = np.clip(np.einsum("...i,...i->...", u, v), -1.0, 1.0)
    return np.degrees(np.arccos(d)) * 3600.0


def _score_similarities(p1, p2, q1, q2, flip, Pc, Qtree, tol):
    """Vectorised: for K catalogue pairs (q1,q2) matched to one detected pair
    (p1,p2) at parity `flip`, build every similarity transform and score each by
    how many detected stars land within `tol` px of a catalogue star.

    Returns (scores K, M Kx2x2, t Kx2); scale-gated transforms score -1.
    """
    f = flip
    a1 = np.array([f * p1[0], p1[1]])
    a2 = np.array([f * p2[0], p2[1]])
    dp = a2 - a1
    den = dp[0] ** 2 + dp[1] ** 2
    dq = q2 - q1                                        # (K,2)
    cs = (dq @ dp) / den
    sn = (dq[:, 1] * dp[0] - dq[:, 0] * dp[1]) / den
    K = len(q1)
    M = np.empty((K, 2, 2))
    M[:, 0, 0] = cs * f
    M[:, 0, 1] = -sn
    M[:, 1, 0] = sn * f
    M[:, 1, 1] = cs
    t = q1 - np.einsum("kij,j->ki", M, p1)             # q = M p + t
    pred = (np.einsum("ni,kji->knj", Pc, M) + t[:, None, :]).reshape(-1, 2)
    d, _ = Qtree.query(pred, distance_upper_bound=tol)
    scores = np.isfinite(d).reshape(K, len(Pc)).sum(axis=1).astype(int)
    scores[np.abs(np.hypot(cs, sn) - 1.0) > SCALE_TOL] = -1
    return scores, M, t


def _pair_hypotheses(Pc, det_n, W, H, cxyz, Qcat, s_arcsec_px, verbose):
    """Blind match: line up long detected star pairs with catalogue pairs.

    Each of the longest detected pairs is matched against every catalogue pair
    of consistent length, in both orders and both parities; every match gives a
    similarity transform, scored by how many detected stars then land on a
    catalogue star.  Returns a de-duplicated, score-sorted list of (score, M, t).
    """
    from scipy.spatial import cKDTree

    frame_diag = math.hypot(W, H)
    band = PAIR_LEN_TOL
    Qtree = cKDTree(Qcat)

    cpair = cKDTree(cxyz).query_pairs(
        2.0 * math.sin(math.radians(
            frame_diag * s_arcsec_px * (1 + band) / 3600.0) / 2.0),
        output_type="ndarray")
    if len(cpair) < 5:
        return []
    cpx = _sep_arcsec(cxyz[cpair[:, 0]], cxyz[cpair[:, 1]]) / s_arcsec_px
    o = np.argsort(cpx)
    cpair, cpx = cpair[o], cpx[o]

    dd = sorted(((float(np.hypot(*(Pc[i] - Pc[j]))), i, j)
                 for i in range(det_n) for j in range(i + 1, det_n)),
                reverse=True)
    cand = []
    for gpx, i, j in dd[:PAIR_TRIES]:
        if gpx < 0.30 * frame_diag:
            break
        lo = int(np.searchsorted(cpx, gpx * (1.0 - band)))
        hi = int(np.searchsorted(cpx, gpx * (1.0 + band)))
        cp = cpair[lo:hi]
        if len(cp) == 0:
            continue
        for ca, cb in ((cp[:, 0], cp[:, 1]), (cp[:, 1], cp[:, 0])):
            q1, q2 = Qcat[ca], Qcat[cb]
            for flip in (1.0, -1.0):
                sc, M, t = _score_similarities(Pc[i], Pc[j], q1, q2, flip,
                                               Pc, Qtree, HYP_TOL_PX)
                for k in np.where(sc >= MIN_INLIERS)[0]:
                    cand.append((int(sc[k]), M[k], t[k]))

    cand.sort(key=lambda z: -z[0])
    kept = []
    for score, M, t in cand:
        if any(np.allclose(M, KM, atol=0.02) and np.allclose(t, Kt, atol=200)
               for _, KM, Kt in kept):
            continue
        kept.append((score, M, t))
        if len(kept) >= HYP_KEEP:
            break
    if verbose:
        print(f"[solve] pair hypotheses kept: {[k[0] for k in kept]}")
    return kept


def solve_field(stars, W, H, cat, ra0_deg, dec0_deg, s_arcsec_px,
                search_deg, verbose=True):
    """Blind-solve pointing from detected star pixels.

    stars : list of {x,y,...} (flux-sorted).  cat : (ra_deg,dec_deg,mag,name).
    Returns a dict; ['ok'] tells you whether to trust it.
    """
    ra_deg, dec_deg, mag, name = cat
    s_deg = s_arcsec_px / 3600.0
    result = {"ok": False, "reason": "", "n_detected": len(stars)}
    if len(stars) < MIN_DETECT:
        result["reason"] = f"only {len(stars)} stars detected (need {MIN_DETECT})"
        return result

    P = np.array([[s["x"], s["y"]] for s in stars], float)
    Pc = P - np.array([W / 2.0, H / 2.0])

    fov_deg = math.hypot(W, H) * s_deg
    cone = search_deg + fov_deg / 2.0 + 2.0
    sep = angsep_deg(ra_deg, dec_deg, ra0_deg, dec0_deg)
    cg = np.where((sep < cone) & (mag <= CAT_MAG))[0]         # global cat indices
    if len(cg) < 15:
        result["reason"] = "too few catalogue stars near the guess position"
        return result
    cxyz = _radec_xyz(ra_deg[cg], dec_deg[cg])

    # catalogue in "focal pixels" on a tangent plane at the guess position
    xi, eta = radec_to_tan(np.radians(ra_deg[cg]), np.radians(dec_deg[cg]),
                           math.radians(ra0_deg), math.radians(dec0_deg))
    Qcat = np.column_stack([np.degrees(xi) / s_deg, np.degrees(eta) / s_deg])

    det_n = min(len(P), DET_TRI_N)
    if verbose:
        print(f"[solve] {len(cg)} catalogue stars in cone, "
              f"{det_n} detected stars used")

    hyps = _pair_hypotheses(Pc, det_n, W, H, cxyz, Qcat, s_arcsec_px, verbose)
    if not hyps:
        result["reason"] = "no star pattern matched the catalogue"
        return result

    passed = []
    for score, M, t0 in hyps:
        sol = _refine_finalize(P, Pc, W, H, M, t0, cat, cg, Qcat,
                               s_arcsec_px, ra0_deg, dec0_deg, fov_deg, verbose)
        if sol is not None:
            passed.append(sol)

    if not passed:
        result["reason"] = "a star pattern matched but no fit met the quality bar"
        return result

    passed.sort(key=lambda s: (-s["n_inliers"], s["rms"]))
    best = passed[0]
    for other in passed[1:]:
        if angsep_deg(np.array([other["center"][0]]),
                      np.array([other["center"][1]]),
                      best["center"][0], best["center"][1])[0] > 0.5:
            result["reason"] = "two inconsistent pointings both fit -- refusing"
            return result

    best.update(n_detected=len(stars), catalog=cat, detected_xy=P)
    return best


def _refine_finalize(P, Pc, W, H, M0, t0, cat, cg, Qcat, s_arcsec_px,
                     ra0_deg, dec0_deg, fov_deg, verbose):
    """From a coarse (M0, t0), iterate a reprojected WCS fit and grade it.

    Returns a solution dict with ok=True, or None if it fails the quality bar.
    """
    from scipy.spatial import cKDTree

    ra_deg, dec_deg, mag, _ = cat
    s_deg = s_arcsec_px / 3600.0

    def project(idx, cen):
        xi, eta = radec_to_tan(np.radians(ra_deg[idx]), np.radians(dec_deg[idx]),
                               math.radians(cen[0]), math.radians(cen[1]))
        return np.column_stack([np.degrees(xi) / s_deg, np.degrees(eta) / s_deg])

    def recenter(t_vec, cen):
        ra_c, dec_c = tan_to_radec(math.radians(t_vec[0] * s_deg),
                                   math.radians(t_vec[1] * s_deg),
                                   math.radians(cen[0]), math.radians(cen[1]))
        return (float(np.degrees(ra_c)) % 360.0, float(np.degrees(dec_c)))

    center = (float(ra0_deg), float(dec0_deg))
    A, t = M0.copy(), np.asarray(t0, float).copy()
    cj = None
    n_in = rms = None
    for it in range(16):
        # --- match all detected stars against a fresh projection at `center` --
        m = angsep_deg(ra_deg, dec_deg, center[0], center[1]) < fov_deg / 2.0 + 0.5
        depth = CAT_MAG if it < 4 else VERIFY_MAG
        fidx = np.where(m & (mag <= depth))[0]
        if len(fidx) < MIN_INLIERS:
            return None
        Qf = project(fidx, center)
        tol = HYP_TOL_PX if it == 0 else (12.0 if it < 4 else REFINE_TOL_PX)
        dist, jj = cKDTree(Qf).query(Pc @ A.T + t, distance_upper_bound=tol)
        good = np.isfinite(dist)
        if good.sum() < MIN_INLIERS:
            return None
        di, cj = np.where(good)[0], fidx[jj[good]]

        # --- fit at `center`, move the centre, refit at the new centre -------
        A, t = _fit_affine(Pc[di], project(cj, center))
        new_center = recenter(t, center)
        moved = angsep_deg(np.array([new_center[0]]), np.array([new_center[1]]),
                           *center)[0]
        A, t = _fit_affine(Pc[di], project(cj, new_center))
        center = new_center

        resid = Pc[di] @ A.T + t - project(cj, center)
        n_in = len(di)
        rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
        if it >= 4 and moved < 1e-6:
            break

    good_idx = np.arange(len(cj))
    matched = [(int(di[k]), int(cj[k])) for k in good_idx]
    det = float(np.linalg.det(A))
    scale = math.sqrt(abs(det)) * s_arcsec_px
    if verbose:
        print(f"[solve]   candidate: inliers={n_in} rms={rms:.2f}px "
              f"scale_ratio={scale / s_arcsec_px:.3f} "
              f"centre=({center[0]:.3f},{center[1]:+.3f})")
    if n_in < MIN_INLIERS or rms > MAX_RMS_PX or abs(scale / s_arcsec_px - 1.0) > SCALE_TOL:
        return None
    return {
        "ok": True, "A": A, "t": t, "center": center, "s_deg": s_deg,
        "n_inliers": n_in, "rms": rms, "parity": -1 if det < 0 else 1,
        "scale_arcsec_px": scale,
        "roll_deg": math.degrees(math.atan2(-A[0, 1], A[1, 1])),
        "matched": matched,
    }


def radec_to_pixel(sol, ra_deg, dec_deg, W, H):
    """Map sky coords to pixel (x, y) using a solved field."""
    xi, eta = radec_to_tan(np.radians(np.atleast_1d(ra_deg)),
                           np.radians(np.atleast_1d(dec_deg)),
                           math.radians(sol["center"][0]),
                           math.radians(sol["center"][1]))
    Q = np.column_stack([np.degrees(xi) / sol["s_deg"],
                         np.degrees(eta) / sol["s_deg"]])
    Ainv = np.linalg.inv(sol["A"])
    Pc = (Q - sol["t"]) @ Ainv.T
    return Pc + np.array([W / 2.0, H / 2.0])


# ==========================================================================
# arrow geometry
# ==========================================================================
def _clip_to_rect(p0, p1, W, H):
    """Return the point where the ray p0->p1 leaves the [0,W]x[0,H] rectangle."""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    ts = []
    if dx != 0:
        for bx in (0.0, W):
            t = (bx - x0) / dx
            if t > 0:
                y = y0 + t * dy
                if -1 <= y <= H + 1:
                    ts.append(t)
    if dy != 0:
        for by in (0.0, H):
            t = (by - y0) / dy
            if t > 0:
                x = x0 + t * dx
                if -1 <= x <= W + 1:
                    ts.append(t)
    if not ts:
        return p1
    t = min(ts)
    return (x0 + t * dx, y0 + t * dy)


def arrow_endpoints(sol, W, H):
    """(start_xy, end_xy, m31_in_frame). start is always the image centre."""
    start = (W / 2.0, H / 2.0)
    m31 = radec_to_pixel(sol, M31_RA_DEG, M31_DEC_DEG, W, H)[0]
    in_frame = (0 <= m31[0] <= W) and (0 <= m31[1] <= H)
    if in_frame:
        return start, (float(m31[0]), float(m31[1])), True
    return start, _clip_to_rect(start, m31, W, H), False


# ==========================================================================
# annotation (headless PNG output)
# ==========================================================================
def _draw_arrow(draw, p0, p1, color, width):
    draw.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    L = 26 + 4 * width
    for da in (math.radians(150), math.radians(-150)):
        draw.line([p1, (p1[0] + L * math.cos(ang + da),
                        p1[1] + L * math.sin(ang + da))], fill=color, width=width)


def annotate(rgb16, stars, sol, gain, path):
    from PIL import Image, ImageDraw

    disp = to_display_gray(rgb16, gain)
    im = Image.fromarray(disp)
    dr = ImageDraw.Draw(im)
    H, W = disp.shape[:2]

    for s in stars:
        r = 6 + 1.5 * math.sqrt(s["area"])
        r = min(r, 60)
        dr.ellipse([s["x"] - r, s["y"] - r, s["x"] + r, s["y"] + r],
                   outline=(255, 80, 80), width=2)

    if sol and sol.get("ok"):
        cat = sol["catalog"]
        for di, ci in sol["matched"]:
            px = radec_to_pixel(sol, cat[0][ci], cat[1][ci], W, H)[0]
            dr.ellipse([px[0] - 10, px[1] - 10, px[0] + 10, px[1] + 10],
                       outline=(80, 160, 255), width=2)
            nm = str(cat[3][ci])
            if nm:
                dr.text((px[0] + 12, px[1] - 6), nm, fill=(120, 200, 255))
        start, end, inframe = arrow_endpoints(sol, W, H)
        _draw_arrow(dr, start, end, (0, 255, 0), 3)
        tag = "M31" if inframe else "M31 (off frame) ->"
        dr.text((end[0] + 6, end[1] + 6), tag, fill=(0, 255, 0))
        dr.text((8, 8),
                f"solved: centre RA {sol['center'][0]:.3f} Dec {sol['center'][1]:+.3f}  "
                f"roll {sol['roll_deg']:+.1f}  scale {sol['scale_arcsec_px']:.2f}\"/px  "
                f"inliers {sol['n_inliers']}  rms {sol['rms']:.2f}px",
                fill=(0, 255, 0))
    else:
        msg = sol["reason"] if sol else "not solved"
        dr.text((8, 8), f"NOT SOLVED - no arrow  ({msg})", fill=(255, 120, 120))

    im.save(path)
    print(f"[out] wrote {path}")


# ==========================================================================
# GUI viewer  (viewport transform lifted from focus_hunt_stars.py)
# ==========================================================================
def render_viewport(img, num, den, cx, cy, view_w, view_h):
    sh, sw = img.shape[:2]
    cw = min(sw, max(1, math.ceil(view_w * den / num)))
    ch = min(sh, max(1, math.ceil(view_h * den / num)))
    x0 = max(0, min(int(round(cx - cw / 2.0)), sw - cw))
    y0 = max(0, min(int(round(cy - ch / 2.0)), sh - ch))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    sub = crop[::den, ::den]
    disp = np.repeat(np.repeat(sub, num, axis=0), num, axis=1) if num > 1 else sub
    disp = disp[:view_h, :view_w]
    frame = np.zeros((view_h, view_w, 3), np.uint8)
    dh, dw = disp.shape[:2]
    oy, ox = max(0, (view_h - dh) // 2), max(0, (view_w - dw) // 2)
    frame[oy:oy + dh, ox:ox + dw] = disp
    return frame, x0 + cw / 2.0, y0 + ch / 2.0, cw, ch, (x0, y0, num, den, ox, oy)


class Viewer:
    """Minimal pan/zoom inspector: linear frame + detected stars + M31 arrow."""

    def __init__(self, ctx):
        import tkinter as tk
        self.tk = tk
        self.ctx = ctx                      # dict shared with re-capture
        self._rebuild_from_ctx()

        self.num, self.den = 1, 1
        h, w = self.disp.shape[:2]
        self.cx, self.cy = w / 2.0, h / 2.0
        self.vt = (0, 0, 1, 1, 0, 0)
        self._photo = None

        self.root = tk.Tk()
        self.root.title("find_it")
        self.root.geometry("1500x900")
        bar = tk.Frame(self.root); bar.pack(side="top", fill="x")
        for txt, cmd in [("Zoom - (-)", self.zoom_out), ("Zoom + (+)", self.zoom_in),
                         ("Fit (f)", self.fit), ("1:1 (1)", self.one_to_one),
                         ("Re-shoot (r)", self.reshoot), ("Quit (q)", self._quit)]:
            tk.Button(bar, text=txt, command=cmd, takefocus=0).pack(side="left", padx=1)
        tk.Label(bar, text="  disp gain").pack(side="left")
        self.gain = tk.Scale(bar, from_=1, to=64, orient="horizontal", length=180,
                             takefocus=0, command=lambda _v: self._regain())
        self.gain.set(int(self.ctx.get("gain", 1))); self.gain.pack(side="left")
        self.info = tk.Label(self.root, anchor="w", font=("TkFixedFont", 10))
        self.info.pack(side="top", fill="x")
        self.view = tk.Label(self.root, bg="black"); self.view.pack(fill="both", expand=True)
        self.view.bind("<Configure>", lambda _e: self.render())
        self.root.bind("<Key>", self._key)
        self.one_to_one()

    # -- state ---------------------------------------------------------
    def _rebuild_from_ctx(self):
        self.rgb16 = self.ctx["rgb16"]
        self.stars = self.ctx["stars"]
        self.sol = self.ctx["sol"]
        self.disp = to_display_gray(self.rgb16, float(self.ctx.get("gain", 1)))
        H, W = self.disp.shape[:2]
        self.overlay = []                  # (kind, *src-coords)
        for s in self.stars:
            self.overlay.append(("circ", s["x"], s["y"],
                                 min(6 + 1.5 * math.sqrt(s["area"]), 60), "#ff5050"))
        if self.sol and self.sol.get("ok"):
            start, end, _ = arrow_endpoints(self.sol, W, H)
            self.overlay.append(("arrow", start[0], start[1], end[0], end[1]))
        self._status = self._solve_line()

    def _solve_line(self):
        if not self.sol:
            return "no solve attempted"
        if self.sol.get("ok"):
            s = self.sol
            return (f"SOLVED  centre RA {s['center'][0]:.3f} Dec {s['center'][1]:+.3f}  "
                    f"roll {s['roll_deg']:+.1f}deg  scale {s['scale_arcsec_px']:.2f}\"/px  "
                    f"inliers {s['n_inliers']}  rms {s['rms']:.2f}px  "
                    f"({s['n_detected']} stars detected)")
        return f"NOT SOLVED - {self.sol['reason']}  ({self.sol['n_detected']} stars)"

    # -- nav ---------------------------------------------------------
    def zoom_in(self):
        if self.den > 1: self.den -= 1
        elif self.num < 16: self.num += 1
        self.render()

    def zoom_out(self):
        mx = self._max_den()
        if self.num > 1: self.num -= 1
        elif self.den < mx: self.den += 1
        self.render()

    def _max_den(self):
        sh, sw = self.disp.shape[:2]
        vw, vh = self._vsize()
        return max(1, math.ceil(max(sw / vw, sh / vh)))

    def _vsize(self):
        w, h = self.view.winfo_width(), self.view.winfo_height()
        return (max(64, w), max(64, h)) if w > 1 else (1400, 760)

    def fit(self):
        sh, sw = self.disp.shape[:2]
        self.num, self.den = 1, self._max_den()
        self.cx, self.cy = sw / 2.0, sh / 2.0
        self.render()

    def one_to_one(self):
        self.num, self.den = 1, 1
        self.render()

    def pan(self, dx, dy):
        self.cx += dx * 200; self.cy += dy * 200
        self.render()

    def _regain(self):
        self.ctx["gain"] = float(self.gain.get())
        self.disp = to_display_gray(self.rgb16, self.ctx["gain"])
        self.render()

    def _key(self, e):
        k = e.keysym
        m = {"plus": self.zoom_in, "equal": self.zoom_in, "minus": self.zoom_out,
             "f": self.fit, "1": self.one_to_one, "r": self.reshoot,
             "q": self._quit, "Escape": self._quit,
             "Left": lambda: self.pan(-1, 0), "Right": lambda: self.pan(1, 0),
             "Up": lambda: self.pan(0, -1), "Down": lambda: self.pan(0, 1)}
        if k in m:
            m[k](); return "break"

    # -- render ------------------------------------------------------
    def render(self):
        from PIL import Image, ImageDraw, ImageTk
        vw, vh = self._vsize()
        frame, self.cx, self.cy, _, _, self.vt = render_viewport(
            self.disp, self.num, self.den, self.cx, self.cy, vw, vh)
        im = Image.fromarray(frame)
        dr = ImageDraw.Draw(im)
        for ov in self.overlay:
            if ov[0] == "circ":
                _, x, y, r, col = ov
                vx, vy = self._to_view(x, y)
                rr = r * self.num / self.den
                dr.ellipse([vx - rr, vy - rr, vx + rr, vy + rr], outline=col, width=2)
            else:
                _, x0, y0, x1, y1 = ov
                p0, p1 = self._to_view(x0, y0), self._to_view(x1, y1)
                _draw_arrow(dr, p0, p1, (0, 255, 0), 3)
                dr.text((p1[0] + 6, p1[1] + 6), "M31", fill=(0, 255, 0))
        self._photo = ImageTk.PhotoImage(im)
        self.view.configure(image=self._photo)
        zoom = f"{self.num}:1" if self.den == 1 else f"1:{self.den}"
        self.info.configure(text=f"{self._status}    zoom={zoom}  "
                                 f"(r re-shoot, +/- zoom, f fit, 1 1:1, arrows pan, q quit)")

    def _to_view(self, sx, sy):
        x0, y0, num, den, ox, oy = self.vt
        return (ox + (sx - x0) / den * num, oy + (sy - y0) / den * num)

    # -- re-shoot --------------------------------------------------
    def reshoot(self):
        from tkinter import messagebox
        self.info.configure(text="RE-SHOOTING ..."); self.root.update()
        try:
            raw, name = capture_one(self.ctx["exposure"], self.ctx["iso"])
            rgb16 = decode_raw(raw, name)
        except Exception as e:
            messagebox.showerror("Re-shoot failed", str(e)); return
        stars = detect_stars(rgb16)
        H, W = rgb16.shape[:2]
        sol = solve_field(stars, W, H, self.ctx["cat"], M31_RA_DEG, M31_DEC_DEG,
                          self.ctx["s_arcsec_px"], self.ctx["search_deg"])
        self.ctx.update(rgb16=rgb16, stars=stars, sol=sol)
        self._rebuild_from_ctx()
        self.render()

    def _quit(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ==========================================================================
# catalogue builder (one-off, needs the HYG csv)
# ==========================================================================
def build_catalog(csv_path: str, out: str = CATALOG_NPZ, mag_max: float = 9.5):
    import csv as _csv
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ra, dec, mag, name = [], [], [], []
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["id"] == "0":
                continue
            try:
                m = float(row["mag"])
            except ValueError:
                continue
            if m > mag_max:
                continue
            ra.append(float(row["ra"]) * 15.0)
            dec.append(float(row["dec"]))
            mag.append(m)
            name.append(row["proper"] or "")
    np.savez_compressed(out, ra_deg=np.array(ra), dec_deg=np.array(dec),
                        mag=np.array(mag), name=np.array(name, dtype=object))
    print(f"[cat] wrote {out}  ({len(ra)} stars, mag <= {mag_max})")


# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", metavar="FILE",
                   help="use an existing raw file instead of shooting")
    p.add_argument("--exposure", type=float, default=None,
                   help="shutter in seconds (default: camera's current setting)")
    p.add_argument("--iso", default=None, help="ISO to set before the shot")
    p.add_argument("--focal-mm", type=float, default=DEFAULT_FOCAL_MM)
    p.add_argument("--search-deg", type=float, default=8.0,
                   help="max angle between the frame centre and M31 (default 12)")
    p.add_argument("--catalog", default=CATALOG_NPZ)
    p.add_argument("--gain", type=float, default=1.0,
                   help="linear display gain (does not affect detection/solve)")
    p.add_argument("--no-gui", action="store_true",
                   help="just detect + solve + write the annotated PNG")
    p.add_argument("--out", default=os.path.join(HERE, "find_it_result.png"))
    p.add_argument("--build-catalog", metavar="HYG_CSV",
                   help="one-off: build the star .npz from a HYG csv and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.build_catalog:
        build_catalog(args.build_catalog, args.catalog)
        return 0

    if args.sim:
        with open(args.sim, "rb") as fh:
            raw = fh.read()
        name = os.path.basename(args.sim)
        print(f"[sim] loaded {name} ({len(raw)} bytes)")
    else:
        raw, name = capture_one(args.exposure, args.iso)

    rgb16 = decode_raw(raw, name)
    H, W = rgb16.shape[:2]
    s_arcsec_px = nominal_arcsec_px(W, args.focal_mm)
    print(f"[img] {W}x{H}  nominal scale {s_arcsec_px:.3f}\"/px  "
          f"FOV {W * s_arcsec_px / 3600:.2f} x {H * s_arcsec_px / 3600:.2f} deg")

    t0 = time.time()
    stars = detect_stars(rgb16)
    print(f"[detect] {len(stars)} stars in {time.time() - t0:.2f}s")

    cat = load_catalog(args.catalog)
    t0 = time.time()
    sol = solve_field(stars, W, H, cat, M31_RA_DEG, M31_DEC_DEG,
                      s_arcsec_px, args.search_deg)
    print(f"[solve] {time.time() - t0:.2f}s -> "
          f"{'OK' if sol.get('ok') else 'FAILED: ' + sol['reason']}")
    if sol.get("ok"):
        start, end, inframe = arrow_endpoints(sol, W, H)
        print(f"[solve] centre RA {sol['center'][0]:.4f} Dec {sol['center'][1]:+.4f}  "
              f"roll {sol['roll_deg']:+.2f}deg  parity {sol['parity']:+d}  "
              f"scale {sol['scale_arcsec_px']:.3f}\"/px")
        print(f"[arrow] image centre {tuple(round(v) for v in start)} -> "
              f"{'M31 at ' if inframe else 'border towards M31, '}"
              f"{tuple(round(v) for v in end)}")

    annotate(rgb16, stars, sol, args.gain, args.out)

    if not args.no_gui:
        ctx = dict(rgb16=rgb16, stars=stars, sol=sol, cat=cat,
                   exposure=args.exposure, iso=args.iso,
                   s_arcsec_px=s_arcsec_px, search_deg=args.search_deg,
                   gain=args.gain)
        try:
            Viewer(ctx).run()
        except Exception as e:
            print(f"[gui] viewer unavailable ({e}); PNG was still written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
