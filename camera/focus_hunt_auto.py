#!/usr/bin/env python
"""Fully automatic star-based focus hunt for the Nikon Z6 + 400mm rig.

Scores focus by *how many catalogue stars the frame can identify*: defocus
spreads each star over more pixels, so the faint ones sink below the detection
threshold and the identified count drops.  The count is turned into a metric
that is robust to hot pixels and cosmic rays by requiring every detection to
coincide with a known catalogue star.

Assumptions (promised by the operator, NOT checked):
  * M31 is somewhere in the frame.
  * Nikon Z6 + 400mm lens, tracking mount.
  * The lens starts already close to focus -- this is a fine-tune, not a search.
  * ISO / shutter are whatever the camera is currently set to; we never touch
    them, so the scores are only comparable inside one run.

Nothing is left behind: each frame is deleted from the card after download and
is only held in memory, unless --save-frames DIR is given.

Search
------
Each "burst" shoots 5 frames at focus positions centre-2s .. centre+2s, swept
monotonically upward so every frame is approached from the same direction.

  * best in the middle 3  -> halve the step, re-centre on the best
  * best on either edge   -> keep the step, re-centre on the best (walks)
  * step reaches MIN_STEP (4) and the best is in the middle -> done

A burst is repeated (up to MAX_REPEATS) until one position wins by more than
the measurement noise; if it never does, the hunt stops early, parks on the
best position so far and says so.

Usage
-----
    camera/focus_hunt_auto.py                      # the real hunt
    camera/focus_hunt_auto.py --once               # one burst, no search, no park
    camera/focus_hunt_auto.py --score a.NEF b.NEF  # score files, no camera

    # one-off, ONLINE, to build the deep catalogue patch around M31:
    camera/focus_hunt_auto.py --build-catalog

The deep catalogue (find_it_data/m31_deep.npz, Gaia DR3 out to mag 16.5 in a
7.5 deg cone around M31: ~305k stars, 5.3 MB) is what the metric counts
against; the shallow HYG catalogue shipped with find_it.py is only used for the
initial blind solve.  Building it takes about 20 minutes of archive time.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

import find_it

DEEP_NPZ = os.path.join(find_it.DATA_DIR, "m31_deep.npz")

# ---- catalogue ------------------------------------------------------------
DEEP_MAG = 16.5      # catalogue depth; must stay ahead of what the frame
                     #  reaches (~14.0 at ISO 8000 / 1 s, ~15.3 at 10 s) or the
                     #  count saturates and the metric goes flat near focus
DEEP_RADIUS_DEG = 7.5  # cone around M31: half the frame diagonal (3.1 deg) for
                     #  "M31 anywhere in frame", plus the same again for the
                     #  frame's own extent, plus margin
GAIA_EPOCH = 2016.0   # Gaia DR3 positions are J2016.0; proper motion is
                      #  propagated to OBS_EPOCH when the catalogue is built
OBS_EPOCH = 2026.0
GAIA_JOB_TIMEOUT_S = 45 * 60   # the archive really does take ~20 min for a
                               #  7.5 deg cone; measured 2026-09-18

# ---- detection ------------------------------------------------------------
DET_K = 4.0          # threshold at median + K*sigma. Lower than find_it's 6
                     #  on purpose: the catalogue match downstream throws out
                     #  the junk that comes with the extra depth
MIN_AREA = 3         # smaller groups are single hot pixels / cosmic rays
MAX_AREA = 4000
MAX_ELONG = 3.0      # reject streaks (satellites, tracking glitches)
MAX_DET = 40000

# ---- matching -------------------------------------------------------------
MATCH_R_PX = 3.0     # a catalogue star counts as identified within this radius
FIT_MAG = 11.0       # only stars brighter than this are used to FIT the sky
                     #  transform: they are detected at every focus setting, so
                     #  the fit quality cannot correlate with focus
FIT_TOL_PX = 40.0    # match gate for the FIRST fit iteration. Generous, so a
                     #  frame whose field has crept since the previous one can
                     #  still be picked up; stars brighter than FIT_MAG are
                     #  ~200 px apart, so a 40 px gate still rarely mispairs.
FIT_TOL_TIGHT = 3.0  # gate for the later iterations, so one bad pairing in a
                     #  crowded field cannot drag the least-squares fit
FIT_CLIP_SIGMA = 3.0  # drop pairs beyond this many residual sigmas, then refit
FIT_ITERS = 4
FIT_DET_N = 600      # only the brightest this many detections may take part in
                     #  the fit. Faint junk hugely outnumbers the bright
                     #  catalogue stars, and letting it into the fit is how
                     #  wrong pairings get in.
SHUFFLE_PX = 80.0    # catalogue offset used to measure the accidental-match
                     #  rate: every match found after this shift is a coincidence
REF_INSET_PX = 60    # the scored sky region is the first frame's field inset
                     #  by this much. The tracker is imperfect, so the field
                     #  creeps during a hunt; the inset lets it creep this far
                     #  before any of the scored patch leaves the sensor (60 px
                     #  is 3 arcmin at 3 arcsec/px), and beyond that the score
                     #  is scaled back up by the fraction still on the sensor.
                     #  Kept small on purpose: drift does not bias the count
                     #  (the counted area is the same wherever the frame
                     #  points), so a big inset would only throw away stars and
                     #  make the metric noisier.
ZP_MAG_LO, ZP_MAG_HI = 10.0, 13.0   # magnitude window for the photometric zero
                     #  point (brighter stars saturate, fainter ones are noisy)

# ---- search ---------------------------------------------------------------
N_POS = 5            # frames per burst
CENTRE_IDX = 2
DEFAULT_STEP = 32    # matches focus_hunt.py's INITIAL_STEP
MIN_STEP = 4         # the focus motor no-ops below ~4 units
MAX_REPEATS = 5      # bursts at one (centre, step) before giving up
WIN_Z = 2.0          # winner must lead by this many combined standard errors
WIN_REL_MARGIN = 0.003  # ...and by at least this fraction of the metric, so a
                     #  freakishly small sigma cannot crown a trivial lead
SETTLE_S = 0.35      # let the optics settle after a focus move (as focus_hunt)
CAP_SETTLE_S = 1.0   # let live view resume after the shutter (as focus_hunt_stars)


# ==========================================================================
# deep catalogue
# ==========================================================================
def build_deep_catalog(out: str = DEEP_NPZ, mag_max: float = DEEP_MAG,
                       radius_deg: float = DEEP_RADIUS_DEG) -> None:
    """Fetch a Gaia DR3 cone around M31 and store it as an npz. Needs network.

    Gaia G is not Johnson V, but it is within a few tenths for ordinary stars
    and we only ever use the magnitudes as a relative depth scale, so the
    difference does not matter here.
    """
    import csv as _csv
    import io as _io

    import requests

    q = (f"SELECT ra, dec, pmra, pmdec, phot_g_mean_mag "
         f"FROM gaiadr3.gaia_source "
         f"WHERE phot_g_mean_mag < {mag_max} "
         f"AND 1=CONTAINS(POINT('ICRS', ra, dec), "
         f"CIRCLE('ICRS', {find_it.M31_RA_DEG}, {find_it.M31_DEC_DEG}, "
         f"{radius_deg}))")
    base = "https://gea.esac.esa.int/tap-server/tap/async"
    print(f"[cat] submitting Gaia DR3 cone: r={radius_deg} deg, G<{mag_max}")
    r = requests.post(base, data={"REQUEST": "doQuery", "LANG": "ADQL",
                                  "FORMAT": "csv", "PHASE": "RUN",
                                  "QUERY": q}, timeout=60)
    r.raise_for_status()
    job = r.url.split("?")[0]
    print(f"[cat] job {job.rsplit('/', 1)[-1]}")
    print(f"[cat] a {radius_deg} deg cone takes the archive on the order of "
          f"20 minutes -- this is normal, not a hang")
    t0 = time.time()
    deadline = t0 + GAIA_JOB_TIMEOUT_S
    last = 0.0
    while time.time() < deadline:
        ph = requests.get(job + "/phase", timeout=30).text.strip()
        if ph in ("COMPLETED", "ERROR", "ABORTED"):
            break
        now = time.time()
        if now - last >= 15.0:           # so a long wait does not look dead
            print(f"[cat] {ph.lower()}, {now - t0:.0f}s elapsed", flush=True)
            last = now
        time.sleep(2.0)
    else:
        raise RuntimeError(
            f"Gaia job still {ph} after {GAIA_JOB_TIMEOUT_S / 60:.0f} minutes. "
            f"It may yet finish -- the job is at {job}")
    if ph != "COMPLETED":
        raise RuntimeError(f"Gaia job ended in phase {ph}: "
                           f"{requests.get(job, timeout=30).text[:2000]}")
    print(f"[cat] completed in {time.time() - t0:.0f}s, downloading")
    body = requests.get(job + "/results/result", timeout=600).text

    ra, dec, mag = [], [], []
    dt = OBS_EPOCH - GAIA_EPOCH
    for row in _csv.DictReader(_io.StringIO(body)):
        try:
            a, d, m = float(row["ra"]), float(row["dec"]), \
                float(row["phot_g_mean_mag"])
        except (ValueError, KeyError):
            continue
        # propagate proper motion (mas/yr) to the observing epoch; only the
        # handful of very high-pm stars move enough to matter at 3 arcsec/px
        try:
            pa, pd = float(row["pmra"]), float(row["pmdec"])
        except (ValueError, KeyError, TypeError):
            pa = pd = 0.0
        if np.isfinite(pa) and np.isfinite(pd):
            d = d + pd * dt / 3.6e6
            a = a + pa * dt / 3.6e6 / max(1e-6, math.cos(math.radians(d)))
        ra.append(a)
        dec.append(d)
        mag.append(m)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, ra_deg=np.array(ra), dec_deg=np.array(dec),
                        mag=np.array(mag, dtype=np.float32))
    size_mb = os.path.getsize(out) / 1e6
    print(f"[cat] wrote {out}  ({len(ra)} stars, G <= {mag_max}, "
          f"{size_mb:.1f} MB, epoch J{OBS_EPOCH:.0f})")


def load_deep_catalog(path: str = DEEP_NPZ):
    if not os.path.exists(path):
        raise RuntimeError(
            f"deep star catalogue not found: {path}\n"
            f"build it once (needs network) with:\n"
            f"    {sys.argv[0]} --build-catalog")
    d = np.load(path)
    return (d["ra_deg"].astype(float), d["dec_deg"].astype(float),
            d["mag"].astype(float))


# ==========================================================================
# detection
# ==========================================================================
def detect_stars(gray: np.ndarray, k: float = DET_K) -> dict:
    """Find star-like blobs. Returns arrays x, y, flux, area, peak.

    Same shape of algorithm as find_it.detect_stars, but vectorised over the
    above-threshold pixels only (there are thousands of blobs here, not 80, and
    scipy's per-label helpers get slow) and with a tunable threshold.
    """
    import scipy.ndimage as ndi

    h, w = gray.shape

    # coarse local background: median on a decimated grid, bilinearly upsampled.
    # Needed because the sky has a gradient, the lens vignettes, and M31 itself
    # is a large diffuse patch that would otherwise read as one huge blob.
    f = 32
    small = gray[::f, ::f]
    bg_small = ndi.median_filter(small, size=5, mode="nearest")
    bg = ndi.zoom(bg_small, (h / bg_small.shape[0], w / bg_small.shape[1]),
                  order=1)[:h, :w]
    resid = gray - bg

    # robust noise level: MAD rather than std, because stars are a tiny
    # fraction of the pixels but enormous in value and would inflate a std
    samp = resid[::7, ::7]
    med = float(np.median(samp))
    sigma = 1.4826 * (float(np.median(np.abs(samp - med))) + 1e-6)
    thr = med + k * sigma

    mask = resid > thr
    lbl, n = ndi.label(mask)
    empty = dict(x=np.zeros(0), y=np.zeros(0), flux=np.zeros(0),
                 area=np.zeros(0, int), peak=np.zeros(0), sigma=sigma, thr=thr)
    if n == 0:
        return empty

    flat = np.flatnonzero(mask.ravel())
    lab = lbl.ravel()[flat]
    val = resid.ravel()[flat].astype(np.float64)
    xs = (flat % w).astype(np.float64)
    ys = (flat // w).astype(np.float64)

    nb = n + 1
    area = np.bincount(lab, minlength=nb).astype(np.int64)
    flux = np.bincount(lab, weights=val, minlength=nb)
    sx = np.bincount(lab, weights=val * xs, minlength=nb)
    sy = np.bincount(lab, weights=val * ys, minlength=nb)
    peak = np.zeros(nb)
    np.maximum.at(peak, lab, val)
    xmin = np.full(nb, np.inf)
    xmax = np.full(nb, -np.inf)
    ymin = np.full(nb, np.inf)
    ymax = np.full(nb, -np.inf)
    np.minimum.at(xmin, lab, xs)
    np.maximum.at(xmax, lab, xs)
    np.minimum.at(ymin, lab, ys)
    np.maximum.at(ymax, lab, ys)

    with np.errstate(invalid="ignore", divide="ignore"):
        cx = sx / flux
        cy = sy / flux
    bw = xmax - xmin + 1.0
    bh = ymax - ymin + 1.0
    elong = np.maximum(bw, bh) / np.maximum(1.0, np.minimum(bw, bh))

    keep = np.ones(nb, bool)
    keep[0] = False
    keep &= (area >= MIN_AREA) & (area <= MAX_AREA)
    keep &= np.isfinite(cx) & np.isfinite(cy) & (flux > 0)
    keep &= ~((elong > MAX_ELONG) & (area > 25))
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return empty
    if idx.size > MAX_DET:                      # keep the brightest
        idx = idx[np.argsort(-flux[idx])[:MAX_DET]]

    order = idx[np.argsort(-flux[idx])]
    return dict(x=cx[order], y=cy[order], flux=flux[order],
                area=area[order], peak=peak[order], sigma=sigma, thr=thr)


# ==========================================================================
# sky transform
# ==========================================================================
def _project(ra_deg, dec_deg, center, s_deg):
    """Catalogue positions -> tangent-plane pixels about `center`."""
    xi, eta = find_it.radec_to_tan(np.radians(ra_deg), np.radians(dec_deg),
                                   math.radians(center[0]),
                                   math.radians(center[1]))
    return np.column_stack([np.degrees(xi) / s_deg, np.degrees(eta) / s_deg])


def refit_wcs(det: dict, deep, sol: dict, w: int, h: int) -> dict | None:
    """Re-fit the pixel<->sky transform of one frame, seeded from `sol`.

    The mount tracks and the camera does not move, so the previous frame's
    solution is always a good starting point and no blind solve is needed.

    Only catalogue stars brighter than FIT_MAG take part.  Those are far above
    the detection limit at every focus setting we try, so the fit is equally
    well determined in a sharp and a blurred frame -- if faint stars were
    allowed in, the transform would be better in the sharp frames and that
    would leak into the very quantity we are measuring.
    """
    from scipy.spatial import cKDTree

    ra, dec, mag = deep
    s_deg = sol["s_deg"]
    fov_deg = math.hypot(w, h) * s_deg
    center = tuple(sol["center"])
    a_mat = np.array(sol["A"], float)
    t_vec = np.array(sol["t"], float)

    if det["x"].size < 8:
        return None
    # detections arrive sorted brightest first, so the head of the array is the
    # bright subset that corresponds to the catalogue stars used for the fit
    p_cen = np.column_stack([det["x"], det["y"]]) - np.array([w / 2.0, h / 2.0])
    p_fit = p_cen[:FIT_DET_N]

    for it in range(FIT_ITERS):
        near = (find_it.angsep_deg(ra, dec, center[0], center[1])
                < fov_deg / 2.0 + 0.5)
        sel = np.flatnonzero(near & (mag <= FIT_MAG))
        if sel.size < find_it.MIN_INLIERS:
            return None
        q_cat = _project(ra[sel], dec[sel], center, s_deg)
        tol = FIT_TOL_PX if it == 0 else FIT_TOL_TIGHT
        # Ask each CATALOGUE star for its nearest detection, not the other way
        # round. Detections outnumber the bright catalogue stars by an order of
        # magnitude, so querying per detection would pair most of the faint
        # ones to some bright catalogue star within the gate and wreck the fit.
        dist, ii = cKDTree(p_fit @ a_mat.T + t_vec).query(
            q_cat, distance_upper_bound=tol)
        good = np.isfinite(dist)
        if good.sum() < find_it.MIN_INLIERS:
            return None
        di = ii[good]
        cj = sel[good]

        # fit at the current centre, move the centre onto the fitted origin,
        # then fit again there so the tangent point stays near the field centre
        a_mat, t_vec = find_it._fit_affine(p_fit[di],
                                           _project(ra[cj], dec[cj], center,
                                                    s_deg))

        # sigma-clip: a dense field offers plenty of wrong nearest neighbours,
        # and a single one at the tolerance limit biases a least-squares fit
        r = (p_fit[di] @ a_mat.T + t_vec
             - _project(ra[cj], dec[cj], center, s_deg))
        d = np.hypot(r[:, 0], r[:, 1])
        cut = FIT_CLIP_SIGMA * 1.4826 * (np.median(np.abs(d - np.median(d))) + 1e-6)
        keep = d <= np.median(d) + cut
        if keep.sum() >= find_it.MIN_INLIERS and keep.sum() < di.size:
            di, cj = di[keep], cj[keep]
            a_mat, t_vec = find_it._fit_affine(
                p_fit[di], _project(ra[cj], dec[cj], center, s_deg))
        ra_c, dec_c = find_it.tan_to_radec(math.radians(t_vec[0] * s_deg),
                                           math.radians(t_vec[1] * s_deg),
                                           math.radians(center[0]),
                                           math.radians(center[1]))
        center = (float(np.degrees(ra_c)) % 360.0, float(np.degrees(dec_c)))
        a_mat, t_vec = find_it._fit_affine(p_fit[di],
                                           _project(ra[cj], dec[cj], center,
                                                    s_deg))

    resid = p_fit[di] @ a_mat.T + t_vec - _project(ra[cj], dec[cj], center,
                                                   s_deg)
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    return {"A": a_mat, "t": t_vec, "center": center, "s_deg": s_deg,
            "n_fit": int(di.size), "rms": rms}


# ==========================================================================
# matching + score
# ==========================================================================
def match_one_to_one(det_xy: np.ndarray, cat_xy: np.ndarray, r: float):
    """Greedy closest-first matching; no detection or catalogue star is reused.

    One-to-one matters: a defocused frame merges close stars into one fat blob,
    and a nearest-neighbour-per-catalogue-star match would let that single blob
    be claimed by several catalogue entries -- inflating the score exactly for
    the frames that deserve the lowest one.
    """
    from scipy.spatial import cKDTree

    if det_xy.shape[0] == 0 or cat_xy.shape[0] == 0:
        return np.zeros(0, int), np.zeros(0, int)
    tree = cKDTree(cat_xy)
    cand = []
    for i, js in enumerate(tree.query_ball_point(det_xy, r)):
        for j in js:
            cand.append((float(np.hypot(*(det_xy[i] - cat_xy[j]))), i, j))
    cand.sort()
    used_d = np.zeros(det_xy.shape[0], bool)
    used_c = np.zeros(cat_xy.shape[0], bool)
    di, cj = [], []
    for _, i, j in cand:
        if used_d[i] or used_c[j]:
            continue
        used_d[i] = used_c[j] = True
        di.append(i)
        cj.append(j)
    return np.array(di, int), np.array(cj, int)


def limiting_mag(cat_mag: np.ndarray, matched: np.ndarray, acc_rate: float,
                 bin_w: float = 0.5, frac: float = 0.5) -> float:
    """Magnitude at which the identified fraction falls through `frac`.

    Less sensitive than a raw count to sky transparency changing between
    bursts, because a transparency change moves every star's brightness
    together and so only shifts the curve sideways by a small amount.
    """
    if cat_mag.size == 0:
        return float("nan")
    lo = math.floor(float(cat_mag.min()))
    hi = math.ceil(float(cat_mag.max()))
    edges = np.arange(lo, hi + bin_w, bin_w)
    which = np.digitize(cat_mag, edges) - 1
    centres, fracs = [], []
    for b in range(len(edges) - 1):
        m = which == b
        tot = int(m.sum())
        if tot < 20:                      # too few to estimate a fraction
            continue
        got = float(matched[m].sum()) - acc_rate * tot
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        fracs.append(max(0.0, got) / tot)
    if len(centres) < 2:
        return float("nan")
    centres = np.array(centres)
    fracs = np.array(fracs)
    for i in range(1, len(fracs)):
        if fracs[i] < frac <= fracs[i - 1]:
            span = fracs[i - 1] - fracs[i]
            w = (fracs[i - 1] - frac) / span if span > 0 else 0.0
            return float(centres[i - 1] + w * (centres[i] - centres[i - 1]))
    # never crossed: either everything was found (catalogue too shallow) or
    # nothing was (frame far out of focus)
    return float(centres[-1]) if fracs[-1] >= frac else float(centres[0])


def score_frame(gray: np.ndarray, deep, sol_seed: dict, k: float = DET_K,
                match_r: float = MATCH_R_PX, ref: np.ndarray | None = None
                ) -> dict:
    """Detect, re-fit the transform, match against the catalogue, score.

    `ref` is the fixed set of catalogue stars to score against, chosen once on
    the first frame of a run.  Pinning it matters because the mount does not
    track perfectly: the field creeps across the sensor during a hunt, so the
    set of stars that happen to fall inside the frame changes from frame to
    frame.  Counting a *fixed* patch of sky instead means the denominator is
    identical everywhere and only focus can move the score.  The returned
    "ref" is the set to pass to every later frame.
    """
    h, w = gray.shape
    det = detect_stars(gray, k)
    out = {"ok": False, "reason": "", "n_det": int(det["x"].size),
           "sigma": det["sigma"]}
    if det["x"].size < 8:
        out["reason"] = f"only {det['x'].size} detections"
        return out

    sol = refit_wcs(det, deep, sol_seed, w, h)
    if sol is None:
        out["reason"] = "could not re-fit the sky transform"
        return out

    ra, dec, mag = deep
    px = find_it.radec_to_pixel(sol, ra, dec, w, h)
    if ref is None:
        # first frame of the run: the scored patch is this frame inset by
        # REF_INSET_PX, so later frames can drift by that much and still
        # contain all of it
        m = ((px[:, 0] > REF_INSET_PX) & (px[:, 0] < w - REF_INSET_PX) &
             (px[:, 1] > REF_INSET_PX) & (px[:, 1] < h - REF_INSET_PX))
        ref = np.flatnonzero(m)
    if ref.size < 50:
        out["reason"] = f"only {ref.size} catalogue stars in the scored patch"
        return out

    # how much of the fixed patch is still on the sensor after the drift
    ref_xy = px[ref]
    on = ((ref_xy[:, 0] >= 0) & (ref_xy[:, 0] < w) &
          (ref_xy[:, 1] >= 0) & (ref_xy[:, 1] < h))
    n_on = int(on.sum())
    if n_on < 50:
        out["reason"] = "the scored sky patch has drifted off the sensor"
        return out
    cat_xy = ref_xy[on]
    cat_mag = mag[ref][on]
    det_xy = np.column_stack([det["x"], det["y"]])

    di, cj = match_one_to_one(det_xy, cat_xy, match_r)

    # accidental-match control: shift the whole catalogue and match again.
    # Every match found now is a coincidence, so this measures the false
    # identification rate directly instead of assuming it.
    shift = np.array([SHUFFLE_PX, SHUFFLE_PX])
    _, acc_cj = match_one_to_one(det_xy, cat_xy + shift, match_r)
    n_acc = int(acc_cj.size)
    acc_rate = n_acc / float(cat_xy.shape[0])

    matched = np.zeros(cat_xy.shape[0], bool)
    matched[cj] = True
    n_match = int(cj.size)

    # unmatched detections, counted only over the scored patch -- detections
    # outside it are mostly real stars with no catalogue entry in the patch,
    # and counting those would make the false rate look far worse than it is
    x0, x1 = cat_xy[:, 0].min(), cat_xy[:, 0].max()
    y0, y1 = cat_xy[:, 1].min(), cat_xy[:, 1].max()
    in_patch = ((det_xy[:, 0] >= x0) & (det_xy[:, 0] <= x1) &
                (det_xy[:, 1] >= y0) & (det_xy[:, 1] <= y1))
    matched_det = np.zeros(det_xy.shape[0], bool)
    matched_det[di] = True
    n_false = int((in_patch & ~matched_det).sum())

    # if part of the patch has drifted off, scale back up to the full patch so
    # the score stays comparable with the frames that held all of it
    drift_corr = ref.size / float(n_on)

    # photometric zero point: catalogue mag minus instrumental mag, over a
    # window that avoids saturated bright stars and noisy faint ones. Its
    # value tracks sky transparency; its scatter is a health check.
    zp = scat = float("nan")
    if di.size:
        inst = -2.5 * np.log10(np.maximum(det["flux"][di], 1e-9))
        win = (cat_mag[cj] >= ZP_MAG_LO) & (cat_mag[cj] <= ZP_MAG_HI)
        if win.sum() >= 10:
            d = cat_mag[cj][win] - inst[win]
            zp = float(np.median(d))
            scat = float(1.4826 * np.median(np.abs(d - zp)))

    out.update(ok=True, n_match=n_match, n_acc=n_acc,
               score=float(n_match - n_acc) * drift_corr,
               n_ref=int(ref.size), n_on=n_on, drift_corr=drift_corr,
               n_false=n_false,
               lim_mag=limiting_mag(cat_mag, matched, acc_rate),
               zp=zp, zp_scatter=scat, rms=sol["rms"], n_fit=sol["n_fit"],
               sol=sol, ref=ref)
    return out


# ==========================================================================
# frame acquisition / decoding
# ==========================================================================
def gray_from_raw(raw: bytes, name: str) -> np.ndarray:
    rgb16 = find_it.decode_raw(raw, name)
    # mean with an explicit dtype: the obvious .astype(float32).mean() would
    # first materialise a 300 MB copy of the full-resolution three-channel frame
    return rgb16.mean(axis=2, dtype=np.float32)


def initial_solve(gray: np.ndarray, focal_mm: float, search_deg: float) -> dict:
    """Blind-solve the first frame with find_it's shallow HYG catalogue.

    The blind solver forms hypotheses from the brightest stars, so it wants a
    sparse catalogue; the deep one would drown it in candidates. After this one
    solve every later frame is handled by refit_wcs.
    """
    h, w = gray.shape
    d = detect_stars(gray, k=6.0)          # find_it's own threshold
    n = min(80, d["x"].size)
    stars = [dict(x=float(d["x"][i]), y=float(d["y"][i]),
                  flux=float(d["flux"][i]), area=int(d["area"][i]),
                  peak=float(d["peak"][i])) for i in range(n)]
    cat = find_it.load_catalog()
    s = find_it.nominal_arcsec_px(w, focal_mm)
    sol = find_it.solve_field(stars, w, h, cat, find_it.M31_RA_DEG,
                              find_it.M31_DEC_DEG, s, search_deg)
    return sol if sol.get("ok") else None


# ==========================================================================
# burst + winner test
# ==========================================================================
def positions_for(centre: int, step: int) -> list[int]:
    return [centre + (i - CENTRE_IDX) * step for i in range(N_POS)]


def noise_sigma(samples: dict) -> tuple[float, int]:
    """Estimate the per-measurement noise, and say how many repeats it used.

    With repeats, pool the within-position spread -- that is the real
    frame-to-frame noise (seeing, transparency). With a single burst there is
    no such spread, so fall back to the scatter of the 5 points about a fitted
    parabola: the focus response is smooth, so whatever does not lie on that
    curve is noise. Only 2 degrees of freedom, hence the WIN_REL_MARGIN floor.
    """
    pos = sorted(samples)
    n_rep = min(len(samples[p]) for p in pos)
    if n_rep >= 2:
        var, dof = 0.0, 0
        for p in pos:
            v = np.asarray(samples[p], float)
            var += float(((v - v.mean()) ** 2).sum())
            dof += len(v) - 1
        return (math.sqrt(var / dof) if dof else float("inf")), n_rep
    x = np.array(pos, float)
    y = np.array([samples[p][0] for p in pos], float)
    if len(x) < 4:
        return float("inf"), n_rep
    c = np.polyfit(x - x.mean(), y, 2)
    r = y - np.polyval(c, x - x.mean())
    return float(np.sqrt((r ** 2).sum() / max(1, len(x) - 3))), n_rep


def pick_winner(samples: dict):
    """Return (winning position, mean, sigma) or None if it is not decided."""
    pos = sorted(samples)
    if len(pos) < 2:
        return None
    means = {p: float(np.mean(samples[p])) for p in pos}
    order = sorted(pos, key=lambda p: -means[p])
    best, second = order[0], order[1]
    sigma, n_rep = noise_sigma(samples)
    if not math.isfinite(sigma):
        return None
    se = sigma * math.sqrt(2.0 / max(1, n_rep))
    lead = means[best] - means[second]
    floor = WIN_REL_MARGIN * max(1.0, abs(means[best]))
    if lead > max(WIN_Z * se, floor):
        return best, means[best], sigma
    return None


def pick_tie(samples: dict):
    """Two neighbouring positions that lead the field but not each other.

    This is not a failure -- it is the optimum sitting between two grid
    positions, which repeating the burst can never resolve because the two are
    genuinely equal. Recognising it lets the search re-centre on the midpoint
    instead of burning every remaining repeat on an undecidable comparison.
    """
    pos = sorted(samples)
    if len(pos) < 3:
        return None
    means = {p: float(np.mean(samples[p])) for p in pos}
    order = sorted(pos, key=lambda p: -means[p])
    sigma, n_rep = noise_sigma(samples)
    if n_rep < 2 or not math.isfinite(sigma):
        return None                       # one burst cannot establish equality
    se = sigma * math.sqrt(2.0 / n_rep)
    a, b, c = order[0], order[1], order[2]
    if abs(pos.index(a) - pos.index(b)) != 1:
        return None                       # only neighbours can bracket a peak
    if means[a] - means[b] > se:
        return None                       # a is pulling ahead; not a tie
    if means[b] - means[c] < WIN_Z * se:
        return None                       # the pair is not clear of the rest
    return (min(a, b), max(a, b))


def print_burst(samples: dict, extra: dict, centre: int, step: int,
                rep: int) -> None:
    print(f"\n[burst] centre={centre:+d} step={step} repeat={rep}")
    print("   pos     N      mean   lim_mag    zp   scat   false  n_det")
    for p in sorted(samples):
        e = extra.get(p, {})
        vals = samples[p]
        print(f"  {p:+5d} {vals[-1]:7.0f} {np.mean(vals):9.1f} "
              f"{e.get('lim_mag', float('nan')):8.2f} "
              f"{e.get('zp', float('nan')):6.2f} "
              f"{e.get('zp_scatter', float('nan')):6.2f} "
              f"{e.get('n_false', -1):7d} {e.get('n_det', -1):6d}")
    sigma, n_rep = noise_sigma(samples)
    print(f"  noise sigma={sigma:.1f} (from {n_rep} repeat(s))")


# ==========================================================================
# the hunt
# ==========================================================================
class Hunt:
    def __init__(self, cam, deep, args, out_dir: str):
        self.cam = cam
        self.deep = deep
        self.args = args
        self.out_dir = out_dir
        self.sol = None            # last good sky transform, seeds the next frame
        self.ref = None            # fixed set of catalogue stars being scored
        self.shots = 0

    # -- one frame ---------------------------------------------------------
    def measure(self, pos: int, tag: str) -> dict | None:
        self.cam.move_to(pos)
        time.sleep(SETTLE_S)
        raw, name = self.cam.capture_raw()
        self.shots += 1
        gray = gray_from_raw(raw, name)
        time.sleep(CAP_SETTLE_S)

        if self.sol is None:
            self.sol = initial_solve(gray, self.args.focal,
                                     self.args.search_deg)
            if self.sol is None:
                print(f"[warn] {tag}: blind solve failed on this frame")
                self._save(raw, name, tag, None)
                return None
            print(f"[solve] locked: centre=({self.sol['center'][0]:.3f}, "
                  f"{self.sol['center'][1]:+.3f}) inliers={self.sol['n_inliers']} "
                  f"rms={self.sol['rms']:.2f}px")

        res = score_frame(gray, self.deep, self.sol, self.args.k,
                          self.args.match_r, self.ref)
        self._save(raw, name, tag, res)
        if not res["ok"]:
            print(f"[warn] {tag}: {res['reason']}")
            return None
        self.sol = res["sol"]          # carry the transform to the next frame
        if self.ref is None:
            self.ref = res["ref"]
            print(f"[patch] scoring a fixed sky patch of {self.ref.size} "
                  f"catalogue stars (frame inset by {REF_INSET_PX} px)")
        elif res["n_on"] < res["n_ref"]:
            print(f"[warn] {tag}: the field has drifted -- {res['n_ref'] - res['n_on']} "
                  f"of {res['n_ref']} patch stars are off the sensor "
                  f"(score scaled by {res['drift_corr']:.3f}); if this keeps "
                  f"growing, re-centre the mount")
        return res

    def _save(self, raw: bytes, name: str, tag: str, res) -> None:
        if not self.out_dir:
            return
        ext = os.path.splitext(name or "")[1] or ".nef"
        s = f"_n{res['score']:.0f}" if res and res.get("ok") else "_fail"
        p = os.path.join(self.out_dir, f"{tag}{s}{ext}")
        try:
            with open(p, "wb") as fh:
                fh.write(raw)
        except OSError as e:
            print(f"[warn] could not save {p}: {e}")

    # -- one burst of N_POS frames -----------------------------------------
    def burst(self, centre: int, step: int, rep: int, samples: dict,
              extra: dict) -> bool:
        """Sweep the 5 positions low->high so all are approached alike."""
        pos = sorted(positions_for(centre, step))
        # Drop below the first position before starting the sweep. The previous
        # burst left the lens at its own highest position, so without this the
        # first frame would be reached by a downward move and the other four by
        # upward ones -- and with any backlash that makes it not comparable.
        self.cam.move_to(pos[0] - 2 * step)
        time.sleep(SETTLE_S)
        got = 0
        for p in pos:
            tag = f"s{step:03d}_c{centre:+05d}_r{rep}_p{p:+05d}"
            res = self.measure(p, tag)
            if res is None:
                continue
            samples.setdefault(p, []).append(res["score"])
            extra[p] = res
            got += 1
        return got == N_POS

    # -- the search --------------------------------------------------------
    def run(self, centre: int, step: int) -> tuple[int, bool]:
        while True:
            samples: dict[int, list[float]] = {}
            extra: dict[int, dict] = {}
            win = tie = None
            for rep in range(1, self.args.repeats + 1):
                if not self.burst(centre, step, rep, samples, extra):
                    print("[warn] burst incomplete -- some frames did not score")
                if len(samples) < N_POS:
                    print("[warn] fewer than 5 positions scored; repeating")
                    continue
                print_burst(samples, extra, centre, step, rep)
                win = pick_winner(samples)
                if win is not None:
                    break
                tie = pick_tie(samples)
                if tie is not None:
                    break

            if tie is not None:
                mid = (tie[0] + tie[1]) // 2
                print(f"[round] {tie[0]:+d} and {tie[1]:+d} lead the field but "
                      f"not each other -> the optimum is between them")
                if step <= MIN_STEP:
                    print(f"[round] at the finest step ({MIN_STEP}) -> take the "
                          f"midpoint {mid:+d} and finish")
                    return mid, True
                centre = mid
                step = max(MIN_STEP, step // 2)
                print(f"[round] re-centre on {mid:+d}, step now {step}")
                continue

            if win is None:
                if not samples:
                    raise RuntimeError("no position produced a usable score")
                best = max(samples, key=lambda p: float(np.mean(samples[p])))
                print(f"\n[stop] no position won after {self.args.repeats} "
                      f"repeats at step={step}. PREMATURE FINISH: the focus is "
                      f"only resolved to about +-{step} units.")
                return best, False

            best, mean, sigma = win
            idx = positions_for(centre, step).index(best)
            print(f"[round] winner pos={best:+d} mean={mean:.1f} "
                  f"sigma={sigma:.1f} (index {idx} of 0..4)")

            if idx in (0, N_POS - 1):
                print(f"[round] winner on the edge -> re-centre on {best:+d}, "
                      f"step stays {step}")
                centre = best
                continue
            if step <= MIN_STEP:
                print(f"[round] winner in the middle at the finest step "
                      f"({MIN_STEP}) -> done")
                return best, True
            centre = best
            step = max(MIN_STEP, step // 2)
            print(f"[round] winner in the middle -> re-centre on {best:+d}, "
                  f"step now {step}")

    def park(self, pos: int, step: int) -> None:
        """Park on `pos`, approached from below like every measured frame was.

        If the lens has any backlash, a position reached from above is not the
        same optical focus as the one that was measured on the way up.
        """
        self.cam.move_to(pos - 2 * max(step, MIN_STEP))
        time.sleep(SETTLE_S)
        self.cam.move_to(pos)
        time.sleep(SETTLE_S)
        print(f"[done] focus parked at tracked position {pos:+d} "
              f"(cam.pos={self.cam.pos:+d}), {self.shots} frames shot")


# ==========================================================================
# offline scoring
# ==========================================================================
def score_files(paths: list[str], args) -> int:
    deep = load_deep_catalog(args.catalog)
    sol = ref = None
    print("  file                       N    n_acc  lim_mag    zp   scat  "
          "false  n_det   rms")
    for p in paths:
        with open(p, "rb") as fh:
            raw = fh.read()
        gray = gray_from_raw(raw, os.path.basename(p))
        if sol is None:
            sol = initial_solve(gray, args.focal, args.search_deg)
            if sol is None:
                print(f"[fatal] blind solve failed on {p}")
                return 2
            print(f"[solve] centre=({sol['center'][0]:.3f}, "
                  f"{sol['center'][1]:+.3f}) inliers={sol['n_inliers']} "
                  f"rms={sol['rms']:.2f}px")
        res = score_frame(gray, deep, sol, args.k, args.match_r, ref)
        if not res["ok"]:
            print(f"  {os.path.basename(p)[:24]:24s} FAILED: {res['reason']}")
            continue
        sol = res["sol"]
        ref = res["ref"]
        print(f"  {os.path.basename(p)[:24]:24s} {res['score']:7.0f} "
              f"{res['n_acc']:6d} {res['lim_mag']:8.2f} {res['zp']:6.2f} "
              f"{res['zp_scatter']:6.2f} {res['n_false']:6d} "
              f"{res['n_det']:6d} {res['rms']:5.2f}")
    return 0


# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", type=int, default=DEFAULT_STEP,
                   help=f"initial focus step (default {DEFAULT_STEP})")
    p.add_argument("--repeats", type=int, default=MAX_REPEATS,
                   help=f"max bursts per round (default {MAX_REPEATS})")
    p.add_argument("--k", type=float, default=DET_K,
                   help=f"detection threshold in sigma (default {DET_K})")
    p.add_argument("--match-r", type=float, default=MATCH_R_PX,
                   help=f"match radius in px (default {MATCH_R_PX})")
    p.add_argument("--focal", type=float, default=find_it.DEFAULT_FOCAL_MM)
    p.add_argument("--search-deg", type=float, default=4.0,
                   help="how far the frame centre may be from M31 (default 4)")
    p.add_argument("--catalog", default=DEEP_NPZ)
    p.add_argument("--save-frames", metavar="DIR", default=None,
                   help="keep the captured frames in DIR (default: keep "
                        "nothing -- frames are deleted from the card by the "
                        "capture and never written to disk here)")
    p.add_argument("--once", action="store_true",
                   help="shoot one burst at the current focus, print the "
                        "scores and stop -- no search, no focus left changed")
    p.add_argument("--score", nargs="+", metavar="FILE",
                   help="score existing raw files and exit (no camera)")
    p.add_argument("--build-catalog", action="store_true",
                   help="fetch the deep Gaia catalogue patch (needs network)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.build_catalog:
        build_deep_catalog(args.catalog)
        return 0
    if args.score:
        return score_files(args.score, args)

    deep = load_deep_catalog(args.catalog)
    print(f"[cat] {len(deep[0])} catalogue stars, "
          f"G <= {deep[2].max():.1f}")

    # imported late: focus_hunt pulls in cv2 at module level, which this
    # script itself never needs (the offline paths above must work without it)
    from focus_hunt import FocusCamera

    out_dir = args.save_frames
    if out_dir:
        out_dir = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        print(f"[info] frames -> {out_dir}/")

    cam = FocusCamera()
    cam.open()
    try:
        cam.start_liveview()     # Nikon only accepts focus drive in live view
        hunt = Hunt(cam, deep, args, out_dir)
        if args.once:
            samples, extra = {}, {}
            hunt.burst(0, args.step, 1, samples, extra)
            if samples:
                print_burst(samples, extra, 0, args.step, 1)
            hunt.park(0, args.step)        # back where we started
            return 0
        best, clean = hunt.run(0, args.step)
        hunt.park(best, MIN_STEP)
        if not clean:
            print("[warn] the result above is the best seen, not a resolved "
                  "optimum -- consider re-running")
        return 0 if clean else 1
    finally:
        try:
            cam.stop_liveview()
        except Exception:
            pass
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
