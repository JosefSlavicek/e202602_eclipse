#!/usr/bin/env python
"""Offline checks for focus_hunt_auto.py -- no camera, no network.

Builds synthetic star fields (real HYG stars for the bright end so the blind
solver has something to lock onto, plus generated faint stars) at a range of
blur levels, and checks that the metric falls as the blur grows.  Then drives
the search loop against a simulated focus curve to check the step-halving and
edge-walking rules converge.

    python camera/focus_hunt_auto_selftest.py           # everything
    python camera/focus_hunt_auto_selftest.py --quick   # skip the image tests
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

import find_it
import focus_hunt_auto as fha

# synthetic rig: half the Z6's pixel count in each axis keeps the tests quick
# while leaving the arcsec/px and the star density per pixel realistic -- the
# focal length is scaled by the same factor so the field of view shrinks too.
W, H = 3024, 2012
FOCAL_MM = find_it.DEFAULT_FOCAL_MM
SENSOR_SCALE = 0.5                      # of the real sensor width

SKY = 400.0            # background counts per pixel
READ = 5.0             # read noise, counts
FLUX_AT_15 = 417.0     # total counts of a mag-15 star (marginal at FWHM 2)
SAT = 65535.0

CONE_DEG = 4.0         # synthetic catalogue cone radius
FAINT_MAG = 16.5       # synthetic catalogue depth (as focus_hunt_auto.DEEP_MAG)
FAINT_SLOPE = 0.35     # log10 N(<m) slope; ~the real count slope off the plane
N_AT_MAG12 = 46.0      # stars per sq deg brighter than mag 12 near M31. Back
                       #  -calculated from the real catalogue built on
                       #  2026-09-18: 304896 stars to G<16.5 in a 7.5 deg cone
                       #  = 1728/deg^2, which at this slope is 46/deg^2 at
                       #  mag 12. Sets the crowding the matcher must cope with,
                       #  so it has to match reality.


def arcsec_px() -> float:
    """arcsec per pixel for the scaled-down test sensor."""
    fov = math.degrees(2.0 * math.atan(
        (find_it.SENSOR_W_MM * SENSOR_SCALE / 2.0) / FOCAL_MM))
    return fov * 3600.0 / W


# ---------------------------------------------------------------------------
# synthetic sky
# ---------------------------------------------------------------------------
def make_catalog(rng, ra0, dec0):
    """Real HYG stars (mag<=9.5) plus generated faint ones, in one cone."""
    ra_h, dec_h, mag_h, _ = find_it.load_catalog()
    near = find_it.angsep_deg(ra_h, dec_h, ra0, dec0) < CONE_DEG
    ra_b, dec_b, mag_b = ra_h[near], dec_h[near], mag_h[near]

    # counts grow as 10^(slope*m); draw magnitudes from that, positions uniform
    area = 2 * math.pi * (1 - math.cos(math.radians(CONE_DEG))) * (180 / math.pi) ** 2
    n_faint = int(N_AT_MAG12 * area * 10 ** (FAINT_SLOPE * (FAINT_MAG - 12.0)))
    u = rng.random(n_faint)
    lo, hi = 10 ** (FAINT_SLOPE * 9.5), 10 ** (FAINT_SLOPE * FAINT_MAG)
    mag_f = np.log10(lo + u * (hi - lo)) / FAINT_SLOPE

    cosr = math.cos(math.radians(CONE_DEG))
    z = 1 - rng.random(n_faint) * (1 - cosr)
    th = np.arccos(z)
    ph = rng.random(n_faint) * 2 * math.pi
    # rotate the local cone onto (ra0, dec0)
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    zc = np.cos(th)
    d0 = math.radians(dec0)
    a0 = math.radians(ra0)
    xr = x * math.sin(d0) + zc * math.cos(d0)
    zr = -x * math.cos(d0) + zc * math.sin(d0)
    ra_f = np.degrees(np.arctan2(y, xr) + a0) % 360.0
    dec_f = np.degrees(np.arcsin(np.clip(zr, -1, 1)))

    return (np.concatenate([ra_b, ra_f]),
            np.concatenate([dec_b, dec_f]),
            np.concatenate([mag_b, mag_f]).astype(float))


def render(cat, ra0, dec0, roll_deg, fwhm_px, rng) -> np.ndarray:
    """Draw the catalogue onto a frame at the given blur, with noise."""
    ra, dec, mag = cat
    s_deg = arcsec_px() / 3600.0
    xi, eta = find_it.radec_to_tan(np.radians(ra), np.radians(dec),
                                   math.radians(ra0), math.radians(dec0))
    xp = np.degrees(xi) / s_deg
    yp = np.degrees(eta) / s_deg
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    x = W / 2.0 + (xp * c - yp * s)
    y = H / 2.0 - (xp * s + yp * c)          # image y grows downward

    sg = fwhm_px / 2.3548
    rad = max(2, int(math.ceil(4 * sg)))
    img = rng.normal(SKY, math.sqrt(SKY + READ ** 2), (H, W)).astype(np.float32)

    flux = FLUX_AT_15 * 10 ** (-0.4 * (mag - 15.0))
    inside = (x > -rad) & (x < W + rad) & (y > -rad) & (y < H + rad)
    ox = np.arange(-rad, rad + 1)
    gx, gy = np.meshgrid(ox, ox)
    for xi_, yi_, f_ in zip(x[inside], y[inside], flux[inside]):
        ix, iy = int(round(xi_)), int(round(yi_))
        x0, x1 = max(0, ix - rad), min(W, ix + rad + 1)
        y0, y1 = max(0, iy - rad), min(H, iy + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        sx = gx[:, (x0 - ix + rad):(x1 - ix + rad)][0]
        sy = gy[(y0 - iy + rad):(y1 - iy + rad), :][:, 0]
        dx = sx + ix - xi_
        dy = sy + iy - yi_
        g = np.exp(-0.5 * ((dx[None, :] ** 2 + dy[:, None] ** 2) / sg ** 2))
        img[y0:y1, x0:x1] += (f_ / (2 * math.pi * sg ** 2) * g).astype(np.float32)

    return np.clip(img, 0, SAT)


def seed_solution(ra0, dec0, roll_deg) -> dict:
    """The exact transform used to render -- the metric only needs a seed."""
    s_deg = arcsec_px() / 3600.0
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    # render() maps tangent-plane pixels -> centred image pixels with
    # [[c, -s], [-s, -c]] (the second row is negated because image y grows
    # downward); the solver wants the other direction.
    a = np.linalg.inv(np.array([[c, -s], [-s, -c]]))
    return {"A": a, "t": np.zeros(2), "center": (ra0, dec0), "s_deg": s_deg}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_match_one_to_one() -> bool:
    print("\n== one-to-one matching ==")
    ok = True
    # one detection sitting between two catalogue stars must claim only one
    det = np.array([[100.0, 100.0]])
    cat = np.array([[99.0, 100.0], [101.5, 100.0]])
    di, cj = fha.match_one_to_one(det, cat, 3.0)
    ok &= len(di) == 1 and cj[0] == 0
    print(f"  blended pair -> {len(di)} match (want 1), "
          f"nearest picked: {cj.tolist() == [0]}")
    # and two detections must not both take the same catalogue star
    det = np.array([[100.0, 100.0], [101.0, 100.0]])
    cat = np.array([[100.2, 100.0]])
    di, cj = fha.match_one_to_one(det, cat, 3.0)
    ok &= len(di) == 1
    print(f"  two detections, one star -> {len(di)} match (want 1)")
    # disjoint pairs all match
    det = np.array([[10.0, 10.0], [200.0, 200.0], [400.0, 50.0]])
    cat = det + 0.5
    di, cj = fha.match_one_to_one(det, cat, 3.0)
    ok &= len(di) == 3 and sorted(cj) == [0, 1, 2]
    print(f"  three separated -> {len(di)} matches (want 3)")
    return ok


def test_limiting_mag() -> bool:
    print("\n== limiting magnitude ==")
    rng = np.random.default_rng(1)
    mags = rng.uniform(8.0, 16.0, 20000)
    # detection probability falls through 0.5 at mag 13.0
    p = 1.0 / (1.0 + np.exp((mags - 13.0) / 0.25))
    matched = rng.random(20000) < p
    lim = fha.limiting_mag(mags, matched, 0.0)
    ok = abs(lim - 13.0) < 0.35
    print(f"  recovered {lim:.2f}, truth 13.00 -> {'ok' if ok else 'FAIL'}")
    return ok


def test_winner_logic() -> bool:
    print("\n== winner test ==")
    ok = True
    flat = {p: [1000.0, 1001.0] for p in (-64, -32, 0, 32, 64)}
    ok &= fha.pick_winner(flat) is None
    print(f"  flat curve -> no winner: {fha.pick_winner(flat) is None}")

    peak = {-64: [800.0, 802.0], -32: [900.0, 898.0], 0: [1000.0, 1001.0],
            32: [900.0, 901.0], 64: [800.0, 799.0]}
    w = fha.pick_winner(peak)
    ok &= w is not None and w[0] == 0
    print(f"  clear peak -> winner at {w[0] if w else None} (want 0)")

    # a lead inside the noise must not win
    noisy = {-64: [800.0, 950.0], -32: [960.0, 800.0], 0: [1000.0, 810.0],
             32: [820.0, 990.0], 64: [990.0, 805.0]}
    ok &= fha.pick_winner(noisy) is None
    print(f"  noise-dominated -> no winner: {fha.pick_winner(noisy) is None}")
    return ok


class _FakeCam:
    def __init__(self):
        self.pos = 0

    def move_to(self, p, **kw):
        self.pos = p


class _SimHunt(fha.Hunt):
    """Hunt with the camera replaced by an analytic focus curve."""

    def __init__(self, true_pos, sigma, rng, args):
        super().__init__(_FakeCam(), None, args, "")
        self.true_pos = true_pos
        self.sigma = sigma
        self.rng = rng
        self.calls = 0

    def measure(self, pos, tag):
        self.calls += 1
        n = 3000.0 * math.exp(-0.5 * ((pos - self.true_pos) / 30.0) ** 2)
        n += self.rng.normal(0.0, self.sigma)
        return {"ok": True, "score": n, "n_det": int(n), "n_false": 0,
                "lim_mag": float("nan"), "zp": float("nan"),
                "zp_scatter": float("nan")}


def test_search(verbose=False) -> bool:
    print("\n== search loop (simulated focus curve) ==")
    args = fha.parse_args([])
    ok = True
    hits = 0
    trials = 8
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        true_pos = int(rng.integers(-70, 71))
        h = _SimHunt(true_pos, 6.0, rng, args)
        if not verbose:
            out = sys.stdout
            sys.stdout = open(os.devnull, "w")
        try:
            best, clean = h.run(0, args.step)
        finally:
            if not verbose:
                sys.stdout.close()
                sys.stdout = out
        err = abs(best - true_pos)
        good = err <= 6
        hits += good
        print(f"  true={true_pos:+4d} -> best={best:+4d} err={err:2d} "
              f"frames={h.calls:3d} {'clean' if clean else 'premature'} "
              f"{'ok' if good else 'FAIL'}")
    ok &= hits >= trials - 1
    print(f"  {hits}/{trials} converged within 6 focus units")
    return ok


def test_metric_vs_blur(fwhms) -> bool:
    print("\n== metric vs blur (synthetic frames) ==")
    rng = np.random.default_rng(7)
    ra0 = find_it.M31_RA_DEG + 0.8
    dec0 = find_it.M31_DEC_DEG + 0.4
    roll = 17.0
    cat = make_catalog(rng, ra0, dec0)
    print(f"  catalogue: {len(cat[0])} stars to mag {cat[2].max():.1f}, "
          f"scale {arcsec_px():.2f} arcsec/px")

    deep = cat
    seed = seed_solution(ra0, dec0, roll)
    scores = []
    print("   fwhm       N   n_acc  lim_mag    zp   scat   false  n_det   rms")
    for fw in fwhms:
        t0 = time.time()
        img = render(cat, ra0, dec0, roll, fw, rng)
        res = fha.score_frame(img, deep, seed)
        if not res["ok"]:
            print(f"  {fw:5.1f}  FAILED: {res['reason']}")
            return False
        scores.append(res["score"])
        print(f"  {fw:5.1f} {res['score']:7.0f} {res['n_acc']:6d} "
              f"{res['lim_mag']:8.2f} {res['zp']:6.2f} {res['zp_scatter']:6.2f} "
              f"{res['n_false']:6d} {res['n_det']:6d} {res['rms']:5.2f} "
              f"  [{time.time() - t0:.1f}s]")

    mono = all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))
    drop = scores[0] / max(1.0, scores[-1])
    print(f"  monotonically falling: {mono};  sharpest/blurriest = {drop:.1f}x")
    return mono and drop > 3.0


def test_drift(n=14, step_px=30.0, fwhm=3.0) -> bool:
    """A creeping field must not move the score.

    The mount is not perfect, so the frame walks across the sky during a hunt
    and different stars fall inside it from frame to frame. Scoring a fixed
    patch of sky should make that invisible; scoring "whatever is in frame"
    should not.
    """
    print("\n== drift stability ==")
    rng = np.random.default_rng(11)
    ra0 = find_it.M31_RA_DEG + 0.8
    dec0 = find_it.M31_DEC_DEG + 0.4
    roll = 17.0
    cat = make_catalog(rng, ra0, dec0)
    d_deg = step_px * arcsec_px() / 3600.0

    pinned, floating = [], []
    ref = None
    sol = seed_solution(ra0, dec0, roll)
    print(f"   frame   drift_px   pinned   floating   on/ref")
    for i in range(n):
        dd = dec0 + i * d_deg
        # identical noise realisation every frame, so the only thing that can
        # move the score is the drift itself, not photon noise
        img = render(cat, ra0, dd, roll, fwhm, np.random.default_rng(99))
        a = fha.score_frame(img, cat, sol, ref=ref)
        # "floating" = score whatever falls in the frame, the behaviour a
        # pinned patch replaces
        inset = fha.REF_INSET_PX
        fha.REF_INSET_PX = 10
        try:
            b = fha.score_frame(img, cat, sol)
        finally:
            fha.REF_INSET_PX = inset
        if not (a["ok"] and b["ok"]):
            print(f"  frame {i}: FAILED "
                  f"({a.get('reason') or b.get('reason')})")
            return False
        sol = a["sol"]
        if ref is None:
            ref = a["ref"]
        pinned.append(a["score"])
        floating.append(b["score"])
        print(f"  {i:5d} {i * step_px:10.0f} {a['score']:8.0f} "
              f"{b['score']:10.0f}   {a['n_on']}/{a['n_ref']}")

    sp = float(np.std(pinned) / np.mean(pinned))
    sf = float(np.std(floating) / np.mean(floating))
    print(f"  spread: pinned {100 * sp:.2f}%, floating {100 * sf:.2f}%")

    # The thing that would actually corrupt a focus search is a systematic
    # TREND with drift -- across a burst that would read as a focus gradient.
    # Random scatter from stars swapping in and out at the edge is harmless.
    x = np.arange(n, dtype=float) * step_px
    y = np.array(pinned, float)
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (slope * x + icept)
    # standard error of the slope, so the trend is judged against its own
    # uncertainty rather than against the raw scatter (which is far larger and
    # would let a real trend through while failing a null result at random)
    se_slope = float(np.sqrt((resid ** 2).sum() / (n - 2)
                             / ((x - x.mean()) ** 2).sum()))
    trend = float(slope * x[-1]) / float(y.mean())
    sigmas = abs(slope) / se_slope if se_slope > 0 else float("inf")
    print(f"  trend over {x[-1]:.0f} px of drift: {100 * trend:+.2f}% "
          f"= {sigmas:.1f} sigma (scatter {100 * sp:.2f}%)")
    ok = sp < 0.025 and sigmas < 3.0 and abs(trend) < 0.04
    print(f"  no significant drift trend: {ok}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="skip the synthetic-image tests")
    p.add_argument("--verbose-search", action="store_true")
    a = p.parse_args(argv)

    results = {
        "match_one_to_one": test_match_one_to_one(),
        "limiting_mag": test_limiting_mag(),
        "winner_logic": test_winner_logic(),
        "search_loop": test_search(a.verbose_search),
    }
    if not a.quick:
        results["metric_vs_blur"] = test_metric_vs_blur(
            [2.0, 2.6, 3.4, 4.5, 6.5, 9.0])
        results["drift"] = test_drift()

    print("\n== summary ==")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
