"""Synthetic end-to-end check of find_it.solve_field / arrow geometry."""
import math, numpy as np
import find_it as F

rng = np.random.default_rng(1)
cat = F.load_catalog()
ra, dec, mag, name = cat
W, H = 6064, 4040
s_ap = F.nominal_arcsec_px(W, 400.0)
s_deg = s_ap / 3600.0


def make_frame(center_ra, center_dec, roll_deg, parity, mag_lim, miss, spurious):
    xi, eta = F.radec_to_tan(np.radians(ra), np.radians(dec),
                             math.radians(center_ra), math.radians(center_dec))
    Q = np.column_stack([np.degrees(xi) / s_deg, np.degrees(eta) / s_deg])
    r = math.radians(roll_deg)
    cs, sn = math.cos(r), math.sin(r)
    A = np.array([[parity * cs, -sn], [parity * sn, cs]])
    Ainv = np.linalg.inv(A)
    P = Q @ Ainv.T + np.array([W / 2.0, H / 2.0])
    sepc = F.angsep_deg(ra, dec, center_ra, center_dec)
    inb = ((P[:, 0] > 0) & (P[:, 0] < W) & (P[:, 1] > 0) & (P[:, 1] < H)
           & (mag <= mag_lim) & (sepc < 5.0))
    P = P[inb] + rng.normal(0, 0.4, (int(inb.sum()), 2))
    m = mag[inb].copy()
    keep = rng.random(len(P)) > miss
    P, m = P[keep], m[keep]
    # realistic: detection order == brightness order; spurious dots are faintest
    for _ in range(spurious):
        P = np.vstack([P, [rng.uniform(0, W), rng.uniform(0, H)]])
        m = np.append(m, 99.0)
    order = np.argsort(m)
    return [dict(x=float(P[i, 0]), y=float(P[i, 1]), flux=1.0 / (k + 1),
                 area=9, peak=100.0) for k, i in enumerate(order)]


cases = [
    dict(center_ra=10.68, center_dec=41.27, roll_deg=0.0, parity=1),
    dict(center_ra=13.5, center_dec=43.0, roll_deg=37.0, parity=-1),
    dict(center_ra=8.0, center_dec=39.0, roll_deg=201.0, parity=1),
    dict(center_ra=10.68, center_dec=44.7, roll_deg=-70.0, parity=-1),   # M31 off frame
    dict(center_ra=15.5, center_dec=38.0, roll_deg=115.0, parity=1),     # ~5 deg away
]

ok = 0
for i, c in enumerate(cases):
    stars = make_frame(c["center_ra"], c["center_dec"], c["roll_deg"], c["parity"],
                       mag_lim=8.7, miss=0.15, spurious=6)
    sol = F.solve_field(stars, W, H, cat, F.M31_RA_DEG, F.M31_DEC_DEG, s_ap, 8.0,
                        verbose=(i == 0))
    if not sol.get("ok"):
        print(f"case {i}: FAIL to solve -- {sol['reason']}  ({len(stars)} stars)")
        continue
    dcen = F.angsep_deg(np.array([sol["center"][0]]), np.array([sol["center"][1]]),
                        c["center_ra"], c["center_dec"])[0]
    # truth arrow: project M31 directly through the same WCS
    xi, eta = F.radec_to_tan(np.radians([F.M31_RA_DEG]), np.radians([F.M31_DEC_DEG]),
                             math.radians(c["center_ra"]), math.radians(c["center_dec"]))
    r = math.radians(c["roll_deg"]); cs, sn = math.cos(r), math.sin(r)
    A = np.array([[c["parity"] * cs, -sn], [c["parity"] * sn, cs]])
    Ainv = np.linalg.inv(A)
    m31_true = (np.array([np.degrees(xi)[0] / s_deg, np.degrees(eta)[0] / s_deg]) @ Ainv.T
                + np.array([W / 2, H / 2]))
    m31_got = F.radec_to_pixel(sol, F.M31_RA_DEG, F.M31_DEC_DEG, W, H)[0]
    perr = float(np.hypot(*(m31_true - m31_got)))
    start, end, inframe = F.arrow_endpoints(sol, W, H)
    good = dcen < 0.05 and perr < 5.0
    ok += good
    print(f"case {i}: {'OK ' if good else 'BAD'}  centre_err={dcen*3600:.1f}\"  "
          f"M31_px_err={perr:.2f}px  inliers={sol['n_inliers']}  rms={sol['rms']:.2f}  "
          f"roll={sol['roll_deg']:+.1f}(true {c['roll_deg']:+.1f})  "
          f"parity={sol['parity']:+d}(true {c['parity']:+d})  M31_in_frame={inframe}")

print(f"\n{ok}/{len(cases)} cases recovered")

# --- sparse / noisy: should still solve --------------------------------------
stars = make_frame(11.5, 42.0, 88.0, -1, mag_lim=8.2, miss=0.35, spurious=10)
sol = F.solve_field(stars, W, H, cat, F.M31_RA_DEG, F.M31_DEC_DEG, s_ap, 8.0,
                    verbose=False)
print(f"sparse case ({len(stars)} stars): "
      f"{'OK inliers=%d rms=%.2f' % (sol['n_inliers'], sol['rms']) if sol.get('ok') else 'FAIL ' + sol['reason']}")

# --- negative control: random dots, nothing real -- MUST NOT solve ----------
rng2 = np.random.default_rng(7)
junk = [dict(x=float(rng2.uniform(0, W)), y=float(rng2.uniform(0, H)),
             flux=1.0 / (k + 1), area=9, peak=100.0) for k in range(40)]
sol = F.solve_field(junk, W, H, cat, F.M31_RA_DEG, F.M31_DEC_DEG, s_ap, 8.0,
                    verbose=False)
print(f"negative control (random dots): "
      f"{'BAD - falsely solved!' if sol.get('ok') else 'OK rejected (' + sol['reason'] + ')'}")

# --- wrong region: real pattern but 30 deg from M31 -- MUST NOT solve -------
stars = make_frame(40.0, 40.0, 0.0, 1, mag_lim=8.0, miss=0.1, spurious=4)
sol = F.solve_field(stars, W, H, cat, F.M31_RA_DEG, F.M31_DEC_DEG, s_ap, 8.0,
                    verbose=False)
print(f"far-off-target (30 deg away): "
      f"{'BAD - falsely solved!' if sol.get('ok') else 'OK rejected (' + sol['reason'] + ')'}")
