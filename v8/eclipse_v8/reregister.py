"""Re-register the exposure pairs once the response curve is known.

Stage 2 has to align the bracket before anything is calibrated, so it pre-scales the longer
frame of each pair by a fitted per-pair exponent and hands the grid search a saturation mask
derived from that same exponent:

    apriori_valid = (img0 * ((t1 / t0) ** (1.0 / gamma)) <= 0.9)      # stage2.py

Neither is physical, and the L1 residual the search minimises is not scale-invariant, so an
error in the assumed brightness ratio biases the alignment — in a corona whose brightness
falls steeply with radius, "too bright" and "slightly displaced radially" are partly
degenerate. Stage 2's exponents therefore remain what they always were: a bootstrap, good
enough to let the calibration find corresponding pixels.

This module runs afterwards and redoes the same grid search on calibrated data:

  * Both frames are converted to radiance through `source.load_radiance`. Radiance is light
    per unit time — the same physical quantity in both frames — so in log space the exposure
    ratio is a pure additive constant, which `clean_polar_fft`'s `remove_lowfeq` already
    strips. The objective becomes exposure-invariant by construction and no brightness
    estimate enters registration at all.
  * The saturation mask comes from each exposure's own validity (`covered`, i.e. how many
    frames of the stack could actually be inverted at that pixel) instead of being predicted
    from the other exposure through a fitted exponent.

`refine_cross_registration` returns a new `cross_reg` and a per-pair report of how far each
transform moved. The caller is expected to refit the calibration on the result: calibration
consumes `cross_reg` to find corresponding pixels, so the two are mutually dependent and
this is one step of an alternation, the same shape as `calib._refine_step` uses for the
shutter corrections.

Measured on the 2026-08-06 JPEG run (`compare_registration_radiance.py`, control
bit-identical to stage 2): 12 of 14 pairs move, median 0.36 px, max 0.72 px — 2 to 4 of the
0.18 px cells the grid search resolves.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm

from eclipse_v8.merge import average_exposure_radiance
from eclipse_v8.utils import grid_search_registration, moon_median

# --- constants ------------------------------------------------------------------------
COVER_THRESH = 0.5            # as merge.MIN_STACK_COVER: below half a frame's worth of valid
                              # contributions the pixel is saturated, off-frame or moon
FLOOR_QUANTILE = 0.05         # log floor, taken on the longer exposure (it measures the faint
                              # end best) and shared by both frames so the offset between them
                              # survives untouched
QUANTILE_SUBSAMPLE = 1 << 22  # torch.quantile refuses very large inputs; stride down to this
INITIAL_SHIFT_HALF = 2.0 * (5.0 * (2.0 + 2.0) + 0.0 + 3.0)   # stage2.register_cross_exposure
SEARCH_RESOLUTION_PX = 0.18   # the grid search stops refining below 0.1 px, which lands on a
                              # final cell of 23 / 2**7; deltas under this are not resolvable
ROT_EVAL_RADIUS_FACTOR = 3.0  # a rotation delta is reported as the arc it sweeps at 3 moon
                              # radii — out where the corona structure actually lives


def displacement(a, b, r_eval):
    """(translation px, rotation arc px at r_eval, total px) between two (si, sj, rot) triples."""
    d_trans = math.hypot(a[0] - b[0], a[1] - b[1])
    d_rot_px = abs(math.radians(a[2] - b[2])) * r_eval
    return d_trans, d_rot_px, d_trans + d_rot_px


def average_radiance(group, opt_results, exp, source, device):
    """One exposure's stack mean in physical brightness, on the stage-1 poses."""
    abs_xy = torch.from_numpy(opt_results[exp]["abs_xy"]).to(device)
    abs_angle_t = torch.from_numpy(opt_results[exp]["abs_angle_t"]).to(device)
    Lbar, covered, _moon_out, _group_weight = average_exposure_radiance(
        group, abs_xy, abs_angle_t, source, device
    )
    return Lbar, covered


class RadianceAverageCache:
    """Rolling cache over consecutive pairs: each exposure is averaged once, used twice.

    Tensors are parked on the CPU (two full-res float32 arrays is ~190 MB at 4000x6000) and
    moved to the GPU inside the pair loop, as stage 2 does with its own averages.
    """

    def __init__(self, exposure_groups, opt_results, source, device):
        self.groups = exposure_groups
        self.opt_results = opt_results
        self.source = source
        self.device = device
        self._cache = {}

    def get(self, exp):
        if exp not in self._cache:
            L, covered = average_radiance(
                self.groups[exp], self.opt_results, exp, self.source, self.device
            )
            self._cache[exp] = {"L": L.cpu(), "covered": covered.cpu()}
            del L, covered
            torch.cuda.empty_cache()
        return self._cache[exp]

    def drop_all_but(self, keep):
        for exp in [e for e in self._cache if e not in keep]:
            del self._cache[exp]


def _low_quantile(x, valid, q):
    """q-quantile of `x` over `valid`, on a strided subsample (torch.quantile caps its input)."""
    v = x[valid]
    assert v.numel() > 0, "no valid pixels to take a quantile over"
    if v.numel() > QUANTILE_SUBSAMPLE:
        v = v[:: (v.numel() // QUANTILE_SUBSAMPLE) + 1]
    return float(torch.quantile(v.float(), q))


def radiance_pair_images(L0, cov0, L1, cov1, space: str = "log"):
    """Both frames of a pair as comparable images, plus the target's validity mask.

    Nothing is scaled: radiance is already the same quantity in both frames, which is the
    whole point. Uncovered pixels are held at the floor rather than zeroed — a zero is a hard
    edge that `remove_lowfeq` would smear across every angular frequency, while a constant is
    exactly what the low-frequency strip removes. Saturated cores need nothing beyond that:
    `clean_polar_fft`'s antiprot clip already flattens the top of the range.

    `space="linear"` keeps the current residual's weighting and only fixes the scale;
    `"log"` additionally makes the objective invariant to the exposure ratio.
    """
    assert space in ("log", "linear"), space
    # A pixel no frame covered has Lbar = 0/EPS, and one bad frame can leave a non-finite
    # behind; either would poison the quantile and then the whole FFT row.
    valid0 = (cov0 >= COVER_THRESH) & torch.isfinite(L0) & (L0 > 0)
    valid1 = (cov1 >= COVER_THRESH) & torch.isfinite(L1) & (L1 > 0)
    floor = _low_quantile(L1, valid1, FLOOR_QUANTILE)
    assert floor > 0, floor

    def prep(L, valid):
        x = torch.where(valid, L, torch.full_like(L, floor)).clamp(min=floor)
        if space == "log":
            return torch.log(x) - math.log(floor)      # >= 0, common offset preserved
        return x / floor                               # common factor: the argmin is unchanged

    # The mask stage 2 approximated by predicting img1's clipping from img0 through the
    # fitted exponent; here both exposures state their own validity. valid1 is used un-warped:
    # the pair is within tens of pixels of alignment already, and the mask edge sits in the
    # saturated core where nothing is being matched anyway.
    apriori_valid = (valid0 & valid1).to(torch.float32)
    return prep(L0, valid0), prep(L1, valid1), apriori_valid


def register_pair(x0, x1, apriori_valid, moon0, moon1, device):
    """Stage 2's grid search, same starting bracket, on calibrated images."""
    return grid_search_registration(
        x0, x1, moon0, moon1, INITIAL_SHIFT_HALF, device, apriori_valid=apriori_valid
    )


def refine_cross_registration(
    exposure_groups,
    opt_results,
    exposure_times_sorted,
    cross_reg,
    source,
    device,
    *,
    space: str = "log",
):
    """Redo the consecutive-pair registration on calibrated radiance.

    `exposure_times_sorted` is the exposure set the calibration covers; pairs outside it keep
    their stage-2 transform (their shutter corrections were never fitted, and nothing
    downstream uses them). Returns `(cross_reg_refined, report_rows)`; `cross_reg` itself is
    not modified.

    `source` must already have `set_calibration()` called on it.
    """
    assert source._calib is not None, "call source.set_calibration(...) before refining"
    refined = dict(cross_reg)
    cache = RadianceAverageCache(exposure_groups, opt_results, source, device)
    pairs = [
        (t0, t1)
        for t0, t1 in zip(exposure_times_sorted, exposure_times_sorted[1:])
        if (t0, t1) in cross_reg
    ]
    kept = len(cross_reg) - len(pairs)
    print(f"Re-registering {len(pairs)} pairs in {space} radiance"
          + (f"; {kept} pair(s) outside the calibration keep their stage-2 transform" if kept else ""))

    rows = []
    for t0, t1 in tqdm.tqdm(pairs, desc="Re-register (radiance)"):
        moon0 = moon_median(exposure_groups[t0])
        moon1 = moon_median(exposure_groups[t1])
        r_eval = ROT_EVAL_RADIUS_FACTOR * moon0[2]
        e0 = cache.get(t0)
        e1 = cache.get(t1)

        L0 = e0["L"].to(device)
        c0 = e0["covered"].to(device)
        L1 = e1["L"].to(device)
        c1 = e1["covered"].to(device)
        x0, x1, apriori_valid = radiance_pair_images(L0, c0, L1, c1, space)
        del L0, L1, c0, c1
        torch.cuda.empty_cache()

        got = register_pair(x0, x1, apriori_valid, moon0, moon1, device)
        del x0, x1, apriori_valid
        torch.cuda.empty_cache()

        d_trans, d_rot, d_total = displacement(got, cross_reg[(t0, t1)], r_eval)
        refined[(t0, t1)] = got
        rows.append({
            "t0": t0, "t1": t1, "r_eval_px": r_eval,
            "stage2": list(cross_reg[(t0, t1)]), "refined": list(got),
            "d_trans_px": d_trans, "d_rot_px": d_rot, "d_total_px": d_total,
        })
        cache.drop_all_but({t1})

    report(rows)
    return refined, rows


def report(rows) -> dict:
    """Print the per-pair movement and return its summary."""
    if not rows:
        print("Re-registration: no pairs refined.")
        return {"n_pairs": 0}
    print(f"{'t0':>9} {'t1':>9} | {'shift_i':>9} {'shift_j':>9} {'rot_deg':>9} "
          f"| {'d_trans':>8} {'d_rot':>8} {'d_total':>8}  (px)")
    for r in rows:
        print(f"{r['t0']:9.5f} {r['t1']:9.5f} | {r['refined'][0]:9.4f} {r['refined'][1]:9.4f} "
              f"{r['refined'][2]:9.4f} | {r['d_trans_px']:8.3f} {r['d_rot_px']:8.3f} "
              f"{r['d_total_px']:8.3f}")
    tot = np.array([r["d_total_px"] for r in rows])
    summary = {
        "n_pairs": len(rows),
        "median_px": float(np.median(tot)),
        "max_px": float(tot.max()),
        "n_moved": int((tot > SEARCH_RESOLUTION_PX).sum()),
    }
    print(f"Moved: median {summary['median_px']:.3f} px, max {summary['max_px']:.3f} px, "
          f"{summary['n_moved']}/{len(rows)} pairs beyond the search's own "
          f"{SEARCH_RESOLUTION_PX} px cell")
    if summary["max_px"] <= SEARCH_RESOLUTION_PX:
        print("Nothing moved by more than the grid resolution — the stage-2 alignment was "
              "already at the achievable optimum for this bracket.")
    return summary


def save_pickle(out_pkl: Path, cross_reg_refined: dict, rows: list) -> Path:
    out_pkl = Path(out_pkl)
    with open(out_pkl, "wb") as fd:
        pickle.dump(cross_reg_refined, fd)
        pickle.dump(rows, fd)
    print(f"Saved {out_pkl} (refined cross_reg, movement report).")
    return out_pkl


def load(path: Path):
    """(cross_reg_refined, rows) from a pickle written by `save_pickle`."""
    with open(Path(path), "rb") as fd:
        cross_reg_refined = pickle.load(fd)
        rows = pickle.load(fd)
    return cross_reg_refined, rows


def run(
    exposure_groups,
    opt_results,
    exposure_times_sorted,
    cross_reg,
    source,
    device,
    *,
    space: str = "log",
    out_pkl: Path | None = None,
):
    """Refine, report, optionally save. Returns `(cross_reg_refined, rows)`."""
    refined, rows = refine_cross_registration(
        exposure_groups, opt_results, exposure_times_sorted, cross_reg, source, device,
        space=space,
    )
    if out_pkl is not None:
        save_pickle(out_pkl, refined, rows)
    return refined, rows
