#!/usr/bin/env python3
"""Smoke test of the post-calibration re-registration, on a scene whose alignment is known.

The planted scene from `test_merge_smoke` is perfectly aligned: every exposure sees the same
corona at the same place, so the true cross-exposure transform is exactly (0, 0, 0).  This
test seeds `cross_reg` with a deliberately wrong transform, as if stage 2's gamma-scaled
alignment had missed, and checks that `refine_cross_registration` recovers the truth.

`grid_search_registration` brackets +-46 px around zero and never reads the incoming
transform, so recovering (0, 0, 0) from a 3.6 px seed is a real measurement of the radiance
path — the log conversion, the coverage mask, the pair loop — and not the seed leaking
through.

Fast (a few seconds on one GPU, 500x700 frames). Run directly or under pytest. Needs CUDA.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v7 import calib as CA                        # noqa: E402
from eclipse_v7 import reregister as RR                   # noqa: E402
from test_merge_smoke import EXPOSURES, MOON_R, SyntheticSource, make_groups  # noqa: E402

SEEDED_ERROR = (2.0, -3.0, 0.0)   # the wrong transform we pretend stage 2 produced
TOL_PX = 0.6                      # a few of the grid search's 0.18 px cells, for stack noise
N_REFINED_EXPOSURES = 3           # keep the test short: 3 exposures = 2 pairs


def test_reregister_smoke():
    assert torch.cuda.is_available(), "this smoke test needs CUDA"
    device = torch.device("cuda")
    source = SyntheticSource()
    groups, opt_results, cross_reg_true = make_groups(source)
    exposure_times_sorted = sorted(groups.keys())
    t_ref = exposure_times_sorted[0]

    # Calibrate on the true alignment, as the pipeline's first calibration does.
    V, valid, exposures, sample_ij = CA.gather_samples(
        groups, opt_results, exposure_times_sorted, cross_reg_true, t_ref, device,
        n_radial_bins=24, samples_per_bin=120,
    )
    result = CA.calibrate(V, valid, exposures, verbose=False)
    assert result.istop != 3, result.istop
    source.set_calibration(result)

    # Pretend stage 2 got every pair wrong by the same amount.
    cross_reg_seeded = {k: SEEDED_ERROR for k in cross_reg_true}
    refine_over = exposure_times_sorted[:N_REFINED_EXPOSURES]

    refined, rows = RR.refine_cross_registration(
        groups, opt_results, refine_over, cross_reg_seeded, source, device, space="log",
    )

    assert len(rows) == N_REFINED_EXPOSURES - 1, len(rows)
    assert len(refined) == len(cross_reg_seeded), "refinement must not drop or add pairs"
    assert cross_reg_seeded[list(cross_reg_seeded)[0]] == SEEDED_ERROR, "input dict was mutated"

    # Pairs outside the refined exposure set keep their seeded transform untouched.
    untouched = [k for k in refined if k not in {(a, b) for a, b in zip(refine_over, refine_over[1:])}]
    assert untouched, "the test should leave some pairs outside the refined set"
    for k in untouched:
        assert refined[k] == SEEDED_ERROR, (k, refined[k])

    # The refined transforms must land on the truth, (0, 0, 0).
    for (t0, t1), got in ((k, refined[k]) for k in refined if k not in untouched):
        shift = float(np.hypot(got[0], got[1]))
        rot_px = abs(np.radians(got[2])) * (RR.ROT_EVAL_RADIUS_FACTOR * MOON_R)
        print(f"  pair {t0:.4f}->{t1:.4f}: shift {shift:.3f} px, rotation {rot_px:.3f} px "
              f"at {RR.ROT_EVAL_RADIUS_FACTOR:g} moon radii (truth is 0)")
        assert shift < TOL_PX, (t0, t1, got)
        assert rot_px < TOL_PX, (t0, t1, got)

    # And the report must show it moved by the seeded error, not by nothing.
    seeded_px = float(np.hypot(SEEDED_ERROR[0], SEEDED_ERROR[1]))
    for r in rows:
        assert abs(r["d_total_px"] - seeded_px) < TOL_PX, (r["d_total_px"], seeded_px)
    summary = RR.report(rows)
    assert summary["n_moved"] == len(rows), summary

    # Pickle round trip: what the pipeline hands to stage 3.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v7-stage2r.pkl"
        RR.save_pickle(path, refined, rows)
        back, rows_back = RR.load(path)
        assert back == refined, "cross_reg did not survive the pickle round trip"
        assert len(rows_back) == len(rows)


if __name__ == "__main__":
    test_reregister_smoke()
    print("OK")
