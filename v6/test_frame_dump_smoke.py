#!/usr/bin/env python3
"""Smoke test for `merge_to_composite`'s `frame_dump_dir` option.

Reuses the synthetic bracket from `test_merge_smoke.py` and checks that
`merge.reconstruct_from_frame_dump` — summing every dumped frame's (weight, wval),
normalised by the dumped `sum_w` and `moon_blank` — exactly reproduces `ctx.composite`.
That equivalence is the whole point of the frame dump: the composite is provably nothing
more than a flat weighted sum over individual frames once the right per-frame weight is
used.

Run directly (`python v6/test_frame_dump_smoke.py`) or under pytest. Needs CUDA.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6 import calib as CA                          # noqa: E402
from eclipse_v6.merge import NO_DATA, merge_to_composite, reconstruct_from_frame_dump  # noqa: E402
from eclipse_v6.stage3 import Stage3Context                  # noqa: E402
from test_merge_smoke import SyntheticSource, make_groups     # noqa: E402


def test_frame_dump_smoke():
    assert torch.cuda.is_available(), "this smoke test needs CUDA"
    device = torch.device("cuda")
    source = SyntheticSource()
    groups, opt_results, cross_reg = make_groups(source)
    exposure_times_sorted = sorted(groups.keys())
    t_ref = exposure_times_sorted[0]

    V, valid, exposures, sample_ij = CA.gather_samples(
        groups, opt_results, exposure_times_sorted, cross_reg, t_ref, device,
        n_radial_bins=24, samples_per_bin=120,
    )
    result = CA.calibrate(V, valid, exposures, verbose=False)
    assert result.istop != 3, result.istop
    source.set_calibration(result)

    ctx = Stage3Context(workdir=Path("/tmp"), device=device)
    ctx.exposure_groups = groups
    ctx.opt_results = opt_results
    ctx.cross_reg = cross_reg
    ctx.exposure_times_sorted = exposure_times_sorted
    ctx.t_ref = t_ref

    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = Path(tmp) / "frames"
        merge_to_composite(ctx, source, frame_dump_dir=dump_dir)

        manifest = json.loads((dump_dir / "manifest.json").read_text())
        n_frames_expected = sum(len(g) for g in groups.values())
        assert len(manifest) <= n_frames_expected, (len(manifest), n_frames_expected)
        assert len(manifest) > 0

        reconstructed = reconstruct_from_frame_dump(dump_dir)

        have = reconstructed > NO_DATA
        np.testing.assert_array_equal(have, ctx.composite > NO_DATA)
        np.testing.assert_allclose(
            reconstructed[have], ctx.composite[have], rtol=1e-4, atol=1e-6
        )
        print(f"  {len(manifest)} frames dumped, reconstruction matches ctx.composite "
              f"over {int(have.sum()):,} pixels")


if __name__ == "__main__":
    test_frame_dump_smoke()
    print("OK")
