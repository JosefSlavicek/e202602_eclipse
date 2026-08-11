#!/usr/bin/env python3
"""Calibrate and merge on top of a cached v2 run, skipping registration (~50 min).

Stages 0-2 of v6 are byte-identical to v2's, so a finished v2 workdir is a perfectly good
input: this reads its `v2-stage1.pkl` / `v2-stage2.pkl` through the module-alias shim,
re-saves them under the v6 names, then runs the calibration and stage 3.

    python v6/run_from_v2_cache.py \\
        --cache-dir /home/slavik/tmp/eclipse_v2_jpg \\
        --workdir   /home/slavik/tmp/eclipse_v6_from_v2

Nothing is written into `--cache-dir`.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6.device import configure_cuda_visible_devices, require_cuda  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/home/slavik/tmp/eclipse_v2_jpg"),
                    help="finished v2 workdir holding v2-stage1.pkl and v2-stage2.pkl")
    ap.add_argument("--workdir", type=Path,
                    default=Path("/home/slavik/tmp/eclipse_v6_from_v2"),
                    help="where v6 writes; never the cache dir")
    ap.add_argument("--input-mode", choices=["jpg", "atmosphere", "nef", "inject"], default="jpg")
    ap.add_argument("--jpg-dir", type=Path,
                    default=Path(os.environ.get("EDA00_DATA_ROOT",
                                                "/home/slavik/e202602_eclipse/data")))
    ap.add_argument("--atmosphere-dir", type=Path,
                    default=Path("/home/slavik/e202602_eclipse/data_atmosphere"))
    ap.add_argument("--nef-dir", type=Path, default=Path("/home/slavik/tmp/eclipse_fake_imgs"))
    ap.add_argument("--n-exposure-refine", type=int, default=None)
    ap.add_argument("--refine-registration", choices=["on", "off"], default="on",
                    help="redo the cross-exposure registration on calibrated radiance and "
                         "refit the calibration on it (one alternation step), as pipeline.py "
                         "does. The cached v2 registration is a gamma-scaled bootstrap.")
    ap.add_argument("--refine-space", choices=["log", "linear"], default="log")
    ap.add_argument("--calib-only", action="store_true",
                    help="stop after the calibration (fast: no full-resolution merge)")
    return ap.parse_args()


def import_v2_pickles(cache_dir: Path, workdir: Path) -> None:
    """Re-save the cached v2 stage pickles under the v6 names, via the alias shim."""
    from eclipse_v6.compat import install_v2_pickle_aliases

    install_v2_pickle_aliases()
    src1, src2 = cache_dir / "v2-stage1.pkl", cache_dir / "v2-stage2.pkl"
    assert src1.is_file() and src2.is_file(), (src1, src2)

    with open(src1, "rb") as fd:
        exposure_groups = pickle.load(fd)
        reg = pickle.load(fd)
        opt_results = pickle.load(fd)
    with open(workdir / "v6-stage1.pkl", "wb") as fd:
        pickle.dump(exposure_groups, fd)
        pickle.dump(reg, fd)
        pickle.dump(opt_results, fd)
    shutil.copyfile(src2, workdir / "v6-stage2.pkl")
    print(f"Imported {src1.name}, {src2.name} from {cache_dir} "
          f"({len(exposure_groups)} exposure groups)")


def main() -> int:
    configure_cuda_visible_devices()
    require_cuda()
    args = _parse_args()
    assert args.workdir.resolve() != args.cache_dir.resolve(), "refusing to write into the cache"
    args.workdir.mkdir(parents=True, exist_ok=True)

    import math

    import numpy as np
    import torch
    from eclipse_v6 import calib as ca
    from eclipse_v6 import reregister as rr
    from eclipse_v6 import stage2 as s2
    from eclipse_v6.inputs import attach_source, make_source
    from eclipse_v6.merge import merge_to_composite
    from eclipse_v6.stage3 import (
        Stage3Context, crop_and_save_composite, fft_unsharp_and_save, load_calibration,
        load_inputs, radial_normalize_display, rgb_vignette_and_radial_pickle,
    )

    import_v2_pickles(args.cache_dir, args.workdir)
    source = make_source(args.input_mode, jpg_dir=args.jpg_dir, nef_dir=args.nef_dir,
                         atmosphere_dir=args.atmosphere_dir)
    device = torch.device("cuda")
    pk_calib = args.workdir / "v6-calib.pkl"
    print(f"Input mode: {source.kind} (is_linear={source.is_linear})")
    print(f"Workdir:    {args.workdir}")

    print("\n=== Calibration: recover the response curve and the shutter corrections ===")
    exposure_groups, _reg, opt_results = s2.load(args.workdir / "v6-stage1.pkl")
    attach_source(exposure_groups, source)
    with open(args.workdir / "v6-stage2.pkl", "rb") as fd:
        cross_reg = pickle.load(fd)
    exposure_times_sorted = sorted(exposure_groups.keys())[2:]
    n_refine = ca.N_EXPOSURE_REFINE if args.n_exposure_refine is None else args.n_exposure_refine
    calib_result = ca.run(
        exposure_groups, opt_results, exposure_times_sorted, cross_reg,
        exposure_times_sorted[0], device, n_exposure_refine=n_refine, out_pkl=pk_calib)

    if args.refine_registration == "on":
        print("\n=== Re-registration: redo the cross-exposure alignment on calibrated radiance ===")
        source.set_calibration(calib_result)
        cross_reg_refined, _rows = rr.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg, source, device,
            space=args.refine_space, out_pkl=args.workdir / "v6-stage2r.pkl",
        )
        print("\n=== Recalibration: refit the response on the refined alignment ===")
        ln_c_before = calib_result.ln_c.copy()
        calib_result = ca.run(
            exposure_groups, opt_results, exposure_times_sorted, cross_reg_refined,
            exposure_times_sorted[0], device, n_exposure_refine=n_refine, out_pkl=pk_calib)
        d_ln_c = float(np.max(np.abs(calib_result.ln_c - ln_c_before)))
        print(f"Alternation: max |change in ln c| = {d_ln_c:.5f} "
              f"({math.expm1(d_ln_c) * 100:+.2f}% on the worst shutter correction)")

    if args.calib_only:
        print("\nCalibration only; stopping before the merge.")
        return 0

    print("\n=== Stage 3: reference merge, radial tone, FFT sharpen, RGB ===")
    ctx = Stage3Context(workdir=args.workdir)
    load_inputs(ctx, use_refined_registration=(args.refine_registration == "on"))
    attach_source(ctx.exposure_groups, source)
    load_calibration(ctx, source, pk_calib)
    merge_to_composite(ctx, source)
    crop_and_save_composite(ctx)
    radial_normalize_display(ctx)
    fft_unsharp_and_save(ctx)
    rgb_vignette_and_radial_pickle(ctx)
    print("\nDone. Outputs in", args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
