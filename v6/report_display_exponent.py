#!/usr/bin/env python3
"""Measure the display exponent instead of guessing it.

The composite is physical brightness now, while every constant in the display chain was
tuned against v2's composite.  `stage3.DISPLAY_GAMMA` is the single knob that reconciles
them, and the honest way to set it is to match the two composites' *measured* brightness
spread (p99/p50):

    spread_new ** (1/gamma) == spread_reference   =>   gamma = ln(spread_new)/ln(spread_ref)

Do NOT derive this from first principles.  The tempting argument — "v2's composite was
heavily compressed by the camera's JPEG encoding, so imitate that" — gives ~1.63 and ruins
the image.  It is wrong because v2 scaled each exposure by `(t_ref/t_k)^(1/g)` with g ~ 1.1,
which is nearly the plain exposure ratio, and across a ~1e4 brightness range those factors
do almost all of the work: v2's composite was already nearly proportional to brightness.
On the JPEG data the measured answer is ~1.065, i.e. essentially no compression.

    python v6/report_display_exponent.py \\
        --reference /home/slavik/tmp/eclipse_v2_run_20260804/v2-stage3_composite.npy \\
        --composite /home/slavik/tmp/eclipse_v6_run/v6-stage3_composite.npy
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eclipse_v6.merge import NO_DATA          # noqa: E402
from eclipse_v6.stage3 import brightness_spread  # noqa: E402

V2_REFERENCE_SPREAD = 167.0   # measured on v2's JPEG-set composite, the chain's tuning point


def _load(path: Path):
    arr = np.load(Path(path)).astype(np.float64)
    no_data = arr <= NO_DATA
    return arr, no_data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--composite", type=Path, required=True,
                    help="v6-stage3_composite.npy — the new merge's physical brightness")
    ap.add_argument("--reference", type=Path, default=None,
                    help="v2-stage3_composite.npy the display chain was tuned against. "
                         f"Omit to use the recorded value {V2_REFERENCE_SPREAD:.0f}.")
    args = ap.parse_args()

    new, new_nd = _load(args.composite)
    spread_new = brightness_spread(new, no_data=new_nd)
    print(f"new merge      {args.composite}")
    print(f"  measured pixels {int((~new_nd & (new > 0)).sum()):,}, "
          f"no-data {int(new_nd.sum()):,}")
    print(f"  brightness spread (p99/p50)  {spread_new:.1f}")

    if args.reference is not None:
        ref, ref_nd = _load(args.reference)
        spread_ref = brightness_spread(ref, no_data=ref_nd)
        print(f"reference      {args.reference}")
        print(f"  brightness spread (p99/p50)  {spread_ref:.1f}")
    else:
        spread_ref = V2_REFERENCE_SPREAD
        print(f"reference      recorded v2 value  {spread_ref:.1f}")

    assert spread_new > 1.0 and spread_ref > 1.0, (spread_new, spread_ref)
    gamma = math.log(spread_new) / math.log(spread_ref)
    print()
    print(f"  matching exponent            {gamma:.4f}")
    print(f"  => set eclipse_v6/stage3.py  DISPLAY_GAMMA = {gamma:.3f}")
    if abs(gamma - 1.0) < 0.10:
        print("  (within 10% of 1.0 — leaving DISPLAY_GAMMA at 1.0 keeps the no-op exact and "
              "costs almost nothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
