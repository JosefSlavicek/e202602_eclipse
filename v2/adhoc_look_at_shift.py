"""Per-pair drift vector V = (shift - Δmoon_center) / Δt for reduced pairs (post stage1 prune).

Loads the v1 post-pruning pickle (default: $ECLIPSE_V1_WORKDIR/v1-eda02.pkl, matching
v1/pipeline.py's WORKDIR convention) without importing torch, then for every ordered
registered pair computes the residual shift after subtracting the moon-center difference,
divides by Δt (EXIF timestamps), and prints mean/std/min/max of the (V_i, V_j) components
per exposure group plus pooled overall.
"""
from __future__ import annotations

import enum
import os
import pickle
import statistics
from pathlib import Path

WORKDIR = Path(os.environ.get("ECLIPSE_V1_WORKDIR", "/home/slavik/tmp/eclipse_v1_run"))
PKL = WORKDIR / "v1-eda02.pkl"


# --- stubs so the pickle (originally from eclipse_v1.stage0) loads without torch ----------

class _ImageInfoStub:
    """Plain placeholder; pickle restores fields straight into __dict__."""


class _MoonInfoOriginStub(enum.Enum):
    """Real Enum so pickle's `cls(value)` reconstruction works; values must match the original."""
    DIRECT = 0
    INTERPOLATED = 1


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "eclipse_v1.stage0":
            if name == "ImageInfo":
                return _ImageInfoStub
            if name == "MoonInfoOrigin":
                return _MoonInfoOriginStub
        return super().find_class(module, name)


def load_eda02(path: Path):
    with open(path, "rb") as fd:
        up = _Unpickler(fd)
        exposure_groups = up.load()
        reg = up.load()
        # opt_results follows in the file; we don't need it.
    return exposure_groups, reg


# --- stats helpers --------------------------------------------------------------------------

def _stats(vals, weights):
    if not vals:
        return None
    n = len(vals)
    wsum = sum(weights)
    mean = sum(w * x for w, x in zip(weights, vals)) / wsum
    if n > 1:
        var = sum(w * (x - mean) ** 2 for w, x in zip(weights, vals)) / wsum
        std = var ** 0.5
    else:
        std = 0.0
    return n, mean, std, min(vals), max(vals)


def _fmt(label, s):
    if s is None:
        return f"    {label}: <empty>"
    n, mean, std, mn, mx = s
    return (
        f"    {label}: n={n:4d}  mean={mean:+9.4f}  std={std:8.4f}  "
        f"min={mn:+9.4f}  max={mx:+9.4f}"
    )


# --- main -----------------------------------------------------------------------------------

def main():
    exposure_groups, reg = load_eda02(PKL)
    print(f"Loaded {PKL}")
    print(f"  exposure groups: {len(exposure_groups)}")
    print(f"  reg pairs (reduced, post pruning): {len(reg)}")
    print()
    print("V is in pixels/second. Components: V_i (row axis), V_j (col axis).")
    print()

    all_vi, all_vj, all_w = [], [], []

    for exposure_time in sorted(exposure_groups):
        group = list(exposure_groups[exposure_time])
        keys = [k for k in reg if k[0] == exposure_time]
        if not keys:
            continue

        v_i_list, v_j_list, w_list, skipped_dt = [], [], [], 0
        max_abs_dt = 0.0
        for et, i, j in keys:
            ii_a = group[i]
            ii_b = group[j]
            shift_i, shift_j, _rot = reg[(et, i, j)]
            dt = ii_a.timestamp - ii_b.timestamp
            if abs(dt) > max_abs_dt:
                max_abs_dt = abs(dt)
            if abs(dt) < 1e-3:
                skipped_dt += 1
                continue
            d_moon_i = ii_a.moon[0] - ii_b.moon[0]
            d_moon_j = ii_a.moon[1] - ii_b.moon[1]
            v_i_list.append((shift_i - d_moon_i) / dt)
            v_j_list.append((shift_j - d_moon_j) / dt)
            w_list.append(abs(dt))

        all_vi.extend(v_i_list)
        all_vj.extend(v_j_list)
        all_w.extend(w_list)

        print(
            f"exposure_time={exposure_time:.6f}s  n_images={len(group):3d}  "
            f"n_pairs={len(v_i_list):4d}  skipped(dt~0)={skipped_dt}  "
            f"max|dt|={max_abs_dt:.3f}s"
        )
        print(_fmt("V_i", _stats(v_i_list, w_list)))
        print(_fmt("V_j", _stats(v_j_list, w_list)))

    print()
    print("Pooled across all exposure groups:")
    print(_fmt("V_i", _stats(all_vi, all_w)))
    print(_fmt("V_j", _stats(all_vj, all_w)))


if __name__ == "__main__":
    main()
