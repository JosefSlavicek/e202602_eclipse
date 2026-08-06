# v5 — bracketed exposures on one physical brightness scale

v5 is a self-contained copy of v2 (no cross-version imports; the `vN/eclipse_vN/` convention
v3 established) with the radiometry replaced as specified in `../PHOTOMETRY_SPEC.md`.

**What changed.** v2 fitted a separate exponent to each adjacent pair of exposures,
multiplied those 16 exponents down a ladder to put everything on one scale, and combined
with a hand-shaped bell weight. That is unsound: the exponents came out spread 0.87–1.39
when a single camera response demands they be equal, each fit silently absorbed whatever
else differed between its two frames, the errors compound along the chain, and the weight
was chosen by eye. v5 recovers the camera's response curve **once** for the whole bracket —
together with corrections to the shutter times, which the camera reports inaccurately —
converts every frame to physical brightness independently, and combines with weights derived
from each measurement's actual uncertainty.

Stages 0–2 are unchanged. Stage 2's per-pair exponents are still computed and still used,
but **for alignment only**: Fourier registration cares about structure, not absolute scale.
They no longer determine any brightness.

## Layout

| file | what it does |
|---|---|
| `eclipse_v5/calib.py` | recovers `ln f` at 64 knots + per-exposure shutter corrections (Debevec–Malik, sparse `lsqr`, corrections by alternation) |
| `eclipse_v5/merge.py` | per-frame brightness, inverse-variance merge, common moon blanking, `NO_DATA` sentinel |
| `eclipse_v5/warp.py` | reference-grid warping and the cross-exposure chain, shared by the calibration, both merges and stage 3 |
| `eclipse_v5/inputs.py` | adds `set_calibration` / `effective_exposure` / `load_radiance` to the source layer |
| `eclipse_v5/compat.py` | module aliases so a cached **v2** run's pickles load under the v5 package name |
| `eclipse_v5/stage3.py` | display chain, unchanged, plus the merge→display handoff and `DISPLAY_GAMMA` |
| `pipeline.py` | the runner: stage0 → 1 → 2 → **calibration** → 3 |
| `run_from_v2_cache.py` | calibrate + merge on top of a finished v2 workdir, skipping ~50 min of registration |
| `report_display_exponent.py` | measures `DISPLAY_GAMMA` from the composites' brightness spread |
| `test_calib_synthetic.py` | plants a known response, asserts it is recovered. No GPU, ~3 s |
| `test_merge_smoke.py` | calibration → `load_radiance` → merge end to end on a small synthetic scene. Needs CUDA, ~6 s |

## Running

```bash
source /home/slavik/usr/anaconda3/etc/profile.d/conda.sh && conda activate e202602_eclipse
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

python v5/test_calib_synthetic.py                 # verify the numerics first (no GPU)
python v5/test_merge_smoke.py                     # verify the plumbing

python v5/pipeline.py                             # full run, jpg mode
python v5/pipeline.py --merge legacy              # v2's merge, for side-by-side comparison
python v5/pipeline.py --n-exposure-refine 0       # honest nominal-shutter-times baseline
python v5/run_from_v2_cache.py --cache-dir /home/slavik/tmp/eclipse_v2_jpg \
                               --workdir /home/slavik/tmp/eclipse_v5_from_v2
```

## What to look at in the output

| number | expected on the real JPEG data |
|---|---|
| `lsqr istop` | 2 (**3 means ill-conditioned and the answer is wrong**) |
| monotonicity violations | 0 |
| residual, core `v ∈ [0.25, 0.85]` | ~0.73 % rms, no bias |
| per-exposure corrections | 0.93–1.10, none pinned at the ±25 % clamp, largest on the shortest exposures |
| response local exponent | ~1.82 at `v=0.20` rising to ~3.82 at `v=0.80` (a power law would be constant) |
| coverage gaps outside the moon | 0 % |
| brightness spread p99/p50 | ~233 (v2's composite, which the display chain is tuned for, was 167) |

If the corrections come out at the clamp, or `istop == 3`, the alternation has been turned
into a joint solve — see `PHOTOMETRY_SPEC.md` §3.5 trap 1.

## Deliberate approximations

- The calibration is fitted on per-exposure averages of **encoded** values while the merge
  averages brightness. Wrong order in principle; the frames within a group are registered
  copies of the same scene, so the spread is noise-level and the bias second order. The curve
  is unknown when those averages are built, which is why.
- One response curve for the **mean of R, G, B**. The three channels have different curves
  and different clipping points; per-channel calibration is strictly better and is the
  natural next step.
- `value_sigma = 1/255` stands in for quantization, JPEG block artefacts and chroma
  subsampling together. Measured residuals came out ~4× smaller than it predicts, mostly
  because each value is already a stack average — relative weighting is right, the absolute
  variance scale is pessimistic. That matters only if the σ maps are read as absolute.
- The two shortest exposures are dropped (inherited from v2, no recorded reason; kept for
  comparability).
- `GAIN_E_PER_UNIT` and `READ_SIGMA` in `inputs.py` are placeholders setting only relative
  weighting between raw frames. Measure them from a flat-field pair: variance against mean is
  a straight line, slope `1/gain`, intercept `read_noise²`.
- `pipeline.ipynb` is inherited from v1: its stage 0–2 cells predate the `FrameSource` API
  and its saved outputs are from an old v1 run. `pipeline.py` is the maintained runner.
