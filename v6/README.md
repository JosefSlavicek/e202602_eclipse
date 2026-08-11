# v6 — bracketed exposures on one physical brightness scale

v6 is, so far, a self-contained copy of v5 (no cross-version imports; the `vN/eclipse_vN/`
convention v3 established) — no changes yet. The rest of this file is v5's history, inherited
unchanged; v5 itself was a copy of v2 with the radiometry replaced as specified in
`../PHOTOMETRY_SPEC.md`.

**What changed.** v2 fitted a separate exponent to each adjacent pair of exposures,
multiplied those 16 exponents down a ladder to put everything on one scale, and combined
with a hand-shaped bell weight. That is unsound: the exponents came out spread 0.87–1.39
when a single camera response demands they be equal, each fit silently absorbed whatever
else differed between its two frames, the errors compound along the chain, and the weight
was chosen by eye. v6 recovers the camera's response curve **once** for the whole bracket —
together with corrections to the shutter times, which the camera reports inaccurately —
converts every frame to physical brightness independently, and combines with weights derived
from each measurement's actual uncertainty.

Stages 0–2 are unchanged. Stage 2's per-pair exponents are still computed, but they are now
only a **bootstrap**: they scale the frames well enough to align the bracket, which is what
the calibration needs in order to find corresponding pixels across exposures. They determine
no brightness.

**And no longer the alignment either.** The registration objective is an L1 residual on
Fourier-cleaned images, which is not scale-invariant, and the saturation mask handed to the
grid search was itself derived from the fitted exponent — so a wrong exponent biases the
alignment, not just the brightness. Once the response curve is known, `reregister.py` redoes
the cross-exposure registration on physical radiance (in log space, where the exposure ratio
is an additive constant that `remove_lowfeq` already strips, so no photometric estimate
enters at all) and the calibration is refit on the result. One alternation step, on by
default; `--refine-registration off` reproduces the old behaviour. Measured on the real JPEG
bracket: 12 of 14 pairs move, median 0.36 px, max 0.72 px — 2 to 4 of the 0.18 px cells the
grid search can resolve.

## Layout

| file | what it does |
|---|---|
| `eclipse_v6/calib.py` | recovers `ln f` at 64 knots + per-exposure shutter corrections (Debevec–Malik, sparse `lsqr`, corrections by alternation) |
| `eclipse_v6/merge.py` | per-frame brightness, inverse-variance merge, common moon blanking, `NO_DATA` sentinel |
| `eclipse_v6/reregister.py` | redoes the cross-exposure registration on calibrated radiance; writes `v6-stage2r.pkl` |
| `eclipse_v6/warp.py` | reference-grid warping and the cross-exposure chain, shared by the calibration, both merges and stage 3 |
| `eclipse_v6/inputs.py` | adds `set_calibration` / `effective_exposure` / `load_radiance` to the source layer |
| `eclipse_v6/compat.py` | module aliases so a cached **v2** run's pickles load under the v6 package name |
| `eclipse_v6/stage3.py` | display chain, unchanged, plus the merge→display handoff and `DISPLAY_GAMMA` |
| `pipeline.py` | the runner: stage0 → 1 → 2 → **calibration → re-registration → recalibration** → 3 |
| `run_from_v2_cache.py` | calibrate + merge on top of a finished v2 workdir, skipping ~50 min of registration |
| `report_display_exponent.py` | measures `DISPLAY_GAMMA` from the composites' brightness spread |
| `compare_registration_radiance.py` | diagnostic: how far the refined registration moves from stage 2's, with a control that must reproduce stage 2 exactly |
| `test_calib_synthetic.py` | plants a known response, asserts it is recovered. No GPU, ~3 s |
| `test_merge_smoke.py` | calibration → `load_radiance` → merge end to end on a small synthetic scene. Needs CUDA, ~6 s |
| `test_reregister_smoke.py` | seeds a wrong transform, asserts the radiance re-registration recovers the truth. Needs CUDA, ~4 s |

## Running

```bash
source /home/slavik/usr/anaconda3/etc/profile.d/conda.sh && conda activate e202602_eclipse
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

python v6/test_calib_synthetic.py                 # verify the numerics first (no GPU)
python v6/test_merge_smoke.py                     # verify the plumbing
python v6/test_reregister_smoke.py                # verify the re-registration

python v6/pipeline.py                             # full run, jpg mode
python v6/pipeline.py --merge legacy              # v2's merge, for side-by-side comparison
python v6/pipeline.py --n-exposure-refine 0       # honest nominal-shutter-times baseline
python v6/pipeline.py --refine-registration off   # stage 2's gamma-scaled alignment, as before
python v6/compare_registration_radiance.py        # how much does the refinement actually move?
python v6/run_from_v2_cache.py --cache-dir /home/slavik/tmp/eclipse_v2_jpg \
                               --workdir /home/slavik/tmp/eclipse_v6_from_v2
```

## What to look at in the output

| number | expected on the real JPEG data |
|---|---|
| `lsqr istop` | 2 (**3 means ill-conditioned and the answer is wrong**) |
| monotonicity violations | 0 |
| residual, core `v ∈ [0.25, 0.85]` | ~0.73 % rms, no bias |
| per-exposure corrections | 0.93–1.10, none pinned at the ±25 % clamp, largest on the shortest exposures |
| response local exponent | ~1.82 at `v=0.20` rising to ~3.82 at `v=0.80` (a power law would be constant) |
| re-registration movement | median ~0.36 px, max ~0.72 px over 14 pairs (all multiples of the 0.18 px search cell) |
| `max \|change in ln c\|` after recalibration | small — if it is not, one alternation step was not enough |
| coverage gaps outside the moon | 0 % |
| brightness spread p99/p50 | ~233 (v2's composite, which the display chain is tuned for, was 167) |

If the corrections come out at the clamp, or `istop == 3`, the alternation has been turned
into a joint solve — see `PHOTOMETRY_SPEC.md` §3.5 trap 1.

## Deliberate approximations

- The calibration is fitted on per-exposure averages of **encoded** values while the merge
  averages brightness. Wrong order in principle; the frames within a group are registered
  copies of the same scene, so the spread is noise-level and the bias second order. The curve
  is unknown when those averages are built, which is why.
- Registration and calibration are mutually dependent — the calibration needs corresponding
  pixels, the registration now needs the response curve — and the pipeline runs **one**
  alternation step, not a loop to convergence. The recalibration prints how far the shutter
  corrections moved; if that is not small, add a second step rather than assuming one sufficed.
- The refined registration masks the target with the longer exposure's coverage **un-warped**.
  The pair is within tens of pixels of alignment before the search starts and the mask edge
  sits in the saturated core, so the error is confined to where nothing is being matched.
- Pairs outside the calibrated exposure set (the two shortest) keep their stage-2 transform.
  Nothing downstream reads them; the merge starts at the third-shortest.
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
