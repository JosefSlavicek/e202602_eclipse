# v7 — bracketed exposures on one physical brightness scale

v7 started as a self-contained copy of v6 (no cross-version imports; the `vN/eclipse_vN/`
convention v3 established). It has since diverged in two ways:

- **NEF-only.** The JPG/atmosphere/inject input modes are gone; there is one source class
  (`NefSource`) and one input contract. `--merge legacy` (v2's per-pair exponent chain) is
  still selectable for comparison, but nothing JPG-specific remains.
- **Merge weight moved to ingestion.** Previously the cross-exposure merge weight
  (`merge._window`) was derived *after* dark/flat/calibration, from each exposure group's
  averaged, calibrated radiance — clamped to `[0, 1/t_eff]` just to give the window a domain
  to operate on. Now `rawprep.py` measures overburn and the merge weight **once per raw
  frame, at ingestion**, directly off the unmodified decode, before dark/flat correction
  ever runs. Those per-frame maps ride through the same warps the image data goes through
  (intra-exposure pose in `merge.average_exposure_radiance`, then the cross-exposure chain
  in `merge.merge_to_composite`), landing pixel-aligned with the data they weight — and
  since the weight is never in radiance units, it needs no `[0, 1/t_eff]` clamp at all.

The rest of this README describes v6's history relative to v5 and v2, which still applies to
v7's calibration/registration machinery (unchanged) even though the input layer and merge
weighting described above have moved on.

v5 is a self-contained copy of v2 with the radiometry replaced as specified in
`../PHOTOMETRY_SPEC.md`.

**What changed.** v2 fitted a separate exponent to each adjacent pair of exposures,
multiplied those 16 exponents down a ladder to put everything on one scale, and combined
with a hand-shaped bell weight. That is unsound: the exponents came out spread 0.87–1.39
when a single camera response demands they be equal, each fit silently absorbed whatever
else differed between its two frames, the errors compound along the chain, and the weight
was chosen by eye. v5 recovers the camera's response curve **once** for the whole bracket —
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
| `eclipse_v7/calib.py` | recovers `ln f` at 64 knots + per-exposure shutter corrections (Debevec–Malik, sparse `lsqr`, corrections by alternation) |
| `eclipse_v7/rawprep.py` | measures per-frame overburn mask + merge weight once, straight off the unmodified raw decode, before dark/flat correction; also bakes dark/flat correction into a cached "corrected" array once fit, so the NEF is decoded exactly once |
| `eclipse_v7/merge.py` | per-frame brightness, inverse-variance merge, common moon blanking, `NO_DATA` sentinel; weight comes from `rawprep`'s per-frame maps, warped alongside the data (not derived from calibrated radiance); also keeps every exposure's own weight map (before summing) and writes it as `v7-stage3_weights.npy` — large by design, see below |
| `eclipse_v7/reregister.py` | redoes the cross-exposure registration on calibrated radiance; writes `v7-stage2r.pkl` |
| `eclipse_v7/warp.py` | reference-grid warping and the cross-exposure chain, shared by the calibration, both merges and stage 3 |
| `eclipse_v7/inputs.py` | `NefSource`: finds `lights/`/`darks/`/`flats/` under the folder it's given; `load_gray` reads the rawprep-corrected cache (`set_cache_dirs`), `load_overburn`/`load_weight` read the raw-measured cache, `load_radiance` divides by the effective exposure (shutter-corrected via `set_calibration`) |
| `eclipse_v7/darkcal.py` | per-pixel dark-current model `bias + rate * t`, fit by OLS over every `darks/` frame at once against the plain EXIF-reported `t` — deliberately no shutter-time correction (see its module docstring for why matched-by-label subtraction doesn't need one); writes `v7-dark.pkl` |
| `eclipse_v7/flatcal.py` | per-pixel flat field, **high-frequency only** (dust shadows + chip sensitivity, not the smooth optical vignetting -- each dark-subtracted frame has its own sigma=24px Gaussian blur, mirror-padded, subtracted from it first), averaged from every `flats/` frame (one exposure time, checked), normalized so it corrects a perfectly uniform frame back to a uniform frame averaging ~1.0, then clamped to `[0.95, 1.05]`; writes `v7-flat.pkl` |
| `eclipse_v7/stage3.py` | display chain, unchanged, plus the merge→display handoff and `DISPLAY_GAMMA` |
| `pipeline.py` | the runner: **raw prep** → **dark model** → **flat model** → **apply dark/flat** → stage0 → 1 → 2 → calibration → re-registration → recalibration → 3 |
| `test_darkcal_synthetic.py` | plants a known per-pixel bias/rate map (with hot pixels) under real per-exposure timing errors, asserts the (uncorrected, EXIF-trusting) fit still recovers bias/rate to noise-level accuracy. No GPU, no files, ~1 s |
| `test_flatcal_synthetic.py` | plants a known vignetting pattern (radial falloff + a dust shadow) under a known dark bias/rate, asserts only the dust shadow is recovered (the radial falloff is dropped by the high-pass step) with the planted defect correctly pinned at the `[0.95, 1.05]` clamp, and that it corrects a uniform frame back to close to average 1.0; separately checks the high-pass step's mirror-padded edges don't bias a uniform frame. No GPU, no files, ~1 s |
| `report_display_exponent.py` | measures `DISPLAY_GAMMA` from the composites' brightness spread |
| `compare_registration_radiance.py` | diagnostic: how far the refined registration moves from stage 2's, with a control that must reproduce stage 2 exactly |
| `test_calib_synthetic.py` | plants a known response, asserts it is recovered. No GPU, ~3 s |
| `test_merge_smoke.py` | calibration → `load_radiance` → merge end to end on a small synthetic scene. Needs CUDA, ~6 s |
| `test_reregister_smoke.py` | seeds a wrong transform, asserts the radiance re-registration recovers the truth. Needs CUDA, ~4 s |

## Running

```bash
source /home/slavik/usr/anaconda3/etc/profile.d/conda.sh && conda activate e202602_eclipse
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2

python v7/test_calib_synthetic.py                 # verify the numerics first (no GPU)
python v7/test_darkcal_synthetic.py               # verify the dark-model numerics (no GPU)
python v7/test_flatcal_synthetic.py               # verify the flat-model numerics (no GPU)
python v7/test_merge_smoke.py                     # verify the plumbing
python v7/test_reregister_smoke.py                # verify the re-registration

python v7/pipeline.py                             # full run
python v7/pipeline.py --merge legacy              # v2's merge, for side-by-side comparison
python v7/pipeline.py --n-exposure-refine 0       # honest nominal-shutter-times baseline
python v7/pipeline.py --refine-registration off   # stage 2's gamma-scaled alignment, as before
python v7/pipeline.py --dark-model off            # skip the dark-current subtraction
python v7/pipeline.py --flat-model off            # skip the flat-field correction
                                                    # (auto by default: used iff nef-dir/flats exists)
python v7/compare_registration_radiance.py        # how much does the refinement actually move?
```

## Per-exposure weight stack

`v7-stage3_weights.npy` — shape `(n_exp, H_crop, W_crop)` float32, one full-resolution
weight map per exposure in the merge's cross-exposure chain, cropped identically to
`v7-stage3_composite.npy`. `v7-stage3_weights.npy[k]` is exactly the weight exposure
`v7-stage3_weights_exposures.npy[k]` (that array's reported shutter time, seconds)
contributed to `composite_variance` at every pixel, *before* summing across exposures —
`v7-stage3_weights.npy.sum(axis=0) == 1 / v7-stage3_variance.npy` at every covered pixel.
Large on purpose (~100MB per exposure, tens of exposures): it exists to let you look at
which exposure dominates a given pixel, not to be loaded casually. `argmax(axis=0)` against
`v7-stage3_weights_exposures.npy` gives the dominant exposure per pixel cheaply, if that is
all you need.

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
- The two shortest exposures are dropped (inherited from v2, no recorded reason; kept for
  comparability).
- `GAIN_E_PER_UNIT` and `READ_SIGMA` in `inputs.py` are placeholders setting only relative
  weighting between raw frames. Measure them from a flat-field pair: variance against mean is
  a straight line, slope `1/gain`, intercept `read_noise²`.
- `test_merge_smoke.py`'s `SyntheticSource` is deliberately still JPEG-like (8-bit,
  non-power-law response) even though production is NEF-only — it exists to exercise
  `calib.py`'s general curve-recovery machinery end to end, which a linear NEF source never
  exercises on its own.
- `pipeline.ipynb` is inherited from v1: its stage 0–2 cells predate the current NefSource-based
  inputs API and its saved outputs are from an old v1 run. `pipeline.py` is the maintained runner.
