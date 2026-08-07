# v6 — remove the static atmosphere before registration (v2: measured, not predicted)

## Why

In 2026 the Sun is **7.9° above the horizon** at mid-totality, seen from the ridge above
Pradoluengo (42.2323283 N, 3.2077811 W, 2131 m). At that altitude the atmosphere deforms and
dims the frames an order of magnitude harder than at 2024's 39.8°: on the order of **40 px of
anisotropic vertical compression** across the Z6 field, against 2.7 px in the dev set.

v5 registers with a rigid model (translation + rotation). A 40 px anisotropic compression has
nowhere to go in a rigid model, so it aliases into spurious translation and rotation —
HANDOVER §7's conclusion, unchanged.

v6 removes the part of the atmosphere that is **static**: predictable per frame, with no
reference to the image content changing shape frame to frame. Seeing, scintillation and
transparency fluctuation are explicitly **out of scope** — they are not static and they do not
warp the field (HANDOVER §10).

## What changed from the first version of this plan

The first version of this document computed the warp from physics: Bennett refraction as a
function of Sun altitude, site pressure and temperature, aerosol optical depth, etc. — all
predicted from the ephemeris and assumed atmospheric constants, then applied feed-forward to
undo the predicted warp. That chain has a lot of unmeasured inputs (pressure, temperature, the
lunar-rate factor, camera roll, haze), and the old plan spent most of its "traps" section on
how wrong each of those inputs could be.

The better question turns out not to need any of that. **Registration only cares about the
shape of the distortion, and that shape is exactly what a field of stars in the frame already
tells you** — where each star lands in the image vs. where its catalog position says it should
be, is the warp. No physics model is required to state this; it only has to be measured. And
because totality is bracket-shot with exposures long enough to reveal stars (see "Already
measured, below"), that measurement is available directly from the same frames being corrected,
several times across the ~100 s of totality.

Concretely: run the plate solve against a handful of the longer bracket exposures during
totality, fit each one's *total* pixel-level distortion (whatever mix of lens, refraction,
roll, and tracking error produced it — see "the degeneracy that doesn't need resolving" below),
track how that measured distortion drifts over the several calibration frames, fit a straight
line to the drift (linear-in-time, since totality is short enough that Sun altitude — and
hence refraction — moves close to linearly across it), and apply the fitted trend, evaluated at
every frame's own timestamp, to undo the frame-to-frame part of the warp.

**If the day doesn't cooperate — too few stars, or a poorly-conditioned fit — skip this
correction entirely and hand frames to v5 unmodified.** This is pure upside on top of v5's
existing rigid registration, not a dependency it needs.

## Already measured: this isn't a leap of faith

The question "will there be enough usable stars in the frame" already has an answer, sitting in
the repo's own verification suite, from a real total eclipse:

- `find_stars/check_06_anchor_census.py` measured, on the 84 real 2024 dev frames, the number
  of forced-photometry anchors (SNR > 5) per exposure. At 1.0 s — **the longest exposure
  `check_03_shooting_plan.py` actually plans for 2026** — the mean is **24.2 anchors**, out of
  25 confirmed stars total. At every exposure ≥ 0.25 s the count is 17 or more. Below ~0.03 s,
  essentially nothing is detected — so short frames were never going to reveal stars, with or
  without this plan.
- The same script fitted a 3-DOF rigid transform to those anchors and got sigma_translation as
  low as **0.022 px** at the best exposures — i.e. the existing star-detection and fitting
  machinery already resolves star positions far below the 40 px this plan needs to correct.
- The 2026 field itself was already censused: **44 Gaia G<9 stars** fall inside the Z6 FOV at
  the actual 2026 pointing, about 15% fewer than the 2024-equivalent count, using the same
  catalog query used to validate against the dev set.

None of this is new work — `starlib.solve_plate`, `starlib.gaia`, and the forced-photometry /
anchor-census pipeline (`dev_peaks`, `dev_forced`, `confirmed_stars`) are already built and
already validated end to end against a real eclipse. What v6 adds is: run that same machinery
on 2026 frames instead of 2024 ones, extend the fitted model by a few coefficients, and track
how the fit drifts across the sequence instead of taking one static solve.

## The degeneracy that doesn't need resolving

`starlib.project` already fits an affine transform plus radial lens distortion (`k1`, `k2`) per
frame (`solve_plate`). The obvious next step is to add the atmosphere's separable shape terms
(quadratic along the compression axis, linear along the other) as more parameters on the same
per-frame fit. **Don't do that naively** — with ~20-25 stars in one frame, a quadratic-in-y
atmosphere term and the lens's own `k1`/`k2` push pixels in almost the same direction, and nothing
in a single frame's data can tell them apart. Every fit will report a good residual and a
different, unstable split between "lens" and "atmosphere."

The resolution is to notice that **this split is not needed.** Registration only breaks on the
part of the warp that changes *between* frames; a component present identically in every frame
(the lens's own distortion, plus whatever part of the atmosphere is constant across the
calibration set) is registration-neutral — v5's existing per-frame affine fit already absorbs a
constant field shape, the same way it already absorbs the ~100 px of bulk refraction offset as
pure translation.

So: fit the full model (affine + k1/k2 + the new separable terms) independently per calibration
frame as one call each to the existing `solve_plate`/`project`, exactly as now — don't attempt a
joint multi-frame fit, and don't try to read the individual coefficients as physically
meaningful. Then, instead of trending the *raw coefficients* over time (unstable, per the
degeneracy above), evaluate each frame's fitted model as a **displacement at a fixed grid of
reference points** (the same idea as the old plan's `exact_offsets`, just measured instead of
computed from Bennett), and fit the straight line in time to those displacement values,
point by point on the grid. Subtract the sequence-mean line value and keep only the
slope-times-(t − t_mean) part — mirroring how `bulk_px` already removes its own mean in
`make_data_atmosphere.py`. That map is what gets applied to every frame. This sidesteps the
coefficient-identifiability problem completely: it never asks "how much was lens vs.
atmosphere," only "how did the star field's shape change over the sequence," which the data
answers directly.

## What's no longer needed from the physics-first version

- `Atmos`, `tau_*`, `n_air`, and the whole Bennett-refraction chain in `make_data_atmosphere.py`
  — not needed for the geometric correction. (Note: `make_data_atmosphere.py` itself is
  untouched and still useful for its original purpose, synthesizing a *known* atmosphere for
  the closed-loop test below.)
- a barometer/thermometer on site — was only there to firm up the physics amplitude.
- "set roll by bubble level, then recover the actual roll from the data" as a two-step process
  — the empirical fit recovers whatever roll is actually present, in one step, same as it
  recovers everything else.
- the amplitude-vs-`k1`/`k2` degeneracy as a *risk* — see above, it's now a non-issue because
  the plan no longer needs to resolve it.
- the assumed R/G/B effective wavelengths (`CHANNELS` in `make_data_atmosphere.py`, flagged
  before as the largest unquantified error in the dispersion correction) — dispersion is a
  per-channel star-position offset, so it can be measured the same way as the geometric warp,
  no wavelength assumption required. See "dispersion" below for the one wrinkle.

## What still needs the physics-first approach — explicitly out of scope for now

The extinction ramp and sky-background gradient are photometric (a brightness ratio across the
field), not astrometric. Star *positions* don't carry that information — it would need stellar
*photometry* (measured flux vs. catalog magnitude, across the field), a different and noisier
measurement, especially in the blue channel where extinction is strongest. **v6-v2 covers
geometry and dispersion only.** If the extinction ramp still matters for the photometry later,
it stays on the old physics-based `Atmos`/`channel_terms` path, or gets its own empirical
pass — not decided here.

## Dispersion: one wrinkle

Fitting each color channel's star positions independently, per frame, gets the dispersion
offset for free — but a star's flux, and hence its detection SNR, splits three ways across
channels, and blue is extinguished hardest. Doing three independent plate solves per frame
risks too few blue detections to fit anything. Better: detect and solve on the full-SNR fused
image (as for the geometric warp), then measure only the small *residual* centroid offset of
each channel relative to that already-known position — a differential measurement, which needs
far less individual-star SNR than an independent detection would.

## The axis assumption gets weaker, so make the model ask instead of assume

The first version of this plan derived which image axis is "vertical" (compression axis) from
the 2026 mount geometry (roll set so image-y = local vertical at mid-totality), and warned that
`make_data_atmosphere.py`'s own axis choice (image-x, from the 2024 plate solve) does not carry
over. That geometric reasoning about the *mount* is still correct and still worth having as a
prior. But since the correction is no longer derived from that reasoning, only informed by it,
**fit quadratic terms on both image axes**, not only the assumed vertical one, and let the
measured drift show which one actually carries the effect. If the mount reasoning is right, the
other axis's fitted term should come out near zero — which is itself a useful check that
nothing is misattributed (parity flip, wrong axis, etc.), rather than a silent assumption baked
into the model shape.

## Order of operations

1. decode raw → linear
2. **calibration pass**: on every bracket exposure empirically long enough to reveal stars
   (≥0.25 s, per the anchor census above), run the extended plate solve (fused-luminance
   detection, affine + k1/k2 + both-axis quadratic terms) against Gaia — one independent
   `solve_plate` call per calibration frame, seeded with an approximate pointing from the
   frame's EXIF timestamp (accurate to seconds, which is more than enough — the ephemeris
   moves the Sun/Moon by arcseconds per second) and `SITE_2026`
3. evaluate each calibration frame's fitted model as a displacement on a fixed reference grid;
   fit a line vs. time to each grid point's displacement across the calibration frames; keep
   only the slope × (t − t_mean) part
4. per-channel dispersion: same idea, but as the differential centroid offset described above,
   also trended linearly in time
5. evaluate both trends at every frame's own timestamp (star-bearing or not) to get that
   frame's correction
6. per-channel dispersion shift, applied post-demosaic (unchanged from the old plan; still the
   right point in the pipeline for the reasons given there)
7. separable geometric de-warp using the fitted (mean-removed) coefficients
8. mask the band de-compression can't fill — **mask, don't edge-extend**; that trick is only
   valid for `make_data_atmosphere.py`'s synthetic use, not real data
9. hand to v5 stage0 unmodified
10. detect stars for v5's own registration on the **un-dewarped** frames, transform the
    coordinates — resampling correlates noise and would invalidate HANDOVER §4.6's false-alarm
    thresholds, same reasoning as before

## Where the code goes

Same `vN/eclipse_vN/` convention as before:

| file | what |
|---|---|
| `v6/eclipse_v6/atmosphere.py` | new — but now a thin wrapper: extend `starlib.project`/`solve_plate` with the `len(p) > 8` block (both-axis quadratic terms), the calibration-frame selection, the per-grid-point linear-in-time trend fit, and evaluation at an arbitrary timestamp. No Bennett/`Atmos` physics needed here |
| `v6/eclipse_v6/inputs.py` | unchanged plan: an optional `correction` on `FrameSource`, applied at the end of `load_gray`/`load_rgb`. Same slot as before |
| `find_stars/starlib.py` | `project()` (line 526) gets the same kind of conditional block it already has for `k1`/`k2`, one level further, for the new terms. `solve_plate()` (571) gets more initial-guess elements. No new dependencies |

## Validation

1. **Closed loop on `data_atmosphere/`**, largely unchanged in spirit but now a better test than
   before: that set is the 2024 JPEGs pushed forward through a *known* synthetic atmosphere by
   `make_data_atmosphere.py`. Run the empirical calibration pass on it and check it recovers the
   known injected warp from `atmosphere.json`. This validates the measurement method itself
   against ground truth, not just "physics matches physics" as the old Round 1 did.
2. **Conditioning check on the real 2024 dev set** (not the synthetic one) — the one genuinely
   new check this version of the plan needs that the old one didn't: `check_06` counted anchors,
   but never tried fitting a quadratic distortion term to them. Fit the extended model to the
   real dev-set anchors and look at the fit's conditioning (covariance / condition number), not
   just the raw count — confirms the quadratic term is actually constrained by the real spatial
   distribution of those ~20-25 stars, not just their number.
3. **Does it help v5?** Run v5 on `data_atmosphere/` with and without the v6 pre-transform and
   compare the rigid-registration residual. Unchanged from before — if this number doesn't drop,
   stop and find out why before touching real data.

## Risks, in order

1. **Haze.** Still the dominant unknown, and now it bites twice: it's still the single biggest
   photometric risk (HANDOVER §7, 1-3 mag), and it now also determines whether enough stars are
   even visible to fit anything. If this correction becomes infeasible on the day, that is very
   likely why.
2. **Temporal coverage of the calibration frames.** The linear-in-time fit is only as good as
   its span; if the star-revealing exposures in the bracket happen to cluster rather than spread
   across totality, the fit is extrapolating right where it matters most (near C2/C3). Worth
   checking against the actual bracket cycle timing (`check_03_shooting_plan.py`) before the
   day, not discovering it live.
3. **Star ID errors.** A mismatched star (plausible if two similar-magnitude candidates fall
   within the pointing tolerance of each other) silently corrupts that frame's calibration
   rather than failing loudly — worth an outlier/residual check in the fit, not blind trust in
   every match `solve_plate` returns.
4. **The lunar-rate factor 2** (HANDOVER §6) still matters for the star field's own apparent
   motion across the sequence, independent of this plan. Worth resolving from the 2024 data
   regardless.
5. **If it doesn't work, it costs nothing.** Per the opening: this whole correction is
   conditional on the day's conditions. No usable stars, or a badly conditioned fit, means
   skip it and fall back to v5 unmodified — decided in advance, not something to agonize over
   live during totality.

## Order of work

| round | what | why first |
|---|---|---|
| 1 | extend `starlib.project`/`solve_plate` with the both-axis quadratic terms; unit-test on synthetic planted stars (recovers a known warp, same shape as `v5/test_calib_synthetic.py`) | the only genuinely new numerics, and it's a pure function — no images, no dependencies |
| 2 | conditioning check on the real 2024 anchors (validation #2) | tells you, before writing anything else, whether this is worth pursuing at all |
| 3 | the reference-grid trend fit (evaluate → fit line → mean-removed correction) and the inverse-transform `FrameSource`; closed-loop test on `data_atmosphere/` (validation #1) | proves the measurement pipeline end to end against known ground truth |
| 4 | v5-with-and-without comparison (validation #3) | the real go/no-go gate |
| 5 | dispersion's differential-centroid measurement | smaller, and depends on the same detections as round 1 |

Rounds 1-2 are cheap and answer the viability question directly — if round 2 shows the real
2024 anchors can't constrain a quadratic term, stop there and skip this correction for 2026,
per the risk #5 above.
