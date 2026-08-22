# Putting bracketed exposures on one physical brightness scale

**Status:** specified and prototyped once against the JPEG data set; not currently present in
any pipeline version. This document is self-contained — it is everything needed to implement
the change from scratch.

---

## 1. What is being changed, in one paragraph

The pipeline photographs the corona with a bracket of 17 shutter times and must reduce them to
one brightness number per pixel. The existing method fits a separate exponent to each adjacent
pair of exposures, multiplies those exponents down the ladder to put everything on one scale,
and combines with a hand-shaped weighting curve. That method is unsound: the exponent it fits
cannot represent the camera's actual response, its errors accumulate multiplicatively along the
chain, and the weighting is a guess. The replacement recovers the camera's response curve once
for the whole bracket — together with corrections to the shutter times, which the camera
reports inaccurately — converts every frame independently to physical brightness, and combines
with weights derived from each measurement's actual uncertainty.

---

## 2. The change explained

### 2.1 What the problem is

You photograph the corona with a bracket of exposures — in this case 17 different shutter
times, from 1/4000 s to 2 s — because no single exposure can hold both the blinding inner
corona and the faint outer streamers. Several frames at each shutter time. The job is to turn
all of them into one number per pixel: how bright the sky actually is there.

Two terms, both needed below.

**Brightness** here means light arriving per unit time — a property of the sky, the same in
every frame. During an exposure of length `t` the sensor accumulates `brightness × t`. That
product is what the camera measures; the brightness is what we want.

**Response curve.** A camera doesn't store the accumulated light. It stores a number (0–255 in
a JPEG) that is some increasing function of it, chosen so the picture looks right on a screen —
a gamma curve plus a contrast S-curve. Write `f` for the function that undoes this: given a
stored value `v`, `f(v)` is the accumulated light that produced it. So for pixel `i` in the
frame with shutter time `t_k`:

```
f(v_ik) = L_i · t_k          L_i = the brightness we want
```

`f` is unknown, because the camera doesn't tell you what it did.

### 2.2 How it was computed before

**Step 1 — match neighbouring exposures.** Take two shutter times, `t₀` shorter and `t₁`
longer. Assume their stored values are related by a single multiplying factor, and that the
factor is the exposure ratio raised to some power. Search for the exponent `g` that best
matches the two frames:

```
minimise  Σ | v_i0  −  v_i1 · (t₀/t₁)^(1/g) |     over mid-range pixels
```

Done separately for each of the 16 adjacent pairs, giving 16 exponents `g₁…g₁₆`.

**Step 2 — put everything on one scale.** The factor taking exposure `k` onto the shortest
exposure is the running product down the ladder:

```
s_k = (t₀/t₁)^(1/g₁) · (t₁/t₂)^(1/g₂) · … · (t_{k-1}/t_k)^(1/g_k)
```

**Step 3 — combine.** A weighted average of the rescaled stored values:

```
result_i = Σ_k w_ik · (v_ik · s_k)  /  Σ_k w_ik       with   w = exp(−((v − 0.5)/0.2)²)
```

That weight is a bell curve peaking at stored value 0.5 — "trust mid-tones, distrust very dark
and very bright pixels."

### 2.3 What is wrong with that

**The single-factor assumption is only valid if `f` is a pure power law.** If `f(v) = v^γ`,
then multiplying the light by a constant does multiply the stored value by a constant, and one
exponent describes the pair exactly. But `f` is a gamma curve *plus* a contrast curve, so it
isn't a power law, and no single number can describe the relationship between two frames. The
evidence is in the fitted numbers themselves: a camera has one response curve, so all 16
exponents must come out equal. They came out spread from **0.87 to 1.39**, each stable and
repeatable. Each fit was silently absorbing whatever else differed between its two frames —
shutter inaccuracy, sky transparency drift — and settling wherever that pair's particular
brightness distribution pulled it.

**The errors multiply.** `s_k` is a product of `k` independently fitted numbers, so a mistake
in one link tilts every exposure below it, and nothing ever checked whether the ladder was
globally consistent. For this data the chained result differed by a factor of 1.6 from what a
single self-consistent exponent would give, with no way to tell which was closer.

**The weight is a guess.** It contains no exposure time, and it isn't derived from how
uncertain each measurement actually is — the shape and width were chosen by eye.

### 2.4 How it is computed now

**Recover the response curve once, instead of 16 exponents.** Take the equation
`f(v_ik) = L_i · t_k`, add an unknown correction `c_k` to each exposure time (the camera's
reported shutter times are not exact), and take logarithms:

```
ln f(v_ik)  =  ln L_i  +  ln t_k  +  ln c_k
```

This is linear in the unknowns, which are: the values of `ln f` at about 64 points along the
stored-value axis; one `ln c_k` per exposure; one `ln L_i` per sample pixel. With thousands of
sample pixels seen across many exposures this is a hugely overdetermined linear system, solved
by least squares, with a smoothness requirement on `f` (it's a physical curve, not noise) and
two normalisations to pin down the two arbitrary overall scales. This is a standard technique
in high-dynamic-range imaging, due to Debevec and Malik.

One subtlety: the exposure corrections cannot be solved at the same time as everything else. In
the equation, `ln t_k + ln c_k` appears only as a sum, so a free `c_k` makes the known shutter
times carry no information at all, and the solution becomes ambiguous. Instead: solve the curve
using the shutter times as reported, then look at each exposure's average leftover discrepancy
— a systematic offset for one exposure *is* its timing error — fold that into `c_k`, and
repeat. A few rounds converge. On this data the corrections came out between 0.93 and 1.10,
largest on the shortest exposures, which is what you'd expect from mechanical shutter timing.

**Convert each frame independently.** No chain, no reference exposure:

```
L̂_ik = f(v_ik) / (t_k · c_k)
```

Every exposure is tied straight to the same scale, so there is nothing to accumulate.

**Also: convert before averaging, not after.** Several frames share each shutter time.
Previously their stored values were averaged and the result converted; now each frame is
converted and the brightnesses averaged. The order matters because `f` is curved — the average
of encoded values is not the encoding of the average.

**Weight by actual uncertainty.** When you average independent measurements of the same
quantity, the most precise result comes from weighting each by `1/σ²`, where `σ` is its
uncertainty. So we compute `σ` for each estimate rather than guessing a weight:

```
σ_ik  =  (noise in the stored value) × (slope of f at v_ik) / (t_k · c_k)   plus a second term, below

w_ik  =  1 / σ_ik²
```

The slope of `f` appears because it converts an error in the stored value into an error in
light: where the curve is steep, one step of the 256 available levels spans a lot of light, so
that reading is intrinsically imprecise. Dividing by `t` appears because the same uncertainty
in accumulated light means a smaller uncertainty in brightness when the exposure was longer.

The second term is the calibration's own error, measured rather than assumed: after solving, we
check how much the exposures still disagree at each stored value and add that in as extra
uncertainty. It is small in the middle of the range and large at both ends.

And stored values at the very top of the scale get weight exactly **zero**, not merely small. A
saturated pixel does not report the light that fell on it; it reports "at least this much", and
its stored value is necessarily too low. Including it at any weight biases the result.

Three useful behaviours now follow from the formula instead of being chosen: long exposures
automatically dominate the faint outer corona, short exposures dominate the bright inner
corona, and readings from the parts of the response curve we know least well are automatically
discounted.

**A note on which noise is modelled.** For an 8-bit JPEG the dominant uncertainty is
quantization — one step of the 256 available levels is 0.4% of full scale, larger than the
random scatter of photon arrivals at mid-tones. Quantization does not shrink with more light,
so the JPEG model uses a constant uncertainty in the stored value, not a `√signal` term.
Longer exposures still come out better, but through `1/t` combined with the curve's slope
rather than through photon statistics: doubling the exposure reduces the uncertainty by about
0.77 for a curve of the measured shape. For raw files, where there are ~16000 levels and
quantization is no longer the bottleneck, photon noise *is* modelled explicitly as
`variance ∝ signal/gain + read_noise²`, giving the familiar `1/√t` improvement. Separately,
averaging `n` frames of the same shutter time divides the variance by `n`, so an exposure with
more frames earns proportionally more weight; that is exact, not approximate.

### 2.5 The result

The old scheme's chained factor was ambiguous by a factor of 1.6. After the change, across the
reliable part of the stored-value range, the 15 exposures agree with each other on the
brightness of the same piece of sky to **0.73% root-mean-square, with no systematic offset**.

A side finding worth keeping: the same measurement shows that below roughly 10% of full scale,
an 8-bit JPEG carries essentially no usable brightness information — the recovered curve is
wrong by 50% and more down there. That is a property of the input format, not of the method,
and it is one reason to prefer raw files.

---

# 3. Implementation specification

Everything below is for the implementing agent. Section 2 is the rationale; this section is
the contract.

## 3.1 Environment

```
python      /home/slavik/usr/anaconda3/envs/e202602_eclipse/bin/python
activate    source /home/slavik/usr/anaconda3/etc/profile.d/conda.sh && conda activate e202602_eclipse
GPU         CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
available   torch 2.10 (+cu130), scipy 1.17, matplotlib 3.10, rawpy, PIL, numpy
```

`eclipse_v2/device.py::configure_cuda_visible_devices` sets those env vars with `setdefault`,
so the shell wins. CUDA is required (`require_cuda`).

## 3.2 Data and reference artefacts

```
/home/slavik/e202602_eclipse/data                 84 real corona JPGs, 17 exposures 0.00025–2 s
                                                  THE ONLY REAL CORONA DATA. No corona raws exist.
/home/slavik/e202602_eclipse/data_atmosphere      same frames re-rendered for low sun
/home/slavik/tmp/eclipse_v2_jpg/                  cached v2 run: v2-stage0/1/2.pkl
                                                  -> reuse these, registration takes ~50 min
/home/slavik/tmp/eclipse_v2_run_20260804/         full v2 run incl. v2-stage3_composite.npy
```

The `.NEF` files under `/home/slavik/tmp/` are camera test and black frames. The `inject` input
mode fabricates corona content from the JPGs, so it cannot validate radiometry — treat
`nef`/`inject` results as plumbing checks only.

## 3.3 Baseline: what exists in v2 and must be understood first

Read these before writing anything. Paths relative to
`/home/slavik/e202602_eclipse/src/e202602_eclipse/`.

| file | what matters in it |
|---|---|
| `v2/eclipse_v2/inputs.py` | `FrameSource` ABC; `JpgSource`, `AtmosphereSource(JpgSource)`, `NefSource`, `NefInjectSource(NefSource)`. `load_gray(ii, device) -> float32 (H,W) in [0,1]`. `is_linear` flag. `make_source()`, `attach_source()` (sources are not pickled). `BLACK_LEVEL=1008`, `WHITE_LEVEL=16383`. |
| `v2/eclipse_v2/stage0.py` | `ImageInfo` (`.path .width .height .exposure_time .timestamp .moon=(i,j,r) .source .link`), moon detection, intra-exposure registration, `find_moon`. |
| `v2/eclipse_v2/stage1.py` | pose fit; `opt_results[exp] = {"abs_xy": (n,2), "abs_angle_t": (n,)}`. |
| `v2/eclipse_v2/stage2.py` | **the code being replaced.** `estimate_gamma` (the per-pair exponent search), `mae_out_of_moon` (restricts to `img0 ∈ [0.01,0.45]`, outside moon), `cross_exposure_consecutive_pairs`, `register_cross_exposure`. `save_pickle` writes `cross_reg{(t0,t1):(di,dj,rot)}` then `gamma_by_pair{(t0,t1):g}`. |
| `v2/eclipse_v2/stage3.py` | `Stage3Context` dataclass; `load_inputs`; `build_per_exposure_averages`; **`warp_merge_to_composite`** (contains `_scale_to_ref` = the chain, `_weight_from_value` = the bell, `_warp_to_ref`, and `mask_0`); then the display chain: `crop_and_save_composite`, `radial_normalize_display`, `fft_unsharp_and_save`, `rgb_vignette_and_radial_pickle`. |
| `v2/eclipse_v2/utils.py` | `compute_weighted_average` (per-exposure stack mean of **encoded** values), `apply_transform_single`, `load_grayscale`, `moon_median`. |
| `v2/pipeline.py` | the runner; stage0 → stage1 → stage2 → stage3. |

**Inherited behaviour to preserve unless deliberately changed:**

- `load_inputs` does `ctx.exposure_times_sorted = sorted(ctx.exposure_groups.keys())[2:]` — the
  **two shortest exposures are dropped**, so 15 of 17 are used and `t_ref = 0.001`. No recorded
  reason. Keep it for comparability; note it.
- `warp_merge_to_composite` multiplies every warped exposure by `mask_0`, the shortest
  exposure's coverage mask. Since the shortest exposure has the largest detected moon, this
  blanks all exposures to the largest disk. **This must be preserved** — the detected moon
  radius drifts 5.94 px across the bracket (316.03 px at 1/4000 s → 310.19 px at 2 s, ordered
  by exposure, scatter within each group only 0.0–0.3 px) because glare in long exposures makes
  the edge-finder place the limb further in. Without a common blanking radius, the annulus
  between the smallest and largest disk is populated only by saturated long exposures.
- `_moon_mask` style blanking uses `dist > r + 2` per frame.
- The whole pipeline is **grayscale** — colour is fabricated at the very end from
  `RGB_DIM_QUOTIENTS`. `load_gray` averages R,G,B, so the recovered response is that of the
  mean of three channels, each of which really has its own curve and its own clipping point.
  Accept this; note it as the natural next improvement.

## 3.4 Work plan

Create a new version directory as a self-contained copy, following the existing `vN/eclipse_vN/`
convention (v3 established this: vendored copies, no cross-version imports).

```
cp -r v2 vN && rm -rf vN/__pycache__ vN/eclipse_v2/__pycache__ && mv vN/eclipse_v2 vN/eclipse_vN
# then sed:  eclipse_v2 -> eclipse_vN,  v2- -> vN-,  ECLIPSE_V2 -> ECLIPSE_VN
```

To consume a cached v2 run's pickles under the new package name, add a module-alias shim —
`sys.modules.setdefault("eclipse_v2", <new package>)` plus each submodule — because the pickles
name `eclipse_v2.stage0.ImageInfo`. The classes are identical, so aliasing is sufficient.

### Step 1 — new module `calib.py`: recover the response curve

#### 1a. Gather samples

Needs corresponding pixels across exposures, so it runs **after** registration. That creates an
apparent circularity: cross-exposure alignment currently uses the fitted exponent. Resolve it by
keeping stage2 exactly as it is and using its exponents **for alignment only** — Fourier
alignment cares about structure, not absolute scale, so a slightly wrong scale still registers.
The exponents stop determining any brightness.

1. Build the per-exposure averages of encoded values (reuse `build_per_exposure_averages`).
2. Build the per-exposure chain of transforms onto the reference grid. Extract this from
   `warp_merge_to_composite` into a shared helper — it is needed by both the calibration and the
   merge:
   ```
   build_chains(exposure_times_sorted, available_exposures, cross_reg, t_ref)
       -> (exposures_with_chain, {exp: [ (di,dj,rot), ... ]})
   ```
3. Compute the coverage common to all exposures by warping a constant image per exposure and
   taking the elementwise minimum; require `>= 0.999`.
4. Sample pixels **stratified in log radius** from the moon, outside `moon_r + 8`. This matters:
   the corona spans ~1e4, so a pixel near the limb is only ever well exposed in the shortest
   frames and one at the frame edge only in the longest. Uniform-over-area sampling piles almost
   everything into one part of the curve. Use ~48 log-radial bins × ~220 samples.
5. Warp **one exposure at a time** and gather values at the fixed sample coordinates; free each
   before the next. Peak memory then is one full-resolution frame, not the whole stack.

Returns `V, valid` of shape `[n_exposures, n_samples]` plus the exposure list.

#### 1b. Solve

Unknowns: `g[0..K-1]` = `ln f` at `K = 64` knots evenly spaced on `[0,1]`; `lnE[0..M-1]` = one
per sample pixel. **`d_k = ln c_k` is NOT an unknown of the linear system** — see the trap in
§3.5. It enters as a fixed offset and is refined by alternation.

Because grayscale values are the mean of three 8-bit channels they are not on the 256-integer
grid, so `f` must be interpolated: `g(v) = (1−α)·g[z] + α·g[z+1]` for the bracketing knots. Still
linear in the unknowns.

Rows of the sparse system, one per usable `(pixel, exposure)`:

```
w · [ (1−α)·g[z] + α·g[z+1] − lnE_i ]  =  w · ( ln t_k + d_k )
```

Plus a smoothness prior on the curve, for `z = 1 … K−2`:

```
λ · s(z) · ( g[z−1] − 2·g[z] + g[z+1] )  =  0        λ ≈ 200,  s(z) = hat(knot z) + 0.05
```

The `s(z)` factor lets the prior govern the unobserved ends of the curve without fighting the
data in the middle.

Plus one gauge row fixing the single remaining global degeneracy (`g += c`, `lnE += c`):

```
BIG · g[K/2] = 0            BIG ≈ 1e4
```

Solve with `scipy.sparse.linalg.lsqr(A, rhs, atol=1e-12, btol=1e-12, iter_lim=8000)`. Check
`istop`: 1 or 2 is fine, **3 means ill-conditioning and the answer is wrong** (see §3.5).

The fit weight `w` is a triangular hat over `[FIT_VALUE_LO, FIT_VALUE_HI]`, zero outside. Note
this is not the same thing as the merge weights and does not contradict replacing them — here it
only selects which samples constrain the calibration, where preferring well-exposed pixels is
correct and standard.

Discard sample pixels usable in fewer than 2 exposures; they constrain nothing.

#### 1c. Refine the exposure corrections by alternation

```
d = zeros(n_exposures)
repeat 6 times:
    solve g, lnE  with  ln_t_effective = ln t + d          # well-posed
    resid = g(v) − lnE_i − (ln t + d)_k                    # per (pixel, exposure)
    step_k = weighted mean of resid over exposure k        # weights = the hat weights
    d += step ; d -= mean(d) ; d = clip(d, ±ln(1.25))
```

Each subproblem is well-posed and the clamp bounds how far the ladder can be rewritten. Expose
`n_exposure_refine=0` as the honest nominal-times baseline for comparison.

#### 1d. Constants, with the measurements that set them

```
FIT_VALUE_LO = 0.10        # NOT 0.02 — see below
FIT_VALUE_HI = 0.98
N_KNOTS = 64
SMOOTHNESS_LAMBDA = 200.0
GAUGE_WEIGHT = 1e4
N_RADIAL_BINS = 48
SAMPLES_PER_BIN = 220
MIN_EXPOSURES_PER_SAMPLE = 2
N_EXPOSURE_REFINE = 6
MAX_LN_CORRECTION = ln(1.25)
```

`FIT_VALUE_LO = 0.10` is measured, not chosen. On a synthetic bracket with a known response and
1/255 noise, the recovered curve's error by band:

```
v ∈ [0.02,0.05)  ~190%       v ∈ [0.20,0.95)  < 1%
v ∈ [0.05,0.10)   ~56%       v ∈ [0.95,0.98)  ~1.7%
```

At `v = 0.02` a code step is ~1/5 of the value at the curve's steepest point — there is no
information there. Cutting at 0.10 costs no coverage: each exposure still spans a factor of ~90
in light between `v = 0.10` and `v = 0.98`, far more than the ~2× steps between exposures.

#### 1e. Derived outputs `calib.py` must expose

```
response_lut(result)         -> (v grid, f(v))                dense, from the knots
response_slope_lut(result)   -> (v grid, df/dv)               np.gradient, floored at 1e-12
systematic_sigma(result)     -> (v grid, sigma_fraction)      see below
effective_gamma(result, lo=0.25, hi=0.85) -> float            log-log least-squares slope
monotonicity_report(result)  -> counts of non-monotone steps
```

`systematic_sigma` is the calibration's own error as a fraction of brightness, measured per value
band from the fit residuals as `sqrt(mean² + var)` — bias and scatter both count. Interpolate
gaps; outside the fitted range multiply the edge value by ~4 rather than pretending
extrapolation is as good as measurement; floor at 0.002. Measured shape on the real JPEG data:

```
v      0.00   0.12   0.19   0.27   0.38   0.52   0.56   0.74   0.85   0.92   0.96   1.00
sigma  44.9%  11.2%   1.8%   0.9%   0.6%   0.4%   0.4%   0.9%   1.3%   4.4%  11.9%  47.8%
```

Note the shape: a U with its minimum 0.38% at `v ≈ 0.56`, **asymmetric** — the dark side degrades
much faster than the bright side. That is the measured version of what the old bell curve was
guessing at.

`effective_gamma` must use the log-log **slope**, not a secant through `v = 1`. The two differ
badly here (1.63 vs 2.62) because the extrapolated toe does not reach zero (`f(0) ≈ 0.106`),
which inflates the secant. It is needed only for display scaling, never for radiometry.

### Step 2 — `inputs.py`: the source layer's contract with the merge

Add to `FrameSource`:

```
value_sigma = 1.0/255.0          # uncertainty of one frame's stored value, in stored-value units
set_calibration(calib_result)    # caches response LUT, slope LUT, systematic sigma, corrections
effective_exposure(ii)           # ii.exposure_time * c_k
load_radiance(ii, device) -> (radiance, variance, valid)
```

`load_gray` keeps its current meaning: the **stored** value, which is what stage0/1/2 register on
and what the calibration is fitted to. Radiometry goes only through `load_radiance`.

`JpgSource.load_radiance`:

```
v        = load_gray(ii)
t_eff    = ii.exposure_time * c_k
radiance = interp(v, response_lut) / t_eff
var      = (value_sigma * interp(v, slope_lut) / t_eff)**2  +  (interp(v, sys_sigma) * radiance)**2
valid    = (v > CLIP_LO) & (v < CLIP_HI)          # CLIP_LO = 0.02, CLIP_HI = 0.98
```

Implement the LUT lookup on GPU with a `torch.searchsorted`-based `np.interp` equivalent.

`NefSource.load_radiance`:

```
signal   = load_gray(ii)                       # rawpy postprocess(gamma=(1,1), no_auto_bright)
signal  -= dark_current_per_s * exposure_time   # hook, None by default
signal  /= vignette                             # hook, None by default
radiance = signal / t_eff
var      = (clamp(signal,0)/GAIN_E_PER_UNIT + READ_SIGMA**2) / t_eff**2
valid    = decoded < CLIP_HI                    # test the DECODED value, not the corrected one
```

No systematic term: there is no fitted curve to be wrong. `GAIN_E_PER_UNIT` and `READ_SIGMA` are
placeholders that set only relative weighting; measure them from a flat-field pair (variance vs
mean is a straight line — slope `1/gain`, intercept `read_noise²`). `set_calibration` is still
honoured for the exposure-time corrections, which apply regardless of file format.

`is_linear` **does not disappear.** It is gone from radiometry, but stage2 still needs it to
scale one exposure roughly onto its neighbour before Fourier alignment, and for raw that scaling
is exactly the exposure ratio. Retitle it as a registration detail in the docstring.

Beware: `AtmosphereSource.__init__` and `NefInjectSource.__init__` call `super().__init__(...)`;
if `FrameSource` gains an `__init__`, the chain must reach it.

### Step 3 — new module `merge.py`: combine in physical brightness

```
average_exposure_radiance(group, abs_xy, abs_angle_t, source, device)
    -> (Lbar, Vbar, covered, moon_out)
```

Per frame: `L, var, valid = source.load_radiance(...)`; `m = valid · moon_mask`; warp
`L·m`, `var·m²`, `m` and the moon mask alone by that frame's stage-1 pose. Then

```
Lbar = Σ(m·L) / Σm                Vbar = Σ(m²·var) / (Σm)²
```

which is the ordinary variance of a weighted mean, so more frames automatically earn more weight.
Return the moon geometry separately from `covered`, because `covered` also excludes saturated
pixels and the merge needs the geometry on its own.

```
merge_to_composite(ctx, source)
```

One exposure at a time — build, warp, accumulate, free. Peak memory one full-res frame.

```
w        = 1/Vbar   where covered and finite, else 0
wL_ref   = warp_to_ref(w · Lbar)          # weight BEFORE warping — see §3.5
w_ref    = warp_to_ref(w)
cover    = warp_to_ref(ones);  inside = cover >= 0.999
accumulate sum_wL, sum_w  in float64 over `inside`
common_moon_out = elementwise min over exposures of warp_to_ref(moon_out)
```

Finally blank the union of the moon disks (equivalently: the largest), which is what v2's
`mask_0` achieved — set `sum_w = sum_wL = 0` where `common_moon_out < 0.999`. Then

```
radiance = sum_wL / sum_w      where sum_w > 0, else NO_DATA = -1.0
variance = 1 / sum_w
```

Record and print, **separately**, pixels masked as moon versus genuine coverage gaps outside the
moon (every exposure saturated). Lumping them together hides the informative number. On the real
JPEG data, coverage gaps outside the moon are 0%.

### Step 4 — display scaling

The composite is now physical brightness, whereas every constant in the display chain
(`_radial_tone_map` breakpoints, `_percentile_stretch`, `UNSHARP_WEIGHTS`, `RGB_DIM_QUOTIENTS`)
was tuned against v2's composite. Provide one knob rather than retuning many:

```
composite_for_display = (L / percentile(L, 99.9)) ** (1/gamma)      gamma == 1.0 -> exact no-op
```

Everything downstream works on ratios to a local mean or per-radius quantiles, so the
normalisation is cosmetic; only the exponent matters. Make `gamma == 1.0` return the input
object unchanged so the no-op is exact.

**For the JPEG data the correct value is ≈ 1.0, i.e. no compression.** Measured brightness spread
(99th percentile ÷ median):

```
v2 composite, which the display chain is tuned for      167
new merge's brightness, uncompressed                    233
exponent that matches them                            1.065
```

**Do not derive this exponent from first principles.** Doing so gives ≈1.63 and ruins the image.
The reasoning that produces 1.63 — "v2's composite was heavily compressed by the camera's JPEG
encoding, so imitate that" — is wrong: v2 scaled exposures by `(t_ref/t_k)^(1/g)` with `g ≈ 1.1`,
which is nearly the plain exposure ratio, and across a ~1e4 brightness range those factors do
almost all the work. v2's composite was therefore already nearly proportional to brightness. Fit
the exponent against the measured spread instead; provide a small script that reports it and
names the constant to set.

### Step 5 — wire the runner

Insert a calibration stage between stage2 and stage3 (it needs registration, and the merge needs
it). Print: number of exposures and samples, `lsqr istop`, monotonicity violations, residual rms
overall and in the core band `v ∈ [0.25, 0.85]`, and the per-exposure corrections. Save the
calibration to a pickle. Keep the old merge reachable behind a flag so the two can be compared
without a checkout.

## 3.5 Traps — every one of these was hit and cost real time

**1. The exposure corrections cannot be solved jointly with the curve.** `ln t_k + d_k` appears
only as a sum, so a free `d_k` makes the known shutter times carry no information and curve shape
trades freely against exposure ratios at identical residual (the Grossberg–Nayar
response/exposure-ratio ambiguity). `lsqr` does not fail loudly — it returns `istop=3` and
answers that looked plausible while the recovered corrections were wrong by **1700%**. Alternate
instead (§1c). Assert on `istop == 3`.

**2. Score the recovered curve anchored mid-range, never at `v = 1`.** Only
`[FIT_VALUE_LO, FIT_VALUE_HI]` is constrained; `v = 1` is extrapolated, so normalising there
smears extrapolation error across every other value and made a 0.3%-accurate fit look 5% wrong.
Anchor at `v = 0.5`.

**3. The synthetic test's planted curve must be monotone over `[0,1]`.** A first attempt used
`v + 0.18·sin(π(v−0.5))·cos(…)`, which leaves `[0,1]` and clips at both ends, making the planted
function non-invertible and the whole test meaningless. Use e.g.
`(1−β)·v + β·(3v² − 2v³)` with `β = 0.5` on top of sRGB — monotone, maps 0→0 and 1→1, and
deliberately **not** a power law, since that is the whole point.

**4. Form merge weights in each exposure's own frame, before warping.** Warping the weights and
the values separately lets bilinear interpolation pair a trusted pixel's value with an untrusted
neighbour's weight.

**5. Saturated pixels need weight exactly zero, not small.** The old scheme gave them ~0.002
while their stored value is necessarily too low — with 15 overlapping exposures that biases the
bright inner corona downward.

**6. Preserve the common moon blanking.** Detected radius drifts 5.94 px across the bracket
(§3.3). Without it, the annulus between the smallest and largest disk is filled only from
saturated long exposures.

**7. Memory.** Never hold all exposures at full resolution — 15 × 2 arrays at 4000×6000 float32
is ~2.9 GB on the GPU. Build/warp/accumulate one at a time. If per-exposure arrays must be kept
for diagnostics, subsample by 4 (full resolution is ~4.3 GB of host RAM for plots rendered at
stride 8 anyway).

**8. `np.arange(H, np.float32)` silently means `stop=np.float32`, not `dtype`.** Use
`np.arange(H, dtype=np.float32)`.

**9. Locating the moon in a rendered PNG:** the disk is one constant value, but so is much of the
dark background, so a value threshold grabs half the frame. Take the connected component
containing the frame centre (`scipy.ndimage.label`).

## 3.6 Known approximations — deliberate, document them

- The calibration is fitted on per-exposure averages of **encoded** values while the merge
  averages brightness. Wrong order in principle, but frames within a group are registered copies
  of the same scene, so the spread is noise-level and the bias second order. The curve is unknown
  when the averages are built, which is why.
- One response curve for the **mean of R,G,B**. The three channels have different curves and
  different clipping points; per-channel calibration is strictly better and is the natural next
  step.
- `value_sigma = 1/255` is a single constant standing in for quantization, JPEG block artefacts
  and chroma subsampling. Measured residuals came out ~4× smaller than it predicts, mostly
  because each value is already a stack average — so relative weighting is right but the absolute
  variance scale is pessimistic. Harmless for a weighted mean; matters only if the σ maps are
  ever read as absolute uncertainties.
- Two shortest exposures dropped (§3.3).

## 3.7 Verification

**Synthetic, first, no GPU, ~10 s.** Plant a known response — sRGB followed by the monotone
S-curve of trap 3 — across the real 17-exposure ladder with 1/255 noise and per-exposure scale
errors drawn at σ = 5%. Assert:

```
median curve error over [FIT_VALUE_LO, FIT_VALUE_HI], anchored at v=0.5   < 2%      (got 0.63%)
max error in the recovered exposure corrections                          < 4%      (got 2.4%, from a planted 9.7% spread)
non-monotone steps inside the fit range                                  == 0
refinement does not make the curve worse than the nominal-times fit
```

Also print accuracy by value band (this is what sets `FIT_VALUE_LO`) and the implied
adjacent-pair multiplier at several stored values — it varies ~35% across the range for a single
2× exposure step, which is the direct demonstration that one exponent per pair cannot work.

**Real data.** Run against the cached v2 run to skip registration. Expect:

```
lsqr istop                      2          (3 = broken)
monotonicity violations         0
residual, core v ∈ [0.25,0.85]  0.73% rms, bias 0.00%
per-exposure corrections        0.93 – 1.10, none pinned at the ±25% clamp,
                                largest on the shortest exposures
response local exponent         1.82 at v=0.20 rising to 3.82 at v=0.80
                                (a pure power law would be constant; sRGB alone would be 2.2)
coverage gaps outside the moon  0%
```

If the per-exposure corrections come out at the clamp, or `istop == 3`, the alternation has been
implemented as a joint solve — see trap 1.

**Display.** Compare the composite's brightness spread (p99/p50) against **167**, v2's value on
the JPEG set, and set the display exponent from that ratio rather than by eye.
