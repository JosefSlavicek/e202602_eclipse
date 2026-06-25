# v2 three-mode inputs — implementation plan (working notes)

Branch: `v2-three-input-modes`. Rollback = `git checkout main`.

## Goal
v2 supports 3 input modes:
1. **jpg** — folder of JPGs, identical to today (v1 behavior).
2. **nef** — folder of `.NEF`, decoded as real linear raw (real corona content).
3. **inject** — folder of `.NEF` (counts/exposures/timestamps/noise from real NEFs) +
   corona content faked from the v1 HDR composite, re-exposed per frame so the linear
   exposure law (`signal ∝ radiance·t`) holds across the synthetic bracket.

Primary purpose of mode 3: validate **runtime/memory at real-eclipse scale** (183 frames,
real bracket structure) on the **same code path** the real corona NEFs will take.

## Key facts that shape the design
- Pipeline is **grayscale/luminance** end to end; color is fabricated only in
  `stage3.rgb_vignette_and_radial_pickle`. So a loader only needs a 2-D `[0,1]` luminance.
- All pixel loading funnels through **`utils.load_grayscale(ii, device)`** (+ two direct
  opens in `stage0.detect_moons` / `stage0.get_image_infos`). That is the chokepoint.
- Cross-exposure brightness "magic" = `stage2.estimate_gamma` fitting a gamma to undo the
  JPG tone curve; `stage3._scale_to_ref` uses `(t0/t1)**(1/gamma)`. For linear raw the
  correct scale is just `t0/t1` (gamma≡1).
- Conda env `e202602_eclipse` (`/home/slavik/usr/anaconda3/envs/e202602_eclipse/bin/python`)
  has torch 2.10 + CUDA + rawpy 0.27 → decode + CUDA in one process.
- v1 composite radiance map: `/home/slavik/tmp/eclipse_v1_run/v1-stage3_composite.npy`
  (float32, cropped to mutual coverage, moon blackened, HDR merge in ref coords). Used as
  the scene radiance ρ for injection.
- Real test NEFs (black, but real EXIF/noise/bracket): `/home/slavik/tmp/eclipse_fake_imgs`
  (183 frames from `camera/pokus9.py`). These mirror the planned eclipse bracket exactly.
- Sensor constants (from make_fake_inputs): W=6064 H=4040 black=1008 white=16383 RGGB.

## Architecture
New module `eclipse_v2/inputs.py`:

```
class FrameSource(ABC):
    kind: str               # "jpg" | "nef" | "inject"
    is_linear: bool
    min_peak_brightness: float   # replaces hard 0.1 assert in get_image_infos
    def scan(self) -> list[ImageInfo]            # fills fields + ii.source=self (+ ii.link for inject)
    def load_gray(self, ii, device) -> Tensor    # (H,W) float32 [0,1]
    def load_rgb(self, ii, device) -> Tensor     # (H,W,3); only detect_moons uses it

class JpgSource(FrameSource):    # is_linear=False, min_peak=0.1  — verbatim current behavior
class NefSource(FrameSource):    # is_linear=True,  min_peak=0.003
class NefInjectSource(NefSource):# is_linear=True; ρ from v1 composite, re-exposed per frame

def attach_source(exposure_groups_or_infos, source)  # re-set ii.source after unpickling
```

### ImageInfo (in stage0) changes
- add `source=None` (excluded from pickle via `__getstate__`) and `link=None` (pickled tuple).
- `__getstate__` returns `__dict__` copy with `source=None` so the 94 MB ρ never gets pickled.

### utils.load_grayscale
```
if getattr(ii,"source",None) is not None: return ii.source.load_gray(ii,device)
# else legacy JPG fallback (keeps mode-1 safe even if a reattach is missed)
```

### stage0
- `get_image_infos(source)`: delegate enumeration to `source.scan()`; replace
  `assert ...avg_brightness>0.1` with `source.min_peak_brightness`.
- `detect_moons`: `img = ii.source.load_rgb(ii, device)` instead of `Image.open`.
- `get_info_from_exif`: reused for NEF (EXIF:ExposureTime + Composite:SubSecDateTimeOriginal).
  If real NEFs lack timezone offset the regex assert may need relaxing — fix if testing hits it.

### stage2 (gamma gating)
In `cross_exposure_consecutive_pairs`: still call `estimate_gamma` and **print** it (diagnostic),
but `gamma_used = 1.0 if source.is_linear else gamma2`; use gamma_used for scale, registration,
and stored `gamma_by_pair`. stage3 unchanged (`(t0/t1)**(1/1)` = t0/t1). Harden
`mae_out_of_moon` to return inf when the valid window is empty (faint linear frames).

### pipeline.py
- argparse `--input-mode {jpg,nef,inject}`, `--nef-dir`, `--jpg-dir`, `--radiance-npy`.
- construct `source` once; reattach after every pickle `load` via `attach_source`.
- stage funcs that scan/detect get `source` passed in.

## Mode-3 injection detail (NefInjectSource)
Construction: load ρ = v1 composite (float32, HxW, moon blackened); know ρ moon center
(≈ ρ array center, refine with find_moon at build OR assume center). t_norm = max exposure.
`scan` stores per-frame `link = (seed,)`, seed = stable hash of filename.

`load_gray(ii)` — deterministic pure fn of fixed NEF file + ρ + ii.{exposure,timestamp,link}:
1. `L_real` = real NEF linear luminance (genuine black-level + read-noise floor).
2. target moon center = base_center + drift·(t−t0) + jitter(seed) (±5px/±0.2° affine, seeded).
3. place/scale ρ into sensor raster so ρ-moon → target center (affine warp; pad outside with 0).
4. `signal = clip(ρ_placed · (exposure / t_norm), 0, 1)`  ← linear exposure law.
5. `gray = clip(signal + L_real + seeded_noise, 0, 1)`.

Reuse `make_fake_inputs._jitter_affine` / `srgb_to_linear` ideas.

## Heuristics that assume gamma-encoded [0,1] (audit)
Relative/scale-invariant (NO change): brightness-outlier prune, radial tone map (vs local
mean), p3 stretch, merge weight Gaussian@0.5 (still "prefer well-exposed" in linear).
Absolute (handled): get_image_infos peak assert → source.min_peak_brightness;
estimate_gamma window → diagnostic+null-safe; warp_merge `orig>0.5→w=1` on ref exposure →
keep, verify on a linear run.

## Build order / milestones (each runnable)
1. inputs.py + JpgSource + chokepoint rewrite → **mode jpg reproduces current output** (safety net).
2. NefSource scan/decode → mode nef runs on black NEFs (empty but path+timing real).
3. stage2 gamma=1 gating → mode nef cross-exposure correct.
4. NefInjectSource → mode inject = the perf/realism target (full 183 run).

## Validation
- mode jpg vs current outputs: compare `v2-stage3_*` artifacts (bitwise-ish).
- mode inject `--dry` subset smoke, then full 183 run for runtime/memory numbers.

## Risk
Milestone 1 is the safety net. Linear-input heuristics (audit list) caught at 2/3.
Mode-3 geometry (ρ placement) is the most code & most likely to need the step-by-step fallback.
