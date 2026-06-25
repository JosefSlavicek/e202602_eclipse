# v2 three-mode inputs — implementation plan (working notes)

Branch: `v2-three-input-modes`. Rollback = `git checkout main`.

## Goal
v2 supports 3 input modes:
1. **jpg** — folder of JPGs, identical to today (v1 behavior).
2. **nef** — folder of `.NEF`, decoded as real linear raw (real corona content).
3. **inject** — folder of `.NEF` (counts/exposures/timestamps/noise from real NEFs) +
   corona content faked per frame from the nearest-log v1 JPG exposure group, placed on
   the real NEF noise floor.

   NOTE (design change during impl): the original idea was to re-expose a single v1-composite
   radiance map by `t/t_norm` for a strictly linear bracket. Rejected after testing: the
   composite is tone-compressed (~100:1) and cannot span the real bracket (~1e4:1) — short
   frames collapse to the noise floor, their moon becomes undetectable, and the O(n^2)
   intra-exposure registration (the dominant cost we measure) would run on fewer frames than
   reality. Per-exposure JPG content gives every exposure detectable structure → stage0 runs
   at true scale. Tradeoff: cross-exposure brightness isn't perfectly linear; mode 2 on real
   corona NEFs remains the true test of linear handling.

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

## Mode-3 injection detail (NefInjectSource) — AS IMPLEMENTED
`__init__(nef_dir, jpg_dir)`. `scan`:
- build ImageInfo per NEF WITHOUT decoding the raw (W/H = SENSOR_W/H constants; exposure +
  timestamp from `get_info_from_exif`) — the repeated-decode cost belongs in load_gray.
- read JPG exposures (exiftool batch), group, sort NEFs by exposure, round-robin the
  nearest-log group so siblings get different JPGs.
- store `ii.link = (seed, jpg_path)`, seed = stable hash of filename (deterministic).
- set `ii.avg_brightness` = mean linear luminance of the assigned JPG (NOT the black NEF!),
  so the brightness sort/assert reflect injected content.

`load_gray(ii)` — deterministic pure fn of (assigned JPG, seed, fixed NEF file):
1. `signal` = `_jitter_luminance(jpg, seed)` — resize to sensor, seeded ±5px/±0.2° affine,
   sRGB→linear, mean over channels.
2. `L_real` = real NEF linear decode (genuine noise floor; decoding also makes mode-3 decode
   timing representative of mode 2).
3. `gray = clip(signal + L_real + seeded_shot_noise, 0, 1)`.

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

## Fixes found during implementation (all committed)
- `get_info_from_exif` regex hardcoded a `-` timezone offset; real/test NEFs use `+01:00`.
  Relaxed to `[-+]` (superset → jpg mode unaffected).
- mode-3 `avg_brightness` must come from the assigned JPG, not the black NEF raw mean.
- `NefSource.min_peak_brightness = 0.0` so the all-black test NEFs pass the scan assert.

## Component tests done (pre-handoff)
- compile + CLI ok; ImageInfo `__getstate__` strips `source`, keeps `link`.
- NEF linear decode ok (4040×6064, faint = real floor); EXIF parse ok after fix.
- mode-3 scan assignment (siblings get distinct JPGs), avg_brightness content-derived, assert passes.
- stage0 ingest + detect_moons on injected subset: **moon detected in every exposure**
  (centers ~(1982,2920), r~317) incl. the 0.0002s frame → registration will run at true scale.
- mode-1 JPG `load_gray` is **bitwise-identical** to the legacy path.
- NOT yet run: full stage0→3 on the 183-frame set (that is the user's manual test).

## Validation
- mode jpg vs current outputs: compare `v2-stage3_*` artifacts (bitwise-ish).
- mode inject `--dry` subset smoke, then full 183 run for runtime/memory numbers.

## Risk
Milestone 1 is the safety net. Linear-input heuristics (audit list) caught at 2/3.
Mode-3 geometry (ρ placement) is the most code & most likely to need the step-by-step fallback.
