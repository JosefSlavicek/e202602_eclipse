"""Input sources for the v2 pipeline: JPG, atmosphere-distorted JPG, real NEF, and
NEF+injected-corona.

The whole pipeline consumes a single 2-D [0,1] luminance per frame (color is fabricated
only at the very end in stage3). So an input mode is fully described by:
  - how to enumerate frames and read their (exposure, timestamp, size, brightness);
  - how to load one frame as a float32 [0,1] grayscale tensor.

Four modes:
  jpg        - folder of JPGs, identical to v1 behavior (gamma-encoded values).
  atmosphere - the same JPG set re-rendered as it would look with the Sun 8 deg up
               (data_atmosphere/, built by make_data_atmosphere.py). Byte-identical
               format to `jpg`, so it is a drop-in for measuring how much the
               pipeline degrades under the 2026 target atmosphere.
  nef        - folder of .NEF, decoded as real *linear* raw luminance.
  inject  - folder of .NEF (counts/exposures/timestamps/noise from the real NEFs) with
            corona content faked per frame from the nearest-log v1 JPG exposure group,
            placed on the real NEF noise floor (see NefInjectSource for the rationale).

`load_grayscale` (utils) and `detect_moons`/`get_image_infos` (stage0) route through the
active source via `ImageInfo.source`, which is re-attached after every pickle load
(`attach_source`) and never pickled itself.
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# NEF decoding (libraw). Imported lazily inside the NEF sources so the jpg mode and the
# non-conda interpreters do not require rawpy.


# --- sensor / format constants (Nikon Z6 frames; mirror make_fake_inputs.py) ----------
SENSOR_W = 6064
SENSOR_H = 4040
BLACK_LEVEL = 1008
WHITE_LEVEL = 16383
RAW_SPAN = float(WHITE_LEVEL - BLACK_LEVEL)

V1_COMPOSITE_NPY = Path("/home/slavik/tmp/eclipse_v1_run/v1-stage3_composite.npy")

# Per-frame synthetic jitter / noise for injection (parallax + sensor noise so stage0
# registration is exercised the same way it will be on real data).
INJECT_JITTER_PX = 5.0
INJECT_JITTER_DEG = 0.2
INJECT_SHOT = 2.0 / RAW_SPAN           # signal-dependent shot-noise factor -> [0,1] units


def _stable_seed(name: str) -> int:
    """Deterministic per-frame seed from a filename (independent of dict/listing order)."""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


class FrameSource(ABC):
    kind: str = "abstract"
    is_linear: bool = False
    min_peak_brightness: float = 0.1

    @abstractmethod
    def scan(self) -> list:
        """Enumerate frames -> list[ImageInfo] with source/link attached."""

    @abstractmethod
    def load_gray(self, ii, device) -> torch.Tensor:
        """Load one frame as a float32 (H, W) tensor in [0,1] on `device`."""

    def load_rgb(self, ii, device) -> torch.Tensor:
        """(H, W, 3) tensor; default broadcasts grayscale. Only detect_moons needs this."""
        gray = self.load_gray(ii, device)
        return gray.unsqueeze(-1).expand(-1, -1, 3)


# --------------------------------------------------------------------------- #
#  JPG  (mode 1) — verbatim current behavior                                  #
# --------------------------------------------------------------------------- #
class JpgSource(FrameSource):
    kind = "jpg"
    is_linear = False
    min_peak_brightness = 0.1

    def __init__(self, data_root):
        self.data_root = Path(data_root)

    def scan(self) -> list:
        from eclipse_v2.stage0 import ImageInfo, get_info_from_exif
        root = self.data_root
        jpg_files = list(root.rglob("*.jpg")) + list(root.rglob("*.JPG"))
        infos = []
        for jpg_file in jpg_files:
            with Image.open(jpg_file) as img:
                width, height = img.size
                avg_brightness = np.array(img).astype(np.float32).mean() / 255.0
            exposure_time, timestamp = get_info_from_exif(jpg_file)
            ii = ImageInfo(
                path=jpg_file, width=width, height=height,
                avg_brightness=avg_brightness, timestamp=timestamp,
                exposure_time=exposure_time,
            )
            ii.source = self
            infos.append(ii)
        return infos

    def load_gray(self, ii, device) -> torch.Tensor:
        with Image.open(ii.path) as img:
            arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def load_rgb(self, ii, device) -> torch.Tensor:
        with Image.open(ii.path) as img:
            arr = np.array(img).astype(np.float32) / 255.0
        assert arr.ndim == 3 and arr.shape[-1] == 3, (ii.path, arr.shape)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Atmosphere-distorted JPG  (mode 1b)                                        #
# --------------------------------------------------------------------------- #
class AtmosphereSource(JpgSource):
    """`data_atmosphere/` — the JPG set re-rendered with the Sun 8 deg above the horizon.

    Same format as `jpg`: 6000x4000 sRGB JPEGs, original filenames, EXIF copied
    verbatim (so exposures and sub-second timestamps still parse). The pipeline
    therefore treats these frames exactly like mode 1, which is the point — the only
    difference between a `jpg` run and an `atmosphere` run is the atmosphere.

    The extra behaviour here is provenance: the directory carries an atmosphere.json
    listing every distortion applied and its size, which is loaded so a run can log
    and archive what it consumed.
    """

    kind = "atmosphere"

    def __init__(self, data_root):
        super().__init__(data_root)
        meta_path = self.data_root / "atmosphere.json"
        assert meta_path.is_file(), (
            f"{self.data_root} has no atmosphere.json — mode 'atmosphere' expects the "
            f"output of make_data_atmosphere.py, not a plain folder of JPGs")
        with open(meta_path) as fh:
            self.meta = json.load(fh)

    def describe(self) -> str:
        """One-paragraph summary of the atmosphere baked into these frames."""
        m = self.meta
        g, s, ch = m["geometry"], m["seeing"], m["channels"]
        mag = lambda c: -2.5 * math.log10(ch[c]["atten"])   # noqa: E731
        return "\n".join([
            f"  target      Sun {m['target']['alt_deg']:.1f} deg, airmass "
            f"{m['target']['airmass']:.2f}, {m['target']['P_hPa']:.0f} hPa"
            f"   (source was {m['source']['alt_deg']:.1f} deg, X "
            f"{m['source']['airmass']:.2f})",
            f"  refraction  {g['compression_px']:.1f} px compression along +x "
            f"({g['compression_percent']:.2f}%), non-affine {g['nonaffine_px']:.1f} px, "
            f"drift {m['drift']['bulk_px_range'][0]:+.1f}.."
            f"{m['drift']['bulk_px_range'][1]:+.1f} px",
            f"  dispersion  R {ch['R']['disp_px_predicted']:+.2f}  "
            f"G {ch['G']['disp_px_predicted']:+.2f}  "
            f"B {ch['B']['disp_px_predicted']:+.2f} px along +x",
            f"  seeing      {s['target_arcsec']:.1f}\" (zenith {s['zenith_arcsec']:.1f}\", "
            f"X^{s['exponent']:.1f}); blur+wander split by exposure, "
            f"{s['per_exposure'][0]['added_blur_px_fwhm']:.2f} px + "
            f"{s['per_exposure'][0]['added_shift_px_rms']:.2f} px rms at "
            f"{s['per_exposure'][0]['exp_s']:.5f} s",
            f"  photometry  extinction R {mag('R'):.2f}  G {mag('G'):.2f}  "
            f"B {mag('B'):.2f} mag, exposure gain x{m['exposure_gain']:.2f}, "
            f"veiling glare {ch['G']['scattered'] * m['aureole']['fraction_in_frame_kernel']:.1%}",
            f"  NOT applied {', '.join(m['not_simulated'])}",
            f"  edges       outer {max(g['edge_extrapolated_px']):.0f} px along x are "
            f"edge-extended, not real data",
        ])


# --------------------------------------------------------------------------- #
#  NEF  (mode 2) — real linear raw                                            #
# --------------------------------------------------------------------------- #
def _decode_nef_linear(path: Path) -> np.ndarray:
    """Decode a .NEF to a linear [0,1] luminance array via rawpy (demosaiced, gamma=1)."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16,
            user_wb=[1.0, 1.0, 1.0, 1.0],
        )
    return (rgb.astype(np.float32).mean(axis=2) / 65535.0)


class NefSource(FrameSource):
    kind = "nef"
    is_linear = True
    # 0.0 so the all-black test NEFs still pass the scan's sanity assert (mode 2 on real
    # corona frames trivially clears it; on the black test frames it lets decode/timing run).
    min_peak_brightness = 0.0

    def __init__(self, nef_dir):
        self.nef_dir = Path(nef_dir)

    def _nef_files(self) -> list:
        return sorted(list(self.nef_dir.glob("*.NEF")) + list(self.nef_dir.glob("*.nef")))

    def scan(self) -> list:
        import rawpy
        from eclipse_v2.stage0 import ImageInfo, get_info_from_exif
        infos = []
        for nef in self._nef_files():
            with rawpy.imread(str(nef)) as raw:
                vis = raw.raw_image_visible
                # read inside the `with` block — the view is freed on exit
                h, w = int(vis.shape[0]), int(vis.shape[1])
                mean_code = float(vis.astype(np.float32).mean())
            avg_brightness = max(0.0, (mean_code - BLACK_LEVEL) / RAW_SPAN)
            exposure_time, timestamp = get_info_from_exif(nef)
            ii = ImageInfo(
                path=nef, width=w, height=h,
                avg_brightness=avg_brightness, timestamp=timestamp,
                exposure_time=exposure_time,
            )
            ii.source = self
            infos.append(ii)
        return infos

    def load_gray(self, ii, device) -> torch.Tensor:
        arr = _decode_nef_linear(ii.path)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  NEF + injected corona  (mode 3)                                            #
# --------------------------------------------------------------------------- #
# A single tone-compressed radiance map (e.g. the v1 composite) cannot span the real
# bracket (~1/4000s..2s, ~1e4:1) by linear re-exposure: short frames collapse to the
# noise floor and their moon becomes undetectable, so the O(n^2) intra-exposure
# registration would run on fewer frames than reality and the runtime/memory measurement
# would be wrong. Instead each NEF gets correctly-exposed content from the nearest-log v1
# JPG exposure group (round-robin within a group so siblings differ), placed on the real
# NEF's noise floor. Cross-exposure brightness is then not perfectly linear (mode 2 on real
# corona NEFs remains the true test of linear handling), but every exposure group has
# detectable, structured content so stage0 runs at true scale -- which is what mode 3 is for.
def _srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _read_exposures(paths: list) -> dict:
    """Map each path -> EXIF ExposureTime (seconds) via a single exiftool batch."""
    import exiftool
    with exiftool.ExifToolHelper() as et:
        md = et.get_metadata([str(p) for p in paths])
    out = {}
    for p, m in zip(paths, md):
        e = m.get("EXIF:ExposureTime")
        assert e is not None and isinstance(e, (int, float)), (p, e)
        out[p] = float(e)
    return out


def _nearest_exposure(target: float, keys: list) -> float:
    return min(keys, key=lambda k: abs(math.log(target) - math.log(k)))


def _jpg_linear_mean(jpg_path: Path) -> float:
    """Cheap brightness proxy: mean linear luminance of a downscaled JPG. Reflects the
    injected content's brightness (monotone with exposure) for the scan's sort/assert."""
    with Image.open(jpg_path) as im:
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        rgb = np.asarray(im)
    return float(_srgb_to_linear(rgb).mean())


def _jitter_luminance(jpg_path: Path, seed: int, out_h: int, out_w: int) -> np.ndarray:
    """Load a JPG, resize to the sensor raster, apply a seeded sub-pixel shift+rotation,
    sRGB-linearize, and return luminance (mean over channels) in [0,1]."""
    rng = np.random.default_rng(seed)
    with Image.open(jpg_path) as im:
        im = im.convert("RGB").resize((out_w, out_h), Image.LANCZOS)
        th = math.radians(float(rng.uniform(-INJECT_JITTER_DEG, INJECT_JITTER_DEG)))
        tx = float(rng.uniform(-INJECT_JITTER_PX, INJECT_JITTER_PX))
        ty = float(rng.uniform(-INJECT_JITTER_PX, INJECT_JITTER_PX))
        cx, cy = out_w / 2.0, out_h / 2.0
        cos_t, sin_t = math.cos(th), math.sin(th)
        a, b = cos_t, -sin_t
        d, e = sin_t, cos_t
        c = cx + tx - a * cx - b * cy
        f = cy + ty - d * cx - e * cy
        im = im.transform((out_w, out_h), Image.AFFINE, (a, b, c, d, e, f),
                          resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        rgb = np.asarray(im)
    return _srgb_to_linear(rgb).mean(axis=2)


class NefInjectSource(NefSource):
    kind = "inject"
    is_linear = True
    min_peak_brightness = 0.02

    def __init__(self, nef_dir, jpg_dir):
        super().__init__(nef_dir)
        self.jpg_dir = Path(jpg_dir)

    def _jpg_files(self) -> list:
        return sorted(list(self.jpg_dir.glob("*.jpg")) + list(self.jpg_dir.glob("*.JPG")))

    def scan(self) -> list:
        # Build metadata without decoding the raw (injected frames are always sensor-sized;
        # the representative repeated-decode cost lives in load_gray, not this one-time scan).
        from eclipse_v2.stage0 import ImageInfo, get_info_from_exif
        infos = []
        for nef in self._nef_files():
            exposure_time, timestamp = get_info_from_exif(nef)
            ii = ImageInfo(
                path=nef, width=SENSOR_W, height=SENSOR_H,
                avg_brightness=0.0, timestamp=timestamp, exposure_time=exposure_time,
            )
            ii.source = self
            infos.append(ii)

        jpgs = self._jpg_files()
        assert jpgs, f"no JPGs in {self.jpg_dir}"
        jpg_exp = _read_exposures(jpgs)
        groups = {}
        for p, e in jpg_exp.items():
            groups.setdefault(e, []).append(p)
        for e in groups:
            groups[e].sort()
        gkeys = sorted(groups.keys())

        # Deterministic assignment: sort NEFs by exposure, round-robin the matched group so
        # sibling frames of one exposure get different JPGs (genuine parallax to register).
        rr = {}
        for ii in sorted(infos, key=lambda x: (x.exposure_time, x.path.name)):
            ge = _nearest_exposure(ii.exposure_time, gkeys)
            grp = groups[ge]
            jpg = grp[rr.get(ge, 0) % len(grp)]
            rr[ge] = rr.get(ge, 0) + 1
            ii.source = self
            ii.link = (_stable_seed(ii.path.name), str(jpg))
            # Brightness must reflect the *injected* content, not the black NEF noise floor.
            ii.avg_brightness = _jpg_linear_mean(jpg)
        return infos

    def load_gray(self, ii, device) -> torch.Tensor:
        assert ii.link is not None, f"inject frame {ii.path} has no JPG linkage (scan not run?)"
        seed, jpg_path = ii.link
        H, W = SENSOR_H, SENSOR_W

        # Correctly-exposed corona content for this exposure (jittered, linear luminance).
        signal = _jitter_luminance(Path(jpg_path), seed, H, W)

        # Real sensor noise floor + decode cost (genuine black-level/read noise from the file;
        # decoding it also makes mode-3 decode timing representative of mode 2 / real data).
        l_real = _decode_nef_linear(ii.path)
        if l_real.shape != (H, W):
            l_real_t = torch.from_numpy(l_real).unsqueeze(0).unsqueeze(0)
            l_real = torch.nn.functional.interpolate(
                l_real_t, size=(H, W), mode="bilinear", align_corners=False
            ).squeeze().numpy()

        # Signal-dependent shot noise (the black frame has no signal-scaled noise of its own).
        rng = np.random.default_rng(seed ^ 0x5EED)
        sigma = INJECT_SHOT * np.sqrt(np.clip(signal, 0.0, None))
        noise = rng.standard_normal(signal.shape).astype(np.float32) * sigma
        gray = np.clip(signal + l_real + noise, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(gray).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Construction + reattachment                                                #
# --------------------------------------------------------------------------- #
def make_source(input_mode: str, *, jpg_dir=None, nef_dir=None, radiance_npy=None,
                atmosphere_dir=None):
    if input_mode == "jpg":
        assert jpg_dir is not None, "jpg mode needs jpg_dir"
        return JpgSource(jpg_dir)
    if input_mode == "atmosphere":
        assert atmosphere_dir is not None, "atmosphere mode needs atmosphere_dir"
        return AtmosphereSource(atmosphere_dir)
    if input_mode == "nef":
        assert nef_dir is not None, "nef mode needs nef_dir"
        return NefSource(nef_dir)
    if input_mode == "inject":
        assert nef_dir is not None and jpg_dir is not None, "inject mode needs nef_dir and jpg_dir"
        return NefInjectSource(nef_dir, jpg_dir)
    raise ValueError(f"unknown input_mode {input_mode!r}")


def _iter_infos(exposure_groups_or_infos):
    if isinstance(exposure_groups_or_infos, dict):
        for group in exposure_groups_or_infos.values():
            yield from group
    else:
        yield from exposure_groups_or_infos


def attach_source(exposure_groups_or_infos, source) -> None:
    """Re-attach the active FrameSource to ImageInfos after a pickle load (source is not pickled)."""
    for ii in _iter_infos(exposure_groups_or_infos):
        ii.source = source
