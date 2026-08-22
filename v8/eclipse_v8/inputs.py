"""Input source for the v8 pipeline: real NEF raw only.

The whole pipeline consumes a single 2-D [0,1] luminance per frame. So the source is fully
described by:
  - how to enumerate frames and read their (exposure, timestamp, size, brightness);
  - how to load one frame as a float32 [0,1] grayscale tensor.

`load_grayscale` (utils) and `detect_moons`/`get_image_infos` (stage0) route through the
active source via `ImageInfo.source`, which is re-attached after every pickle load
(`attach_source`) and never pickled itself.

**Two contracts, not one.**  `load_gray` returns the *stored* value — what stage0/1/2
register on and what the calibration is fitted to. As of the rawprep rework this is the
dark+flat-corrected value (see `rawprep.apply_corrections`), read straight from that cache —
`load_gray` no longer decodes or corrects anything itself.  `load_radiance` returns physical
brightness with its variance, and is the only path radiometry may take.  A source becomes
able to answer the second one after `set_calibration`.

Overburn and the merge's per-frame weight are *not* derived here at all: they are computed
once, per raw frame, straight off the unmodified decode, in `rawprep.py` — before dark/flat
correction ever runs — and just read back by `load_overburn`/`load_weight`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

# NEF decoding (libraw). Imported lazily inside scan()/_decode_nef_linear so a non-conda
# interpreter does not require rawpy just to import this module.


# --- sensor / format constants (Nikon Z6 frames) ---------------------------------------
SENSOR_W = 6064
SENSOR_H = 4040
BLACK_LEVEL = 1008
WHITE_LEVEL = 16383
RAW_SPAN = float(WHITE_LEVEL - BLACK_LEVEL)

# Placeholders. They set only the *relative* weighting between raw frames, and both are
# measurable from a flat-field pair: plot variance against mean over many patches and the
# line's slope is 1/gain, its intercept read_noise^2.
GAIN_E_PER_UNIT = 4.0e4                # electrons per unit of decoded [0,1] signal
READ_SIGMA = 2.0 / RAW_SPAN            # read noise, in the same [0,1] units


def _decode_nef_linear(path: Path) -> np.ndarray:
    """Decode a .NEF to a linear [0,1] luminance array via rawpy (demosaiced, gamma=1).

    `user_flip=0` is required, not optional: rawpy/libraw's default is to auto-rotate/flip
    the output according to the camera's own orientation sensor at capture time, so shots
    taken with the camera physically rotated (portrait vs. landscape, or any other angle)
    would otherwise come back pre-rotated by different amounts. Every fixed-pattern thing
    downstream -- dust shadows, vignetting, dark current, pixel (i,j) meaning the same
    physical photosite across frames at all -- depends on every frame staying in the sensor's
    own, unrotated layout regardless of how the camera was held.
    """
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16,
            user_wb=[1.0, 1.0, 1.0, 1.0], user_flip=0,
        )
    return (rgb.astype(np.float32).mean(axis=2) / 65535.0)


class NefSource:
    kind = "nef"
    is_linear = True        # registration detail only: one exposure scales onto its
                             # neighbour by the plain exposure ratio, which stage2 uses to
                             # bring two frames close before Fourier alignment. Says nothing
                             # about radiometry — that is load_radiance's job.
    min_peak_brightness = 0.0

    _calib = None
    _corrections: dict | None = None

    def __init__(self, nef_dir):
        # nef_dir is the my_raws/-style parent holding lights/ and darks/; the actual
        # frames are read from its lights/ subfolder.
        self.nef_dir = Path(nef_dir) / "lights"
        self._raw_cache_dir = None
        self._corrected_dir = None

    def _nef_files(self) -> list:
        return sorted(list(self.nef_dir.glob("*.NEF")) + list(self.nef_dir.glob("*.nef")))

    def scan(self) -> list:
        import rawpy
        from eclipse_v8.stage0 import ImageInfo, get_info_from_exif
        infos = []
        for nef in self._nef_files():
            with rawpy.imread(str(nef)) as raw:
                # raw_image_visible is the sensor data before rawpy's processing pipeline
                # runs, so unlike postprocess() (see _decode_nef_linear) it is never
                # auto-rotated/flipped by the camera's orientation sensor -- already in the
                # sensor's own unrotated layout, no user_flip equivalent needed here.
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

    # --- rawprep cache wiring ---------------------------------------------------------
    def set_cache_dirs(self, raw_cache_dir, corrected_dir) -> None:
        """Point this source at the rawprep caches: raw (overburn/weight) and corrected
        (dark+flat-baked) values. Must be called before load_gray/load_radiance/
        load_overburn/load_weight are used."""
        self._raw_cache_dir = Path(raw_cache_dir)
        self._corrected_dir = Path(corrected_dir)

    def load_gray(self, ii, device) -> torch.Tensor:
        """The STORED value: dark+flat-corrected, read straight from the rawprep cache."""
        from eclipse_v8 import rawprep as rp
        assert self._corrected_dir is not None, (
            "set_cache_dirs() has not been called on this source")
        arr = rp.load_corrected(self._corrected_dir, ii)
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def load_overburn(self, ii, device) -> torch.Tensor:
        """Per-pixel saturation mask, measured once from the unmodified raw (rawprep.py)."""
        from eclipse_v8 import rawprep as rp
        assert self._raw_cache_dir is not None, (
            "set_cache_dirs() has not been called on this source")
        overburn, _weight = rp.load_measurements(self._raw_cache_dir, ii)
        return torch.from_numpy(overburn).to(device=device, dtype=torch.bool)

    def load_weight(self, ii, device) -> torch.Tensor:
        """Per-pixel merge weight, measured once from the unmodified raw (rawprep.py)."""
        from eclipse_v8 import rawprep as rp
        assert self._raw_cache_dir is not None, (
            "set_cache_dirs() has not been called on this source")
        _overburn, weight = rp.load_measurements(self._raw_cache_dir, ii)
        return torch.from_numpy(weight).to(device=device, dtype=torch.float32)

    def load_rgb(self, ii, device) -> torch.Tensor:
        """(H, W, 3) tensor; broadcasts grayscale. Only detect_moons needs this."""
        gray = self.load_gray(ii, device)
        return gray.unsqueeze(-1).expand(-1, -1, 3)

    # --- radiometry ----------------------------------------------------------------
    def set_calibration(self, calib_result) -> None:
        """Cache the per-exposure shutter-time corrections.

        Raw is already linear, so there is no response curve to invert — only the shutter
        corrections apply, regardless of format.
        """
        self._calib = calib_result
        self._corrections = calib_result.correction_by_exposure()

    def exposure_correction(self, ii) -> float:
        """c_k for this frame's exposure; 1.0 for exposures the calibration never saw."""
        if not self._corrections:
            return 1.0
        t = float(ii.exposure_time)
        if t in self._corrections:
            return self._corrections[t]
        near = min(self._corrections, key=lambda k: abs(math.log(k) - math.log(t)))
        return self._corrections[near] if abs(near - t) <= 1e-9 * max(near, t) else 1.0

    def effective_exposure(self, ii) -> float:
        """The shutter time the camera actually delivered: reported time x its correction."""
        return float(ii.exposure_time) * self.exposure_correction(ii)

    def load_radiance(self, ii, device):
        """(radiance, variance, usable) as float32 (H, W) tensors: physical brightness per
        pixel.

        `radiance` is light per unit time — a property of the sky, identical in every frame.
        `variance` is that estimate's own uncertainty squared, modelled as
        `signal/gain + read_noise^2`, giving the familiar 1/sqrt(t) improvement. `usable` is
        the per-frame overburn mask, inverted — every frame that clears it is trusted
        equally in the exposure-group average.

        `signal` (via load_gray) is already dark-subtracted and flat-divided.
        """
        signal = self.load_gray(ii, device)
        t_eff = self.effective_exposure(ii)
        assert t_eff > 0, (ii.path, t_eff)

        radiance = signal / t_eff
        var = (signal.clamp(min=0.0) / GAIN_E_PER_UNIT + READ_SIGMA ** 2) / (t_eff ** 2)
        usable = ~self.load_overburn(ii, device)
        return radiance, var, usable


# --------------------------------------------------------------------------- #
#  Construction + reattachment                                                #
# --------------------------------------------------------------------------- #
def make_source(nef_dir) -> NefSource:
    return NefSource(nef_dir)


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
