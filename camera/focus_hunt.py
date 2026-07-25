#!/usr/bin/env python3
"""Auto-hunt the optimal focus of the filtered Sun via full-resolution shots.

Target image: a bright disk with a sharp border on a dark background. The
in-focus position is the one that MAXIMIZES an edge-sharpness figure of merit
computed from an ACTUALLY CAPTURED image (not a live-view preview).

Figure of merit
---------------
On the grayscale of the captured frame we compute the image gradient (Sobel
derivatives wrt x and y) and its magnitude. We then find the TOP_N pixels with
the largest gradient magnitude and mark a pixel VALID iff it is among that
top set AND at least two of its 8-neighbours are also in the top set (this
rejects isolated hot pixels / noise spikes that are not part of a real edge).
The merit is the MINIMUM gradient magnitude over the valid pixels: a frame with
a higher value has a sharper, more consistently strong border and is preferred.

Assumptions / procedure
------------------------
* The lens starts already ROUGHLY FOCUSED (not at a hard stop), so the true
  peak is expected to be close on either side. We therefore begin with a
  modest step and probe both directions; the first probe just uses a nominal
  direction (negative) as a tie-break.
* Nikon `manualfocusdrive` is a RANGE *action*: each set drives the focus motor
  RELATIVELY by the given (signed) amount. There is no way to read back an
  absolute focus position, so we track our own software position counter `pos`
  (units = focus-drive steps relative to the starting position).
* Search = neighbour-probe hill-climb with step halving. Starting from a large
  step we keep stepping in the improving direction; when neither neighbour
  improves we halve the step and try again, down to MIN_STEP, then return to
  the best position found. This reproduces the "hunt, halve, settle at the
  finest step" behaviour and is self-correcting: because every decision is made
  on a freshly
  MEASURED sharpness, small position-tracking errors (e.g. a dropped move after
  a USB hiccup) do not accumulate into wrong decisions.

Robustness
----------
The original PoC kept a single camera handle and became non-deterministic after
the first exception. Here every camera operation is funnelled through a guard:
on ANY exception the camera is torn down and re-initialised, then the call is
retried (for idempotent ops: preview capture, config reads/writes) or skipped
(for focus drive, which is NOT re-issued to avoid an unknown double-move -- the
closed-loop hunt absorbs the at-most-one-chunk position error). A `finally`
block always restores control mode / live view and releases the camera.

Run with the python that has gphoto2 + OpenCV bound (the machine wired to the
camera), e.g.:
    python3 camera/focus_hunt.py
"""
from __future__ import annotations

import argparse
import base64
import os
import time
import sys

# Silence OpenCV's libtiff WARN chatter (null-padded EXIF ASCII tags, unknown
# Nikon EXIF tags) that floods the log when decoding camera TIFFs. This env var
# is read by OpenCV at import, so it MUST be set before `import cv2`; it is the
# reliable path on builds whose cv2 lacks setLogLevel (see the guarded call
# below, kept as belt-and-suspenders).
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import numpy as np
import cv2
import gphoto2 as gp

# tkinter is stdlib but can be missing (headless / minimal Python builds) and
# the actual Tk runtime may lack PNG support (< 8.6). We import it defensively;
# if anything is off the display simply falls back to saving PNGs (see _show).
try:
    import tkinter as tk
    _TK_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on the host Python build
    tk = None
    _TK_IMPORT_ERROR = e

# Silence libtiff's benign per-frame chatter (null-padded EXIF ASCII tags and
# unknown Nikon EXIF tags) emitted by cv2.imdecode on camera TIFFs. Not present
# on older OpenCV builds, so guard it -- it is a nicety, not a requirement.
try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass

# ---- tunable parameters ---------------------------------------------------
INITIAL_STEP = 32       # first (coarse) focus step; halved down to MIN_STEP.
                        #  Modest because the lens starts already roughly
                        #  focused, so the peak is expected nearby
MIN_STEP = 4            # finest step; hunt ends after refining at this size
                        #  (not smaller than 4: the focus motor starts to fail
                        #   / no-op on sub-4-unit moves)
DRIVE_CHUNK = 100       # max focus units per single manualfocusdrive command
                        # (kept at the PoC's known-good magnitude; large moves
                        #  are split into several chunked commands)
MOVE_PAUSE = 0.15       # s between chunked drive commands
SETTLE_S = 0.35         # s to let optics settle after a move, before shooting
AVG_FRAMES = 1          # full-res shots averaged per measurement (>=1); each is
                        #  a real shutter actuation, so keep this small
NOISE_MARGIN = 1.005    # a candidate must beat the best by this factor to win
                        #  (guards against chasing measurement noise)
TOP_N = 300             # number of strongest-gradient pixels considered "edge"
BORDER_CROP = 8         # px cropped off each edge before the gradient metric
MAX_MOVES = 400         # hard safety cap on total focus moves
MAX_RETRY = 3           # camera-reset retries for idempotent operations
BACKLASH = 0            # extra units to take up mechanical slack on reversal
                        #  (0 = disabled; raise if reversals read soft)

# ---- 'radius' metric parameters -------------------------------------------
EXPECTED_RADIUS = 320   # px, expected solar-disk radius in the frame
RADIUS_KEEP = int(round(2 * 2 * np.pi * EXPECTED_RADIUS))  # ~4021 strongest
                        #  edge pixels kept for circle-fitting (a few * limb
                        #  circumference so the whole limb is represented)
CIRCUMCENTER_MAX_DIST = 500  # px, a triplet's circumcircle centre must lie
                        #  within this of the image centre to be valid
MIN_PAIR_DIST = 100     # px, every pair of a triplet's 3 points must be at
                        #  least this far apart (spreads points around the limb)
RADIUS_COS_MIN = 0.5    # min cosine between (point->circumcentre) and the
                        #  gradient-toward-higher-intensity (=> within ~60 deg;
                        #  the limb gradient must point inward, at the centre)
RADIUS_N_VALID = 1024   # valid triplets collected before scoring
RADIUS_TOP_SCORE = 128  # highest-score valids whose radii are averaged
RADIUS_SAMPLE_BATCH = 4096   # triplets evaluated per vectorised batch
DEFAULT_MAX_SAMPLES = 512 * 1024  # crash if this many samples fail to yield
                        #  RADIUS_N_VALID valid triplets (cmdline overridable)
COLLINEAR_EPS = 1e-6    # |2*signed-area| below this => collinear, reject early

# ---- auto-exposure (for the 'radius' metric) ------------------------------
CLIP_VALUE = 250        # a grayscale pixel at/above this counts as saturated
CLIP_ONSET_FRAC = 5e-4  # frame fraction saturated that marks "clipping began"
OVEREXPOSE_STOPS = 2.0  # default stops PAST onset to expose (~4x; the ~over=4
                        #  regime that made the radius metric focus-sensitive)
BASE_ISO = "100"        # ISO held during auto-exposure (base = least noise)

# ---- capture format -------------------------------------------------------
DEFAULT_IMAGE_QUALITY = "JPEG Fine"  # cv2.imdecode needs a JPEG/TIFF frame; a
                        #  NEF/raw only yields a small embedded PREVIEW to
                        #  cv2.imdecode, so the sharpness metrics would run on a
                        #  low-res, camera-processed image. JPEG Fine is full
                        #  resolution and decodes natively. The camera's original
                        #  quality is snapshotted at startup and restored on exit.

WINDOW = "Focus Hunt (filtered Sun)"
PNG_DIR = "focus_hunt"  # only used if there is no display (headless fallback)
GUI_INIT_W = 800        # tkinter window initial width; frames are downscaled to
GUI_INIT_H = 600        #  fit the current window size (preserving aspect ratio)


def focus_merit(bgr: np.ndarray) -> float:
    """Edge-sharpness figure of merit (higher = sharper).

    Grayscale the frame, take the Sobel gradient (d/dx, d/dy) and its magnitude.
    Keep the TOP_N pixels with the strongest gradient. A pixel is VALID iff it is
    in that top set AND at least two of its 8-neighbours are also in the top set
    (isolated hot pixels / noise spikes therefore drop out). The merit is the
    MINIMUM gradient magnitude over the valid pixels -- the weakest link of the
    coherent strong-edge cluster, which rises as the border comes into focus.

    Returns 0.0 if no pixel survives the validity test (e.g. a blank frame).
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = bgr.astype(np.float64)
    c = BORDER_CROP
    if gray.shape[0] > 2 * c and gray.shape[1] > 2 * c:
        gray = gray[c:-c, c:-c]

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    flat = mag.ravel()
    if flat.size <= TOP_N:
        top_flat = np.ones(flat.size, dtype=bool)
    else:
        top_flat = np.zeros(flat.size, dtype=bool)
        top_flat[np.argpartition(flat, -TOP_N)[-TOP_N:]] = True
    top = top_flat.reshape(mag.shape)

    # Count how many of each pixel's 8 neighbours are also in the top set.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.float32)
    neigh = cv2.filter2D(top.astype(np.float32), -1, kernel,
                         borderType=cv2.BORDER_CONSTANT)
    valid = top & (neigh >= 2)
    if not valid.any():
        return 0.0
    return float(mag[valid].min())


def focus_merit_laplacian(bgr: np.ndarray) -> float:
    """Laplacian variance -- standard optical autofocus figure of merit.

    Applies the discrete Laplacian (second spatial derivative) to the
    grayscale frame and returns the variance of the result.  Sharp edges
    produce large second-derivative values; the variance therefore rises
    monotonically as the image comes into focus.

    Particularly well-suited to solar / eclipse images: the hard limb of
    the solar disk and the silhouette of the moon both generate strong,
    localised Laplacian responses that increase rapidly with focus quality.
    Being a difference operator the metric is insensitive to overall frame
    brightness, so it needs no normalisation for illumination variation.
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = bgr.astype(np.float64)
    c = BORDER_CROP
    if gray.shape[0] > 2 * c and gray.shape[1] > 2 * c:
        gray = gray[c:-c, c:-c]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def focus_merit_radius(bgr: np.ndarray,
                       max_samples: int = DEFAULT_MAX_SAMPLES) -> float:
    """Circle-fit figure of merit -- averages the fitted solar-disk radius.

    Rationale: the filtered Sun is a bright disk whose sharp limb, when in
    focus, fits a tight circle; a blurred / bloomed limb reads as a LARGER
    apparent radius. We therefore estimate the disk radius by fitting circles
    to random triplets of strong-gradient limb pixels and return a merit that
    RISES as that radius shrinks (sharper focus).

    Steps:
      1. Sobel gradient of the grayscale; keep the RADIUS_KEEP pixels with the
         strongest magnitude, remembering position, gradient vector (gx, gy)
         -- which points toward higher intensity -- and magnitude.
      2. Draw random triplets. A triplet is VALID iff:
           * its circumcircle centre lies within CIRCUMCENTER_MAX_DIST of the
             image centre,
           * for each point, the cosine between (point->circumcentre) and its
             gradient exceeds RADIUS_COS_MIN (gradient points inward),
           * all three pairwise distances exceed MIN_PAIR_DIST.
         Near-collinear triplets are rejected up front via the circumcentre
         denominator (twice the signed triangle area) to avoid blow-up.
      3. For each valid triplet record score = min of its 3 gradient
         magnitudes, and radius = circumradius.
      4. Collect RADIUS_N_VALID valid triplets (crashing if max_samples is
         exhausted first), average the radii of the RADIUS_TOP_SCORE
         highest-score ones, and return 1000 / (avg_radius + 1).
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = bgr.astype(np.float64)
    c = BORDER_CROP
    if gray.shape[0] > 2 * c and gray.shape[1] > 2 * c:
        gray = gray[c:-c, c:-c]

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    H, W = mag.shape

    flat = mag.ravel()
    n_keep = min(RADIUS_KEEP, flat.size)
    keep = np.argpartition(flat, -n_keep)[-n_keep:]
    xs = (keep % W).astype(np.float64)
    ys = (keep // W).astype(np.float64)
    gxs = gx.ravel()[keep]
    gys = gy.ravel()[keep]
    mags = flat[keep]
    cx0 = (W - 1) / 2.0
    cy0 = (H - 1) / 2.0
    n = n_keep

    scores: list[float] = []
    radii: list[float] = []
    samples_done = 0
    min_pair_sq = float(MIN_PAIR_DIST) ** 2
    center_sq = float(CIRCUMCENTER_MAX_DIST) ** 2

    while len(scores) < RADIUS_N_VALID:
        if samples_done >= max_samples:
            raise RuntimeError(
                f"radius metric: only {len(scores)}/{RADIUS_N_VALID} valid "
                f"triplets after {samples_done} samples "
                f"(budget {max_samples}); giving up")
        b = min(RADIUS_SAMPLE_BATCH, max_samples - samples_done)
        i1 = np.random.randint(0, n, b)
        i2 = np.random.randint(0, n, b)
        i3 = np.random.randint(0, n, b)
        samples_done += b

        x1, y1 = xs[i1], ys[i1]
        x2, y2 = xs[i2], ys[i2]
        x3, y3 = xs[i3], ys[i3]

        # all three pairwise distances must exceed MIN_PAIR_DIST
        ok = (((x1 - x2) ** 2 + (y1 - y2) ** 2) > min_pair_sq)
        ok &= (((x2 - x3) ** 2 + (y2 - y3) ** 2) > min_pair_sq)
        ok &= (((x3 - x1) ** 2 + (y3 - y1) ** 2) > min_pair_sq)

        # circumcentre denominator = 2 * signed triangle area; ~0 => collinear
        d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        ok &= np.abs(d) > COLLINEAR_EPS

        d_safe = np.where(ok, d, 1.0)  # avoid div-by-zero; masked out below
        s1 = x1 * x1 + y1 * y1
        s2 = x2 * x2 + y2 * y2
        s3 = x3 * x3 + y3 * y3
        ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d_safe
        uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d_safe

        # circumcentre within CIRCUMCENTER_MAX_DIST of the image centre
        ok &= (((ux - cx0) ** 2 + (uy - cy0) ** 2) <= center_sq)

        # each point's gradient must point toward the circumcentre:
        # cos(v, g) > t  <=>  dot(v, g) > t * |v| * |g|   (t > 0)
        for xp, yp, gxp, gyp in (
            (x1, y1, gxs[i1], gys[i1]),
            (x2, y2, gxs[i2], gys[i2]),
            (x3, y3, gxs[i3], gys[i3]),
        ):
            vx = ux - xp
            vy = uy - yp
            dot = vx * gxp + vy * gyp
            vg = np.sqrt(vx * vx + vy * vy) * np.sqrt(gxp * gxp + gyp * gyp)
            ok &= dot > RADIUS_COS_MIN * vg

        if ok.any():
            r = np.sqrt((ux[ok] - x1[ok]) ** 2 + (uy[ok] - y1[ok]) ** 2)
            sc = np.minimum(np.minimum(mags[i1][ok], mags[i2][ok]),
                            mags[i3][ok])
            scores.extend(sc.tolist())
            radii.extend(r.tolist())

    # keep exactly RADIUS_N_VALID (last batch may overshoot; order is random)
    scores_a = np.asarray(scores[:RADIUS_N_VALID])
    radii_a = np.asarray(radii[:RADIUS_N_VALID])
    k = min(RADIUS_TOP_SCORE, scores_a.size)
    top = np.argpartition(scores_a, -k)[-k:]
    avg_radius = float(radii_a[top].mean())
    merit = 1000.0 / (avg_radius + 1.0)
    print(f"[radius] {samples_done} samples -> {len(scores)} valid triplets; "
          f"avg top-{k} radius={avg_radius:.2f} merit={merit:.4f}")
    return merit


def parse_shutter_seconds(s: str) -> float | None:
    """Parse a gphoto2 shutter-speed choice into seconds.

    Handles the two numeric forms Nikon reports -- fractions like '1/2000'
    and decimals like '0.5' or '1.3' (an optional trailing 's' is tolerated).
    Non-numeric choices (e.g. 'Bulb', 'Time') return None so the caller can
    drop them from the searchable range.
    """
    s = s.strip().rstrip("s").strip()
    try:
        if "/" in s:
            num, den = s.split("/")
            den = float(den)
            return float(num) / den if den else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _gp_error_code(exc: Exception) -> int | None:
    """Extract the gphoto2 numeric error code from a raised _set_config error.

    _set_config asserts with a message tuple whose LAST element is the gphoto2
    error code, e.g. ``assert err == GP_OK, ("set_config", name, err)`` -> the
    raised AssertionError has ``args == (("set_config", name, err),)``. So the
    code lives at ``args[0][-1]`` (not ``args[-1]`` -- that is the whole tuple,
    a subtle trap that silently defeated the old -110 busy check). Some asserts
    pass a bare int message; handle that too. Returns None if no int is found.
    """
    if isinstance(exc, AssertionError) and exc.args:
        msg = exc.args[0]
        if isinstance(msg, (tuple, list)) and msg and isinstance(msg[-1], int):
            return msg[-1]
        if isinstance(msg, int):
            return msg
    return None


def _fit_and_annotate(bgr: np.ndarray, lines: list[str],
                      max_w: int, max_h: int) -> np.ndarray:
    """Downscale `bgr` to fit (max_w, max_h) keeping aspect, then draw `lines`.

    Only ever shrinks (never upscales). The font size scales with the resized
    frame height so the overlay stays readable at any window size. Text is drawn
    green over a thin black outline for legibility on both bright and dark areas.
    Shared by the tkinter display and the headless PNG fallback.
    """
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    fs = max(0.4, nh / 900.0)
    for i, line in enumerate(lines):
        y = int(round(24 * fs)) + int(round(26 * fs)) * i
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 0), 1, cv2.LINE_AA)
    return out


class TkDisplay:
    """Minimal Tk image window driven from a procedural loop (no mainloop()).

    The focus hunt is a blocking procedural loop, so we cannot hand control to
    Tk's mainloop() during the run. Instead each show() rebuilds the frame as a
    PhotoImage and pumps pending Tk events with update(). The frame is encoded
    to PNG (via cv2) and handed to Tk as base64 -- this needs Tk >= 8.6 (PNG
    support) but avoids a Pillow dependency.

    The constructor raises RuntimeError if Tk is unavailable or there is no
    display, so the caller can fall back to headless PNG saving. Closing the
    window sets `closed` (it is NOT destroyed until destroy()/wait_close()), so
    a mid-hunt close cleanly flips the caller to the headless path.
    """

    def __init__(self, title: str, w: int, h: int) -> None:
        if tk is None:
            raise RuntimeError(f"tkinter unavailable: {_TK_IMPORT_ERROR}")
        try:
            self.root = tk.Tk()
            self.root.title(title)
            self.root.geometry(f"{w}x{h}")
            self.label = tk.Label(self.root, bg="black")
            self.label.pack(fill="both", expand=True)
            self.view_w, self.view_h = w, h
            self.closed = False
            self._imgref = None  # keep a ref so Tk does not GC the live image
            self.label.bind("<Configure>", self._on_resize)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.update()
        except tk.TclError as e:  # typically: no $DISPLAY
            raise RuntimeError(f"no display for tkinter: {e}") from e

    def _on_resize(self, event) -> None:
        # track the live widget size so frames scale with the window
        self.view_w, self.view_h = max(1, event.width), max(1, event.height)

    def _on_close(self) -> None:
        self.closed = True  # do NOT destroy here; the driver reacts to the flag

    def show(self, bgr: np.ndarray, lines: list[str]) -> None:
        if self.closed:
            return
        out = _fit_and_annotate(bgr, lines, self.view_w, self.view_h)
        ok, png = cv2.imencode(".png", out)
        if not ok:
            return
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self.label.configure(image=photo)
        self._imgref = photo  # prevent garbage collection of the shown image
        try:
            self.root.update()
        except tk.TclError:
            self.closed = True

    def wait_close(self) -> None:
        """Block (pumping events) until the user closes the window."""
        while not self.closed:
            try:
                self.root.update()
            except tk.TclError:
                break
            time.sleep(0.05)
        self.destroy()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


class FocusCamera:
    """gphoto2 camera wrapper with self-healing (re-init on error) operations."""

    def __init__(self) -> None:
        self.camera = None
        self.context = gp.gp_context_new()
        self.pos = 0                 # software focus position (rel. to start)
        self.headless = False        # set True if no display / GUI unavailable
        self.saved_frames = 0        # headless overlay PNGs written
        self.saved_captures = 0      # raw camera files saved to cwd
        self.current_round = 0       # 0 = baseline; increments each step-halving
        self.metric = "custom"       # custom|normalized|laplacian|radius
        self.max_samples = DEFAULT_MAX_SAMPLES  # 'radius' triplet-sample budget
        self.display = None          # lazily-created TkDisplay (None until first
                                     #  frame, or if we fell back to headless)
        self.liveview = False        # True while live view is (meant to be) on;
                                     #  reset() re-arms it so focus drive keeps
                                     #  working after a re-init

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        self._init_handle()

    def _init_handle(self) -> None:
        err, self.camera = gp.gp_camera_new()
        assert err == gp.GP_OK, err
        err = gp.gp_camera_init(self.camera, self.context)
        assert err == gp.GP_OK, err

    def reset(self) -> None:
        """Tear the handle fully down and re-create it -- clears any bad state."""
        try:
            if self.camera is not None:
                gp.gp_camera_exit(self.camera, self.context)
        except Exception:
            pass
        self.camera = None
        time.sleep(1.0)
        self._init_handle()
        # Re-arm live view if the hunt was running in it. Nikon manualfocusdrive
        # ONLY works in live view, so without this every focus drive after a
        # reset fails -- turning one camera hiccup into an endless -113 cascade.
        if self.liveview:
            self._arm_liveview_quiet()

    def _arm_liveview_quiet(self) -> None:
        """Enable live view best-effort WITHOUT triggering a reset.

        Used from reset() (and start_liveview); it calls _set_config directly
        rather than set_config_guarded so a failure cannot recurse back into
        reset() -> _arm_liveview_quiet -> reset() ...
        """
        for name, val in (("controlmode", "0"), ("viewfinder", 1)):
            try:
                self._set_config(name, val)
            except Exception as e:
                print(f"[warn] re-arm live view: set {name}={val} failed: {e}")
        time.sleep(0.8)

    def close(self) -> None:
        try:
            if self.camera is not None:
                gp.gp_camera_exit(self.camera, self.context)
        except Exception:
            pass
        self.camera = None
        if self.display is not None:
            self.display.destroy()
            self.display = None

    # -- config helpers ----------------------------------------------------
    def _set_config(self, name: str, value) -> None:
        """Set a single config leaf and push it to the camera (raises on error)."""
        err, config = gp.gp_camera_get_config(self.camera, self.context)
        assert err == gp.GP_OK, ("get_config", err)
        err, child = gp.gp_widget_get_child_by_name(config, name)
        assert err == gp.GP_OK, ("get_child", name, err)
        gp.gp_widget_set_value(child, value)
        err = gp.gp_camera_set_config(self.camera, config, self.context)
        assert err == gp.GP_OK, ("set_config", name, err)

    def set_config_guarded(self, name: str, value) -> bool:
        """Idempotent config set with reset+retry. Returns True on success."""
        for attempt in range(MAX_RETRY + 1):
            try:
                self._set_config(name, value)
                return True
            except Exception as e:
                print(f"[warn] set {name}={value} failed "
                      f"(try {attempt + 1}/{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        print(f"[warn] giving up on set {name}={value}; continuing")
        return False

    def get_config_choices(self, name: str) -> list[str]:
        """Read the list of allowed values for a radio/menu config leaf."""
        for attempt in range(MAX_RETRY + 1):
            try:
                err, config = gp.gp_camera_get_config(self.camera, self.context)
                assert err == gp.GP_OK, ("get_config", err)
                err, child = gp.gp_widget_get_child_by_name(config, name)
                assert err == gp.GP_OK, ("get_child", name, err)
                count = gp.gp_widget_count_choices(child)
                if isinstance(count, (tuple, list)):  # some bindings return (err, n)
                    count = count[1]
                out = []
                for i in range(count):
                    res = gp.gp_widget_get_choice(child, i)
                    out.append(res[1] if isinstance(res, (tuple, list)) else res)
                return out
            except Exception as e:
                print(f"[warn] read choices {name} failed "
                      f"(try {attempt + 1}/{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        raise RuntimeError(f"could not read config choices for {name}")

    def get_config_value(self, name: str):
        """Read the current value of a config leaf (best-effort, None on error)."""
        try:
            err, config = gp.gp_camera_get_config(self.camera, self.context)
            assert err == gp.GP_OK, ("get_config", err)
            err, child = gp.gp_widget_get_child_by_name(config, name)
            assert err == gp.GP_OK, ("get_child", name, err)
            res = gp.gp_widget_get_value(child)
            return res[1] if isinstance(res, (tuple, list)) else res
        except Exception as e:
            print(f"[warn] read value {name} failed: {e}")
            return None

    def auto_expose_to_clip(self, overexpose_stops: float = OVEREXPOSE_STOPS,
                            iso: str = BASE_ISO) -> str:
        """Pick a shutter speed that saturates the disk core, for 'radius'.

        The radius metric only tracks focus when the bright disk clips (a
        symmetric, unclipped limb is focus-insensitive -- see module notes).
        We therefore hold the camera at base ISO and a fixed aperture and drive
        the SHUTTER to a deliberate overexposure:

          1. Find the clipping "onset" -- the fastest shutter speed at which at
             least CLIP_ONSET_FRAC of the frame is saturated. Saturation grows
             monotonically with exposure time, so instead of probing every
             speed we bracket the onset: starting fast we double the exposure
             time each probe until it overshoots, then bisect that last bracket
             (stepping back down) to pin the exact onset in ~log2 shots.
          2. Multiply that exposure by 2**overexpose_stops (default 2 stops =>
             ~4x, the regime we validated) and set the nearest available speed.

        Aperture is left untouched (yours to choose -- it sets depth of field);
        only ISO and shutter are managed here. Exposure-mode is nudged to
        Manual best-effort, but on bodies where the mode dial is mechanical
        that set is a harmless no-op, so put the camera in M yourself.

        Returns the shutter-speed string it settled on. Raises RuntimeError if
        even the slowest available speed cannot reach the clipping onset (disk
        too dim -- open the aperture or raise ISO).
        """
        # Best-effort manual mode + base ISO; aperture stays as the user set it.
        self.set_config_guarded("expprogram", "M")
        self.set_config_guarded("iso", iso)

        cand = []
        for c in self.get_config_choices("shutterspeed"):
            secs = parse_shutter_seconds(c)
            if secs is not None and secs > 0:
                cand.append((secs, c))
        if not cand:
            raise RuntimeError("auto-expose: no numeric shutter speeds found")
        cand.sort()  # ascending seconds: fastest (least light) -> slowest
        secs = [s for s, _ in cand]
        vals = [v for _, v in cand]

        def clip_frac(idx: int) -> float:
            self.set_config_guarded("shutterspeed", vals[idx])
            time.sleep(SETTLE_S)
            actual = self.get_config_value("shutterspeed")
            if actual is not None and str(actual) != str(vals[idx]):
                print(f"[auto-expose] WARNING: requested shutterspeed "
                      f"{vals[idx]} but camera reports {actual} -- the write "
                      f"did not stick (mode not Manual, or liveview override?)")
            img, _, _ = self.capture_image()
            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if img.ndim == 3 else img)
            return float((gray >= CLIP_VALUE).mean())

        # Find the clipping onset. clip_frac() is monotonic in exposure time
        # (more time => more saturation), so we don't need to probe every speed:
        #   Phase 1 (coarse, doubling up): start at the fastest speed and, while
        #   still too short, jump to the first speed with at least ~2x the
        #   exposure time. This races up in ~log2 probes instead of one-at-a-time.
        #   Phase 2 (fine, stepping down): once a probe reaches the onset we have
        #   overshot; bisect the last (too-short, clips] bracket -- smaller and
        #   smaller steps -- to pin the FASTEST speed that still clips.
        def probe(i: int) -> float:
            frac = clip_frac(i)
            print(f"[auto-expose] probe {vals[i]}s "
                  f"({secs[i] * 1000:.3f} ms): clipped {frac * 100:.3f}% "
                  f"{'(onset)' if frac >= CLIP_ONSET_FRAC else '(too short)'}")
            return frac

        onset = None
        lo = -1  # fastest index verified BELOW onset (too short); -1 => none yet
        i = 0
        while i < len(vals):
            if probe(i) >= CLIP_ONSET_FRAC:
                onset = i
                break
            lo = i
            # jump to the first speed with >= 2x this exposure time, but never
            # skip past the slowest speed (so we always test it before giving up)
            target2 = secs[i] * 2.0
            nxt = i + 1
            while nxt < len(vals) - 1 and secs[nxt] < target2:
                nxt += 1
            i = max(nxt, i + 1)

        if onset is None:
            raise RuntimeError(
                f"auto-expose: even {vals[-1]}s does not reach the clipping "
                f"onset ({CLIP_ONSET_FRAC * 100:.3f}% of frame). Open the "
                f"aperture or raise ISO -- the disk is too dim to saturate.")

        # Phase 2: bisect (lo, onset] for the smallest exposure that still clips.
        # The doubling in phase 1 may have leapt over the true onset; this walks
        # back down with halving steps to the exact boundary the linear scan
        # would have found -- but in ~log2 probes instead of one per speed.
        hi = onset
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if probe(mid) >= CLIP_ONSET_FRAC:
                hi = mid
            else:
                lo = mid
        onset = hi

        target = secs[onset] * (2.0 ** overexpose_stops)
        pick = min(range(len(secs)), key=lambda i: abs(secs[i] - target))
        clamped = " (clamped to slowest available)" if secs[pick] < target \
            and pick == len(secs) - 1 else ""
        frac = clip_frac(pick)
        print(f"[auto-expose] onset={vals[onset]}s "
              f"({secs[onset] * 1000:.3f} ms); +{overexpose_stops} stops -> "
              f"target {target * 1000:.3f} ms; set {vals[pick]}s "
              f"({secs[pick] * 1000:.3f} ms){clamped}; "
              f"frame now clipped {frac * 100:.2f}%")
        return vals[pick]

    def _normalize_merit(self, bgr: np.ndarray, merit: float) -> float:
        """Divide merit by the 400th brightest pixel value of the frame.

        Compensates for exposure / brightness variation between frames so that
        sharpness comparisons are not confused by illumination differences.
        Returns `merit` unchanged if the image is too small or the reference
        pixel is zero (avoids division by zero on a blank frame).
        """
        gray = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3
                else bgr).ravel().astype(np.float64)
        if gray.size < 400:
            return merit
        ref = float(np.partition(gray, -400)[-400])
        if ref == 0.0:
            return merit
        return merit / ref

    def start_liveview(self) -> None:
        # Hand PC control to the camera logic and raise the mirror into live
        # view. Focus driving on Nikon happens in live view, so we keep it on
        # for the manualfocusdrive commands; the actual measurement is still a
        # full-resolution shot taken via gp_camera_capture. Failures non-fatal.
        self.set_config_guarded("controlmode", "0")
        self.set_config_guarded("viewfinder", 1)
        time.sleep(0.8)
        # Set the flag LAST: only now should a later reset() try to re-arm live
        # view (setting it earlier could recurse through the guarded sets above).
        self.liveview = True

    def stop_liveview(self) -> None:
        # Clear the flag FIRST so the guarded set below (or any reset it spawns)
        # does not try to re-arm the live view we are intentionally turning off.
        self.liveview = False
        self.set_config_guarded("viewfinder", 0)

    # -- image capture -----------------------------------------------------
    def _capture_image_once(self) -> tuple[np.ndarray, bytes, str]:
        # Trigger a real exposure, download the resulting file off the card,
        # decode it, then delete it from the card so we do not fill it during a
        # long hunt. NOTE: the camera must be shooting a decodable format (JPEG);
        # a RAW-only (.NEF) frame will NOT decode with cv2.imdecode.
        # Returns (decoded BGR image, raw file bytes, original filename) so the
        # caller can persist the exact downloaded file under a merit-tagged name.
        err, path = gp.gp_camera_capture(
            self.camera, gp.GP_CAPTURE_IMAGE, self.context)
        assert err == gp.GP_OK, ("capture", err)
        err, cam_file = gp.gp_file_new()
        assert err == gp.GP_OK, ("file_new", err)
        err = gp.gp_camera_file_get(
            self.camera, path.folder, path.name,
            gp.GP_FILE_TYPE_NORMAL, cam_file, self.context)
        assert err == gp.GP_OK, ("file_get", err)
        err, data = gp.gp_file_get_data_and_size(cam_file)
        assert err == gp.GP_OK, ("get_data", err)
        raw = memoryview(data).tobytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # best-effort cleanup; failure to delete is non-fatal
        try:
            gp.gp_camera_file_delete(
                self.camera, path.folder, path.name, self.context)
        except Exception:
            pass
        if img is None:
            raise RuntimeError(
                f"captured image {path.name} failed to decode "
                f"(shooting RAW/.NEF? set the camera to JPEG)")
        return img, raw, path.name

    def capture_image(self) -> tuple[np.ndarray, bytes, str]:
        """Shoot one full-resolution frame, self-healing on error.

        Returns (decoded BGR image, raw file bytes, original camera filename).
        """
        for attempt in range(MAX_RETRY + 1):
            try:
                return self._capture_image_once()
            except Exception as e:
                print(f"[warn] capture failed (try {attempt + 1}/"
                      f"{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        raise RuntimeError("image capture failed after retries")

    # -- focus drive -------------------------------------------------------
    def drive_focus(self, delta: int, strict: bool = False) -> None:
        """Move focus RELATIVELY by `delta` units, split into safe chunks.

        In normal mode (strict=False) a failed chunk is NOT retried blindly (the
        motor may have moved a partial, unknown amount). How we react depends on
        the camera's error code, and in every failure case we ABANDON the rest
        of this move rather than hammering the remaining chunks -- the closed-
        loop hunt corrects the resulting small offset on the next measurement:

          * -110 GP_ERROR_CAMERA_BUSY -- fully rejected, motor provably did not
            move; wait 1 s and retry the chunk ONCE, then give up the move.
          * -113 GP_ERROR_CAMERA_ERROR -- the camera refused the drive, almost
            always because the lens is at the end of its focus travel (or is not
            in a drivable state). The handle is fine, so we do NOT reset (a reset
            would drop live view and make every later drive fail); we just stop
            this move.
          * anything else -- possibly a broken handle; reset once, then stop.

        In strict mode (strict=True) any failure raises RuntimeError so the
        caller knows the final position was not reached.
        """
        if delta == 0:
            return
        sign = 1 if delta > 0 else -1
        remaining = abs(delta)
        while remaining > 0:
            move = sign * min(DRIVE_CHUNK, remaining)
            try:
                self._set_config("manualfocusdrive", move)
                self.pos += move
                remaining -= abs(move)
                time.sleep(MOVE_PAUSE)
                continue
            except Exception as e:
                # Stash the exception: Python clears `e` when the except block
                # exits, so the branches below must use `err`, not `e`.
                err = e
                code = _gp_error_code(e)

            if code == -110:  # busy: motor did not move -> safe to retry once
                print(f"[warn] focus drive {move} busy (-110); "
                      f"waiting 1 s and retrying once")
                time.sleep(1.0)
                try:
                    self._set_config("manualfocusdrive", move)
                    self.pos += move
                    remaining -= abs(move)
                    time.sleep(MOVE_PAUSE)
                    continue
                except Exception as e2:
                    if strict:
                        raise RuntimeError(
                            f"final focus move chunk {move} failed after "
                            f"busy-retry: {e2}") from e2
                    print(f"[warn] focus drive {move} still busy; "
                          f"abandoning this move (closed loop will re-measure)")
                    return
            elif code == -113:  # camera refused: end of travel / not drivable
                if strict:
                    raise RuntimeError(
                        f"final focus move chunk {move} refused (-113): "
                        f"{err}") from err
                print(f"[warn] focus drive {move} refused by camera (-113) -- "
                      f"likely end of focus travel; abandoning this move")
                return
            else:  # unknown: the handle may be bad -> reset once, then stop
                if strict:
                    raise RuntimeError(
                        f"final focus move chunk {move} failed: {err}") from err
                print(f"[warn] focus drive {move} failed: {err} "
                      f"(assuming no move; resetting, then abandoning this move)")
                self.reset()
                return

    def move_to(self, target_pos: int, reversing: bool = False,
                strict: bool = False) -> None:
        """Drive to an absolute (tracked) position from the current position."""
        if reversing and BACKLASH:
            # take up slack in the new direction, then approach the target
            direction = 1 if target_pos > self.pos else -1
            self.drive_focus(direction * BACKLASH, strict=strict)
        self.drive_focus(target_pos - self.pos, strict=strict)

    # -- measurement + display --------------------------------------------
    def measure(self, label: str) -> float:
        """Settle, shoot AVG_FRAMES real frames, average their focus merit.

        Each downloaded file is saved into the current directory under a name
        that carries its own merit value (and tracked position), so the frames
        are self-documenting and sort by sharpness. Returns the mean merit and
        shows the last frame to the user.
        """
        time.sleep(SETTLE_S)
        scores = []
        last = None
        for _ in range(max(1, AVG_FRAMES)):
            img, raw, name = self.capture_image()
            if self.metric == "laplacian":
                value = focus_merit_laplacian(img)
            elif self.metric == "radius":
                value = focus_merit_radius(img, self.max_samples)
            else:
                value = focus_merit(img)
                if self.metric == "normalized":
                    value = self._normalize_merit(img, value)
            scores.append(value)
            last = img
            self._save_capture(raw, name, value)
        sharp = float(np.mean(scores))
        # Always report the ACTUAL current focus position (cam.pos), not just
        # the intended target -- the two diverge if a drive chunk is dropped.
        print(f"[measure] pos={self.pos:+d} value={sharp:.3f} ({label})")
        self._show(last, label, sharp)
        # Give the camera time to fully resume live view after the shutter
        # fired; without this pause the next manualfocusdrive arrives while
        # the camera is still transitioning and returns GP_ERROR_CAMERA_BUSY.
        time.sleep(1.0)
        return sharp

    def _save_capture(self, raw: bytes, name: str, value: float) -> None:
        """Write the raw downloaded file to cwd, tagging the name with `value`."""
        import os
        stem, ext = os.path.splitext(name)
        # zero-padded merit so a plain lexicographic sort orders by sharpness;
        # pos and original stem keep each frame traceable, seq avoids collisions.
        out = (f"round{self.current_round:02d}_merit_{value:012.3f}_pos{self.pos:+06d}_"
               f"{stem}_{self.saved_captures:03d}{ext}")
        try:
            with open(out, "wb") as f:
                f.write(raw)
        except OSError as e:
            print(f"[warn] could not save {out}: {e}")
        self.saved_captures += 1

    def _show(self, bgr: np.ndarray, label: str, sharp: float) -> None:
        lines = [label, f"pos={self.pos}  value={sharp:.3f}"]
        if not self.headless:
            # Lazily create the Tk window on the first frame.
            if self.display is None:
                try:
                    self.display = TkDisplay(WINDOW, GUI_INIT_W, GUI_INIT_H)
                except RuntimeError as e:
                    print(f"[warn] no GUI ({e}); falling back to saving PNGs "
                          f"into {PNG_DIR}/")
                    self.headless = True
            if self.display is not None:
                if self.display.closed:
                    # user closed the window mid-hunt -> go headless
                    print(f"[info] display window closed; saving PNGs into "
                          f"{PNG_DIR}/ from here on")
                    self.display.destroy()
                    self.display = None
                    self.headless = True
                else:
                    try:
                        self.display.show(bgr, lines)
                        return
                    except Exception as e:
                        print(f"[warn] GUI update failed ({e}); falling back to "
                              f"saving PNGs into {PNG_DIR}/")
                        self.headless = True
                        self.display = None
        # headless fallback: same downscaled+annotated frame, written to disk
        import os
        vis = _fit_and_annotate(bgr, lines, GUI_INIT_W, GUI_INIT_H)
        os.makedirs(PNG_DIR, exist_ok=True)
        path = os.path.join(PNG_DIR, f"frame_{self.saved_frames:03d}.png")
        cv2.imwrite(path, vis)
        self.saved_frames += 1


def hunt(cam: FocusCamera) -> None:
    moves_used = 0

    # Baseline at the starting (roughly-focused) position.
    best_pos = 0
    cam.current_round = 0
    best_sharp = cam.measure("start (roughly focused)")
    print(f"[baseline] pos={best_pos} value={best_sharp:.3f}")

    step = INITIAL_STEP
    pref = -1  # nominal first-probe direction (tie-break); both sides are probed
    round_num = 1

    while step >= MIN_STEP:
        # Re-measure best_pos fresh at the start of each round so the
        # comparison threshold reflects current conditions, not a stale
        # reading from a coarser pass.
        cam.current_round = round_num
        cam.move_to(best_pos)
        best_sharp = cam.measure(f"round {round_num} start (step={step})")
        print(f"[round {round_num}] re-measured pos={best_pos} "
              f"value={best_sharp:.3f} step={step}")

        # probed: all positions measured so far this round mapped to their merit.
        # Misses are counted globally (all positions above/below current best_pos
        # that were measured and lost), so history is never wiped when best_pos
        # changes.  This prevents both re-probing already-measured positions and
        # the "best_pos bounce resets all history" bug.
        probed: dict[int, float] = {best_pos: best_sharp}
        # Track raw best (ignoring NOISE_MARGIN) to correct best_pos at round end.
        raw_best_pos, raw_best_sharp = best_pos, best_sharp

        while moves_used < MAX_MOVES:
            # Misses on each side = probed positions that are not the current
            # best and whose merit did not beat best_sharp * NOISE_MARGIN.
            misses_above = sum(1 for p, m in probed.items()
                               if p > best_pos
                               and m <= best_sharp * NOISE_MARGIN)
            misses_below = sum(1 for p, m in probed.items()
                               if p < best_pos
                               and m <= best_sharp * NOISE_MARGIN)
            if misses_above >= 2 and misses_below >= 2:
                break

            # Pick side with fewer misses; tie-break by pref direction.
            if misses_above >= 2:
                d = -1
            elif misses_below >= 2:
                d = 1
            elif misses_above < misses_below:
                d = 1
            elif misses_below < misses_above:
                d = -1
            else:
                d = pref

            # Find nearest position in direction d not yet measured this round.
            k = 1
            while (best_pos + d * k * step) in probed:
                k += 1
            cand = best_pos + d * k * step

            reversing = (d != pref)
            cam.move_to(cand, reversing=reversing)
            moves_used += 1
            sharp = cam.measure(
                f"round={round_num} step={step} dir={'+' if d > 0 else '-'} "
                f"cand={cand}"
            )
            probed[cand] = sharp
            if sharp > raw_best_sharp:
                raw_best_pos, raw_best_sharp = cand, sharp

            better = sharp > best_sharp * NOISE_MARGIN
            print(f"[hunt] round={round_num} step={step:<4d} "
                  f"dir={'+' if d > 0 else '-'} "
                  f"pos={cand:<6d} value={sharp:<12.3f} "
                  f"best={best_sharp:.3f}@{best_pos} "
                  f"ma={misses_above} mb={misses_below} "
                  f"{'<-- new best' if better else ''}")
            if better:
                best_sharp, best_pos, pref = sharp, cand, d

        if moves_used >= MAX_MOVES:
            print("[warn] hit MAX_MOVES safety cap; stopping hunt")

        # If NOISE_MARGIN prevented the raw merit winner from being tracked,
        # correct best_pos now so the next round starts from the right place.
        if raw_best_pos != best_pos:
            print(f"[hunt] round {round_num}: raw-best correction "
                  f"{best_pos}({best_sharp:.3f}) -> "
                  f"{raw_best_pos}({raw_best_sharp:.3f})")
            best_pos, best_sharp = raw_best_pos, raw_best_sharp
        print(f"[hunt] round {round_num} (step={step}) exhausted; "
              f"best={best_sharp:.3f}@{best_pos}")
        step //= 2
        round_num += 1

    # Settle exactly on the best position (approach with backlash comp if set).
    # strict=True: any chunk failure raises immediately rather than silently
    # skipping -- we must know if the lens did not reach the target.
    cam.current_round = round_num
    cam.move_to(best_pos, reversing=True, strict=True)
    final = cam.measure(f"FINAL pos={best_pos}")
    print("=" * 60)
    print(f"FINAL focus position = {cam.pos} steps from start "
          f"(target {best_pos})")
    print(f"FINAL value          = {final:.3f} (peak seen {best_sharp:.3f})")
    print("=" * 60)
    if not cam.headless and cam.display is not None:
        print("Close the image window to exit...")
        cam.display.wait_close()
        cam.display = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-hunt optimal focus of the filtered Sun via full-res shots.")
    parser.add_argument(
        "--metric", choices=["custom", "normalized", "laplacian", "radius"],
        default="custom",
        help=(
            "Sharpness metric used for focus decisions (default: custom). "
            "'custom': edge-cluster minimum over top-N Sobel gradient pixels "
            "(coherent-edge focus, noise-spike resistant). "
            "'normalized': same divided by the 400th brightest pixel value "
            "(compensates for exposure variation between frames). "
            "'laplacian': Laplacian variance -- standard optical autofocus metric, "
            "well-suited to solar/eclipse images with hard limb edges; "
            "insensitive to overall brightness changes. "
            "'radius': fits circles to random triplets of strong-gradient limb "
            "pixels and rewards a smaller fitted disk radius "
            "(1000 / (avg_radius + 1)); tuned to the solar limb."
        ))
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help=(
            "'radius' metric only: max random triplets sampled per frame while "
            f"collecting {RADIUS_N_VALID} valid ones (default: {DEFAULT_MAX_SAMPLES}). "
            "The measurement crashes if this budget is exhausted first."
        ))
    parser.add_argument(
        "--auto-expose", action="store_true",
        help=(
            "Before hunting, drive the shutter to saturate the disk core "
            "(required for 'radius' to be focus-sensitive; harmless but "
            "pointless for the other metrics). Automatically enabled for "
            "'radius' unless --no-auto-expose is given."
        ))
    parser.add_argument(
        "--no-auto-expose", action="store_true",
        help="Skip auto-exposure even when --metric radius is selected.")
    parser.add_argument(
        "--overexpose-stops", type=float, default=OVEREXPOSE_STOPS,
        help=(
            "Auto-exposure: stops past the clipping onset to expose "
            f"(default: {OVEREXPOSE_STOPS}; ~4x saturation of the disk core)."
        ))
    parser.add_argument(
        "--iso", default=BASE_ISO,
        help=f"ISO held during auto-exposure (default: {BASE_ISO}).")
    parser.add_argument(
        "--image-quality", default=DEFAULT_IMAGE_QUALITY,
        help=(
            "Camera image-quality setting used DURING the hunt "
            f"(default: '{DEFAULT_IMAGE_QUALITY}'). Frames are decoded with "
            "cv2.imdecode, which needs a JPEG/TIFF frame -- a RAW/.NEF only "
            "decodes as a small embedded preview, so the metrics would run on a "
            "low-res image. JPEG Fine keeps full resolution and decodes "
            "natively. The camera's current quality is read at startup and "
            "restored on exit. Pass the exact camera label (e.g. 'JPEG Fine', "
            "'JPEG Normal', 'TIFF', 'NEF (Raw)'); use the empty string to leave "
            "the camera's quality untouched."
        ))
    args = parser.parse_args()

    cam = FocusCamera()
    cam.metric = args.metric
    cam.max_samples = args.max_samples
    print(f"[info] sharpness metric: {args.metric}")
    do_auto_expose = (args.auto_expose
                      or (args.metric == "radius" and not args.no_auto_expose))

    orig_quality = None    # camera's image quality as found at startup
    quality_changed = False  # True once we have written a different quality

    try:
        cam.open()
        # Snapshot the camera's current image quality, then switch to the
        # requested capture format for the hunt. Skipped if --image-quality is
        # empty (leave the camera as-is) or already matches.
        if args.image_quality:
            orig_quality = cam.get_config_value("imagequality")
            print(f"[info] image quality currently '{orig_quality}'")
            if orig_quality is None:
                print("[warn] could not read the current image quality; will "
                      "still switch for the hunt but CANNOT auto-restore it")
            if orig_quality != args.image_quality:
                if cam.set_config_guarded("imagequality", args.image_quality):
                    quality_changed = True
                    print(f"[info] image quality set to "
                          f"'{args.image_quality}' for the hunt")
                else:
                    print(f"[warn] could not set image quality to "
                          f"'{args.image_quality}'; continuing as-is")
            else:
                print(f"[info] image quality already '{args.image_quality}'; "
                      f"leaving it")
        cam.start_liveview()
        if do_auto_expose:
            cam.auto_expose_to_clip(args.overexpose_stops, args.iso)
        hunt(cam)
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    finally:
        cam.stop_liveview()
        # Restore the original image quality while the camera handle is still
        # alive. We build the single final status line here, then close the
        # camera, then print that line LAST so it is unmistakably the last thing
        # the script emits (blank line above it for visibility).
        restore_line = None
        if quality_changed:
            if orig_quality is None:
                # We switched formats but never learned the original value.
                restore_line = (
                    f"WARNING: image quality was NOT restored -- original value "
                    f"was unknown; it is now '{args.image_quality}', set the "
                    f"quality you want manually in the camera menu")
                print(restore_line)  # now too, in case anything below fails
            elif cam.set_config_guarded("imagequality", orig_quality):
                restore_line = f"Image quality restored to {orig_quality}"
            else:
                restore_line = (
                    f"WARNING: image quality was NOT restored -- set it back to "
                    f"'{orig_quality}' manually in the camera menu")
                print(restore_line)  # now too, in case anything below fails
        cam.close()
        if restore_line is not None:
            print()  # one empty line so the final status stands out
            print(restore_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
