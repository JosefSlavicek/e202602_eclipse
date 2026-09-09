#!/usr/bin/env python3
"""Find the optimal focus on a star, deep-sky style (intended for M31).

This is a close relative of ``focus_hunt_visual.py``. That script captures a
9-frame focus bracket and lets you eyeball it in a zoomable viewer -- and that
is all. This one keeps the same viewer but adds the two things you need to turn
"looking at nine pictures" into "converging on best focus":

  * ``Re-capture``  -- asks for a new focus step, then shoots 9 fresh frames
    CENTRED on the focus of the frame you are currently looking at. Nothing is
    displayed while it captures (a plain "CAPTURING k/9" screen is shown); when
    it finishes the frames reappear at the SAME zoom and pan you had. Repeat,
    narrowing the step, to zoom in on the peak.

  * ``Select star`` -- draw a freehand closed curve around a star. Its outline is
    then drawn on every frame (it survives re-capture -- the mount is tracking,
    so the star does not move in the frame). A status line shows a sharpness
    figure of merit for that region on ALL nine frames at once, so the sharpest
    frame is obvious.

Figure of merit
---------------
Inside the selected region we estimate the local sky background (median of a
ring just outside the curve), subtract it, clip negatives to zero, normalise the
remaining pixel values to sum to 1 (so p_i is "fraction of the star's light in
pixel i") and report::

    sum_i p_i^2            higher  = light more concentrated = SHARPER
    1 / sum_i p_i^2        "effective pixel count", lower = sharper

sum p_i^2 is the order-2 Renyi participation ratio; it is exposure-invariant and,
unlike Shannon entropy, is not dominated by the large tail of near-zero
background pixels.

Over-burn indicator
-------------------
If ANY pixel inside the selected region is >= 0.7 * the maximum possible value
(255 for JPEG, the raw white level for NEF, measured on the LINEAR data), a big
red ``OVERBURN`` bar is drawn across the viewer and the tripped frames are
flagged in the status line. A clipped star core makes every sharpness metric
lie, so treat those frames as unusable and pick a fainter star or a shorter sub.

NEF vs JPEG (``--format``)
-------------------------
Default is ``nef``: the linear raw star profile is what ``sum p_i^2`` is really
measuring, and the in-camera JPEG pipeline (noise reduction, sharpening, tone
curve) distorts small star PSFs. NEF frames are decoded linearly with rawpy and
shown truly linearly (pixel value / raw white level, no screen stretch), so a
faint field reads dark -- lean on zoom and the sharpness metric, not the look.
``--format jpeg`` shoots 'JPEG Fine' instead and shows it as-is.

On close
--------
When you close the window the script drives the lens to the focus of the frame
you were last looking at, prints what it is doing, and prints ``SUCCESS`` or
``FAILED`` (with the manual-move delta on failure). So on a clean exit the lens
is left AT the focus you chose. The camera is then returned to shooting NEF.

Memory note: nine full-resolution frames are held in RAM (roughly 0.3-0.6 GB per
frame for a 45 MP NEF). Run on the machine you process on.

Run with the python that has gphoto2 + OpenCV bound (the camera machine); NEF
mode additionally needs ``rawpy`` importable there.

Usage
-----
    camera/focus_hunt_stars.py ABS_POS FIRST_STEP STEP [--format {nef,jpeg}]
    e.g.  camera/focus_hunt_stars.py -1750 0 12

ABS_POS     absolute focus position of the CURRENT focus (LABEL ONLY -- it does
            not move the lens; it just makes the printed numbers absolute).
FIRST_STEP  offset from the current focus at which the first bracket is centred
            (image index 4 of 9).
STEP        focus-unit spacing between consecutive images of the first bracket.
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import cv2

# Reuse the self-healing camera wrapper and timing from focus_hunt.py (same
# directory). Importing it also sets OPENCV_LOG_LEVEL before cv2 loads and pulls
# in gphoto2.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_hunt import FocusCamera, SETTLE_S, set_final_image_quality  # noqa: E402

try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except Exception as e:  # pragma: no cover - GUI is required for this tool
    print(f"[fatal] tkinter is required for the viewer but is unavailable: {e}")
    sys.exit(1)

try:
    import rawpy  # only needed for --format nef
except Exception:  # pragma: no cover - optional dependency
    rawpy = None

# ---- tunables -----------------------------------------------------------------
N_IMAGES = 9              # frames per bracket
CENTRE_INDEX = 4         # 0-based index that lands on the bracket centre
VIEW_W, VIEW_H = 1100, 800   # initial viewer window size (resizable)
MAX_MAG = 16             # deepest integer magnification (Nx)
CAP_SETTLE_S = 1.0       # let live view resume after a full-res shot before the
                         #  next focus move (avoids GP_ERROR_CAMERA_BUSY)
OUT_DIR = "focus_stars"  # where downloaded frames are saved on the PC

# metric / region
RING_GAP_PX = 4          # blank margin between the drawn curve and the bg ring
RING_WIDTH_PX = 10       # width of the background-estimation ring
OVERBURN_FRAC = 0.7      # pixel >= this * max-possible -> over-burn warning

NEF_WHITE = 65535.0      # rawpy postprocess(output_bps=16) normalises the raw
                         #  white point to full 16-bit; the viewer shows NEF
                         #  frames as gray / NEF_WHITE (pure linear, no stretch)


# ---------------------------------------------------------------------------
# Frame model + decoding
# ---------------------------------------------------------------------------
@dataclass
class FocusFrame:
    """One captured frame: linear data for metrics + a display image."""

    gray: np.ndarray            # float32 HxW, linear-ish; metric + over-burn
    white: float                # max-possible value of `gray` (clip reference)
    fmt: str                    # "nef" | "jpeg"
    jpeg_bgr: np.ndarray | None = None   # kept for the JPEG display path
    disp: np.ndarray | None = None       # uint8 HxWx3 BGR shown by the viewport

    def apply_display(self) -> None:
        """(Re)build `disp`. NEF is shown pure-linear (gray / white); JPEG as-is."""
        if self.fmt == "jpeg":
            self.disp = self.jpeg_bgr
            return
        y = np.clip(self.gray / self.white, 0.0, 1.0)
        g = (y * 255.0).astype(np.uint8)
        self.disp = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def decode_frame(raw: bytes, name: str, fmt: str) -> FocusFrame:
    """Decode one downloaded camera file into a FocusFrame."""
    if fmt == "nef":
        if rawpy is None:
            raise RuntimeError(
                "--format nef needs rawpy, which is not importable in this "
                "python (pip install rawpy)")
        with rawpy.imread(io.BytesIO(raw)) as r:
            # Linear (gamma 1), no auto-brightening, unit white balance and the
            # raw colour space: keep every output value proportional to photons
            # so the sharpness metric and the clip test are meaningful.
            rgb16 = r.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16,
                user_wb=[1.0, 1.0, 1.0, 1.0], user_flip=0,
                output_color=rawpy.ColorSpace.raw)
        gray = rgb16.astype(np.float32).mean(axis=2)
        return FocusFrame(gray=gray, white=NEF_WHITE, fmt="nef")

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(
            f"{name}: cv2 could not decode this frame -- is the camera really "
            f"on JPEG? (use --format nef for raw)")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return FocusFrame(gray=gray, white=255.0, fmt="jpeg", jpeg_bgr=bgr)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def capture_bracket(cam: FocusCamera, center_pos: int, step: int, fmt: str,
                    status_cb=None):
    """Shoot a 9-frame bracket centred (in tracked focus units) on center_pos.

    The lens is swept MONOTONICALLY from wherever it currently is through the
    nine target positions (no return move afterwards -- cam.pos is left at the
    last, highest target). Returns (frames, meta, raws, names).
    """
    targets = [center_pos + (i - CENTRE_INDEX) * step for i in range(N_IMAGES)]
    order = sorted(range(N_IMAGES), key=lambda i: targets[i])
    raws: list[bytes | None] = [None] * N_IMAGES
    names: list[str | None] = [None] * N_IMAGES

    if not cam.liveview:
        cam.start_liveview()

    for n, i in enumerate(order):
        if status_cb:
            status_cb(n, N_IMAGES, f"moving to tracked_pos {targets[i]:+d}")
        cam.move_to(targets[i])
        time.sleep(SETTLE_S)
        if status_cb:
            status_cb(n, N_IMAGES, f"exposing frame {i}")
        raw, name = cam.capture_raw()
        raws[i], names[i] = raw, name
        print(f"[capture] {n + 1}/{N_IMAGES} idx={i} target={targets[i]:+d} "
              f"pos={cam.pos:+d} bytes={len(raw)} name={name}")
        time.sleep(CAP_SETTLE_S)

    frames: list[FocusFrame] = []
    for i in range(N_IMAGES):
        if status_cb:
            status_cb(i, N_IMAGES, f"decoding frame {i}")
        frames.append(decode_frame(raws[i], names[i], fmt))

    for f in frames:
        f.apply_display()

    meta = [{"index": i, "tracked_pos": targets[i],
             "offset": targets[i] - center_pos} for i in range(N_IMAGES)]
    return frames, meta, raws, names


def save_bracket(raws, names, meta, out_root: str, tag: str) -> None:
    """Persist the nine downloaded raw files under out_root/tag/."""
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    for i, (raw, name) in enumerate(zip(raws, names)):
        _, ext = os.path.splitext(name or "")
        p = os.path.join(d, f"img{i}_pos{meta[i]['tracked_pos']:+d}{ext}")
        try:
            with open(p, "wb") as fh:
                fh.write(raw)
        except OSError as e:
            print(f"[warn] could not save {p}: {e}")
    print(f"[info] saved bracket -> {d}/")


# ---------------------------------------------------------------------------
# Viewport transform (adapted from focus_hunt_visual.render_viewport, with the
# same integer-zoom / no-interpolation behaviour, plus the crop geometry `vt`
# returned so the caller can map image <-> screen coordinates for the overlay).
# ---------------------------------------------------------------------------
def render_viewport(img: np.ndarray, num: int, den: int, cx: float, cy: float,
                    view_w: int, view_h: int):
    """Render a view_w x view_h BGR frame of `img` at integer zoom num/den.

    Returns (frame, new_cx, new_cy, cw, ch, vt) where
    vt = (x0, y0, num, den, ox, oy) is everything needed to map a source pixel
    (sx, sy) to a screen pixel:  vx = ox + (sx - x0) / den * num  (and same y).
    """
    sh, sw = img.shape[:2]
    cw = min(sw, max(1, math.ceil(view_w * den / num)))
    ch = min(sh, max(1, math.ceil(view_h * den / num)))
    x0 = int(round(cx - cw / 2.0))
    y0 = int(round(cy - ch / 2.0))
    x0 = max(0, min(x0, sw - cw))
    y0 = max(0, min(y0, sh - ch))
    new_cx, new_cy = x0 + cw / 2.0, y0 + ch / 2.0

    crop = img[y0:y0 + ch, x0:x0 + cw]
    sub = crop[::den, ::den]
    if num > 1:
        disp = np.repeat(np.repeat(sub, num, axis=0), num, axis=1)
    else:
        disp = sub
    disp = disp[:view_h, :view_w]

    frame = np.zeros((view_h, view_w, 3), np.uint8)
    dh, dw = disp.shape[:2]
    oy, ox = max(0, (view_h - dh) // 2), max(0, (view_w - dw) // 2)
    frame[oy:oy + dh, ox:ox + dw] = disp
    return frame, new_cx, new_cy, cw, ch, (x0, y0, num, den, ox, oy)


def _to_photo(frame_bgr: np.ndarray) -> "tk.PhotoImage":
    """Encode a BGR frame to a Tk PhotoImage (PNG+base64, no Pillow)."""
    ok, png = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class Viewer:
    """Integer-zoom pannable inspector with re-capture + star-sharpness tools."""

    def __init__(self, cam: FocusCamera, frames: list[FocusFrame],
                 meta: list[dict], abs_pos: int, step: int, fmt: str,
                 out_root: str) -> None:
        self.cam = cam
        self.frames = frames
        self.meta = meta
        self.abs_pos = abs_pos
        self.step = step
        self.fmt = fmt
        self.out_root = out_root
        self.idx = CENTRE_INDEX
        self.bracket_seq = 0

        self.num, self.den = 1, 1
        h, w = frames[0].disp.shape[:2]
        self.cx, self.cy = w / 2.0, h / 2.0
        self.vis_w, self.vis_h = w, h
        self._photo = None
        self._base_frame = None      # rendered frame before overlays (lasso use)
        self.vt = (0, 0, 1, 1, 0, 0)

        self._poly = None            # (K, 2) float32 source-pixel polygon
        self._region = None          # (mask_bool, ring_bool)
        self.metrics = None          # list[dict] per frame, or None
        self._select_mode = False
        self._lasso: list[tuple[int, int]] = []
        self._capturing = False

        self._build_ui()
        self.root.update_idletasks()
        self.one_to_one()

    # -- UI --------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Focus hunt (stars)")
        self.root.geometry(f"{VIEW_W}x{VIEW_H}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        bar = tk.Frame(self.root)
        bar.pack(side="top", fill="x")

        def btn(text, cmd):
            b = tk.Button(bar, text=text, command=cmd, takefocus=0)
            b.pack(side="left", padx=1, pady=2)
            return b

        btn("◀ Prev (p)", self.prev_image)
        btn("Next (n) ▶", self.next_image)
        tk.Label(bar, text=" ").pack(side="left")
        btn("Zoom − (-)", self.zoom_out)
        btn("Zoom + (+)", self.zoom_in)
        btn("Fit (f)", self.fit)
        btn("1:1 (1)", self.one_to_one)
        tk.Label(bar, text=" ").pack(side="left")
        btn("◀", lambda: self.pan(-1, 0))
        btn("▲", lambda: self.pan(0, -1))
        btn("▼", lambda: self.pan(0, 1))
        btn("▶", lambda: self.pan(1, 0))
        tk.Label(bar, text=" ").pack(side="left")
        btn("Re-capture (r)", self.recapture)
        btn("Select star (x)", self.start_select)
        btn("Clear sel (c)", self.clear_select)
        btn("Quit (q)", self._on_quit)

        self.info = tk.Label(self.root, anchor="w", font=("TkFixedFont", 10))
        self.info.pack(side="top", fill="x")
        self.metricbar = tk.Label(self.root, anchor="w",
                                  font=("TkFixedFont", 10))
        self.metricbar.pack(side="top", fill="x")
        self.hint = tk.Label(self.root, anchor="w", fg="#666",
                             font=("TkFixedFont", 9))
        self.hint.pack(side="top", fill="x")

        self.view = tk.Label(self.root, bg="black")
        self.view.pack(side="top", fill="both", expand=True)
        self.view.bind("<Configure>", lambda _e: self.render())
        self.view.bind("<Button-1>", self._on_press)
        self.view.bind("<B1-Motion>", self._on_drag)
        self.view.bind("<ButtonRelease-1>", self._on_release)

        self.root.bind("<Key>", self._on_key)

    def _on_key(self, event):
        if self._capturing:
            return "break"
        k = event.keysym
        if k in ("plus", "equal", "KP_Add", "i", "I"):
            self.zoom_in()
        elif k in ("minus", "underscore", "KP_Subtract", "o", "O"):
            self.zoom_out()
        elif k in ("Left", "h"):
            self.pan(-1, 0)
        elif k in ("Right", "l"):
            self.pan(1, 0)
        elif k in ("Up", "k"):
            self.pan(0, -1)
        elif k in ("Down", "j"):
            self.pan(0, 1)
        elif k in ("n", "bracketright", "space", "period"):
            self.next_image()
        elif k in ("p", "bracketleft", "comma"):
            self.prev_image()
        elif k in ("f", "F"):
            self.fit()
        elif k == "1":
            self.one_to_one()
        elif k in ("r", "R"):
            self.recapture()
        elif k in ("x", "X"):
            self.start_select()
        elif k in ("c", "C"):
            self.clear_select()
        elif k in ("q", "Escape"):
            self._on_quit()
        else:
            return None
        return "break"

    # -- viewport size -------------------------------------------------------
    def _view_size(self) -> tuple[int, int]:
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if w <= 1 or h <= 1:
            w, h = VIEW_W, VIEW_H
        return max(64, w), max(64, h)

    def _max_den(self) -> int:
        sh, sw = self.frames[self.idx].disp.shape[:2]
        vw, vh = self._view_size()
        return max(1, math.ceil(max(sw / vw, sh / vh)))

    # -- navigation --------------------------------------------------------
    def zoom_in(self) -> None:
        if self.den > 1:
            self.den -= 1
        elif self.num < MAX_MAG:
            self.num += 1
        self.render()

    def zoom_out(self) -> None:
        if self.num > 1:
            self.num -= 1
        elif self.den < self._max_den():
            self.den += 1
        self.render()

    def fit(self) -> None:
        sh, sw = self.frames[self.idx].disp.shape[:2]
        self.num, self.den = 1, self._max_den()
        self.cx, self.cy = sw / 2.0, sh / 2.0
        self.render()

    def one_to_one(self) -> None:
        self.num, self.den = 1, 1
        self.render()

    def pan(self, dxs: int, dys: int) -> None:
        self.cx += dxs * max(1, self.vis_w // 6)
        self.cy += dys * max(1, self.vis_h // 6)
        self.render()

    def next_image(self) -> None:
        self.idx = min(self.idx + 1, N_IMAGES - 1)
        self.render()

    def prev_image(self) -> None:
        self.idx = max(self.idx - 1, 0)
        self.render()

    # -- star selection --------------------------------------------------
    def start_select(self) -> None:
        if self._capturing:
            return
        self._select_mode = True
        self._lasso = []
        self.view.configure(cursor="crosshair")
        self._set_hint("draw a closed curve around ONE star; release to finish")

    def clear_select(self) -> None:
        self._poly = None
        self._region = None
        self.metrics = None
        self._set_hint("selection cleared")
        self.render()

    def _on_press(self, e) -> None:
        if not self._select_mode:
            return
        self._lasso = [(e.x, e.y)]

    def _on_drag(self, e) -> None:
        if not self._select_mode or not self._lasso:
            return
        lx, ly = self._lasso[-1]
        if abs(e.x - lx) + abs(e.y - ly) >= 2:
            self._lasso.append((e.x, e.y))
            self._draw_lasso_live()

    def _on_release(self, e) -> None:
        if not self._select_mode:
            return
        self._select_mode = False
        self.view.configure(cursor="")
        if len(self._lasso) < 3:
            self._set_hint("selection too small -- try again (Select star)")
            self._lasso = []
            self.render()
            return
        h, w = self.frames[0].gray.shape
        pts = np.array([self._view_to_src(x, y) for x, y in self._lasso],
                       dtype=np.float64)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        self._poly = pts.astype(np.float32)
        self._lasso = []
        self._build_region()
        self._recompute_metrics()
        self.render()

    def _view_to_src(self, vx: float, vy: float) -> tuple[float, float]:
        x0, y0, num, den, ox, oy = self.vt
        return (x0 + (vx - ox) * den / num, y0 + (vy - oy) * den / num)

    def _poly_to_view(self, poly: np.ndarray) -> np.ndarray:
        x0, y0, num, den, ox, oy = self.vt
        vx = ox + (poly[:, 0] - x0) / den * num
        vy = oy + (poly[:, 1] - y0) / den * num
        return np.stack([vx, vy], axis=1).round().astype(np.int32)

    def _draw_lasso_live(self) -> None:
        if self._base_frame is None:
            return
        f = self._base_frame.copy()
        cv2.polylines(f, [np.array(self._lasso, np.int32)], False,
                      (0, 255, 255), 1, cv2.LINE_AA)
        self._photo = _to_photo(f)
        self.view.configure(image=self._photo)

    # -- metric -----------------------------------------------------------
    def _build_region(self) -> None:
        if self._poly is None:
            self._region = None
            return
        h, w = self.frames[0].gray.shape
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(self._poly).astype(np.int32)], 1)
        if int(m.sum()) == 0:
            self._region = None
            return
        gap = cv2.dilate(m, _disk(RING_GAP_PX))
        outer = cv2.dilate(m, _disk(RING_GAP_PX + RING_WIDTH_PX))
        ring = (outer > 0) & (gap == 0)
        self._region = (m.astype(bool), ring)

    def _metric(self, gray: np.ndarray) -> dict:
        mask, ring = self._region
        vals = gray[mask].astype(np.float64)
        over = bool(np.any(vals >= OVERBURN_FRAC * self.frames[0].white))
        bg = (float(np.median(gray[ring])) if ring.any()
              else float(np.percentile(gray, 25)))
        v = np.clip(vals - bg, 0.0, None)
        s = float(v.sum())
        if s <= 0.0:
            return {"sump2": float("nan"), "eff": float("nan"), "over": over}
        p = v / s
        sump2 = float(np.dot(p, p))
        return {"sump2": sump2, "eff": 1.0 / sump2, "over": over}

    def _recompute_metrics(self) -> None:
        if self._region is None:
            self.metrics = None
            return
        self.metrics = [self._metric(f.gray) for f in self.frames]
        finite = [(m["sump2"], i) for i, m in enumerate(self.metrics)
                  if math.isfinite(m["sump2"])]
        best = max(finite)[1] if finite else None
        cells = []
        for i, m in enumerate(self.metrics):
            s = m["sump2"]
            cells.append(f"{i}:nan" if not math.isfinite(s)
                         else f"{i}:{s * 1e3:.2f}")
        tail = f"   best={best}" if best is not None else ""
        print("[metric] sum p^2 x1e3: " + "  ".join(cells) + tail)

    # -- re-capture -----------------------------------------------------
    def recapture(self) -> None:
        if self._capturing:
            return
        step = simpledialog.askinteger(
            "Re-capture", "New focus step (units, e.g. 8):",
            parent=self.root, minvalue=1,
            initialvalue=abs(self.step) or 8)
        if not step:
            return
        center = self.meta[self.idx]["tracked_pos"]
        self._capturing = True
        self._set_hint(f"capturing 9 frames around tracked_pos {center:+d}, "
                       f"step {step} ... (images hidden)")
        try:
            frames, meta, raws, names = capture_bracket(
                self.cam, center, step, self.fmt,
                status_cb=self._capture_status)
        except Exception as e:
            self._capturing = False
            messagebox.showerror("Re-capture failed", str(e))
            self.render()
            return
        self._capturing = False
        self.bracket_seq += 1
        save_bracket(raws, names, meta, self.out_root,
                     f"b{self.bracket_seq:02d}_c{center:+d}_s{step}")
        self.frames, self.meta, self.step = frames, meta, step
        self.idx = CENTRE_INDEX
        self._build_region()      # same polygon, new frames
        self._recompute_metrics()
        self._set_hint("re-capture done")
        self.render()

    def _capture_status(self, n: int, total: int, msg: str) -> None:
        vw, vh = self._view_size()
        f = np.zeros((vh, vw, 3), np.uint8)
        cv2.putText(f, f"CAPTURING  {n + 1}/{total}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(f, msg, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(f, "images hidden until capture completes", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                    cv2.LINE_AA)
        self._photo = _to_photo(f)
        self.view.configure(image=self._photo)
        self.info.configure(text=f"CAPTURING {n + 1}/{total} -- {msg}")
        self.root.update()

    # -- render ---------------------------------------------------------
    def render(self) -> None:
        if self._capturing:
            return
        vw, vh = self._view_size()
        frame, self.cx, self.cy, self.vis_w, self.vis_h, self.vt = \
            render_viewport(self.frames[self.idx].disp, self.num, self.den,
                            self.cx, self.cy, vw, vh)
        self._base_frame = frame.copy()

        if self._poly is not None:
            pts = self._poly_to_view(self._poly)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

        mets = self.metrics[self.idx] if self.metrics else None
        top = 0
        if mets and mets["over"]:
            cv2.rectangle(frame, (0, 0), (vw, 46), (0, 0, 255), -1)
            cv2.putText(frame, "OVERBURN  pixels near saturation in selection",
                        (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            top = 46

        m = self.meta[self.idx]
        zoom = f"{self.num}:1" if self.den == 1 else f"1:{self.den}"
        line = (f"[{self.idx}/{N_IMAGES - 1}] tracked_pos={m['tracked_pos']:+d} "
                f"offset={m['offset']:+d} "
                f"abs_focus={self.abs_pos + m['tracked_pos']:+d} "
                f"step={self.step} zoom={zoom}")
        cv2.putText(frame, line, (8, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        self._photo = _to_photo(frame)
        self.view.configure(image=self._photo)
        self._update_bars()

    def _update_bars(self) -> None:
        m = self.meta[self.idx]
        self.info.configure(text=(
            f"[{self.idx}/{N_IMAGES - 1}] tracked_pos={m['tracked_pos']:+d}  "
            f"offset={m['offset']:+d}  "
            f"abs_focus={self.abs_pos + m['tracked_pos']:+d}  "
            f"step={self.step}  fmt={self.fmt}   "
            f"(r re-capture, x select star, c clear, q quit)"))

        if not self.metrics:
            self.metricbar.configure(
                text="Σp²: select a star (x) to measure sharpness")
            return
        finite = [(mm["sump2"], i) for i, mm in enumerate(self.metrics)
                  if math.isfinite(mm["sump2"])]
        best = max(finite)[1] if finite else None
        cells = []
        for i, mm in enumerate(self.metrics):
            s = mm["sump2"]
            t = "nan" if not math.isfinite(s) else f"{s * 1e3:.2f}"
            if i == best:
                t += "*"
            if i == self.idx:
                t = f"[{t}]"
            cells.append(f"{i}:{t}")
        cur = self.metrics[self.idx]
        eff = "nan" if not math.isfinite(cur["eff"]) else f"{cur['eff']:.0f}"
        over = [str(i) for i, mm in enumerate(self.metrics) if mm["over"]]
        txt = ("Σp²×1e3  " + "  ".join(cells)
               + f"    best={best if best is not None else '-'}"
               + f"   eff.px[{self.idx}]={eff}   (higher Σp² = sharper)")
        if over:
            txt += f"   OVERBURN:{','.join(over)}"
        self.metricbar.configure(text=txt)

    def _set_hint(self, msg: str) -> None:
        self.hint.configure(text=msg)

    # -- exit ---------------------------------------------------------
    def _on_quit(self) -> None:
        if self._capturing:
            return
        target = self.meta[self.idx]["tracked_pos"]
        delta = target - self.cam.pos
        print("=" * 64)
        print(f"[close] chosen frame [{self.idx}]  tracked_pos={target:+d}  "
              f"abs_focus={self.abs_pos + target:+d}")
        print(f"[close] driving lens from tracked_pos {self.cam.pos:+d} "
              f"by {delta:+d} ...")
        ok = True
        try:
            self.cam.move_to(target, reversing=True, strict=True)
        except Exception as e:
            ok = False
            print(f"[close] FAILED to reach the chosen focus: {e}")
            print(f"[close] lens now at tracked_pos {self.cam.pos:+d} "
                  f"(position uncertain). To finish manually, run:")
            print(f"[close]     camera/focus_hunt_move_focus.py "
                  f"{target - self.cam.pos:+d}")
        if ok:
            print(f"[close] SUCCESS: lens at tracked_pos {self.cam.pos:+d}  "
                  f"(abs_focus {self.abs_pos + self.cam.pos:+d})")
        print("=" * 64)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Converge on best star focus with a re-capturable 9-frame "
                    "bracket and a per-region sharpness metric.")
    p.add_argument("abs_pos", type=int,
                   help="absolute focus position of the CURRENT focus (label "
                        "only; does not move the lens)")
    p.add_argument("first_step", type=int,
                   help="offset (from current focus) of the first bracket's "
                        "centre, i.e. of image index 4")
    p.add_argument("step", type=int,
                   help="focus-unit spacing between consecutive images of the "
                        "first bracket")
    p.add_argument("--format", choices=["nef", "jpeg"], default="nef",
                   help="capture format (default: nef -- linear raw, best for "
                        "the sharpness metric; needs rawpy). 'jpeg' shoots "
                        "'JPEG Fine' and shows it as-is.")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    fmt = args.format
    if fmt == "nef" and rawpy is None:
        print("[fatal] --format nef needs rawpy, not importable in this python "
              "(pip install rawpy), or use --format jpeg")
        return 1

    quality = "NEF (Raw)" if fmt == "nef" else "JPEG Fine"
    out_root = os.path.join(OUT_DIR, time.strftime("session_%Y%m%d_%H%M%S"))
    print(f"[info] abs_pos={args.abs_pos:+d} first_step={args.first_step:+d} "
          f"step={args.step} format={fmt}")

    cam = FocusCamera()
    opened = False
    try:
        cam.open()
        opened = True
        cur = cam.get_config_value("imagequality")
        print(f"[info] camera image quality: {cur}")
        if cur != quality:
            if cam.set_config_guarded("imagequality", quality):
                print(f"[info] set image quality to '{quality}'")
            else:
                print(f"[warn] could not set image quality to '{quality}'; "
                      f"frames may not decode")
        cam.start_liveview()

        center = args.first_step
        print(f"[info] first bracket: centre tracked_pos={center:+d} "
              f"step={args.step}")
        frames, meta, raws, names = capture_bracket(
            cam, center, args.step, fmt,
            status_cb=lambda n, t, msg: print(f"[capture] {n + 1}/{t} {msg}"))
        save_bracket(raws, names, meta, out_root,
                     f"b00_c{center:+d}_s{args.step}")

        print("[info] opening viewer")
        Viewer(cam, frames, meta, args.abs_pos, args.step, fmt, out_root).run()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    finally:
        cam.stop_liveview()
        final_line = set_final_image_quality(cam) if opened else None
        cam.close()
        if final_line is not None:
            print()
            print(final_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
