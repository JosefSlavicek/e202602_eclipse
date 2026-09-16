#!/usr/bin/env python3
"""Shoot one NEF frame and look at its RGB histograms.

A stripped-down relative of ``focus_hunt_stars.py``. It keeps that script's
zoomable / pannable image viewer and its freehand-lasso region tool, and throws
away everything else: no sharpness metric, no over-burn bar, no focus drive, no
image-quality changes. The camera is used exactly as it is currently set -- the
script only opens it, takes a full-resolution shot, downloads it and closes.

What it does
------------
  * Shows the captured frame in an integer-zoom viewer (same controls as
    focus_hunt_stars: zoom, fit, 1:1, pan by button or arrow keys).

  * ``Stretch`` slider -- an asinh DISPLAY stretch so a faint deep-sky target
    (M31) is actually visible on screen. It changes only what you see; every
    histogram is computed on the raw linear data and is unaffected.

  * ``Select area`` -- draw a freehand closed curve. The histograms for "the
    selected area" are then computed inside that polygon. The polygon is saved
    to a file in the current folder and reloaded on the next run.

  * ``Re-capture`` -- take another shot with whatever settings are on the camera
    right now (change ISO / shutter on the camera body first if you like). The
    lasso and the zoom / pan are kept.

  * Histogram panel (right) -- six curves: R, G, B for the whole image, and
    R, G, B for the selected area (blank until something is selected). Linear
    16-bit data, 256 bins, 0..65535. ``log Y`` toggle.

  * Clipping readout -- fraction of pixels at or above 0.99 * white level, per
    channel, for the whole image and the selection.

NEF only
--------
The frame is decoded with rawpy using the same settings as focus_hunt_stars
(linear gamma, no auto-bright, unit white balance, raw colour, 16-bit), so the
histogram values are proportional to photons and the per-channel clipping point
is the real one. If the camera is not shooting NEF the decode fails and the
script exits.

Run with the python that has gphoto2 + rawpy + OpenCV + matplotlib.

Usage
-----
    camera/get_histogram.py [--lasso-file PATH] [--debug-layout]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_hunt import FocusCamera  # noqa: E402

try:
    import tkinter as tk
    from tkinter import messagebox
except Exception as e:  # pragma: no cover - GUI is required for this tool
    print(f"[fatal] tkinter is required for the viewer but is unavailable: {e}")
    sys.exit(1)

try:
    import rawpy
except Exception:  # pragma: no cover
    rawpy = None

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception as e:  # pragma: no cover
    print(f"[fatal] matplotlib is required for the histograms but is "
          f"unavailable: {e}")
    sys.exit(1)

# ---- tunables ---------------------------------------------------------------
VIEW_W, VIEW_H = 1500, 850   # initial window size (resizable)
MAX_MAG = 16                 # deepest integer magnification (Nx)
NEF_WHITE = 65535.0          # rawpy output_bps=16 white point
HIST_BINS = 256
CLIP_FRAC = 0.99             # pixel >= this * white -> counted as clipped
STRETCH_MAX = 3000           # slider range for the asinh display stretch
STRETCH_DEFAULT = 400
DEFAULT_LASSO_FILE = "get_histogram_lasso.json"
RESIZE_DELAY_MS = 60         # wait this long after the last resize to redraw

# channel -> (matplotlib colour, label)
CHANNELS = [(0, "#d62728", "R"), (1, "#2ca02c", "G"), (2, "#1f77b4", "B")]


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def decode_nef(raw: bytes, name: str) -> np.ndarray:
    """Decode one downloaded NEF into a linear uint16 HxWx3 RGB array.

    Same rawpy settings as focus_hunt_stars: linear, no auto-bright, unit white
    balance, raw colour space, 16-bit output.
    """
    if rawpy is None:
        raise RuntimeError(
            "NEF decode needs rawpy, which is not importable in this python "
            "(pip install rawpy)")
    try:
        with rawpy.imread(io.BytesIO(raw)) as r:
            rgb16 = r.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16,
                user_wb=[1.0, 1.0, 1.0, 1.0], user_flip=0,
                output_color=rawpy.ColorSpace.raw)
    except Exception as e:
        raise RuntimeError(
            f"{name}: rawpy could not decode this frame -- is the camera "
            f"really shooting NEF? ({e})")
    return np.ascontiguousarray(rgb16)


# ---------------------------------------------------------------------------
# Viewport transform (verbatim from focus_hunt_stars.render_viewport)
# ---------------------------------------------------------------------------
def render_viewport(img: np.ndarray, num: int, den: int, cx: float, cy: float,
                    view_w: int, view_h: int):
    """Render a view_w x view_h BGR frame of `img` at integer zoom num/den.

    Returns (frame, new_cx, new_cy, cw, ch, vt) where
    vt = (x0, y0, num, den, ox, oy) maps a source pixel (sx, sy) to a screen
    pixel:  vx = ox + (sx - x0) / den * num  (and same for y).
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


def asinh_stretch(y01: np.ndarray, amount: float) -> np.ndarray:
    """Display-only asinh stretch of data already scaled to [0, 1]."""
    if amount <= 0:
        return y01
    return np.arcsinh(y01 * amount) / math.asinh(amount)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class Viewer:
    """Integer-zoom pannable inspector with a lasso and RGB histograms."""

    def __init__(self, cam: FocusCamera, rgb16: np.ndarray,
                 lasso_file: str, debug_layout: bool = False) -> None:
        self.cam = cam
        self.rgb16 = rgb16
        self.lasso_file = lasso_file

        self.num, self.den = 1, 1
        h, w = rgb16.shape[:2]
        self.cx, self.cy = w / 2.0, h / 2.0
        self.vis_w, self.vis_h = w, h
        self.disp = None                 # uint8 HxWx3 BGR shown by the viewport
        self._photo = None
        self._base_frame = None          # rendered frame before overlays
        self.vt = (0, 0, 1, 1, 0, 0)

        self._poly = None                # (K, 2) float32 source-pixel polygon
        self._mask = None                # bool HxW inside the polygon
        self._select_mode = False
        self._lasso: list[tuple[int, int]] = []
        self._capturing = False
        self._last_view_size = None      # last <Configure> size of the view
        self._resize_job = None          # pending after() id for a resize
        self.debug_layout = debug_layout
        self._n_render = 0

        self._build_ui()
        self.root.update_idletasks()
        self._load_lasso()
        self.apply_display()
        self.one_to_one()
        self.update_histograms()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Histogram")
        self.root.geometry(f"{VIEW_W}x{VIEW_H}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        bar = tk.Frame(self.root)
        bar.pack(side="top", fill="x")

        def btn(text, cmd):
            b = tk.Button(bar, text=text, command=cmd, takefocus=0)
            b.pack(side="left", padx=1, pady=2)
            return b

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
        btn("Select area (x)", self.start_select)
        btn("Clear sel (c)", self.clear_select)
        tk.Label(bar, text="  stretch").pack(side="left")
        self.stretch = tk.Scale(bar, from_=0, to=STRETCH_MAX, orient="horizontal",
                                length=220, showvalue=True, takefocus=0,
                                command=self._on_stretch)
        self.stretch.set(STRETCH_DEFAULT)
        self.stretch.pack(side="left")
        self.logy = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="log Y", variable=self.logy, takefocus=0,
                       command=self.update_histograms).pack(side="left", padx=6)
        btn("Quit (q)", self._on_quit)

        self.info = tk.Label(self.root, anchor="w", font=("TkFixedFont", 10))
        self.info.pack(side="top", fill="x")
        self.clipbar = tk.Label(self.root, anchor="w", font=("TkFixedFont", 10))
        self.clipbar.pack(side="top", fill="x")
        self.hint = tk.Label(self.root, anchor="w", fg="#666",
                             font=("TkFixedFont", 9))
        self.hint.pack(side="top", fill="x")

        body = tk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # The histogram canvas is packed BEFORE the image view, so pack gives
        # it its width first and the view only gets what is left. Otherwise a
        # view that asks for a few px more than it has (image + border) keeps
        # taking width from the histogram panel, which fires <Configure>,
        # which re-renders a bigger image, and so on until the panel is gone.
        self.fig = Figure(figsize=(4.4, 7.0), dpi=100)
        self.ax_full = self.fig.add_subplot(211)
        self.ax_sel = self.fig.add_subplot(212)
        self.fig.subplots_adjust(left=0.16, right=0.97, top=0.94, bottom=0.08,
                                 hspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().configure(width=460)
        self.canvas.get_tk_widget().pack(side="right", fill="y")

        # No border / padding, so the rendered image asks for exactly the
        # size the label already has.
        self.view = tk.Label(body, bg="black", borderwidth=0,
                             highlightthickness=0, padx=0, pady=0)
        self.view.pack(side="left", fill="both", expand=True)
        self.view.bind("<Configure>", self._on_view_configure)
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

    # -- viewport size ----------------------------------------------------
    def _on_view_configure(self, e) -> None:
        """Re-render after the view really changed size, at most once per
        burst of resize events."""
        size = (e.width, e.height)
        if self.debug_layout:
            print(f"[layout] <Configure> view={e.width}x{e.height} "
                  f"hist_panel={self.canvas.get_tk_widget().winfo_width()}x"
                  f"{self.canvas.get_tk_widget().winfo_height()}")
        if size == self._last_view_size:
            return
        self._last_view_size = size
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(RESIZE_DELAY_MS, self._resize_render)

    def _resize_render(self) -> None:
        self._resize_job = None
        self.render()

    def _view_size(self) -> tuple[int, int]:
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if w <= 1 or h <= 1:
            w, h = VIEW_W - 480, VIEW_H - 120
        return max(64, w), max(64, h)

    def _max_den(self) -> int:
        sh, sw = self.disp.shape[:2]
        vw, vh = self._view_size()
        return max(1, math.ceil(max(sw / vw, sh / vh)))

    # -- navigation -----------------------------------------------------
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
        sh, sw = self.disp.shape[:2]
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

    # -- display --------------------------------------------------------
    def apply_display(self) -> None:
        """(Re)build `self.disp` from the linear data + the current stretch."""
        amount = float(self.stretch.get())
        y = self.rgb16.astype(np.float32) / NEF_WHITE
        y = asinh_stretch(np.clip(y, 0.0, 1.0), amount)
        rgb8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        self.disp = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)

    def _on_stretch(self, _v) -> None:
        if self._capturing:
            return
        self.apply_display()
        self.render()

    # -- area selection ------------------------------------------------
    def start_select(self) -> None:
        if self._capturing:
            return
        self._select_mode = True
        self._lasso = []
        self.view.configure(cursor="crosshair")
        self._set_hint("draw a closed curve around the area; release to finish")

    def clear_select(self) -> None:
        self._poly = None
        self._mask = None
        self._save_lasso()
        self._set_hint("selection cleared")
        self.update_histograms()
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
            self._set_hint("selection too small -- try again (Select area)")
            self._lasso = []
            self.render()
            return
        h, w = self.rgb16.shape[:2]
        pts = np.array([self._view_to_src(x, y) for x, y in self._lasso],
                       dtype=np.float64)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        self._poly = pts.astype(np.float32)
        self._lasso = []
        self._build_mask()
        self._save_lasso()
        self.update_histograms()
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

    def _build_mask(self) -> None:
        if self._poly is None:
            self._mask = None
            return
        h, w = self.rgb16.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(self._poly).astype(np.int32)], 1)
        self._mask = m.astype(bool) if int(m.sum()) else None

    # -- lasso persistence -------------------------------------------
    def _load_lasso(self) -> None:
        if not os.path.exists(self.lasso_file):
            return
        try:
            with open(self.lasso_file) as fh:
                data = json.load(fh)
            pts = np.array(data["polygon"], dtype=np.float32)
        except Exception as e:
            print(f"[warn] could not read {self.lasso_file}: {e}")
            return
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            print(f"[warn] {self.lasso_file}: not a usable polygon; ignoring")
            return
        h, w = self.rgb16.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        self._poly = pts
        self._build_mask()
        print(f"[info] loaded selection ({len(pts)} pts) from {self.lasso_file}")

    def _save_lasso(self) -> None:
        try:
            if self._poly is None:
                if os.path.exists(self.lasso_file):
                    os.remove(self.lasso_file)
                return
            h, w = self.rgb16.shape[:2]
            with open(self.lasso_file, "w") as fh:
                json.dump({"image_size": [w, h],
                           "polygon": self._poly.tolist()}, fh, indent=1)
        except OSError as e:
            print(f"[warn] could not write {self.lasso_file}: {e}")

    # -- histograms --------------------------------------------------
    def _draw_hist(self, ax, sel: bool) -> None:
        ax.clear()
        log = self.logy.get()
        title = "selected area" if sel else "whole image"
        if sel and self._mask is None:
            ax.text(0.5, 0.5, "no selection", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_title(title, fontsize=9)
            ax.set_xlim(0, NEF_WHITE)
            return
        for c, colour, label in CHANNELS:
            plane = self.rgb16[..., c]
            vals = plane[self._mask] if sel else plane.ravel()
            hist, edges = np.histogram(vals, bins=HIST_BINS,
                                       range=(0, NEF_WHITE))
            ax.plot(edges[:-1], np.maximum(hist, 0.5) if log else hist,
                    color=colour, lw=0.9, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, NEF_WHITE)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    def update_histograms(self) -> None:
        if self._capturing:
            return
        self._draw_hist(self.ax_full, sel=False)
        self._draw_hist(self.ax_sel, sel=True)
        self.canvas.draw_idle()
        self._update_clipbar()

    def _clip_fractions(self, sel: bool) -> str:
        thr = CLIP_FRAC * NEF_WHITE
        out = []
        for c, _colour, label in CHANNELS:
            plane = self.rgb16[..., c]
            vals = plane[self._mask] if sel else plane.ravel()
            frac = float(np.count_nonzero(vals >= thr)) / max(1, vals.size)
            out.append(f"{label}={frac * 100:.3f}%")
        return "  ".join(out)

    def _update_clipbar(self) -> None:
        whole = self._clip_fractions(sel=False)
        if self._mask is not None:
            sel = self._clip_fractions(sel=True)
            npx = int(self._mask.sum())
            self.clipbar.configure(
                text=f"clipped (>= {CLIP_FRAC:g}*white)   whole: {whole}     "
                     f"sel ({npx} px): {sel}")
        else:
            self.clipbar.configure(
                text=f"clipped (>= {CLIP_FRAC:g}*white)   whole: {whole}     "
                     f"sel: -")

    # -- re-capture -------------------------------------------------
    def recapture(self) -> None:
        if self._capturing:
            return
        self._capturing = True
        self._show_capturing()
        try:
            raw, name = self.cam.capture_raw()
            print(f"[capture] {name} bytes={len(raw)}")
            rgb16 = decode_nef(raw, name)
        except Exception as e:
            self._capturing = False
            messagebox.showerror("Re-capture failed", str(e))
            self.render()
            return
        self._capturing = False
        self.rgb16 = rgb16
        h, w = rgb16.shape[:2]
        # keep the polygon; clamp it to the new frame and rebuild the mask
        if self._poly is not None:
            self._poly[:, 0] = np.clip(self._poly[:, 0], 0, w - 1)
            self._poly[:, 1] = np.clip(self._poly[:, 1], 0, h - 1)
            self._build_mask()
        self.apply_display()
        self._set_hint("re-capture done")
        self.render()
        self.update_histograms()

    def _show_capturing(self) -> None:
        vw, vh = self._view_size()
        f = np.zeros((vh, vw, 3), np.uint8)
        cv2.putText(f, "CAPTURING ...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(f, "shooting one frame with the camera's current settings",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180),
                    1, cv2.LINE_AA)
        self._photo = _to_photo(f)
        self.view.configure(image=self._photo)
        self.info.configure(text="CAPTURING ...")
        self.root.update()

    # -- render ----------------------------------------------------
    def render(self) -> None:
        if self._capturing:
            return
        vw, vh = self._view_size()
        self._n_render += 1
        if self.debug_layout:
            hw = self.canvas.get_tk_widget()
            print(f"[layout] render #{self._n_render}: image={vw}x{vh} "
                  f"view={self.view.winfo_width()}x{self.view.winfo_height()} "
                  f"hist_panel={hw.winfo_width()}x{hw.winfo_height()} "
                  f"window={self.root.winfo_width()}x"
                  f"{self.root.winfo_height()}")
        frame, self.cx, self.cy, self.vis_w, self.vis_h, self.vt = \
            render_viewport(self.disp, self.num, self.den,
                            self.cx, self.cy, vw, vh)
        self._base_frame = frame.copy()

        if self._poly is not None:
            pts = self._poly_to_view(self._poly)
            if len(pts) >= 2:
                cv2.polylines(frame, [pts], True, (0, 255, 0), 1, cv2.LINE_AA)

        zoom = f"{self.num}:1" if self.den == 1 else f"1:{self.den}"
        h, w = self.rgb16.shape[:2]
        line = f"{w}x{h}  zoom={zoom}  stretch={int(self.stretch.get())}"
        cv2.putText(frame, line, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        self._photo = _to_photo(frame)
        self.view.configure(image=self._photo)
        self.info.configure(text=(
            f"{w}x{h}  zoom={zoom}   "
            f"(r re-capture, x select area, c clear, arrows pan, +/- zoom, "
            f"f fit, 1 one-to-one, q quit)"))

    def _set_hint(self, msg: str) -> None:
        self.hint.configure(text=msg)

    # -- exit ----------------------------------------------------
    def _on_quit(self) -> None:
        if self._capturing:
            return
        self._save_lasso()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shoot one NEF frame and inspect its RGB histograms "
                    "(whole image + a freehand-selected area).")
    p.add_argument("--lasso-file", default=DEFAULT_LASSO_FILE,
                   help=f"where the selection polygon is saved / loaded "
                        f"(default: ./{DEFAULT_LASSO_FILE} in the current "
                        f"folder)")
    p.add_argument("--debug-layout", action="store_true",
                   help="print widget sizes on every resize event and "
                        "every image render (to diagnose redraw loops)")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    if rawpy is None:
        print("[fatal] this tool needs rawpy (pip install rawpy)")
        return 1

    cam = FocusCamera()
    try:
        cam.open()
        cur = cam.get_config_value("imagequality")
        print(f"[info] camera image quality (left as-is): {cur}")
        print("[info] taking one frame with the camera's current settings")
        raw, name = cam.capture_raw()
        print(f"[capture] {name} bytes={len(raw)}")
        rgb16 = decode_nef(raw, name)
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
        cam.close()
        return 1
    except Exception as e:
        print(f"[fatal] {e}")
        cam.close()
        return 1

    try:
        print("[info] opening viewer")
        Viewer(cam, rgb16, args.lasso_file,
               debug_layout=args.debug_layout).run()
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
