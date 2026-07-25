#!/usr/bin/env python3
"""Reveal how the (filtered) Sun drifts in the frame over time.

This does NOT change focus/exposure/any CAPTURE parameter and does NOT save any
image. It streams LIVE-VIEW preview frames off the camera in a tight loop and
keeps two of them:

* ``first`` -- the very first preview frame obtained after start-up (frozen).
* ``last``  -- the most recent preview frame (continuously overwritten).

The view flips every ``FIRST_S`` / ``LAST_S`` seconds between the frozen
``first`` (labelled "first", green) and the current ``last`` (labelled "last",
red). Because ``first`` never changes, blinking it against ``last`` makes any
positional drift of the Sun over the elapsed interval pop out by eye. On each
flip the elapsed time between the two displayed frames' capture instants is
printed to stdout.

Live view is inherently required to pull preview frames, so we enable it the
same known-good way ``focus_hunt.py`` does (controlmode -> PC, viewfinder on)
and restore ``viewfinder=0`` on exit. No focus/exposure parameter is touched.

Window (the tkinter console from the old ``liveview_focus.py``)
--------------------------------------------------------------
* Left: the preview in a scrollable canvas. Frames are pulled back-to-back, as
  fast as the USB link allows (scheduled with ``after(1)`` so the UI stays
  responsive). Pan by dragging with the mouse, with the scrollbars, or with the
  arrow keys / hjkl. Only the visible part of the magnified frame is ever
  rasterised, so 4x costs the same per frame as 1x.
* Right: the controls.
    - View scale: 1x / 2x / 4x nearest-neighbour magnification of the preview on
      screen (display only -- does not touch the camera). Upscaling is pure
      pixel replication: no antialiasing / interpolation, so individual preview
      pixels are shown as solid blocks.
    - Zoom: a popup 0 / 2048 / 1024 / 512 driving ``liveviewzoomarea``
      (empirically: 0 = whole chip, smaller = more magnified; the number appears
      to be how many chip pixels map to the preview's longer side).
    - AF area: the arrow pad driving ``changeafarea`` ("XxY", chip pixels,
      0x0 = top-left). Each press steps by ``zoom // 4`` px; the position is
      tracked in software and clamped to the sensor. When Zoom is 0 the step is
      0, so the arrows are inert (nothing to pan at the full-chip view).
    - Re-arm: re-freeze "first" from the next frame.

``liveviewzoomarea`` and ``changeafarea`` are live-view-only settings -- no
focus, no exposure -- and the zoom is restored to 0 on exit. Changing either
re-arms the frozen ``first`` frame, since a reframed preview cannot be blinked
against the old reference.

Keys: + / = / i and - / o scale, 1 or f back to 1:1, arrows or hjkl pan,
r re-arm, q / Esc quit.

Run with the python that has gphoto2 + Pillow bound (the machine wired to the
camera):
    python3 camera/drift_reveal.py
"""
from __future__ import annotations

import io
import sys
import time
import traceback
import tkinter as tk

from PIL import Image, ImageTk
import gphoto2 as gp

# ---- tunable parameters ---------------------------------------------------
FIRST_S = 1.0           # seconds the frozen "first" frame is shown per cycle
LAST_S = 3.0            # seconds the current "last" frame is shown per cycle
MAX_RETRY = 3           # camera-reset retries for the (idempotent) preview grab
BUSY_WAIT_S = 1.0       # pause before retrying a -110 (CAMERA_BUSY) config op
COLOR_FIRST = "#00ff00"     # green label for the frozen "first" frame
COLOR_LAST = "#ff4040"      # red label for the live "last" frame

# ---- view / zoom ----------------------------------------------------------
CANVAS_W, CANVAS_H = 1024, 768      # initial canvas size (px); freely resizable
VIEW_SCALES = ["1x", "2x", "4x"]    # nearest-neighbour display magnification
PAN_FRAC = 0.25                     # keyboard pan step, as a fraction of the canvas
ZOOM_CHOICES = ["0", "2048", "1024", "512"]   # liveviewzoomarea values
CHIP_W = 6048                       # Nikon Z6 sensor width  (px) -- AF-area clamp
CHIP_H = 4024                       # Nikon Z6 sensor height (px) -- AF-area clamp

WINDOW = "Drift Reveal (filtered Sun)"


def print_error(what: str, exc: Exception) -> None:
    """Print an exception with everything available about it."""
    print(f"\n[error] {what}: {type(exc).__name__}: {exc}")
    print(f"        args = {exc.args!r}")
    traceback.print_exc()
    print()


def _is_busy(exc: Exception) -> bool:
    """True iff `exc` is a GP_ERROR_CAMERA_BUSY (-110) assertion from a config op."""
    return isinstance(exc, AssertionError) and bool(exc.args) and exc.args[-1] == -110


# =====================================================================
#  Camera wrapper (self-healing, single USB handle)
# =====================================================================
class DriftCamera:
    """gphoto2 wrapper with self-healing (re-init on error) preview capture."""

    def __init__(self) -> None:
        self.camera = None
        self.context = gp.gp_context_new()

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

    def close(self) -> None:
        try:
            if self.camera is not None:
                gp.gp_camera_exit(self.camera, self.context)
        except Exception:
            pass
        self.camera = None

    # -- config helpers ----------------------------------------------------
    def _set_config(self, name: str, value) -> None:
        err, config = gp.gp_camera_get_config(self.camera, self.context)
        assert err == gp.GP_OK, ("get_config", err)
        err, child = gp.gp_widget_get_child_by_name(config, name)
        assert err == gp.GP_OK, ("get_child", name, err)
        gp.gp_widget_set_value(child, value)
        err = gp.gp_camera_set_config(self.camera, config, self.context)
        assert err == gp.GP_OK, ("set_config", name, err)

    def _set_config_guarded(self, name: str, value) -> bool:
        """Set a config leaf, waiting out -110 (BUSY) and resetting on anything else.

        Idempotent config writes are safe to re-issue, so on any other failure we
        reset the handle and try again up to MAX_RETRY times.
        """
        if self.camera is None:
            return False        # never opened (or already closed) -- nothing to do
        for attempt in range(MAX_RETRY + 1):
            try:
                self._set_config(name, value)
                return True
            except Exception as e:
                if _is_busy(e):
                    print(f"[warn] set {name}={value!r} busy (-110); "
                          f"waiting {BUSY_WAIT_S}s and retrying")
                    time.sleep(BUSY_WAIT_S)
                    continue
                print(f"[warn] set {name}={value} failed "
                      f"(try {attempt + 1}/{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        print(f"[warn] giving up on set {name}={value}; continuing")
        return False

    def start_liveview(self) -> None:
        # Hand PC control to the camera and raise the mirror into live view so
        # preview frames can be pulled. This does NOT alter focus/exposure.
        self._set_config_guarded("controlmode", "0")
        self._set_config_guarded("viewfinder", 1)
        time.sleep(0.8)

    def stop_liveview(self) -> None:
        self._set_config_guarded("viewfinder", 0)

    # -- live-view zoom / pan ----------------------------------------------
    def set_zoom(self, value: str) -> bool:
        """Set ``liveviewzoomarea`` (live-view magnification, camera side)."""
        return self._set_config_guarded("liveviewzoomarea", value)

    def set_af_area(self, x: int, y: int) -> bool:
        """Set ``changeafarea`` to the chip pixel "XxY" (0x0 = top-left)."""
        return self._set_config_guarded("changeafarea", f"{x}x{y}")

    # -- preview capture ---------------------------------------------------
    def _capture_preview_once(self) -> Image.Image:
        err, cam_file = gp.gp_file_new()
        assert err == gp.GP_OK, ("file_new", err)
        err = gp.gp_camera_capture_preview(self.camera, cam_file, self.context)
        assert err == gp.GP_OK, ("capture_preview", err)
        err, data = gp.gp_file_get_data_and_size(cam_file)
        assert err == gp.GP_OK, ("get_data", err)
        raw = memoryview(data).tobytes()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def capture_preview(self) -> Image.Image:
        """Grab one live-view preview frame, self-healing on error."""
        for attempt in range(MAX_RETRY + 1):
            try:
                return self._capture_preview_once()
            except Exception as e:
                if _is_busy(e):
                    time.sleep(BUSY_WAIT_S)
                    continue
                print(f"[warn] preview failed (try {attempt + 1}/"
                      f"{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        raise RuntimeError("preview capture failed after retries")


# =====================================================================
#  GUI
# =====================================================================
class DriftApp:
    def __init__(self, cam: DriftCamera) -> None:
        self.cam = cam
        self.af_x = 0
        self.af_y = 0

        self.first = None       # frozen reference frame (PIL RGB)
        self.t_first = 0.0
        self.last = None        # most recent frame
        self.t_last = 0.0
        self.show_first = True
        self.last_toggle = time.monotonic()

        self.root = tk.Tk()
        self.root.title(WINDOW)

        # -- left: scrollable image --------------------------------------
        left = tk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                bg="black", highlightthickness=0)
        hbar = tk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._img_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self._label_id = self.canvas.create_text(
            0, 0, anchor=tk.NW, text="", fill=COLOR_FIRST,
            font=("TkDefaultFont", 20, "bold"))
        self._photo = None

        # drag-to-pan, the same gesture as the scrollbars
        self.canvas.bind("<ButtonPress-1>",
                         lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>",
                         lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        # -- right: controls ---------------------------------------------
        right = tk.Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="ns")

        self.dims_var = tk.StringVar(value="frame: (none yet)")
        tk.Label(right, textvariable=self.dims_var,
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.elapsed_var = tk.StringVar(value="first -> last: --")
        tk.Label(right, textvariable=self.elapsed_var,
                 font=("TkDefaultFont", 10)).pack(anchor="w", pady=(0, 12))

        # View scale -----------------------------------------------------
        vbox = tk.LabelFrame(right, text="View scale (display only)",
                             padx=6, pady=6)
        vbox.pack(anchor="w", fill="x", pady=6)
        self.scale_var = tk.StringVar(value=VIEW_SCALES[0])
        tk.OptionMenu(vbox, self.scale_var, *VIEW_SCALES).pack(anchor="w")

        # Zoom -----------------------------------------------------------
        zbox = tk.LabelFrame(right, text="Zoom (liveviewzoomarea)", padx=6, pady=6)
        zbox.pack(anchor="w", fill="x", pady=6)
        self.zoom_var = tk.StringVar(value=ZOOM_CHOICES[0])
        tk.OptionMenu(zbox, self.zoom_var, *ZOOM_CHOICES,
                      command=self.on_zoom).pack(anchor="w")

        # AF area --------------------------------------------------------
        abox = tk.LabelFrame(right, text="AF area (changeafarea)", padx=6, pady=6)
        abox.pack(anchor="w", fill="x", pady=6)
        self.af_var = tk.StringVar()
        tk.Label(abox, textvariable=self.af_var,
                 font=("TkDefaultFont", 11)).pack(anchor="w", pady=(0, 6))
        pad = tk.Frame(abox)
        pad.pack(anchor="w")
        #      col0   col1   col2
        # row0        up
        # row1 left          right
        # row2        down
        tk.Button(pad, text="↑", width=3,
                  command=lambda: self.on_af(0, -1)).grid(row=0, column=1)
        tk.Button(pad, text="←", width=3,
                  command=lambda: self.on_af(-1, 0)).grid(row=1, column=0)
        tk.Button(pad, text="→", width=3,
                  command=lambda: self.on_af(1, 0)).grid(row=1, column=2)
        tk.Button(pad, text="↓", width=3,
                  command=lambda: self.on_af(0, 1)).grid(row=2, column=1)
        self._update_af_label()

        # Reference ------------------------------------------------------
        tk.Button(right, text="Re-arm \"first\"",
                  command=self.on_rearm).pack(anchor="w", fill="x", pady=6)

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self._bind_keys()

    # -- keys --------------------------------------------------------------
    def _bind_keys(self) -> None:
        b = self.root.bind
        b("<Escape>", lambda e: self.root.destroy())
        b("q", lambda e: self.root.destroy())
        for k in ("plus", "equal", "i"):
            b(f"<{k}>", lambda e: self.on_view_scale(+1))
        for k in ("minus", "o"):
            b(f"<{k}>", lambda e: self.on_view_scale(-1))
        for k in ("1", "f"):
            b(f"<{k}>", lambda e: self.on_view_reset())
        for k, (dx, dy) in (("Left", (-1, 0)), ("Right", (1, 0)),
                            ("Up", (0, -1)), ("Down", (0, 1)),
                            ("h", (-1, 0)), ("l", (1, 0)),
                            ("k", (0, -1)), ("j", (0, 1))):
            b(f"<{k}>", lambda e, dx=dx, dy=dy: self.on_pan(dx, dy))
        b("r", lambda e: self.on_rearm())

    # -- label helpers ------------------------------------------------------
    def _update_af_label(self) -> None:
        self.af_var.set(f"{self.af_x}x{self.af_y}")

    @property
    def scale(self) -> int:
        return int(self.scale_var.get().rstrip("x"))

    # -- control handlers ---------------------------------------------------
    def on_view_scale(self, step: int) -> None:
        idx = VIEW_SCALES.index(self.scale_var.get())
        self.scale_var.set(VIEW_SCALES[min(len(VIEW_SCALES) - 1,
                                           max(0, idx + step))])

    def on_view_reset(self) -> None:
        self.scale_var.set(VIEW_SCALES[0])
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def on_pan(self, dx: int, dy: int) -> None:
        # Work in scrollbar fractions: (hi - lo) is exactly one viewport, so a
        # press moves PAN_FRAC of the visible area whatever the scale is.
        if dx:
            lo, hi = self.canvas.xview()
            self.canvas.xview_moveto(lo + dx * (hi - lo) * PAN_FRAC)
        if dy:
            lo, hi = self.canvas.yview()
            self.canvas.yview_moveto(lo + dy * (hi - lo) * PAN_FRAC)

    def on_zoom(self, value: str) -> None:
        print(f"[zoom] liveviewzoomarea = {value!r}")
        self.cam.set_zoom(value)
        self.rearm("camera zoom changed")

    def on_af(self, dx: int, dy: int) -> None:
        step = int(self.zoom_var.get()) // 4
        if step == 0:
            print("[afarea] zoom is 0 (whole chip); arrows inert")
            return
        self.af_x = min(CHIP_W, max(0, self.af_x + dx * step))
        self.af_y = min(CHIP_H, max(0, self.af_y + dy * step))
        value = f"{self.af_x}x{self.af_y}"
        print(f"[afarea] changeafarea = {value!r} (step {step})")
        self.cam.set_af_area(self.af_x, self.af_y)
        self._update_af_label()
        self.rearm("AF area moved")

    def on_rearm(self) -> None:
        self.rearm("requested")

    def rearm(self, why: str) -> None:
        """Drop the frozen reference; the next frame becomes the new "first"."""
        print(f"[ref] re-arming the frozen 'first' frame ({why})")
        self.first = None
        # Show the new reference before blinking it against anything.
        self.show_first = True
        self.last_toggle = time.monotonic()
        self.elapsed_var.set("first -> last: --")

    # -- drawing ------------------------------------------------------------
    def _draw(self, frame: Image.Image, text: str, color: str) -> None:
        """Blit the visible part of `frame` (magnified) into the canvas.

        Only the region the canvas actually shows is cropped and magnified, so
        the per-frame cost is set by the window size, not by the view scale.
        """
        s = self.scale
        full_w, full_h = frame.width * s, frame.height * s
        self.canvas.config(scrollregion=(0, 0, full_w, full_h))

        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        ox = max(0, min(int(self.canvas.canvasx(0)), max(0, full_w - view_w)))
        oy = max(0, min(int(self.canvas.canvasy(0)), max(0, full_h - view_h)))

        # Source rect covering the visible canvas, snapped to source pixels.
        sx0, sy0 = ox // s, oy // s
        sx1 = min(frame.width, -(-(ox + view_w) // s))      # ceil division
        sy1 = min(frame.height, -(-(oy + view_h) // s))
        crop = frame.crop((sx0, sy0, sx1, sy1))
        if s != 1:
            # NEAREST = pure pixel replication: no antialiasing, no
            # interpolation -- one source pixel becomes an s*s block.
            crop = crop.resize(((sx1 - sx0) * s, (sy1 - sy0) * s), Image.NEAREST)

        self._photo = ImageTk.PhotoImage(crop)
        self.canvas.coords(self._img_id, sx0 * s, sy0 * s)
        self.canvas.itemconfig(self._img_id, image=self._photo)
        # Keep the label pinned to the visible top-left corner while scrolling.
        self.canvas.coords(self._label_id, ox + 8, oy + 8)
        self.canvas.itemconfig(self._label_id, text=text, fill=color)
        self.canvas.tag_raise(self._label_id)

        self.dims_var.set(f"frame: {frame.width} x {frame.height} px "
                          f"(shown {s}:1)")

    # -- capture loop -------------------------------------------------------
    def _tick(self) -> None:
        try:
            img = self.cam.capture_preview()
            now = time.monotonic()
            if self.first is None:
                self.first, self.t_first = img, now
            self.last, self.t_last = img, now

            dwell = FIRST_S if self.show_first else LAST_S
            if now - self.last_toggle >= dwell:
                self.show_first = not self.show_first
                self.last_toggle = now
                elapsed = self.t_last - self.t_first
                print(f"elapsed first->last: {elapsed:.1f} s")
                self.elapsed_var.set(f"first -> last: {elapsed:.1f} s")

            if self.show_first:
                self._draw(self.first, "first", COLOR_FIRST)
            else:
                self._draw(self.last, "last", COLOR_LAST)
        except Exception as e:
            print_error("capture_preview", e)
        # back-to-back: reschedule ASAP while still servicing UI events
        self.root.after(1, self._tick)

    def run(self) -> None:
        self.root.after(200, self._tick)
        self.root.mainloop()


def main() -> int:
    print("Blinking the frozen 'first' preview against the live 'last' one.")
    print("View scale: 1x/2x/4x popup or +/-, 1 = 1:1. Pan: drag, scrollbars,")
    print("arrows or hjkl. Zoom: popup. AF area: arrow pad. r re-arms 'first'.")
    print("q, Esc or closing the window quits.\n")

    cam = DriftCamera()
    try:
        cam.open()
        cam.start_liveview()
        DriftApp(cam).run()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    except Exception as e:
        print_error("fatal", e)
    finally:
        cam.set_zoom(ZOOM_CHOICES[0])   # back to the whole chip
        cam.stop_liveview()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
