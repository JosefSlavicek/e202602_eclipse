#!/usr/bin/env python3
"""Interactive live-view + focus / zoom / AF-area control for the Nikon Z6.

This combines two earlier probes into one operator console:

* ``liveview_probe.py`` -- how to pull the live-view preview stream and push
  ``liveviewsize`` / ``liveviewzoomarea`` / ``changeafarea`` to the camera.
* ``focus_hunt.py``     -- how to drive the focus motor with ``manualfocusdrive``
  (a RELATIVE action; there is no absolute focus read-back) plus the
  self-healing camera wrapper (reset + retry on error, wait-and-retry on the
  GP_ERROR_CAMERA_BUSY / -110 that live-view focus moves occasionally return).

None of the focus-HUNT logic (metrics, hill-climb, full-res shots) is used
here: focus is driven purely by the six on-screen buttons, one
``manualfocusdrive`` command per click.

Window
------
* Left: the live-view preview shown ALWAYS 1:1 (one preview pixel = one screen
  pixel) in a scrollable canvas. Frames are pulled back-to-back, as fast as the
  USB link allows (scheduled with ``after(1)`` so the UI stays responsive).
* Right: the controls.
    - Focus:  <<<  <<  <  >  >>  >>>   (∓400 / ∓40 / ∓4 focus-drive units;
              left = negative direction, right = positive). A label shows the
              running tracked position (starts at 0 at launch -- there is no
              absolute read-back, so this is our own software counter).
    - Zoom:   a popup 0 / 2048 / 1024 / 512 driving ``liveviewzoomarea``
              (empirically: 0 = whole chip, smaller = more magnified; the
              number appears to be how many chip pixels map to the preview's
              longer side). Applied to the camera immediately on selection.
    - AF area: ← ↑ → ↓ buttons driving ``changeafarea`` ("XxY", chip pixels,
              0x0 = top-left). Each press steps by ``zoom // 4`` px in that
              direction; the position is tracked in software and clamped to the
              sensor. When Zoom is 0 the step is 0, so the arrows are inert
              (nothing to pan at the full-chip view).
    - View scale: 1x / 2x / 4x nearest-neighbour magnification of the preview
              on screen (display only -- does not touch the camera). Upscaling
              is pure pixel replication: no antialiasing / interpolation, so
              individual preview pixels are shown as solid blocks.

Every camera error is printed in full and the loop continues.

Run with the python that has gphoto2 + Pillow bound (the machine wired to the
camera):
    python3 camera/liveview_focus.py
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
LIVEVIEW_SIZE = "XGA"        # fixed preview resolution for this tool
ZOOM_CHOICES = ["0", "2048", "1024", "512"]   # liveviewzoomarea values
FOCUS_STEPS = [
    ("<<<", -400), ("<<", -40), ("<", -4),
    (">", 4), (">>", 40), (">>>", 400),
]
VIEW_SCALES = ["1x", "2x", "4x"]   # nearest-neighbour display magnification
CHIP_W = 6048                # Nikon Z6 sensor width  (px) -- AF-area clamp
CHIP_H = 4024                # Nikon Z6 sensor height (px) -- AF-area clamp
MAX_RETRY = 3                # camera-reset retries for idempotent ops
BUSY_WAIT_S = 1.0            # pause before retrying a -110 (CAMERA_BUSY) op

WINDOW = "Live View + Focus (Nikon Z6)"


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
class LiveCamera:
    def __init__(self) -> None:
        self.camera = None
        self.context = gp.gp_context_new()
        self.pos = 0        # software focus position, relative to launch

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

    # -- raw config helpers ------------------------------------------------
    def _set_config(self, name: str, value) -> None:
        err, config = gp.gp_camera_get_config(self.camera, self.context)
        assert err == gp.GP_OK, ("get_config", err)
        err, child = gp.gp_widget_get_child_by_name(config, name)
        assert err == gp.GP_OK, ("get_child", name, err)
        gp.gp_widget_set_value(child, value)
        err = gp.gp_camera_set_config(self.camera, config, self.context)
        assert err == gp.GP_OK, ("set_config", name, err)

    def set_config_guarded(self, name: str, value) -> bool:
        """Set a config leaf with -110 wait-retry and reset-on-error retry.

        Returns True on success. Idempotent config writes are safe to re-issue,
        so on any failure we reset the handle and try again up to MAX_RETRY.
        """
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
                print(f"[warn] set {name}={value!r} failed "
                      f"(try {attempt + 1}/{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        print(f"[warn] giving up on set {name}={value!r}; continuing")
        return False

    # -- live view ---------------------------------------------------------
    def start_liveview(self) -> None:
        self.set_config_guarded("controlmode", "0")
        self.set_config_guarded("viewfinder", 1)
        self.set_config_guarded("liveviewsize", LIVEVIEW_SIZE)
        time.sleep(0.8)

    def stop_liveview(self) -> None:
        try:
            self.set_config_guarded("viewfinder", 0)
        except Exception as e:
            print_error("stop_liveview", e)

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
        """Grab one preview frame, self-healing on error (reset + retry)."""
        for attempt in range(MAX_RETRY + 1):
            try:
                return self._capture_preview_once()
            except Exception as e:
                if _is_busy(e):
                    time.sleep(BUSY_WAIT_S)
                    continue
                print(f"[warn] capture_preview failed "
                      f"(try {attempt + 1}/{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        raise RuntimeError("preview capture failed after retries")

    # -- focus -------------------------------------------------------------
    def drive_focus(self, delta: int) -> None:
        """Move focus RELATIVELY by `delta` units (one manualfocusdrive command).

        On -110 (CAMERA_BUSY) the command was fully rejected -- the motor
        provably did not move -- so we wait and retry once. On any other error
        we assume an unknown partial move, reset the handle and DO NOT re-issue
        (avoids an unknown double-move). `pos` is only advanced on success.
        """
        if delta == 0:
            return
        try:
            self._set_config("manualfocusdrive", delta)
            self.pos += delta
        except Exception as e:
            if _is_busy(e):
                print(f"[warn] focus drive {delta:+d} busy (-110); "
                      f"waiting {BUSY_WAIT_S}s and retrying")
                time.sleep(BUSY_WAIT_S)
                try:
                    self._set_config("manualfocusdrive", delta)
                    self.pos += delta
                except Exception as e2:
                    print(f"[warn] focus drive {delta:+d} retry failed: {e2} "
                          f"(assuming no move; resetting)")
                    self.reset()
            else:
                print(f"[warn] focus drive {delta:+d} failed: {e} "
                      f"(assuming no move; resetting)")
                self.reset()


# =====================================================================
#  GUI
# =====================================================================
class LiveApp:
    def __init__(self, cam: LiveCamera) -> None:
        self.cam = cam
        self.af_x = 0
        self.af_y = 0

        self.root = tk.Tk()
        self.root.title(WINDOW)

        # -- left: scrollable 1:1 image ----------------------------------
        left = tk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(left, width=1024, height=768, bg="black",
                                highlightthickness=0)
        hbar = tk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._img_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self._photo = None

        # -- right: controls ---------------------------------------------
        right = tk.Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="ns")

        self.dims_var = tk.StringVar(value="frame: (none yet)")
        tk.Label(right, textvariable=self.dims_var,
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w",
                                                          pady=(0, 12))

        # Focus ----------------------------------------------------------
        fbox = tk.LabelFrame(right, text="Focus", padx=6, pady=6)
        fbox.pack(anchor="w", fill="x", pady=6)
        self.focus_var = tk.StringVar()
        tk.Label(fbox, textvariable=self.focus_var,
                 font=("TkDefaultFont", 11)).pack(anchor="w", pady=(0, 6))
        row = tk.Frame(fbox)
        row.pack(anchor="w")
        for text, delta in FOCUS_STEPS:
            tk.Button(row, text=text, width=3,
                      command=lambda d=delta: self.on_focus(d)).pack(side="left")
        self._update_focus_label()

        # Zoom -----------------------------------------------------------
        zbox = tk.LabelFrame(right, text="Zoom (liveviewzoomarea)", padx=6, pady=6)
        zbox.pack(anchor="w", fill="x", pady=6)
        self.zoom_var = tk.StringVar(value=ZOOM_CHOICES[0])
        tk.OptionMenu(zbox, self.zoom_var, *ZOOM_CHOICES,
                      command=self.on_zoom).pack(anchor="w")

        # View scale -----------------------------------------------------
        vbox = tk.LabelFrame(right, text="View scale (display only)",
                             padx=6, pady=6)
        vbox.pack(anchor="w", fill="x", pady=6)
        self.scale_var = tk.StringVar(value=VIEW_SCALES[0])
        tk.OptionMenu(vbox, self.scale_var, *VIEW_SCALES).pack(anchor="w")

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

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # -- label helpers ----------------------------------------------------
    def _update_focus_label(self) -> None:
        self.focus_var.set(f"position: {self.cam.pos:+d}")

    def _update_af_label(self) -> None:
        self.af_var.set(f"{self.af_x}x{self.af_y}")

    # -- control handlers -------------------------------------------------
    def on_focus(self, delta: int) -> None:
        print(f"[focus] drive {delta:+d}")
        self.cam.drive_focus(delta)
        self._update_focus_label()

    def on_zoom(self, value: str) -> None:
        print(f"[zoom] liveviewzoomarea = {value!r}")
        self.cam.set_config_guarded("liveviewzoomarea", value)

    def on_af(self, dx: int, dy: int) -> None:
        step = int(self.zoom_var.get()) // 4
        if step == 0:
            print("[afarea] zoom is 0 (whole chip); arrows inert")
            return
        self.af_x = min(CHIP_W, max(0, self.af_x + dx * step))
        self.af_y = min(CHIP_H, max(0, self.af_y + dy * step))
        value = f"{self.af_x}x{self.af_y}"
        print(f"[afarea] changeafarea = {value!r} (step {step})")
        self.cam.set_config_guarded("changeafarea", value)
        self._update_af_label()

    # -- capture loop -----------------------------------------------------
    def _tick(self) -> None:
        try:
            img = self.cam.capture_preview()
            w, h = img.size
            scale = int(self.scale_var.get().rstrip("x"))
            if scale != 1:
                # NEAREST = pure pixel replication: no antialiasing, no
                # interpolation -- one source pixel becomes a scale*scale block.
                img = img.resize((w * scale, h * scale), Image.NEAREST)
            self.dims_var.set(f"frame: {w} x {h} px (shown {scale}:1)")
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self._img_id, image=self._photo)
            self.canvas.config(scrollregion=(0, 0, w * scale, h * scale))
        except Exception as e:
            print_error("capture_preview", e)
        # back-to-back: reschedule ASAP while still servicing UI events
        self.root.after(1, self._tick)

    def run(self) -> None:
        self.root.after(200, self._tick)
        self.root.mainloop()


def main() -> int:
    print(f"Live view {LIVEVIEW_SIZE}, refreshed as fast as the link allows.")
    print("Focus: 6 buttons (+-100/10/1). Zoom: popup. AF area: arrow pad.")
    print("Esc or close the window to quit.\n")

    cam = LiveCamera()
    try:
        cam.open()
        cam.start_liveview()
        LiveApp(cam).run()
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    except Exception as e:
        print_error("fatal", e)
    finally:
        cam.stop_liveview()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
