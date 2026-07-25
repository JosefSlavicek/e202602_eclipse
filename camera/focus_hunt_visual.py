#!/usr/bin/env python3
"""Capture a short focus-bracket sequence and inspect it in a zoomable viewer.

Usage
-----
    camera/focus_hunt_visual.py ABS_POS FIRST_STEP STEP
    e.g.  camera/focus_hunt_visual.py -1750 -20 4

Arguments (three integers)
--------------------------
* ABS_POS     -- the ABSOLUTE focus position of the camera's CURRENT focus
                 (e.g. the value focus_hunt printed as FINAL). It is only
                 remembered and used to LABEL the captured images; it does not
                 move the lens.
* FIRST_STEP  -- where the CENTRE of the bracket sits relative to the current
                 focus. Image index 4 (0-based, the middle of 9) is captured at
                 `current_focus + FIRST_STEP`.
* STEP        -- focus-unit spacing between consecutive images.

The 9 images therefore sit at focus offsets (relative to the current focus):
    offset(i) = FIRST_STEP + (i - 4) * STEP        for i in 0..8
and each is labelled with the absolute focus  ABS_POS + offset(i).

What it does
------------
1. Opens the camera, enters live view, and drives the focus to each offset in
   turn (moving monotonically to keep backlash consistent), shooting one
   full-resolution frame per position, downloading it to the PC, saving a copy
   under focus_visual/, and DELETING it from the camera card.
2. Returns the lens to the bracket centre (FIRST_STEP).
3. Opens a tkinter viewer to inspect the 9 frames:
     * zoom IN is INTEGER magnification with NO interpolation / antialiasing
       (each source pixel becomes an NxN block); zoom OUT decimates (every
       Nth pixel), also with no interpolation.
     * pan within a zoomed frame; switch between frames PRESERVING the current
       zoom and pan.
   Everything is driven by on-screen buttons and laptop-friendly keys.

Keyboard (also on-screen buttons)
---------------------------------
    + / = / i      zoom in            - / o          zoom out
    arrows / hjkl  pan                f              fit whole frame to window
    1              1:1 actual pixels  n / ] / space  next frame
    p / [ / ,      previous frame     q / Esc        quit

Run with the python that has gphoto2 + OpenCV bound (the camera machine).
"""
from __future__ import annotations

import argparse
import base64
import math
import os
import sys
import time

import numpy as np
import cv2

# Reuse the self-healing camera wrapper and timing from focus_hunt.py, which
# lives in this same directory. Importing it also sets OPENCV_LOG_LEVEL before
# cv2 loads (silencing libtiff chatter) and pulls in gphoto2.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_hunt import FocusCamera, SETTLE_S  # noqa: E402

try:
    import tkinter as tk
except Exception as e:  # pragma: no cover - GUI is required for this tool
    print(f"[fatal] tkinter is required for the viewer but is unavailable: {e}")
    sys.exit(1)

# ---- tunables -------------------------------------------------------------
N_IMAGES = 9            # frames per bracket
CENTRE_INDEX = 4        # 0-based index that lands on current_focus + FIRST_STEP
VIEW_W, VIEW_H = 1000, 720   # initial viewer window size (resizable)
MAX_MAG = 16            # deepest integer magnification (Nx)
CAP_SETTLE_S = 1.0      # let live view resume after a full-res shot before the
                        #  next focus move (avoids GP_ERROR_CAMERA_BUSY)
OUT_DIR = "focus_visual"  # where downloaded frames are saved on the PC


# ---------------------------------------------------------------------------
# Viewport transform (pure, unit-testable: no camera, no Tk)
# ---------------------------------------------------------------------------
def render_viewport(img: np.ndarray, num: int, den: int,
                    cx: float, cy: float, view_w: int, view_h: int):
    """Render a view_w x view_h BGR frame of `img` at integer zoom num/den.

    Zoom is a rational num/den with only one side > 1 at a time:
      * num > 1, den == 1  -> MAGNIFY: each source pixel becomes an num x num
        block (np.repeat -> pure nearest-neighbour, no interpolation).
      * num == 1, den > 1  -> SHRINK: keep every den-th source pixel
        (decimation -> also no interpolation).
      * num == 1, den == 1 -> 1:1.

    (cx, cy) is the source-image coordinate kept at the viewport centre; it is
    re-clamped so the crop stays inside the image and the clamped value is
    returned so the caller can store it. Also returns the source-pixel span
    (cw, ch) currently visible, which the caller uses to size pan steps.

    Returns (frame, new_cx, new_cy, cw, ch).
    """
    sh, sw = img.shape[:2]
    # source-pixel span needed to fill the viewport at this zoom
    cw = min(sw, max(1, math.ceil(view_w * den / num)))
    ch = min(sh, max(1, math.ceil(view_h * den / num)))
    # centre-anchored crop, clamped into the image
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

    # centre the (possibly smaller) rendered image on a black canvas
    frame = np.zeros((view_h, view_w, 3), np.uint8)
    dh, dw = disp.shape[:2]
    oy, ox = max(0, (view_h - dh) // 2), max(0, (view_w - dw) // 2)
    frame[oy:oy + dh, ox:ox + dw] = disp
    return frame, new_cx, new_cy, cw, ch


def _to_photo(frame_bgr: np.ndarray) -> "tk.PhotoImage":
    """Encode a BGR frame to a Tk PhotoImage (PNG+base64, no Pillow)."""
    ok, png = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def capture_sequence(abs_pos: int, first_step: int, step: int,
                     image_quality: str = "JPEG Fine"):
    """Shoot the 9-frame focus bracket; return (images, meta).

    images[i] is the decoded BGR frame; meta[i] carries index/offset/abs_focus.

    The camera is switched to `image_quality` (default 'JPEG Fine') for the
    captures and restored afterwards. This matters: a NEF/raw is TIFF-based and
    cv2.imdecode can only pull its small EMBEDDED PREVIEW, so the viewer would
    show a tiny, pixelated sun. A full-res JPEG/TIFF decodes as the real
    full-resolution image. Pass image_quality='' to leave the setting untouched.
    """
    offsets = [first_step + (i - CENTRE_INDEX) * step for i in range(N_IMAGES)]
    images: list[np.ndarray | None] = [None] * N_IMAGES
    meta: list[dict] = [None] * N_IMAGES  # type: ignore[list-item]
    os.makedirs(OUT_DIR, exist_ok=True)

    cam = FocusCamera()
    orig_quality = None
    quality_changed = False
    try:
        cam.open()
        orig_quality = cam.get_config_value("imagequality")
        print(f"[info] camera image quality: {orig_quality}")
        if image_quality and orig_quality != image_quality:
            if cam.set_config_guarded("imagequality", image_quality):
                quality_changed = True
                print(f"[info] set image quality to '{image_quality}' for "
                      f"full-resolution, decodable frames")
            else:
                print(f"[warn] could not set image quality to "
                      f"'{image_quality}'; frames may be low-res / undecodable")

        cam.start_liveview()

        # Capture in tracked-position order so the motor sweeps monotonically.
        order = sorted(range(N_IMAGES), key=lambda i: offsets[i])
        for n, i in enumerate(order):
            cam.move_to(offsets[i])
            time.sleep(SETTLE_S)
            img, raw, name = cam.capture_image()
            images[i] = img
            h, w = img.shape[:2]
            abs_focus = abs_pos + offsets[i]
            meta[i] = {"index": i, "offset": offsets[i], "abs_focus": abs_focus}

            _, ext = os.path.splitext(name)
            out = os.path.join(
                OUT_DIR, f"img{i}_abs{abs_focus:+d}_off{offsets[i]:+d}{ext}")
            try:
                with open(out, "wb") as f:
                    f.write(raw)
            except OSError as e:
                print(f"[warn] could not save {out}: {e}")
                out = "(not saved)"
            print(f"[capture] {n + 1}/{N_IMAGES} index={i} off={offsets[i]:+d} "
                  f"abs_focus={abs_focus:+d} pos={cam.pos:+d} res={w}x{h} "
                  f"-> {out}")
            time.sleep(CAP_SETTLE_S)

        # Return the lens to the initial focus position (tracked pos 0, where
        # the camera's focus was when the script started).
        cam.move_to(0)
        print(f"[info] returned lens to the initial focus position "
              f"(abs {abs_pos:+d}); tracked pos={cam.pos:+d}")
    finally:
        # Restore the original quality while the camera handle is still alive.
        if quality_changed and orig_quality:
            if cam.set_config_guarded("imagequality", orig_quality):
                print(f"[info] restored image quality to '{orig_quality}'")
            else:
                print(f"[warn] could NOT restore image quality -- set it back "
                      f"to '{orig_quality}' manually in the camera menu")
        cam.stop_liveview()
        cam.close()

    return images, meta


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class Viewer:
    """tkinter window: integer-zoom, pannable, frame-switching image inspector."""

    def __init__(self, images: list[np.ndarray], meta: list[dict],
                 abs_pos: int, step: int) -> None:
        self.images = images
        self.meta = meta
        self.abs_pos = abs_pos
        self.step = step
        self.idx = CENTRE_INDEX

        # zoom is num/den (only one side > 1 at a time); (cx, cy) is the source
        # coordinate held at the viewport centre; vis_w/h = visible source span.
        self.num, self.den = 1, 1
        h, w = images[0].shape[:2]
        self.cx, self.cy = w / 2.0, h / 2.0
        self.vis_w, self.vis_h = w, h
        self._photo = None  # keep a ref so Tk does not GC the shown image

        self._build_ui()
        # Fit once the window has a real size.
        self.root.update_idletasks()
        self.fit()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Focus bracket viewer")
        self.root.geometry(f"{VIEW_W}x{VIEW_H}")

        bar = tk.Frame(self.root)
        bar.pack(side="top", fill="x")

        def btn(parent, text, cmd):
            b = tk.Button(parent, text=text, command=cmd, takefocus=0)
            b.pack(side="left", padx=1, pady=2)
            return b

        btn(bar, "◀ Prev (p)", self.prev_image)
        btn(bar, "Next (n) ▶", self.next_image)
        tk.Label(bar, text="  ").pack(side="left")
        btn(bar, "Zoom − (-)", self.zoom_out)
        btn(bar, "Zoom + (+)", self.zoom_in)
        btn(bar, "Fit (f)", self.fit)
        btn(bar, "1:1 (1)", self.one_to_one)
        tk.Label(bar, text="  ").pack(side="left")
        btn(bar, "◀", lambda: self.pan(-1, 0))
        btn(bar, "▲", lambda: self.pan(0, -1))
        btn(bar, "▼", lambda: self.pan(0, 1))
        btn(bar, "▶", lambda: self.pan(1, 0))
        btn(bar, "Quit (q)", self.root.destroy)

        self.info = tk.Label(self.root, anchor="w", font=("TkFixedFont", 10))
        self.info.pack(side="top", fill="x")

        self.view = tk.Label(self.root, bg="black")
        self.view.pack(side="top", fill="both", expand=True)
        self.view.bind("<Configure>", lambda _e: self.render())

        self.root.bind("<Key>", self._on_key)

    def _on_key(self, event):
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
        elif k in ("q", "Escape"):
            self.root.destroy()
        else:
            return None
        return "break"  # swallow (e.g. stop arrows moving button focus)

    # -- viewport size -----------------------------------------------------
    def _view_size(self) -> tuple[int, int]:
        w, h = self.view.winfo_width(), self.view.winfo_height()
        if w <= 1 or h <= 1:  # not realised yet
            w, h = VIEW_W, VIEW_H
        return max(64, w), max(64, h)

    def _max_den(self) -> int:
        """Deepest zoom-out (den) that still just fits the whole frame."""
        sh, sw = self.images[self.idx].shape[:2]
        vw, vh = self._view_size()
        return max(1, math.ceil(max(sw / vw, sh / vh)))

    # -- actions -----------------------------------------------------------
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
        sh, sw = self.images[self.idx].shape[:2]
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
        # clamp at the last frame (do NOT wrap around to the first)
        self.idx = min(self.idx + 1, N_IMAGES - 1)  # zoom & pan preserved
        self.render()

    def prev_image(self) -> None:
        # clamp at the first frame (do NOT wrap around to the last)
        self.idx = max(self.idx - 1, 0)
        self.render()

    # -- render ------------------------------------------------------------
    def render(self) -> None:
        vw, vh = self._view_size()
        frame, self.cx, self.cy, self.vis_w, self.vis_h = render_viewport(
            self.images[self.idx], self.num, self.den, self.cx, self.cy, vw, vh)

        m = self.meta[self.idx]
        zoom = f"{self.num}:1" if self.den == 1 else f"1:{self.den}"
        info = (f"[{self.idx}/{N_IMAGES - 1}]  abs_focus={m['abs_focus']:+d}  "
                f"offset={m['offset']:+d}  step={self.step}  zoom={zoom}")
        cv2.putText(frame, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        self._photo = _to_photo(frame)
        self.view.configure(image=self._photo)
        self.info.configure(text=info + "   (+/- zoom, arrows/hjkl pan, "
                                        "n/p switch, f fit, 1 actual, q quit)")

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture a 9-frame focus bracket and inspect it zoomed.")
    p.add_argument("abs_pos", type=int,
                   help="absolute focus position of the CURRENT focus (label "
                        "only; does not move the lens)")
    p.add_argument("first_step", type=int,
                   help="offset (from current focus) of the bracket CENTRE, "
                        "i.e. of image index 4")
    p.add_argument("step", type=int,
                   help="focus-unit spacing between consecutive images")
    p.add_argument("--image-quality", default="JPEG Fine",
                   help="camera image quality to shoot for the viewer (default "
                        "'JPEG Fine' = full-res and cv2-decodable). A NEF/raw "
                        "only yields a small embedded preview to the viewer "
                        "(tiny, pixelated sun). Restored on exit. Pass an empty "
                        "string to leave the camera's setting unchanged.")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(f"[info] abs_pos={args.abs_pos:+d} first_step={args.first_step:+d} "
          f"step={args.step}")
    try:
        images, meta = capture_sequence(args.abs_pos, args.first_step,
                                        args.step, args.image_quality)
    except RuntimeError as e:
        print(f"[fatal] capture failed: {e}")
        print("        (if frames won't decode, set the camera to JPEG or TIFF "
              "-- a RAW/.NEF cannot be decoded for the viewer)")
        return 1
    print("[info] all frames captured; opening viewer "
          "(+/- zoom, arrows/hjkl pan, n/p switch, f fit, 1 actual, q quit)")
    Viewer(images, meta, args.abs_pos, args.step).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
