#!/usr/bin/env python3
"""Reveal how the (filtered) Sun drifts in the frame over time.

This does NOT change focus/exposure/any capture parameter and does NOT save any
image. It streams LIVE-VIEW preview frames off the camera in a tight loop and
keeps two of them:

* ``first`` -- the very first preview frame obtained after start-up (frozen).
* ``last``  -- the most recent preview frame (continuously overwritten).

A single window flips every ``TOGGLE_S`` seconds between the frozen ``first``
(labelled "first") and the current ``last`` (labelled "last"). Because ``first``
never changes, blinking it against ``last`` makes any positional drift of the
Sun over the elapsed interval pop out by eye. On each flip the elapsed time
between the two displayed frames' capture instants is printed to stdout.

Live view is inherently required to pull preview frames, so we enable it the
same known-good way ``focus_hunt.py`` does (controlmode -> PC, viewfinder on)
and restore ``viewfinder=0`` on exit. No focus/exposure parameter is touched.

Run with the python that has gphoto2 + OpenCV bound (the machine wired to the
camera):
    python3 camera/drift_reveal.py
"""
from __future__ import annotations

import time
import sys

import numpy as np
import cv2
import gphoto2 as gp

# ---- tunable parameters ---------------------------------------------------
TOGGLE_S = 1.0          # seconds each of first/last is shown before flipping
MAX_RETRY = 3           # camera-reset retries for the (idempotent) preview grab
FONT_SCALE = 0.28       # ~1/3 of focus_hunt's 0.8 -- small corner label
LABEL_ORG = (8, 22)     # top-left anchor of the label text

WINDOW = "Drift Reveal (filtered Sun)"


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
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

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

    def start_liveview(self) -> None:
        # Hand PC control to the camera and raise the mirror into live view so
        # preview frames can be pulled. This does NOT alter focus/exposure.
        self._set_config_guarded("controlmode", "0")
        self._set_config_guarded("viewfinder", 1)
        time.sleep(0.8)

    def stop_liveview(self) -> None:
        self._set_config_guarded("viewfinder", 0)

    # -- preview capture ---------------------------------------------------
    def _capture_preview_once(self) -> np.ndarray:
        err, cam_file = gp.gp_file_new()
        assert err == gp.GP_OK, ("file_new", err)
        err = gp.gp_camera_capture_preview(self.camera, cam_file, self.context)
        assert err == gp.GP_OK, ("capture_preview", err)
        err, data = gp.gp_file_get_data_and_size(cam_file)
        assert err == gp.GP_OK, ("get_data", err)
        arr = np.frombuffer(memoryview(data).tobytes(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("preview frame failed to decode")
        return img

    def capture_preview(self) -> np.ndarray:
        """Grab one live-view preview frame, self-healing on error."""
        for attempt in range(MAX_RETRY + 1):
            try:
                return self._capture_preview_once()
            except Exception as e:
                print(f"[warn] preview failed (try {attempt + 1}/"
                      f"{MAX_RETRY + 1}): {e}")
                if attempt < MAX_RETRY:
                    self.reset()
        raise RuntimeError("preview capture failed after retries")


def _draw_label(bgr: np.ndarray, text: str) -> np.ndarray:
    """Return a copy of `bgr` with a small label in the top-left corner."""
    vis = bgr.copy()
    cv2.putText(vis, text, LABEL_ORG, cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, text, LABEL_ORG, cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def run(cam: DriftCamera) -> None:
    first = None
    t_first = 0.0
    last = None
    t_last = 0.0

    show_first = True
    last_toggle = time.monotonic()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Streaming live-view preview; press q or Esc in the window to quit.")

    while True:
        img = cam.capture_preview()
        now = time.monotonic()
        if first is None:
            first, t_first = img, now
        last, t_last = img, now

        if now - last_toggle >= TOGGLE_S:
            show_first = not show_first
            last_toggle = now
            print(f"elapsed first->last: {t_last - t_first:.1f} s")

        frame = first if show_first else last
        label = "first" if show_first else "last"
        cv2.imshow(WINDOW, _draw_label(frame, label))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break


def main() -> int:
    cam = DriftCamera()
    try:
        cam.open()
        cam.start_liveview()
        run(cam)
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    finally:
        cam.stop_liveview()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
