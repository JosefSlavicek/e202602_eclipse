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
* The lens starts at INFINITY. We therefore probe in the NEGATIVE focus
  direction first (the same sign the original PoC used: manualfocusdrive -N).
* Nikon `manualfocusdrive` is a RANGE *action*: each set drives the focus motor
  RELATIVELY by the given (signed) amount. There is no way to read back an
  absolute focus position, so we track our own software position counter `pos`
  (units = focus-drive steps relative to the infinity start).
* Search = neighbour-probe hill-climb with step halving. Starting from a large
  step we keep stepping in the improving direction; when neither neighbour
  improves we halve the step and try again, down to step size 1, then return to
  the best position found. This reproduces the "hunt, halve, settle at step 1"
  behaviour and is self-correcting: because every decision is made on a freshly
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
import time
import sys

import numpy as np
import cv2
import gphoto2 as gp

# ---- tunable parameters ---------------------------------------------------
INITIAL_STEP = 256      # first (coarse) focus step; halved down to 1
MIN_STEP = 1            # finest step; hunt ends after refining at this size
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

WINDOW = "Focus Hunt (filtered Sun)"
PNG_DIR = "focus_hunt"  # only used if there is no display (headless fallback)


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


class FocusCamera:
    """gphoto2 camera wrapper with self-healing (re-init on error) operations."""

    def __init__(self) -> None:
        self.camera = None
        self.context = gp.gp_context_new()
        self.pos = 0                 # software focus position (rel. to infinity)
        self.headless = False        # set True if no display for cv2 windows
        self.saved_frames = 0        # headless overlay PNGs written
        self.saved_captures = 0      # raw camera files saved to cwd
        self.current_round = 0       # 0 = baseline; increments each step-halving
        self.metric = "custom"       # "custom" | "normalized" | "laplacian"

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
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

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

    def stop_liveview(self) -> None:
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

        In normal mode (strict=False) a failed chunk is NOT retried (the motor
        may have moved a partial, unknown amount): we reset the camera, assume
        the chunk did not move, drop it from the tracked position, and continue.
        The closed-loop hunt corrects any resulting small offset on the next
        measurement.

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
            except Exception as e:
                # -110 = GP_ERROR_CAMERA_BUSY: the command was fully rejected,
                # the motor provably did not move, so retrying is safe.
                is_busy = (isinstance(e, AssertionError) and e.args
                           and e.args[-1] == -110)
                if is_busy:
                    print(f"[warn] focus drive {move} busy (-110); "
                          f"waiting 1 s and retrying")
                    time.sleep(1.0)
                    try:
                        self._set_config("manualfocusdrive", move)
                        self.pos += move
                    except Exception as e2:
                        if strict:
                            raise RuntimeError(
                                f"final focus move chunk {move} failed after "
                                f"busy-retry: {e2}") from e2
                        print(f"[warn] focus drive {move} retry failed: {e2} "
                              f"(assuming no move; resetting)")
                        self.reset()
                else:
                    if strict:
                        raise RuntimeError(
                            f"final focus move chunk {move} failed: {e}") from e
                    print(f"[warn] focus drive {move} failed: {e} "
                          f"(assuming no move; resetting)")
                    self.reset()
            remaining -= abs(move)
            time.sleep(MOVE_PAUSE)

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
            else:
                value = focus_merit(img)
                if self.metric == "normalized":
                    value = self._normalize_merit(img, value)
            scores.append(value)
            last = img
            self._save_capture(raw, name, value)
        sharp = float(np.mean(scores))
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
        vis = bgr.copy()
        for i, line in enumerate(
            [label, f"pos={self.pos}  value={sharp:.3f}"]
        ):
            cv2.putText(vis, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 1, cv2.LINE_AA)
        if not self.headless:
            try:
                cv2.imshow(WINDOW, vis)
                cv2.waitKey(1)
                return
            except cv2.error as e:
                print(f"[warn] no display ({e}); falling back to saving PNGs "
                      f"into {PNG_DIR}/")
                self.headless = True
        # headless fallback
        import os
        os.makedirs(PNG_DIR, exist_ok=True)
        path = os.path.join(PNG_DIR, f"frame_{self.saved_frames:03d}.png")
        cv2.imwrite(path, vis)
        self.saved_frames += 1


def hunt(cam: FocusCamera) -> None:
    moves_used = 0

    # Baseline at the starting (infinity) position.
    best_pos = 0
    cam.current_round = 0
    best_sharp = cam.measure("start (infinity)")
    print(f"[baseline] pos={best_pos} value={best_sharp:.3f}")

    step = INITIAL_STEP
    pref = -1  # probe the NEGATIVE direction first (away from infinity)
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
    print(f"FINAL focus position = {best_pos} steps from infinity")
    print(f"FINAL value          = {final:.3f} (peak seen {best_sharp:.3f})")
    print("=" * 60)
    if not cam.headless:
        print("Press any key in the image window to exit...")
        try:
            cv2.waitKey(0)
        except cv2.error:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-hunt optimal focus of the filtered Sun via full-res shots.")
    parser.add_argument(
        "--metric", choices=["custom", "normalized", "laplacian"],
        default="custom",
        help=(
            "Sharpness metric used for focus decisions (default: custom). "
            "'custom': edge-cluster minimum over top-N Sobel gradient pixels "
            "(coherent-edge focus, noise-spike resistant). "
            "'normalized': same divided by the 400th brightest pixel value "
            "(compensates for exposure variation between frames). "
            "'laplacian': Laplacian variance -- standard optical autofocus metric, "
            "well-suited to solar/eclipse images with hard limb edges; "
            "insensitive to overall brightness changes."
        ))
    args = parser.parse_args()

    cam = FocusCamera()
    cam.metric = args.metric
    print(f"[info] sharpness metric: {args.metric}")
    try:
        cam.open()
        cam.start_liveview()
        hunt(cam)
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    finally:
        cam.stop_liveview()
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
