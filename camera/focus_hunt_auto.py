#!/usr/bin/env python
"""Star-based focus scan for the Nikon Z6 + 400mm rig, with a live plot.

    camera/focus_hunt_auto.py WIDTH
    camera/focus_hunt_auto.py WIDTH --fake-data    # test with the lens covered

Shoots frames at random focus positions in [start - WIDTH, start + WIDTH]
(uniform, multiples of 4 -- the motor does not move below 4 units), scores each
frame and plots the result.  It runs until the plot window is closed; focus is
then left wherever the last shot put it.  Read the best position off the plot
(hover the mouse to get a vertical guide line) and drive there yourself.

Two quantities are plotted against the focus offset from the start position:
  * blue  -- identified stars (star_metric.score_frame "score"): highest at focus
  * red   -- typical star size, FWHM in px (star_metric.star_fwhm): lowest at
             focus

Assumptions (promised by the operator, NOT checked) are those of star_metric:
M31 somewhere in the frame, tracking mount, lens near focus, ISO / shutter as
set on the camera.  The deep catalogue must exist (star_metric.py
--build-catalog).

How it runs
-----------
  * a background thread owns the camera: move -> settle -> capture (download
    + delete from the card) -> queue -> live-view settle -> next.  The first
    frame is shot at the start position and blind-solved in this thread; it
    fixes the sky transform and the scored star set for the whole run.
  * WORKERS processes decode and score frames from the queue, so shooting
    never waits for computing (unless the queue fills up).
  * the main thread runs the plot and redraws it once per arriving point.

Positions are visited in random order, so they are approached from both
directions: any backlash in the focus drive shows up as extra scatter.

--fake-data
-----------
For testing with the lens covered.  Everything runs as normal -- frames are
shot, downloaded and decoded -- but the decoded frame is thrown away and a
synthetic star field is scored in its place (the same one focus_speed_test.py
uses: full Z6 size, star density as near M31, scored against a matching
synthetic catalogue, so no plate solve is needed).  Its blur grows linearly
from FWHM FAKE_FWHM_BEST px at a fake best focus of about +WIDTH/3 (so the
plot shows whether the peak lands where it should, not just at 0) to
FAKE_FWHM_EDGE px at the far end of the window.  The plot title says FAKE DATA.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time

import numpy as np

import find_it
import star_metric as sm

WORKERS = 4          # decode + score processes; one frame costs ~1.2 s of CPU
QUEUE_MAX = 8        # frames waiting to be scored before the camera waits
SOLVE_TRIES = 3      # frames to try for the first blind solve before giving up
POINT_PX = 3         # plotted point diameter
STARS_COLOR = "tab:blue"
FWHM_COLOR = "tab:red"

FAKE_LEVELS = 9      # pre-rendered blur levels for --fake-data
FAKE_FWHM_BEST = 2.0
FAKE_FWHM_EDGE = 9.0


# ==========================================================================
# --fake-data: synthetic frames to score in place of the black ones
# ==========================================================================
def fake_level(offset: int, fake: dict) -> int:
    d = abs(offset - fake["focus"]) / fake["dmax"]
    return min(FAKE_LEVELS - 1, int(round(d * (FAKE_LEVELS - 1))))


def prepare_fake(width: int, tmp: str) -> dict:
    """Render the blur levels into `tmp`; return what the workers need."""
    import focus_speed_test as fst
    import star_metric_selftest as fhs

    fst.full_size_rig()
    focus = width // 3 // 4 * 4
    fake = dict(dir=tmp, focus=focus, dmax=float(width + focus))
    rng = np.random.default_rng(7)
    cat = fhs.make_catalog(rng, fst.FIELD_RA, fst.FIELD_DEC)
    np.savez(os.path.join(tmp, "cat.npz"), ra=cat[0], dec=cat[1], mag=cat[2])
    jobs = [(i, FAKE_FWHM_BEST + (FAKE_FWHM_EDGE - FAKE_FWHM_BEST) * i
             / (FAKE_LEVELS - 1), os.path.join(tmp, "cat.npz"),
             os.path.join(tmp, f"level{i}.npy"), 100 + i)
            for i in range(FAKE_LEVELS)]
    with mp.get_context("spawn").Pool(FAKE_LEVELS) as pool:
        pool.map(fst._render_level, jobs)

    # fix the scored star set from the sharpest frame, as the real run does
    # from its first frame
    seed = fhs.seed_solution(fst.FIELD_RA, fst.FIELD_DEC, fst.FIELD_ROLL)
    img = np.load(os.path.join(tmp, "level0.npy"), mmap_mode="r")
    res = sm.score_frame(img, cat, seed)
    if not res["ok"]:
        raise RuntimeError(f"synthetic frame did not score: {res['reason']}")
    fake.update(seed=res["sol"], ref=res["ref"])
    print(f"[fake] FAKE DATA: synthetic frames, best focus at offset "
          f"{focus:+d}, FWHM {FAKE_FWHM_BEST} -> {FAKE_FWHM_EDGE} px over "
          f"{fake['dmax']:.0f} units")
    return fake


# ==========================================================================
# worker process: decode + score
# ==========================================================================
def worker(wid: int, tasks, results, fake: dict | None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # main handles Ctrl+C
    if fake:
        import focus_speed_test as fst
        fst.full_size_rig()
        c = np.load(os.path.join(fake["dir"], "cat.npz"))
        deep = (c["ra"], c["dec"], c["mag"])
        # memory-mapped: all workers share one copy through the page cache
        frames = [np.load(os.path.join(fake["dir"], f"level{i}.npy"),
                          mmap_mode="r") for i in range(FAKE_LEVELS)]
    else:
        deep = sm.load_deep_catalog()
    sol = None             # this worker's last good transform, seeds the next
    while True:
        job = tasks.get()
        if job is None:
            return
        idx, offset, raw, name, seed, ref = job
        t0 = time.time()
        gray, rgb16 = sm.frame_from_raw(raw, name)
        del raw
        if fake:              # decoded honestly above, but scored synthetic
            gray, rgb16 = frames[fake_level(offset, fake)], None
        t1 = time.time()
        res = sm.score_frame(gray, deep, sol or seed, ref=ref, rgb16=rgb16)
        if not res["ok"] and sol is not None:
            # this worker's transform may be stale; the shared seed is not
            res = sm.score_frame(gray, deep, seed, ref=ref, rgb16=rgb16)
        t2 = time.time()
        if res["ok"]:
            sol = res["sol"]
        results.put(result_of(idx, offset, res, wid, t1 - t0, t2 - t1))


PER_STAR = ("m_mag", "m_rank", "m_size", "m_usable", "m_sat", "acc_mag",
            "drift_corr")


def result_of(idx, offset, res, wid, decode_s, score_s) -> dict:
    out = dict(idx=idx, offset=offset, ok=res["ok"],
               reason=res.get("reason", ""),
               stars=res.get("score", float("nan")),
               fwhm=res.get("fwhm", float("nan")),
               n_det=res["n_det"], n_on=res.get("n_on", 0),
               n_ref=res.get("n_ref", 0), wid=wid,
               decode=decode_s, score=score_s)
    if res["ok"]:             # per-star detail, for re-scoring star subsets
        out.update({k: res[k] for k in PER_STAR})
    return out


# ==========================================================================
# camera thread
# ==========================================================================
def camera_loop(width: int, tasks, results, stop: threading.Event,
                status: list, fake: dict | None) -> None:
    from focus_hunt import FocusCamera

    rng = np.random.default_rng()
    n_steps = width // 4
    cam = FocusCamera()
    cam.open()
    try:
        cam.start_liveview()     # Nikon only accepts focus drive in live view

        # first frame, at the start position: blind solve + fix the star set
        # (--fake-data: both come with the synthetic frames, nothing to solve)
        seed = ref = None
        idx = 1
        if fake:
            seed, ref, idx = fake["seed"], fake["ref"], 0
        deep = None if fake else sm.load_deep_catalog()
        for attempt in range(0 if fake else SOLVE_TRIES):
            if stop.is_set():
                return
            status[0] = f"solving the first frame (try {attempt + 1})"
            raw, name = cam.capture_raw()
            t0 = time.time()
            gray, rgb16 = sm.frame_from_raw(raw, name)
            del raw
            t1 = time.time()
            time.sleep(sm.CAP_SETTLE_S)
            sol = sm.initial_solve(gray, find_it.DEFAULT_FOCAL_MM, 4.0)
            if sol is None:
                print(f"[warn] blind solve failed on {name}")
                continue
            res = sm.score_frame(gray, deep, sol, rgb16=rgb16)
            if not res["ok"]:
                print(f"[warn] first frame did not score: {res['reason']}")
                continue
            seed, ref = res["sol"], res["ref"]
            print(f"[solve] locked: centre=({sol['center'][0]:.3f}, "
                  f"{sol['center'][1]:+.3f}) inliers={sol['n_inliers']}; "
                  f"scoring a fixed patch of {ref.size} catalogue stars")
            results.put(result_of(0, cam.pos, res, -1, t1 - t0,
                                  time.time() - t1))
            break
        if seed is None:
            status[0] = "FAILED: could not solve the first frame (see terminal)"
            print("[fatal] no plate solve -- is M31 in the frame and in "
                  "rough focus? Close the window to exit.")
            return
        status[0] = ""

        while not stop.is_set():
            offset = int(rng.integers(-n_steps, n_steps + 1)) * 4
            t0 = time.time()
            cam.move_to(offset)
            time.sleep(sm.SETTLE_S)
            t1 = time.time()
            raw, name = cam.capture_raw()
            t2 = time.time()
            while not stop.is_set():     # wait for room, but not past a close
                try:
                    tasks.put((idx, cam.pos, raw, name, seed, ref), timeout=0.5)
                    break
                except queue.Full:
                    pass
            t3 = time.time()
            del raw
            time.sleep(sm.CAP_SETTLE_S)
            print(f"shot #{idx:<4d} pos {cam.pos:+4d}  capture {t2 - t1:5.2f}  "
                  f"queue-wait {t3 - t2:5.2f}  cycle {time.time() - t0:5.2f} s")
            idx += 1
    except Exception as e:
        status[0] = f"FAILED: {e}"
        print(f"[fatal] camera loop: {e!r}")
    finally:
        try:
            cam.stop_liveview()
        except Exception:
            pass
        cam.close()
        print(f"[stop] camera closed, focus left at offset {cam.pos:+d}")


# ==========================================================================
# plot
# ==========================================================================
def means_by_position(x: np.ndarray, y: np.ndarray):
    """Mean of y at each distinct x, sorted by x; NaN values are left out."""
    good = np.isfinite(y)
    x, y = x[good], y[good]
    ux, inv = np.unique(x, return_inverse=True)
    return ux, np.bincount(inv, weights=y) / np.bincount(inv)


class Plot:
    def __init__(self, width: int, results, status: list, fake: bool):
        self.results = results
        self.status = status
        self.data = []            # (offset, stars, fwhm)
        self.prefix = "FAKE DATA -- " if fake else ""

        self.fig = self.make_figure()
        ax = self.fig.add_subplot(111, facecolor="white")
        ax2 = ax.twinx()
        self.ax, self.ax2 = ax, ax2
        ms = POINT_PX * 72.0 / self.fig.dpi
        style = dict(linestyle="", marker="o", markersize=ms,
                     markeredgewidth=0)
        (self.stars_pts,) = ax.plot([], [], color=STARS_COLOR, **style)
        (self.fwhm_pts,) = ax2.plot([], [], color=FWHM_COLOR, **style)
        # lines through the per-position means (all points are still shown)
        (self.stars_line,) = ax.plot([], [], color=STARS_COLOR, linewidth=1)
        (self.fwhm_line,) = ax2.plot([], [], color=FWHM_COLOR, linewidth=1)

        ticks = np.arange(-width, width + 1, 4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:+d}" if t else "0" for t in ticks],
                           rotation=90 if ticks.size > 33 else 0)
        ax.set_xlim(-width - 2, width + 2)
        ax.set_xlabel("focus offset from start (drive units)")
        ax.set_ylabel("identified stars", color=STARS_COLOR)
        ax2.set_ylabel("star FWHM (px)", color=FWHM_COLOR)
        ax.tick_params(axis="y", colors=STARS_COLOR)
        ax2.tick_params(axis="y", colors=FWHM_COLOR)
        ax.grid(True, axis="x", color="0.92")
        self.title = ax.set_title(self.prefix + "waiting for the first frame")

        # mouse guide line: drawn by blitting over a saved background, so
        # moving the mouse never triggers a full redraw
        self.bg = None
        self.mouse_x = None
        self.vline = ax2.axvline(0, color="0.4", linewidth=0.8,
                                 animated=True, visible=False)
        self.vlabel = ax2.text(0, 1.0, "", transform=ax2.get_xaxis_transform(),
                               ha="left", va="bottom", fontsize=9,
                               color="0.3", animated=True, visible=False)
        c = self.fig.canvas
        c.mpl_connect("draw_event", self.on_draw)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("axes_leave_event", self.on_leave)

        self.timer = c.new_timer(interval=200)
        self.timer.add_callback(self.poll)
        self.timer.start()
        self.fig.tight_layout()

    # -- window ------------------------------------------------------------
    def make_figure(self):
        import matplotlib
        matplotlib.use("TkAgg")
        matplotlib.rcParams["toolbar"] = "None"
        import matplotlib.pyplot as plt
        self.plt = plt
        fig = plt.figure(figsize=(12, 6), facecolor="white")
        fig.canvas.manager.set_window_title("focus hunt")
        return fig

    def on_close(self, callback) -> None:
        self.fig.canvas.mpl_connect("close_event", lambda _e: callback())

    def close(self) -> None:
        self.plt.close(self.fig)

    def run(self) -> None:
        self.plt.show()

    # -- data --------------------------------------------------------------
    def add(self, r: dict) -> None:
        self.data.append((r["offset"], r["stars"], r["fwhm"]))

    def series(self):
        """(offsets, star counts, FWHMs) of all frames, as arrays."""
        d = np.array(self.data, float).reshape(-1, 3)
        return d[:, 0], d[:, 1], d[:, 2]

    def poll(self) -> None:
        new = False
        while True:
            try:
                r = self.results.get_nowait()
            except queue.Empty:
                break
            if not r["ok"]:
                print(f"  scored #{r['idx']:<4d} pos {r['offset']:+4d}  "
                      f"FAILED: {r['reason']}")
                continue
            print(f"  scored #{r['idx']:<4d} pos {r['offset']:+4d}  "
                  f"stars {r['stars']:6.0f}  fwhm {r['fwhm']:5.2f} px  "
                  f"(decode {r['decode']:4.2f} s, score {r['score']:4.2f} s)")
            if r["n_on"] < r["n_ref"]:
                print(f"  [warn] field drifted: {r['n_ref'] - r['n_on']} of "
                      f"{r['n_ref']} patch stars off the sensor (count scaled "
                      f"up to match); re-centre the mount if this grows")
            self.add(r)
            new = True
        if new or self.status[0] != getattr(self, "_shown_status", None):
            self.redraw()

    def redraw(self) -> None:
        x, stars, fwhm = self.series()
        self.stars_pts.set_data(x, stars)
        self.fwhm_pts.set_data(x, fwhm)
        self.stars_line.set_data(*means_by_position(x, stars))
        self.fwhm_line.set_data(*means_by_position(x, fwhm))
        for a in (self.ax, self.ax2):
            a.relim()
            a.autoscale_view(scalex=False)
        self._shown_status = self.status[0]
        self.title.set_text(self.prefix +
                            (self.status[0] or f"{len(self.data)} frames"))
        self.fig.canvas.draw()

    # -- mouse guide line ----------------------------------------------------
    def on_draw(self, _event) -> None:
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_guide()

    def on_move(self, event) -> None:
        self.mouse_x = event.xdata if event.inaxes else None
        self.draw_guide()

    def on_leave(self, _event) -> None:
        self.mouse_x = None
        self.draw_guide()

    def draw_guide(self) -> None:
        if self.bg is None:
            return
        c = self.fig.canvas
        c.restore_region(self.bg)
        if self.mouse_x is not None:
            x = self.mouse_x
            self.vline.set_xdata([x, x])
            self.vline.set_visible(True)
            self.vlabel.set_x(x)
            self.vlabel.set_text(f" {x:+.0f}")
            self.vlabel.set_visible(True)
            self.ax2.draw_artist(self.vline)
            self.ax2.draw_artist(self.vlabel)
        c.blit(self.fig.bbox)



# ==========================================================================
def parse_args(argv=None, doc=__doc__):
    p = argparse.ArgumentParser(
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("width", type=int,
                   help="scan focus in [start - WIDTH, start + WIDTH] "
                        "(rounded down to a multiple of 4)")
    p.add_argument("--fake-data", action="store_true",
                   help="shoot and decode as normal, but score synthetic "
                        "star fields (for testing with the lens covered)")
    return p.parse_args(argv)


def main(argv=None, plot_cls=Plot, doc=__doc__) -> int:
    args = parse_args(argv, doc)
    width = args.width // 4 * 4
    if width < 4:
        print("WIDTH must be at least 4")
        return 2
    fake = tmp = None
    if args.fake_data:
        tmp = tempfile.mkdtemp(prefix="focus_hunt_fake_")
        fake = prepare_fake(width, tmp)
    else:
        sm.load_deep_catalog()      # fail early if it has not been built

    # workers are spawned before the camera is opened, so they never inherit
    # any gphoto2 state
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue(maxsize=QUEUE_MAX)
    results = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, tasks, results, fake),
                         daemon=True)
             for w in range(WORKERS)]
    for p in procs:
        p.start()

    stop = threading.Event()
    status = ["starting the camera"]
    plot = plot_cls(width, results, status, fake is not None)
    cam_thread = threading.Thread(target=camera_loop, daemon=True,
                                  args=(width, tasks, results, stop, status,
                                        fake))
    cam_thread.start()

    # closing the window ends the run; Ctrl+C in the terminal closes it
    plot.on_close(stop.set)
    signal.signal(signal.SIGINT, lambda *_: plot.close())
    try:
        plot.run()
    finally:
        stop.set()
        print("[stop] window closed, finishing the current shot ...")
        cam_thread.join(timeout=60)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
