#!/usr/bin/env python
"""Star-based focus scan with a live plot and a per-magnitude star filter.

    camera/focus_hunt_auto2.py WIDTH
    camera/focus_hunt_auto2.py WIDTH --fake-data    # test with the lens covered

Same scan as focus_hunt_auto.py (random focus positions in +/- WIDTH, shot and
scored in the background, runs until the window is closed, focus left where
the last shot put it), with one addition: a panel right of the plot lists the
catalogue magnitude bins (1 mag wide) of the identified stars.  Each row shows
the largest number of stars of that bin identified in any one frame so far,
and, in that same frame, the fraction of them with at least one saturated
pixel (any colour channel >= star_metric.SAT_LEVEL, 90% of the sensor's
range).  Each row has a checkbox, checked by default.  The plot uses only the stars of the
checked bins:

  * blue -- identified stars in the checked bins, minus the accidental matches
            that fall in those bins, scaled for drift like the full count
  * red  -- median FWHM of the FWHM_N brightest unsaturated identified stars
            in the checked bins

Unchecking the faint bins, for example, shows whether the count near focus is
driven by the faint stars (which appear only near focus) or by all of them.
"""
from __future__ import annotations

import sys

import numpy as np

import focus_hunt_auto as base
import star_metric as sm


class Plot2(base.Plot):
    """The focus_hunt_auto plot inside a Tk window with a magnitude panel."""

    def __init__(self, width: int, results, status: list, fake: bool):
        self.frames = []          # per frame: offset, bins and per-star data
        self.cache = {}           # frame index -> (stars, fwhm) for `checked`
        self.rows = {}            # mag bin -> (BooleanVar, Checkbutton,
                                  #             count Label, saturated Label)
        self.max_n = {}           # mag bin -> most stars in one frame
        super().__init__(width, results, status, fake)

    # -- window ------------------------------------------------------------
    def make_figure(self):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("focus hunt")
        self.root.configure(bg="white")
        self.root.geometry("1400x700")
        self.closed_cb = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # packed first, so a narrow window shrinks the plot, not the panel
        self.panel = tk.Frame(self.root, bg="white")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        tk.Label(self.panel, text="mag", bg="white",
                 font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, sticky="w")
        tk.Label(self.panel, text="max stars", bg="white",
                 font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=1, sticky="e", padx=(12, 0))
        tk.Label(self.panel, text="saturated", bg="white",
                 font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=2, sticky="e", padx=(12, 0))

        fig = Figure(figsize=(12, 6), facecolor="white")
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return fig

    def on_close(self, callback) -> None:
        self.closed_cb = callback

    def close(self) -> None:
        if self.closed_cb:
            self.closed_cb()
        self.root.quit()

    def run(self) -> None:
        self.root.mainloop()
        self.root.destroy()

    # -- magnitude panel -----------------------------------------------------
    def add_row(self, b: int) -> None:
        tk = self.tk
        var = tk.BooleanVar(master=self.root, value=True)
        cb = tk.Checkbutton(self.panel, text=f"{b} - {b + 1}", variable=var,
                            command=self.on_toggle, bg="white",
                            activebackground="white", highlightthickness=0,
                            anchor="w")
        lbl = tk.Label(self.panel, text="0", bg="white", anchor="e")
        sat = tk.Label(self.panel, text="", bg="white", anchor="e")
        self.rows[b] = (var, cb, lbl, sat)
        for i, k in enumerate(sorted(self.rows)):     # keep rows in mag order
            self.rows[k][1].grid(row=i + 1, column=0, sticky="w")
            self.rows[k][2].grid(row=i + 1, column=1, sticky="e", padx=(12, 0))
            self.rows[k][3].grid(row=i + 1, column=2, sticky="e", padx=(12, 0))

    def checked(self) -> np.ndarray:
        return np.array([b for b, (v, *_) in self.rows.items() if v.get()],
                        int)

    def on_toggle(self) -> None:
        self.cache.clear()
        self.redraw()

    # -- data --------------------------------------------------------------
    def add(self, r: dict) -> None:
        super().add(r)
        f = dict(offset=r["offset"], drift_corr=r["drift_corr"],
                 m_bin=np.floor(r["m_mag"]).astype(int),
                 acc_bin=np.floor(r["acc_mag"]).astype(int),
                 m_rank=r["m_rank"], m_size=r["m_size"],
                 m_usable=r["m_usable"])
        self.frames.append(f)
        bins, inv, counts = np.unique(f["m_bin"], return_inverse=True,
                                      return_counts=True)
        n_sat = np.bincount(inv, weights=r["m_sat"], minlength=bins.size)
        for b, n, ns in zip(bins.tolist(), counts.tolist(), n_sat.tolist()):
            if b not in self.rows:
                self.add_row(b)
            if n > self.max_n.get(b, 0):
                self.max_n[b] = n
                self.rows[b][2].configure(text=str(n))
                self.rows[b][3].configure(text=f"{ns / n:.4f}")

    def score(self, i: int, sel: np.ndarray):
        if i not in self.cache:
            f = self.frames[i]
            m = np.isin(f["m_bin"], sel)
            n_acc = int(np.isin(f["acc_bin"], sel).sum())
            stars = (int(m.sum()) - n_acc) * f["drift_corr"]
            fwhm = sm.fwhm_of_sizes(f["m_rank"][m], f["m_size"][m],
                                    f["m_usable"][m])
            self.cache[i] = (stars, fwhm)
        return self.cache[i]

    def series(self):
        sel = self.checked()
        x = np.array([f["offset"] for f in self.frames], float)
        s = np.array([self.score(i, sel) for i in range(len(self.frames))],
                     float).reshape(-1, 2)
        return x, s[:, 0], s[:, 1]


if __name__ == "__main__":
    sys.exit(base.main(plot_cls=Plot2, doc=__doc__))
