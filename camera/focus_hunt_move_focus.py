#!/usr/bin/env python3
"""Move the camera focus by a relative amount and exit.

Usage
-----
    camera/focus_hunt_move_focus.py DELTA
    e.g.  camera/focus_hunt_move_focus.py -40      # drive focus 40 units nearer
          camera/focus_hunt_move_focus.py 40       # drive focus 40 units farther

DELTA is a signed integer in focus-drive units (the same units focus_hunt uses).
The move is RELATIVE to wherever the lens currently is. Live view is entered
(Nikon manualfocusdrive only works in live view), the move is issued in safe
chunks, and the camera is released on exit.

Run with the python that has gphoto2 bound (the camera machine).
"""
from __future__ import annotations

import argparse
import os
import sys

# Reuse the self-healing camera wrapper from focus_hunt.py (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focus_hunt import FocusCamera, set_final_image_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move the camera focus by a relative (signed) amount.")
    parser.add_argument("delta", type=int,
                        help="signed focus-drive units to move (negative = one "
                             "way, positive = the other)")
    args = parser.parse_args()

    cam = FocusCamera()
    opened = False
    try:
        cam.open()
        opened = True
        cam.start_liveview()
        print(f"[info] moving focus by {args.delta:+d} units")
        cam.drive_focus(args.delta)
        print(f"[info] done; tracked position now {cam.pos:+d} "
              f"(relative to start)")
    except KeyboardInterrupt:
        print("\n[abort] interrupted by user")
    finally:
        cam.stop_liveview()
        # Leave the camera shooting raw (.NEF) for the eclipse; print last.
        final_line = set_final_image_quality(cam) if opened else None
        cam.close()
        if final_line is not None:
            print()
            print(final_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
