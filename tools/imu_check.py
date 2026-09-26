"""Guided check that config.IMU.mount_rotation matches the real sensor.

Touches no actuators. Reads the BNO085 through the same reader the control
loop uses, so what passes here is what the tilt interlock (and later a
walking policy) will see. Base frame: +x forward, +y left, +z up.

Each step waits for Enter, then watches (30 s by default, --watch) with a live
reading, and passes as soon as the motion is seen:

  1. LEVEL       hold the robot still and upright   gravity ~ (0, 0, -1)
  2. NOSE DOWN   tip the front down ~20 deg          gravity x goes positive
  3. LEFT DOWN   tip the left side down ~20 deg      gravity y goes positive
  4. TURN LEFT   rotate it left (CCW seen from above) gyro z goes positive

A level reading alone only proves which chip axis points up; steps 2-4 pin
the other two, including a chip mounted turned about the vertical.

    source setup.sh
    python tools/imu_check.py [--watch 60]
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEC  # noqa: E402
from sensors.imu import Bno085Reader  # noqa: E402

LEVEL_TOL_DEG = 10.0
TILT_MIN = math.sin(math.radians(10))      # gravity component for a 10 deg tip
RATE_MIN = 0.5                             # rad/s


def watch(reader, test, seconds):
    """Sample until test(sample) passes or time runs out, showing the reading."""
    end = time.monotonic() + seconds
    best, last_print = None, 0.0
    while time.monotonic() < end:
        s = reader.latest()
        ok, detail, score = test(s)
        if best is None or score > best[2]:
            best = (ok, detail, score)
        if ok:
            print()
            return best
        now = time.monotonic()
        if now - last_print > 0.25:
            print(f"\r   {end - now:4.0f} s left   {detail}      ", end="", flush=True)
            last_print = now
        time.sleep(0.02)
    print()
    return best


def dominant(v, axis):
    """Component `axis` is positive and larger than the other two."""
    v = np.asarray(v)
    others = [abs(v[i]) for i in range(3) if i != axis]
    return v[axis] > 0 and v[axis] > max(others)


def main():
    p = argparse.ArgumentParser(description="Check the IMU mount rotation.")
    p.add_argument('--watch', type=float, default=30.0,
                   help="Seconds to wait for each motion (default: %(default)s).")
    args = p.parse_args()

    reader = Bno085Reader(SPEC.imu)
    reader.start()
    results = []

    steps = [
        ("LEVEL", "Hold the robot still, upright and level.",
         lambda s: (s.tilt_rad < math.radians(LEVEL_TOL_DEG)
                    and s.projected_gravity[2] < 0,
                    f"gravity {np.round(s.projected_gravity, 2)}, "
                    f"tilt {math.degrees(s.tilt_rad):.1f} deg",
                    -s.tilt_rad)),
        ("NOSE DOWN", "Tip the FRONT of the robot down about 20 deg and hold.",
         lambda s: (s.projected_gravity[0] > TILT_MIN and dominant(
                    [s.projected_gravity[0], s.projected_gravity[1], 0], 0),
                    f"gravity {np.round(s.projected_gravity, 2)}",
                    s.projected_gravity[0])),
        ("LEFT DOWN", "Tip the robot's LEFT side down about 20 deg and hold.",
         lambda s: (s.projected_gravity[1] > TILT_MIN and dominant(
                    [s.projected_gravity[0], s.projected_gravity[1], 0], 1),
                    f"gravity {np.round(s.projected_gravity, 2)}",
                    s.projected_gravity[1])),
        ("TURN LEFT", "Level again, then turn the robot LEFT (anticlockwise "
         "seen from above), briskly.",
         lambda s: (s.ang_vel[2] > RATE_MIN and dominant(s.ang_vel, 2),
                    f"gyro {np.round(s.ang_vel, 2)} rad/s",
                    s.ang_vel[2])),
    ]

    try:
        for name, instruction, test in steps:
            print(f"\n>> {name}: {instruction}")
            input(f"   Press Enter, then do it (you have {args.watch:.0f} s)...")
            ok, detail, _ = watch(reader, test, args.watch)
            print(f"   [{'PASS' if ok else 'FAIL'}] {detail}")
            results.append((name, ok))
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        reader.stop()

    print("\n--- Summary ---")
    for name, ok in results:
        print(f"  {name:<10} {'PASS' if ok else 'FAIL'}")
    if len(results) == len(steps) and all(ok for _, ok in results):
        print("config.IMU.mount_rotation matches the sensor.")
        sys.exit(0)
    print("Mismatch: note which steps failed and the readings shown.")
    sys.exit(1)


if __name__ == "__main__":
    main()
