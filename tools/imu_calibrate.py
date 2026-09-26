"""Guided one-off calibration of the BNO085's accelerometer and gyro.

Touches no actuators. The chip calibrates itself while it moves; this turns
that on for the accelerometer and gyro (not the magnetometer, which this robot
does not use), walks you through the motions it needs, shows the chip's own
accuracy estimate live (0 unreliable .. 3 high), and on your say-so saves the
result to the chip's flash, from which it loads at every power-up. So it is
done once, and again only if the IMU is remounted.

The calibration lives in the chip's RAM until saved, and the driver resets the
chip whenever it connects, so save before quitting: quitting unsaved loses the
session.

The steps wait for Enter; take as long as the robot needs. For the tilts,
larger is better within reason -- the accelerometer calibrates best from
clearly different orientations -- but ~20-30 deg is useful.

    source setup.sh
    python tools/imu_calibrate.py
"""

import math
import os
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEC  # noqa: E402
from sensors.imu import Bno085Reader  # noqa: E402

STEPS = [
    ("Leave the robot completely STILL, upright (gyro).", 5),
    ("Hold it LEVEL and still.", 3),
    ("Tip the FRONT down and hold it there.", 3),
    ("Tip the FRONT up and hold it there.", 3),
    ("Tip the LEFT side down and hold it there.", 3),
    ("Tip the RIGHT side down and hold it there.", 3),
    ("Back to LEVEL and still.", 3),
]


def status_line(reader):
    s = reader.latest()
    acc = reader.accuracy()
    g = s.projected_gravity
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, g[0]))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, g[1]))))
    return (f"accel {acc['accel']}/3  gyro {acc['gyro']}/3   "
            f"pitch {pitch:+5.1f}  roll {roll:+5.1f}  "
            f"|gyro| {math.sqrt(sum(v * v for v in s.ang_vel)):.2f} rad/s")


def wait_enter(reader, prompt):
    """Show the live status until Enter. Returns the line typed."""
    print(f"\n>> {prompt}")
    print("   Press Enter when done.")
    while True:
        print(f"\r   {status_line(reader)}      ", end="", flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            line = sys.stdin.readline()
            print()
            return line.strip().lower()


def main():
    reader = Bno085Reader(SPEC.imu)
    reader.start()
    try:
        reader.begin_calibration()
        print("[INFO] Accelerometer + gyro calibration enabled on the chip.")
        for instruction, hold_s in STEPS:
            wait_enter(reader, f"{instruction} (about {hold_s} s, then Enter)")

        acc = reader.accuracy()
        print(f"\nFinal accuracy: accel {acc['accel']}/3, gyro {acc['gyro']}/3.")
        if min(acc.values()) < 2:
            print("That is low; you can repeat the motions before saving "
                  "(answer 'r'), or save anyway.")
        while True:
            # A plain prompt: no live status line redrawing over the typing.
            print(f"\nNow: {status_line(reader)}")
            answer = input("Save this calibration to the chip? "
                           "[y = save / r = repeat the tilts / q = quit unsaved]: ")
            answer = answer.strip().lower()
            if answer in ("r", "repeat"):
                for instruction, hold_s in STEPS[1:]:
                    wait_enter(reader, f"{instruction} (about {hold_s} s, then Enter)")
                continue
            if answer in ("y", "yes"):
                try:
                    reader.save_calibration()
                except RuntimeError as e:
                    print(f"[ERROR] The chip did not confirm the save: {e}")
                    continue
                print("[OK] The chip confirmed: calibration saved to its flash, "
                      "and it loads at every power-up.")
                break
            if answer in ("q", "quit"):
                print("Not saved. This session's calibration is lost when the tool "
                      "exits (the driver resets the chip on connect).")
                break
            print(f"   Got {answer!r}; please answer y, r or q.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped without saving.")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
