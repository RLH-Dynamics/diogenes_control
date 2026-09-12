"""BNO085 bring-up and mount-rotation calibration aid. Touches no actuators.

DETERMINING THE MOUNT ROTATION
------------------------------
config.IMU.mount_rotation maps the SENSOR frame to the ROBOT BASE frame. Get it
wrong and the policy will fight gravity, so measure it rather than guessing:

  1. Sit the robot level and upright. `proj grav` should read approximately
     (0, 0, -1). Whichever axis shows -1 is the sensor axis pointing up.
  2. Pitch the robot nose-down about the base's Y axis. The component that grows
     is the sensor axis pointing forward.
  3. Roll the robot about the base's X axis to confirm the third axis.

Build the matrix whose rows map sensor axes onto base axes, write it into
config.IMU.mount_rotation, and re-run: level should read (0, 0, -1) and each
rotation should move the expected base axis.

The rate figures printed at exit are the ones that matter for the control loop:
if the effective rate is far below 1 / report_interval_s, or errors are frequent,
consider lowering i2c_arm_baudrate or moving to the sensor's UART-RVC mode.
"""

import argparse
import sys
import time

import numpy as np

from config import SPEC
from sensors.imu import Bno085Reader
from utils.exceptions import HardwareError


def parse_args():
    p = argparse.ArgumentParser(description="Read and characterise the IMU.")
    p.add_argument('--rate', type=float, default=10.0, help="Display rate in Hz.")
    p.add_argument('--duration', type=float, default=None,
                   help="Exit after this many seconds.")
    p.add_argument('--i2c-hz', type=int, default=50_000,
                   help="I2C bus frequency (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()
    reader = Bno085Reader(SPEC.imu, i2c_frequency=args.i2c_hz)

    ages, started = [], time.perf_counter()
    try:
        reader.start()
        last_count, last_t = reader.sample_count, time.perf_counter()

        while True:
            s = reader.latest()
            age = s.age()
            ages.append(age)

            now = time.perf_counter()
            elapsed = now - last_t
            rate = (reader.sample_count - last_count) / elapsed if elapsed > 0 else 0.0
            last_count, last_t = reader.sample_count, now

            print("\033[H\033[J", end="")
            print("--- BNO085 ---")
            print(f"  quat (wxyz) : " + "  ".join(f"{v:>7.4f}" for v in s.quat))
            print(f"  ang vel     : "
                  + "  ".join(f"{a}={v:>8.4f}" for a, v in zip('xyz', s.ang_vel)))
            print(f"  lin accel   : "
                  + "  ".join(f"{a}={v:>8.4f}" for a, v in zip('xyz', s.lin_accel)))
            print(f"  proj grav   : "
                  + "  ".join(f"{a}={v:>8.4f}" for a, v in zip('xyz', s.projected_gravity)))
            print(f"  tilt        : {s.tilt_rad * 57.2958:.2f} deg")
            print(f"\n  accuracy    : {s.status}/3")
            print(f"  sample age  : {age * 1000:>6.1f} ms "
                  f"(limit {SPEC.imu.max_age_s * 1000:.0f} ms)")
            print(f"  rate        : {rate:>6.1f} Hz")
            print(f"  samples     : {reader.sample_count}   "
                  f"errors: {reader.read_errors}")
            print("\nLevel and upright should read proj grav ~ (0, 0, -1).")

            if args.duration is not None and (now - started) >= args.duration:
                break
            time.sleep(1.0 / args.rate)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping.")
    except HardwareError as e:
        print(f"\n[CRITICAL] {e}")
        sys.exit(3)
    finally:
        if ages:
            a = np.array(ages) * 1000.0
            total = time.perf_counter() - started
            print(f"\n--- Summary over {total:.1f} s ---")
            print(f"  samples      : {reader.sample_count} "
                  f"({reader.sample_count / total:.1f} Hz effective)")
            print(f"  read errors  : {reader.read_errors}")
            print(f"  sample age   : mean {a.mean():.1f} ms, "
                  f"p99 {np.percentile(a, 99):.1f} ms, max {a.max():.1f} ms")
        reader.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
