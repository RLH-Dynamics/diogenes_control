"""Passive state read across every bus. Nothing moves.

Bring-up tool. With kp = kd = 0 the actuators stay limp, so this is safe to run
as the first thing after wiring a leg. Use it to confirm each CAN id lands on the
channel config.py says it does, and to check joint direction signs by moving
each joint by hand and watching which reading changes.
"""

import argparse
import sys
import time

from config import SPEC
from robot.session import RobotSession
from utils.exceptions import HardwareError


def parse_args():
    p = argparse.ArgumentParser(description="Read joint state passively.")
    p.add_argument('--rate', type=float, default=10.0, help="Refresh rate in Hz.")
    p.add_argument('--no-imu', action='store_true', help="Skip the IMU.")
    p.add_argument('--once', action='store_true', help="Print one frame and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    period = 1.0 / args.rate

    try:
        with RobotSession(SPEC, use_imu=not args.no_imu, realtime=False) as robot:
            print("[INFO] Motors enabled and limp. Reading state...")
            time.sleep(0.5)

            while True:
                state = robot.read_limp_state(timeout=0.05)

                if not args.once:
                    print("\033[H\033[J", end="")   # clear screen
                print(f"--- Joint state ({SPEC.num_joints} joints, "
                      f"{len(SPEC.buses)} buses) ---")
                print(f"{'joint':<14}{'bus':<7}{'id':<5}"
                      f"{'pos (rad)':>12}{'vel (rad/s)':>14}"
                      f"{'torque (Nm)':>14}{'temp (C)':>11}")
                for joint in SPEC.joints:
                    r = state[joint.name]
                    print(f"{joint.name:<14}{joint.bus:<7}{joint.can_id:<5}"
                          f"{r['pos']:>12.4f}{r['vel']:>14.4f}"
                          f"{r['torque']:>14.4f}{r['temp']:>11.1f}")

                if not args.no_imu and SPEC.imu is not None:
                    s = robot.read_imu()
                    print(f"\n--- Base (IMU) ---")
                    print(f"  tilt      : {s.tilt_rad * 57.2958:>7.2f} deg "
                          f"(age {s.age() * 1000:.1f} ms, accuracy {s.status})")
                    print(f"  ang vel   : "
                          + "  ".join(f"{a}={v:>7.3f}" for a, v in zip('xyz', s.ang_vel)))
                    print(f"  proj grav : "
                          + "  ".join(f"{a}={v:>7.3f}" for a, v in zip('xyz', s.projected_gravity)))

                stats = robot.network.bus_statistics()
                print(f"\nexchanges: {stats['exchanges']}  "
                      f"timeouts: {stats['timeouts']}  "
                      f"last: {stats['last_exchange_ms']:.2f} ms")

                if args.once:
                    break
                time.sleep(period)

    except HardwareError as e:
        print(f"\n[CRITICAL] Hardware failure: {e}")
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
