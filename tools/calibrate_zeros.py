"""Interactive zero calibration against physical references.

Sets each motor's zero (SET_ZERO_POSITION, stored in the motor) at a reference
you can reproduce by hand, one joint at a time:

  1. HIPS    align each hip with a straight edge (sim zero), press Enter.
             The hip is zeroed and then HELD FIRM there for the next steps.
  2. THIGHS  push each thigh against the body and hold it, press Enter.
             Zeroed there, then left limp.
  3. CALVES  first place the thighs where it is comfortable and press Enter to
             lock them; then push each calf to its stop and hold it, press Enter.

Where those references sit in the sim frame (0, 102.9 and 75 deg) lives in
config.py as each joint's `sim_offset`, with its sign in ZERO_OFFSET_SIGN; check
them in the live viewer afterwards (tools/stream_joints.py).

Works in the motors' RAW frame throughout: no sim frame, no whole-turn
correction, no joint limits (the stops are outside them). Only the hips, and
in step 3 the thighs, are ever under gain, at --hold-kp. A joint must be still
for 0.3 s before it is zeroed, and everything is released if a held joint is
pushed more than 15 deg from where it is held. Ctrl-C releases everything.

    source setup.sh
    python tools/calibrate_zeros.py                   # all three steps
    python tools/calibrate_zeros.py --only calves     # redo one step
"""

import argparse
import collections
import dataclasses
import math
import os
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEC, ZERO_REFERENCE_DEG  # noqa: E402
from robot.session import RobotSession  # noqa: E402
from robstride.protocol import CommunicationType  # noqa: E402
from utils.exceptions import HardwareError  # noqa: E402

STEPS = ("hips", "thighs", "calves")
# Stillness is judged on position alone: the motors' velocity feedback is too
# noisy at rest (and worse against a stop) to be a reliable test.
STILL_WINDOW_S = 0.3
STILL_POS_RAD = math.radians(0.5)
MAX_HOLD_ERROR = math.radians(15)
ZERO_TOLERANCE = 0.02              # rad, read-back after SET_ZERO


class CalibrationAborted(Exception):
    pass


class Calibrator:
    """Keeps every motor fed at the control rate while waiting on the user."""

    def __init__(self, robot, hold_kp, hold_kd):
        self.robot = robot
        self.hold_kp, self.hold_kd = hold_kp, hold_kd
        self.targets = robot.network.limp_targets()
        self.kp = {n: 0.0 for n in SPEC.names}
        self.kd = {n: 0.0 for n in SPEC.names}
        self.held = set()
        self.state = {}
        self.history = collections.deque()       # (t, {name: (pos, vel)})
        self.period = SPEC.dt

    # ------------------------------------------------------------- loop --

    def tick(self):
        self.state = self.robot.exchange(self.targets, self.kp, self.kd, timeout=0.02)
        now = time.monotonic()
        self.history.append((now, {n: (s['pos'], s['vel']) for n, s in self.state.items()}))
        while self.history and now - self.history[0][0] > 1.0:
            self.history.popleft()
        for name in self.held:
            error = self.state[name]['pos'] - self.targets[name]['pos']
            if abs(error) > MAX_HOLD_ERROR:
                raise CalibrationAborted(
                    f"{name} was pushed {math.degrees(error):+.1f} deg from where it "
                    f"is held (limit {math.degrees(MAX_HOLD_ERROR):.0f}).")

    def run_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.tick()
            time.sleep(self.period)

    def status(self, names):
        return "   ".join(f"{n} {math.degrees(self.state[n]['pos']):+7.1f} deg"
                          for n in names)

    def wait_enter(self, prompt, show):
        """Tick until the user presses Enter. 'q' + Enter aborts."""
        print(f"\n>> {prompt}\n   Press Enter when ready ('q' + Enter to quit).")
        last_print = 0.0
        while True:
            self.tick()
            now = time.monotonic()
            if now - last_print > 0.2 and sys.stdin.isatty():
                print(f"\r   {self.status(show)}   ", end="", flush=True)
                last_print = now
            ready, _, _ = select.select([sys.stdin], [], [], self.period)
            if ready:
                line = sys.stdin.readline()
                if sys.stdin.isatty():
                    print()
                if line == "" or line.strip().lower() == "q":
                    raise CalibrationAborted("stopped by user.")
                return

    # ----------------------------------------------------------- joints --

    def still_spread(self, name):
        """Position spread (rad) over the last STILL_WINDOW_S, after waiting
        for the history to span it."""
        start = time.monotonic()
        while time.monotonic() - start < STILL_WINDOW_S:
            self.tick()
            time.sleep(self.period)
        now = time.monotonic()
        pos = [s[name][0] for t, s in self.history if now - t <= STILL_WINDOW_S]
        return max(pos) - min(pos)

    def zero(self, name):
        if name in self.held:
            raise RuntimeError(f"refusing to zero {name} while it is held under gain")
        joint = SPEC.joint(name)
        self.robot.network.buses[joint.bus].transmit(
            comm_type=CommunicationType.SET_ZERO_POSITION,
            extra_data=SPEC.host_id,
            destination_id=joint.can_id,
            data=b'\x01' + b'\x00' * 7,
        )
        # Keep feeding every motor (held hips would time out on a blocking wait).
        self.run_for(0.3)
        pos = self.state[name]['pos']
        if abs(pos) > ZERO_TOLERANCE:
            raise CalibrationAborted(
                f"{name} reads {pos:+.4f} rad after zeroing, not ~0.")
        print(f"   [OK] {name} zeroed (reads {math.degrees(pos):+.2f} deg).")

    def hold(self, name):
        """Hold a joint firm exactly where it is now."""
        self.targets[name] = {'pos': self.state[name]['pos'], 'vel': 0.0, 'torque': 0.0}
        self.kp[name], self.kd[name] = self.hold_kp, self.hold_kd
        self.held.add(name)

    def zero_at_reference(self, name, instruction):
        while True:
            self.wait_enter(instruction, show=[name])
            spread = self.still_spread(name)
            if spread < STILL_POS_RAD:
                break
            print(f"   {name} moved {math.degrees(spread):.2f} deg in the last "
                  f"{STILL_WINDOW_S:.1f} s (limit {math.degrees(STILL_POS_RAD):.1f}); "
                  f"hold it steady and press Enter again.")
        self.zero(name)


def joints_of(link):
    return [f"{side}_{link}" for side in ("left", "right")]


def run(cal, steps):
    hips, thighs, calves = joints_of("hip"), joints_of("thigh"), joints_of("calf")
    cal.run_for(0.3)

    if "hips" in steps:
        print("\n=== Step 1: hips, zero at the straight edge ===")
        for name in hips:
            side = name.split("_")[0].upper()
            cal.zero_at_reference(
                name, f"Align the {side} HIP with the straight edge and let go.")
            cal.hold(name)
            print(f"   {name} is now held firm at its zero.")

    if "thighs" in steps or "calves" in steps:
        unheld = [n for n in hips if n not in cal.held]
        if unheld:
            cal.wait_enter("Set the HIPS at their zero (straight edge); they will be "
                           "held firm where they are.", show=unheld)
            for name in unheld:
                cal.hold(name)

    if "thighs" in steps:
        print(f"\n=== Step 2: thighs, zero against the body "
              f"({ZERO_REFERENCE_DEG['thigh']} deg from sim zero) ===")
        for name in thighs:
            side = name.split("_")[0].upper()
            cal.zero_at_reference(
                name, f"Push the {side} THIGH against the body and hold it there.")

    if "calves" in steps:
        print(f"\n=== Step 3: calves, zero against their stop "
              f"({ZERO_REFERENCE_DEG['calf']} deg from sim zero) ===")
        cal.wait_enter("Place both THIGHS where comfortable; they will be held "
                       "firm where they are.", show=thighs)
        for name in thighs:
            cal.hold(name)
        for name in calves:
            side = name.split("_")[0].upper()
            cal.zero_at_reference(
                name, f"Push the {side} CALF against its stop and hold it there.")

    print("\n=== Done ===")
    print("   " + cal.status(SPEC.names))


def build_spec(virtual):
    if not virtual:
        return SPEC, None
    buses = [dataclasses.replace(b, interface="virtual") for b in SPEC.buses]
    spec = dataclasses.replace(SPEC, buses=buses)
    from tools.fake_motors import FakeRobot
    return spec, FakeRobot(spec)


def main():
    p = argparse.ArgumentParser(description="Zero the joints at physical references.")
    p.add_argument('--only', choices=STEPS, action='append',
                   help="Run only this step (repeatable). Default: all three.")
    p.add_argument('--hold-kp', type=float, default=40.0,
                   help="Stiffness holding the hips/thighs, N.m/rad.")
    p.add_argument('--hold-kd', type=float, default=2.0,
                   help="Damping holding the hips/thighs, N.m.s/rad.")
    p.add_argument('--virtual', action='store_true',
                   help="Run against tools/fake_motors.py; no hardware.")
    args = p.parse_args()
    steps = args.only or list(STEPS)

    spec, simulator = build_spec(args.virtual)
    if simulator is not None:
        simulator.start()
    try:
        with RobotSession(spec, use_imu=False, realtime=False, raw_frame=True) as robot:
            print("[INFO] All motors enabled and limp.")
            run(Calibrator(robot, args.hold_kp, args.hold_kd), steps)
            print("\nReleasing all motors. The zeros are stored in the motors.")
            print("Next: tools/stream_joints.py + the live viewer, to find the signs "
                  "in config.ZERO_OFFSET_SIGN.")
    except CalibrationAborted as e:
        print(f"\n[ABORTED] {e} All motors released; zeros already set are kept.")
        sys.exit(2)
    except HardwareError as e:
        print(f"\n[CRITICAL] Hardware failure: {e}")
        sys.exit(3)
    finally:
        if simulator is not None:
            simulator.stop()


if __name__ == "__main__":
    main()
