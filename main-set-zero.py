"""Set the mechanical zero position of the actuators.

WARNING: this overwrites the motors' internal zero. Hold the robot physically
still at the intended zero pose before confirming.

Now scoped: `--leg left` or `--joint left_calf` zeroes a subset, which matters
with two legs since you will usually be jigging one at a time.

This zeroes a joint WHEREVER IT IS. Thighs and calves are zeroed against hard
stops by tools/calibrate_zeros.py instead (their config `sim_offset` is not 0),
so they are refused here unless --force.
"""

import argparse
import sys
import time

from config import SPEC
from robot.session import RobotSession
from robstride.protocol import CommunicationType
from utils.exceptions import HardwareError

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
TOLERANCE_RAD = 0.05


def parse_args():
    p = argparse.ArgumentParser(description="Set actuator mechanical zero.")
    g = p.add_mutually_exclusive_group()
    g.add_argument('--leg', choices=SPEC.leg_names,
                   help="Zero only this leg's joints.")
    g.add_argument('--joint', choices=SPEC.names, action='append',
                   help="Zero only this joint (repeatable).")
    p.add_argument('--yes', action='store_true', help="Skip the confirmation prompt.")
    p.add_argument('--force', action='store_true',
                   help="Allow joints whose zero reference is not sim zero.")
    return p.parse_args()


def selected_joints(args):
    if args.leg:
        return SPEC.joints_in_leg(args.leg)
    if args.joint:
        return [SPEC.joint(n) for n in args.joint]
    return list(SPEC.joints)


def main():
    args = parse_args()
    targets = selected_joints(args)
    offset = [j.name for j in targets if j.sim_offset != 0.0]
    if offset and not args.force:
        print(f"[ERROR] {offset} are zeroed against hard stops, not where they are "
              f"now. Use tools/calibrate_zeros.py (or --force if you really mean it).")
        sys.exit(1)

    print("--- RobStride RS03 Zero Point Setter ---")
    print("WARNING: this overwrites the internal mechanical zero position.")
    print("Ensure the following joints are held still at the desired zero:")
    for j in targets:
        print(f"  - {j.name}  ({j.bus}, id {j.can_id})")

    if not args.yes and input("\nType 'YES' to proceed: ") != "YES":
        print("Aborting.")
        sys.exit(0)

    try:
        with RobotSession(SPEC, use_imu=False, realtime=False,
                          limp_only=True, raw_frame=True) as robot:
            print("\n[INFO] Transmitting SET_ZERO_POSITION commands...")
            for joint in targets:
                print(f"  -> {joint.name} ({joint.bus}, id {joint.can_id})...")
                robot.network.buses[joint.bus].transmit(
                    comm_type=CommunicationType.SET_ZERO_POSITION,
                    extra_data=SPEC.host_id,
                    destination_id=joint.can_id,
                    data=b'\x01\x00\x00\x00\x00\x00\x00\x00',
                )
                # Let the motor MCU process the command and flash it.
                time.sleep(0.5)

            print("\n[INFO] Verifying new mechanical zero positions...")
            state = robot.read_limp_state(timeout=0.05)

            all_zeroed = True
            for joint in targets:
                pos = state[joint.name]['pos']
                if abs(pos) <= TOLERANCE_RAD:
                    print(f"  {GREEN}[SUCCESS]{RESET} {joint.name:<14}{pos:>9.4f} rad")
                else:
                    print(f"  {RED}[FAIL]{RESET}    {joint.name:<14}{pos:>9.4f} rad "
                          f"(exceeds {TOLERANCE_RAD} rad tolerance)")
                    all_zeroed = False

            if all_zeroed:
                print(f"\n{GREEN}[DONE] All selected motors zeroed and verified.{RESET}")
                print("Power cycle the motors to confirm the calibration persisted.")
            else:
                print(f"\n{RED}[ERROR] One or more motors failed to verify.{RESET}")
                sys.exit(1)

    except HardwareError as e:
        print(f"\n{RED}[CRITICAL] Hardware failure: {e}{RESET}")
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
