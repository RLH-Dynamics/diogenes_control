"""Read-only actuator health check: run after a fall, a reconnect or a reboot.

Opens both buses without applying gain (limp), reads every joint several times
and reports, per joint: its position in the sim frame and whether that is inside
the joint's range, temperature, mode, the fault flags the motor sets in its
status replies (under-voltage, over-current, over-temperature, encoder faults,
not calibrated), fault frames, and how many reads it answered. Nothing moves.

A joint reading far outside its range after a fall usually means a wrong turn
count or a slipped zero; check it in the live viewer (tools/stream_joints.py)
before running a policy.

    source setup.sh
    python tools/health_check.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPEC  # noqa: E402
from robot.session import RobotSession  # noqa: E402
from robstride.bus import RobstrideBus  # noqa: E402
from utils.exceptions import ActuatorFault, HardwareError  # noqa: E402

READS = 20
HOT_C = 60.0


def main():
    reads, missing = [], {j.name: 0 for j in SPEC.joints}
    fault_frames = "none"
    with RobotSession(SPEC, use_imu=False, limp_only=True) as robot:
        for _ in range(READS):
            try:
                state = robot.read_limp_state()
                reads.append(state)
            except HardwareError as e:
                for name in getattr(e, "missing", []) or []:
                    missing[name] += 1
            time.sleep(0.05)
        try:
            robot.check_faults()
        except ActuatorFault as e:
            fault_frames = str(e)
        if not reads:
            sys.exit("[FAIL] no complete read of the joints")
        sim_pos, _ = robot.to_sim_frame(*robot.state_arrays(reads[-1]))

    print(f"\n{'joint':12s} {'sim deg':>8s} {'range deg':>15s} {'in range':>9s} "
          f"{'temp C':>7s} {'mode':>5s}  faults")
    ok = True
    for i, j in enumerate(SPEC.joints):
        last = reads[-1][j.name]
        lo, hi = np.degrees(j.sim_pos_limits)
        pos = np.degrees(sim_pos[i])
        flags = 0
        for r in reads:
            flags |= r[j.name].get("faults", 0)
        names = [RobstrideBus.STATUS_FAULT_BITS[b] for b in range(6) if flags >> b & 1]
        inside = lo <= pos <= hi
        hot = last["temp"] > HOT_C
        ok &= inside and not names and not hot and missing[j.name] == 0
        print(f"{j.name:12s} {pos:+8.1f} {lo:+6.1f}..{hi:+6.1f} {'yes' if inside else 'NO':>9s} "
              f"{last['temp']:7.1f}{' HOT' if hot else '    '} {last['mode']:3d}  "
              f"{', '.join(names) or 'none'}"
              + (f"  ({missing[j.name]} missed reads)" if missing[j.name] else ""))
    print(f"\ncomplete reads: {len(reads)}/{READS}; fault frames: {fault_frames}")
    ok &= fault_frames == "none" and len(reads) == READS
    print("[OK] all actuators healthy" if ok else "[CHECK] see the flagged joints above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
