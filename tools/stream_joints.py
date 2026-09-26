"""Stream limp joint state to a laptop over UDP for live visualisation.

Bring-up tool for checking the joint mapping. Nothing moves: every exchange uses
kp = kd = 0 exactly like main-read-state.py, so the actuators are already limp
and stay limp if this process dies, even on motors whose firmware lacks the
CAN-timeout parameter.

Each packet carries both the raw hardware reading and the SIM-frame value the
policy would see (hardware * direction). The viewer poses the MuJoCo model from
the sim values, so moving a joint by hand and watching the model checks the
whole chain: bus -> CAN id -> joint name -> direction sign -> sim joint.

    source setup.sh
    python tools/stream_joints.py --host 192.168.0.10
    python tools/stream_joints.py --host 192.168.0.10 --imu     # + base orientation
    python tools/stream_joints.py --host 127.0.0.1 --virtual   # no hardware

With --imu each packet also carries the base orientation from the BNO085,
through config.IMU.mount_rotation (base frame: +x forward, +y left, +z up), so
the viewer can tilt the model to match and show whether the mount is right.

Pair with diogenes_mjlab/src/diogenes_mjlab/tools/live_joint_viewer.py.
"""

import argparse
import dataclasses
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from config import SPEC  # noqa: E402
from robot.session import RobotSession  # noqa: E402
from sensors.base import quat_to_rotation_matrix  # noqa: E402
from utils.exceptions import HardwareError  # noqa: E402

DEFAULT_PORT = 9870


def parse_args():
    p = argparse.ArgumentParser(description="Stream limp joint state over UDP.")
    p.add_argument('--host', required=True, help="Laptop IP running the viewer.")
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--rate', type=float, default=50.0, help="Stream rate in Hz.")
    p.add_argument('--imu', action='store_true',
                   help="Also stream the IMU's base orientation, gravity and gyro.")
    p.add_argument('--virtual', action='store_true',
                   help="Use tools/fake_motors.py instead of hardware. The fake "
                        "joints drift slowly so the viewer has something to show.")
    return p.parse_args()


def build_spec(virtual: bool):
    if not virtual:
        return SPEC, None
    buses = [dataclasses.replace(b, interface="virtual") for b in SPEC.buses]
    spec = dataclasses.replace(SPEC, buses=buses)
    from tools.fake_motors import FakeRobot
    return spec, FakeRobot(spec)


def main():
    args = parse_args()
    spec, simulator = build_spec(args.virtual)
    period = 1.0 / args.rate
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)

    if simulator is not None:
        simulator.start()
    try:
        use_imu = args.imu and simulator is None
        mount = spec.imu.mount_matrix if use_imu else None
        with RobotSession(spec, use_imu=use_imu, realtime=False,
                          limp_only=True) as robot:
            print(f"[INFO] Motors enabled and limp. Streaming to {dest[0]}:{dest[1]} "
                  f"at {args.rate:.0f} Hz. Ctrl-C to stop.")
            seq = 0
            next_t = time.perf_counter()
            last_print = 0.0
            while True:
                if simulator is not None:
                    # Give the fake joints something to do: each one swings in turn.
                    t = time.monotonic()
                    for b, bus in enumerate(simulator.buses):
                        for m in bus.motors.values():
                            m.pos = 0.4 * math.sin(0.5 * t + m.can_id + 3 * b)

                state = robot.read_limp_state(timeout=0.02)
                hw_pos, vel = robot.state_arrays(state)
                sim_pos, sim_vel = robot.to_sim_frame(hw_pos, vel)

                joints = {}
                for i, j in enumerate(spec.joints):
                    r = state[j.name]
                    joints[j.name] = {
                        'bus': j.bus, 'id': j.can_id,
                        'hw': float(hw_pos[i]), 'sim': float(sim_pos[i]),
                        'vel': float(sim_vel[i]),
                        'torque': float(r['torque']), 'temp': float(r['temp']),
                    }
                packet = {'seq': seq, 't': time.time(), 'joints': joints}
                if use_imu:
                    s = robot.read_imu()
                    # sample.quat is the CHIP's orientation in the IMU's world
                    # frame; v_base = M v_chip, so R_world_base = R_world_chip M^T.
                    r_wb = quat_to_rotation_matrix(np.asarray(s.quat)) @ mount.T
                    packet['imu'] = {
                        'R_world_base': r_wb.ravel().tolist(),
                        'gravity': [float(v) for v in s.projected_gravity],
                        'gyro': [float(v) for v in s.ang_vel],
                        'tilt_deg': float(np.degrees(s.tilt_rad)),
                        'status': int(s.status),
                        'age_ms': float(s.age() * 1000.0),
                    }
                sock.sendto(json.dumps(packet).encode(), dest)
                seq += 1

                now = time.perf_counter()
                if now - last_print > 1.0:
                    last_print = now
                    print("  " + "  ".join(f"{n}={v['sim']:+.2f}"
                                           for n, v in joints.items()), flush=True)

                next_t += period
                time.sleep(max(0.0, next_t - time.perf_counter()))
                if time.perf_counter() - next_t > 0.5:
                    next_t = time.perf_counter()   # fell far behind; don't burst

    except HardwareError as e:
        print(f"\n[CRITICAL] Hardware failure: {e}")
        sys.exit(3)
    finally:
        if simulator is not None:
            simulator.stop()


if __name__ == "__main__":
    main()
