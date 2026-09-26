"""Sim-to-sim: run a walking policy through THIS stack's policy code in MuJoCo.

The policy, its observation builder, the chip-frame IMU conversion, the action
transform and the command clamp are the ones main-rl.py uses on the robot; only
the robot is simulated. The model comes from diogenes_mjlab:

    # in diogenes_mjlab (training env)
    uv run python src/diogenes_mjlab/tools/export_walk_model.py /tmp/harold_walk.mjb
    # here, with mujoco + onnxruntime
    python tools/sim2sim_walk.py policy.onnx /tmp/harold_walk.mjb

The simulated BNO085 is read like the real one: gyro and gravity in the chip's
own axes at its site, turned into the base frame through config's mount
rotation (which Policy turns back). The motors are the training task's RS03
model: PD with the robot's gains, torque-speed limited at the nominal battery
voltage. No noise, delay or DR: a policy that fails here fails for a reason in
the deployment code or the policy itself, not in the sim-to-real gap.

A scripted command sequence runs (--script to change it); each segment reports
the speed and turn rate achieved after a settling second.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import POLICY_PROFILES, SPEC  # noqa: E402
from control.policy import Policy  # noqa: E402
from sensors.base import ImuSample  # noqa: E402

# (seconds, vx m/s, wz rad/s)
DEFAULT_SCRIPT = [
    (4.0, 0.0, 0.0),
    (5.0, 0.15, 0.0),
    (5.0, 0.3, 0.0),
    (5.0, 0.15, 0.4),
    (5.0, 0.15, -0.4),
    (4.0, -0.1, 0.0),
    (3.0, 0.0, 0.0),
]
FALL_TILT = np.radians(45.0)
FALL_HEIGHT = 0.25


def parse_script(text):
    segs = []
    for part in text.split(';'):
        d, vx, wz = (float(v) for v in part.split(','))
        segs.append((d, vx, wz))
    return segs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=Path)
    p.add_argument("model", type=Path, help=".mjb from export_walk_model.py")
    p.add_argument("--script", type=parse_script, default=DEFAULT_SCRIPT,
                   help='"secs,vx,wz;secs,vx,wz;..."')
    p.add_argument("--save", type=Path, default=None,
                   help="Save the trajectory (time, qpos, command) as .npz")
    args = p.parse_args()

    info = json.loads(args.model.with_suffix(".json").read_text())
    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(info["keyframe"]).id)
    mujoco.mj_forward(model, data)

    names = SPEC.names
    assert names == info["joint_names"], (names, info["joint_names"])
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    act = np.array([model.actuator(n).id for n in names])
    site = model.site(info["imu_site"]).id
    mount = np.asarray(SPEC.imu.mount_rotation, dtype=np.float64)  # base <- chip
    kp, kd = info["kp"], info["kd"]
    peak, sat, w0 = info["peak_torque"], info["saturation_torque"], info["no_load_speed"]
    substeps = int(round(SPEC.dt / model.opt.timestep))

    policy = Policy(SPEC, str(args.policy), POLICY_PROFILES, clock=lambda: data.time)
    if not policy.profile.uses_command:
        sys.exit(f"{policy.profile.task_id} is not a walking policy.")

    def imu_sample():
        r_site = data.site_xmat[site].reshape(3, 3)          # world <- chip
        g_chip = r_site.T @ np.array([0.0, 0.0, -1.0])
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, vel, 1)
        return ImuSample(t=data.time, quat=data.qpos[3:7].copy(), ang_vel=mount @ vel[:3],
                         lin_accel=np.zeros(3), projected_gravity=mount @ g_chip)

    def torque(target):
        q, qd = data.qpos[qadr], data.qvel[dadr]
        tau = kp * (target - q) - kd * qd
        hi = np.minimum(peak, sat * (1.0 - qd / w0))
        lo = np.maximum(-peak, sat * (-1.0 - qd / w0))
        return np.clip(tau, lo, hi)

    policy.reset()
    log_t, log_q, log_cmd = [], [], []
    fell = None
    results = []
    peak_tau = np.zeros(len(names))
    for dur, vx, wz in args.script:
        policy.set_command(vx, 0.0, wz)
        seg_start = data.time
        vel_sum, yaw_sum, n = 0.0, 0.0, 0
        while data.time < seg_start + dur - 1e-9:
            target = policy.act(data.qpos[qadr].copy(), data.qvel[dadr].copy(), imu_sample())
            for _ in range(substeps):
                tau = torque(target)
                peak_tau = np.maximum(peak_tau, np.abs(tau))
                data.ctrl[act] = tau
                mujoco.mj_step(model, data)
            # Base state in the heading frame.
            w, x, y, z = data.qpos[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            v = data.qvel[0:3]
            v_fwd = np.cos(yaw) * v[0] + np.sin(yaw) * v[1]
            up_z = 1 - 2 * (x * x + y * y)                  # world z of the base z axis
            tilt = np.arccos(np.clip(up_z, -1, 1))
            log_t.append(data.time)
            log_q.append(data.qpos.copy())
            log_cmd.append((vx, wz))
            if data.time - seg_start > 1.0:
                vel_sum += v_fwd
                yaw_sum += data.qvel[5]
                n += 1
            if tilt > FALL_TILT or data.qpos[2] < FALL_HEIGHT:
                fell = (data.time, np.degrees(tilt), data.qpos[2])
                break
        results.append((dur, vx, wz, vel_sum / max(n, 1), yaw_sum / max(n, 1)))
        if fell:
            break

    print(f"\n{'segment':>22} {'achieved vx':>12} {'achieved wz':>12}")
    for dur, vx, wz, got_vx, got_wz in results:
        print(f"  {dur:4.1f} s vx {vx:+.2f} wz {wz:+.2f}  {got_vx:+11.3f}  {got_wz:+11.3f}")
    print(f"peak |torque| per joint (N.m): {np.round(peak_tau, 1).tolist()}")
    if fell:
        print(f"[FAIL] fell at t={fell[0]:.2f} s (tilt {fell[1]:.0f} deg, height {fell[2]:.3f} m)")
    else:
        print(f"[OK] walked the whole script ({data.time:.1f} s) without falling")
    if args.save:
        np.savez(args.save, t=np.array(log_t), qpos=np.array(log_q), command=np.array(log_cmd),
                 joint_names=np.array(names))
        print(f"[OK] trajectory saved to {args.save}")
    sys.exit(1 if fell else 0)


if __name__ == "__main__":
    main()
