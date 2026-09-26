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


class WalkSim:
    """The walking robot in MuJoCo, read and driven like the real one: joint
    state in policy order, a BNO085 sample in the base frame (through the mount
    rotation, as the real reader gives it), and RS03 PD targets per 20 ms."""

    def __init__(self, model_path: Path):
        self.info = json.loads(model_path.with_suffix(".json").read_text())
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        m, info = self.model, self.info
        names = SPEC.names
        assert names == info["joint_names"], (names, info["joint_names"])
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        self.act = np.array([m.actuator(n).id for n in names])
        self.site = m.site(info["imu_site"]).id
        self.mount = np.asarray(SPEC.imu.mount_rotation, dtype=np.float64)  # base <- chip
        self.substeps = int(round(SPEC.dt / m.opt.timestep))
        self.crouch = np.array([info["crouch"][n] for n in names])
        self.peak_tau = np.zeros(len(names))
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key(self.info["keyframe"]).id)
        mujoco.mj_forward(self.model, self.data)
        self.home = self.data.qpos[:7].copy()

    def joints(self):
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.dadr].copy()

    def imu_sample(self) -> ImuSample:
        d = self.data
        r_site = d.site_xmat[self.site].reshape(3, 3)           # world <- chip
        g_chip = r_site.T @ np.array([0.0, 0.0, -1.0])
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_SITE, self.site, vel, 1)
        return ImuSample(t=d.time, quat=d.qpos[3:7].copy(), ang_vel=self.mount @ vel[:3],
                         lin_accel=np.zeros(3), projected_gravity=self.mount @ g_chip)

    def _torque(self, target):
        info, d = self.info, self.data
        q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
        tau = info["kp"] * (target - q) - info["kd"] * qd
        sat, w0, peak = info["saturation_torque"], info["no_load_speed"], info["peak_torque"]
        hi = np.minimum(peak, sat * (1.0 - qd / w0))
        lo = np.maximum(-peak, sat * (-1.0 - qd / w0))
        return np.clip(tau, lo, hi)

    def step(self, target=None, hold_base=False):
        """One 20 ms control period. target=None leaves the motors limp;
        hold_base pins the torso where it started (the rope)."""
        for _ in range(self.substeps):
            if target is None:
                self.data.ctrl[self.act] = 0.0
            else:
                tau = self._torque(target)
                self.peak_tau = np.maximum(self.peak_tau, np.abs(tau))
                self.data.ctrl[self.act] = tau
            if hold_base:
                self.data.qpos[:7] = self.home
                self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)

    def base_state(self):
        """(forward speed in the heading frame m/s, yaw rate rad/s, tilt rad, height m)."""
        d = self.data
        w, x, y, z = d.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = d.qvel[0:3]
        v_fwd = np.cos(yaw) * v[0] + np.sin(yaw) * v[1]
        tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
        return v_fwd, d.qvel[5], tilt, d.qpos[2]

    def fallen(self):
        _, _, tilt, height = self.base_state()
        return tilt > FALL_TILT or height < FALL_HEIGHT


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=Path)
    p.add_argument("model", type=Path, help=".mjb from export_walk_model.py")
    p.add_argument("--script", type=parse_script, default=DEFAULT_SCRIPT,
                   help='"secs,vx,wz;secs,vx,wz;..."')
    p.add_argument("--save", type=Path, default=None,
                   help="Save the trajectory (time, qpos, command) as .npz")
    args = p.parse_args()

    sim = WalkSim(args.model)
    policy = Policy(SPEC, str(args.policy), POLICY_PROFILES, clock=lambda: sim.data.time)
    if not policy.profile.uses_command:
        sys.exit(f"{policy.profile.task_id} is not a walking policy.")

    policy.reset()
    log_t, log_q, log_cmd = [], [], []
    fell = None
    results = []
    for dur, vx, wz in args.script:
        policy.set_command(vx, 0.0, wz)
        seg_start = sim.data.time
        vel_sum, yaw_sum, n = 0.0, 0.0, 0
        while sim.data.time < seg_start + dur - 1e-9:
            target = policy.act(*sim.joints(), sim.imu_sample())
            sim.step(target)
            v_fwd, yaw_rate, tilt, height = sim.base_state()
            log_t.append(sim.data.time)
            log_q.append(sim.data.qpos.copy())
            log_cmd.append((vx, wz))
            if sim.data.time - seg_start > 1.0:
                vel_sum += v_fwd
                yaw_sum += yaw_rate
                n += 1
            if sim.fallen():
                fell = (sim.data.time, np.degrees(tilt), height)
                break
        results.append((dur, vx, wz, vel_sum / max(n, 1), yaw_sum / max(n, 1)))
        if fell:
            break

    print(f"\n{'segment':>22} {'achieved vx':>12} {'achieved wz':>12}")
    for dur, vx, wz, got_vx, got_wz in results:
        print(f"  {dur:4.1f} s vx {vx:+.2f} wz {wz:+.2f}  {got_vx:+11.3f}  {got_wz:+11.3f}")
    print(f"peak |torque| per joint (N.m): {np.round(sim.peak_tau, 1).tolist()}")
    if fell:
        print(f"[FAIL] fell at t={fell[0]:.2f} s (tilt {fell[1]:.0f} deg, height {fell[2]:.3f} m)")
    else:
        print(f"[OK] walked the whole script ({sim.data.time:.1f} s) without falling")
    if args.save:
        np.savez(args.save, t=np.array(log_t), qpos=np.array(log_q), command=np.array(log_cmd),
                 joint_names=np.array(SPEC.names))
        print(f"[OK] trajectory saved to {args.save}")
    sys.exit(1 if fell else 0)


if __name__ == "__main__":
    main()
