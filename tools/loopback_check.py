"""End-to-end smoke test of the control stack with no hardware attached.

Runs the real Robot, RobstrideNetwork, SafetyMonitor, ObservationBuilder,
Recorder and LoopPacer against tools/fake_motors.py over python-can's
in-process 'virtual' backend. Catches the failure modes that matter when
rewiring: joints mapped to the wrong channel, replies attributed to the wrong
joint, a gather that silently drops a bus, and observation terms whose widths
have drifted from the model.

    python tools/loopback_check.py [--cycles 100]
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from control.observation import ObsContext, ObservationBuilder
from control.policy import HoldPolicy
from robot.robot import Robot
from sensors.base import NullImu
from tools.fake_motors import FakeRobot
from utils.exceptions import SafetyLimitError
from utils.realtime import LoopPacer
from utils.recorder import Recorder
from utils.safety import SafetyMonitor

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


def virtual_spec():
    """config.SPEC, rebound to the in-process virtual CAN backend."""
    buses = [dataclasses.replace(b, interface="virtual") for b in config.SPEC.buses]
    return dataclasses.replace(config.SPEC, buses=buses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cycles', type=int, default=100)
    args = ap.parse_args()

    spec = virtual_spec()
    print(spec.describe())

    sim = FakeRobot(spec)
    sim.start()

    robot = Robot(spec, imu=NullImu())
    safety = SafetyMonitor(spec)
    recorder = Recorder(spec)
    policy = HoldPolicy(spec)
    pacer = LoopPacer(spec.dt)

    exchange_ms = []
    try:
        robot.start(control_mode='MIT', verify_watchdogs=True)

        print("\n--- Transport ---")
        state = robot.read_limp_state(timeout=0.05)
        check("every joint reported", set(state) == set(spec.names),
              f"{len(state)}/{spec.num_joints} joints")
        check("both channels answered",
              len({j.bus for j in spec.joints if j.name in state}) == len(spec.buses))
        safety.verify_measured_state(state)
        check("initial state within safety limits", True)

        print("\n--- Joint identity ---")
        # Drive one joint to a distinctive angle and confirm only that joint
        # moves. This is the check that catches a cross-wired bus or an id
        # attributed to the wrong name.
        probe = spec.names[-1]           # right_calf: last joint, second bus
        probe_hw = spec.joint(probe).pos_limits[0] * 0.5
        targets = robot.targets_from_array(np.zeros(spec.num_joints))
        targets[probe] = {'pos': probe_hw, 'vel': 0.0, 'torque': 0.0}
        # Paced at the control rate so the simulated first-order actuator has
        # real time to converge on its setpoint.
        for _ in range(60):
            state = robot.exchange(targets, kp=60.0, kd=4.0, timeout=0.05)
            time.sleep(spec.dt)
        moved = [n for n in spec.names if abs(state[n]['pos']) > 1e-3]
        check(f"only '{probe}' moved", moved == [probe], f"moved: {moved}")
        check(f"'{probe}' tracked its setpoint",
              abs(state[probe]['pos'] - probe_hw) < 0.05,
              f"{state[probe]['pos']:.4f} vs {probe_hw:.4f}")

        print("\n--- Observation layout ---")
        builder = ObservationBuilder(spec, config.OBSERVATION_TERMS)
        sim_pos, sim_vel = robot.to_sim_frame(*robot.state_arrays(state))
        obs = builder.build(ObsContext(
            sim_pos=sim_pos, sim_vel=sim_vel,
            default_pos=spec.default_pos_vector(),
            last_raw_action=np.zeros(spec.num_joints, dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            projected_gravity=np.array([0, 0, -1], dtype=np.float32),
            phase=np.array([0.0, 1.0], dtype=np.float32),
            commands=np.zeros(3, dtype=np.float32),
        ))
        check("observation width matches declared layout",
              obs.shape == (1, builder.total_width), f"shape {obs.shape}")
        check("observation is finite", bool(np.all(np.isfinite(obs))))
        check("joint_pos_rel block reflects the probed joint",
              abs(obs[0, spec.index_of(probe)] -
                  (probe_hw * spec.joint(probe).direction)) < 0.05)

        print("\n--- Safety interlocks ---")
        try:
            bad = np.zeros(spec.num_joints)
            bad[0] = 99.0
            safety.validate_commanded_targets(bad)
            check("rogue command rejected", False, "no exception raised")
        except SafetyLimitError:
            check("rogue command rejected", True)
        try:
            safety.validate_commanded_targets(np.full(spec.num_joints, np.nan))
            check("non-finite command rejected", False, "no exception raised")
        except SafetyLimitError:
            check("non-finite command rejected", True)
        try:
            safety.verify_measured_state({n: state[n] for n in spec.names[:-1]})
            check("missing joint rejected", False, "no exception raised")
        except SafetyLimitError:
            check("missing joint rejected", True)

        print(f"\n--- Control loop ({args.cycles} cycles @ {spec.loop_rate_hz:g} Hz) ---")
        targets = robot.hold_targets(state)
        policy.reset()
        t0 = time.perf_counter()
        pacer.start()
        overrun_s = 0.0
        for _ in range(args.cycles):
            state = robot.exchange(targets, timeout=0.05)
            robot.check_faults()
            safety.verify_measured_state(state)
            imu_sample = robot.read_imu()
            safety.verify_imu_sample(imu_sample)

            sim_pos, sim_vel = robot.to_sim_frame(*robot.state_arrays(state))
            hw = robot.to_hardware_frame(policy.act(sim_pos, sim_vel, imu_sample))
            safety.validate_commanded_targets(hw)
            targets = robot.targets_from_array(hw)
            robot.send(targets)

            exchange_ms.append(robot.network.last_exchange_s * 1000.0)
            recorder.record(t=time.perf_counter() - t0, state=state, targets=targets,
                            imu_sample=imu_sample,
                            exchange_s=robot.network.last_exchange_s,
                            overrun_s=overrun_s)
            overrun_s = pacer.sleep()

        elapsed = time.perf_counter() - t0
        rate = args.cycles / elapsed
        e = np.array(exchange_ms)
        check("no gather timeouts", robot.network.timeout_count == 0,
              f"{robot.network.timeout_count} timeouts")
        check("loop held its rate", abs(rate - spec.loop_rate_hz) < 1.0,
              f"{rate:.2f} Hz over {elapsed:.2f} s")
        check("recorder captured every cycle", len(recorder.rows) == args.cycles)
        check("recorder schema is complete",
              set(recorder.rows[0]) == set(recorder.headers))
        print(f"      exchange: mean {e.mean():.2f} ms, p99 "
              f"{np.percentile(e, 99):.2f} ms, max {e.max():.2f} ms")
        print(f"      pacer: {pacer.summary()}")
        print(f"      simulated frames handled: {sim.frames_seen}")

    finally:
        robot.shutdown()
        sim.stop()

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
