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
from utils.exceptions import (
    ActuatorFault, DirectionsUnverifiedError, HardwareIOError, MissedReplies,
    MotorDisabled, SafetyLimitError, WatchdogUnverifiedError,
)
from utils.realtime import LoopPacer
from utils.recorder import Recorder
from utils.safety import SafetyMonitor

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f" -- {detail}" if detail else ""))


def virtual_spec():
    """config.SPEC, rebound to the in-process virtual CAN backend.

    Marked zero-offset-verified: this tests the stack, not the robot's
    calibration, which Robot.start gates separately on real hardware.
    """
    buses = [dataclasses.replace(b, interface="virtual") for b in config.SPEC.buses]
    return dataclasses.replace(config.SPEC, buses=buses, zero_offsets_verified=True)


# Harold's three RS03s whose firmware lacks the CAN-timeout register.
LEGACY_MOTORS = {("can0", 1), ("can0", 2), ("can1", 3)}


def watchdog_checks(spec):
    """Robot.start's watchdog gate, against fresh fake buses per scenario."""
    print("\n--- Watchdog gate ---")
    scenarios = [
        ("all motors have the register", (), None, True),
        ("legacy motors that do time out", LEGACY_MOTORS, 0.1, True),
        ("legacy motors that never time out", LEGACY_MOTORS, None, False),
    ]
    for label, legacy, timeout_s, should_start in scenarios:
        sim = FakeRobot(spec, legacy_motors=legacy, legacy_timeout_s=timeout_s)
        sim.start()
        robot = Robot(spec, imu=NullImu())
        enabled_at_refusal = None
        try:
            robot.start(control_mode='MIT', require_watchdogs=True)
            started = True
        except WatchdogUnverifiedError:
            started = False
            enabled_at_refusal = [
                (b.channel, m.can_id) for b in sim.buses
                for m in b.motors.values() if m.enabled
            ]
        finally:
            robot.shutdown()
            sim.stop()
        outcome = "started" if started else "refused"
        check(f"{label}: {outcome}", started == should_start)
        if not started:
            check(f"{label}: nothing left enabled on refusal",
                  enabled_at_refusal == [], f"enabled: {enabled_at_refusal}")


def obs_context(spec, sim_pos, sim_vel, phase=(0.0, 1.0)):
    n = spec.num_joints
    return ObsContext(
        sim_pos=np.asarray(sim_pos, dtype=np.float32),
        sim_vel=np.asarray(sim_vel, dtype=np.float32),
        default_pos=spec.default_pos_vector(),
        last_raw_action=np.zeros(n, dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        projected_gravity=np.array([0, 0, -1], dtype=np.float32),
        imu_ang_vel_chip=np.zeros(3, dtype=np.float32),
        imu_gravity_chip=np.array([0, -1, 0], dtype=np.float32),
        phase=np.asarray(phase, dtype=np.float32),
        commands=np.zeros(3, dtype=np.float32),
    )


def history_checks(spec):
    """The history must match mjlab: term-major, oldest first, backfilled."""
    n, H = spec.num_joints, 3
    b = ObservationBuilder(spec, ["joint_pos_rel", "phase_clock"], H)
    step = lambda v: b.build(obs_context(spec, np.full(n, v), np.zeros(n),
                                         phase=(v, -v))).copy()[0]
    first = step(1.0)
    check("history backfilled from the first observation",
          np.allclose(first[:H * n], 1.0) and np.allclose(first[H * n:], [1.0, -1.0] * H))
    step(2.0)
    third = step(3.0)
    pos = third[:H * n].reshape(H, n)[:, 0]
    clock = third[H * n:].reshape(H, 2)[:, 0]
    check("history is term-major, oldest first",
          pos.tolist() == [1.0, 2.0, 3.0] and clock.tolist() == [1.0, 2.0, 3.0],
          f"pos {pos.tolist()}, clock {clock.tolist()}")
    b.reset()
    check("reset refills the history", np.allclose(step(7.0)[:H * n], 7.0))


SUSPENDED = config.POLICY_PROFILES["Diogenes-Biped-Suspended"]
WALK = config.POLICY_PROFILES["Diogenes-Biped-Walk"]


def policy_checks(spec):
    """Policy against stand-in ONNX files: profile selection, metadata gate,
    action transform, command clamp and the chip-frame IMU terms."""
    print("\n--- Policy wiring ---")
    try:
        import onnx
        from onnx import TensorProto, helper
        import onnxruntime  # noqa: F401
    except ImportError as e:
        print(f"  [SKIP] needs onnx + onnxruntime ({e.name} missing)")
        return
    import tempfile
    from control.policy import Policy
    from sensors.base import ImuSample

    n = spec.num_joints
    # Constant output of 5 rad per joint: past every ctrl range.
    bias = np.full(n, 5.0, dtype=np.float32)

    def make_model(meta, width, out=bias):
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["obs", "W", "b"], ["actions"])],
            "standin",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n])],
            [helper.make_tensor("W", TensorProto.FLOAT, [width, n], [0.0] * (width * n)),
             helper.make_tensor("b", TensorProto.FLOAT, [n], np.asarray(out).tolist())],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        for k, v in meta.items():
            model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
        onnx.save(model, path)
        return path

    def meta_for(profile, **extra):
        meta = {
            'task_id': profile.task_id,
            'joint_names': ",".join(spec.names),
            'default_joint_pos': ",".join(
                f"{profile.default_pos[j]:.5f}" for j in spec.names),
            'action_scale': "1.0",
            'observation_names': ",".join(profile.sim_observation_names),
            'actor_history_length': str(profile.history_length),
            'phase_period': "2.0",
            'step_dt': str(spec.dt),
            'joint_stiffness': ",".join(f"{spec.kp:.3f}" for _ in range(n)),
            'joint_damping': ",".join(f"{spec.kd:.3f}" for _ in range(n)),
        }
        meta.update(extra)
        return {k: v for k, v in meta.items() if v is not None}

    def load(meta, width, out=bias):
        return Policy(spec, make_model(meta, width, out), config.POLICY_PROFILES)

    def refused(meta, width):
        meta = {k: v for k, v in meta.items() if v is not None}
        try:
            load(meta, width)
            return False
        except SystemExit:
            return True

    # Suspended: 10 x (6 + 6 + 6 + 2).
    s_width = max(1, SUSPENDED.history_length) * (3 * n + 2)
    policy = load(meta_for(SUSPENDED), s_width)
    check("suspended export accepted, its profile picked",
          policy.profile is SUSPENDED and policy.builder.total_width == s_width)
    targets = policy.act(np.zeros(n), np.zeros(n))
    check("targets clamped to the sim ctrl range",
          np.allclose(targets, policy.ctrl_hi), f"{np.round(targets, 3).tolist()}")
    check("last_action keeps the raw, unclamped output",
          np.allclose(policy.last_raw_action, bias))

    swapped = list(spec.names)
    swapped[0], swapped[3] = swapped[3], swapped[0]
    for label, change in [
        ("swapped joint order", {'joint_names': ",".join(swapped)}),
        ("wrong history length", {'actor_history_length': "5"}),
        ("wrong observation terms", {'observation_names': "joint_pos,joint_vel"}),
        ("missing metadata", {'phase_period': None}),
        ("unknown task", {'task_id': "Diogenes-Biped-Somersault"}),
    ]:
        check(f"{label} refused", refused(meta_for(SUSPENDED, **change), s_width))

    # Walking: 10 x (3 + 3 + 6 + 6 + 6 + 3 + 2) = 290, crouch default, per-joint
    # scale, command ranges.
    w_width = max(1, WALK.history_length) * (3 + 3 + 3 * n + 3 + 2)
    scale = [0.25, 0.5, 0.5, 0.25, 0.5, 0.5]
    walk_meta = meta_for(WALK, action_scale=",".join(map(str, scale)),
                         phase_period="0.714286",
                         command_ranges="-0.1,0.3,0.0,0.0,-0.5,0.5")
    small = np.array([0.1, 0.2, -0.2, -0.1, -0.2, -0.2], dtype=np.float32)
    policy = load(walk_meta, w_width, out=small)
    check("walking export accepted, its profile picked",
          policy.profile is WALK and policy.builder.total_width == w_width)
    crouch = WALK.pose_vector(WALK.default_pos, spec.names)
    targets = policy.act(crouch, np.zeros(n))
    check("targets = crouch + per-joint scale x action",
          np.allclose(targets, crouch + np.array(scale) * small, atol=1e-6),
          f"{np.round(targets, 4).tolist()}")
    policy.set_command(0.8, 0.2, -1.0)
    check("command clamped to the trained ranges",
          np.allclose(policy.command, [0.3, 0.0, -0.5]), f"{policy.command.tolist()}")
    # Turning left while tipped nose down: base frame gyro +z, gravity +x.
    imu = ImuSample(t=0.0, quat=np.array([1.0, 0, 0, 0]), ang_vel=np.array([0.0, 0.0, 1.0]),
                    lin_accel=np.zeros(3), projected_gravity=np.array([0.5, 0.0, -0.866]))
    policy.reset()
    obs = policy.builder._buffer
    policy.act(crouch, np.zeros(n), imu)
    H = WALK.history_length
    gyro = obs[0, (H - 1) * 3:H * 3]
    grav = obs[0, H * 3 + (H - 1) * 3:H * 6]
    # BNO085 axes: +x left, +y up, +z forward.
    check("IMU terms in the chip frame",
          np.allclose(gyro, [0, 1, 0]) and np.allclose(grav, [0, -0.866, 0.5], atol=1e-6),
          f"gyro {gyro.tolist()}, gravity {np.round(grav, 3).tolist()}")
    for label, change in [
        ("walking without command ranges", {'command_ranges': None}),
        ("walking with the suspended default pose",
         {'default_joint_pos': ",".join("0.0" for _ in range(n))}),
    ]:
        check(f"{label} refused", refused({**walk_meta, **change}, w_width))


def teleop_checks():
    """The UDP command link: deadman, link timeout, START/STOP, sender lock."""
    print("\n--- Teleop link ---")
    import socket
    from control.command_source import (
        BUTTON_DEADMAN, BUTTON_START, BUTTON_STOP, UdpCommandSource, encode,
    )
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    src = UdpCommandSource(port=port, timeout_s=0.2)
    src.start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(seq, vx, wz, buttons):
        tx.sendto(encode(seq, vx, 0.0, wz, buttons), ("127.0.0.1", port))
        time.sleep(0.05)

    try:
        check("no packets: step in place", src.command() == (0.0, 0.0, 0.0))
        send(1, 0.2, 0.3, 0)
        check("deadman released: step in place", src.command() == (0.0, 0.0, 0.0))
        send(2, 0.2, 0.3, BUTTON_DEADMAN)
        vx, _, wz = src.command()
        check("deadman held: command passes", abs(vx - 0.2) < 1e-6 and abs(wz - 0.3) < 1e-6)
        send(1, 0.9, 0.9, BUTTON_DEADMAN)
        check("stale packet ignored", abs(src.command()[0] - 0.2) < 1e-6)
        time.sleep(0.25)
        check("link lost: step in place", src.command() == (0.0, 0.0, 0.0))
        check("START not yet given", not src.start_requested())
        send(3, 0.0, 0.0, BUTTON_START)
        send(4, 0.0, 0.0, 0)
        check("START latched", src.start_requested() and not src.stop_requested())
        send(5, 0.0, 0.0, BUTTON_STOP)
        check("STOP latched", src.stop_requested())
    finally:
        src.stop()
        tx.close()


def turn_checks(spec):
    """A joint that powers up a whole turn out is read and held correctly."""
    print("\n--- Whole-turn correction ---")
    joint = spec.joint("right_thigh")
    true_hw = joint.direction * (joint.default_pos - joint.sim_offset)
    raw = true_hw + 2 * np.pi
    sim = FakeRobot(spec, initial_pos={"right_thigh": raw})
    sim.start()
    robot = Robot(spec, imu=NullImu())
    try:
        robot.start(control_mode='MIT')
        state = robot.read_limp_state(timeout=0.05)
        check("reading corrected by one turn",
              abs(state["right_thigh"]['pos'] - true_hw) < 1e-3,
              f"{state['right_thigh']['pos']:+.3f} vs {true_hw:+.3f} rad")
        targets = robot.hold_targets(state)
        for _ in range(30):
            robot.exchange(targets, kp=60.0, kd=4.0, timeout=0.05)
            time.sleep(spec.dt)
        motor = sim.bus_for(joint.bus).motors[joint.can_id]
        check("holding it does not spin the motor a turn",
              abs(motor.pos - raw) < 1e-3, f"motor at {motor.pos:+.3f}, raw {raw:+.3f}")
    finally:
        robot.shutdown()
        sim.stop()


def offset_checks(spec):
    """Zero offsets: frame round trip and derived hardware limits."""
    print("\n--- Zero offsets ---")
    robot = Robot(spec, imu=NullImu())
    sim_pos = np.linspace(-0.5, 0.5, spec.num_joints).astype(np.float32)
    back, _ = robot.to_sim_frame(robot.to_hardware_frame(sim_pos),
                                 np.zeros(spec.num_joints, dtype=np.float32))
    check("sim -> hardware -> sim round trip", np.allclose(back, sim_pos, atol=1e-5))
    ok = all(lo <= j.direction * (j.default_pos - j.sim_offset) <= hi
             for j in spec.joints for lo, hi in [j.pos_limits])
    check("default pose maps inside every hardware limit", ok)
    within = all(abs(v) < np.pi for j in spec.joints for v in j.pos_limits)
    check("every hardware limit within +-pi of the motor zero", within,
          "needed for the whole-turn correction")


def calibration_checks():
    """tools/calibrate_zeros.py end to end against fake motors."""
    print("\n--- Zero calibration tool ---")
    import subprocess
    tool = Path(__file__).resolve().parent / "calibrate_zeros.py"
    result = subprocess.run([sys.executable, str(tool), "--virtual"],
                            input="\n" * 12, capture_output=True, text=True, timeout=120)
    zeroed = result.stdout.count("[OK]")
    check("runs to completion", result.returncode == 0,
          f"exit {result.returncode}" + ("" if result.returncode == 0 else
                                         f": {result.stdout[-300:]}{result.stderr[-300:]}"))
    check("all six joints zeroed", zeroed == 6, f"{zeroed} zeroed")


def reply_tolerance_checks(spec):
    """Isolated missed replies are ridden through; a run of them is not."""
    print("\n--- Missed replies ---")
    sim = FakeRobot(spec)
    sim.start()
    robot = Robot(spec, imu=NullImu())
    try:
        robot.start(control_mode='MIT')
        last = robot.read_limp_state(timeout=0.05)
        targets = robot.hold_targets(last)
        limit = config.MAX_CONSECUTIVE_MISSED_REPLIES

        sim.drop_replies("left_calf", 1)
        state, miss = robot.exchange_tolerant(targets, last, limit)
        check("one missed reply is ridden through",
              miss is not None and miss.missing == ["left_calf"]
              and state["left_calf"] is last["left_calf"]
              and all(state[n] is not last[n] for n in spec.names if n != "left_calf"),
              f"missing {miss.missing if miss else None}")
        state, miss = robot.exchange_tolerant(targets, state, limit)
        check("a good reply resets the count",
              miss is None and robot.consecutive_misses == 0)

        sim.drop_replies("left_calf", limit)
        raised = False
        try:
            for _ in range(limit):
                state, miss = robot.exchange_tolerant(targets, state, limit)
        except MissedReplies:
            raised = True
        check(f"{limit} misses in a row stop the run", raised,
              f"after {robot.consecutive_misses}")
        robot.consecutive_misses = 0
        robot.read_limp_state(timeout=0.05)

        sim.drop_replies("right_thigh", 1, as_fault=True)
        state, miss = robot.exchange_tolerant(targets, state, limit)
        check("a fault sent instead of a reply is reported",
              miss is not None and any("type 21" in s for s in miss.stray),
              f"{miss.stray if miss else None}")
        try:
            robot.check_faults()
            faulted = False
        except ActuatorFault:
            faulted = True
        check("... and still stops the run as a fault", faulted)

        sim.drop_replies("left_hip", 1)
        try:
            robot.exchange(targets)
            plain = False
        except MissedReplies as e:
            plain = isinstance(e, HardwareIOError)
        check("plain exchange still raises (limp tools unchanged)", plain)
    finally:
        robot.shutdown()
        sim.stop()


class SlowImu(NullImu):
    """An IMU whose start() blocks, like the real BNO085 coming up over I2C."""

    def start(self):
        time.sleep(0.3)


def enabled_mode_checks(spec):
    """Motors must still be running when the loop starts, and a motor that
    drops out must be noticed. Regression for 2026-09-25, when the IMU started
    after the motors were enabled, their CAN timeout fired during the pause,
    and the whole run commanded limp motors without anything noticing."""
    print("\n--- Motors stay enabled ---")
    everyone = {(j.bus, j.can_id) for j in spec.joints}
    sim = FakeRobot(spec, legacy_motors=everyone, legacy_timeout_s=0.1)
    sim.start()
    robot = Robot(spec, imu=SlowImu())
    try:
        robot.start(control_mode='MIT')
        try:
            state = robot.read_limp_state(timeout=0.05)
            running = True
        except MotorDisabled as e:
            running, state = False, None
        check("slow IMU start does not let the motors time out", running,
              "" if running else str(e)[:80])
        check("every reply reports run mode",
              state is not None and all(s['mode'] == 2 for s in state.values()))

        sim.motor("right_thigh").enabled = False
        try:
            robot.read_limp_state(timeout=0.05)
            caught = ""
        except MotorDisabled as e:
            caught = str(e)
        check("a motor that drops out is caught", "right_thigh" in caught,
              caught[:60])
    finally:
        robot.shutdown()
        sim.stop()


def soft_start_checks(spec):
    """The ramp from the resting pose to the start pose."""
    print("\n--- Soft start ---")
    from control.soft_start import SoftStart
    rest = np.radians([6, 13, -17, -8, -15, -23]).astype(np.float32)
    goal = SUSPENDED.pose_vector(SUSPENDED.start_pose, spec.names)
    ramp = SoftStart(rest, goal, config.SOFT_START_S, config.SOFT_START_HOLD_S)
    check("starts at the resting pose", np.allclose(ramp.target(0.0), rest))
    check("reaches the start pose and holds it",
          np.allclose(ramp.target(config.SOFT_START_S), goal) and
          np.allclose(ramp.target(config.SOFT_START_S + config.SOFT_START_HOLD_S * 0.9), goal))
    steps = [ramp.target(k * spec.dt) for k in range(int(config.SOFT_START_S / spec.dt) + 2)]
    worst = max(np.abs(b - a).max() for a, b in zip(steps, steps[1:]))
    check("no step larger than the ramp slope",
          worst <= np.abs(goal - rest).max() * spec.dt / config.SOFT_START_S + 1e-6,
          f"worst step {np.degrees(worst):.2f} deg")
    check("hands over only after ramp + hold",
          not ramp.done(config.SOFT_START_S) and
          ramp.done(config.SOFT_START_S + config.SOFT_START_HOLD_S))
    lo, hi = zip(*(j.sim_pos_limits for j in spec.joints))
    check("start pose inside every joint limit",
          bool(np.all(goal >= np.array(lo)) and np.all(goal <= np.array(hi))))


def contract_checks(spec):
    """The sim joint contract gate and the limits derived from it."""
    print("\n--- Sim joint contract ---")
    check("config directions verified against the shipped contract",
          spec.directions_verified,
          f"signature {spec.sim_contract['direction_signature']}")

    # A changed sim convention must stop Robot.start before any CAN traffic.
    changed = dict(spec.sim_contract, direction_signature="0" * 16)
    stale = dataclasses.replace(spec, sim_contract=changed)
    sim = FakeRobot(stale)
    sim.start()
    robot = Robot(stale, imu=NullImu())
    try:
        robot.start(control_mode='MIT')
        refused = False
    except DirectionsUnverifiedError:
        refused = True
    finally:
        robot.shutdown()
        sim.stop()
    check("changed sim convention refused", refused)
    check("refused before any CAN traffic", sim.frames_seen == 0,
          f"{sim.frames_seen} frames")

    # Limits must stay inside the sim range plus the margin.
    j = spec.joints[0]
    lo, hi = spec.sim_contract['joints'][j.name]['range']
    too_wide = dataclasses.replace(j, sim_pos_limits=(lo, hi + spec.sim_limit_margin + 0.1))
    try:
        dataclasses.replace(spec, joints=[too_wide, *spec.joints[1:]])
        rejected = False
    except ValueError:
        rejected = True
    check("limits beyond the sim range rejected", rejected)

    for jt in spec.joints:
        hw = tuple(sorted(jt.direction * (v - jt.sim_offset) for v in jt.sim_pos_limits))
        if hw != jt.pos_limits:
            check(f"{jt.name} hardware limits follow direction", False, f"{jt.pos_limits}")
            break
    else:
        check("hardware limits follow each joint's direction", True)


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
        lo, hi = spec.joint(probe).pos_limits
        probe_hw = lo + 0.25 * (hi - lo)
        initial = robot.read_limp_state(timeout=0.05)
        targets = robot.hold_targets(initial)
        targets[probe] = {'pos': probe_hw, 'vel': 0.0, 'torque': 0.0}
        # Paced at the control rate so the simulated first-order actuator has
        # real time to converge on its setpoint.
        for _ in range(60):
            state = robot.exchange(targets, kp=60.0, kd=4.0, timeout=0.05)
            time.sleep(spec.dt)
        moved = [n for n in spec.names
                 if abs(state[n]['pos'] - initial[n]['pos']) > 1e-3]
        check(f"only '{probe}' moved", moved == [probe], f"moved: {moved}")
        check(f"'{probe}' tracked its setpoint",
              abs(state[probe]['pos'] - probe_hw) < 0.05,
              f"{state[probe]['pos']:.4f} vs {probe_hw:.4f}")

        print("\n--- Observation layout ---")
        history = SUSPENDED.history_length
        builder = ObservationBuilder(spec, list(SUSPENDED.observation_terms), history)
        sim_pos, sim_vel = robot.to_sim_frame(*robot.state_arrays(state))
        obs = builder.build(obs_context(spec, sim_pos, sim_vel)).copy()
        n = spec.num_joints
        frame = 3 * n + 2
        check("observation width matches declared layout",
              obs.shape == (1, builder.total_width) and
              builder.total_width == max(1, history) * frame,
              f"shape {obs.shape}")
        check("observation is finite", bool(np.all(np.isfinite(obs))))
        newest = (max(1, history) - 1) * n + spec.index_of(probe)
        i = spec.index_of(probe)
        check("joint_pos_rel block reflects the probed joint",
              abs(obs[0, newest] - (sim_pos[i] - spec.default_pos_vector()[i])) < 1e-5
              and abs(sim_pos[i] - robot.to_sim_frame(
                  np.full(n, probe_hw), np.zeros(n))[0][i]) < 0.05)
        history_checks(spec)

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

    watchdog_checks(spec)
    contract_checks(spec)
    policy_checks(spec)
    teleop_checks()
    turn_checks(spec)
    offset_checks(spec)
    soft_start_checks(spec)
    reply_tolerance_checks(spec)
    enabled_mode_checks(spec)
    calibration_checks()

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
