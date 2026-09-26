"""Run the RL policy on hardware.

SINGLE-RATE LOOP
----------------
The policy and the actuator command stream both run at config.LOOP_RATE_HZ
(50 Hz). The previous dual-rate arrangement -- a 200 Hz command loop with
low-pass filters on the measured position, measured velocity and policy output,
wrapped around a 50 Hz policy tick -- has been removed. The filters were running
with alpha = 1.0 (pass-through) in any case, and the extra rate added bus traffic
without adding control authority.

CYCLE STRUCTURE
---------------
Each cycle sends setpoints twice, on purpose:

  1. `exchange()` re-sends the setpoints computed last cycle and collects the
     status replies they trigger. Re-sending the currently-active command is a
     no-op mechanically, and the OPERATION_CONTROL frame is what makes the
     actuators report.
  2. `send()` applies the freshly computed setpoints immediately.

That keeps sensor-to-actuator latency at a couple of milliseconds instead of a
full 20 ms cycle, which matters because the policy was trained with the action
applied in the same step its observation was taken.

SOFT START
----------
The loop opens with a soft start: targets ramp from the resting pose to the
policy profile's start pose (the default pose for --dry-run) and hold there, and
only then is the policy reset and run, so it starts where training episodes
start. Safety checks, fault checks and logging run identically in every phase,
and the soft start counts towards --duration.

WALKING
-------
A walking policy (its profile takes a velocity command) needs a command source:
--teleop (tools/gamepad_teleop.py on the laptop) or --command VX,WZ for a fixed
command. With --teleop the loop holds the crouch after the soft start until
START (OPTIONS on the pad) is pressed: lower the robot on its rope until the
feet carry it, then press it. STOP (CIRCLE) ends the run at any time; the
motors then go limp.

    python main-rl.py --model policy_walk.onnx --teleop   # on the Pi
    python3 tools/gamepad_teleop.py                       # on the laptop
"""

import argparse
import dataclasses
import sys
import time

import numpy as np

from config import (
    MAX_CONSECUTIVE_MISSED_REPLIES, MODEL_PATH, POLICY_PROFILES,
    SOFT_START_HOLD_S, SOFT_START_S, SPEC,
)
from control.command_source import DEFAULT_PORT, FixedCommandSource, UdpCommandSource
from control.policy import HoldPolicy, Policy
from control.soft_start import SoftStart
from robot.session import RobotSession
from utils.exceptions import (
    ActuatorFault, HardwareError, HardwareIOError, SafetyLimitError,
)
from utils.realtime import LoopPacer, restore_gc
from utils.recorder import Recorder
from utils.safety import SafetyMonitor


def parse_args():
    p = argparse.ArgumentParser(description="Run the RL policy on the robot.")
    p.add_argument('--model', default=MODEL_PATH,
                   help="Path to the ONNX actor (default: %(default)s)")
    p.add_argument('--dry-run', action='store_true',
                   help="Command the default pose instead of loading a policy. "
                        "Exercises CAN, IMU, safety, timing and logging without "
                        "a trained model.")
    p.add_argument('--no-imu', action='store_true',
                   help="Run with a synthetic level IMU (bench use only).")
    p.add_argument('--duration', type=float, default=None,
                   help="Stop automatically after this many seconds.")
    p.add_argument('--log-prefix', default='rl_log',
                   help="Filename prefix for the CSV log (default: %(default)s)")
    p.add_argument('--no-log', action='store_true', help="Disable CSV logging.")
    p.add_argument('--allow-unverified-watchdogs', action='store_true',
                   help="Run even if some motor-side CAN timeouts cannot be "
                        "verified, by register readback or, on firmware without "
                        "parameter 0x7028, by the zero-force silence test. Those "
                        "motors may NOT go limp if this process stalls -- only "
                        "use with a hardware kill switch in reach.")
    p.add_argument('--virtual', action='store_true',
                   help="Run against tools/fake_motors.py over an in-process "
                        "virtual CAN bus. No hardware required; implies --no-imu.")
    p.add_argument('--teleop', action='store_true',
                   help="Walking: take commands from tools/gamepad_teleop.py over "
                        "UDP, and wait for START after the soft start.")
    p.add_argument('--teleop-port', type=int, default=DEFAULT_PORT)
    p.add_argument('--teleop-peer', default=None,
                   help="Only accept teleop packets from this address (default: "
                        "lock onto the first sender).")
    p.add_argument('--command', default=None, metavar="VX,WZ",
                   help="Walking: a fixed command (m/s, rad/s) instead of teleop; "
                        "the policy starts right after the soft start.")
    return p.parse_args()


def command_source(args, policy):
    """The velocity-command source a walking policy needs, or None."""
    profile = getattr(policy, "profile", None)
    if profile is None or not profile.uses_command:
        if args.teleop or args.command:
            print("[WARN] This policy takes no velocity command; ignoring "
                  "--teleop/--command.")
        return None
    if args.command:
        vx, wz = (float(v) for v in args.command.split(','))
        return FixedCommandSource(vx=vx, wz=wz)
    if args.teleop:
        return UdpCommandSource(port=args.teleop_port, peer=args.teleop_peer)
    print("[ERROR] A walking policy needs --teleop or --command VX,WZ.")
    sys.exit(1)


def build_spec(virtual: bool):
    """The configured robot, optionally rebound to the virtual CAN backend."""
    if not virtual:
        return SPEC, None
    buses = [dataclasses.replace(b, interface="virtual") for b in SPEC.buses]
    spec = dataclasses.replace(SPEC, buses=buses)
    from tools.fake_motors import FakeRobot
    return spec, FakeRobot(spec)


def main():
    args = parse_args()
    spec, simulator = build_spec(args.virtual)
    use_imu = not (args.no_imu or args.virtual)

    print(f"[INFO] RL control loop at {spec.loop_rate_hz:g} Hz "
          f"({spec.dt * 1000:.1f} ms per cycle)")
    if simulator is not None:
        print("[INFO] VIRTUAL: running against simulated actuators.")
        simulator.start()

    if args.dry_run:
        print("[INFO] DRY RUN: commanding the default pose, no policy loaded.")
        policy = HoldPolicy(spec)
    else:
        policy = Policy(spec=spec, model_path=args.model, profiles=POLICY_PROFILES)
    commands = command_source(args, policy)
    wait_for_start = isinstance(commands, UdpCommandSource)

    safety = SafetyMonitor(spec)
    recorder = None if args.no_log else Recorder(spec)
    pacer = LoopPacer(spec.dt)

    exit_code = 0
    if commands is not None:
        commands.start()
    try:
        with RobotSession(spec, use_imu=use_imu,
                          allow_unverified_watchdogs=args.allow_unverified_watchdogs) as robot:
            # Confirm the robot is communicative and within bounds while limp,
            # before any gain is applied.
            print("[INFO] Checking initial hardware state...")
            state = robot.read_limp_state()
            safety.verify_measured_state(state)

            imu_sample = robot.read_imu()
            safety.verify_imu_sample(imu_sample)
            print(f"[INFO] Base tilt at start: "
                  f"{imu_sample.tilt_rad * 57.2958:.1f} deg")

            # Start from where the robot actually is, then ramp to the start
            # pose before the policy takes over.
            targets = robot.hold_targets(state)
            rest_sim, _ = robot.to_sim_frame(*robot.state_arrays(state))
            goal_sim = spec.default_pos_vector() if args.dry_run else policy.start_pose
            soft_start = SoftStart(rest_sim, goal_sim, SOFT_START_S, SOFT_START_HOLD_S)
            policy_running = False
            announced_wait = False
            last_status = 0.0

            print(f"[INFO] Soft start: ramping to the "
                  f"{'default' if args.dry_run else 'start'} pose over "
                  f"{SOFT_START_S:.1f} s, holding {SOFT_START_HOLD_S:.1f} s "
                  f"(Ctrl+C to stop)...")
            start_time = time.perf_counter()
            pacer.start()
            overrun_s = 0.0
            last_state = state

            while True:
                if commands is not None and commands.stop_requested():
                    print("\n[INFO] STOP from the controller.")
                    break

                # 1. Apply last cycle's setpoints and collect the state they
                #    provoke. An isolated missing reply is ridden through on the
                #    joint's previous reading; MAX_CONSECUTIVE in a row stops.
                state, miss = robot.exchange_tolerant(
                    targets, last_state, MAX_CONSECUTIVE_MISSED_REPLIES)
                if miss is not None:
                    print(f"\n[WARN] t={time.perf_counter() - start_time:6.2f} s: no reply "
                          f"from {miss.missing} "
                          f"({robot.consecutive_misses}/{MAX_CONSECUTIVE_MISSED_REPLIES} "
                          f"in a row, {robot.total_misses} total)"
                          + (f"; they sent instead: {miss.stray}" if miss.stray else ""))
                robot.check_faults()
                last_state = state
                safety.verify_measured_state(state)

                # 2. Base state. Never blocks; staleness is a hard fault.
                imu_sample = robot.read_imu()
                safety.verify_imu_sample(imu_sample)

                # 3. Soft-start ramp, then the policy; both in the sim frame.
                sim_pos, sim_vel = robot.to_sim_frame(*robot.state_arrays(state))
                elapsed = time.perf_counter() - start_time
                ready = soft_start.done(elapsed) and (
                    not wait_for_start or commands.start_requested())
                if not ready:
                    sim_targets = soft_start.target(elapsed)
                    if soft_start.done(elapsed) and not announced_wait:
                        announced_wait = True
                        print("[INFO] Holding the start pose. Lower the robot until "
                              "its feet carry it, then press OPTIONS (CIRCLE stops).")
                else:
                    if not policy_running:
                        # Fresh clock and observation history, as at an episode start.
                        policy.reset()
                        policy_running = True
                        print(f"[INFO] Soft start complete; "
                              f"{'holding' if args.dry_run else 'policy running'}.")
                    if commands is not None:
                        policy.set_command(*commands.command())
                    sim_targets = policy.act(sim_pos, sim_vel, imu_sample)

                # 4. Back to the hardware frame, and validate before it can move
                #    anything.
                hw_targets = robot.to_hardware_frame(sim_targets)
                safety.validate_commanded_targets(hw_targets)
                targets = robot.targets_from_array(hw_targets)

                # 5. Apply this cycle's setpoints now rather than next cycle.
                robot.send(targets)

                now = time.perf_counter()
                if recorder is not None:
                    recorder.record(
                        t=now - start_time,
                        state=state,
                        targets=targets,
                        imu_sample=imu_sample,
                        exchange_s=robot.network.last_exchange_s,
                        overrun_s=overrun_s,
                        missed=miss.missing if miss is not None else (),
                        phase="policy" if policy_running else "soft_start",
                        command=policy.command if policy_running and commands else (0.0, 0.0, 0.0),
                    )
                if commands is not None and now - last_status > 1.0:
                    last_status = now
                    print(f"\r[{now - start_time:6.1f} s] {commands.status()}      ",
                          end="", flush=True)

                if args.duration is not None and (now - start_time) >= args.duration:
                    print(f"\n[INFO] Reached {args.duration:.1f} s limit.")
                    break

                overrun_s = pacer.sleep()

    except SafetyLimitError as e:
        print(f"\n[EMERGENCY STOP] Safety interlock tripped: {e}")
        exit_code = 2
    except (HardwareIOError, ActuatorFault, HardwareError) as e:
        print(f"\n[CRITICAL] Hardware failure: {e}")
        exit_code = 3
    except Exception as e:
        print(f"\n[FATAL] Unexpected error: {e}")
        exit_code = 4
    finally:
        if commands is not None:
            commands.stop()
        if simulator is not None:
            simulator.stop()
        restore_gc()
        print(f"[INFO] Loop timing: {pacer.summary()}")
        if 'robot' in locals() and robot.total_misses:
            print(f"[INFO] Missed replies ridden through: {robot.total_misses}")
        if recorder is not None:
            recorder.save(prefix=args.log_prefix)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
