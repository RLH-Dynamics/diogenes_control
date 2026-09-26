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
The loop opens with a soft start: targets ramp from the resting pose to
config.START_POSE (the default pose for --dry-run) and hold there, and only
then is the policy reset and run, so it starts where training episodes start.
Safety checks, fault checks and logging run identically in both phases, and the
soft start counts towards --duration.
"""

import argparse
import dataclasses
import sys
import time

import numpy as np

from config import (
    ACTION_SCALE, CYCLE_PERIOD, MAX_CONSECUTIVE_MISSED_REPLIES, MODEL_PATH,
    OBSERVATION_HISTORY, OBSERVATION_TERMS, SIM_OBSERVATION_NAMES,
    SOFT_START_HOLD_S, SOFT_START_S, SPEC, START_POSE,
)
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
    return p.parse_args()


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
        policy = Policy(
            spec=spec,
            model_path=args.model,
            term_names=OBSERVATION_TERMS,
            action_scale=ACTION_SCALE,
            period=CYCLE_PERIOD,
            history_length=OBSERVATION_HISTORY,
            sim_observation_names=SIM_OBSERVATION_NAMES,
        )

    safety = SafetyMonitor(spec)
    recorder = None if args.no_log else Recorder(spec)
    pacer = LoopPacer(spec.dt)

    exit_code = 0
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
            goal_sim = (spec.default_pos_vector() if args.dry_run else
                        np.array([START_POSE[n] for n in spec.names], dtype=np.float32))
            soft_start = SoftStart(rest_sim, goal_sim, SOFT_START_S, SOFT_START_HOLD_S)
            policy_running = False

            print(f"[INFO] Soft start: ramping to the "
                  f"{'default' if args.dry_run else 'start'} pose over "
                  f"{SOFT_START_S:.1f} s, holding {SOFT_START_HOLD_S:.1f} s "
                  f"(Ctrl+C to stop)...")
            start_time = time.perf_counter()
            pacer.start()
            overrun_s = 0.0
            last_state = state

            while True:
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
                if not soft_start.done(elapsed):
                    sim_targets = soft_start.target(elapsed)
                else:
                    if not policy_running:
                        # Fresh clock and observation history, as at an episode start.
                        policy.reset()
                        policy_running = True
                        print(f"[INFO] Soft start complete; "
                              f"{'holding' if args.dry_run else 'policy running'}.")
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
                    )

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
