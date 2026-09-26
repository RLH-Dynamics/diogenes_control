"""ONNX actor inference.

The policy is now a thin object: it owns the network, the phase clock, and the
previous raw action, and it speaks the SIMULATION frame exclusively. Converting
between hardware and sim frames is the Robot's job; assembling the observation
is the ObservationBuilder's job. What is left here is inference plus the
action transform, which is the part that genuinely belongs to the policy.

The startup checks are deliberately fatal. A policy whose observation layout
disagrees with the deploy code will still produce plausible-looking numbers, and
those numbers drive 60 N.m actuators. Besides the input width, the metadata
diogenes_mjlab's export_onnx.py attaches (task, joint order, default pose,
observation terms, history length, control period) is checked against the
config's profile for that task; action scale, clock period and command ranges
are taken from it.

Targets are clamped to each actuator's sim control range before they leave the
policy: the sim's position actuators clamp every target to `ctrlrange`, so the
policy was trained to rely on that, and without it a target just past a joint
limit would trip the (non-clamping) safety layer instead.
"""

import sys
import time

import numpy as np

from control.observation import ObsContext, ObservationBuilder


# Relative tolerance for float metadata (mjlab writes floats to 3 decimals).
_META_TOL = 1e-3


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split(',')], dtype=np.float64)


class Policy:
    """An exported actor plus the observation layout and action transform its
    training task used, picked and checked through its metadata.

    `profiles` maps task ids to PolicyProfile (config.POLICY_PROFILES); the
    file's task_id selects one. Action scale, gait-clock period and command
    ranges come from the file: they differ between runs of one task.
    """

    def __init__(self, spec, model_path, profiles: dict, clock=time.perf_counter):
        self.spec = spec
        self.num_joints = spec.num_joints
        # Seconds, for the gait clock; sim-to-sim runs pass simulated time.
        self.clock = clock

        # Imported here, not at module scope, so HoldPolicy (and therefore
        # `main-rl.py --dry-run`) works on a machine with no onnxruntime.
        try:
            import onnxruntime as ort
        except ImportError as e:
            print(f"[ERROR] onnxruntime is required to load a policy: {e}")
            sys.exit(1)

        print(f"[INFO] Loading model: {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            sys.exit(1)
        self.meta = self.session.get_modelmeta().custom_metadata_map

        task = self.meta.get('task_id')
        if task not in profiles:
            print(f"[ERROR] Policy task {task!r} has no profile in config.POLICY_PROFILES "
                  f"(known: {sorted(profiles)}). Refusing to run.")
            sys.exit(1)
        self.profile = profiles[task]
        self.history_length = self.profile.history_length
        self.sim_observation_names = list(self.profile.sim_observation_names)
        self.default_pos = self.profile.pose_vector(self.profile.default_pos, spec.names)
        self.start_pose = self.profile.pose_vector(self.profile.start_pose, spec.names)
        self.builder = ObservationBuilder(spec, list(self.profile.observation_terms),
                                          self.history_length)

        # The sim clamps each target to its joint's range (ctrl_range, SIM frame).
        if spec.sim_contract is None:
            raise ValueError("Policy needs the sim joint contract for the "
                             "actuator control ranges.")
        ranges = [spec.sim_contract['joints'][n]['ctrl_range'] for n in spec.names]
        self.ctrl_lo = np.array([r[0] for r in ranges], dtype=np.float32)
        self.ctrl_hi = np.array([r[1] for r in ranges], dtype=np.float32)

        # IMU chip frame <- base frame: the transpose of the mount rotation.
        if spec.imu is not None:
            self.chip_from_base = np.asarray(spec.imu.mount_rotation, dtype=np.float32).T
        else:
            self.chip_from_base = np.eye(3, dtype=np.float32)

        # Previous RAW network output. Zeroed at reset to match sim, where the
        # action history starts zeroed on the first inference of an episode.
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)

        print(self.builder.describe())
        self._verify_input_dim()
        self._verify_metadata()
        self.start_time = self.clock()

    def _verify_input_dim(self):
        expected = self.builder.total_width
        try:
            width = self.session.get_inputs()[0].shape[-1]
        except Exception as e:
            print(f"[WARN] Could not introspect model input shape: {e}")
            return

        if isinstance(width, int) and width != expected:
            print(
                f"\n[ERROR] Observation width mismatch.\n"
                f"  ONNX model expects : {width}\n"
                f"  This layout builds : {expected}\n"
                f"{self.builder.describe()}\n"
                f"The {self.profile.task_id} profile in config.py does not match the "
                f"trained policy's actor observation group. Refusing to run with a "
                f"mismatched observation vector."
            )
            sys.exit(1)
        print(f"[INFO] Observation width OK ({expected} dims).")

    def _verify_metadata(self):
        """Check the export's metadata against the profile and config, and read
        the per-run values (action scale, clock period, command ranges)."""
        meta = self.meta
        spec = self.spec
        errors, warnings = [], []

        def need(key):
            if key not in meta:
                errors.append(f"missing metadata '{key}' (re-export with "
                              f"diogenes_mjlab tools/export_onnx.py)")
                return None
            return meta[key]

        if (v := need('joint_names')) is not None and v.split(',') != spec.names:
            errors.append(f"joint order: policy {v.split(',')}, config {spec.names}")
        if (v := need('default_joint_pos')) is not None:
            got = _floats(v)
            if got.shape != self.default_pos.shape or not np.allclose(
                    got, self.default_pos, atol=_META_TOL):
                errors.append(f"default joint pos: policy {got.tolist()}, "
                              f"profile {self.default_pos.tolist()}")
        if (v := need('observation_names')) is not None:
            if v.split(',') != self.sim_observation_names:
                errors.append(f"observation terms: policy {v.split(',')}, profile "
                              f"{self.sim_observation_names}")
        if (v := need('actor_history_length')) is not None and int(v) != self.history_length:
            errors.append(f"observation history: policy {v}, profile {self.history_length}")
        if (v := need('step_dt')) is not None and abs(float(v) - spec.dt) > _META_TOL:
            errors.append(f"control period: policy {v} s, config {spec.dt} s")

        self.action_scale = np.ones(self.num_joints, dtype=np.float32)
        if (v := need('action_scale')) is not None:
            scale = _floats(v)
            if scale.size not in (1, self.num_joints) or not np.all(scale > 0):
                errors.append(f"action scale: {v!r} is not 1 or {self.num_joints} positive values")
            else:
                self.action_scale = np.broadcast_to(scale, (self.num_joints,)).astype(np.float32)
        self.period = None
        if (v := need('phase_period')) is not None:
            self.period = float(v)
            if not self.period > 0:
                errors.append(f"phase clock period {v} is not positive")

        # Commands are clamped to the ranges the policy was trained on.
        self.command_lo = np.zeros(3, dtype=np.float32)
        self.command_hi = np.zeros(3, dtype=np.float32)
        if self.profile.uses_command and (v := need('command_ranges')) is not None:
            r = _floats(v)
            if r.size != 6:
                errors.append(f"command_ranges: expected 6 values, got {v!r}")
            else:
                self.command_lo = r[0::2].astype(np.float32)
                self.command_hi = r[1::2].astype(np.float32)

        # Gains are a free choice on hardware and were randomised in training,
        # so a mismatch is worth knowing but not fatal.
        for key, value in (('joint_stiffness', spec.kp), ('joint_damping', spec.kd)):
            if key in meta and not np.allclose(_floats(meta[key]), value, rtol=0.01):
                warnings.append(f"{key}: sim {meta[key]}, config {value}")

        for w in warnings:
            print(f"[WARN] Policy metadata differs from config: {w}")
        if errors:
            print("\n[ERROR] Policy does not match this deployment's config:")
            for e in errors:
                print(f"  - {e}")
            print("Refusing to run.")
            sys.exit(1)
        print(f"[INFO] Policy metadata OK (task {self.profile.task_id}): action scale "
              f"{np.round(self.action_scale, 3).tolist()}, gait period {self.period:.3f} s"
              + (f", commands vx {self.command_lo[0]:+.2f}..{self.command_hi[0]:+.2f} m/s, "
                 f"wz {self.command_lo[2]:+.2f}..{self.command_hi[2]:+.2f} rad/s"
                 if self.profile.uses_command else ""))

    def reset(self):
        """Reset the phase-clock origin and action history before a run."""
        self.start_time = self.clock()
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)
        self.builder.reset()

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        """Velocity command, clamped to the ranges the policy was trained on."""
        cmd = np.array([vx, vy, wz], dtype=np.float32)
        self.command = np.clip(cmd, self.command_lo, self.command_hi)

    def phase(self) -> np.ndarray:
        """[sin, cos] of the gait phase, wrapping every `period` seconds."""
        elapsed = self.clock() - self.start_time
        angle = 2.0 * np.pi * (elapsed / self.period)
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

    def act(self, sim_pos, sim_vel, imu_sample=None) -> np.ndarray:
        """Run one policy tick. Takes and returns SIM-frame joint positions."""
        if imu_sample is not None:
            ang_vel = np.asarray(imu_sample.ang_vel, dtype=np.float32)
            proj_g = np.asarray(imu_sample.projected_gravity, dtype=np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)
            proj_g = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        ctx = ObsContext(
            sim_pos=np.asarray(sim_pos, dtype=np.float32),
            sim_vel=np.asarray(sim_vel, dtype=np.float32),
            default_pos=self.default_pos,
            last_raw_action=self.last_raw_action,
            base_ang_vel=ang_vel,
            projected_gravity=proj_g,
            imu_ang_vel_chip=self.chip_from_base @ ang_vel,
            imu_gravity_chip=self.chip_from_base @ proj_g,
            phase=self.phase(),
            commands=self.command,
        )

        obs = self.builder.build(ctx)
        raw_actions = np.asarray(
            self.session.run(None, {self.input_name: obs})[0][0], dtype=np.float32
        )

        # Feed back the PRE-scale/offset network output, as mjlab does.
        self.last_raw_action = raw_actions.copy()

        # use_default_offset=True: the absolute target is scale*action + default,
        # then clamped to the joint's range as the sim does.
        targets = raw_actions * self.action_scale + self.default_pos
        return np.clip(targets, self.ctrl_lo, self.ctrl_hi)


class HoldPolicy:
    """Stand-in that always commands the default pose.

    Lets the full loop -- CAN exchange, safety interlocks, IMU staleness checks,
    logging and timing -- be exercised on hardware before the retrained ONNX
    exists. `main-rl.py --dry-run` selects it.
    """

    def __init__(self, spec):
        self.spec = spec
        self.default_pos = spec.default_pos_vector()

    def reset(self):
        pass

    def set_command(self, vx=0.0, vy=0.0, wz=0.0):
        pass

    def act(self, sim_pos, sim_vel, imu_sample=None) -> np.ndarray:
        return self.default_pos
