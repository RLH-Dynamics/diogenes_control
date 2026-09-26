"""ONNX actor inference.

The policy is now a thin object: it owns the network, the phase clock, and the
previous raw action, and it speaks the SIMULATION frame exclusively. Converting
between hardware and sim frames is the Robot's job; assembling the observation
is the ObservationBuilder's job. What is left here is inference plus the
action transform, which is the part that genuinely belongs to the policy.

The startup checks are deliberately fatal. A policy whose observation layout
disagrees with the deploy code will still produce plausible-looking numbers, and
those numbers drive 60 N.m actuators. Besides the input width, the metadata
diogenes_mjlab's export_onnx.py attaches (joint order, default pose, action
scale, observation terms, history length, clock period, control period) is
checked against the config.

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
    def __init__(self, spec, model_path, term_names, action_scale, period,
                 history_length: int = 0, sim_observation_names=None):
        self.spec = spec
        self.num_joints = spec.num_joints
        self.action_scale = action_scale
        self.period = period
        self.history_length = history_length
        self.sim_observation_names = sim_observation_names

        self.default_pos = spec.default_pos_vector()
        self.builder = ObservationBuilder(spec, term_names, history_length)

        # The sim clamps each target to its actuator's ctrlrange (SIM frame).
        if spec.sim_contract is None:
            raise ValueError("Policy needs the sim joint contract for the "
                             "actuator control ranges.")
        ranges = [spec.sim_contract['joints'][n]['ctrl_range'] for n in spec.names]
        self.ctrl_lo = np.array([r[0] for r in ranges], dtype=np.float32)
        self.ctrl_hi = np.array([r[1] for r in ranges], dtype=np.float32)

        # Previous RAW network output. Zeroed at reset to match sim, where the
        # action history starts zeroed on the first inference of an episode.
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)

        # Velocity command, if the layout includes a command term.
        self.command = np.zeros(3, dtype=np.float32)

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

        print(self.builder.describe())
        self._verify_input_dim()
        self._verify_metadata()
        self.start_time = time.perf_counter()

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
                f"The deployed observation layout does not match the trained "
                f"policy. Update OBSERVATION_TERMS in config.py to match the "
                f"training environment's actor observation group. Refusing to "
                f"run with a mismatched observation vector."
            )
            sys.exit(1)
        print(f"[INFO] Observation width OK ({expected} dims).")

    def _verify_metadata(self):
        """Check the export's metadata against this deployment's config."""
        meta = self.session.get_modelmeta().custom_metadata_map
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
                              f"config {self.default_pos.tolist()}")
        if (v := need('action_scale')) is not None:
            got = _floats(v)
            if not np.allclose(got, self.action_scale, atol=_META_TOL):
                errors.append(f"action scale: policy {got.tolist()}, "
                              f"config {self.action_scale}")
        if (v := need('observation_names')) is not None and self.sim_observation_names is not None:
            if v.split(',') != list(self.sim_observation_names):
                errors.append(f"observation terms: policy {v.split(',')}, config "
                              f"{list(self.sim_observation_names)}")
        if (v := need('actor_history_length')) is not None and int(v) != self.history_length:
            errors.append(f"observation history: policy {v}, config {self.history_length}")
        if (v := need('phase_period')) is not None and abs(float(v) - self.period) > _META_TOL:
            errors.append(f"phase clock period: policy {v} s, config {self.period} s")
        if (v := need('step_dt')) is not None and abs(float(v) - spec.dt) > _META_TOL:
            errors.append(f"control period: policy {v} s, config {spec.dt} s")

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
        print(f"[INFO] Policy metadata OK (task {meta.get('task_id', 'unknown')}).")

    def reset(self):
        """Reset the phase-clock origin and action history before a run."""
        self.start_time = time.perf_counter()
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)
        self.builder.reset()

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self.command = np.array([vx, vy, wz], dtype=np.float32)

    def phase(self) -> np.ndarray:
        """[sin, cos] of the global hop phase, wrapping every `period` seconds."""
        elapsed = time.perf_counter() - self.start_time
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
        # then clamped to the actuator's ctrlrange as the sim does.
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
