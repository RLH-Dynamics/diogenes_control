"""ONNX actor inference.

The policy is now a thin object: it owns the network, the phase clock, and the
previous raw action, and it speaks the SIMULATION frame exclusively. Converting
between hardware and sim frames is the Robot's job; assembling the observation
is the ObservationBuilder's job. What is left here is inference plus the
action transform, which is the part that genuinely belongs to the policy.

The startup width check is deliberately fatal. A policy whose observation layout
disagrees with the deploy code will still produce plausible-looking numbers, and
those numbers drive 60 N.m actuators.
"""

import sys
import time

import numpy as np

from control.observation import ObsContext, ObservationBuilder


class Policy:
    def __init__(self, spec, model_path, term_names, action_scale, period):
        self.spec = spec
        self.num_joints = spec.num_joints
        self.action_scale = action_scale
        self.period = period

        self.default_pos = spec.default_pos_vector()
        self.builder = ObservationBuilder(spec, term_names)

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

    def reset(self):
        """Reset the phase-clock origin and action history before a run."""
        self.start_time = time.perf_counter()
        self.last_raw_action = np.zeros(self.num_joints, dtype=np.float32)

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

        # use_default_offset=True: the absolute target is scale*action + default.
        return raw_actions * self.action_scale + self.default_pos


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
