"""Supervisory limits. Everything here raises rather than clamps.

Keyed by JOINT NAME, not CAN id. With two buses the id alone is no longer a
unique identity, and a limit lookup that silently missed would be a safety
interlock that silently did nothing.

Three checks:
  * measured joint state against per-joint position and velocity bounds,
  * commanded joint targets against the same position bounds,
  * base attitude and angular rate from the IMU.

The attitude check is new with the two-legged machine: on a robot that can fall
over, tilt is the limit that protects the hardware when a policy misbehaves.
"""

import numpy as np

from utils.exceptions import SafetyLimitError


class SafetyMonitor:
    def __init__(self, spec):
        self.spec = spec
        self.names = spec.names
        self.pos_lo, self.pos_hi = spec.pos_limit_arrays()
        self.vel_lo, self.vel_hi = spec.vel_limit_arrays()
        self.safety = spec.safety
        self.imu_spec = spec.imu

    # ---------------------------------------------------------------- joints --

    def verify_measured_state(self, state: dict):
        """Hard-fault if any joint has strayed outside its physical bounds."""
        for i, name in enumerate(self.names):
            reading = state.get(name)
            if reading is None:
                raise SafetyLimitError(
                    f"No state reported for joint '{name}'. Refusing to continue "
                    f"with an incomplete picture of the robot."
                )

            pos = reading.get('pos', 0.0)
            if pos < self.pos_lo[i] or pos > self.pos_hi[i]:
                raise SafetyLimitError(
                    f"Joint '{name}' out of position bounds! Pos: {pos:.3f} rad. "
                    f"Limits: [{self.pos_lo[i]}, {self.pos_hi[i]}]"
                )

            vel = reading.get('vel', 0.0)
            if vel < self.vel_lo[i] or vel > self.vel_hi[i]:
                raise SafetyLimitError(
                    f"Joint '{name}' out of velocity bounds! Vel: {vel:.3f} rad/s. "
                    f"Limits: [{self.vel_lo[i]}, {self.vel_hi[i]}]"
                )

    def validate_commanded_targets(self, hw_targets):
        """Hard-fault on a policy command outside the safe envelope."""
        targets = np.asarray(hw_targets, dtype=np.float64)
        if targets.shape[0] != len(self.names):
            raise SafetyLimitError(
                f"Command vector has {targets.shape[0]} entries, expected "
                f"{len(self.names)} (one per joint)."
            )

        if not np.all(np.isfinite(targets)):
            bad = [self.names[i] for i in np.flatnonzero(~np.isfinite(targets))]
            raise SafetyLimitError(
                f"Policy produced non-finite commands for joints: {bad}"
            )

        below = np.flatnonzero(targets < self.pos_lo)
        above = np.flatnonzero(targets > self.pos_hi)
        for i in np.concatenate([below, above]):
            raise SafetyLimitError(
                f"Rogue policy command! Requested {targets[i]:.3f} rad for "
                f"'{self.names[i]}'. Limits: "
                f"[{self.pos_lo[i]}, {self.pos_hi[i]}]"
            )

    # ------------------------------------------------------------------- base --

    def verify_imu_sample(self, sample, now: float = None):
        """Hard-fault on a stale, implausible, or over-tilted base reading."""
        if not self.safety.enforce_attitude or self.imu_spec is None:
            return

        age = sample.age(now)
        if age > self.imu_spec.max_age_s:
            raise SafetyLimitError(
                f"IMU sample is stale ({age * 1000:.1f} ms old, limit "
                f"{self.imu_spec.max_age_s * 1000:.1f} ms). A frozen base "
                f"estimate is more dangerous than none."
            )

        if not np.all(np.isfinite(sample.ang_vel)) or not np.all(np.isfinite(sample.projected_gravity)):
            raise SafetyLimitError("IMU reported non-finite values.")

        tilt = sample.tilt_rad
        if tilt > self.safety.max_tilt_rad:
            raise SafetyLimitError(
                f"Base tilt {np.degrees(tilt):.1f} deg exceeds limit "
                f"{np.degrees(self.safety.max_tilt_rad):.1f} deg. Robot is falling."
            )

        rate = float(np.linalg.norm(sample.ang_vel))
        if rate > self.safety.max_base_ang_vel:
            raise SafetyLimitError(
                f"Base angular rate {rate:.2f} rad/s exceeds limit "
                f"{self.safety.max_base_ang_vel:.2f} rad/s."
            )
