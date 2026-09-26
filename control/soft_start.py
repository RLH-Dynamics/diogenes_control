"""Bring the joints from wherever they rest to the policy's start pose.

Without it, the first commanded step would jump straight from the resting pose
to the policy's output: at kp 60 that is a snap of up to ~24 N.m with the legs
hanging ~20 deg off, and it would also start the policy outside the pose range
it was trained from (the gait's nominal pose, +-0.15 rad).

Targets are blended linearly in the SIM frame from the measured start pose to
the goal over `ramp_s`, then held at the goal for `hold_s`.
"""

import numpy as np


class SoftStart:
    def __init__(self, start_sim: np.ndarray, goal_sim: np.ndarray,
                 ramp_s: float, hold_s: float):
        self.start = np.asarray(start_sim, dtype=np.float32)
        self.goal = np.asarray(goal_sim, dtype=np.float32)
        self.ramp_s = max(ramp_s, 1e-6)
        self.duration = ramp_s + hold_s

    def done(self, elapsed: float) -> bool:
        return elapsed >= self.duration

    def target(self, elapsed: float) -> np.ndarray:
        alpha = min(1.0, max(0.0, elapsed / self.ramp_s))
        return self.start + alpha * (self.goal - self.start)
