"""Sensor-agnostic interfaces for base-state estimation.

The control loop talks to an `ImuSource`, never to a specific driver. That keeps
three things possible without touching the loop: running the whole stack on a
bench with no IMU attached (`NullImu`), replaying a recorded run
(`ReplayImu`), and swapping the BNO085's I2C driver for its UART-RVC mode or a
separate reader process if I2C latency proves unreliable.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

# Gravity direction in the world frame, normalised. An upright robot sees this
# same vector in its own base frame.
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class ImuSample:
    """One immutable snapshot of base state, already in the ROBOT BASE frame.

    Frozen on purpose: the reader thread publishes by swapping the whole object,
    so the control loop can never observe a half-updated sample.
    """
    t: float                     # time.perf_counter() at acquisition
    quat: np.ndarray             # (4,) [w, x, y, z], base orientation in world
    ang_vel: np.ndarray          # (3,) rad/s, base frame
    lin_accel: np.ndarray        # (3,) m/s^2, base frame
    projected_gravity: np.ndarray  # (3,) unit gravity direction, base frame
    status: int = 0              # driver-reported accuracy, 0..3 on the BNO085

    @property
    def tilt_rad(self) -> float:
        """Angle between the base's 'down' and true down. 0 means upright."""
        g = self.projected_gravity
        norm = float(np.linalg.norm(g))
        if norm < 1e-6:
            return 0.0
        cos_tilt = float(np.dot(g / norm, GRAVITY_WORLD))
        return float(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))

    def age(self, now: Optional[float] = None) -> float:
        import time
        return (time.perf_counter() if now is None else now) - self.t


class ImuSource(Protocol):
    """What the control loop requires of any IMU implementation."""

    def start(self) -> None:
        """Begin acquiring. Must return only once a first sample is available."""

    def latest(self) -> ImuSample:
        """Return the most recent sample. MUST NOT block."""

    def stop(self) -> None:
        """Stop acquiring and release the device."""


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a [w, x, y, z] quaternion (body -> world)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - z * w),       s * (x * z + y * w)],
        [s * (x * y + z * w),       1.0 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),       s * (y * z + x * w),       1.0 - s * (x * x + y * y)],
    ])


def projected_gravity_from_quat(q: np.ndarray) -> np.ndarray:
    """Express the world gravity direction in the body frame."""
    return quat_to_rotation_matrix(q).T @ GRAVITY_WORLD


class NullImu:
    """A perfectly level, perfectly still IMU.

    Lets the full observation pipeline, safety layer and logging run with no
    hardware attached. Never stale, so staleness checks pass trivially.
    """

    def __init__(self):
        self._started = False

    def start(self):
        self._started = True

    def latest(self) -> ImuSample:
        import time
        return ImuSample(
            t=time.perf_counter(),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.zeros(3),
            lin_accel=np.array([0.0, 0.0, 9.81]),
            projected_gravity=GRAVITY_WORLD.copy(),
            status=3,
        )

    def stop(self):
        self._started = False
