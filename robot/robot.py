"""The whole machine: every actuator on every bus, plus the base IMU.

This is the object entry-point scripts hold. It owns the actuator network and
the IMU, converts between the name-keyed dictionaries the transport layer speaks
and the canonical-order arrays the policy speaks, and exposes legs as named
views for diagnostics.
"""

import numpy as np

from robot.leg import Leg
from robstride.network import RobstrideNetwork
from sensors.base import ImuSample, NullImu


class Robot:
    def __init__(self, spec, imu=None):
        self.spec = spec
        self.network = RobstrideNetwork(spec)
        self.imu = imu if imu is not None else NullImu()
        self._legs = {name: Leg(self, name) for name in spec.leg_names}

        # Cached canonical-order vectors, built once.
        self._directions = spec.direction_vector()
        self._default_pos = spec.default_pos_vector()

    # ------------------------------------------------------------ lifecycle --

    def start(self, control_mode: str = 'MIT', verify_watchdogs: bool = True):
        """Bring up every CAN channel, enable the motors, start the IMU."""
        print(self.spec.describe())
        self.network.open()

        if verify_watchdogs and not self.network.verify_watchdogs():
            print("[WARN] One or more hardware watchdogs could not be verified. "
                  "The motors may not go limp on their own if the host stops "
                  "transmitting.")

        self.network.enable(control_mode=control_mode)
        self.imu.start()

    def shutdown(self):
        """Disable motors and release both subsystems. Safe to call twice."""
        try:
            self.imu.stop()
        except Exception as e:
            print(f"[WARN] Non-fatal error stopping IMU: {e}")
        self.network.shutdown()

    # --------------------------------------------------------------- control --

    def exchange(self, targets: dict, kp: float = None, kd: float = None,
                 timeout: float = 0.005) -> dict:
        """Command every joint and collect every reply. Returns name -> state."""
        kp = self.spec.kp if kp is None else kp
        kd = self.spec.kd if kd is None else kd
        return self.network.exchange(targets, kp=kp, kd=kd, timeout=timeout)

    def send(self, targets: dict, kp: float = None, kd: float = None):
        """Command every joint without waiting for replies."""
        kp = self.spec.kp if kp is None else kp
        kd = self.spec.kd if kd is None else kd
        self.network.send(targets, kp=kp, kd=kd)

    def read_limp_state(self, timeout: float = 0.02) -> dict:
        """Read the full joint state with zero gains, so nothing moves."""
        return self.exchange(self.network.limp_targets(), kp=0.0, kd=0.0,
                             timeout=timeout)

    def check_faults(self):
        self.network.check_faults()

    # ---------------------------------------------------------------- sensor --

    def read_imu(self) -> ImuSample:
        """Latest base-state snapshot. Never blocks."""
        return self.imu.latest()

    # ------------------------------------------------- array <-> dict bridge --

    def state_arrays(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Hardware-frame (pos, vel) arrays in canonical joint order."""
        names = self.spec.names
        pos = np.array([state[n]['pos'] for n in names], dtype=np.float32)
        vel = np.array([state[n]['vel'] for n in names], dtype=np.float32)
        return pos, vel

    def to_sim_frame(self, pos: np.ndarray, vel: np.ndarray):
        """Apply the per-joint direction flip, hardware frame -> sim frame."""
        return pos * self._directions, vel * self._directions

    def to_hardware_frame(self, sim_pos: np.ndarray) -> np.ndarray:
        """Apply the per-joint direction flip, sim frame -> hardware frame."""
        return sim_pos * self._directions

    def targets_from_array(self, hw_positions) -> dict:
        """Build a transport-layer target dict from a canonical-order array."""
        return {
            name: {'pos': float(hw_positions[i]), 'vel': 0.0, 'torque': 0.0}
            for i, name in enumerate(self.spec.names)
        }

    def hold_targets(self, state: dict) -> dict:
        """Targets that hold the joints exactly where they currently are."""
        return {name: {'pos': state[name]['pos'], 'vel': 0.0, 'torque': 0.0}
                for name in self.spec.names}

    # ------------------------------------------------------------------ legs --

    @property
    def legs(self) -> dict:
        return dict(self._legs)

    def leg(self, name: str) -> Leg:
        return self._legs[name]
