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
from utils.exceptions import (
    DirectionsUnverifiedError, MissedReplies, WatchdogUnverifiedError,
)


class Robot:
    def __init__(self, spec, imu=None):
        self.spec = spec
        self.network = RobstrideNetwork(spec)
        self.imu = imu if imu is not None else NullImu()
        self._legs = {name: Leg(self, name) for name in spec.leg_names}

        # Missed-reply bookkeeping for exchange_tolerant().
        self.consecutive_misses = 0
        self.total_misses = 0

        # Cached canonical-order vectors, built once.
        self._directions = spec.direction_vector()
        self._offsets = spec.offset_vector()
        self._default_pos = spec.default_pos_vector()

    # ------------------------------------------------------------ lifecycle --

    def start(self, control_mode: str = 'MIT', verify_watchdogs: bool = True,
              require_watchdogs: bool = True,
              require_verified_directions: bool = True,
              resolve_turns: bool = True):
        """Bring up every CAN channel, enable the motors, start the IMU.

        Refuses to enable anything unless every motor-side CAN timeout is
        confirmed (`require_watchdogs`) and the joint direction signs and zero
        offsets were verified (`require_verified_directions`). Only tools that
        never apply gain through the sim frame (limp reads, the mapping test,
        zero calibration) should turn that off.

        `resolve_turns` corrects each motor's power-up reading by whole turns
        (see RobstrideNetwork.resolve_turns). Calibration works in the motors'
        raw frame and turns it off.
        """
        print(self.spec.describe())
        if not self.spec.zero_offsets_verified:
            message = ("Joint zero-offset signs (config.py) have not been verified "
                       "on the robot. Check them with tools/stream_joints.py + the "
                       "live viewer, then set ZERO_OFFSETS_VERIFIED.")
            if require_verified_directions:
                raise DirectionsUnverifiedError(message + " Refusing to enable.")
            print(f"[WARN] {message}")
        if not self.spec.directions_verified:
            contract = self.spec.sim_contract
            found = contract['direction_signature'] if contract else "no contract"
            message = (
                "Joint direction signs were verified against sim contract "
                f"{self.spec.directions_verified_signature}, but the loaded contract "
                f"is {found}: the sim's joint conventions changed since. Re-run the "
                "limp mapping test (tools/stream_joints.py + live_joint_viewer.py), "
                "fix any `direction` in config.py, then update "
                "DIRECTIONS_VERIFIED_SIGNATURE."
            )
            if require_verified_directions:
                raise DirectionsUnverifiedError(message + " Refusing to enable.")
            print(f"[WARN] {message} Continuing because this tool applies no gain.")

        self.network.open()
        if resolve_turns:
            self.network.resolve_turns()

        if verify_watchdogs and not self.network.verify_watchdogs():
            if require_watchdogs:
                raise WatchdogUnverifiedError(
                    "One or more motor-side CAN timeouts could not be verified "
                    "(see warnings above), so those motors may not go limp if "
                    "the host stops transmitting. Refusing to enable."
                )
            print("[WARN] One or more hardware watchdogs could not be verified. "
                  "The motors may not go limp on their own if the host stops "
                  "transmitting. Continuing because this tool applies no gain.")

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

    def exchange(self, targets: dict, kp=None, kd=None,
                 timeout: float = 0.010) -> dict:
        """Command every joint and collect every reply. Returns name -> state.

        `kp` / `kd` are a single gain for every joint or a dict name -> gain.
        """
        kp = self.spec.kp if kp is None else kp
        kd = self.spec.kd if kd is None else kd
        return self.network.exchange(targets, kp=kp, kd=kd, timeout=timeout)

    def exchange_tolerant(self, targets: dict, last_state: dict,
                          max_consecutive: int, kp=None, kd=None,
                          timeout: float = 0.010):
        """exchange(), riding through isolated missed replies.

        A joint that does not reply this cycle keeps its reading from
        `last_state`; the motor itself carries on executing its last command.
        The MissedReplies is re-raised once `max_consecutive` cycles in a row
        have misses, or if there is no previous state to fall back on.

        Returns (state, miss) where miss is the MissedReplies or None.
        """
        try:
            state = self.exchange(targets, kp, kd, timeout)
        except MissedReplies as miss:
            self.consecutive_misses += 1
            self.total_misses += 1
            if last_state is None or self.consecutive_misses >= max_consecutive:
                raise
            return {**last_state, **miss.state}, miss
        self.consecutive_misses = 0
        return state, None

    def send(self, targets: dict, kp=None, kd=None):
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
        """Hardware frame -> sim frame: direction flip, then zero offset."""
        return pos * self._directions + self._offsets, vel * self._directions

    def to_hardware_frame(self, sim_pos: np.ndarray) -> np.ndarray:
        """Sim frame -> hardware frame: remove zero offset, then direction flip."""
        return (sim_pos - self._offsets) * self._directions

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
