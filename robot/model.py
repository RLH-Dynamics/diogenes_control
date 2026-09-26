"""Structured description of the robot's hardware.

This module defines the data model only -- no I/O, no hardware access. The
concrete values live in `config.py`, which is the single source of truth.

THE ORDERING CONTRACT
---------------------
`RobotSpec.joints` is an ORDERED list, and that order defines:

  * the order of every per-joint block in the policy observation vector,
  * the order of the policy's action output,
  * the order of every derived vector (default positions, direction flips,
    position/velocity limits).

It MUST match the joint ordering used by the training environment. Nothing in
this codebase can verify that for you, so `RobotSpec` cross-checks the declared
order against an explicit `policy_joint_order` list which you copy verbatim out
of the trained env config. If the two disagree, construction fails loudly rather
than producing a robot that moves the wrong limb.

CAN identity is `(bus, can_id)`, never `can_id` alone. Joint *names* are the key
used everywhere above the transport layer.

THE SIM JOINT CONTRACT
----------------------
`sim_joint_contract.json` is exported from the MJCF by diogenes_mjlab (see
harold_biped/joint_contract.py). It records each joint's sim range and, as
signs, what a POSITIVE angle does. `direction` signs here can only be checked on
the real robot, so the config records the contract's `direction_signature` that
the check was done against. `RobotSpec.directions_verified` is False whenever
the two differ, and `Robot.start` then refuses to apply gain.

Position limits are declared in the SIM frame, where they can be checked against
the contract, and converted to the hardware frame through `direction`. A sign
fix therefore can never leave the limits pointing the wrong way.

ZERO OFFSETS
------------
A motor's zero is set at a physical reference (tools/calibrate_zeros.py): the
hips against a straight edge at sim zero, the thighs and calves against hard
stops outside the sim range. `sim_offset` is the sim-frame angle of the motor's
zero, so

    sim = direction * hw + sim_offset        hw = direction * (sim - sim_offset)

`RobotSpec.zero_offsets_verified` records whether the offsets' signs were
checked on the robot; like unverified directions, Robot.start then refuses gain.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BusSpec:
    """One CAN channel.

    `interface` is the python-can backend. It is configurable so the whole stack
    can be exercised against the in-process 'virtual' backend (see
    tools/fake_motors.py) or a vcan pair, with no hardware attached.
    """
    channel: str
    bitrate: int = 1_000_000
    interface: str = "socketcan"


@dataclass(frozen=True)
class JointSpec:
    """One actuator and the software conventions that surround it."""
    name: str
    leg: str
    bus: str
    can_id: int

    # Sign flip mapping hardware frame -> simulation frame (and back).
    direction: float

    # Sim-frame neutral pose. Must equal the training env's default joint pos,
    # because the observation is relative to it and the action is offset by it.
    default_pos: float

    # Hard position bounds in the SIM frame, radians. See `pos_limits` for the
    # hardware-frame view the safety layer uses.
    sim_pos_limits: tuple[float, float]

    # Hard velocity bounds, radians/second. Symmetric, so frame-independent.
    vel_limits: tuple[float, float]

    # Sim-frame angle at which the motor reads zero, radians. See module docstring.
    sim_offset: float = 0.0

    @property
    def key(self) -> tuple[str, int]:
        """Transport-level identity: the channel plus the CAN id on it."""
        return (self.bus, self.can_id)

    @property
    def pos_limits(self) -> tuple[float, float]:
        """Position bounds in the HARDWARE frame (what the motor reports)."""
        a, b = (self.direction * (lim - self.sim_offset) for lim in self.sim_pos_limits)
        return (min(a, b), max(a, b))


@dataclass(frozen=True)
class ImuSpec:
    """Configuration for the base-mounted IMU."""
    i2c_address: int = 0x4A

    # Rotation mapping SENSOR frame -> ROBOT BASE frame. Determine this
    # empirically during bring-up (see main-imu-read.py) rather than guessing.
    mount_rotation: tuple = ((1.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0),
                             (0.0, 0.0, 1.0))

    # A sample older than this is treated as a hardware fault. A frozen IMU
    # feeding a locomotion policy is more dangerous than no IMU at all.
    max_age_s: float = 0.06

    # Target report interval requested from the BNO085, in seconds.
    report_interval_s: float = 0.005

    # How long the reader waits after the first sample before reporting ready.
    # The BNO085's fusion reports an idealised level orientation at first and
    # converged to the real tilt ~0.4 s later on the robot (2026-09-25).
    settle_s: float = 0.6

    @property
    def mount_matrix(self) -> np.ndarray:
        return np.array(self.mount_rotation, dtype=np.float64)


@dataclass(frozen=True)
class SafetySpec:
    """Supervisory limits that are not per-joint."""
    # Maximum tilt of the base away from upright before we e-stop, in radians.
    max_tilt_rad: float = 0.70
    # Maximum base angular rate before we e-stop, in rad/s.
    max_base_ang_vel: float = 12.0
    # Whether IMU-derived limits are enforced at all (disable for bench work).
    enforce_attitude: bool = True


@dataclass
class RobotSpec:
    """The whole machine, assembled and validated."""
    buses: list[BusSpec]
    joints: list[JointSpec]
    actuator_limits: dict          # RS03 scaling bounds (P_MIN, P_MAX, ...)
    host_id: int
    kp: float
    kd: float
    loop_rate_hz: float
    policy_joint_order: list[str]
    imu: Optional[ImuSpec] = None
    safety: SafetySpec = field(default_factory=SafetySpec)
    watchdog_ms: int = 100
    # The sim joint contract (parsed JSON) and the direction_signature the
    # `direction` signs were verified against on hardware. See module docstring.
    sim_contract: Optional[dict] = None
    directions_verified_signature: Optional[str] = None
    # How far past the sim range a joint's sim_pos_limits may reach, radians.
    sim_limit_margin: float = 0.0
    # Whether each joint's sim_offset sign was checked on the robot.
    zero_offsets_verified: bool = True

    def __post_init__(self):
        self._validate()
        # Cached lookups, built once.
        self._index = {j.name: i for i, j in enumerate(self.joints)}
        self._by_key = {j.key: j for j in self.joints}
        self._by_name = {j.name: j for j in self.joints}

    # ---------------------------------------------------------------- checks

    def _validate(self):
        if not self.buses:
            raise ValueError("RobotSpec requires at least one bus.")
        if not self.joints:
            raise ValueError("RobotSpec requires at least one joint.")

        channels = [b.channel for b in self.buses]
        if len(set(channels)) != len(channels):
            raise ValueError(f"Duplicate CAN channel in bus list: {channels}")

        names = [j.name for j in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate joint name: {names}")

        # Transport identity must be unique. With one bus this was implied;
        # with two it is the single most likely wiring mistake.
        keys = [j.key for j in self.joints]
        if len(set(keys)) != len(keys):
            dupes = {k for k in keys if keys.count(k) > 1}
            raise ValueError(f"Duplicate (bus, can_id) pairs: {sorted(dupes)}")

        for j in self.joints:
            if j.bus not in channels:
                raise ValueError(
                    f"Joint '{j.name}' references unknown bus '{j.bus}'. "
                    f"Known buses: {channels}"
                )
            lo, hi = j.sim_pos_limits
            if lo >= hi:
                raise ValueError(
                    f"Joint '{j.name}' has empty sim_pos_limits {j.sim_pos_limits}.")
            lo, hi = j.vel_limits
            if lo >= hi:
                raise ValueError(f"Joint '{j.name}' has empty vel_limits {j.vel_limits}.")
            if j.direction not in (1.0, -1.0):
                raise ValueError(
                    f"Joint '{j.name}' direction must be +1.0 or -1.0, got {j.direction}."
                )

        # The ordering contract, checked explicitly.
        if names != list(self.policy_joint_order):
            raise ValueError(
                "Joint order does not match the trained policy's joint order.\n"
                f"  config JOINTS order : {names}\n"
                f"  POLICY_JOINT_ORDER  : {list(self.policy_joint_order)}\n"
                "These must be identical. POLICY_JOINT_ORDER is copied from the "
                "training environment; reorder JOINTS to match it."
            )

        # A joint whose default pose is already outside its own safety limits
        # would trip the interlock on the first commanded step.
        for j in self.joints:
            lo, hi = j.sim_pos_limits
            if not (lo <= j.default_pos <= hi):
                raise ValueError(
                    f"Joint '{j.name}' default_pos {j.default_pos} is outside "
                    f"sim_pos_limits {j.sim_pos_limits}."
                )

        if self.sim_contract is not None:
            self._validate_against_contract()

    def _validate_against_contract(self):
        contract = self.sim_contract['joints']
        names = [j.name for j in self.joints]
        if set(names) != set(contract):
            raise ValueError(
                "Joint names do not match the sim joint contract.\n"
                f"  config JOINTS : {sorted(names)}\n"
                f"  contract      : {sorted(contract)}"
            )
        # Limits may be tighter than the sim range, never looser than it plus
        # the margin the config allows (a real joint can overshoot slightly).
        margin = self.sim_limit_margin
        for j in self.joints:
            c_lo, c_hi = contract[j.name]['range']
            lo, hi = j.sim_pos_limits
            if lo < c_lo - margin - 1e-9 or hi > c_hi + margin + 1e-9:
                raise ValueError(
                    f"Joint '{j.name}' sim_pos_limits {j.sim_pos_limits} exceed the "
                    f"sim range [{c_lo}, {c_hi}] plus the {margin} rad margin."
                )

    @property
    def directions_verified(self) -> bool:
        """True when the `direction` signs were checked against this sim contract."""
        return (self.sim_contract is not None
                and self.directions_verified_signature
                == self.sim_contract['direction_signature'])

    # --------------------------------------------------------------- lookups

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def channels(self) -> list[str]:
        return [b.channel for b in self.buses]

    @property
    def dt(self) -> float:
        return 1.0 / self.loop_rate_hz

    def index_of(self, name: str) -> int:
        return self._index[name]

    def joint(self, name: str) -> JointSpec:
        return self._by_name[name]

    def joint_by_key(self, bus: str, can_id: int) -> JointSpec:
        return self._by_key[(bus, can_id)]

    def joints_on(self, channel: str) -> list[JointSpec]:
        return [j for j in self.joints if j.bus == channel]

    def can_ids_on(self, channel: str) -> list[int]:
        return [j.can_id for j in self.joints_on(channel)]

    @property
    def leg_names(self) -> list[str]:
        seen: list[str] = []
        for j in self.joints:
            if j.leg not in seen:
                seen.append(j.leg)
        return seen

    def joints_in_leg(self, leg: str) -> list[JointSpec]:
        return [j for j in self.joints if j.leg == leg]

    # ------------------------------------------------- derived array vectors
    # All returned in canonical joint order.

    def default_pos_vector(self) -> np.ndarray:
        return np.array([j.default_pos for j in self.joints], dtype=np.float32)

    def offset_vector(self) -> np.ndarray:
        return np.array([j.sim_offset for j in self.joints], dtype=np.float32)

    def direction_vector(self) -> np.ndarray:
        return np.array([j.direction for j in self.joints], dtype=np.float32)

    def pos_limit_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([j.pos_limits[0] for j in self.joints], dtype=np.float64)
        hi = np.array([j.pos_limits[1] for j in self.joints], dtype=np.float64)
        return lo, hi

    def vel_limit_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([j.vel_limits[0] for j in self.joints], dtype=np.float64)
        hi = np.array([j.vel_limits[1] for j in self.joints], dtype=np.float64)
        return lo, hi

    def describe(self) -> str:
        lines = [f"Robot: {self.num_joints} joints across {len(self.buses)} buses "
                 f"@ {self.loop_rate_hz:g} Hz"]
        for ch in self.channels:
            on_bus = self.joints_on(ch)
            ids = ", ".join(f"{j.can_id}:{j.name}" for j in on_bus)
            lines.append(f"  {ch}: {ids}")
        lines.append(f"  IMU: {'enabled' if self.imu else 'disabled'}")
        return "\n".join(lines)
