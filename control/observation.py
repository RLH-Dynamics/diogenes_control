"""Declarative assembly of the policy's observation vector.

WHY THIS IS DATA AND NOT A hand-written concatenate
---------------------------------------------------
The observation layout is the part of this system that changes every time the
policy is retrained. Previously it was a hand-maintained `np.concatenate` plus a
docstring describing the index math. At three joints and four terms that was
tractable; at six joints with IMU terms it is where the bugs come from.

Here the layout is a list of term names in a policy profile
(`config.POLICY_PROFILES`). Each term
declares its own width, the builder sums them, and `Policy` checks that total
against the ONNX model's declared input width before any torque is applied. A
mismatch names the terms rather than just printing two numbers.

CONVENTIONS (carried over from the MJLab deployment notes in the previous
policy.py, and still true):
  * `joint_pos_rel` subtracts the default pose, so `default_pos` must equal the
    sim's default joint positions.
  * `last_action` is the RAW network output from the previous tick, BEFORE
    per-term scale and offset -- not the physical motor target.
  * obs normalisation is baked into the exported ONNX graph, so raw,
    unnormalised observations are fed in.
  * No obs scale or clip is applied; none is configured on the sim side.

OBSERVATION HISTORY
-------------------
With `history_length` H > 0 each term keeps its last H values, reproducing
mjlab's ObservationManager + CircularBuffer exactly:
  * layout is TERM-MAJOR: all H steps of the first term, then all H steps of
    the next, each term's block ordered oldest -> newest;
  * after a reset, the first observation fills every history slot, rather than
    padding with zeros.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ObsContext:
    """Everything a term might need, assembled once per policy tick."""
    sim_pos: np.ndarray          # (n,) joint position, sim frame
    sim_vel: np.ndarray          # (n,) joint velocity, sim frame
    default_pos: np.ndarray      # (n,)
    last_raw_action: np.ndarray  # (n,)
    base_ang_vel: np.ndarray     # (3,) base frame
    projected_gravity: np.ndarray  # (3,) base frame
    imu_ang_vel_chip: np.ndarray   # (3,) the BNO085's own axes
    imu_gravity_chip: np.ndarray   # (3,) the BNO085's own axes
    phase: np.ndarray            # (2,) [sin, cos]
    commands: np.ndarray         # (3,) [vx, vy, wz] velocity command


@dataclass(frozen=True)
class ObsTerm:
    name: str
    width: Callable[[int], int]      # num_joints -> width
    extract: Callable[[ObsContext], np.ndarray]


# The registry of terms this deployment knows how to produce. Adding a term the
# retrained policy needs means adding one entry here, then naming it in
# the profile's observation_terms.
TERM_REGISTRY: dict[str, ObsTerm] = {
    'joint_pos_rel': ObsTerm(
        'joint_pos_rel', lambda n: n,
        lambda c: c.sim_pos - c.default_pos),
    'joint_vel_rel': ObsTerm(
        # default joint velocity is zero, so 'relative' velocity is just velocity
        'joint_vel_rel', lambda n: n,
        lambda c: c.sim_vel),
    'joint_pos': ObsTerm(
        'joint_pos', lambda n: n,
        lambda c: c.sim_pos),
    'joint_vel': ObsTerm(
        'joint_vel', lambda n: n,
        lambda c: c.sim_vel),
    'last_action': ObsTerm(
        'last_action', lambda n: n,
        lambda c: c.last_raw_action),
    'base_ang_vel': ObsTerm(
        'base_ang_vel', lambda n: 3,
        lambda c: c.base_ang_vel),
    'projected_gravity': ObsTerm(
        'projected_gravity', lambda n: 3,
        lambda c: c.projected_gravity),
    # The walking policy reads the IMU as the sim does, at the chip's site in
    # its own axes (+x left, +y up, +z forward on Harold): the base-frame
    # sample rotated back through the mount rotation.
    'imu_ang_vel_chip': ObsTerm(
        'imu_ang_vel_chip', lambda n: 3,
        lambda c: c.imu_ang_vel_chip),
    'imu_gravity_chip': ObsTerm(
        'imu_gravity_chip', lambda n: 3,
        lambda c: c.imu_gravity_chip),
    'phase_clock': ObsTerm(
        'phase_clock', lambda n: 2,
        lambda c: c.phase),
    'velocity_commands': ObsTerm(
        'velocity_commands', lambda n: 3,
        lambda c: c.commands),
}


class ObservationBuilder:
    """Builds the flat observation vector from a declared list of term names."""

    def __init__(self, spec, term_names: list[str], history_length: int = 0):
        unknown = [n for n in term_names if n not in TERM_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown observation term(s): {unknown}. "
                f"Known terms: {sorted(TERM_REGISTRY)}"
            )
        if not term_names:
            raise ValueError("The observation layout is empty.")

        self.spec = spec
        self.num_joints = spec.num_joints
        self.terms = [TERM_REGISTRY[n] for n in term_names]
        self.widths = [t.width(self.num_joints) for t in self.terms]
        self.history_length = int(history_length)
        self.steps = max(1, self.history_length)
        self.frame_width = sum(self.widths)
        self.total_width = self.frame_width * self.steps

        # Per-term history, rows oldest -> newest. Filled on the first build()
        # after a reset.
        self._history = [np.zeros((self.steps, w), dtype=np.float32)
                         for w in self.widths]
        self._primed = False

        needs_imu = {'base_ang_vel', 'projected_gravity',
                     'imu_ang_vel_chip', 'imu_gravity_chip'}
        self.requires_imu = bool(needs_imu.intersection(term_names))
        if self.requires_imu and spec.imu is None:
            raise ValueError(
                f"Observation layout includes IMU terms "
                f"{sorted(needs_imu.intersection(term_names))} but no IMU is "
                f"configured in the robot spec."
            )

        self._buffer = np.zeros((1, self.total_width), dtype=np.float32)

    def describe(self) -> str:
        history = (f", history {self.history_length} x {self.frame_width}"
                   if self.history_length else "")
        lines = [f"Observation layout ({self.total_width} dims{history}):"]
        offset = 0
        for term, width in zip(self.terms, self.widths):
            span = width * self.steps
            steps = f"  ({self.steps} x {width}, oldest first)" if self.history_length else ""
            lines.append(f"  [{offset:>3}:{offset + span:>3}] {term.name}{steps}")
            offset += span
        return "\n".join(lines)

    def reset(self):
        """Start a new episode: the next build() refills every history slot."""
        self._primed = False

    def build(self, ctx: ObsContext) -> np.ndarray:
        """Advance the history by one step and return the (1, total_width)
        observation. The returned buffer is reused between calls."""
        offset = 0
        for term, width, hist in zip(self.terms, self.widths, self._history):
            value = np.asarray(term.extract(ctx), dtype=np.float32).ravel()
            if value.size != width:
                raise ValueError(
                    f"Observation term '{term.name}' produced {value.size} values, "
                    f"expected {width}."
                )
            if self._primed:
                hist[:-1] = hist[1:]
                hist[-1] = value
            else:
                hist[:] = value
            span = width * self.steps
            self._buffer[0, offset:offset + span] = hist.ravel()
            offset += span
        self._primed = True
        return self._buffer
