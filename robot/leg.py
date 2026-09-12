"""A named group of joints.

A leg is a VIEW, not an owner. It holds no bus and no state of its own -- it
exists so diagnostics, zeroing and per-limb IK can address "the left leg"
without hard-coding which channel or which CAN ids that happens to mean today.
Bus assignment lives in config; rewiring is a data change, not a code change.
"""


class Leg:
    def __init__(self, robot, name: str):
        self.robot = robot
        self.name = name
        self.joints = robot.spec.joints_in_leg(name)
        if not self.joints:
            raise ValueError(f"No joints configured for leg '{name}'.")

    def __repr__(self):
        return f"<Leg {self.name}: {', '.join(self.names)}>"

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def channels(self) -> list[str]:
        seen = []
        for j in self.joints:
            if j.bus not in seen:
                seen.append(j.bus)
        return seen

    def slice_state(self, state: dict) -> dict:
        """Narrow a whole-robot state dict down to this leg's joints."""
        return {n: state[n] for n in self.names}

    def angles(self, state: dict) -> list[float]:
        """This leg's joint positions, in configured order (hip, thigh, calf)."""
        return [state[n]['pos'] for n in self.names]
