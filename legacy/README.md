# Legacy scripts

These target the pre-dual-bus API (`robot.leg.Leg(limits, channel, host_id,
motor_ids)`, the flat `JOINT_CONFIG`/`CAN_CHANNEL` constants, and the 200 Hz
filtered loop). They are kept for reference because several encode hard-won
characterisation work, but **none of them run against the current codebase.**

| script | what it did | to revive |
|---|---|---|
| `main-coulomb.py` | Coulomb friction identification, TORQUE mode | port to `RobstrideBus` + `RobotSession`; per-joint, so scope it with `--joint` |
| `main-viscous.py` | Viscous friction identification, MIT mode | same |
| `main-ik-step.py` | 3D stepping trajectory via IK | use `robot.kinematics.calculate_leg_ik` and a `Leg` view; `is_left_stance` already selects the leg |
| `main-ik-linear.py` | Linear foot trajectory via IK | same |
| `main-ik-hold-still.py` | Position hold at the startup pose | largely superseded by `main-rl.py --dry-run` |
| `main-ik-passive-read-joints.py` | Passive joint state print | superseded by `main-read-state.py` |

The duplicated `calculate_leg_ik` from these scripts now lives in
`robot/kinematics.py`; `ik-visualizer.py` (pure simulation, no hardware) still
works and has been pointed at it.
