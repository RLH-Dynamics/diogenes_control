"""Single source of truth for hardware layout and control configuration.

Everything the robot knows about itself lives here as plain data. Modules take a
`RobotSpec` and never reach back into this file, so a rewire or a retrain is a
config edit rather than a code change.
"""

import json
from pathlib import Path

import numpy as np

from robot.model import BusSpec, ImuSpec, JointSpec, RobotSpec, SafetySpec

### ------------------------------------------------------------- CAN BUSES ###
# The Waveshare 2-CH CAN HAT presents two MCP2515 controllers as can0/can1.
# They are electrically independent buses but share the SPI0 master, so their
# register traffic serialises in the kernel. See README for the dtoverlay setup.
BUSES = [
    BusSpec(channel="can0", bitrate=1_000_000),   # left leg
    BusSpec(channel="can1", bitrate=1_000_000),   # right leg
]

# 0xFD (253) is the host's CAN id. Low ids win arbitration, so keeping the host
# high ensures motor feedback takes priority, and 253 is unlikely to clash with
# a motor id.
HOST_ID = 0xFD

### ---------------------------------------------------------------- JOINTS ###
# Each leg's bus uses the same ids: hip = 1, thigh = 2, calf = 3. Identity is
# (bus, can_id), so the repeat across buses is fine. Verified with
# tools/can_id_scan.py.
#
# ORDER MATTERS. This list defines the observation and action ordering for the
# policy. See the ordering contract in robot/model.py.
#
# `direction` maps hardware frame -> sim frame. The signs below were verified
# on the real robot on 2026-09-24 by moving each joint by hand with
# tools/stream_joints.py + the Viser live joint viewer. The right calf was
# found reversed and flipped to +1. Note the sim's thigh axes are mirrored
# left/right while both calves share one convention, so the calves' signs are
# expected to differ from each other even though the thighs' do not.
#
# That check only holds for the sim joint meanings it was made against, so
# DIRECTIONS_VERIFIED_SIGNATURE records the contract's direction_signature at
# the time. If the sim's joint conventions change, the exported contract gets
# a new signature and Robot.start refuses to apply gain until the mapping test
# is repeated and this value updated. Limp tools still run, so you can.
DIRECTIONS_VERIFIED_SIGNATURE = "51cd6f1d87d8b245"

# `default_pos` is in the SIM frame and must equal the training env's default.
# Position limits are the sim joint ranges (from the contract) widened by
# SIM_LIMIT_MARGIN, since a real joint can overshoot where the sim constraint
# would not. They are converted to the hardware frame through `direction`.
SIM_CONTRACT = json.loads(
    (Path(__file__).parent / "sim_joint_contract.json").read_text())
SIM_LIMIT_MARGIN = 0.05        # rad, ~3 deg


# Zero references, set with tools/calibrate_zeros.py. Each motor's zero is set
# at a physical reference, and `sim_offset` is where that reference sits in the
# sim frame:
#   hips   -- aligned with a straight edge, AT sim zero;
#   thighs -- pushed against the body, 102.9 deg from sim zero;
#   calves -- pushed against their stop, 75 deg from sim zero.
# The magnitudes are geometry; the SIGNS say which side of sim zero each stop
# is on. The signs below come from the first calibration (2026-09-24): just
# before each joint was zeroed at its stop, it read, against the old by-eye
# zero near sim zero, left thigh +100.1, right thigh -100.2, both calves -64.1
# deg (sim frame). Consistent with the sim's mirrored thigh axes and shared
# calf axes. Confirmed on the robot in the live viewer (tools/stream_joints.py)
# on 2026-09-24, at rest and moving each joint. The calves' 64.1 deg was the old
# by-eye zero being ~11 deg off; the 75 deg stop reference is correct.
ZERO_REFERENCE_DEG = {"hip": 0.0, "thigh": 102.9, "calf": 75.0}
ZERO_OFFSET_SIGN = {
    "left_thigh": +1, "left_calf": -1,
    "right_thigh": -1, "right_calf": -1,
}
ZERO_OFFSETS_VERIFIED = True


def sim_offset(name: str) -> float:
    link = name.split("_", 1)[1]
    return np.radians(ZERO_REFERENCE_DEG[link]) * ZERO_OFFSET_SIGN.get(name, 0)


def sim_limits(name: str) -> tuple[float, float]:
    lo, hi = SIM_CONTRACT["joints"][name]["range"]
    return (lo - SIM_LIMIT_MARGIN, hi + SIM_LIMIT_MARGIN)


JOINTS = [
    JointSpec(name="left_hip",    leg="left",  bus="can0", can_id=1,
              direction= 1.0, default_pos=0.0,
              sim_offset=sim_offset("left_hip"),
              sim_pos_limits=sim_limits("left_hip"),    vel_limits=(-20.94, 20.94)),
    JointSpec(name="left_thigh",  leg="left",  bus="can0", can_id=2,
              direction= 1.0, default_pos=0.0,
              sim_offset=sim_offset("left_thigh"),
              sim_pos_limits=sim_limits("left_thigh"),  vel_limits=(-20.94, 20.94)),
    JointSpec(name="left_calf",   leg="left",  bus="can0", can_id=3,
              direction=-1.0, default_pos=0.0,
              sim_offset=sim_offset("left_calf"),
              sim_pos_limits=sim_limits("left_calf"),   vel_limits=(-20.94, 20.94)),

    JointSpec(name="right_hip",   leg="right", bus="can1", can_id=1,
              direction= 1.0, default_pos=0.0,
              sim_offset=sim_offset("right_hip"),
              sim_pos_limits=sim_limits("right_hip"),   vel_limits=(-20.94, 20.94)),
    JointSpec(name="right_thigh", leg="right", bus="can1", can_id=2,
              direction= 1.0, default_pos=0.0,
              sim_offset=sim_offset("right_thigh"),
              sim_pos_limits=sim_limits("right_thigh"), vel_limits=(-20.94, 20.94)),
    JointSpec(name="right_calf",  leg="right", bus="can1", can_id=3,
              direction= 1.0, default_pos=0.0,
              sim_offset=sim_offset("right_calf"),
              sim_pos_limits=sim_limits("right_calf"),  vel_limits=(-20.94, 20.94)),
]

# The training environment's joint ordering: HAROLD_JOINT_NAMES in
# diogenes_mjlab, which is also the MJCF actuator order (checked 2026-09-24).
# RobotSpec refuses to build if JOINTS above disagrees with this, and Policy
# refuses a model whose exported joint_names disagree with it.
POLICY_JOINT_ORDER = [
    "left_hip", "left_thigh", "left_calf",
    "right_hip", "right_thigh", "right_calf",
]

### ----------------------------------------------------- ACTUATOR SCALING ###
# Fixed-point scaling bounds for the RS03's MIT-mode frames. These are protocol
# constants, not safety limits -- per-joint safety lives in JOINTS above.
RS03_LIMITS = {
    'P_MIN': -12.57, 'P_MAX': 12.57,
    'V_MIN': -20.0,  'V_MAX': 20.0,
    'T_MIN': -60.0,  'T_MAX': 60.0,
}

# MIT-mode gains, N.m/rad and N.m.s/rad. Equal to the sim's position actuators
# (kp=60, kv=4 in harold_biped.xml), which training randomised by +-30%.
KP_GAIN = 60.0
KD_GAIN = 4.0

# Motor-side CAN timeout. If the host stops talking for this long the actuators
# go limp on their own. At 50 Hz this is a five-cycle grace period.
WATCHDOG_MS = 100

# main-rl.py rides through a joint's missing status reply by reusing its last
# reading (the motor keeps executing its last command), but stops when this
# many cycles IN A ROW have a miss: 3 cycles = 60 ms, inside WATCHDOG_MS.
# Isolated misses were seen on the real robot on 2026-09-24 with a clean bus.
MAX_CONSECUTIVE_MISSED_REPLIES = 3

### --------------------------------------------------------------- SENSORS ###
# Adafruit BNO085 on the Pi's primary I2C bus. `mount_rotation` maps the sensor
# frame to the robot base frame; the identity below is a placeholder to be
# replaced with the matrix you measure during bring-up.
IMU = ImuSpec(
    i2c_address=0x4A,
    mount_rotation=((1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0)),
    max_age_s=0.06,            # three control cycles at 50 Hz
    report_interval_s=0.005,
)

SAFETY = SafetySpec(
    max_tilt_rad=0.70,         # ~40 degrees off upright
    max_base_ang_vel=12.0,
    enforce_attitude=True,
)

### ----------------------------------------------------------- CONTROL LOOP ###
# Single-rate loop. The policy runs at the same rate the actuators are
# commanded; there is no separate high-rate filtering stage.
LOOP_RATE_HZ = 50
DT = 1.0 / LOOP_RATE_HZ

### ------------------------------------------------------------ RL  POLICY ###
# Matches the Diogenes-Biped-Suspended task (diogenes_mjlab
# harold_biped/env_cfg.py), checked 2026-09-24. Policy verifies every value
# below against the metadata export_onnx.py attaches, and refuses a mismatch.
MODEL_PATH = "policy.onnx"
ACTION_SCALE = 1.0             # JointPositionActionCfg scale, default offset on
CYCLE_PERIOD = 2.0             # gait-clock period (gait.GAIT_PERIOD), seconds

# The actor observation group, in order: each entry is a control-side term
# (control/observation.py), and SIM_OBSERVATION_NAMES the sim's name for it as
# exported in the policy metadata. The suspended robot's torso is welded to the
# world in training, so the actor has no IMU terms.
OBSERVATION_TERMS = [
    "joint_pos_rel",        # num_joints
    "joint_vel_rel",        # num_joints
    "last_action",          # num_joints
    "phase_clock",          # 2
]
SIM_OBSERVATION_NAMES = ["joint_pos", "joint_vel", "last_action", "gait_clock"]

# Past steps stacked per actor term (OBS_HISTORY_LENGTH in diogenes_mjlab).
# Input width = 10 x (6 + 6 + 6 + 2) = 200.
OBSERVATION_HISTORY = 10

# Soft start (control/soft_start.py): before the policy runs, targets ramp from
# the resting pose to START_POSE over SOFT_START_S, then hold for
# SOFT_START_HOLD_S. START_POSE is where training episodes start, the gait's
# nominal pose (diogenes_mjlab gait.nominal_joint_pos(): hips abducted by
# HIP_ABDUCTION = 10 deg, thighs and calves at 0), in the SIM frame.
START_POSE = {
    "left_hip": np.radians(10.0), "left_thigh": 0.0, "left_calf": 0.0,
    "right_hip": np.radians(-10.0), "right_thigh": 0.0, "right_calf": 0.0,
}
SOFT_START_S = 2.0
SOFT_START_HOLD_S = 0.5

### -------------------------------------------------------------- ASSEMBLY ###
SPEC = RobotSpec(
    buses=BUSES,
    joints=JOINTS,
    actuator_limits=RS03_LIMITS,
    host_id=HOST_ID,
    kp=KP_GAIN,
    kd=KD_GAIN,
    loop_rate_hz=LOOP_RATE_HZ,
    policy_joint_order=POLICY_JOINT_ORDER,
    imu=IMU,
    safety=SAFETY,
    watchdog_ms=WATCHDOG_MS,
    sim_contract=SIM_CONTRACT,
    directions_verified_signature=DIRECTIONS_VERIFIED_SIGNATURE,
    sim_limit_margin=SIM_LIMIT_MARGIN,
    zero_offsets_verified=ZERO_OFFSETS_VERIFIED,
)

NUM_JOINTS = SPEC.num_joints
