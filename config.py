"""Single source of truth for hardware layout and control configuration.

Everything the robot knows about itself lives here as plain data. Modules take a
`RobotSpec` and never reach back into this file, so a rewire or a retrain is a
config edit rather than a code change.
"""

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
# ORDER MATTERS. This list defines the observation and action ordering for the
# policy. See the ordering contract in robot/model.py.
#
# `direction` maps hardware frame -> sim frame.
# `default_pos` is in the SIM frame and must equal the training env's default.
# `pos_limits` / `vel_limits` are in the HARDWARE frame (what the motor reports).
#
# TODO(bring-up): the right leg currently copies the left leg's direction signs
# and limits verbatim. If the right leg is mechanically mirrored, some of these
# signs and the asymmetric thigh/calf limits need flipping. Confirm joint by
# joint with main-imu-read.py / main-read-state.py before enabling torque.
JOINTS = [
    JointSpec(name="left_hip",    leg="left",  bus="can0", can_id=1,
              direction= 1.0, default_pos=0.0,
              pos_limits=(-0.785, 0.785), vel_limits=(-20.94, 20.94)),
    JointSpec(name="left_thigh",  leg="left",  bus="can0", can_id=2,
              direction= 1.0, default_pos=0.0,
              pos_limits=(-1.57, 0.261),  vel_limits=(-20.94, 20.94)),
    JointSpec(name="left_calf",   leg="left",  bus="can0", can_id=3,
              direction=-1.0, default_pos=0.0,
              pos_limits=(-1.57, 0.018),  vel_limits=(-20.94, 20.94)),

    JointSpec(name="right_hip",   leg="right", bus="can1", can_id=4,
              direction= 1.0, default_pos=0.0,
              pos_limits=(-0.785, 0.785), vel_limits=(-20.94, 20.94)),
    JointSpec(name="right_thigh", leg="right", bus="can1", can_id=5,
              direction= 1.0, default_pos=0.0,
              pos_limits=(-1.57, 0.261),  vel_limits=(-20.94, 20.94)),
    JointSpec(name="right_calf",  leg="right", bus="can1", can_id=6,
              direction=-1.0, default_pos=0.0,
              pos_limits=(-1.57, 0.018),  vel_limits=(-20.94, 20.94)),
]

# Copied verbatim from the training environment's joint ordering. RobotSpec
# refuses to build if JOINTS above disagrees with this.
#
# TODO(retrain): confirm against the new MJLab env config. Some frameworks group
# by joint type (all hips, then all thighs, then all calves) rather than by leg.
# Getting this wrong swaps limbs silently -- it is the highest-consequence
# single line in this file.
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

KP_GAIN = 60.0
KD_GAIN = 4.0

# Motor-side CAN timeout. If the host stops talking for this long the actuators
# go limp on their own. At 50 Hz this is a five-cycle grace period.
WATCHDOG_MS = 100

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
MODEL_PATH = "policy.onnx"
ACTION_SCALE = 1.0
CYCLE_PERIOD = 2.0             # phase-clock period, seconds

# The actor observation layout, in order. Each entry is a term name understood
# by control/observation.py. The summed widths are checked against the ONNX
# model's declared input width at startup, so a mismatch fails before any torque
# is applied.
#
# TODO(retrain): copy this from the new MJLab actor observation group. The list
# below is the expected six-joint + IMU layout, NOT a verified one.
OBSERVATION_TERMS = [
    "joint_pos_rel",        # num_joints
    "joint_vel_rel",        # num_joints
    "base_ang_vel",         # 3   (IMU)
    "projected_gravity",    # 3   (IMU)
    "last_action",          # num_joints
    "phase_clock",          # 2
]

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
)

NUM_JOINTS = SPEC.num_joints
