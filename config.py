### HARDWARE CONFIG ###
CAN_CHANNEL = "can0"
HOST_ID = 0xFD # 0xFD is generally the preferred default CAN ID for the host. Lower ID numbers have priority, ensuring that the motors' feedback wins. 0xFD (253 in decimal) is unlikely to clash with commonly used motor CAN IDs.
JOINT_CONFIG = {
    'hip':   {'id': 1, 'direction': 1.0, 'pos_limits': (-0.785, 0.785), 'vel_limits':(-20.94, 20.94)}, # pos_limits: radians
    'thigh': {'id': 2, 'direction': 1.0, 'pos_limits': (-1.57, 0.261), 'vel_limits':(-20.94, 20.94)},  # vel_limits: radians/second
    'knee':  {'id': 3, 'direction': -1.0, 'pos_limits': (-1.57, 0.0), 'vel_limits':(-20.94, 20.94)}
}
RS03_LIMITS = {
    'P_MIN': -12.57,
    'P_MAX': 12.57,
    'V_MIN': -20.0,
    'V_MAX': 20.0,
    'T_MIN': -60.0,
    'T_MAX': 60.0
}
KP_GAIN = 6.0
KD_GAIN = 0.4

### MODEL CONFIG ###
CYCLE_PERIOD = 2.0
MODEL_PATH = "policy.onnx"
NUM_JOINTS = 3
LOOP_RATE_HZ = 200
DT = 1.0 / LOOP_RATE_HZ
POLICY_UPDATE_INTERVAL = 4
LPF_ALPHA = 1.0
ACTION_LPF_ALPHA = 1.0
HISTORY_LEN = 10
ACTION_SCALE = 1.0
DEFAULT_POS = [0.0, 0.0, 0.0]