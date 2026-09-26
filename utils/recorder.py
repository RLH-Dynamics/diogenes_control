"""CSV logging that derives its own schema from the robot spec.

The previous implementation hand-built its header list inline in the control
loop. With six joints plus IMU channels that is a lot of index bookkeeping to
maintain by hand, and it silently goes stale whenever the robot changes.
"""

import csv
from datetime import datetime


class Recorder:
    def __init__(self, spec, log_imu: bool = True):
        self.spec = spec
        self.log_imu = log_imu and spec.imu is not None
        self.rows = []

        headers = ['time', 'exchange_ms', 'overrun_ms', 'missed']
        for name in spec.names:
            headers += [f'meas_pos_{name}', f'meas_vel_{name}',
                        f'meas_torque_{name}', f'temp_{name}', f'cmd_pos_{name}']
        if self.log_imu:
            headers += ['imu_age_ms', 'imu_tilt_rad', 'imu_status']
            headers += [f'imu_ang_vel_{a}' for a in 'xyz']
            headers += [f'imu_proj_grav_{a}' for a in 'xyz']
            headers += [f'imu_quat_{a}' for a in 'wxyz']
        self.headers = headers

    def record(self, t, state, targets, imu_sample=None,
               exchange_s=0.0, overrun_s=0.0, missed=()):
        """`missed`: joints with no reply this cycle, whose `state` entry is
        their previous reading (see Robot.exchange_tolerant)."""
        row = {
            'time': t,
            'exchange_ms': exchange_s * 1000.0,
            'overrun_ms': overrun_s * 1000.0,
            'missed': ";".join(missed),
        }
        for name in self.spec.names:
            reading = state[name]
            row[f'meas_pos_{name}'] = reading['pos']
            row[f'meas_vel_{name}'] = reading['vel']
            row[f'meas_torque_{name}'] = reading.get('torque', 0.0)
            row[f'temp_{name}'] = reading.get('temp', 0.0)
            row[f'cmd_pos_{name}'] = targets[name]['pos']

        if self.log_imu and imu_sample is not None:
            row['imu_age_ms'] = imu_sample.age() * 1000.0
            row['imu_tilt_rad'] = imu_sample.tilt_rad
            row['imu_status'] = imu_sample.status
            for i, axis in enumerate('xyz'):
                row[f'imu_ang_vel_{axis}'] = float(imu_sample.ang_vel[i])
                row[f'imu_proj_grav_{axis}'] = float(imu_sample.projected_gravity[i])
            for i, axis in enumerate('wxyz'):
                row[f'imu_quat_{axis}'] = float(imu_sample.quat[i])

        self.rows.append(row)

    def save(self, prefix: str = "rl_log") -> str | None:
        if not self.rows:
            print("[INFO] No data recorded; nothing to save.")
            return None

        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\n[INFO] Saving {len(self.rows)} samples to {filename}...")
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.rows)
        except IOError as e:
            print(f"[ERROR] Failed to save log: {e}")
            return None
        print("[INFO] Log saved.")
        return filename
