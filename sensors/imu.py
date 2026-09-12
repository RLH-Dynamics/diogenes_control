"""Adafruit BNO085 reader, running off the control loop's critical path.

WHY THIS IS A THREAD
--------------------
The CircuitPython BNO08x driver performs multi-transaction I2C reads and can
block for tens of milliseconds when the sensor is busy. Calling it inline would
blow the control loop's budget and, at worst, trip the motor-side CAN watchdogs.
So acquisition runs in its own thread and publishes an immutable `ImuSample` by
atomic reference swap; the control loop's `latest()` never blocks and never sees
a partially written sample.

GIL NOTE
--------
CPython's default thread switch interval is 5 ms. On a 20 ms control cycle that
is a meaningful share of the budget if this thread holds the GIL through a burst
of driver work. utils/realtime.py lowers the switch interval at startup. If the
loop's timing tail still degrades once this is live, the next move is to run
acquisition in a separate PROCESS publishing through shared memory -- the
`ImuSource` protocol makes that a one-file swap.

HARDWARE NOTES
--------------
  * The Pi's I2C controller handles the BNO085's clock stretching poorly. Set
    `dtparam=i2c_arm_baudrate=50000` (or lower) in /boot/firmware/config.txt.
  * Intermittent OSError/RuntimeError from the driver is expected, not
    exceptional. They are counted, not raised; sustained failure surfaces as
    sample staleness, which the safety layer already treats as a fault.
"""

import threading
import time

import numpy as np

from sensors.base import ImuSample, projected_gravity_from_quat
from utils.exceptions import HardwareIOError


class Bno085Reader:
    """Threaded BNO085 reader publishing base-frame samples."""

    def __init__(self, spec, i2c_frequency: int = 50_000):
        self.spec = spec
        self.i2c_frequency = i2c_frequency
        self.mount = spec.mount_matrix

        self._sensor = None
        self._i2c = None
        self._thread = None
        self._stop_event = threading.Event()
        self._sample: ImuSample | None = None

        # Diagnostics, read by the bring-up script and the recorder.
        self.read_errors = 0
        self.sample_count = 0

    # ------------------------------------------------------------ lifecycle --

    def _open_device(self):
        # Imported lazily so this module stays importable on a dev machine with
        # no CircuitPython stack and no I2C bus.
        try:
            import board
            import busio
            from adafruit_bno08x import (
                BNO_REPORT_ACCELEROMETER,
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_ROTATION_VECTOR,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
        except ImportError as e:
            raise HardwareIOError(
                "BNO085 support requires the Adafruit CircuitPython stack "
                "(adafruit-circuitpython-bno08x, adafruit-blinka). "
                f"Import failed: {e}"
            )

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA, frequency=self.i2c_frequency)
            self._sensor = BNO08X_I2C(self._i2c, address=self.spec.i2c_address)
            self._sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self._sensor.enable_feature(BNO_REPORT_GYROSCOPE)
            self._sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
        except Exception as e:
            raise HardwareIOError(
                f"Failed to initialise BNO085 at address "
                f"{hex(self.spec.i2c_address)}: {e}"
            )

    def start(self, first_sample_timeout: float = 5.0):
        """Open the device and spin up acquisition. Blocks for a first sample."""
        print("[INFO] Starting BNO085 reader...")
        self._open_device()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="imu-reader", daemon=True)
        self._thread.start()

        deadline = time.perf_counter() + first_sample_timeout
        while self._sample is None:
            if time.perf_counter() > deadline:
                self.stop()
                raise HardwareIOError(
                    f"BNO085 produced no sample within {first_sample_timeout:.1f} s "
                    f"({self.read_errors} read errors). Check wiring and "
                    f"i2c_arm_baudrate."
                )
            time.sleep(0.01)

        print(f"[INFO] BNO085 online (first sample after "
              f"{self.sample_count} reads, {self.read_errors} errors).")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
        self._sensor = None
        print("[INFO] BNO085 reader stopped.")

    # ------------------------------------------------------------ acquisition --

    def _run(self):
        interval = self.spec.report_interval_s
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()
            try:
                self._sample = self._read_once()
                self.sample_count += 1
            except Exception:
                # Driver hiccups are routine on this part. Count and continue;
                # sustained failure shows up downstream as a stale sample.
                self.read_errors += 1

            elapsed = time.perf_counter() - cycle_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _read_once(self) -> ImuSample:
        # Adafruit reports the rotation vector as (i, j, k, real).
        qi, qj, qk, qr = self._sensor.quaternion
        gyro = self._sensor.gyro
        accel = self._sensor.acceleration
        t = time.perf_counter()

        quat_sensor = np.array([qr, qi, qj, qk], dtype=np.float64)   # [w, x, y, z]

        # Rotate the vector quantities from the sensor frame into the base frame.
        ang_vel = self.mount @ np.asarray(gyro, dtype=np.float64)
        lin_accel = self.mount @ np.asarray(accel, dtype=np.float64)
        proj_g = self.mount @ projected_gravity_from_quat(quat_sensor)

        status = int(getattr(self._sensor, "quaternion_accuracy", 0) or 0)

        return ImuSample(
            t=t,
            quat=quat_sensor,
            ang_vel=ang_vel,
            lin_accel=lin_accel,
            projected_gravity=proj_g,
            status=status,
        )

    # ---------------------------------------------------------------- access --

    def latest(self) -> ImuSample:
        """Most recent sample. Never blocks; never returns a partial object."""
        sample = self._sample
        if sample is None:
            raise HardwareIOError("IMU has not produced a sample yet.")
        return sample
