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

ORIENTATION: GAME ROTATION VECTOR
---------------------------------
Orientation comes from the BNO085's GAME rotation vector (accelerometer + gyro
fusion, no magnetometer). Nothing here needs a compass heading -- the tilt
interlock and the walking env's IMU terms use gravity and angular rate only --
and a magnetometer sitting among six motors full of permanent magnets, carrying
changing currents, is more liability than help. Its heading drifts slowly;
tilt is unaffected.

CALIBRATION
-----------
The chip loads its saved calibration from flash at every reset (the driver
resets it on connect), so the full calibration is a one-off
(tools/imu_calibrate.py), redone only if the IMU is remounted. That reset also
turns the chip's own background self-calibration OFF -- measured on the robot:
gyro accuracy stays at 0 until it is re-enabled, then reaches 3 within ~1 s --
so start() turns accelerometer and gyro self-calibration back on every time. `status` is the lower of the
accelerometer's and gyro's own accuracy estimates, 0 (unreliable) to 3 (high).

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
        self.mount = spec.base_from_chip  # nominal axes + the measured pitch trim

        self._sensor = None
        self._i2c = None
        self._thread = None
        self._stop_event = threading.Event()
        # Serialises driver access between the acquisition thread and the
        # calibration commands, which share one I2C conversation.
        self._lock = threading.Lock()
        self.report_accuracy: dict[int, int] = {}
        self._report_ids: dict[str, int] = {}
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
                BNO_REPORT_GAME_ROTATION_VECTOR,
                BNO_REPORT_GYROSCOPE,
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
            self._capture_accuracy(self._sensor)
            self._sensor.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
            self._sensor.enable_feature(BNO_REPORT_GYROSCOPE)
            self._sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
            self._report_ids = {"accel": BNO_REPORT_ACCELEROMETER,
                                "gyro": BNO_REPORT_GYROSCOPE}
        except Exception as e:
            raise HardwareIOError(
                f"Failed to initialise BNO085 at address "
                f"{hex(self.spec.i2c_address)}: {e}"
            )

    def _capture_accuracy(self, sensor):
        """Record each report's accuracy bits, which the driver discards.

        Every sensor report's third byte is a status byte whose low two bits
        are the sensor's own accuracy estimate (0-3). The driver keeps it only
        for the magnetometer, so wrap its report parser to keep it for all.
        """
        original = sensor._process_report

        def process_report(report_id, report_bytes):
            if len(report_bytes) > 2:
                self.report_accuracy[report_id] = report_bytes[2] & 0x3
            original(report_id, report_bytes)

        sensor._process_report = process_report

    def accuracy(self) -> dict:
        """Latest accuracy per sensor, 0 (unreliable) .. 3 (high)."""
        return {name: self.report_accuracy.get(rid, 0)
                for name, rid in self._report_ids.items()}

    def begin_calibration(self):
        """Turn on the chip's calibration for the accelerometer and gyro only.

        The driver's own begin_calibration() also calibrates the magnetometer,
        which this robot does not use (see module docstring).
        """
        from adafruit_bno08x import _ME_CAL_CONFIG
        with self._lock:
            # accel, gyro, mag, subcommand, planar accel, on-table, reserved x3
            self._sensor._send_me_command([1, 1, 0, _ME_CAL_CONFIG, 0, 0, 0, 0, 0])

    def save_calibration(self):
        """Store the current calibration in the chip's flash (loaded at boot)."""
        with self._lock:
            self._sensor.save_calibration_data()

    def start(self, first_sample_timeout: float = 5.0):
        """Open the device and spin up acquisition. Blocks for a first sample."""
        print("[INFO] Starting BNO085 reader...")
        self._open_device()

        self._stop_event.clear()
        # The driver's reset on connect leaves background self-calibration off;
        # without it the gyro bias is never refined (see module docstring).
        self.begin_calibration()

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
              f"{self.sample_count} reads, {self.read_errors} errors); "
              f"settling {self.spec.settle_s:.1f} s...")
        # Its first reports are an idealised level orientation; wait for the
        # fusion to converge so a start-up tilt check sees the real tilt.
        time.sleep(self.spec.settle_s)

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
                with self._lock:
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
        qi, qj, qk, qr = self._sensor.game_quaternion
        gyro = self._sensor.gyro
        accel = self._sensor.acceleration
        t = time.perf_counter()

        quat_sensor = np.array([qr, qi, qj, qk], dtype=np.float64)   # [w, x, y, z]

        # Rotate the vector quantities from the sensor frame into the base frame.
        ang_vel = self.mount @ np.asarray(gyro, dtype=np.float64)
        lin_accel = self.mount @ np.asarray(accel, dtype=np.float64)
        proj_g = self.mount @ projected_gravity_from_quat(quat_sensor)

        status = min(self.accuracy().values())

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
