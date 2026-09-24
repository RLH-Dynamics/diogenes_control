"""Context manager wrapping robot bring-up and guaranteed teardown.

Every entry-point script previously open-coded the same instantiate / init /
try / finally / shutdown sequence. With two buses and an IMU that boilerplate
roughly doubles, and every copy is somewhere teardown can be forgotten. Using
the session makes the disable-and-verify path run on EVERY exit, including
exceptions and Ctrl-C.

    with RobotSession(SPEC) as robot:
        state = robot.read_limp_state()
"""

from robot.robot import Robot
from sensors.base import NullImu
from utils.realtime import configure_realtime


class RobotSession:
    def __init__(self, spec, control_mode: str = 'MIT', use_imu: bool = True,
                 imu=None, realtime: bool = True, limp_only: bool = False,
                 allow_unverified_watchdogs: bool = False):
        """`limp_only`: the caller never applies gain, so the start-up gates
        (watchdogs, verified joint directions) warn instead of refusing. See
        Robot.start. `allow_unverified_watchdogs` relaxes only the watchdog gate.
        """
        self.spec = spec
        self.control_mode = control_mode
        self.realtime = realtime
        self.require_watchdogs = not (limp_only or allow_unverified_watchdogs)
        self.require_verified_directions = not limp_only

        if imu is not None:
            self.imu = imu
        elif use_imu and spec.imu is not None:
            # Imported here so a bench run with use_imu=False never needs the
            # CircuitPython stack installed.
            from sensors.imu import Bno085Reader
            self.imu = Bno085Reader(spec.imu)
        else:
            self.imu = NullImu()

        self.robot = None

    def __enter__(self) -> Robot:
        if self.realtime:
            configure_realtime()
        self.robot = Robot(self.spec, imu=self.imu)
        try:
            self.robot.start(
                control_mode=self.control_mode,
                require_watchdogs=self.require_watchdogs,
                require_verified_directions=self.require_verified_directions,
            )
        except Exception:
            # A partially-open robot still has motors that may be enabled.
            self.robot.shutdown()
            raise
        return self.robot

    def __exit__(self, exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt -- shutting down.")
        self.robot.shutdown()
        # Suppress only Ctrl-C; everything else propagates to the caller.
        return exc_type is KeyboardInterrupt
