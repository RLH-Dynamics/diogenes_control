class HardwareError(Exception):
    """Base class for all physical hardware failures."""

class HardwareIOError(HardwareError):
    """Raised when reading/writing to a physical bus fails (e.g., CAN socket crashes)."""

class MissedReplies(HardwareIOError):
    """Raised by RobstrideNetwork.gather when some joints did not reply in time.

    Carries what did arrive, so a caller that can tolerate an isolated miss
    (Robot.exchange_tolerant) can carry on:
      missing -- joint names with no status reply this cycle
      state   -- name -> state for the joints that did reply (hardware frame)
      stray   -- other frames the missing motors sent this cycle, e.g. a fault
                 report instead of a status reply, as readable strings
    """

    def __init__(self, message, missing, state, stray):
        super().__init__(message)
        self.missing = missing
        self.state = state
        self.stray = stray


class ParameterRejected(HardwareIOError):
    """Raised when a motor answers a parameter read with its failure flag set.

    Typically the parameter does not exist in that motor's firmware. Without
    this check the reply's zero-filled payload reads as a genuine value of 0.
    """

class WatchdogUnverifiedError(HardwareError):
    """Raised before enabling when a motor-side CAN timeout could not be confirmed."""

class DirectionsUnverifiedError(HardwareError):
    """Raised before enabling when the joint direction signs were verified
    against a different sim joint contract than the one now loaded."""

class ActuatorFault(HardwareError):
    """Raised when a motor reports an internal hardware fault (e.g., overtemp)."""
    pass

class MotorDisabled(ActuatorFault):
    """Raised when a motor that should be running reports itself disabled.

    Its status replies still arrive, so without this check a motor that has
    dropped out (e.g. its CAN timeout fired) looks healthy while producing no
    torque.
    """


class SafetyLimitError(HardwareError):
    """Raised by the supervisory layer when states violate safe operating bounds."""
    pass