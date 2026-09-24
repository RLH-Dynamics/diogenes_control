class HardwareError(Exception):
    """Base class for all physical hardware failures."""

class HardwareIOError(HardwareError):
    """Raised when reading/writing to a physical bus fails (e.g., CAN socket crashes)."""

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

class SafetyLimitError(HardwareError):
    """Raised by the supervisory layer when states violate safe operating bounds."""
    pass