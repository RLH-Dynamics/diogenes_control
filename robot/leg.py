from robstride.robstride import Robstride

class Leg:
    # Strictly a hardware abstraction layer for the robstride actuators.
    # Does not contain previous state information -> this is handled in policy.py!
    def __init__(self, limits: dict, channel: str, host_id: int, motor_ids: list, control_mode: str = 'MIT'):
        self.robstride = Robstride(channel, host_id, motor_ids)
        self.limits = limits
        self.control_mode = control_mode
        return
    
    # Verified as working.
    # We are not getting any verification that the hardware watchdogs actually worked succesfully.
    # Our previous function to verify that the hardware watchdogs was trying to read a parameter that did not actually get written back.
    # We should try writing a new function to verify that this was succesful.
    def init_leg(self):
        self.robstride.init_CAN_bus()
        self.robstride.enable_and_verify_all(limits=self.limits, control_mode=self.control_mode)
    
    def get_latest_state_vector(self, target_states: dict, kp: float, kd: float):
        """
        Flushes the CAN bus, sends MIT target states to trigger a status reply, 
        and waits to collect the full state vector from all actuators.
        """
        self.robstride.flush_CAN_bus()
        self.robstride.send_all_target_state_vectors(target_states=target_states, kp=kp, kd=kd, limits=self.limits)
        return self.robstride.wait_for_all_replies(limits=self.limits)
    
    def get_position_and_velocity(self, motor_id: int) -> tuple[float, float]:
        """
        Reads the current position (rad) and velocity (rad/s) of a specific actuator.
        Works in any control mode.
        """
        return self.robstride.get_position_and_velocity(motor_id)
    
    def set_output_state_vector(self, physical_targets: dict, kp: float, kd: float):
        """Sends target state vectors to all actuators configured in MIT mode."""
        self.robstride.send_all_target_state_vectors(target_states=physical_targets, kp=kp, kd=kd, limits=self.limits)
        return

    def set_output_torques(self, torque_targets: dict):
        """Sends target torques to all actuators configured in TORQUE mode."""
        self.robstride.set_output_torques(torque_targets=torque_targets, limits=self.limits)
        return

    def set_output_velocities(self, velocity_targets: dict):
        """Sends target velocities to all actuators configured in VELOCITY mode."""
        self.robstride.set_output_velocities(velocity_targets=velocity_targets, limits=self.limits)
        return
    
    def shutdown(self):
        """
        Safely shuts down the leg by delegating the hardware teardown 
        sequence to the robstride interface.
        """
        print("[INFO] Initiating Leg shutdown sequence...")
        self.robstride.shutdown()
        print("[INFO] Leg shutdown sequence complete.")