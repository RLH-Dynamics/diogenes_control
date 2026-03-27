from robstride.robstride import Robstride


class Leg:
    # Strictly a hardware abstraction layer for the robstride actuators.
    # Does not contain previous state information -> this is handled in policy.py!
    def __init__(self, limits: dict, channel, host_id, motor_ids):
        self.robstride = Robstride(channel, host_id, motor_ids)
        self.limits = limits
        return
    
    # Verified as  working.
    # We are not getting any verification that the hardware watchdogs actually worked succesfully.
    # Our previous function to verify that the hardware watchdogs was trying to read a parameter that did not actually get written back.
    # We should try writing a new function to verify that this was succesful.
    def init_leg(self):
        self.robstride.init_CAN_bus()
        self.robstride.enable_and_verify_all_MIT(limits=self.limits)
    
    def get_latest_state_vector(self, target_states: dict, kp: float, kd: float):
        self.robstride.flush_CAN_bus()
        self.robstride.send_all_target_state_vectors(target_states=target_states, kp=kp, kd=kd, limits=self.limits)
        return self.robstride.wait_for_all_replies(limits=self.limits)
    
    def set_output_state_vector(self, physical_targets: dict, kp: float, kd: float):
        self.robstride.send_all_target_state_vectors(target_states=physical_targets, kp=kp, kd=kd, limits=self.limits)
        return
    
    def shutdown(self):
        """
        Safely shuts down the leg by delegating the hardware teardown 
        sequence to the robstride interface.
        """
        print("[INFO] Initiating Leg shutdown sequence...")
        self.robstride.shutdown()
        print("[INFO] Leg shutdown sequence complete.")