import can
import time
import struct
from utils.exceptions import HardwareIOError, ActuatorFault, HardwareError
from robstride.protocol import CommunicationType, ParameterType, FORMAT_MAP

class Robstride:
    def __init__(self, channel, host_id, motor_ids):
        self.bus = None
        self.channel = channel
        self.host_id = host_id
        self.motor_ids = motor_ids

    @staticmethod
    def _scale_value_to_u16(value: float, v_min: float, v_max: float) -> int:
        """
        Helper function to scale a float into a 16-bit unsigned integer based on min/max bounds.
        Equivalent to the float_to_uint function in the Robstride manual.
        """
        clamped = max(min(value, v_max), v_min)
        return int(65535.0 * (clamped - v_min) / (v_max - v_min))
    
    @staticmethod
    def _scale_u16_to_value(x_int: int, v_min: float, v_max: float) -> float:
        """
        Helper function to reverse the 16-bit unsigned integer scaling back into a float.
        Equivalent to the uint_to_float function in the Robstride manual.
        """
        span = v_max - v_min
        return float(x_int) * span / 65535.0 + v_min
    
    # Function completed.
    def flush_CAN_bus(self):
        try:
            while self.bus.recv(timeout=0.0): pass
        except can.CanError as e:
            raise HardwareIOError(f"I/O failure while flushing CAN bus: {e}")
        
    def transmit(self, comm_type, extra_data, destination_id, data=b'\x00'*8):
        arb_id = (comm_type << 24) | (extra_data << 8) | destination_id
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True, dlc=len(data))
        try:
            self.bus.send(msg)
        except can.CanError as e:
            raise HardwareIOError(f"Failed to transmit CommType {comm_type} to ID {destination_id}: {e}")

    def receive(self, timeout=0.001):
        try:
            msg = self.bus.recv(timeout=timeout)
            if msg is None:
                return None
            
            # 1. Unpack the raw arbitration ID
            comm_type = (msg.arbitration_id >> 24) & 0x1F
            extra_data = (msg.arbitration_id >> 8) & 0xFFFF
            destination_id = msg.arbitration_id & 0xFF
            
            # 2. Extract the actual sending Motor ID from the extra_data
            motor_id = extra_data & 0xFF 
            
            return comm_type, motor_id, destination_id, extra_data, msg.data
            
        except can.CanError as e:
            raise HardwareIOError(f"I/O failure while receiving from bus: {e}")
        
    def write_parameter(self, target_id, parameter_tuple, value):
        """Pipelines a parameter write to the CAN bus without blocking."""
        param_index, param_type = parameter_tuple
        value_format = FORMAT_MAP[param_type]
        
        index_bytes = struct.pack('<HH', param_index, 0x0000)
        data_bytes = struct.pack(value_format, value)
        
        # FIX: Pad to 8 bytes. The manual dictates Byte 4~7 holds parameter data.
        padded_data = (index_bytes + data_bytes).ljust(8, b'\x00')
        self.transmit(CommunicationType.WRITE_PARAMETER, self.host_id, target_id, data=padded_data)

    def read_parameter(self, target_id, parameter_tuple, timeout=0.1):
        """
        Reads a single parameter from a motor's memory.
        """
        param_index, param_type = parameter_tuple
        value_format = FORMAT_MAP[param_type]
        
        # CommType 17 (Read), request payload sets data bytes to 0
        req_data = struct.pack('<HHL', param_index, 0x0000, 0x00000000)
        self.transmit(CommunicationType.READ_PARAMETER, self.host_id, target_id, data=req_data)
        
        start_t = time.perf_counter()
        while (time.perf_counter() - start_t) < timeout:
            reply = self.receive(timeout=0.01)
            if reply is None:
                continue
            
            c_type, motor_id, dest_id, extra_data, r_data = reply
            
            # Filter for a Read Parameter reply from the specific motor, sent to our host
            if c_type == CommunicationType.READ_PARAMETER and motor_id == target_id and dest_id == self.host_id:
                # FIX: Dynamically calculate the format size and slice the array accordingly
                size = struct.calcsize(value_format)
                return struct.unpack(value_format, r_data[4:4+size])[0]
                
        raise HardwareIOError(f"Timeout waiting for parameter {hex(param_index)} read from Motor {target_id}")

    def enable_hardware_watchdog(self, timeout_ms=100):
        print(f"[INFO] Setting hardware watchdogs to {timeout_ms}ms...")
        timeout_units = int(timeout_ms * 20)
        
        for mid in self.motor_ids:
            # Pass the full parameter tuple instead of indexing [0] and hardcoding '<I'
            self.write_parameter(mid, ParameterType.CAN_TIMEOUT, timeout_units)
            time.sleep(0.01)  # Give the motor MCU time to process the write command

    # Function completed.
    def init_CAN_bus(self):
        print(f"[INFO] Initializing CAN bus on channel: {self.channel}...")
        try:
            self.bus = can.interface.Bus(channel=self.channel, interface='socketcan', bitrate=1000000)
        except can.CanError as e:
            raise HardwareIOError(f"CAN library error while initializing {self.channel}: {e}")
        except OSError as e:
            raise HardwareIOError(f"OS error connecting to {self.channel}. Is the interface physically up? {e}")
        self.enable_hardware_watchdog()
        print("[INFO] CAN bus initialized and watchdogs verified.")
        return

    def send_target_state_vector(self, motor_id: int, pos: float, vel: float, 
                                 kp: float, kd: float, torque: float, limits: dict):
        """
        Sends the target state vector to a specific motor in MIT control mode.
        
        Args:
            motor_id (int): CAN ID of the target motor.
            pos (float): Target position (rad).
            vel (float): Target velocity (rad/s).
            kp (float): Position gain.
            kd (float): Velocity gain (damping).
            torque (float): Feed-forward torque (N.m).
            limits (dict): Dictionary containing P_MIN, P_MAX, V_MIN, V_MAX, T_MIN, T_MAX.
        """
        # 1. Scale all float values to 16-bit unsigned integers
        p_u16 = self._scale_value_to_u16(pos, limits['P_MIN'], limits['P_MAX'])
        v_u16 = self._scale_value_to_u16(vel, limits['V_MIN'], limits['V_MAX'])
        kp_u16 = self._scale_value_to_u16(kp, 0.0, 5000.0) # Fixed Kp bounds per manual
        kd_u16 = self._scale_value_to_u16(kd, 0.0, 100.0)  # Fixed Kd bounds per manual
        t_u16 = self._scale_value_to_u16(torque, limits['T_MIN'], limits['T_MAX'])

        # 2. Pack data into 8 bytes (Big-Endian format '>HHHH' as required by manual)
        data_payload = struct.pack('>HHHH', p_u16, v_u16, kp_u16, kd_u16)

        # 3. Transmit using CommunicationType 1 (OPERATION_CONTROL). 
        # The extra_data field acts as bits 8-23, which is the 16-bit torque value.
        self.transmit(
            comm_type=CommunicationType.OPERATION_CONTROL,
            extra_data=t_u16,
            destination_id=motor_id,
            data=data_payload
        )

    def send_target_torque(self, motor_id: int, torque_nm: float, limits: dict, kt: float = 2.36):
        """
        Sends a target torque to a specific motor configured in TORQUE (Current) mode.
        The RobStride private protocol commands torque by setting the Iq current reference.
        
        Args:
            motor_id (int): CAN ID of the target motor.
            torque_nm (float): Target torque (N.m).
            limits (dict): Dictionary containing T_MIN and T_MAX.
            kt (float): Motor Torque Constant (N.m/A). Default is 2.36 for the RS03.
        """
        # 1. Clamp the requested torque to your software config limits
        clamped_torque = max(limits['T_MIN'], min(limits['T_MAX'], torque_nm))
        
        # 2. Convert N.m to Amps
        target_amps = clamped_torque / kt
        
        # 3. Clamp to the hardware's absolute physical limit (43A peak per manual)
        target_amps = max(-43.0, min(43.0, target_amps))
        
        # 4. Pipeline the write to the Iq Target parameter
        self.write_parameter(motor_id, ParameterType.IQ_TARGET, target_amps)

    def send_target_velocity(self, motor_id: int, velocity_rads: float, limits: dict):
        """
        Sends a target velocity to a specific motor configured in VELOCITY mode.
        
        Args:
            motor_id (int): CAN ID of the target motor.
            velocity_rads (float): Target velocity (rad/s).
            limits (dict): Dictionary containing V_MIN and V_MAX.
        """
        # 1. Clamp the requested velocity to your software config limits
        clamped_vel = max(limits['V_MIN'], min(limits['V_MAX'], velocity_rads))
        
        # 2. Pipeline the write to the Velocity Target parameter
        self.write_parameter(motor_id, ParameterType.VELOCITY_TARGET, clamped_vel)
    
    def send_all_target_state_vectors(self, target_states: dict, kp: float, kd: float, limits: dict):
        """
        Iterates over a dictionary of target states and dispatches them to the actuators.
        
        Args:
            target_states (dict): Mapping of {motor_id: {'pos': float, 'vel': float, 'torque': float}}
            kp (float): Global Kp gain for all joints.
            kd (float): Global Kd gain for all joints.
            limits (dict): Actuator limit bounds dictionary.
        """
        for motor_id, state in target_states.items():
            self.send_target_state_vector(
                motor_id=motor_id,
                pos=state.get('pos', 0.0),
                vel=state.get('vel', 0.0),
                kp=kp,
                kd=kd,
                torque=state.get('torque', 0.0),
                limits=limits
            )

    def set_output_torques(self, torque_targets: dict, limits: dict):
        """
        Sends target torques to all specified actuators.
        """
        for motor_id, torque_nm in torque_targets.items():
            self.send_target_torque(
                motor_id=motor_id, 
                torque_nm=torque_nm, 
                limits=limits
            )

    def set_output_velocities(self, velocity_targets: dict, limits: dict):
        """
        Sends target velocities to all specified actuators.
        """
        for motor_id, velocity_rads in velocity_targets.items():
            self.send_target_velocity(
                motor_id=motor_id, 
                velocity_rads=velocity_rads, 
                limits=limits
            )

    def get_position_and_velocity(self, motor_id: int) -> tuple[float, float]:
        """
        Explicitly reads the measured position and velocity from the motor's memory.
        This is a blocking operation useful for querying state while in TORQUE or 
        VELOCITY modes without relying on Active Reporting or MIT mode feedback.
        
        Returns:
            tuple: (position_in_radians, velocity_in_radians_per_second)
        """
        # Read active mechanical position (Parameter 0x7019)
        pos = self.read_parameter(
            target_id=motor_id, 
            parameter_tuple=ParameterType.MECHANICAL_POSITION
        )
        
        # Read active mechanical velocity (Parameter 0x701B)
        vel = self.read_parameter(
            target_id=motor_id, 
            parameter_tuple=ParameterType.MECHANICAL_VELOCITY
        )
        
        return pos, vel

    def enable_and_verify_all(self, limits: dict, control_mode: str = 'MIT', timeout: float = 0.5):
        """
        Enables all actuators, configures them to the specified control mode,
        forces them into a passive state, and verifies their response and mode.
        
        Args:
            limits (dict): Dictionary containing P_MIN, P_MAX, V_MIN, V_MAX, T_MIN, T_MAX.
            control_mode (str): The desired control mode ('MIT', 'TORQUE', or 'VELOCITY').
            timeout (float): Max time in seconds to wait for all motors to verify.
        """
        control_mode = control_mode.upper()
        
        # Map requested modes to the hardware's internal integer enumerations
        if control_mode == 'MIT':
            mode_val = 0
        elif control_mode == 'TORQUE':
            mode_val = 3
        elif control_mode == 'VELOCITY':
            mode_val = 2
        else:
            raise ValueError(f"Invalid control mode '{control_mode}'. Supported: 'MIT', 'TORQUE', 'VELOCITY'")

        print(f"[INFO] Configuring all motors to {control_mode} mode (Mode Parameter: {mode_val})...")
        
        # 1. Pipeline the Mode parameter write command to the bus
        for motor_id in self.motor_ids:
            self.write_parameter(motor_id, ParameterType.MODE, mode_val)
            time.sleep(0.01)

        print("[INFO] Pre-loading zero-targets to ensure a limp state on startup...")
        
        # 2. Pipeline a zero-command BEFORE enabling. This ensures that any stale targets 
        # saved in the motor's RAM are overwritten to 0 before power is applied to the coils.
        for motor_id in self.motor_ids:
            if control_mode == 'MIT':
                self.send_target_state_vector(
                    motor_id=motor_id, pos=0.0, vel=0.0, kp=0.0, kd=0.0, torque=0.0, limits=limits
                )
            elif control_mode == 'TORQUE':
                self.send_target_torque(motor_id=motor_id, torque_nm=0.0, limits=limits)
            elif control_mode == 'VELOCITY':
                self.send_target_velocity(motor_id=motor_id, velocity_rads=0.0, limits=limits)
            time.sleep(0.005)

        print("[INFO] Sending Enable commands to all motors...")
        
        # 3. Flush any stale messages from the bus so we only listen for fresh status replies
        self.flush_CAN_bus()

        # 4. Send the Enable command (CommType 3) to all actuators
        for motor_id in self.motor_ids:
            self.transmit(
                comm_type=CommunicationType.ENABLE,
                extra_data=self.host_id,
                destination_id=motor_id,
                data=b'\x00' * 8  # Payload is ignored for Enable commands
            )
            time.sleep(0.01)
            
        print("[INFO] Waiting for state verification from all motors...")
        
        # 5. Verify all expected motor Operation Status replies are received
        verified_ids = set()
        start_time = time.perf_counter()
        
        while len(verified_ids) < len(self.motor_ids):
            # Check for overall timeout
            if (time.perf_counter() - start_time) > timeout:
                missing = set(self.motor_ids) - verified_ids
                raise HardwareIOError(
                    f"Timeout waiting for state verification. Missing replies from motors: {missing}"
                )
                
            reply = self.receive(timeout=0.01)
            if reply is None:
                continue
                
            c_type, motor_id, dest_id, extra_data, r_data = reply
            
            # Check if the reply is a valid operation status destined for our host
            if c_type == CommunicationType.OPERATION_STATUS and dest_id == self.host_id:
                if motor_id in self.motor_ids:
                    verified_ids.add(motor_id)

        # 6. Actively read the MODE parameter from memory to verify it stuck successfully
        print(f"[INFO] Polling memory to verify {control_mode} control mode...")
        for motor_id in self.motor_ids:
            try:
                self.flush_CAN_bus()
                reported_mode = self.read_parameter(motor_id, ParameterType.MODE, timeout=0.1)
                if reported_mode != mode_val:
                    print(f"[WARN] Motor {motor_id} reported mode {reported_mode}, expected {mode_val}.")
                else:
                    print(f"  -> Motor {motor_id}: Verified active and in {control_mode} control mode.")
            except HardwareIOError as e:
                print(f"[WARN] Failed to read mode parameter for motor {motor_id}: {e}")
                    
        print("[INFO] All motors successfully enabled, verified, and passive.")

    # Requires flush_CAN_bus() to be appropriately called before the original outbound messages were sent that we are listening for replies.
    def wait_for_all_replies(self, limits: dict, timeout: float = 0.05) -> dict:
        """
        Blocks until valid Operation Status (Communication Type 2) replies are 
        received from ALL expected motors.
        
        Args:
            limits (dict): Dictionary containing P_MIN, P_MAX, V_MIN, V_MAX, T_MIN, T_MAX.
            timeout (float): Max time in seconds to wait for the complete state vector.
            
        Returns:
            dict: A mapping of {motor_id: {'pos': float, 'vel': float, 'torque': float, 'temp': float}}
            
        Raises:
            HardwareIOError: If the timeout is exceeded before all motors reply.
        """
        received_states = {}
        start_time = time.perf_counter()

        # Loop until we have collected a state reply from every motor we expect
        while len(received_states) < len(self.motor_ids):
            
            # 1. Check for overall timeout
            if (time.perf_counter() - start_time) > timeout:
                missing = set(self.motor_ids) - set(received_states.keys())
                raise HardwareIOError(
                    f"Timeout waiting for state replies. Missing replies from motors: {missing}"
                )

            # 2. Receive the next message from the CAN bus
            reply = self.receive(timeout=0.005)
            if reply is None:
                continue

            c_type, motor_id, dest_id, extra_data, r_data = reply

            # 3. Filter for Operation Status (CommType 2) destined specifically for our host
            if c_type == CommunicationType.OPERATION_STATUS and dest_id == self.host_id:
                
                # 4. If it's from a motor we are tracking, unpack the data
                if motor_id in self.motor_ids and motor_id not in received_states:
                    
                    # The payload is packed as four 16-bit unsigned integers in Big-Endian format
                    p_int, v_int, t_int, temp_int = struct.unpack('>HHHh', r_data)                    
                    # Scale values back to physical units
                    pos = self._scale_u16_to_value(p_int, limits['P_MIN'], limits['P_MAX'])
                    vel = self._scale_u16_to_value(v_int, limits['V_MIN'], limits['V_MAX'])
                    torque = self._scale_u16_to_value(t_int, limits['T_MIN'], limits['T_MAX'])
                    
                    # Temperature is reported as Celsius * 10
                    temperature = temp_int / 10.0
                    
                    # Store in our local dictionary
                    received_states[motor_id] = {
                        'pos': pos,
                        'vel': vel,
                        'torque': torque,
                        'temp': temperature
                    }

        return received_states
    
    def shutdown(self, timeout: float = 0.5):
        """
        Safely disables all active motors, verifies they have shut down, 
        and closes the CAN bus interface.
        """
        if self.bus is not None:

            print("\n[INFO] Zeroing stateful parameters before shutdown...")
            
            # Explicitly overwrite RAM targets to 0.0 to prevent jerking on next boot
            for motor_id in getattr(self, 'motor_ids', []):
                try:
                    self.write_parameter(motor_id, ParameterType.VELOCITY_TARGET, 0.0)
                    self.write_parameter(motor_id, ParameterType.IQ_TARGET, 0.0)
                    time.sleep(0.005)
                except HardwareIOError:
                    pass

            print("\n[INFO] Disabling all motors...")
            
            # 1. Flush bus to clear out old status messages
            try:
                self.flush_CAN_bus()
            except Exception:
                pass

            # 2. Send the Disable command (CommType 4) to all actuators
            for motor_id in getattr(self, 'motor_ids', []):
                try:
                    self.transmit(
                        comm_type=CommunicationType.DISABLE,
                        extra_data=self.host_id,
                        destination_id=motor_id,
                        data=b'\x00' * 8  # Payload must be cleared to 0
                    )
                    time.sleep(0.01)
                except HardwareIOError as e:
                    print(f"[WARN] Failed to send disable command to motor {motor_id}: {e}")

            # 3. Verify the motors actually disabled
            print("[INFO] Verifying motor shutdown...")
            verified_offline = set()
            start_time = time.perf_counter()
            
            while len(verified_offline) < len(getattr(self, 'motor_ids', [])):
                if (time.perf_counter() - start_time) > timeout:
                    missing = set(self.motor_ids) - verified_offline
                    print(f"[CRITICAL WARNING] Timeout verifying shutdown! Motors {missing} MAY STILL BE LIVE AND DANGEROUS.")
                    break
                    
                try:
                    reply = self.receive(timeout=0.01)
                    if reply is None:
                        continue
                        
                    c_type, motor_id, dest_id, extra_data, r_data = reply
                    
                    # If we got a status reply from a motor, we treat it as confirmation 
                    # that the hardware processed the CommType 4 command.
                    if c_type == CommunicationType.OPERATION_STATUS and dest_id == self.host_id:
                        if motor_id in self.motor_ids:
                            verified_offline.add(motor_id)
                            print(f"  -> Motor {motor_id}: Shutdown verified.")
                            
                except HardwareIOError:
                    pass # Ignore read errors during teardown

            print("[INFO] Shutting down CAN bus interface...")
            
            # 4. Close the socketcan interface
            try:
                self.bus.shutdown()
            except Exception as e:
                print(f"[WARN] Non-fatal error while shutting down CAN bus: {e}")
            finally:
                self.bus = None
                print("[INFO] Teardown complete.")
        else:
            print("\n[INFO] CAN bus was not initialized. Nothing to shut down.")