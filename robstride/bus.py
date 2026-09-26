"""Driver for the RobStride actuators sitting on ONE CAN channel.

This is the transport layer. It knows about CAN frames, the RobStride private
protocol, and the motors physically reachable on its own channel. It knows
nothing about joints, legs, policies, or the robot as a whole -- that lives in
robstride/network.py and above.

Scoping this class to a single channel is what makes multi-bus operation
possible: the fan-out logic (which motor is on which wire, how to gather replies
from several channels against one deadline) belongs to the caller, not here.
"""

import struct
import time

import can

from robstride.protocol import CommunicationType, FORMAT_MAP, ParameterType
from utils.exceptions import ActuatorFault, HardwareIOError, ParameterRejected

# Kp/Kd are scaled against fixed bounds defined by the RobStride manual, not
# against the per-robot actuator limits.
KP_SCALE_MAX = 5000.0
KD_SCALE_MAX = 100.0


class RobstrideBus:
    """One socketcan channel and the RobStride motors attached to it."""

    def __init__(self, channel: str, host_id: int, can_ids: list[int],
                 limits: dict, bitrate: int = 1_000_000, watchdog_ms: int = 100,
                 interface: str = "socketcan"):
        self.bus = None
        self.channel = channel
        self.host_id = host_id
        self.can_ids = list(can_ids)
        self.limits = limits
        self.bitrate = bitrate
        self.watchdog_ms = watchdog_ms
        self.interface = interface

        # Faults reported by motors since the last drain, as {can_id: raw_payload}.
        # Populated by drain_into(); the supervisory layer decides what to do.
        self.pending_faults: dict[int, bytes] = {}

        # Frames drain_into() did not use as a status reply, as
        # (motor_id, comm_type, extra_data, payload). Cleared by the network at
        # the start of each gather; read when a motor fails to reply.
        self.other_frames: list[tuple[int, int, int, bytes]] = []

    def __repr__(self):
        return f"<RobstrideBus {self.channel} ids={self.can_ids}>"

    # ------------------------------------------------------------- scaling --

    @staticmethod
    def _scale_value_to_u16(value: float, v_min: float, v_max: float) -> int:
        """Scale a float into a 16-bit unsigned int (float_to_uint in the manual)."""
        clamped = max(min(value, v_max), v_min)
        return int(65535.0 * (clamped - v_min) / (v_max - v_min))

    @staticmethod
    def _scale_u16_to_value(x_int: int, v_min: float, v_max: float) -> float:
        """Reverse the 16-bit scaling back into a float (uint_to_float)."""
        span = v_max - v_min
        return float(x_int) * span / 65535.0 + v_min

    # ------------------------------------------------------------ lifecycle --

    def open(self):
        """Bind to the socketcan interface and arm the motor-side watchdogs."""
        print(f"[INFO] Opening CAN channel {self.channel}...")
        if self.interface == "socketcan":
            # A socket binds fine to a DOWN interface and only the first send
            # fails, so check first. Down is normal after a reboot.
            try:
                with open(f"/sys/class/net/{self.channel}/flags") as f:
                    up = int(f.read(), 16) & 0x1        # IFF_UP
            except OSError:
                raise HardwareIOError(
                    f"{self.channel} does not exist. Check the mcp2515 overlays "
                    f"in /boot/firmware/config.txt (see README).")
            if not up:
                raise HardwareIOError(
                    f"{self.channel} is down (normal after a reboot). "
                    f"Run `source setup.sh` first.")
        try:
            self.bus = can.interface.Bus(
                channel=self.channel, interface=self.interface, bitrate=self.bitrate
            )
        except can.CanError as e:
            raise HardwareIOError(f"CAN library error initialising {self.channel}: {e}")
        except OSError as e:
            raise HardwareIOError(
                f"OS error connecting to {self.channel}. Is the interface up? {e}"
            )
        self.enable_hardware_watchdog(self.watchdog_ms)

    def fileno(self) -> int:
        """Underlying socket fd, so a caller can select() across several buses."""
        return self.bus.fileno()

    def flush(self):
        """Discard anything already queued on this channel."""
        try:
            while self.bus.recv(timeout=0.0):
                pass
        except can.CanError as e:
            raise HardwareIOError(f"I/O failure flushing {self.channel}: {e}")

    # --------------------------------------------------------- raw frame io --

    def transmit(self, comm_type, extra_data, destination_id, data=b'\x00' * 8):
        arb_id = (comm_type << 24) | (extra_data << 8) | destination_id
        msg = can.Message(arbitration_id=arb_id, data=data,
                          is_extended_id=True, dlc=len(data))
        try:
            self.bus.send(msg)
        except can.CanError as e:
            raise HardwareIOError(
                f"[{self.channel}] failed to transmit CommType {comm_type} "
                f"to id {destination_id}: {e}"
            )

    def receive(self, timeout=0.001):
        try:
            msg = self.bus.recv(timeout=timeout)
            if msg is None:
                return None

            comm_type = (msg.arbitration_id >> 24) & 0x1F
            extra_data = (msg.arbitration_id >> 8) & 0xFFFF
            destination_id = msg.arbitration_id & 0xFF

            # The sending motor's id is the low byte of the extra_data field.
            motor_id = extra_data & 0xFF

            return comm_type, motor_id, destination_id, extra_data, msg.data
        except can.CanError as e:
            raise HardwareIOError(f"I/O failure receiving on {self.channel}: {e}")

    # -------------------------------------------------------- parameter i/o --

    def write_parameter(self, target_id, parameter_tuple, value):
        """Pipeline a parameter write without blocking."""
        param_index, param_type = parameter_tuple
        value_format = FORMAT_MAP[param_type]

        index_bytes = struct.pack('<HH', param_index, 0x0000)
        data_bytes = struct.pack(value_format, value)

        # Pad to 8 bytes; the manual dictates bytes 4~7 hold the parameter data.
        padded_data = (index_bytes + data_bytes).ljust(8, b'\x00')
        self.transmit(CommunicationType.WRITE_PARAMETER, self.host_id,
                      target_id, data=padded_data)

    def read_parameter(self, target_id, parameter_tuple, timeout=0.1):
        """Blocking read of a single parameter out of a motor's memory."""
        param_index, param_type = parameter_tuple
        value_format = FORMAT_MAP[param_type]

        req_data = struct.pack('<HHL', param_index, 0x0000, 0x00000000)
        self.transmit(CommunicationType.READ_PARAMETER, self.host_id,
                      target_id, data=req_data)

        start_t = time.perf_counter()
        while (time.perf_counter() - start_t) < timeout:
            reply = self.receive(timeout=0.01)
            if reply is None:
                continue

            c_type, motor_id, dest_id, extra_data, r_data = reply
            if (c_type == CommunicationType.READ_PARAMETER
                    and motor_id == target_id and dest_id == self.host_id):
                # Skip a late reply to some earlier read of a different index.
                if struct.unpack('<H', r_data[0:2])[0] != param_index:
                    continue
                # The high byte of extra_data is a status flag: 0 = success,
                # nonzero = the motor rejected the read (e.g. the parameter is
                # missing from its firmware). The payload is then zero-filled.
                status = (extra_data >> 8) & 0xFF
                if status:
                    raise ParameterRejected(
                        f"[{self.channel}] motor {target_id} rejected read of "
                        f"parameter {hex(param_index)} (status {status:#04x}); "
                        f"not supported by its firmware?"
                    )
                size = struct.calcsize(value_format)
                return struct.unpack(value_format, r_data[4:4 + size])[0]

        raise HardwareIOError(
            f"[{self.channel}] timeout reading parameter {hex(param_index)} "
            f"from motor {target_id}"
        )

    def enable_hardware_watchdog(self, timeout_ms=100):
        """Arm the motor-side CAN timeout so silence makes the actuators go limp."""
        print(f"[INFO] [{self.channel}] setting hardware watchdogs to {timeout_ms} ms...")
        timeout_units = int(timeout_ms * 20)
        for mid in self.can_ids:
            self.write_parameter(mid, ParameterType.CAN_TIMEOUT, timeout_units)
            time.sleep(0.01)   # let the motor MCU process the write

    def verify_hardware_watchdog(self, timeout_ms=100, tolerance_units=1) -> bool:
        """Read the timeout back out of motor memory to confirm the write stuck.

        The previous implementation of this check read a parameter that is never
        written back, so it could not actually fail. This one reads 0x7028,
        the same address the watchdog is written to.
        """
        expected = int(timeout_ms * 20)
        all_ok = True
        no_register = []
        for mid in self.can_ids:
            try:
                self.flush()
                actual = self.read_parameter(mid, ParameterType.CAN_TIMEOUT, timeout=0.1)
                if abs(int(actual) - expected) > tolerance_units:
                    print(f"[WARN] [{self.channel}] motor {mid} watchdog reads "
                          f"{actual}, expected {expected}.")
                    all_ok = False
            except ParameterRejected:
                # Older firmware has no 0x7028, but may still time out on a
                # stored setting. Test the behaviour directly instead.
                no_register.append(mid)
            except HardwareIOError as e:
                print(f"[WARN] [{self.channel}] could not verify watchdog on motor {mid}: {e}")
                all_ok = False

        if no_register:
            print(f"[INFO] [{self.channel}] motors {no_register} have no readable "
                  f"CAN-timeout register; testing timeout behaviour instead...")
            results = self.verify_timeout_by_silence(no_register,
                                                     silence_s=2 * timeout_ms / 1000)
            for mid, ok in results.items():
                if ok:
                    print(f"  -> [{self.channel}] motor {mid}: disabled itself "
                          f"within {2 * timeout_ms} ms of silence.")
                else:
                    all_ok = False
        return all_ok

    def verify_timeout_by_silence(self, motor_ids, silence_s: float = 0.2) -> dict:
        """Behavioural watchdog check: does each motor disable itself when the
        host goes quiet? Returns {can_id: passed}.

        Zero-force throughout. The motors are put in MIT mode with a
        kp = kd = torque = 0 setpoint, enabled, confirmed running from the mode
        bits of their status reply, then left without traffic for `silence_s`.
        One more zero-force frame then asks for their state: a motor that timed
        out reports reset mode. Every motor tested is sent DISABLE afterwards
        whatever the outcome.

        A pass proves the motor's timeout is no longer than `silence_s`; it
        cannot tell 50 ms from 150 ms.
        """
        results = {}
        try:
            for mid in motor_ids:
                self.write_parameter(mid, ParameterType.MODE, self.MODE_VALUES['MIT'])
                time.sleep(0.01)
                self._send_limp(mid)
                time.sleep(0.005)
            self.flush()
            for mid in motor_ids:
                self.transmit(CommunicationType.ENABLE, self.host_id, mid)
                time.sleep(0.01)
            time.sleep(0.02)

            before = {mid: self._limp_probe(mid) for mid in motor_ids}
            time.sleep(silence_s)
            after = {mid: self._limp_probe(mid) for mid in motor_ids}

            for mid in motor_ids:
                b, a = before[mid], after[mid]
                if b != self.MOTOR_STATE_RUN:
                    print(f"[WARN] [{self.channel}] motor {mid}: timeout test "
                          f"inconclusive, did not enter run mode (state {b}).")
                    results[mid] = False
                elif a != self.MOTOR_STATE_RESET:
                    print(f"[WARN] [{self.channel}] motor {mid} was still enabled "
                          f"after {silence_s * 1000:.0f} ms of silence (state {a}): "
                          f"its CAN timeout is off or longer than that.")
                    results[mid] = False
                else:
                    results[mid] = True
        finally:
            for mid in motor_ids:
                try:
                    self.transmit(CommunicationType.DISABLE, self.host_id, mid)
                    time.sleep(0.01)
                except HardwareIOError as e:
                    print(f"[WARN] [{self.channel}] failed to disable motor {mid}: {e}")
            self.flush()
        return results

    def _send_limp(self, motor_id: int):
        self.send_target_state_vector(motor_id, pos=0.0, vel=0.0,
                                      kp=0.0, kd=0.0, torque=0.0)

    def _limp_probe(self, motor_id: int, timeout: float = 0.05):
        """Send one zero-force frame; return the mode state from the reply, or None."""
        self.flush()
        self._send_limp(motor_id)
        start_t = time.perf_counter()
        while (time.perf_counter() - start_t) < timeout:
            reply = self.receive(timeout=0.01)
            if reply is None:
                continue
            c_type, mid, dest_id, extra_data, _ = reply
            if (c_type == CommunicationType.OPERATION_STATUS
                    and mid == motor_id and dest_id == self.host_id):
                # Status frames carry the mode state in bits 22..23 of the
                # arbitration id, i.e. bits 14..15 of extra_data.
                return (extra_data >> 14) & 0x3
        return None

    # ------------------------------------------------------------ commands --

    def send_target_state_vector(self, motor_id: int, pos: float, vel: float,
                                 kp: float, kd: float, torque: float):
        """Send one MIT-mode setpoint frame."""
        lim = self.limits
        p_u16 = self._scale_value_to_u16(pos, lim['P_MIN'], lim['P_MAX'])
        v_u16 = self._scale_value_to_u16(vel, lim['V_MIN'], lim['V_MAX'])
        kp_u16 = self._scale_value_to_u16(kp, 0.0, KP_SCALE_MAX)
        kd_u16 = self._scale_value_to_u16(kd, 0.0, KD_SCALE_MAX)
        t_u16 = self._scale_value_to_u16(torque, lim['T_MIN'], lim['T_MAX'])

        # Big-endian '>HHHH' per the manual; the torque rides in the arbitration
        # id's extra_data field rather than the payload.
        data_payload = struct.pack('>HHHH', p_u16, v_u16, kp_u16, kd_u16)
        self.transmit(
            comm_type=CommunicationType.OPERATION_CONTROL,
            extra_data=t_u16,
            destination_id=motor_id,
            data=data_payload,
        )

    def send_target_torque(self, motor_id: int, torque_nm: float, kt: float = 2.36):
        """Command torque on a motor configured in TORQUE (current) mode."""
        clamped_torque = max(self.limits['T_MIN'], min(self.limits['T_MAX'], torque_nm))
        target_amps = clamped_torque / kt
        target_amps = max(-43.0, min(43.0, target_amps))   # RS03 peak current
        self.write_parameter(motor_id, ParameterType.IQ_TARGET, target_amps)

    def send_target_velocity(self, motor_id: int, velocity_rads: float):
        """Command velocity on a motor configured in VELOCITY mode."""
        clamped_vel = max(self.limits['V_MIN'], min(self.limits['V_MAX'], velocity_rads))
        self.write_parameter(motor_id, ParameterType.VELOCITY_TARGET, clamped_vel)

    def get_position_and_velocity(self, motor_id: int) -> tuple[float, float]:
        """Blocking read of measured position and velocity from motor memory.

        Works in any control mode, unlike the MIT-mode status reply.
        """
        pos = self.read_parameter(motor_id, ParameterType.MECHANICAL_POSITION)
        vel = self.read_parameter(motor_id, ParameterType.MECHANICAL_VELOCITY)
        return pos, vel

    # ------------------------------------------------------------- replies --

    def _decode_status(self, payload: bytes, extra_data: int = 0) -> dict:
        p_int, v_int, t_int, temp_int = struct.unpack('>HHHh', payload)
        lim = self.limits
        return {
            'pos': self._scale_u16_to_value(p_int, lim['P_MIN'], lim['P_MAX']),
            'vel': self._scale_u16_to_value(v_int, lim['V_MIN'], lim['V_MAX']),
            'torque': self._scale_u16_to_value(t_int, lim['T_MIN'], lim['T_MAX']),
            'temp': temp_int / 10.0,      # reported as Celsius * 10
            # Bits 22..23 of the arbitration id (14..15 of extra_data): 0 reset
            # (disabled), 1 calibration, 2 run. See MOTOR_STATE_*.
            'mode': (extra_data >> 14) & 0x3,
        }

    def drain_into(self, received: dict) -> int:
        """Non-blocking. Decode every frame currently queued on this channel.

        Status replies are written into `received` keyed by `(channel, can_id)`.
        Fault frames are accumulated on `self.pending_faults` for the supervisory
        layer to act on. Returns the number of new status replies decoded.

        Draining promptly matters more on the MCP2515 than it did on the USB
        adapter: each controller has only two RX buffers, so a slow drain shows
        up as a silent overrun rather than a late frame.
        """
        new_replies = 0
        while True:
            reply = self.receive(timeout=0.0)
            if reply is None:
                return new_replies

            c_type, motor_id, dest_id, extra_data, r_data = reply

            if c_type == CommunicationType.FAULT_REPORT:
                if motor_id in self.can_ids:
                    self.pending_faults[motor_id] = bytes(r_data)
                self.other_frames.append((motor_id, c_type, extra_data, bytes(r_data)))
                continue

            if (c_type != CommunicationType.OPERATION_STATUS or dest_id != self.host_id
                    or motor_id not in self.can_ids):
                if len(self.other_frames) < 64:
                    self.other_frames.append((motor_id, c_type, extra_data, bytes(r_data)))
                continue

            key = (self.channel, motor_id)
            if key not in received:
                received[key] = self._decode_status(r_data, extra_data)
                new_replies += 1

    def raise_pending_faults(self):
        """Convert any accumulated fault frames into an ActuatorFault."""
        if not self.pending_faults:
            return
        faults = dict(self.pending_faults)
        self.pending_faults.clear()
        detail = ", ".join(f"motor {mid}: {payload.hex()}" for mid, payload in faults.items())
        raise ActuatorFault(f"[{self.channel}] motor fault report -- {detail}")

    # -------------------------------------------------------------- enable --

    MODE_VALUES = {'MIT': 0, 'VELOCITY': 2, 'TORQUE': 3}

    # Mode state reported in bits 22..23 of every status frame's arbitration id.
    MOTOR_STATE_RESET = 0      # disabled
    MOTOR_STATE_CALIBRATION = 1
    MOTOR_STATE_RUN = 2        # enabled

    def enable_and_verify(self, control_mode: str = 'MIT', timeout: float = 0.5):
        """Configure mode, pre-load zero targets, enable, and verify the motors.

        The zero-target pre-load matters: it overwrites whatever stale setpoint
        is sitting in the motor's RAM before power reaches the coils.
        """
        control_mode = control_mode.upper()
        if control_mode not in self.MODE_VALUES:
            raise ValueError(
                f"Invalid control mode '{control_mode}'. "
                f"Supported: {sorted(self.MODE_VALUES)}"
            )
        mode_val = self.MODE_VALUES[control_mode]

        print(f"[INFO] [{self.channel}] configuring motors to {control_mode} mode...")
        for motor_id in self.can_ids:
            self.write_parameter(motor_id, ParameterType.MODE, mode_val)
            time.sleep(0.01)

        print(f"[INFO] [{self.channel}] pre-loading zero targets...")
        for motor_id in self.can_ids:
            if control_mode == 'MIT':
                self.send_target_state_vector(motor_id, pos=0.0, vel=0.0,
                                              kp=0.0, kd=0.0, torque=0.0)
            elif control_mode == 'TORQUE':
                self.send_target_torque(motor_id, 0.0)
            else:
                self.send_target_velocity(motor_id, 0.0)
            time.sleep(0.005)

        print(f"[INFO] [{self.channel}] sending enable commands...")
        self.flush()
        for motor_id in self.can_ids:
            self.transmit(
                comm_type=CommunicationType.ENABLE,
                extra_data=self.host_id,
                destination_id=motor_id,
                data=b'\x00' * 8,
            )
            time.sleep(0.01)

        verified = set()
        start_time = time.perf_counter()
        while len(verified) < len(self.can_ids):
            if (time.perf_counter() - start_time) > timeout:
                missing = set(self.can_ids) - verified
                raise HardwareIOError(
                    f"[{self.channel}] timeout verifying enable. "
                    f"No reply from motors: {sorted(missing)}"
                )
            reply = self.receive(timeout=0.01)
            if reply is None:
                continue
            c_type, motor_id, dest_id, _, _ = reply
            if (c_type == CommunicationType.OPERATION_STATUS
                    and dest_id == self.host_id and motor_id in self.can_ids):
                verified.add(motor_id)

        print(f"[INFO] [{self.channel}] verifying {control_mode} mode took effect...")
        for motor_id in self.can_ids:
            try:
                self.flush()
                reported = self.read_parameter(motor_id, ParameterType.MODE, timeout=0.1)
                if reported != mode_val:
                    print(f"[WARN] [{self.channel}] motor {motor_id} reports mode "
                          f"{reported}, expected {mode_val}.")
                else:
                    print(f"  -> [{self.channel}] motor {motor_id}: "
                          f"active in {control_mode} mode.")
            except HardwareIOError as e:
                print(f"[WARN] [{self.channel}] could not read mode for motor {motor_id}: {e}")

    # ------------------------------------------------------------ shutdown --

    def shutdown(self, timeout: float = 0.5):
        """Zero stateful targets, disable every motor, verify, close the socket."""
        if self.bus is None:
            return

        print(f"\n[INFO] [{self.channel}] zeroing stateful parameters...")
        for motor_id in self.can_ids:
            try:
                self.write_parameter(motor_id, ParameterType.VELOCITY_TARGET, 0.0)
                self.write_parameter(motor_id, ParameterType.IQ_TARGET, 0.0)
                time.sleep(0.005)
            except HardwareIOError:
                pass

        print(f"[INFO] [{self.channel}] disabling all motors...")
        try:
            self.flush()
        except Exception:
            pass

        for motor_id in self.can_ids:
            try:
                self.transmit(
                    comm_type=CommunicationType.DISABLE,
                    extra_data=self.host_id,
                    destination_id=motor_id,
                    data=b'\x00' * 8,
                )
                time.sleep(0.01)
            except HardwareIOError as e:
                print(f"[WARN] [{self.channel}] failed to disable motor {motor_id}: {e}")

        verified_offline = set()
        start_time = time.perf_counter()
        while len(verified_offline) < len(self.can_ids):
            if (time.perf_counter() - start_time) > timeout:
                missing = set(self.can_ids) - verified_offline
                print(f"[CRITICAL WARNING] [{self.channel}] timeout verifying shutdown! "
                      f"Motors {sorted(missing)} MAY STILL BE LIVE AND DANGEROUS.")
                break
            try:
                reply = self.receive(timeout=0.01)
                if reply is None:
                    continue
                c_type, motor_id, dest_id, _, _ = reply
                if (c_type == CommunicationType.OPERATION_STATUS
                        and dest_id == self.host_id and motor_id in self.can_ids):
                    if motor_id not in verified_offline:
                        verified_offline.add(motor_id)
                        print(f"  -> [{self.channel}] motor {motor_id}: shutdown verified.")
            except HardwareIOError:
                pass   # ignore read errors during teardown

        try:
            self.bus.shutdown()
        except Exception as e:
            print(f"[WARN] [{self.channel}] non-fatal error closing bus: {e}")
        finally:
            self.bus = None
            print(f"[INFO] [{self.channel}] teardown complete.")
