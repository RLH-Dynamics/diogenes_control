"""A simulated RobStride bus, so the full stack runs with no hardware.

Each `FakeMotorBus` thread attaches to one python-can channel and impersonates
the motors configured on it: it answers OPERATION_CONTROL with an
OPERATION_STATUS frame, answers parameter reads with whatever was last written,
and tracks enable/disable state. Joint positions relax toward the commanded
setpoint so a control loop sees plausible, moving data.

A motor can also be marked as running older firmware (`legacy_motors`): it
rejects reads of parameters 0x7026 and up, like three of Harold's real RS03s,
and times out after a fixed `legacy_timeout_s` whatever the host writes, or
never if that is None. Timeouts are only simulated on those motors, so ordinary
loop tests do not depend on host-side gaps.

This exists so the ordering contract, the multi-bus gather, the safety
interlocks, the observation layout and the logging can all be exercised on a
laptop -- and so that a regression in any of them is catchable without powering
up 60 N.m actuators.

Usage (see tools/loopback_check.py):
    sim = FakeRobot(spec)
    sim.start()
    ... run the real stack against spec ...
    sim.stop()
"""

import struct
import threading
import time

import can

from robstride.protocol import CommunicationType, FORMAT_MAP


def _scale_to_u16(value, v_min, v_max):
    clamped = max(min(value, v_max), v_min)
    return int(65535.0 * (clamped - v_min) / (v_max - v_min))


def _scale_from_u16(x_int, v_min, v_max):
    return float(x_int) * (v_max - v_min) / 65535.0 + v_min


class FakeMotor:
    """Minimal first-order model of one actuator."""

    def __init__(self, can_id, tau=0.05, legacy=False, legacy_timeout_s=None):
        self.can_id = can_id
        self.tau = tau
        self.legacy = legacy
        self.legacy_timeout_s = legacy_timeout_s
        self.last_rx = None
        self.last_t = None
        self.pos = 0.0
        self.vel = 0.0
        self.torque = 0.0
        self.temp = 31.5
        self.enabled = False
        self.mode = 0
        self.parameters = {}

    def step(self, now, target_pos, kp):
        dt = 0.0 if self.last_t is None else now - self.last_t
        self.last_t = now
        if not self.enabled or kp <= 0.0:
            self.vel = 0.0
            self.torque = 0.0
            return
        # Relax toward the setpoint with a fixed time constant. Not a physical
        # model -- just enough for the state to move coherently.
        alpha = min(1.0, dt / self.tau)
        new_pos = self.pos + alpha * (target_pos - self.pos)
        self.vel = (new_pos - self.pos) / dt if dt > 0 else 0.0
        self.pos = new_pos
        self.torque = kp * (target_pos - self.pos)


class FakeMotorBus(threading.Thread):
    """Impersonates every motor on one channel."""

    def __init__(self, channel, host_id, can_ids, limits, interface="virtual",
                 legacy_ids=(), legacy_timeout_s=None):
        super().__init__(daemon=True, name=f"fake-{channel}")
        self.channel = channel
        self.host_id = host_id
        self.limits = limits
        self.interface = interface
        self.motors = {
            mid: FakeMotor(mid, legacy=mid in legacy_ids,
                           legacy_timeout_s=legacy_timeout_s)
            for mid in can_ids
        }
        self._stop_event = threading.Event()
        self.bus = None
        self.frames_seen = 0

    def start(self):
        self.bus = can.interface.Bus(channel=self.channel, interface=self.interface)
        super().start()

    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        if self.bus is not None:
            self.bus.shutdown()
            self.bus = None

    # ------------------------------------------------------------- protocol --

    def _send(self, comm_type, extra_data, dest_id, data):
        arb_id = (comm_type << 24) | (extra_data << 8) | dest_id
        self.bus.send(can.Message(arbitration_id=arb_id, data=data,
                                  is_extended_id=True, dlc=len(data)))

    def _send_status(self, motor: FakeMotor):
        lim = self.limits
        payload = struct.pack(
            '>HHHh',
            _scale_to_u16(motor.pos, lim['P_MIN'], lim['P_MAX']),
            _scale_to_u16(motor.vel, lim['V_MIN'], lim['V_MAX']),
            _scale_to_u16(motor.torque, lim['T_MIN'], lim['T_MAX']),
            int(motor.temp * 10),
        )
        # The motor's own id rides in the low byte of extra_data and its mode
        # state (0 reset, 2 run) in bits 14..15; the frame is addressed to the host.
        state = 2 if motor.enabled else 0
        self._send(CommunicationType.OPERATION_STATUS,
                   (state << 14) | motor.can_id, self.host_id, payload)

    def run(self):
        while not self._stop_event.is_set():
            msg = self.bus.recv(timeout=0.002)
            now = time.perf_counter()

            if msg is None:
                continue
            self.frames_seen += 1

            comm_type = (msg.arbitration_id >> 24) & 0x1F
            extra_data = (msg.arbitration_id >> 8) & 0xFFFF
            dest_id = msg.arbitration_id & 0xFF

            motor = self.motors.get(dest_id)
            if motor is None:
                continue

            # A real motor times out on its own clock; checking lazily on the
            # next frame is indistinguishable from the host's side.
            if (motor.legacy and motor.enabled and motor.legacy_timeout_s is not None
                    and motor.last_rx is not None
                    and now - motor.last_rx > motor.legacy_timeout_s):
                motor.enabled = False
            motor.last_rx = now

            if comm_type == CommunicationType.OPERATION_CONTROL:
                p_u16, v_u16, kp_u16, kd_u16 = struct.unpack('>HHHH', msg.data)
                target_pos = _scale_from_u16(p_u16, self.limits['P_MIN'],
                                             self.limits['P_MAX'])
                kp = _scale_from_u16(kp_u16, 0.0, 5000.0)
                motor.step(now, target_pos, kp)
                self._send_status(motor)

            elif comm_type == CommunicationType.ENABLE:
                motor.enabled = True
                self._send_status(motor)

            elif comm_type == CommunicationType.DISABLE:
                motor.enabled = False
                self._send_status(motor)

            elif comm_type == CommunicationType.SET_ZERO_POSITION:
                motor.pos = 0.0
                motor.vel = 0.0
                self._send_status(motor)

            elif comm_type == CommunicationType.WRITE_PARAMETER:
                index, _ = struct.unpack('<HH', msg.data[0:4])
                if motor.legacy and index >= 0x7026:
                    continue   # older firmware ignores writes it doesn't know
                motor.parameters[index] = msg.data[4:8]
                if index == 0x7005:   # MODE
                    motor.mode = msg.data[4]

            elif comm_type == CommunicationType.READ_PARAMETER:
                index, _ = struct.unpack('<HH', msg.data[0:4])
                if motor.legacy and index >= 0x7026:
                    # Rejected: status flag 1 in the high byte, zero payload.
                    self._send(CommunicationType.READ_PARAMETER,
                               (1 << 8) | motor.can_id, self.host_id,
                               struct.pack('<HHI', index, 0, 0))
                    continue
                stored = motor.parameters.get(index, b'\x00\x00\x00\x00')
                payload = struct.pack('<HH', index, 0x0000) + stored
                self._send(CommunicationType.READ_PARAMETER, motor.can_id,
                           self.host_id, payload)


class FakeRobot:
    """Every fake bus described by a RobotSpec."""

    def __init__(self, spec, legacy_motors=(), legacy_timeout_s=None):
        """`legacy_motors` is a collection of (channel, can_id) pairs."""
        self.spec = spec
        self.buses = [
            FakeMotorBus(
                channel=b.channel,
                host_id=spec.host_id,
                can_ids=spec.can_ids_on(b.channel),
                limits=spec.actuator_limits,
                interface=b.interface,
                legacy_ids={mid for ch, mid in legacy_motors if ch == b.channel},
                legacy_timeout_s=legacy_timeout_s,
            )
            for b in spec.buses
        ]

    def start(self):
        for b in self.buses:
            b.start()
        time.sleep(0.05)   # let the reader threads attach

    def stop(self):
        for b in self.buses:
            b.stop()

    @property
    def frames_seen(self) -> int:
        return sum(b.frames_seen for b in self.buses)
