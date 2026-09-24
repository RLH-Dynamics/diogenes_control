"""Fan-out across every CAN bus on the robot.

One `RobstrideBus` owns one channel. `RobstrideNetwork` owns all of them and is
responsible for the two things that only make sense at the whole-robot level:

  1. TRANSMIT ORDERING. Commands are interleaved across channels
     (can0:id1, can1:id4, can0:id2, can1:id5, ...) rather than sent bus by bus.
     Each MCP2515 has only two RX buffers, so three replies arriving as a burst
     can overrun it. Interleaving spaces each bus's replies by the other bus's
     frame time, at no cost.

  2. GATHERING AGAINST ONE DEADLINE. Replies from both channels are collected
     concurrently with select(), so latency is the slowest single bus rather
     than the sum of both.

Above this layer, joints are identified by NAME. CAN ids never escape upward.
"""

import select
import time

from robstride.bus import RobstrideBus
from utils.exceptions import HardwareIOError


class RobstrideNetwork:
    """Every actuator on the robot, across every channel."""

    def __init__(self, spec):
        self.spec = spec
        self.buses: dict[str, RobstrideBus] = {
            b.channel: RobstrideBus(
                channel=b.channel,
                host_id=spec.host_id,
                can_ids=spec.can_ids_on(b.channel),
                limits=spec.actuator_limits,
                bitrate=b.bitrate,
                watchdog_ms=spec.watchdog_ms,
                interface=b.interface,
            )
            for b in spec.buses
        }
        self._tx_order = self._build_tx_order()
        self._fd_map: dict[int, RobstrideBus] = {}
        self._use_select = True

        # Rolling diagnostics, read by the recorder and the bring-up scripts.
        self.last_exchange_s = 0.0
        self.exchange_count = 0
        self.timeout_count = 0

    # ------------------------------------------------------------ lifecycle --

    def _build_tx_order(self):
        """Round-robin the joints across channels to space out the replies."""
        per_bus = {ch: self.spec.joints_on(ch) for ch in self.spec.channels}
        depth = max(len(v) for v in per_bus.values())
        order = []
        for i in range(depth):
            for ch in self.spec.channels:
                if i < len(per_bus[ch]):
                    order.append(per_bus[ch][i])
        return order

    def open(self):
        """Open every channel and arm the motor-side watchdogs."""
        for bus in self.buses.values():
            bus.open()

        try:
            self._fd_map = {bus.fileno(): bus for bus in self.buses.values()}
        except (AttributeError, NotImplementedError):
            # Non-socketcan backends (e.g. the 'virtual' interface used in
            # off-hardware tests) may not expose a file descriptor. Fall back to
            # polling each bus in turn; correctness is unaffected, only latency.
            print("[WARN] CAN backend exposes no fileno(); falling back to polled gather.")
            self._use_select = False

    def verify_watchdogs(self) -> bool:
        # A list, not a generator: all() would stop at the first failing bus and
        # leave the remaining buses unchecked and unreported.
        results = [bus.verify_hardware_watchdog(self.spec.watchdog_ms)
                   for bus in self.buses.values()]
        return all(results)

    def enable(self, control_mode: str = 'MIT'):
        for bus in self.buses.values():
            bus.enable_and_verify(control_mode=control_mode)

    def flush(self):
        for bus in self.buses.values():
            bus.flush()

    def shutdown(self):
        for bus in self.buses.values():
            bus.shutdown()

    # -------------------------------------------------------------- control --

    def send(self, targets: dict, kp: float, kd: float):
        """Transmit one setpoint per joint. Non-blocking.

        `targets` maps joint name -> {'pos', 'vel', 'torque'}; missing keys
        default to zero. Frames go out in interleaved bus order.
        """
        for joint in self._tx_order:
            state = targets[joint.name]
            self.buses[joint.bus].send_target_state_vector(
                motor_id=joint.can_id,
                pos=state.get('pos', 0.0),
                vel=state.get('vel', 0.0),
                kp=kp,
                kd=kd,
                torque=state.get('torque', 0.0),
            )

    def gather(self, timeout: float) -> dict:
        """Block until every joint has reported, or the deadline passes.

        Returns joint name -> {'pos', 'vel', 'torque', 'temp'} in hardware frame.
        """
        received: dict[tuple[str, int], dict] = {}
        expected = self.spec.num_joints
        deadline = time.perf_counter() + timeout

        # Frames may already be queued from the transmit we just did.
        for bus in self.buses.values():
            bus.drain_into(received)

        while len(received) < expected:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                self.timeout_count += 1
                missing = [j.name for j in self.spec.joints
                           if j.key not in received]
                raise HardwareIOError(
                    f"Timeout after {timeout * 1000:.1f} ms waiting for state "
                    f"replies. Missing joints: {missing}"
                )

            if self._use_select:
                ready, _, _ = select.select(list(self._fd_map), [], [], remaining)
                for fd in ready:
                    self._fd_map[fd].drain_into(received)
            else:
                for bus in self.buses.values():
                    bus.drain_into(received)
                if len(received) < expected:
                    time.sleep(0.0002)

        return {self.spec.joint_by_key(*key).name: state
                for key, state in received.items()}

    def exchange(self, targets: dict, kp: float, kd: float,
                 timeout: float = 0.005) -> dict:
        """Send every setpoint, then collect every reply against one deadline."""
        started = time.perf_counter()
        self.flush()
        self.send(targets, kp, kd)
        state = self.gather(timeout)
        self.last_exchange_s = time.perf_counter() - started
        self.exchange_count += 1
        return state

    def check_faults(self):
        """Raise ActuatorFault if any motor reported one since the last check."""
        for bus in self.buses.values():
            bus.raise_pending_faults()

    # ---------------------------------------------------------- diagnostics --

    def bus_statistics(self) -> dict:
        """Per-channel counters, for logging alongside the control data."""
        return {
            'exchanges': self.exchange_count,
            'timeouts': self.timeout_count,
            'last_exchange_ms': self.last_exchange_s * 1000.0,
        }

    def limp_targets(self) -> dict:
        """A zero setpoint for every joint. Use with kp=kd=0 to stay passive."""
        return {j.name: {'pos': 0.0, 'vel': 0.0, 'torque': 0.0}
                for j in self.spec.joints}
