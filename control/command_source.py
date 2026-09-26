"""Velocity commands for a walking policy.

UdpCommandSource listens for packets from tools/gamepad_teleop.py, which runs on
the laptop with the gamepad plugged in. Each packet carries the command and the
button state; the laptop sends them at ~50 Hz.

SAFETY BEHAVIOUR
----------------
  * Motion needs the deadman button held on the gamepad: without it, or with no
    packet for LINK_TIMEOUT_S (link lost, laptop script stopped), the command is
    zero -- step in place. A point-foot biped cannot stand still, so zero is the
    neutral command, not a stop.
  * STOP (a latched button press) asks the control loop to end the run; the
    motors then go limp as on any exit. Keep the hardware kill switch in reach.
  * Only the first sender is accepted, so another machine on the network
    cannot take over a running robot. Pass `peer` to fix it in advance.

PACKET (little-endian, 21 bytes)
--------------------------------
  4s  magic b"HRLD"
  I   sequence number (packets arriving out of order are dropped)
  f   vx  m/s, + forward
  f   vy  m/s, + left
  f   wz  rad/s, + turn left
  B   buttons: bit 0 deadman held, bit 1 START pressed, bit 2 STOP pressed
"""

import socket
import struct
import threading
import time

PACKET = struct.Struct("<4sIfffB")
MAGIC = b"HRLD"
DEFAULT_PORT = 5566
LINK_TIMEOUT_S = 0.5

BUTTON_DEADMAN = 1 << 0
BUTTON_START = 1 << 1
BUTTON_STOP = 1 << 2


def encode(seq: int, vx: float, vy: float, wz: float, buttons: int) -> bytes:
    return PACKET.pack(MAGIC, seq & 0xFFFFFFFF, vx, vy, wz, buttons)


class FixedCommandSource:
    """A constant command with START already given: scripted tests, no gamepad."""

    def __init__(self, vx=0.0, vy=0.0, wz=0.0):
        self._cmd = (float(vx), float(vy), float(wz))

    def start(self):
        pass

    def stop(self):
        pass

    def command(self):
        return self._cmd

    def start_requested(self) -> bool:
        return True

    def stop_requested(self) -> bool:
        return False

    def status(self) -> str:
        return f"fixed command {self._cmd}"


class UdpCommandSource:
    """Commands from tools/gamepad_teleop.py over UDP, read in a thread."""

    def __init__(self, port: int = DEFAULT_PORT, peer: str | None = None,
                 timeout_s: float = LINK_TIMEOUT_S):
        self.port = port
        self.peer = peer
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._buttons = 0
        self._last_rx = None
        self._seq = None
        self._start_latched = False
        self._stop_latched = False
        self._rejected = set()
        self._running = False
        self._sock = None
        self._thread = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._rx_loop, name="teleop-rx", daemon=True)
        self._thread.start()
        print(f"[INFO] Teleop: listening on UDP {self.port}"
              + (f" for {self.peer}" if self.peer else " (first sender is locked in)"))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()

    def _rx_loop(self):
        while self._running:
            try:
                data, (host, _) = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != PACKET.size:
                continue
            magic, seq, vx, vy, wz, buttons = PACKET.unpack(data)
            if magic != MAGIC:
                continue
            with self._lock:
                if self.peer is None:
                    self.peer = host
                    print(f"\n[INFO] Teleop: controller at {host}")
                if host != self.peer:
                    if host not in self._rejected:
                        self._rejected.add(host)
                        print(f"\n[WARN] Teleop: ignoring packets from {host}")
                    continue
                if self._seq is not None and 0 < (self._seq - seq) % 2**32 < 2**31:
                    continue  # older than one already applied
                self._seq = seq
                self._cmd = (vx, vy, wz)
                self._buttons = buttons
                self._last_rx = time.monotonic()
                self._start_latched |= bool(buttons & BUTTON_START)
                self._stop_latched |= bool(buttons & BUTTON_STOP)

    def link_ok(self) -> bool:
        with self._lock:
            return (self._last_rx is not None
                    and time.monotonic() - self._last_rx < self.timeout_s)

    def command(self):
        """(vx, vy, wz); zero unless the link is live and the deadman is held."""
        with self._lock:
            live = (self._last_rx is not None
                    and time.monotonic() - self._last_rx < self.timeout_s)
            if live and self._buttons & BUTTON_DEADMAN:
                return self._cmd
            return (0.0, 0.0, 0.0)

    def start_requested(self) -> bool:
        with self._lock:
            return self._start_latched

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_latched

    def status(self) -> str:
        if not self.link_ok():
            return "teleop: NO LINK (stepping in place)"
        with self._lock:
            deadman = "held" if self._buttons & BUTTON_DEADMAN else "released"
        vx, vy, wz = self.command()
        return f"teleop: deadman {deadman}, vx {vx:+.2f} wz {wz:+.2f}"
