"""Drive the walking policy from a gamepad on the laptop.

Runs on the LAPTOP (standard library only), reads the gamepad through the Linux
joystick device (/dev/input/js0) and sends velocity commands to the Pi over UDP
at 50 Hz (control/command_source.py has the packet and the Pi-side safety
behaviour).

Controls. The defaults match Harold's Bluetooth pad, a PlayStation-layout
"Wireless Controller" (checked with --probe 2026-09-26); remap with the flags
for another pad:

  left stick Y    forward / backward speed                         axis 1
  right stick X   turn left / right                                axis 3
  OPTIONS         start the policy (after the soft start, with the
                  feet down)                                       button 9
  CIRCLE          stop: ends the run, the motors go limp           button 1

    python3 tools/gamepad_teleop.py --pi 192.168.0.11
    python3 tools/gamepad_teleop.py --probe      # find a pad's button and stick numbers

Sticks centred (they spring back), the robot steps in place. Speeds are capped
to what the walking policy was trained for, and the command is rate-limited so
the stick can't ask for an instant jump.
"""

import argparse
import os
import select
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.command_source import (  # noqa: E402
    BUTTON_START, BUTTON_STOP, DEFAULT_PORT, encode,
)

JS_EVENT = struct.Struct("<IhBB")  # time ms, value, type, number
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80
AXIS_MAX = 32767.0
DEADZONE = 0.08


def parse_args():
    p = argparse.ArgumentParser(description="Gamepad teleop for the walking policy.")
    p.add_argument("--pi", default="192.168.0.11", help="Pi address (default: %(default)s)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--device", default="/dev/input/js0")
    p.add_argument("--probe", action="store_true",
                   help="Print each button press and stick move, to find their numbers.")
    p.add_argument("--max-forward", type=float, default=0.3, help="m/s (default: %(default)s)")
    p.add_argument("--max-backward", type=float, default=0.1, help="m/s (default: %(default)s)")
    # Turning capped below the policy's 0.5 rad/s and ramped gently: the v3
    # policy fell after a full-stick turn on foam (2026-09-26, run 5).
    p.add_argument("--max-turn", type=float, default=0.3, help="rad/s (default: %(default)s)")
    p.add_argument("--accel", type=float, default=0.5, help="m/s^2 speed slew limit")
    p.add_argument("--turn-accel", type=float, default=1.0, help="rad/s^2 turn slew limit")
    p.add_argument("--axis-forward", type=int, default=1, help="left stick Y")
    p.add_argument("--axis-turn", type=int, default=3, help="right stick X")
    p.add_argument("--button-start", type=int, default=9, help="OPTIONS")
    p.add_argument("--button-stop", type=int, default=1, help="CIRCLE")
    return p.parse_args()


def stick(raw: int) -> float:
    v = raw / AXIS_MAX
    if abs(v) < DEADZONE:
        return 0.0
    return (v - DEADZONE * (1 if v > 0 else -1)) / (1.0 - DEADZONE)


def slew(current: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    return current + max(-step, min(step, target - current))


def main():
    args = parse_args()
    try:
        fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        sys.exit(f"[ERROR] Cannot open {args.device}: {e}. Is the gamepad connected?")
    axes, buttons = {}, {}

    def read_events():
        while True:
            try:
                data = os.read(fd, JS_EVENT.size * 32)
            except BlockingIOError:
                return
            for i in range(0, len(data) - JS_EVENT.size + 1, JS_EVENT.size):
                _, value, kind, number = JS_EVENT.unpack_from(data, i)
                if kind & JS_AXIS:
                    axes[number] = value
                elif kind & JS_BUTTON:
                    buttons[number] = value

    if args.probe:
        print("Press each button and push each stick all the way; one line per event. "
              "Ctrl+C to quit.")
        read_events()  # the driver's initial state
        seen_buttons = dict(buttons)
        axis_side = {k: round(v / AXIS_MAX) for k, v in axes.items()}
        try:
            while True:
                select.select([fd], [], [], 0.1)
                read_events()
                for k, v in sorted(buttons.items()):
                    if v != seen_buttons.get(k, 0):
                        print(f"button {k} {'pressed' if v else 'released'}", flush=True)
                        seen_buttons[k] = v
                for k, v in sorted(axes.items()):
                    side = round(v / AXIS_MAX)  # -1, 0 or +1 (past half travel)
                    if side != axis_side.get(k, 0):
                        print(f"axis {k} -> {v / AXIS_MAX:+.2f}", flush=True)
                        axis_side[k] = side
        except KeyboardInterrupt:
            print()
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.pi, args.port)
    print(f"[INFO] Sending to {dest[0]}:{dest[1]} at 50 Hz. Sticks to move, "
          f"OPTIONS to start the policy, CIRCLE to stop the run. Ctrl+C quits "
          f"(the robot then steps in place until you stop it).")
    seq, vx, wz = 0, 0.0, 0.0
    period = 0.02
    next_t = time.monotonic()
    try:
        while True:
            select.select([fd], [], [], max(0.0, next_t - time.monotonic()))
            read_events()
            now = time.monotonic()
            if now < next_t:
                continue
            next_t += period
            if now - next_t > 0.5:
                next_t = now + period  # fell behind (laptop suspended?)

            fwd = -stick(axes.get(args.axis_forward, 0))    # stick up is negative
            turn = -stick(axes.get(args.axis_turn, 0))      # stick right is positive
            target_vx = fwd * (args.max_forward if fwd >= 0 else args.max_backward)
            target_wz = turn * args.max_turn
            vx = slew(vx, target_vx, args.accel, period)
            wz = slew(wz, target_wz, args.turn_accel, period)

            flags = BUTTON_START if buttons.get(args.button_start) else 0
            flags |= BUTTON_STOP if buttons.get(args.button_stop) else 0
            sock.sendto(encode(seq, vx, 0.0, wz, flags), dest)
            seq += 1
            print(f"\rvx {vx:+.2f} m/s  wz {wz:+.2f} rad/s"
                  + ("  START" if flags & BUTTON_START else "")
                  + ("  STOP" if flags & BUTTON_STOP else "") + "      ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n[INFO] Teleop stopped; the Pi falls back to stepping in place.")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
