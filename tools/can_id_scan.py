"""Scan each CAN channel for RobStride actuators and check them against config.

Read-only. Sends only GET_DEVICE_ID (type 0), which makes a motor reply with its
64-bit MCU unique id; no motor is enabled, zeroed or reconfigured. Safe to run
with the robot powered and suspended.

Every reply during a probe window is kept, so two motors answering to the same
id on one bus show up as two different UIDs rather than silently collapsing.

    source setup.sh
    python tools/can_id_scan.py                 # ids 1..127 on every config bus
    python tools/can_id_scan.py --ids 1-10 --channels can0
"""

import argparse
import os
import sys
import time

import can

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HOST_ID, SPEC  # noqa: E402
from robstride.protocol import CommunicationType  # noqa: E402


def parse_ids(text: str) -> list[int]:
    ids = set()
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        ids.update(range(int(lo), int(hi or lo) + 1))
    return sorted(ids)


def parse_args():
    p = argparse.ArgumentParser(description="Scan CAN channels for RobStride ids.")
    p.add_argument('--channels', default=",".join(b.channel for b in SPEC.buses),
                   help="Comma-separated channels (default: every bus in config).")
    p.add_argument('--ids', default="1-127", help="Ids to probe, e.g. 1-127 or 1,2,5-6.")
    p.add_argument('--window', type=float, default=0.02,
                   help="Seconds to listen after each probe.")
    return p.parse_args()


def source_id(arb_id: int) -> int:
    """Replies carry the motor's own id in bits 15..8 of the extended id."""
    return (arb_id >> 8) & 0xFF


def scan_channel(channel: str, ids: list[int], window: float) -> dict[int, set[str]]:
    found: dict[int, set[str]] = {}
    stray = 0
    with can.Bus(interface='socketcan', channel=channel) as bus:
        while bus.recv(timeout=0.0):
            pass

        # Anything already chattering on the bus before we say a word.
        t_end = time.monotonic() + 0.2
        while (msg := bus.recv(timeout=max(0.0, t_end - time.monotonic()))):
            stray += 1

        for target in ids:
            arb = (CommunicationType.GET_DEVICE_ID << 24) | (HOST_ID << 8) | target
            try:
                bus.send(can.Message(arbitration_id=arb, data=bytes(8),
                                     is_extended_id=True), timeout=0.1)
            except can.CanError as e:
                print(f"  [{channel}] TX failed at id {target}: {e}")
                print(f"  [{channel}] (no ACK usually means nothing powered on this bus,"
                      f" or a wiring/termination fault)")
                break

            t_end = time.monotonic() + window
            while (remaining := t_end - time.monotonic()) > 0:
                msg = bus.recv(timeout=remaining)
                if msg is None:
                    break
                if not msg.is_extended_id or msg.is_error_frame:
                    continue
                comm = (msg.arbitration_id >> 24) & 0x1F
                if comm != CommunicationType.GET_DEVICE_ID:
                    continue
                sid = source_id(msg.arbitration_id)
                found.setdefault(sid, set()).add(bytes(msg.data).hex())

    if stray:
        print(f"  [{channel}] note: {stray} unsolicited frame(s) seen before probing")
    return found


def main():
    args = parse_args()
    channels = [c.strip() for c in args.channels.split(',') if c.strip()]
    ids = parse_ids(args.ids)

    print(f"Probing ids {ids[0]}..{ids[-1]} ({len(ids)}) on {', '.join(channels)} "
          f"with GET_DEVICE_ID (read-only)\n")

    results = {}
    for ch in channels:
        t0 = time.monotonic()
        results[ch] = scan_channel(ch, ids, args.window)
        print(f"[{ch}] {len(results[ch])} id(s) answered "
              f"({time.monotonic() - t0:.1f} s)")
        for sid in sorted(results[ch]):
            uids = sorted(results[ch][sid])
            flag = "   <-- DUPLICATE ID ON THIS BUS" if len(uids) > 1 else ""
            print(f"    id {sid:>3}  uid {', '.join(uids)}{flag}")
        print()

    # ---- compare against config.JOINTS ------------------------------------
    print("Config check:")
    ok = True
    for j in SPEC.joints:
        seen_on = [ch for ch, f in results.items() if j.can_id in f]
        if j.bus in seen_on:
            status = "OK"
        elif seen_on:
            status = f"MISMATCH - found on {', '.join(seen_on)} instead"
            ok = False
        elif j.bus not in results:
            status = "not scanned"
        else:
            status = "MISSING"
            ok = False
        print(f"    {j.name:<13} {j.bus} id {j.can_id:<3} {status}")

    expected = {(j.bus, j.can_id) for j in SPEC.joints}
    extras = [(ch, sid) for ch, f in results.items() for sid in f
              if (ch, sid) not in expected]
    for ch, sid in extras:
        print(f"    UNEXPECTED    {ch} id {sid}")
        ok = False

    all_uids = [u for f in results.values() for s in f.values() for u in s]
    if len(all_uids) != len(set(all_uids)):
        print("    note: the same UID answered on more than one bus/id")

    print("\nRESULT:", "all joints match config" if ok else "see issues above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
