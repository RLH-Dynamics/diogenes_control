#!/bin/bash
# Prepare the Harold workspace: bring up both CAN channels, activate the venv.
# Source it, do not execute it:  source setup.sh

# Must match the channels declared in config.py BUSES.
CHANNELS="can0 can1"
BITRATE=1000000

setup_failed=0

for ch in $CHANNELS; do
    if ! ip link show "$ch" > /dev/null 2>&1; then
        echo "[ERROR] Interface $ch does not exist."
        echo "        Check the mcp2515 dtoverlay lines in /boot/firmware/config.txt"
        echo "        and 'dmesg | grep -i mcp'."
        setup_failed=1
        continue
    fi

    echo "[INFO] Configuring $ch at ${BITRATE} bit/s..."
    # restart-ms lets the controller recover automatically from a bus-off
    # condition instead of staying wedged until a manual restart.
    sudo ip link set "$ch" down 2>/dev/null
    if ! sudo ip link set "$ch" type can bitrate "$BITRATE" restart-ms 100; then
        echo "[ERROR] Failed to configure $ch."
        setup_failed=1
        continue
    fi
    # A short TX queue is deliberate on a control loop: a deep queue means stale
    # setpoints backing up behind fresh ones, which is worse than dropping them.
    sudo ip link set "$ch" txqueuelen 10
    if ! sudo ip link set "$ch" up; then
        echo "[ERROR] Failed to bring up $ch."
        setup_failed=1
        continue
    fi
    echo "[OK]   $ch is up."
done

if [ ! -d ".venv" ]; then
    echo "[ERROR] No .venv found. Create it with:"
    echo "        python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    setup_failed=1
else
    echo "[INFO] Activating Python virtual environment..."
    source .venv/bin/activate
fi

if [ "$setup_failed" -ne 0 ]; then
    echo ""
    echo "[FAILED] Setup did not complete. Do not run the control scripts."
else
    echo ""
    echo "[SUCCESS] Harold workspace is ready."
fi
