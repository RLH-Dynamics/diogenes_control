#!/bin/bash
# Prepare the Harold workspace: bring up both CAN channels, apply the real-time
# tuning, activate the venv. Everything here is lost at a reboot, so run it
# after every boot. Source it, do not execute it:  source setup.sh
#
# The venv defaults to ./.venv; set HAROLD_VENV to use another, e.g.
#   HAROLD_VENV=~/harold/2_embedded/.venv source setup.sh

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

# Real-time tuning. Both CAN channels hang off one SPI controller, whose kernel
# worker (spi0) and interrupt threads run at normal priority by default; an
# occasional stall there delays whole cycles of CAN traffic, which main-rl.py
# sees as missing replies. The ondemand governor adds latency as it ramps.
echo "[INFO] Setting CPU governor to performance..."
if echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null; then
    echo "[OK]   governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
else
    echo "[ERROR] Failed to set the CPU governor."
    setup_failed=1
fi

echo "[INFO] Raising SPI worker and CAN interrupt threads to SCHED_FIFO 85..."
spi_threads="$(pgrep -x spi0) $(pgrep 'irq/[0-9]+-spi0')"
if [ -z "${spi_threads// /}" ]; then
    echo "[ERROR] No spi0 threads found. Is SPI enabled (dtparam=spi=on)?"
    setup_failed=1
fi
for pid in $spi_threads; do
    if sudo chrt -f -p 85 "$pid"; then
        echo "[OK]   $(cat /proc/$pid/comm) -> $(chrt -p "$pid" | sed -n 's/.*policy: //p' | head -1), priority 85"
    else
        echo "[ERROR] Failed to set priority of PID $pid."
        setup_failed=1
    fi
done

venv="${HAROLD_VENV:-.venv}"
if [ ! -d "$venv" ]; then
    echo "[ERROR] No virtual environment at $venv. Create one with:"
    echo "        python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    echo "        or point HAROLD_VENV at an existing one."
    setup_failed=1
else
    echo "[INFO] Activating Python virtual environment ($venv)..."
    source "$venv/bin/activate"
fi

if [ "$setup_failed" -ne 0 ]; then
    echo ""
    echo "[FAILED] Setup did not complete. Do not run the control scripts."
else
    echo ""
    echo "[SUCCESS] Harold workspace is ready."
fi
