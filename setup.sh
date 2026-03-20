#!/bin/bash

echo "[INFO] Configuring CAN bus (can0) at 1Mbit/s..."
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

echo "[INFO] Activating Python virtual environment..."
source .venv/bin/activate

echo "[SUCCESS] Harold workspace is ready!"