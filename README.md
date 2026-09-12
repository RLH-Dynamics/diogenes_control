# Harold — robot control

Control stack for a two-legged robot: six RobStride RS03 actuators across two
CAN buses, plus a base-mounted BNO085 IMU, running an ONNX policy on a
Raspberry Pi 5.

## Hardware

| | |
|---|---|
| Compute | Raspberry Pi 5 |
| CAN | Waveshare 2-CH CAN HAT (2 × MCP2515, SPI0, isolated) |
| Actuators | 6 × RobStride RS03 — `can0`: left leg (ids 1–3), `can1`: right leg (ids 4–6) |
| IMU | Adafruit BNO085 over I2C |

Both MCP2515 controllers share the SPI0 master, so their register traffic
serialises in the kernel even though the CAN buses are electrically independent.
Each controller has only **two RX buffers**, which is why commands are
interleaved across channels rather than sent bus-by-bus — see
`robstride/network.py`.

### One-time Pi configuration

In `/boot/firmware/config.txt`:

```
dtparam=spi=on
dtoverlay=mcp2515-can0,oscillator=12000000,interrupt=25,spimaxfrequency=10000000
dtoverlay=mcp2515-can1,oscillator=12000000,interrupt=23,spimaxfrequency=10000000

# The Pi's I2C controller handles the BNO085's clock stretching poorly.
dtparam=i2c_arm_baudrate=50000
```

Confirm the interrupt GPIOs against the Waveshare wiki and your board's solder
jumpers — they are selectable. Then check `dmesg | grep -i mcp` for two
successful initialisations.

## Running

```bash
source setup.sh                      # brings up can0 + can1, activates .venv

python main-read-state.py            # passive joint state, nothing moves
python main-imu-read.py              # IMU bring-up + mount-rotation calibration
python main-set-zero.py --leg left   # set mechanical zero, one leg at a time
python main-rl.py                    # the policy
```

Useful `main-rl.py` flags: `--dry-run` commands the default pose instead of
loading a policy, `--virtual` runs against simulated actuators with no hardware,
`--no-imu` substitutes a synthetic level IMU, `--duration N` stops after N
seconds.

For tighter loop timing, launch under `sudo chrt -f 80 .venv/bin/python ...`.

## Off-hardware testing

```bash
python tools/loopback_check.py       # full stack against simulated actuators
python main-rl.py --dry-run --virtual --duration 5
```

`tools/fake_motors.py` impersonates the RS03 protocol over python-can's
in-process virtual backend, so joint identity, the multi-bus gather, the safety
interlocks, the observation layout and the logging are all testable on a laptop.

## Architecture

```
config.py            all hardware/control constants as data; the only file
                     you edit to rewire the robot
robot/
  model.py           JointSpec/BusSpec/RobotSpec + validation
  robot.py           the machine: actuator network + IMU
  leg.py             a named view over a subset of joints
  session.py         context manager; guarantees motor teardown
  kinematics.py      leg IK
robstride/
  protocol.py        RS03 private protocol constants
  bus.py             one CAN channel and its motors
  network.py         all channels: interleaved TX, select() gather
sensors/
  base.py            ImuSample, ImuSource protocol, NullImu
  imu.py             threaded BNO085 reader
control/
  observation.py     declarative observation layout
  policy.py          ONNX inference + action transform
utils/
  safety.py          joint limits + base attitude interlocks
  realtime.py        process tuning + absolute-deadline loop pacing
  recorder.py        CSV logging, schema derived from the spec
tools/
  fake_motors.py     simulated RS03 bus
  loopback_check.py  end-to-end smoke test
legacy/              pre-dual-bus scripts, kept for reference (see its README)
```

### The ordering contract

`config.JOINTS` is an ordered list, and that order defines the policy's
observation and action ordering. It is cross-checked at import against
`POLICY_JOINT_ORDER`, which you copy verbatim from the training environment.
CAN id is never used as a joint identity above the transport layer — with two
buses it is not unique enough to be one.

### Observation layout

`config.OBSERVATION_TERMS` names the terms in order; `control/observation.py`
owns their widths. The total is checked against the ONNX model's declared input
width at startup, before any bus is opened, and a mismatch is fatal.

## Open items

- `POLICY_JOINT_ORDER` and `OBSERVATION_TERMS` are the expected six-joint + IMU
  layout, not a verified one. Confirm both against the retrained MJLab env.
- The right leg's `direction` signs and asymmetric joint limits currently copy
  the left leg's. Confirm each against the real mechanism before applying gain.
- `IMU.mount_rotation` is the identity placeholder. Measure it with
  `main-imu-read.py`.
- Loop timing and MCP2515 RX-overrun counters have not been measured on the real
  HAT. Watch `ip -details -statistics link show can0` during a soak test.
