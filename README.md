# Harold — robot control

Control stack for a two-legged robot: six RobStride RS03 actuators across two
CAN buses, plus a base-mounted BNO085 IMU, running an ONNX policy on a
Raspberry Pi 5.

## Hardware

| | |
|---|---|
| Compute | Raspberry Pi 5 |
| CAN | Waveshare 2-CH CAN HAT (2 × MCP2515, SPI0, isolated) |
| Actuators | 6 × RobStride RS03 — `can0`: left leg, `can1`: right leg; on each bus hip = 1, thigh = 2, calf = 3 |
| IMU | Adafruit BNO085 over I2C; game rotation vector (no magnetometer), accel + gyro calibration saved on the chip |

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
source setup.sh                      # after EVERY boot: brings up can0 + can1, sets the
                                     # performance governor and real-time priority for
                                     # the SPI/CAN threads, activates .venv
                                     # (HAROLD_VENV=/path/to/venv to use another)

python main-read-state.py            # passive joint state, nothing moves
python tools/can_id_scan.py          # read-only: which CAN ids answer on which bus
python tools/stream_joints.py --host <laptop-ip>   # limp joint state -> live Viser model
python main-imu-read.py              # IMU bring-up + mount-rotation calibration
python tools/imu_check.py            # guided pass/fail check of the mount rotation
python tools/imu_calibrate.py        # one-off accel + gyro calibration, saved on the chip
python main-set-zero.py --leg left   # set mechanical zero, one leg at a time
python main-rl.py                    # the suspended policy (policy.onnx)
python main-rl.py --model policy_walk.onnx --teleop   # walking, driven from the laptop
```

Useful `main-rl.py` flags: `--dry-run` commands the default pose instead of
loading a policy, `--virtual` runs against simulated actuators with no hardware,
`--no-imu` substitutes a synthetic level IMU, `--duration N` stops after N
seconds, `--command VX,WZ` gives a walking policy a fixed command instead of
the gamepad.

The policy file names its training task; `config.POLICY_PROFILES` holds each
task's observation layout, default and start poses, and the policy is refused if
its metadata disagrees. Action scale, gait-clock period and command ranges are
read from the file.

### Walking

`policy_walk.onnx` is the Diogenes-Biped-Walk policy (v3 rewards, 1.8 steps/s,
trained 2026-09-26). On the laptop, with the gamepad plugged in:

```bash
python3 tools/gamepad_teleop.py --probe       # a new pad: find its button/stick numbers
python3 tools/gamepad_teleop.py --pi <pi-ip>  # then drive
```

On the Pi, `main-rl.py --model policy_walk.onnx --teleop` soft-starts into the
crouch and holds it. Lower the robot on its rope until the feet carry it, then
press OPTIONS (START). The left stick sets speed (-0.1..0.3 m/s) and the right
stick turns (+-0.5 rad/s); with the sticks centred, or the link lost for 0.5 s,
it steps in place. CIRCLE stops the run and the motors go
limp. (Buttons as on Harold's PlayStation-layout pad; see the tool's flags for
another.)

`tools/sim2sim_walk.py` runs a walking policy through this stack's policy code
against the training robot in MuJoCo (model from diogenes_mjlab
`tools/export_walk_model.py`), with a scripted command sequence.

For tighter loop timing, launch under `sudo chrt -f 80 .venv/bin/python ...`.

### Zero calibration

`tools/calibrate_zeros.py` sets each motor's zero at a physical reference, one
joint at a time: hips against a straight edge (sim zero), then thighs pushed
against the body (100.9 deg from sim zero; the CAD's 102.9 made the walking
policy drift backward) while the hips are held firm, then
calves pushed to their stop (75 deg) while the thighs are held. `sim_offset` in
`config.py` places each reference in the sim frame; the offset signs are
confirmed afterwards in the live viewer and recorded by setting
`ZERO_OFFSETS_VERIFIED`, without which `main-rl.py` refuses to apply gain.

At every start-up each motor's position is also corrected by whole turns into
±180 deg of its zero, since RobStride motors only know their angle to within
one turn at power-up. Every calibrated working position lies inside that band.

### Live joint visualisation

`tools/stream_joints.py` streams limp joint state (kp = kd = 0, like
`main-read-state.py`) over UDP port 9870. On the laptop, in `diogenes_mjlab`, run
`python src/diogenes_mjlab/tools/live_joint_viewer.py` and open
http://localhost:8080. The suspended MJCF model follows the robot in the SIM
frame, so moving a joint by hand checks bus, id, name and direction sign in one
go. A link turns yellow while its joint moves and red when its joint is outside
the MJCF range.

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
  can_id_scan.py     read-only CAN id discovery per bus
  calibrate_zeros.py interactive zero calibration at physical references
  stream_joints.py   limp joint state -> live Viser viewer
legacy/              pre-dual-bus scripts, kept for reference (see its README)
```

### The ordering contract

`config.JOINTS` is an ordered list, and that order defines the policy's
observation and action ordering. It is cross-checked at import against
`POLICY_JOINT_ORDER`, which you copy verbatim from the training environment.
CAN id is never used as a joint identity above the transport layer — with two
buses it is not unique enough to be one.

### The sim joint contract

`sim_joint_contract.json` is exported from the MJCF by
`diogenes_mjlab/src/diogenes_mjlab/harold_biped/joint_contract.py`. It gives
each joint's sim range and, as signs, which way a positive angle moves the foot,
plus a `direction_signature` hash of those signs.

- Position limits are declared in the sim frame, derived from the contract's
  ranges plus `SIM_LIMIT_MARGIN`, and converted to the hardware frame through
  each joint's `direction`. Correcting a sign cannot leave a limit reversed.
- `direction` signs were verified by hand on 2026-09-24 (the right calf was
  found reversed and fixed), and `DIRECTIONS_VERIFIED_SIGNATURE` records the
  contract they were checked against. If the sim's joint conventions change,
  the re-exported contract has a new signature and `Robot.start` refuses to
  apply gain until the mapping test is repeated and the signature updated.
  Limp tools still run, so the test itself stays possible.
- On the sim side, `tests/test_joint_contract.py` pins the contract, so the
  change is caught when it happens rather than on the robot.

### Observation layout and policy metadata

`config.OBSERVATION_TERMS` names the terms in order; `control/observation.py`
owns their widths. With `OBSERVATION_HISTORY` > 0 each term is stacked over the
last N steps exactly as mjlab does it: term-major, oldest first, and backfilled
from the first observation after a reset. The suspended policy uses 10 steps of
joint pos, joint vel, last action and gait clock: 200 inputs, no IMU.

At startup `Policy` checks the ONNX input width and the metadata that
`diogenes_mjlab/src/diogenes_mjlab/tools/export_onnx.py` attaches (joint order,
default pose, action scale, observation terms, history length, clock period,
control period) against `config.py`. Any mismatch is fatal; a gain mismatch
only warns, as gains were randomised ±30% in training.

Policy targets are clamped to each actuator's sim `ctrlrange` (from the joint
contract), as the sim's position actuators do, before the safety layer sees
them.

### Motor-side CAN timeout

Every start-up arms each motor's CAN timeout (`WATCHDOG_MS`) and refuses to
enable unless it is confirmed. Motors with the `0x7028` register are checked by
readback. Three of Harold's RS03s (can0 ids 1 and 2, can1 id 3) run firmware
without it, so they get a zero-force silence test instead: enabled at
kp = kd = torque = 0, left without traffic for 2 × `WATCHDOG_MS`, and required
to report themselves disabled. It adds about half a second to start-up. Limp-only
tools skip the requirement; `main-rl.py --allow-unverified-watchdogs` overrides it.

## Open items

- The walking policy has not yet run on the robot: first test on the rope,
  stepping in place before any forward command.
- Loop timing and MCP2515 RX-overrun counters have not been measured on the real
  HAT. Watch `ip -details -statistics link show can0` during a soak test.
