"""Drive the simulated robot live with the gamepad, and watch it in Viser.

A rehearsal of the walking procedure with only the robot simulated: commands
come from tools/gamepad_teleop.py through the same UDP receiver main-rl.py uses
on the Pi, and the policy runs through this stack's policy code against the
training robot in MuJoCo (tools/sim2sim_walk.py has the details), in real time.

  * Before START the torso is held in place with the legs in the crouch --
    the rope. OPTIONS (START) lets go and the policy takes over.
  * Left stick for speed, right stick to turn, as on the robot; centred, it
    steps in place.
  * CIRCLE (STOP) makes the motors go limp, as on the robot.
  * "Reset" (in the Viser panel) puts it back on the rope, ready for START.
    A fall resets by itself after 3 s.

    # terminal 1 (mujoco + onnxruntime + viser)
    python tools/sim2sim_live.py policy_walk.onnx /tmp/harold_walk.mjb
    # terminal 2
    python3 tools/gamepad_teleop.py --pi 127.0.0.1
    # then open http://localhost:8080
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import viser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import POLICY_PROFILES, SPEC  # noqa: E402
from control.command_source import DEFAULT_PORT, UdpCommandSource  # noqa: E402
from control.policy import Policy  # noqa: E402
from tools.sim2sim_walk import WalkSim  # noqa: E402

GRID_CELL = 0.25
FALL_RESET_S = 3.0
SPEED_FILTER_S = 0.5


def body_meshes(model: mujoco.MjModel):
    """Visual meshes merged per (body, colour), in each body's own frame."""
    merged = {}
    rot = np.zeros(9)
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 2:
            continue
        m = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[m], model.mesh_vertnum[m]
        fa, fn = model.mesh_faceadr[m], model.mesh_facenum[m]
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        verts = model.mesh_vert[va:va + vn] @ rot.reshape(3, 3).T + model.geom_pos[g]
        faces = model.mesh_face[fa:fa + fn]
        mat = model.geom_matid[g]
        rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]
        color = tuple(int(255 * c) for c in rgba[:3])
        merged.setdefault((int(model.geom_bodyid[g]), color), []).append((verts, faces))
    out = {}
    for (body, color), parts in merged.items():
        offsets = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
        verts = np.concatenate([v for v, _ in parts]).astype(np.float32)
        faces = np.concatenate([f + o for (_, f), o in zip(parts, offsets)]).astype(np.uint32)
        out.setdefault(body, []).append((color, verts, faces))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("policy", type=Path)
    p.add_argument("model", type=Path, help=".mjb from diogenes_mjlab export_walk_model.py")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="teleop UDP port")
    p.add_argument("--viser-port", type=int, default=8080)
    args = p.parse_args()

    sim = WalkSim(args.model)
    policy = Policy(SPEC, str(args.policy), POLICY_PROFILES, clock=lambda: sim.data.time)
    if not policy.profile.uses_command:
        sys.exit(f"{policy.profile.task_id} is not a walking policy.")
    commands = UdpCommandSource(port=args.port)
    commands.start()

    model, data = sim.model, sim.data
    server = viser.ViserServer(port=args.viser_port, label="Harold sim-to-sim")
    grid = server.scene.add_grid("/floor", width=6.0, height=6.0, cell_size=GRID_CELL)
    frames, meshes = {}, body_meshes(model)
    for body, parts in meshes.items():
        frames[body] = server.scene.add_frame(f"/robot/b{body}", show_axes=False)
        for i, (color, verts, faces) in enumerate(parts):
            server.scene.add_mesh_simple(f"/robot/b{body}/m{i}", verts, faces, color=color)
    status = server.gui.add_markdown("starting...")
    follow = server.gui.add_checkbox("Follow robot", initial_value=True)
    reset_button = server.gui.add_button("Reset (back on the rope)")

    state = {"mode": "rope", "since": 0.0, "reset": False}

    @reset_button.on_click
    def _(_):
        state["reset"] = True

    def go(mode):
        state["mode"], state["since"] = mode, time.monotonic()

    def reset():
        sim.reset()
        commands.reset_latches()
        go("rope")

    print("[INFO] Open http://localhost:%d . Press OPTIONS to let go of the rope." % args.viser_port)
    v_fwd_f = yaw_f = 0.0
    alpha = SPEC.dt / SPEED_FILTER_S
    next_t = time.monotonic()
    step = 0
    try:
        while True:
            if state["reset"]:
                state["reset"] = False
                reset()
            mode = state["mode"]
            if mode == "rope":
                sim.step(sim.crouch, hold_base=True)
                if commands.start_requested():
                    policy.reset()
                    go("walking")
            elif mode == "walking":
                if commands.stop_requested():
                    go("limp")
                else:
                    policy.set_command(*commands.command())
                    sim.step(policy.act(*sim.joints(), sim.imu_sample()))
                    if sim.fallen():
                        go("fallen")
            else:  # limp or fallen: motors off
                sim.step(None)
                if mode == "fallen" and time.monotonic() - state["since"] > FALL_RESET_S:
                    reset()

            v_fwd, yaw_rate, tilt, height = sim.base_state()
            v_fwd_f += alpha * (v_fwd - v_fwd_f)
            yaw_f += alpha * (yaw_rate - yaw_f)

            step += 1
            if step % 2 == 0:  # draw at 25 Hz
                base = data.qpos[:3].copy()
                offset = np.array([base[0], base[1], 0.0]) if follow.value else np.zeros(3)
                for body, frame in frames.items():
                    frame.position = data.xpos[body] - offset
                    frame.wxyz = data.xquat[body]
                grid.position = ((-(base[0] % GRID_CELL), -(base[1] % GRID_CELL), 0.0)
                                 if follow.value else (0.0, 0.0, 0.0))
            if step % 10 == 0:
                cmd = policy.command if mode == "walking" else np.zeros(3)
                label = {"rope": "ON THE ROPE -- press OPTIONS",
                         "walking": "WALKING", "limp": "STOPPED (limp) -- Reset to restart",
                         "fallen": "FELL -- resetting"}[mode]
                status.content = (
                    f"**{label}**  \n{commands.status()}  \n"
                    f"command: vx {cmd[0]:+.2f} m/s, wz {cmd[2]:+.2f} rad/s  \n"
                    f"actual: vx {v_fwd_f:+.2f} m/s, wz {yaw_f:+.2f} rad/s  \n"
                    f"tilt {np.degrees(tilt):4.1f} deg, height {height:.3f} m")

            next_t += SPEC.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_t = time.monotonic()  # fell behind; don't try to catch up
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        commands.stop()


if __name__ == "__main__":
    main()
