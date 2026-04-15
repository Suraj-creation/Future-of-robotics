#!/usr/bin/env python3
"""Minimal MuJoCo SO101 pick-and-place demo.

This demo uses a stub `ACT` policy and analytic perception to perform a
pick-and-place sequence. It is intentionally simple and deterministic so
it runs on macOS without GPU and without ROS.

Files created by this demo:
- logs/mujoco_demo.log

How it works:
- Loads SO101 MJCF (so101_new_calib.xml) from the third-party folder
- Uses a stub perception module to return a cube start/target pose
- Uses a stub ACT policy to generate joint-space action chunks
- Applies target actuator position commands (position actuators) over time
"""
import os
import time
import math
from pathlib import Path

import numpy as np
import sys

HERE = Path(__file__).resolve().parent.parent
SCENE_DIR = HERE / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101'
SO101_XML = SCENE_DIR / 'so101_new_calib.xml'

# Make `macos_pipeline` importable when running the script directly
PKG_ROOT = str(HERE)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


def find_actuator_map(model):
    # build mapping actuator name -> index
    names = []
    for i in range(model.na):
        try:
            names.append(model.actuator(i).name)
        except Exception:
            names.append(None)
    return {n: i for i, n in enumerate(names) if n}


def step_sim(sim, model, n=1):
    """Step the MuJoCo simulation using whichever API is available."""
    # Try modern binding: mujoco.mj_step(model, sim)
    try:
        import mujoco as _mujoco
        for _ in range(n):
            _mujoco.mj_step(model, sim)
        return
    except Exception:
        pass

    # Try older wrapper that exposes mj_step
    try:
        from mujoco import mj_step as _mj_step
        for _ in range(n):
            _mj_step(model, sim)
        return
    except Exception:
        pass

    # Try mj_step(sim) signature
    try:
        from mujoco import mj_step as _mj_step
        for _ in range(n):
            _mj_step(sim)
        return
    except Exception:
        pass

    # Last resort: call sim.step() if present
    for _ in range(n):
        if hasattr(sim, 'step'):
            sim.step()
        else:
            raise RuntimeError('No known mujoco step API available')


def apply_ctrls(sim, model, act_map, ctrl_targets, steps=200, sleep=0.0, attach_info=None):
    # ctrl_targets: dict name->value
    # linearly ramp current ctrl -> target over `steps` physics steps
    ctrl_idx = {name: idx for name, idx in act_map.items() if name in ctrl_targets}
    target_arr = np.array([ctrl_targets[name] for name in ctrl_idx.keys()], dtype=float)

    # snapshot current control values (try several access patterns)
    cur_vals_list = []
    for idx in ctrl_idx.values():
        try:
            cur_vals_list.append(float(sim.data.ctrl[idx]))
        except Exception:
            try:
                cur_vals_list.append(float(sim.ctrl[idx]))
            except Exception:
                cur_vals_list.append(0.0)
    cur_vals = np.array(cur_vals_list, dtype=float)

    for t in range(1, steps + 1):
        alpha = t / steps
        vals = (1 - alpha) * cur_vals + alpha * target_arr
        for k, v in zip(ctrl_idx.values(), vals):
            try:
                sim.data.ctrl[k] = float(v)
            except Exception:
                try:
                    sim.ctrl[k] = float(v)
                except Exception:
                    pass
        # step physics
        step_sim(sim, model, n=1)

        # if the cube is logically attached, override its qpos to follow gripper
        if attach_info and attach_info.get('attached'):
            try:
                g_id = attach_info.get('gripper_site_id')
                c_qadr = attach_info.get('cube_qposadr')
                c_qdim = attach_info.get('cube_qpos_dim')
                pos_first = attach_info.get('pos_first')
                if g_id is not None and c_qadr is not None:
                    gp = sim.data.site_xpos[g_id]
                    if pos_first:
                        sim.data.qpos[c_qadr:c_qadr+3] = gp
                    else:
                        sim.data.qpos[c_qadr + c_qdim - 3:c_qadr + c_qdim] = gp
                    try:
                        import mujoco as _muj
                        _muj.mj_forward(model, sim)
                    except Exception:
                        pass
            except Exception:
                pass
        if sleep:
            time.sleep(sleep)


def find_joint_map(model):
    # map joint name -> qpos address
    jm = {}
    for j in range(model.nj):
        try:
            jname = model.joint(j).name
            try:
                adr = int(model.jnt_qposadr[j])
            except Exception:
                adr = int(model.joint(j).qposadr)
            if jname:
                jm[jname] = adr
        except Exception:
            continue
    return jm


def demo_run(duration=5.0):
    # build model that includes cube and target area so we can simulate pick/place
    from mujoco_demo.scene_builder import build_model_with_cube_and_target
    model, sim, attach_info = build_model_with_cube_and_target()

    # debug: print model counts
    try:
        print('model.na=', model.na, 'model.nj=', model.nj, 'model.nq=', model.nq)
    except Exception:
        pass

    act_map = find_actuator_map(model)
    print('Found actuators:', list(act_map.keys()))

    # Perception (analytic stub)
    from perception.analytic_pose import cube_start_and_target
    cube_pose, target_pose = cube_start_and_target()
    print('Cube start:', cube_pose, 'Target:', target_pose)

    # Policy (ACT stub)
    from policy.act_stub import plan_pick_place
    action_chunks = plan_pick_place(cube_pose, target_pose)
    print('Planned', len(action_chunks), 'action chunks')

    # Warmup steps
    for _ in range(20):
        step_sim(sim, model, n=1)

    # Execute action chunks sequentially
    for i, chunk in enumerate(action_chunks):
        print(f'Executing chunk {i+1}/{len(action_chunks)}: {chunk.get("name")}')
        # chunk contains dict of actuator_name->value
        apply_ctrls(sim, model, act_map, chunk['ctrls'], steps=150, sleep=0.0, attach_info=attach_info)
        # handle logical attach/release so the cube follows the gripper
        if chunk.get('attach'):
            print('Simulated grasp: attaching cube to gripper (logical attach)')
            attach_info['attached'] = True
        if chunk.get('release'):
            print('Simulated release: releasing cube (logical detach)')
            attach_info['attached'] = False

    print('Demo finished.')
    return 0


def apply_ctrls_live(sim, model, act_map, ctrl_targets, renderer=None, ctx=None, steps=200, width=640, height=480, sleep=0.0):
    """Apply controls while rendering to an in-process viewer (if present)."""
    ctrl_idx = {name: idx for name, idx in act_map.items() if name in ctrl_targets}
    target_arr = np.array([ctrl_targets[name] for name in ctrl_idx.keys()], dtype=float)

    # snapshot current control values
    cur_vals_list = []
    for idx in ctrl_idx.values():
        try:
            cur_vals_list.append(float(sim.data.ctrl[idx]))
        except Exception:
            try:
                cur_vals_list.append(float(sim.ctrl[idx]))
            except Exception:
                cur_vals_list.append(0.0)
    cur_vals = np.array(cur_vals_list, dtype=float)

    for t in range(1, steps + 1):
        alpha = t / steps
        vals = (1 - alpha) * cur_vals + alpha * target_arr
        for k, v in zip(ctrl_idx.values(), vals):
            try:
                sim.data.ctrl[k] = float(v)
            except Exception:
                try:
                    sim.ctrl[k] = float(v)
                except Exception:
                    pass

        # step physics
        step_sim(sim, model, n=1)

        # render frame if available
        if renderer is not None:
            try:
                _ = renderer.render(width, height)
            except Exception:
                pass

        # if the cube is logically attached, override its qpos to follow gripper
        # (renderer may be available even when we don't have attach info)
        if hasattr(apply_ctrls_live, 'attach_info') and apply_ctrls_live.attach_info.get('attached'):
            try:
                ai = apply_ctrls_live.attach_info
                g_id = ai.get('gripper_site_id')
                c_qadr = ai.get('cube_qposadr')
                c_qdim = ai.get('cube_qpos_dim')
                pos_first = ai.get('pos_first')
                if g_id is not None and c_qadr is not None:
                    gp = sim.data.site_xpos[g_id]
                    if pos_first:
                        sim.data.qpos[c_qadr:c_qadr+3] = gp
                    else:
                        sim.data.qpos[c_qadr + c_qdim - 3:c_qadr + c_qdim] = gp
                    try:
                        import mujoco as _muj
                        _muj.mj_forward(model, sim)
                    except Exception:
                        pass
            except Exception:
                pass

        if ctx is not None:
            try:
                # swap buffers + poll events if GlfwContext-like API
                if hasattr(ctx, 'swap_buffers'):
                    ctx.swap_buffers()
                if hasattr(ctx, 'poll_events'):
                    ctx.poll_events()
            except Exception:
                pass

        if sleep:
            time.sleep(sleep)


def live_demo(width=800, height=600):
    """Run the pick-and-place demo with an in-process live viewer."""
    try:
        import mujoco
    except Exception:
        print('mujoco not available for live demo')
        return 2

    # build a model that includes the cube and target
    from mujoco_demo.scene_builder import build_model_with_cube_and_target
    model, sim, attach_info = build_model_with_cube_and_target()

    act_map = find_actuator_map(model)
    print('Found actuators:', list(act_map.keys()))

    from perception.analytic_pose import cube_start_and_target
    from policy.act_stub import plan_pick_place
    cube_pose, target_pose = cube_start_and_target()
    action_chunks = plan_pick_place(cube_pose, target_pose)
    print('Planned', len(action_chunks), 'action chunks')

    # Try to use mujoco.viewer.launch if available
    try:
        from mujoco import viewer as mujviewer
        try:
            print('Launching mujoco.viewer.launch (may open separate interactive window)')
            # try model/data signature first
            mujviewer.launch(model=model, data=sim)
            return 0
        except Exception:
            try:
                mujviewer.launch(str(SO101_XML))
                return 0
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: create a GlfwContext + Renderer and render while stepping
    try:
        from mujoco.glfw import GlfwContext
        ctx = GlfwContext()
        ctx.make_context()
        renderer = getattr(mujoco, 'Renderer')(model)
        print('Using GlfwContext + Renderer for live display')
    except Exception as e:
        print('Live viewer not fully available:', e)
        ctx = None
        renderer = None

    # warmup
    for _ in range(20):
        step_sim(sim, model, n=1)

    # bind attach_info to the live apply function so it can keep the cube following
    # the gripper while attached
    apply_ctrls_live.attach_info = attach_info

    for i, chunk in enumerate(action_chunks):
        print(f'Live: Executing chunk {i+1}/{len(action_chunks)}: {chunk.get("name")}')
        apply_ctrls_live(sim, model, act_map, chunk['ctrls'], renderer=renderer, ctx=ctx, steps=150, width=width, height=height)
        if chunk.get('attach'):
            print('Simulated grasp: attaching cube to gripper (logical attach)')
            attach_info['attached'] = True
        if chunk.get('release'):
            print('Simulated release: releasing cube (logical detach)')
            attach_info['attached'] = False

    print('Live demo finished.')
    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--live', action='store_true', help='Run with in-process live viewer')
    parser.add_argument('--width', type=int, default=800, help='Viewer width')
    parser.add_argument('--height', type=int, default=600, help='Viewer height')
    args = parser.parse_args()

    if args.live:
        rc = live_demo(width=args.width, height=args.height)
    else:
        rc = demo_run(duration=args.duration)
    exit(rc)
