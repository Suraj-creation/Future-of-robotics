#!/usr/bin/env python3
"""Run the existing demo and record frames to an MP4.

This script runs a deterministic pick-and-place using the stub ACT policy
and analytic perception (same logic as `runner.py`) and records frames
using `mujoco.Renderer` (if available). It then tries to encode frames into
`demo.mp4` using `ffmpeg` (if installed).

Usage:
  cd physical-ai-challenge-2026/macos_pipeline/mujoco_demo
  conda activate robothon
  python record_demo.py --out ../datasets/synthetic_cube/demo --w 640 --h 480

Output:
  - frames in `<out>/frames/`
  - `demo.mp4` in `<out>/` (if ffmpeg is present)
"""
from pathlib import Path
import argparse
import sys
import time
import shutil
import subprocess

try:
    import mujoco
except Exception as e:
    print('mujoco not available:', e)
    sys.exit(1)

from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
SO101_XML = HERE.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'so101_new_calib.xml'

# Ensure package root (macos_pipeline) is importable so sibling packages
# like `perception` and `policy` can be imported when running from
# `mujoco_demo` working directory.
PKG_ROOT = str(HERE.parent)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


def step_sim(sim, model, n=1):
    try:
        import mujoco as _mujoco
        for _ in range(n):
            _mujoco.mj_step(model, sim)
        return
    except Exception:
        pass
    try:
        from mujoco import mj_step as _mj_step
        for _ in range(n):
            _mj_step(model, sim)
        return
    except Exception:
        pass
    try:
        from mujoco import mj_step as _mj_step
        for _ in range(n):
            _mj_step(sim)
        return
    except Exception:
        pass
    for _ in range(n):
        if hasattr(sim, 'step'):
            sim.step()
        else:
            raise RuntimeError('No known mujoco step API available')


def find_actuator_map(model):
    names = []
    for i in range(model.na):
        try:
            names.append(model.actuator(i).name)
        except Exception:
            names.append(None)
    return {n: i for i, n in enumerate(names) if n}


def apply_ctrls_record(sim, model, act_map, ctrl_targets, renderer, frame_dir, steps=200, width=640, height=480, sleep=0.0, start_idx=0, attach_info=None):
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

    frame_idx = start_idx
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

        # render and save if renderer present
        if renderer is not None:
            try:
                img = renderer.render(width, height)
                if img is not None:
                    Image.fromarray(img).save(frame_dir / f'frame_{frame_idx:06d}.png')
                    frame_idx += 1
            except Exception as e:
                print('Renderer frame failed:', e)

        # enforce attachment if requested: override cube qpos to follow gripper
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

    return frame_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE.parent / 'datasets' / 'demo_record')
    parser.add_argument('--w', type=int, default=640)
    parser.add_argument('--h', type=int, default=480)
    parser.add_argument('--frames', type=int, default=2000)
    args = parser.parse_args()

    outdir = args.out
    frames_dir = outdir / 'frames'
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # build a model that includes the cube and target area
    from mujoco_demo.scene_builder import build_model_with_cube_and_target
    model, sim, attach_info = build_model_with_cube_and_target()

    act_map = find_actuator_map(model)
    print('Actuators:', list(act_map.keys()))

    # perception & policy
    from perception.analytic_pose import cube_start_and_target
    from policy.act_stub import plan_pick_place
    cube_pose, target_pose = cube_start_and_target()
    action_chunks = plan_pick_place(cube_pose, target_pose)
    print('Planned', len(action_chunks), 'action chunks')

    # try to create renderer
    try:
        Renderer = getattr(mujoco, 'Renderer')
        renderer = Renderer(model)
        print('Renderer available')
    except Exception as e:
        print('Renderer not available, frames will not be recorded:', e)
        renderer = None

    # warmup
    for _ in range(20):
        step_sim(sim, model, n=1)
        if renderer is not None:
            try:
                img = renderer.render(args.w, args.h)
                if img is not None:
                    Image.fromarray(img).save(frames_dir / f'frame_{_ :06d}.png')
            except Exception:
                pass

    fi = 0
    for i, chunk in enumerate(action_chunks):
        print('Executing chunk', i + 1, '/', len(action_chunks), chunk.get('name'))
        fi = apply_ctrls_record(sim, model, act_map, chunk['ctrls'], renderer, frames_dir, steps=150, width=args.w, height=args.h, start_idx=fi, attach_info=attach_info)
        if chunk.get('attach'):
            print('Simulated grasp')
            attach_info['attached'] = True
        if chunk.get('release'):
            print('Simulated release')
            attach_info['attached'] = False

    # encode with ffmpeg if available
    demo_mp4 = outdir / 'demo.mp4'
    ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', '30', '-i', str(frames_dir / 'frame_%06d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(demo_mp4)]
    try:
        print('Encoding video (ffmpeg)...')
        subprocess.run(ffmpeg_cmd, check=True)
        print('Saved video to', demo_mp4)
        # try to open video
        try:
            subprocess.run(['open', str(demo_mp4)])
        except Exception:
            pass
    except Exception as e:
        print('ffmpeg encode failed or not installed:', e)
        print('Frames are in', frames_dir)

    print('Done.')


if __name__ == '__main__':
    main()
