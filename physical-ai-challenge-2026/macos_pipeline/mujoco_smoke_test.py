#!/usr/bin/env python3
"""MuJoCo smoke test — load the SO101 MJCF and step the simulator a few times.

This script tries the modern `mujoco` Python API first and falls back to
`mujoco_py` if available. It is intentionally simple so it works on macOS
environments where only CPU is available.
"""
import glob
import os
import sys


def find_model():
    here = os.path.dirname(__file__)
    robot_dir = os.path.join(here, 'sim', 'robots', 'SO101')
    if not os.path.isdir(robot_dir):
        return None
    candidates = glob.glob(os.path.join(robot_dir, '*.xml')) + glob.glob(os.path.join(robot_dir, '*.mjcf'))
    if not candidates:
        return None
    # Prefer typical SO101 filenames
    for c in candidates:
        base = os.path.basename(c).lower()
        if base.startswith('so101') or base.startswith('so101_new'):
            return c
    return candidates[0]


def main():
    model_path = find_model()
    if not model_path:
        print('No SO101 MJCF/MJCF-like XML found in sim/robots/SO101.')
        print('Please place your SO101 MJCF at: sim/robots/SO101/')
        sys.exit(2)

    print('Using model:', model_path)

    try:
        import mujoco
    except Exception as e:
        print('Importing `mujoco` failed:', e)
        mujoco = None

    if mujoco is not None:
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
            data = mujoco.MjData(model)
            # Step the sim a few times to verify it runs
            for _ in range(10):
                mujoco.mj_step(model, data)
            print('Success: loaded and stepped model with `mujoco` package.')
            return 0
        except Exception as e:
            print('`mujoco` API run failed:', e)

    # Fallback: mujoco_py
    try:
        import mujoco_py
        model = mujoco_py.load_model_from_path(model_path)
        sim = mujoco_py.MjSim(model)
        sim.step()
        print('Success: loaded and stepped model with `mujoco_py` fallback.')
        return 0
    except Exception as e:
        print('`mujoco_py` fallback failed:', e)

    print('Failed to run any MuJoCo backend. Ensure MuJoCo is installed and available.')
    return 4


if __name__ == '__main__':
    sys.exit(main())
