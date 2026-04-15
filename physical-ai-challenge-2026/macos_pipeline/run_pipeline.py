#!/usr/bin/env python3
import argparse
import glob
import importlib
import os
import subprocess
import sys


HERE = os.path.dirname(__file__)


def find_so101_model():
    robot_dir = os.path.join(HERE, 'sim', 'robots', 'SO101')
    if not os.path.isdir(robot_dir):
        return None
    candidates = glob.glob(os.path.join(robot_dir, '*.xml')) + glob.glob(os.path.join(robot_dir, '*.mjcf'))
    if not candidates:
        return None
    for c in candidates:
        base = os.path.basename(c).lower()
        if base.startswith('so101'):
            return c
    return candidates[0]


def test_imports():
    pkgs = [
        ('mujoco', 'mujoco'),
        ('ultralytics', 'ultralytics'),
        ('onnxruntime', 'onnxruntime'),
        ('open3d', 'open3d'),
        ('cv2', 'cv2'),
        ('numpy', 'numpy'),
    ]
    for name, mod in pkgs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, '__version__', getattr(m, 'version', 'unknown'))
            print(f'{name}: OK (version {ver})')
        except Exception as e:
            print(f'{name}: FAILED -> {e}')

    sofile = find_so101_model()
    if sofile:
        print('SO101 model found at:', sofile)
    else:
        print('SO101 model not found. Place MJCF/XML in:', os.path.join(HERE, 'sim', 'robots', 'SO101'))


def start_pipeline():
    print('Pipeline start requested.')
    print('This is an early scaffold. Steps to run once models and SO101 are available:')
    print('- Ensure sim/robots/SO101 contains your MJCF/XML model (so101_new_calib.xml is fine)')
    print('- Place YOLO/DenseFusion/ACT ONNX models in macos_pipeline/models/')
    print('- Then implement or run the real-time loop (not implemented in this scaffold)')


def run_mujoco_smoke_test():
    script = os.path.join(HERE, 'mujoco_smoke_test.py')
    if not os.path.exists(script):
        print('Smoke test script not found:', script)
        return 2
    print('Invoking MuJoCo smoke test...')
    proc = subprocess.run([sys.executable, script])
    return proc.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-imports', action='store_true')
    parser.add_argument('--test-mujoco', action='store_true', help='Run the MuJoCo smoke test')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()

    if args.test_imports:
        test_imports()
    elif args.test_mujoco:
        rc = run_mujoco_smoke_test()
        sys.exit(rc)
    elif args.start:
        start_pipeline()
    else:
        parser.print_help()
