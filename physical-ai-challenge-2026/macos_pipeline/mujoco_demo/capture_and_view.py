#!/usr/bin/env python3
"""Open the SO101 scene in the MuJoCo app (macOS) and optionally capture a few frames.

This script attempts to:
- Open the `scene.xml` in the system MuJoCo application (`open -a MuJoCo ...`).
- Launch the Python viewer (in a subprocess) so you can see the sim window.
- Try an offscreen capture using `mujoco.Renderer` if available and save a few frames.

If rendering is not available in the installed mujoco Python package, the
script will still try to open the scene in your MuJoCo app so you can view it.
"""
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'scene.xml'
VIEWER_SCRIPT = HERE / 'viewer.py'
OUTDIR = HERE.parent / 'datasets' / 'synthetic_cube'


def open_external_app():
    # Try to open with the MuJoCo app explicitly, then fallback to default open
    try:
        print('Opening scene in MuJoCo app...')
        subprocess.run(['open', '-a', 'MuJoCo', str(SCENE)], check=True)
        return True
    except Exception:
        try:
            print('Fallback: opening scene with default app...')
            subprocess.run(['open', str(SCENE)], check=True)
            return True
        except Exception as e:
            print('Could not open external app:', e)
            return False


def launch_python_viewer():
    # Launch viewer.py in a separate process so we can continue
    if not VIEWER_SCRIPT.exists():
        print('Viewer script not found:', VIEWER_SCRIPT)
        return None
    print('Launching Python viewer subprocess...')
    return subprocess.Popen([sys.executable, str(VIEWER_SCRIPT)])


def try_offscreen_capture(frames=5, width=640, height=480):
    try:
        import mujoco
        from PIL import Image
    except Exception as e:
        print('Required modules for capture missing:', e)
        return False

    try:
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        sim = mujoco.MjData(model)
    except Exception as e:
        print('Failed to load model for capture:', e)
        return False

    # Try modern Renderer API
    try:
        Renderer = getattr(mujoco, 'Renderer')
        renderer = Renderer(model)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            mujoco.mj_step(model, sim)
            try:
                img = renderer.render(width, height)
                if img is None:
                    print('Renderer returned None for frame', i)
                    break
                # Save image (assuming uint8 HxWx3)
                Image.fromarray(img).save(OUTDIR / f'frame_{i:03d}.png')
                print('Saved', OUTDIR / f'frame_{i:03d}.png')
            except Exception as e:
                print('Renderer.render failed:', e)
                break
        return True
    except Exception as e:
        print('Renderer not available or failed:', e)

    # If Renderer not available, try a minimal fallback: step sim and dump joint state
    try:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for i in range(frames):
            mujoco.mj_step(model, sim)
            # Save simple metadata as placeholder
            (OUTDIR / f'frame_{i:03d}.txt').write_text(str(sim.qpos.tolist()))
            print('Saved qpos for frame', i)
        return True
    except Exception as e:
        print('Fallback capture failed:', e)
        return False


def main(open_app=True, capture=True, keep_viewer=True):
    if open_app:
        open_external_app()

    viewer_proc = launch_python_viewer()

    if capture:
        ok = try_offscreen_capture(frames=8)
        print('Capture result:', ok)

    if not keep_viewer and viewer_proc is not None:
        time.sleep(2)
        viewer_proc.terminate()

    print('Done. If you opened the external MuJoCo app, check it now.')


if __name__ == '__main__':
    main(open_app=True, capture=True, keep_viewer=True)
