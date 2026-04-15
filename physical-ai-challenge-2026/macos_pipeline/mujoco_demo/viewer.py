#!/usr/bin/env python3
"""Simple MuJoCo viewer launcher for the SO101 scene.

This tries to use the installed `mujoco` Python package's viewer API. It will
fall back to printing the scene path so you can open it manually in the MuJoCo
app if the Python viewer is not available.
"""
from pathlib import Path
import time


SCENE = Path(__file__).resolve().parent.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'scene.xml'


def main():
    try:
        import mujoco
        # New-style viewer
        try:
            from mujoco import viewer
            print('Starting mujoco.viewer.show for', SCENE)
            viewer.launch(str(SCENE))
            return
        except Exception:
            pass

        # Older style: use mujoco.MjModel / MjData with glfw/GL
        try:
            model = mujoco.MjModel.from_xml_path(str(SCENE))
            data = mujoco.MjData(model)
            # Use simple GLFW window if available
            try:
                from mujoco.glfw import GlfwContext
                print('Using mujoco.glfw viewer...')
                ctx = GlfwContext()
                ctx.make_context()
                while True:
                    mujoco.mj_step(model, data)
                    time.sleep(0.01)
            except Exception as e:
                print('No glfw viewer available:', e)
                print('Model loaded. Open the scene in MuJoCo app:', SCENE)
                return
        except Exception as e:
            print('Failed to load model for viewer:', e)
            print('Open scene manually:', SCENE)
            return

    except Exception as e:
        print('MuJoCo python package not available:', e)
        print('Open scene manually in your MuJoCo app:', SCENE)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Launch a MuJoCo Python viewer for the SO101 scene.

This script tries the modern `mujoco.viewer.launch` entry point and
exits with a non-zero code if it cannot open a viewer.
"""
from pathlib import Path
import sys

SCENE = Path(__file__).resolve().parent.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'scene.xml'

def main():
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
    except Exception as e:
        print('Failed to load model for viewer:', e)
        return 2

    # Prefer the high-level viewer API if available
    try:
        import mujoco.viewer as mv
        print('Launching mujoco.viewer...')
        mv.launch(model, data)
        return 0
    except Exception:
        pass

    # Fallback: try older viewer entry point
    try:
        from mujoco import viewer
        print('Launching mujoco.viewer (fallback)...')
        viewer.launch(model, data)
        return 0
    except Exception as e:
        print('Could not launch Python viewer:', e)
        return 3


if __name__ == '__main__':
    sys.exit(main())
