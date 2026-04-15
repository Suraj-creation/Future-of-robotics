import mujoco
import numpy as np
import scipy.optimize
import os

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"
os.chdir(os.path.dirname(SCENE_XML_PATH))

with open(SCENE_XML_PATH, "r") as f:
    orig_xml = f.read()

injections = f"""
    <body name="table" pos="0.25 0 0.025">
        <geom type="box" size="0.2 0.2 0.025" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="red_cube" pos="0.18 0.0 0.07">
        <freejoint/>
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.05" friction="2.0 0.1 0.001"/>
    </body>
"""
xml_content = orig_xml.replace('</worldbody>', f'    {injections}\n    </worldbody>')

model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")

# Extracted from earlier IK
APPROACH = [-0., -0.818, 0.745, 1.66, 0.]
GRAB = [-0., -0.158, 0.244, 1.65, -0.]
LIFT = [-0., -0.993, 0.177, 1.65, -0.]

seq = [
    (list(APPROACH) + [1.5], 1000),
    (list(GRAB) + [1.5], 1000),
    (list(GRAB) + [-0.10], 1000),
    (list(LIFT) + [-0.10], 1000)
]

for ctrl, steps in seq:
    data.ctrl[:6] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)
        
success = data.xpos[cube_id][2] > 0.15
print(f"Force-Wait Test -> Lift Success: {success} (Final Z: {data.xpos[cube_id][2]:.3f})")

