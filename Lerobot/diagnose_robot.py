"""Quick check: what does qpos look like with the freejoint cube?"""
import mujoco
import numpy as np
import os

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"

with open(SCENE_XML_PATH, "r") as f:
    xml = f.read()
inject = """
    <body name="table" pos="0.25 0 0.025">
        <geom type="box" size="0.2 0.2 0.025" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="red_cube" pos="0.18 0.0 0.07">
        <freejoint/>
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.05"/>
    </body>
"""
xml = xml.replace('</worldbody>', f'    {inject}\n    </worldbody>')
os.chdir(os.path.dirname(SCENE_XML_PATH))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print(f"model.nq = {model.nq} (total qpos dimension)")
print(f"model.nv = {model.nv} (total qvel dimension)")
print(f"model.nu = {model.nu} (number of actuators)")
print(f"model.njnt = {model.njnt} (number of joints)")
print()

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i]
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    print(f"  Joint {i}: name={str(name):25s} type={type_names.get(jnt_type, '?'):6s} qpos_adr={qpos_adr}  dof_adr={dof_adr}")

print(f"\nActuator ctrl-to-qpos mapping:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    # Position actuators target a joint; find which joint
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    qpos_adr = model.jnt_qposadr[jnt_id]
    print(f"  ctrl[{i}] = actuator '{name}' -> joint '{jnt_name}' -> qpos[{qpos_adr}]")

# Now set all robot joints to 0 and step
mujoco.mj_forward(model, data)
print(f"\nqpos = {data.qpos}")
print(f"ctrl  = {data.ctrl}")
