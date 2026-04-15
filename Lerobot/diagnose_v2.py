"""
Targeted diagnostic: measure exact gripper position for every waypoint
in the current WAYPOINT_SEQUENCE and validate the cube is reachable.
"""
import mujoco
import numpy as np
import os
import math

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"
CUBE_START_POS = np.array([0.18, 0.0, 0.07])
CUBE_PLACE_POS = np.array([0.18, -0.12, 0.07])
PLACE_PAN = math.atan2(CUBE_PLACE_POS[1], CUBE_PLACE_POS[0] - 0.0388)

WAYPOINTS = [
    ("NEUTRAL",      [0.0,  0.0,    0.0,    0.0,    0.0,   0.7]),
    ("PRE_APPROACH", [0.0,  0.104, -0.172,  1.583,  0.0,   0.7]),
    ("APPROACH",     [0.0,  0.035, -0.034,  1.650,  0.0,   0.7]),
    ("GRAB_ADVANCE", [0.0, -0.173,  0.103,  1.650,  0.0,   0.7]),
    ("CLOSE_GRIP",   [0.0, -0.173,  0.103,  1.650,  0.0,  -0.10]),
    ("LIFT",         [0.0, -0.729, -0.103,  1.650,  0.0,  -0.10]),
    ("PAN_TO_PLACE", [PLACE_PAN, -0.729, -0.103, 1.650, 0.0, -0.10]),
    ("PLACE_LOWER",  [PLACE_PAN, -0.173,  0.103,  1.650,  0.0,  -0.10]),
    ("RELEASE",      [PLACE_PAN, -0.173,  0.103,  1.650,  0.0,   0.7]),
    ("RETURN",       [0.0,  0.0,    0.0,    0.0,    0.0,   0.7]),
]

with open(SCENE_XML_PATH, "r") as f:
    xml = f.read()

inject = f"""
    <body name="table" pos="0.25 0 0.025">
        <geom type="box" size="0.2 0.2 0.025" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="red_cube" pos="{CUBE_START_POS[0]} {CUBE_START_POS[1]} {CUBE_START_POS[2]}">
        <freejoint/>
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.05" friction="1.5 0.005 0.0001"/>
    </body>
    <body name="place_marker" pos="{CUBE_PLACE_POS[0]} {CUBE_PLACE_POS[1]} {CUBE_PLACE_POS[2]}">
        <geom type="box" size="0.025 0.025 0.003" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
    <camera name="rgbd_cam" pos="0.15 0.5 0.4" fovy="50" xyaxes="1 0 0 0 -0.8 0.6"/>
"""
xml = xml.replace('</worldbody>', f'    {inject}\n    </worldbody>')
os.chdir(os.path.dirname(SCENE_XML_PATH))
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "red_cube")

print(f"\nCUBE START : {CUBE_START_POS}")
print(f"CUBE PLACE : {CUBE_PLACE_POS}")
print(f"PLACE_PAN  : {PLACE_PAN:.4f} rad  ({math.degrees(PLACE_PAN):.1f} deg)")
print(f"\nnq={model.nq}, nu={model.nu}")
print("="*80)
print(f"{'WAYPOINT':15s}  {'gripper_x':>9} {'gripper_y':>9} {'gripper_z':>9}  {'dist2cube':>9}  {'gz_axis':>7}")
print("="*80)

for name, joints in WAYPOINTS:
    data.ctrl[:6] = joints
    # Settle for 2000 steps (= 4 seconds at 2ms timestep)
    for _ in range(2000):
        mujoco.mj_step(model, data)

    gpos = data.site_xpos[site_id].copy()
    # gripper Z-axis (pointing direction)
    gmat = data.site_xmat[site_id].reshape(3,3)
    gz   = gmat[:, 2]   # Z-column = gripper forward axis

    dist = np.linalg.norm(gpos - CUBE_START_POS)
    print(f"{name:15s}  {gpos[0]:9.4f} {gpos[1]:9.4f} {gpos[2]:9.4f}  {dist:9.4f}  [{gz[0]:+.3f},{gz[1]:+.3f},{gz[2]:+.3f}]")

    # Reset cube position after each waypoint
    if cube_id >= 0:
        cube_qpos_adr = model.jnt_qposadr[model.body_jntadr[cube_id]]
        data.qpos[cube_qpos_adr:cube_qpos_adr+3] = CUBE_START_POS
        data.qpos[cube_qpos_adr+3:cube_qpos_adr+7] = [1,0,0,0]  # identity quat

print("="*80)
print(f"\n[GRAB_ADVANCE gripper vs cube]")
# Run GRAB_ADVANCE specifically and show delta
data.ctrl[:6] = [0.0, -0.173, 0.103, 1.650, 0.0, 0.7]
for _ in range(3000):
    mujoco.mj_step(model, data)
gp = data.site_xpos[site_id].copy()
delta = gp - CUBE_START_POS
print(f"  Gripper : {gp}")
print(f"  Cube    : {CUBE_START_POS}")
print(f"  Delta   : {delta}  (need X-delta ≈ 0.02 for side approach)")

# Find the best pan/tilt for horizontal reach to cube
print("\n[Searching for best approach to cube via horizontal side-grasp...]")
best_dist = 9999
best_cfg = None
for pan in np.linspace(-0.3, 0.3, 7):
    for s_lift in np.linspace(-0.3, 0.1, 5):
        for e_flex in np.linspace(-0.2, 0.4, 5):
            for w_flex in [1.60, 1.65, 1.70]:
                cfg = [pan, s_lift, e_flex, w_flex, 0.0, 0.7]
                data.ctrl[:6] = cfg
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                gp = data.site_xpos[site_id].copy()
                d = np.linalg.norm(gp - CUBE_START_POS)
                if d < best_dist:
                    best_dist = d
                    best_cfg = cfg[:]
                    best_pos = gp.copy()

print(f"  Best distance: {best_dist:.4f} m")
print(f"  Best gripper : {best_pos}")
print(f"  Best joints  : pan={best_cfg[0]:.3f}, s_lift={best_cfg[1]:.3f}, e_flex={best_cfg[2]:.3f}, w_flex={best_cfg[3]:.3f}")
