"""
Find best PLACE waypoint: gripper needs to be at CUBE_PLACE_POS=[0.18, -0.12, 0.07]
Uses a full grid search across all joint ranges.
"""
import mujoco, numpy as np, os, math

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"
CUBE_START_POS = np.array([0.18, 0.0, 0.07])
CUBE_PLACE_POS = np.array([0.18, -0.12, 0.07])

with open(SCENE_XML_PATH) as f: xml = f.read()
inject = f"""
    <body name="table" pos="0.25 0 0.025">
        <geom type="box" size="0.2 0.2 0.025" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="red_cube" pos="{CUBE_START_POS[0]} {CUBE_START_POS[1]} {CUBE_START_POS[2]}">
        <freejoint/>
        <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.05"/>
    </body>
"""
xml = xml.replace('</worldbody>', f'    {inject}\n    </worldbody>')
os.chdir(os.path.dirname(SCENE_XML_PATH))
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

print(f"Searching for PLACE config closest to {CUBE_PLACE_POS} ...")
print(f"(placing with gripper closed, i.e. lowering cube onto table)\n")

targets = [
    ("GRAB",  CUBE_START_POS),
    ("PLACE", CUBE_PLACE_POS),
]

for target_name, target in targets:
    print(f"=== Searching for {target_name} config (target={target}) ===")
    best_dist = 9999
    best_cfg = None
    best_pos = None

    for pan in np.linspace(-1.5, 1.5, 13):
        for s_lift in np.linspace(-0.8, 0.2, 9):
            for e_flex in np.linspace(-0.3, 0.5, 7):
                for w_flex in [1.55, 1.60, 1.65, 1.70]:
                    cfg = [pan, s_lift, e_flex, w_flex, 0.0, 0.7]
                    data.ctrl[:6] = cfg
                    for _ in range(500):
                        mujoco.mj_step(model, data)
                    gp = data.site_xpos[site_id].copy()
                    d = np.linalg.norm(gp - target)
                    if d < best_dist:
                        best_dist = d
                        best_cfg = cfg[:]
                        best_pos = gp.copy()

    print(f"  Best distance : {best_dist*1000:.1f} mm")
    print(f"  Best gripper  : [{best_pos[0]:.4f}, {best_pos[1]:.4f}, {best_pos[2]:.4f}]")
    print(f"  Best joints   : pan={best_cfg[0]:.4f}, s_lift={best_cfg[1]:.4f}, e_flex={best_cfg[2]:.4f}, w_flex={best_cfg[3]:.4f}")
    print()
