"""
SO-101 MuJoCo Pick-and-Place — HEADLESS DATA GENERATOR
Matched with the DEFINITIVE v4 workspace geometry (side grasp).
Optimized for headless environment, dataset recording, and high speed.
"""
import mujoco
import time
import os
import math
import numpy as np

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"

# Cube at X=0.18 in front of robot, Y=0, on table (table top at Z=0.05, cube half=0.02)
CUBE_START_POS = np.array([0.18, 0.0, 0.07])
# Place target: rotated to the side
CUBE_PLACE_POS = np.array([0.18, -0.12, 0.07])

# ──────────────── HARDCODED JOINT WAYPOINTS ────────────────
# Format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
# Gripper: 0.7 = open, -0.10 = closed

PLACE_PAN = math.atan2(CUBE_PLACE_POS[1] - 0.0, CUBE_PLACE_POS[0] - 0.0388)

WAYPOINT_SEQUENCE = [
    # (name,         [pan,  s_lift, e_flex, w_flex, w_roll, grip], wait_s)
    ("NEUTRAL",       [0.0,  0.0,    0.0,    0.0,    0.0,   0.7],  0.5),
    ("PRE_APPROACH",  [0.0,  0.104, -0.172,  1.583,  0.0,   0.7],  0.2),
    ("APPROACH",      [0.0,  0.035, -0.034,  1.650,  0.0,   0.7],  0.2),
    ("GRAB_ADVANCE",  [0.0, -0.173,  0.103,  1.650,  0.0,   0.7],  0.2),
    ("CLOSE_GRIP",    [0.0, -0.173,  0.103,  1.650,  0.0,  -0.10], 0.5),
    ("LIFT",          [0.0, -0.729, -0.103,  1.650,  0.0,  -0.10], 0.5),
    ("PAN_TO_PLACE",  [PLACE_PAN, -0.729, -0.103,  1.650,  0.0,  -0.10], 0.5),
    ("PLACE_LOWER",   [PLACE_PAN, -0.173,  0.103,  1.650,  0.0,  -0.10], 0.5),
    ("RELEASE",       [PLACE_PAN, -0.173,  0.103,  1.650,  0.0,   0.7],  0.5),
    ("RETRACT",       [PLACE_PAN, -0.729, -0.103,  1.650,  0.0,   0.7],  0.5),
    ("RETURN",        [0.0,  0.0,    0.0,    0.0,    0.0,   0.7],  0.5),
]


def create_simulation():
    print(f"[*] Loading SO-101 from: {SCENE_XML_PATH}")
    with open(SCENE_XML_PATH, "r") as f:
        xml_content = f.read()

    injections = f"""
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

    xml_content = xml_content.replace('</worldbody>', f'    {injections}\n    </worldbody>')
    os.chdir(os.path.dirname(SCENE_XML_PATH))

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    return model, data

def run_simulation(model, data):
    MAX_EPISODES = 500
    current_episode = 0
    dataset_dir = "ACT_Dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    renderer = mujoco.Renderer(model, 480, 640)

    waypoint_idx = 0
    sim_steps_at_waypoint = 0
    SETTLE_STEPS = 500

    dataset_buffer = {"images": [], "qpos": [], "ctrl": []}
    step_count = 0

    print("[*] ═══════════════════════════════════════════════════")
    print("[*]  SO-101 HEADLESS DATA GENERATOR — DEFINITIVE v4")
    print("[*] ═══════════════════════════════════════════════════")

    data.ctrl[:6] = WAYPOINT_SEQUENCE[0][1]

    start_time = time.time()

    while current_episode < MAX_EPISODES:
        renderer.update_scene(data, camera="rgbd_cam")
        img = renderer.render()

        if waypoint_idx > 0 and step_count % 10 == 0:
            dataset_buffer["images"].append(img)
            dataset_buffer["qpos"].append(np.copy(data.qpos[:6]))
            dataset_buffer["ctrl"].append(np.copy(data.ctrl[:6]))

        if waypoint_idx < len(WAYPOINT_SEQUENCE):
            name, target_ctrl, wait_s = WAYPOINT_SEQUENCE[waypoint_idx]
            wait_steps = max(int(wait_s / model.opt.timestep), SETTLE_STEPS)

            data.ctrl[:6] = target_ctrl
            sim_steps_at_waypoint += 1

            joint_err = np.linalg.norm(data.qpos[:6] - np.array(target_ctrl))

            if sim_steps_at_waypoint >= wait_steps and (joint_err < 0.35 or sim_steps_at_waypoint > wait_steps * 3):
                waypoint_idx += 1
                sim_steps_at_waypoint = 0
                if waypoint_idx < len(WAYPOINT_SEQUENCE):
                    data.ctrl[:6] = WAYPOINT_SEQUENCE[waypoint_idx][1]
                else:
                    # Episode Complete
                    filepath = os.path.join(dataset_dir, f"episode_{current_episode}.npz")
                    np.savez(filepath,
                             images=np.array(dataset_buffer["images"]),
                             qpos=np.array(dataset_buffer["qpos"]),
                             ctrl=np.array(dataset_buffer["ctrl"]))
                    
                    elapsed = time.time() - start_time
                    fps = step_count / elapsed
                    print(f"[✓] Simulated Episode {current_episode+1}/{MAX_EPISODES} | Saved {len(dataset_buffer['images'])} steps | sim speed: {fps:.0f} Hz")
                    
                    current_episode += 1

                    data.ctrl[:6] = POSE_NEUTRAL
                    mujoco.mj_resetData(model, data)
                    
                    data.ctrl[:6] = POSE_NEUTRAL
                    waypoint_idx = 0
                    sim_steps_at_waypoint = 0
                    dataset_buffer = {"images": [], "qpos": [], "ctrl": []}

        step_count += 1
        mujoco.mj_step(model, data)

if __name__ == "__main__":
    model, data = create_simulation()
    
    # We define POSE_NEUTRAL so mj_resetData doesn't crash on reference
    POSE_NEUTRAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    
    run_simulation(model, data)
