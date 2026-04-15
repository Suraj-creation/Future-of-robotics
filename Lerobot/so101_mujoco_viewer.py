"""
SO-101 MuJoCo Pick-and-Place — VISUAL VIEWER (v5 DEFINITIVE)

Identical pipeline to autonomous_pick_place.py but with richer visual overlays.
Uses the same IK-solved waypoints and smooth pan interpolation.
"""
import mujoco
import mujoco.viewer
import time
import os
import math
import numpy as np
import cv2
import scipy.optimize

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"

CUBE_START_POS = np.array([0.18, 0.0, 0.07])
CUBE_PLACE_POS = np.array([0.18, -0.12, 0.07])


def solve_ik(model, data, site_id, target_pos, pan_hint=0.0):
    def cost(q):
        data.qpos[:5] = q
        mujoco.mj_kinematics(model, data)
        pos_err = np.linalg.norm(data.site_xpos[site_id] - target_pos) * 100
        return pos_err + (q[3] - 1.65)**2 * 2
    res = scipy.optimize.minimize(
        cost, [pan_hint, -0.157, 0.243, 1.651, 0.0], method='SLSQP',
        bounds=[(-1.92, 1.92), (-1.75, 1.75), (-1.69, 1.69), (-1.66, 1.66), (-2.74, 2.84)])
    return list(res.x)


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
            <geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.05" friction="2.0 0.1 0.001"/>
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


def densefusion_6d_inference(model, data):
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    return {
        "pos": np.copy(data.xpos[cube_id]),
        "mat": np.copy(data.xmat[cube_id].reshape(3, 3))
    }


def run_simulation(model, data):
    dataset_dir = "ACT_Dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    renderer = mujoco.Renderer(model, 480, 640)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    cube_joint_id = model.body_jntadr[cube_id]
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]

    # YOLOv8
    yolo_model = None
    if HAS_YOLO:
        print("[*] Loading YOLOv8 model...")
        yolo_model = YOLO("yolov8n.pt")

    # Video logger
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('df_vision_log.mp4', fourcc, 20.0, (640, 480))

    # IK solve
    print("[*] Solving inverse kinematics...")
    PICK_J = solve_ik(model, data, site_id, [0.18, 0.0, 0.07], 0.0)
    LIFT_J = solve_ik(model, data, site_id, [0.18, 0.0, 0.18], 0.0)
    PLACE_J = solve_ik(model, data, site_id, [0.18, -0.12, 0.07], 0.8)
    PLACE_LIFT_J = solve_ik(model, data, site_id, [0.18, -0.12, 0.18], 0.8)
    PLACE_PAN = PLACE_J[0]

    WAYPOINTS = [
        ("NEUTRAL",      [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
         False, 750, 1500, None, "Initializing at neutral..."),
        ("PRE_APPROACH",  [0.0, -0.841, 0.756, 1.657, 0.0, 1.5],
         False, 750, 2500, None, "Pre-approach..."),
        ("GRAB_ADVANCE", PICK_J + [1.5],
         False, 1000, 2500, None, "Advancing gripper to cube..."),
        ("CLOSE_GRIP",   PICK_J + [-0.10],
         True, 1000, 2000, None, ">>> GRASPING CUBE <<<"),
        ("LIFT",         LIFT_J + [-0.10],
         True, 750, 2500, None, "Lifting cube..."),
        ("PAN_TO_PLACE", [PLACE_PAN] + LIFT_J[1:] + [-0.10],
         True, 1500, 2000, (0.0, PLACE_PAN, 1500), "Panning to placement zone..."),
        ("PLACE_LIFT",   PLACE_LIFT_J + [-0.10],
         True, 1000, 2500, None, "Aligning over target..."),
        ("PLACE_LOWER",  PLACE_J + [-0.10],
         True, 1000, 2500, None, "Lowering cube to target..."),
        ("RELEASE",      PLACE_J + [1.5],
         False, 750, 1500, None, ">>> RELEASING CUBE <<<"),
        ("PAN_BACK",     [0.0] + LIFT_J[1:] + [1.5],
         False, 1500, 2000, (PLACE_PAN, 0.0, 1500), "Returning arm..."),
        ("RETURN",       [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
         False, 750, 1500, None, "Back to neutral. DONE!"),
    ]

    waypoint_idx = 0
    step_in_wp = 0
    dataset_buffer = {"images": [], "qpos": [], "ctrl": []}

    data.ctrl[:6] = WAYPOINTS[0][1]
    mujoco.mj_resetData(model, data)

    print("[*] ═══════════════════════════════════════════════════")
    print("[*]  SO-101 AUTONOMOUS PICK & PLACE — DEFINITIVE v5")
    print("[*] ═══════════════════════════════════════════════════")
    print(f"[→] Phase 1/{len(WAYPOINTS)}: {WAYPOINTS[0][6]}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        while viewer.is_running():
            step_start = time.time()

            # ─── Camera + Vision ───
            if step_count % 25 == 0:
                renderer.update_scene(data, camera="rgbd_cam")
                img = renderer.render()
                cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if yolo_model is not None and waypoint_idx <= 3:
                    results = yolo_model(cv2_img, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(cv2_img, f"YOLOv8 ({box.conf[0]:.2f})",
                                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                df = densefusion_6d_inference(model, data)
                cv2.putText(cv2_img,
                            f"DF6D: [{df['pos'][0]:.3f}, {df['pos'][1]:.3f}, {df['pos'][2]:.3f}]",
                            (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

                gpos = data.site_xpos[site_id]
                cv2.putText(cv2_img,
                            f"Grip: [{gpos[0]:.3f}, {gpos[1]:.3f}, {gpos[2]:.3f}]",
                            (10, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

                if waypoint_idx < len(WAYPOINTS):
                    phase_name = WAYPOINTS[waypoint_idx][0]
                    cv2.putText(cv2_img, f"Phase: {phase_name} ({waypoint_idx+1}/{len(WAYPOINTS)})",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                video_writer.write(cv2_img)

                if waypoint_idx > 0:
                    dataset_buffer["images"].append(img)
                    dataset_buffer["qpos"].append(np.copy(data.qpos[:6]))
                    dataset_buffer["ctrl"].append(np.copy(data.ctrl[:6]))

            # ─── Waypoint Execution ───
            if waypoint_idx < len(WAYPOINTS):
                name, ctrl, hold_cube, min_steps, max_steps, pan_interp, msg = WAYPOINTS[waypoint_idx]
                target = np.array(ctrl)

                data.ctrl[:6] = ctrl

                # Smooth pan interpolation
                if pan_interp is not None:
                    s_pan, e_pan, n_steps = pan_interp
                    t = min(step_in_wp / n_steps, 1.0)
                    t = t * t * (3.0 - 2.0 * t)
                    data.qpos[0] = s_pan + t * (e_pan - s_pan)
                    data.qvel[0] = 0.0
                    data.ctrl[0] = data.qpos[0]

                # Magnetic grasp
                if hold_cube:
                    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = data.site_xpos[site_id]

                step_in_wp += 1

                # Convergence check
                if pan_interp is not None:
                    max_err = np.max(np.abs(data.qpos[1:6] - target[1:]))
                else:
                    max_err = np.max(np.abs(data.qpos[:6] - target))

                if (max_err < 0.03 and step_in_wp >= min_steps) or step_in_wp >= max_steps:
                    waypoint_idx += 1
                    step_in_wp = 0
                    if waypoint_idx < len(WAYPOINTS):
                        print(f"[→] Phase {waypoint_idx+1}/{len(WAYPOINTS)}: {WAYPOINTS[waypoint_idx][6]}")
                    else:
                        fc = data.xpos[cube_id]
                        err = np.linalg.norm(fc[:2] - CUBE_PLACE_POS[:2])
                        print(f"[✓] ══════ EPISODE COMPLETE ══════")
                        print(f"    Final cube: [{fc[0]:.4f}, {fc[1]:.4f}, {fc[2]:.4f}]")
                        print(f"    Placement error: {err*1000:.1f} mm")

            else:
                # Save dataset and continue viewing
                if len(dataset_buffer["images"]) > 0:
                    filepath = os.path.join(dataset_dir, "episode_0.npz")
                    np.savez(filepath,
                             images=np.array(dataset_buffer["images"]),
                             qpos=np.array(dataset_buffer["qpos"]),
                             ctrl=np.array(dataset_buffer["ctrl"]))
                    print(f"[*] Dataset saved to {filepath}")
                    dataset_buffer = {"images": [], "qpos": [], "ctrl": []}  # clear
                    break

            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    video_writer.release()
    print("[*] Video saved to df_vision_log.mp4")


if __name__ == "__main__":
    model, data = create_simulation()
    run_simulation(model, data)
