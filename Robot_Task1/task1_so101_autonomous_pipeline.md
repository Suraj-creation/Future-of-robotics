# Adapted Pipeline for SO101 Task 1: Autonomous Cube Pick-and-Place

## 1. Executive Summary

For Physical AI Hackathon 2026 Task 1, the `LeRobot SO101` must autonomously detect a graspable object, pick it, move it to a target region, and place it with tight positional tolerance. Based on the provided repo, the strongest path is:

- `MuJoCo` for fast autonomy prototyping and perception testing.
- `MoveIt + Gazebo` for constrained motion planning and collision-aware execution.
- `RGB-D` perception from the simulated `d435i` camera.
- A task-level autonomy node that detects the object and target, selects a safe grasp direction, calls the existing pick/place motion stack, and verifies final placement.

This is better than a fully open-loop script because the repo already contains:

- camera topics from the MuJoCo bridge,
- a MoveIt server with `/pick_front`, `/pick_left`, `/pick_right`, `/pick_rear`, and `/place_object`,
- world files with graspable objects,
- an example autonomy structure in `task3_autonomy.py`.

## 2. What The Challenge Actually Requires

From `robotics_challenge.pdf`, Task 1 requires:

1. Detect the object using onboard sensors.
2. Grasp it securely.
3. Move it to a specified target location.
4. Place it precisely and release.

Important constraint:

- Semifinal is simulation-only.
- Final round uses the physical `LeRobot SO101`.

So the pipeline should be sim-first, but designed so the same perception and planning logic can transfer to the real arm.

## 3. Repo Assets We Should Reuse

Relevant assets already present in the repo:

- `workshop/dev/docker/workspace/src/so101_mujoco/scripts/so101_mujoco_bridge.py`
  - publishes `/joint_states`, `/d435i/image`, `/d435i/depth_image`, `/d435i/camera_info`, `/d435i/points`
- `workshop/dev/docker/workspace/src/so101_mujoco/scripts/task3_autonomy.py`
  - example of camera-based detection, depth projection, TF conversion, IK fallback, and task sequencing
- `workshop/dev/docker/workspace/src/so101_unified_bringup/src/moveit_server.cpp`
  - exposes `/create_traj`, `/move_to_joint_states`, `/rotate_effector`, `/pick_object`, `/pick_front`, `/pick_left`, `/pick_right`, `/pick_rear`, `/place_object`
- `workshop/dev/docker/workspace/src/so101_unified_bringup/worlds/empty_world.sdf`
  - already contains a graspable `red_box`
- `workshop/dev/docker/workspace/src/so101_mujoco/mujoco/scene.xml`
  - already contains a box and a cylinder for MuJoCo testing

Conclusion:

- We do not need to build the autonomy stack from zero.
- We should build `task1_autonomy.py` by adapting the structure of `task3_autonomy.py`.

## 4. Proposed System Architecture

### Perception Layer

Inputs:

- `/d435i/image`
- `/d435i/depth_image`
- `/d435i/camera_info`
- TF from camera frame to `base_link`

Outputs:

- cube center in `base_link`
- target square center in `base_link`
- estimated grasp approach direction

Recommended method:

1. Detect the cube in RGB.
2. Detect the outer pick zone and inner place zone.
3. Use depth to recover 3D points.
4. Transform all detections into `base_link`.
5. Validate that the estimated points lie inside the reachable workspace.

Best practical choice for Task 1:

- For hackathon speed, start with color/shape segmentation and depth projection.
- Keep a fallback prior if detection is temporarily missing.
- Add a heavier detector only if the scene becomes visually complex.

Why this is the right adaptation:

- The repo already demonstrates RGB + depth + TF conversion in `task3_autonomy.py`.
- Task 1 geometry is simple, so classical vision is likely faster and easier to debug than training a full detector.

### Decision and Policy Layer

The autonomy node should:

1. Move to a scan pose.
2. Detect cube and placement square.
3. Choose pick direction based on cube position relative to the robot.
4. Select an approach waypoint and a grasp waypoint.
5. Call the matching pick service.
6. Move above the placement square.
7. Call the placement service.
8. Re-observe and verify success.

Pick policy:

- If object is left of center, use `/pick_left`.
- If object is right of center, use `/pick_right`.
- If object is centered and reachable from front, use `/pick_front`.
- Use `/pick_rear` only if object lies in a geometry that makes front access unsafe.

Placement policy:

- Use a hover pose above the inner square.
- Lower slowly with orientation constraint enabled.
- Release only after positional error is below tolerance.
- Retreat vertically before any lateral motion.

### Motion Planning Layer

Use the existing MoveIt server for:

- collision-aware planning,
- joint-space fallback,
- orientation-constrained pose moves,
- safe retracts and pre-grasp motion.

This is preferable to direct open-loop teleop-style control because `moveit_server.cpp` already includes:

- repeated planning attempts,
- fallback from constrained pose planning,
- trajectory execution hooks,
- predefined directional pick behaviors.

### Control Layer

For this repo, the practical control split is:

- high-level control: autonomy node,
- motion generation: MoveIt services,
- low-level execution: trajectory controllers from Gazebo or MuJoCo bridge.

For Task 1, prioritize:

- low velocity scaling,
- smooth vertical lift after grasp,
- no large wrist rotations while carrying the cube,
- place with a slow final descent.

Even if real force sensing is not available in simulation, we should still design the logic as if contact uncertainty exists:

- close gripper,
- pause briefly,
- lift a small amount,
- verify object moved with the gripper before committing to transport.

## 5. End-to-End Task 1 Pipeline

### Step A: Initialization

- launch sim
- start camera bridge
- start MoveIt server
- move arm to a scan/home pose
- open gripper

### Step B: Perception

- acquire RGB, depth, camera info
- detect cube centroid
- detect target square centroid
- project both to 3D
- transform to `base_link`

### Step C: Grasp Decision

- classify best approach direction
- compute approach and grasp offsets
- reject poses outside workspace or too close to table edge

### Step D: Pick Execution

- move to pre-grasp
- descend to grasp pose
- close gripper
- lift vertically
- verify object follows

### Step E: Transport

- move to hover pose above target square
- keep cube orientation fixed
- use slower motion than the empty-arm move

### Step F: Placement

- descend to place height
- center over inner square
- open gripper
- retreat upward

### Step G: Verification and Recovery

- re-detect cube
- check if cube center lies inside target square tolerance
- if not, attempt one corrective re-grasp or micro-place adjustment

## 6. Recommended Implementation for This Repo

Create one new node:

- `workshop/dev/docker/workspace/src/so101_mujoco/scripts/task1_autonomy.py`

Use `task3_autonomy.py` as the template, but replace bottle/cup logic with:

- cube detection,
- place-zone detection,
- pick direction selection,
- pick/place service calls,
- placement verification.

Recommended node responsibilities:

- subscribe to camera topics,
- use TF for camera-to-base projection,
- choose scan pose,
- call `/pick_front|left|right|rear`,
- call `/place_object` or `/create_traj`,
- verify task completion.

Recommended world additions:

- add explicit square boundary markers in MuJoCo and Gazebo worlds,
- give the pick square and place square high visual contrast,
- keep the object color distinct from the table.

## 7. Suggested Topic and Service Flow

Perception inputs:

- `/d435i/image`
- `/d435i/depth_image`
- `/d435i/camera_info`

Planning and execution services:

- `/pick_front`
- `/pick_left`
- `/pick_right`
- `/pick_rear`
- `/place_object`
- `/create_traj`
- `/move_to_joint_states`

Verification signals:

- `/joint_states`
- camera re-observation after place

## 8. Practical Detection Strategy

For the cube:

- threshold by color if object color is controlled,
- optionally verify shape by contour area and rectangularity,
- use median depth around the centroid.

For the square boundaries:

- detect the square outline using edges or HSV thresholding,
- estimate the center from contour geometry,
- for the smaller place square, compute center and side length to define acceptance tolerance.

Recommended tolerance check:

- object center must lie inside the target square,
- object yaw does not matter unless the rulebook explicitly requires orientation alignment,
- height after release should match table height.

## 9. Failure Points and Recovery

### Likely failures

- depth noise on object edges
- false contour detection from table textures
- grasp from the wrong side causing collision
- cube slipping after pickup
- release slightly outside the small square

### Mitigations

- use median depth over a local pixel patch
- enforce workspace bounds before planning
- choose pick side from object lateral offset
- lift-and-check after grasp
- use hover-place-retreat structure
- allow one retry with updated perception

## 10. What Not To Do

- Do not rely on a single hard-coded object pose.
- Do not place using one-shot open-loop motion without visual recheck.
- Do not transport the cube with aggressive velocity scaling.
- Do not mix direct joint teleop and MoveIt planning in the same phase unless clearly synchronized.
- Do not overcomplicate perception with large models before basic color/depth detection works.

## 11. Trade-Off Summary

### Best balance for hackathon speed

- perception: classical RGB-D segmentation first
- planning: MoveIt service stack already in repo
- control: slow, stable pick/place rather than maximum speed
- recovery: one retry, not full replanning loops forever

### Why this balance works

- fastest to implement,
- uses existing repo capabilities,
- debuggable in simulation,
- transferable to the final physical SO101 setup.

## 12. Recommended Build Order

1. Confirm the cube can be detected from a scan pose in MuJoCo.
2. Add visible square pick/place boundaries to the scene.
3. Implement 2D-to-3D projection for cube and target center.
4. Build `task1_autonomy.py` from `task3_autonomy.py`.
5. Connect service calls to `/pick_*` and `/place_object`.
6. Add placement verification.
7. Tune offsets, speeds, and retry thresholds.
8. Port the same logic to the Gazebo + MoveIt launch if needed for final evaluation.

## 13. Decision Summary Sheet

| Stage | Key Decision | Rationale |
|---|---|---|
| Perception | Use RGB-D + contour-based detection first | Fastest reliable path for simple cube/square geometry |
| Coordinate Mapping | Project detections to 3D and transform to `base_link` | Required for robot-referenced planning |
| Pick Strategy | Use directional pick services already in repo | Reuses tested motion primitives |
| Motion Planning | Use MoveIt server for grasp and place | Better than open-loop joint scripting |
| Control | Slow descent, vertical lift, stable transport | Reduces slip and placement error |
| Verification | Re-detect after grasp and after place | Prevents silent failure |
| Simulation | Prototype in MuJoCo, validate in Gazebo/MoveIt | Fast iteration plus stronger collision-aware check |
| Implementation | Create `task1_autonomy.py` by adapting `task3_autonomy.py` | Lowest engineering risk |

## 14. Final Recommendation

For your team, the best Task 1 solution is not a generic AI pipeline. It is:

- a `task1_autonomy.py` node,
- using the repo’s simulated camera topics,
- detecting the cube and place square with RGB-D,
- selecting one of the existing directional pick services,
- placing via the MoveIt server,
- and verifying success with a final camera check.

That gives you an autonomous SO101 pipeline that is realistic, hackathon-friendly, and aligned with the codebase you already have.
