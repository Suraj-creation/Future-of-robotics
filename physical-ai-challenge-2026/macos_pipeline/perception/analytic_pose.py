#!/usr/bin/env python3
"""Analytic perception stub for cube-only scenarios.

Returns a fixed start and goal pose for a cube in the workspace. The demo
uses these poses as if a YOLOv8 + DenseFusion pipeline produced them.
"""
from typing import Tuple

def cube_start_and_target() -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """Return (cube_pose, target_pose), each as (x,y,z) in meters.

These are simple fixed coordinates relative to the robot base used for the
local demo. Replace with real perception code (YOLO + DenseFusion) later.
"""
    # cube start position (in front of the robot)
    cube_pose = (0.25, 0.00, 0.02)
    # target place position (to the right)
    target_pose = (0.0, 0.25, 0.02)
    return cube_pose, target_pose
