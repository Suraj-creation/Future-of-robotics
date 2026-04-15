#!/usr/bin/env python3
"""ACT policy stub (Action Chunking Transformer) — simplified.

This file provides `plan_pick_place(cube_pose, target_pose)` which returns a
small list of action chunks. Each chunk is a dict with:
- 'name': human label
- 'ctrls': mapping actuator_name -> target_value (radians or opening)
- optional 'attach' / 'release' booleans to indicate gripper events

Replace with the real ACT inference model once trained and exported.
"""
from typing import Tuple, List, Dict


def plan_pick_place(cube_pose: Tuple[float,float,float], target_pose: Tuple[float,float,float]) -> List[Dict]:
    # actuator names expected by the runner
    # order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    home = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 1.0,  # open
    }

    approach = {
        'shoulder_pan': -0.4,
        'shoulder_lift': 0.8,
        'elbow_flex': -1.0,
        'wrist_flex': 0.3,
        'wrist_roll': 0.0,
        'gripper': 1.0,
    }

    lower = {k: v for k, v in approach.items()}
    # small adjustment to lower
    lower['shoulder_lift'] = 1.0
    lower['wrist_flex'] = 0.0

    grasp = {k: v for k, v in lower.items()}
    grasp['gripper'] = 0.0  # close gripper

    lift = {k: v for k, v in grasp.items()}
    lift['shoulder_lift'] = 0.5

    deliver = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.6,
        'elbow_flex': -0.7,
        'wrist_flex': 0.1,
        'wrist_roll': 0.0,
        'gripper': 0.0,
    }

    release = {k: v for k, v in deliver.items()}
    release['gripper'] = 1.0

    # Compose action chunks (coarse actions / chunks)
    chunks = [
        {'name': 'move_home', 'ctrls': home},
        {'name': 'approach', 'ctrls': approach},
        {'name': 'lower', 'ctrls': lower},
        {'name': 'grasp', 'ctrls': grasp, 'attach': True},
        {'name': 'lift', 'ctrls': lift},
        {'name': 'move_deliver', 'ctrls': deliver},
        {'name': 'release', 'ctrls': release, 'release': True},
        {'name': 'return_home', 'ctrls': home},
    ]

    return chunks
