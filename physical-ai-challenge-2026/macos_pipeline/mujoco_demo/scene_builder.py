"""Helpers to build a MuJoCo model that includes the SO101 robot plus
an interactable cube and a target boundary area.

Provides `build_model_with_cube_and_target(cube_pose, target_pose, cube_size, target_size)`
which returns (model, sim, attach_info) where `attach_info` contains the
indices and metadata needed to attach the cube to the gripper during the demo.
"""
from pathlib import Path
import numpy as np
import os

try:
    import mujoco
except Exception:
    mujoco = None

HERE = Path(__file__).resolve().parent
SO101_XML = HERE.parent / 'third_party' / 'SO-ARM100' / 'Simulation' / 'SO101' / 'so101_new_calib.xml'


def build_model_with_cube_and_target(cube_pose=(0.25, 0.0, 0.02), target_pose=(0.0, 0.25, 0.02),
                                     cube_size=(0.02, 0.02, 0.02), target_size=(0.08, 0.08, 0.002)):
    if mujoco is None:
        raise RuntimeError('mujoco python package is required')

    if not SO101_XML.exists():
        raise FileNotFoundError(f'SO101 XML not found: {SO101_XML}')

    so_text = SO101_XML.read_text()

    cx, cy, cz = cube_pose
    sx, sy, sz = cube_size
    tx, ty, tz = target_pose
    bx, by, bth = target_size

    cube_xml = f'''
    <body name="cube_gt" pos="{cx} {cy} {cz}">
      <joint name="cube_free" type="free"/>
      <geom name="cube_geom" type="box" size="{sx} {sy} {sz}" rgba="1 0 0 1" mass="0.1"/>
      <site name="cube_site" pos="0 0 0" size="{max(sx,sy,sz)}" rgba="1 0 0 1"/>
    </body>
    '''

    target_xml = f'''
    <body name="target_area" pos="{tx} {ty} {tz}">
      <geom name="target_geom" type="box" size="{bx} {by} {bth}" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
    '''

    # Insert additional bodies before the closing </worldbody> tag
    idx = so_text.rfind('</worldbody>')
    if idx == -1:
        raise RuntimeError('Could not find </worldbody> in SO101 XML')

    new_xml = so_text[:idx] + cube_xml + target_xml + so_text[idx:]

    # Ensure any 'assets/...' references resolve to the absolute assets directory
    assets_dir = SO101_XML.parent / 'assets'
    if assets_dir.exists():
        new_xml = new_xml.replace('assets/', assets_dir.as_posix() + '/')

    # Some included files and mesh references are relative to the SO101
    # directory. Temporarily change cwd so MuJoCo resolves those relative
    # paths correctly when loading from the XML string.
    prev_cwd = Path.cwd()
    try:
        os.chdir(SO101_XML.parent)
        model = mujoco.MjModel.from_xml_string(new_xml)
    finally:
        os.chdir(prev_cwd)
    data = mujoco.MjData(model)
    # forward to compute positions
    try:
        mujoco.mj_forward(model, data)
    except Exception:
        # some bindings may not expose mj_forward; ignore if unavailable
        pass

    # locate cube joint and qpos address
    cube_jidx = None
    num_joints = model.njnt if hasattr(model, 'njnt') else model.nj
    for j in range(num_joints):
        try:
            if model.joint(j).name == 'cube_free':
                cube_jidx = j
                break
        except Exception:
            continue
    if cube_jidx is None:
        raise RuntimeError('cube_free joint not found in built model')

    qposadr = int(model.jnt_qposadr[cube_jidx])
    # compute qpos dimension for this joint
    qpos_addrs = [int(model.jnt_qposadr[j]) for j in range(num_joints) if int(model.jnt_qposadr[j]) >= 0]
    next_addrs = [a for a in qpos_addrs if a > qposadr]
    if next_addrs:
        qpos_dim = min(next_addrs) - qposadr
    else:
        qpos_dim = int(model.nq) - qposadr

    # find site ids
    cube_site_id = None
    gripper_site_id = None
    num_sites = model.nsite if hasattr(model, 'nsite') else 0
    for s in range(num_sites):
        try:
            nm = model.site(s).name
        except Exception:
            nm = None
        if nm == 'cube_site':
            cube_site_id = s
        if nm == 'gripperframe':
            gripper_site_id = s

    # detect whether position components are at start or end of qpos slice
    qpos_seg = np.array(data.qpos[qposadr:qposadr+qpos_dim])
    pos_first = False
    if cube_site_id is not None:
        cube_pos = np.array(data.site_xpos[cube_site_id])
        if qpos_dim >= 3 and np.allclose(qpos_seg[:3], cube_pos, atol=1e-6):
            pos_first = True
        elif qpos_dim >= 3 and np.allclose(qpos_seg[-3:], cube_pos, atol=1e-6):
            pos_first = False
        else:
            # fallback: assume pos is first
            pos_first = True

    attach_info = {
        'cube_qposadr': int(qposadr),
        'cube_qpos_dim': int(qpos_dim),
        'pos_first': bool(pos_first),
        'cube_site_id': cube_site_id,
        'gripper_site_id': gripper_site_id,
        'attached': False,
    }

    return model, data, attach_info
