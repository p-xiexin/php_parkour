"""Public CLI, generated-motion editing, and terrain-helper regressions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from motion_matching.io.motion_io import save_edited_motion_npz
from motion_matching.motion_visualizer import load_motion_data
from motion_matching.utils import PKG_ROOT, ROBOT_MOTIONS_DIR
from motion_matching.utils.tracking_format import qpos_to_tracking


def _qpos(n=60):
    qpos = np.zeros((n, 36), dtype=np.float32)
    qpos[:, 0] = 1
    qpos[:, 4] = np.linspace(0, 1, n)
    qpos[:, 6] = 0.8
    qpos[:, 7:] = np.linspace(0, 0.2, 29)
    return qpos


def test_viewer_round_trip_preserves_joint_order_and_commands(tmp_path):
    qpos = _qpos()
    commands = np.eye(14, dtype=np.float32)[np.arange(len(qpos)) % 14]
    source = tmp_path / "source_motion.npz"
    np.savez(source, **qpos_to_tracking(qpos, 30), vel_cmd=commands)
    _, _, loaded, fps, loaded_commands = load_motion_data(source)
    np.testing.assert_allclose(loaded, qpos, atol=1e-7)
    np.testing.assert_array_equal(loaded_commands, commands)
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(loaded, output, source_fps=fps, target_fps=fps, vel_cmd=loaded_commands)
    with np.load(output) as saved:
        np.testing.assert_array_equal(saved["vel_cmd"], commands)
        np.testing.assert_allclose(saved["joint_pos"], qpos_to_tracking(qpos, 30)["joint_pos"], atol=1e-7)


def test_speed_and_fps_edits_preserve_discrete_command_timing(tmp_path):
    qpos = _qpos()
    commands = np.eye(14, dtype=np.float32)[np.arange(len(qpos)) // 5]
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(qpos, output, source_fps=30, target_fps=50, speed=1.5, vel_cmd=commands)
    speed_indices = np.floor(np.linspace(0, 59, 40)).astype(int)
    fps_indices = np.floor(np.linspace(0, 39, 66)).astype(int)
    with np.load(output) as saved:
        assert saved["fps"].item() == 50
        assert saved["joint_pos"].shape[0] == 66
        assert saved["vel_cmd"].dtype == np.float32
        np.testing.assert_array_equal(saved["vel_cmd"], commands[speed_indices[fps_indices]])
        np.testing.assert_array_equal(saved["vel_cmd"].sum(axis=1), np.ones(66))


def test_editing_motion_without_commands_keeps_them_optional(tmp_path):
    output = tmp_path / "edited.npz"
    save_edited_motion_npz(_qpos(), output, source_fps=30, target_fps=30)
    with np.load(output) as saved:
        assert "vel_cmd" not in saved


def test_viewer_rejects_source_clip_with_actionable_error():
    with pytest.raises(ValueError, match="generated tracking motion"):
        load_motion_data(ROBOT_MOTIONS_DIR / "climb_76_high_speed_up.npz")


def test_viewer_rejects_misaligned_commands(tmp_path):
    source = tmp_path / "bad_motion.npz"
    np.savez(source, **qpos_to_tracking(_qpos(), 30), vel_cmd=np.zeros((2, 14)))
    with pytest.raises(ValueError, match="vel_cmd.*matching joint_pos"):
        load_motion_data(source)


def _run_python(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(PKG_ROOT.parent)}
    return subprocess.run([sys.executable, *args], env=env, cwd=cwd, capture_output=True, text=True, check=False)


def test_headless_entry_points_do_not_import_gui_packages(tmp_path):
    result = _run_python(["-c", """
import importlib.abc
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tkinter', '_tkinter', 'viser', 'matplotlib', 'robot_descriptions'}:
            raise AssertionError('Headless import requested GUI package: ' + fullname)
sys.meta_path.insert(0, NoGui())
import motion_matching.run
import motion_matching.generate_database
"""], tmp_path)
    assert result.returncode == 0, result.stderr


def test_viewer_requires_explicit_motion_file(tmp_path):
    result = _run_python(["-m", "motion_matching.motion_visualizer"], tmp_path)
    assert result.returncode != 0
    assert "--motion-file" in result.stdout + result.stderr
    assert "Viser server started" not in result.stdout


def test_swap_xy_helper_swaps_geometry_without_gui_imports(tmp_path):
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    transform = trimesh.transformations.rotation_matrix(0.3, [0, 0, 1])
    transform[:3, 3] = [4.0, 2.0, 0.25]
    mesh.apply_transform(transform)
    source = tmp_path / "box.obj"
    mesh.export(source)
    result = _run_python([
        "-m", "motion_matching.utils.data_utils", "--obj_file", str(source), "--swap_xy"
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(source.with_name("box_xy_swapped.json").read_text())
    np.testing.assert_allclose(data["pos"], [2.0, 4.0, 0.25], atol=1e-7)
    rotation = trimesh.transformations.quaternion_matrix(data["quat"])[:3, :3]
    local_vertices = (mesh.vertices[:, [1, 0, 2]] - data["pos"]) @ rotation
    np.testing.assert_allclose(np.abs(local_vertices), np.broadcast_to(np.asarray(data["size"]) / 2, (8, 3)), atol=1e-7)


def test_contact_face_binding_uses_overlapping_support():
    from motion_matching.postprocessing.contacts import _bind, _surfaces

    from retargeting.retarget_contacts import ContactPhase

    phase = ContactPhase(
        body=1,
        local_point=np.zeros(3),
        start=0,
        stop=4,
        anchor=np.array([0.0, 0.0, 1.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        plane_height=1.0,
        tangent_axes=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        tangent_lower=np.array([0.0, -0.2]),
        tangent_upper=np.array([0.6, 0.2]),
    )
    near = dict(pos=[-0.07, 0.0, 0.5], quat=[1.0, 0.0, 0.0, 0.0], size=[0.1, 0.4, 1.0])
    support = dict(pos=[0.3, 0.0, 0.5], quat=[1.0, 0.0, 0.0, 0.0], size=[0.2, 0.4, 1.0])
    # The closer box is outside the source patch. Adding it cannot invalidate
    # the farther box whose top face intersects the patch.
    for boxes in ([support], [near, support], [support, near]):
        _, bound = _bind(phase, _surfaces(boxes))
        np.testing.assert_allclose(bound.tangent_lower, [0.2, -0.2], atol=1e-12)
        np.testing.assert_allclose(bound.tangent_upper, [0.4, 0.2], atol=1e-12)
        np.testing.assert_array_equal(bound.anchor, phase.anchor)
    with pytest.raises(ValueError, match="overlap"):
        _bind(phase, _surfaces([near]))


def test_t2_retarget_contacts_to_training_npz(tmp_path):
    """Use real consolidated retarget results through contact export and solving."""
    source = os.environ.get("T2_BANK_DIR")
    if not source:
        pytest.skip("Set T2_BANK_DIR to a retargeted bank containing jump_35")
    import mujoco as mj
    from motion_matching.core.database import MotionMatchingDatabase
    from motion_matching.io import save_motion_npz
    from motion_matching.postprocessing.contacts import (
        BankContactSource,
        export_contact_bundle,
    )
    from motion_matching.postprocessing.solver import generate_transition
    from motion_matching.postprocessing.validation import MotionConstraintVerifier
    from motion_matching.robots.t2 import make_t2_spec

    bank = Path(source)
    work = bank.with_name(bank.name + "_work")
    spec = make_t2_spec(PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml")
    bundle = tmp_path / "contacts"
    export_contact_bundle(bank, work, spec, bundle)
    name = "database_g1_jump_35.bin"
    db = MotionMatchingDatabase()
    db.load_from_file(str(bank / name))
    start, stop = map(int, (db.range_starts[0], db.range_stops[0]))
    assert stop - start == 34
    records = [dict(db_key="jump_35", source_frame=frame,
                    transform_quat=[1, 0, 0, 0], transform_pos=[0, 0, 0])
               for frame in range(start, stop)]
    scene = mj.MjModel.from_xml_path(str(work / Path(name).stem / "range_00/scene.xml"))
    box = scene.geom("scene_box")
    boxes = [dict(pos=box.pos.tolist(), quat=box.quat.tolist(), size=(2*box.size).tolist())]
    contacts = BankContactSource(bank, bundle, spec).build(records, boxes)
    expected = spec.decode_bones_to_qpos(db.bone_positions[start:stop], db.bone_rotations[start:stop])
    np.testing.assert_allclose(contacts.target_qpos[:, 4:], expected[:, 4:], atol=1e-12)
    saved_phases = json.loads((work / Path(name).stem / "range_00/phases.json").read_text())
    with np.load(work / Path(name).stem / "range_00/trajectory.npz") as saved:
        payload = json.loads((bundle / "database_g1_jump_35.contacts.json").read_text())
        exported = payload["ranges"][0]["phases"]
        assert len(exported) == len(saved_phases)
        np.testing.assert_array_equal([p["anchor"] for p in exported], saved["anchors"])
    target = contacts.target_qpos
    fixed = {frame: target[frame].copy() for frame in (0, 1, len(target)-2, len(target)-1)}
    qpos = generate_transition(spec, boxes, target, phases=contacts.phases,
                               fixed_poses=fixed, fixed_anchors=contacts.fixed_anchors,
                               iterations=80)
    output = tmp_path / "jump_motion.npz"
    save_motion_npz(qpos, output, target_fps=60, source_fps=60, robot_spec=spec)
    verifier = MotionConstraintVerifier(spec, boxes)
    assert not verifier.check_npz(output)
    # Removing redundant input checks must not weaken physical acceptance.
    bad = qpos[:2].copy()
    bad[:, 6] -= 0.1
    assert "collision_penetration" in verifier.check(bad, 60)
    bad = qpos[:2].copy()
    bad[0, 7] = spec.model.jnt_range[spec.joint_ids[0], 1] + 0.01
    assert "hard_joint_limits" in verifier.check(bad, 60)
    bad = qpos[:2].copy()
    bad[0, 7 + verifier.wrist_columns[0]] = 0.01
    assert "locked_wrist_angle" in verifier.check(bad, 60)
    for frame, pose in fixed.items():
        np.testing.assert_allclose(qpos[frame], pose, atol=1e-12)

    with np.load(output) as saved:
        arrays = {key: saved[key] for key in saved.files}
    foot = spec.body_names.index("left_ankle_pitch_link")
    edited = tmp_path / "edited_motion.npz"
    positions = arrays["body_pos_w"].copy()
    positions[:, foot, 2] = -1.0
    np.savez(edited, **{**arrays, "body_pos_w": positions})
    assert "body_position_mismatch" in verifier.check_npz(edited)

    rotations = arrays["body_quat_w"].copy()
    rotations[0, foot] = [0.0, 1.0, 0.0, 0.0]
    np.savez(edited, **{**arrays, "body_quat_w": rotations})
    assert "body_rotation_mismatch" in verifier.check_npz(edited)
    # q and -q encode the same rotation, including at the floating root.
    np.savez(edited, **{**arrays, "body_quat_w": -arrays["body_quat_w"]})
    assert not verifier.check_npz(edited)
