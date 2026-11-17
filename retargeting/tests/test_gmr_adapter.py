"""Check the shared trajectory solver and native T2 contact geometry."""

import numpy as np
import pytest
from motion_matching.robots.t2 import make_t2_spec


def test_contact_mapping_preserves_sole_patch_and_uses_exposed_toe():
    """A T2 toe protrudes beyond its two front sole landmarks on a wall."""
    from motion_matching.utils import PKG_ROOT

    from retargeting.retarget_contacts import exposed_contact

    model = make_t2_spec(
        PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml"
    ).model
    body = model.body("left_ankle_roll_link").id
    preferred = np.array([0.12, 0.03, -0.03])
    point, radius = exposed_contact(
        model, body, preferred, np.array([0.0, 0.0, 1.0]), np.eye(3)
    )
    np.testing.assert_allclose(point, preferred, atol=1e-8)
    assert radius == pytest.approx(0.005)
    point, radius = exposed_contact(
        model, body, preferred, np.array([-1.0, 0.0, 0.0]), np.eye(3)
    )
    # A .12 m sole landmark would bury the .15 m toe in the wall.
    assert point[0] + radius >= 0.155 - 1e-6


def test_nearby_boxes_use_true_distance_and_preserve_contact_manifold():
    import mujoco as mj
    import numpy as np

    from retargeting.trajectory_optimization import TrajectorySQP

    # Two separated T2 shank boxes: inflated native box-box contacts report
    # spurious negative depths on MuJoCo 3.8.1. True signed distance is positive.
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0.18078304186714542 -1.2139551875018018 0.24220844596858826"
            quat="0.528049247761905 -0.5010805512296506
                  0.3080205158446261 0.6125403129070673">
        <freejoint/><geom type="box" size="0.0525 0.045 0.085" mass="1"/>
      </body>
      <body pos="0.25136723601552846 -0.9980629419620833 0.18681602516953186"
            quat="0.5376232549775548 -0.20503924998715545
                  0.16750572229339666 0.8005385528948692">
        <freejoint/><geom type="box" size="0.0525 0.045 0.085" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    solver = TrajectorySQP.__new__(TrajectorySQP)
    solver.model, solver.data = model, mj.MjData(model)
    mj.mj_forward(model, solver.data)
    witness = np.zeros(6)
    distance = mj.mj_geomDistance(model, solver.data, 0, 1, 0.1, witness)
    assert distance > 0.05
    assert solver.collisions(nearby=True) == []
    np.testing.assert_array_equal(model.geom_margin, 0.0)
    assert solver.data.ncon == 0

    normal = (witness[3:] - witness[:3]) / distance
    solver.data.qpos[7:10] -= (distance - 0.01) * normal
    mj.mj_forward(model, solver.data)
    near = solver.collisions(nearby=True)
    assert len(near) == 1
    assert abs(near[0][4] - 0.01) < 1e-7
    np.testing.assert_allclose(np.linalg.norm(near[0][3]), 1.0, atol=1e-7)
    assert solver.data.ncon == 0

    solver.data.qpos[7:10] -= 0.02 * normal
    mj.mj_forward(model, solver.data)
    actual = solver.collisions()
    assert actual and min(contact[4] for contact in actual) < 0.0
    near = solver.collisions(nearby=True)
    assert len(near) == len(actual)
    for before, after in zip(actual, near):
        assert before[:2] == after[:2]
        np.testing.assert_array_equal(before[2], after[2])
        np.testing.assert_array_equal(before[3], after[3])
        assert before[4] == after[4]
    np.testing.assert_array_equal(model.geom_margin, 0.0)


def test_collision_merit_and_slack_count_do_not_depend_on_manifold_multiplicity():
    import mujoco as mj
    import numpy as np

    from retargeting.trajectory_optimization import TrajectorySQP

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="body" pos="0 0 .09"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    qpos = np.tile(model.qpos0, (3, 1))
    solver = TrajectorySQP(model, 60, [1], qpos[:, None, :3], qpos[:, None, 3:7], [])
    solver.forward(qpos[0])
    contact = solver.collisions()[0]
    anchors = np.empty((0, 3))
    solver.collisions = lambda nearby=False: [contact]
    one = solver.measure(qpos, anchors)
    P1, c1, *_ = solver.linearize(qpos, anchors, 0.1)
    solver.collisions = lambda nearby=False: [contact, contact]
    repeated = solver.measure(qpos, anchors)
    P2, c2, *_ = solver.linearize(qpos, anchors, 0.1)
    assert repeated["merit"] == one["merit"]
    assert repeated["penetration"] == one["penetration"]
    assert P1.shape == P2.shape == (solver.base_size + 3, solver.base_size + 3)
    np.testing.assert_array_equal(c1, c2)
    assert (P1 != P2).nnz == 0


def test_joint_contact_states_match_independent_exhaustive_objective():
    """Compare whole paths, including one label observation per rigid foot."""
    import itertools

    from retargeting.retarget_contacts import joint_contact_states

    rng = np.random.default_rng(143)
    cases = [(rng.uniform(-0.4, 3.7, (4, 2, 3)), [True, False, True, True])]
    # A positive foot label must not reward both probes independently.
    cases.append((np.tile([[2.0, 2.3], [2.0, 2.5]], (3, 1, 1)), [True] * 3))
    # Strong geometry can outweigh one uncertain negative foot annotation.
    cases.append((np.tile([[2.0, 0.5], [2.0, 0.5]], (3, 1, 1)), [False] * 3))
    fps, switch_cost = 60.0, 0.06
    for emissions, labels in cases:
        frames, probes, choices = emissions.shape

        def objective(path):
            total = 0.0
            for frame, modes in enumerate(path):
                total += (
                    sum(emissions[frame, p, mode] for p, mode in enumerate(modes)) / fps
                )
                supported = any(mode != 0 for mode in modes)
                total += (-1.0 if supported == labels[frame] else 1.0) / fps
                if frame:
                    total += switch_cost * sum(
                        a != b for a, b in zip(path[frame - 1], modes)
                    )
            return total

        selected = joint_contact_states(emissions, labels, fps, switch_cost).T
        optimum = min(
            objective(np.array(path).reshape(frames, probes))
            for path in itertools.product(range(choices), repeat=frames * probes)
        )
        assert objective(selected) == pytest.approx(optimum, abs=1e-12)


def test_finite_box_mapping_preserves_forefoot_and_has_real_clearance():
    """A raised heel may cross a wall plane while staying above the finite box."""
    import xml.etree.ElementTree as ET

    import mujoco as mj
    from motion_matching.utils import PKG_ROOT
    from scipy.spatial.transform import Rotation

    from retargeting.retarget_contacts import finite_contact_mapping

    path = PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml"
    tree = ET.parse(path)
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir")).resolve()))
    center = np.array([-0.538207345, 0.075905535, 0.469044945])
    half = np.array([0.387471911302781, 0.25000000181693904, 0.469044945])
    box_quat = [0.612483212802931, 0.0, 0.0, 0.7904835950445774]
    ET.SubElement(
        tree.getroot().find("worldbody"),
        "geom",
        name="scene_box",
        type="box",
        pos=" ".join(map(str, center)),
        size=" ".join(map(str, half)),
        quat=" ".join(map(str, box_quat)),
        contype="1",
        conaffinity="1",
    )
    model = mj.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    model.opt.ccd_tolerance = 1e-9
    body = model.body("left_ankle_roll_link").id
    box_rotation = Rotation.from_quat(box_quat, scalar_first=True).as_matrix()
    normal, axes = -box_rotation[:, 0], box_rotation[:, 1:].T
    lower, upper = axes @ center - half[1:], axes @ center + half[1:]
    rotation = Rotation.from_quat(
        [
            0.7205648121951577,
            0.15550181975161276,
            0.251084031953435,
            -0.6273454745019689,
        ],
        scalar_first=True,
    ).as_matrix()
    anchor = np.array([-0.2956368939606073, -0.2616858827741666, 0.9292233429919463])
    preferred = np.array([0.12, 0.03, -0.03])
    point, radius, selected, _ = finite_contact_mapping(
        model, body, preferred, 0.005, normal, rotation, anchor, axes, lower, upper
    )
    np.testing.assert_allclose(point, preferred, atol=1e-8)
    assert radius == pytest.approx(0.005)
    assert abs(normal @ (selected - anchor)) < 1e-12
    assert np.all(axes @ selected >= lower - 1e-12)
    assert np.all(axes @ selected <= upper + 1e-12)
    assert np.linalg.norm(selected - anchor) < 0.015

    # Independently update the actual body transform and run full MuJoCo FK;
    # do not reuse the selector's manually positioned geom data or its report.
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    parent = int(model.body_parentid[body])
    parent_rotation = data.xmat[parent].reshape(3, 3).copy()
    parent_position = data.xpos[parent].copy()
    placement = selected + radius * normal - rotation @ point
    model.body_pos[body] = parent_rotation.T @ (placement - parent_position)
    model.body_quat[body] = Rotation.from_matrix(parent_rotation.T @ rotation).as_quat(
        scalar_first=True
    )
    mj.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[body], placement, atol=1e-12)
    np.testing.assert_allclose(data.xmat[body].reshape(3, 3), rotation, atol=1e-12)
    foot_geoms = []
    for geom in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom])
        while ancestor not in (0, body):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == body and (
            model.geom_contype[geom] or model.geom_conaffinity[geom]
        ):
            foot_geoms.append(geom)
    terrain = [model.geom("ground").id, model.geom("scene_box").id]
    clearances = [
        float(mj.mj_geomDistance(model, data, a, b, 0.1, np.zeros(6)))
        for a in terrain
        for b in foot_geoms
        if (
            (model.geom_contype[a] & model.geom_conaffinity[b])
            or (model.geom_contype[b] & model.geom_conaffinity[a])
        )
    ]
    assert clearances and min(clearances) >= -1e-6
    assert min(abs(distance) for distance in clearances) <= 1e-6
    for contact in data.contact:
        if (contact.geom1 in terrain and contact.geom2 in foot_geoms) or (
            contact.geom2 in terrain and contact.geom1 in foot_geoms
        ):
            assert contact.dist >= -1e-6


def test_qp_numerical_failure_keeps_only_a_complete_feasible_trajectory(monkeypatch):
    """A QP failure preserves a certified whole iterate, never an infeasible seed."""
    from types import SimpleNamespace

    import clarabel
    import mujoco as mj

    from retargeting.retarget_contacts import ContactPhase
    from retargeting.trajectory_optimization import TrajectorySQP

    calls = []

    class FailedQP:
        def __init__(self, *args, **kwargs):
            calls.append(1)
            self.width = len(args[1])

        def solve(self):
            return SimpleNamespace(
                status="NumericalError", iterations=0, x=np.full(self.width, np.nan)
            )

    monkeypatch.setattr(clarabel, "DefaultSolver", FailedQP)
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="foot" pos=".02 -.01 .1"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    poses = np.tile(model.qpos0, (3, 1))
    poses[:, 0] += [-0.00005, 0.0, 0.00005]
    anchors = np.array([[0.02, -0.01, 0.0]])
    phase = ContactPhase(
        body=1,
        local_point=np.zeros(3),
        start=0,
        stop=3,
        anchor=np.array([0.025, -0.01, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        plane_height=0.0,
        tangent_axes=None,
        tangent_lower=None,
        tangent_upper=None,
        radius=0.1,
        cap_axis_local=None,
    )
    solver = TrajectorySQP(
        model, 60.0, [1], poses[:, None, :3], poses[:, None, 3:7], [phase]
    )
    before = poses.copy()
    result, returned_anchors, history = solver.solve(
        poses, iterations=1, initial_anchors=anchors
    )
    np.testing.assert_array_equal(result, before)
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_array_equal(returned_anchors, anchors)
    assert history[-1]["termination"] == "qp_failure_with_feasible_incumbent"
    assert history[-1]["returned_iteration"] == -1
    assert history[-1]["returned_feasible_incumbent"]
    assert any(row.get("qp", {}).get("attempts") for row in history)
    assert all(
        attempt["status"] == "NumericalError"
        for row in history
        for attempt in row.get("qp", {}).get("attempts", [])
    )
    assert not history[-1]["accepted"]
    assert len(calls) == 2  # The ordinary and equivalent-scale attempts both fail.

    penetrating = before.copy()
    penetrating[:, 2] -= 0.01
    with pytest.raises(RuntimeError, match="certified model step"):
        solver.solve(penetrating, iterations=1, initial_anchors=anchors)
    assert len(calls) == 4


def test_empty_bank_is_rejected_without_creating_output(tmp_path):
    from retargeting.retarget_database import convert_trajectory_bank

    source, target = tmp_path / "empty", tmp_path / "output"
    source.mkdir()
    with pytest.raises(ValueError, match="No G1 databases"):
        convert_trajectory_bank(source, target, None, None)
    assert not target.exists()


def test_unlocked_fist_model_is_rejected(tmp_path):
    from motion_matching.utils import PKG_ROOT

    from retargeting.retarget_database import convert_trajectory_bank

    source, target = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "database_g1_example.bin").touch()
    model = PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_fc.xml"
    with pytest.raises(ValueError, match="all six wrists locked"):
        convert_trajectory_bank(source, target, model, None)
    assert not target.exists()


def test_window_resume_preserves_overlap_and_shared_anchor(tmp_path):
    import mujoco as mj

    from retargeting.retarget_contacts import ContactPhase
    from retargeting.retarget_database import load_window_state, save_window_state
    from retargeting.trajectory_optimization import (
        TrajectorySQP,
        solve_trajectory_windows,
    )

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body><joint name="hinge"/><geom size=".01" mass=".01"
          contype="0" conaffinity="0"/></body>
      </body></worldbody></mujoco>""")
    initial = np.tile(model.qpos0, (7, 1))
    phase = ContactPhase(
        1, np.array([0.0, 0.0, -0.1]), 0, 7, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0
    )
    problem = TrajectorySQP(
        model, 60, [1], initial[:, None, :3], initial[:, None, 3:7], [phase]
    )
    checkpoint = tmp_path / "window.npz"
    committed = []

    def record_first(state):
        committed.append(state.committed)
        if len(committed) == 1:
            save_window_state(checkpoint, state)

    poses, anchors, _ = solve_trajectory_windows(
        problem, initial, 80, 4, on_window=record_first
    )
    state = load_window_state(checkpoint, initial)
    resumed, resumed_anchors, _ = solve_trajectory_windows(
        problem, initial, 80, 4, state=state
    )
    assert committed == [4, 6, 7]
    np.testing.assert_array_equal(poses[:4], state.poses[:4])
    np.testing.assert_array_equal(resumed, poses)
    np.testing.assert_array_equal(resumed_anchors, anchors)
    assert problem.feasible(poses, anchors, problem.measure(poses, anchors))


def test_gmr_sqp_bank_roundtrip(tmp_path):
    """Exercise the real default pipeline on the two ranges of one small BIN."""
    import os
    import shutil
    from pathlib import Path

    from motion_matching.core.database import MotionMatchingDatabase
    from motion_matching.robots.base import locked_wrist_names
    from motion_matching.utils import PKG_ROOT

    from retargeting.retarget_contacts import (
        foot_contact_labels,
    )
    from retargeting.retarget_database import (
        convert_trajectory_bank,
        load_contact_phases,
    )
    from retargeting.verify_t2_bank import verify_trajectory_range

    gmr_root = os.environ.get("GMR_ROOT")
    bank = os.environ.get("G1_BANK_DIR")
    if not gmr_root or not bank:
        pytest.skip("Set GMR_ROOT and G1_BANK_DIR for real GMR + SQP conversion")
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    name = "database_g1_jump_35.bin"
    shutil.copyfile(Path(bank) / name, source / name)
    shutil.copytree(Path(bank) / "terrain/jump", source / "terrain/jump")
    model = PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml"
    summary = convert_trajectory_bank(source, target, model, Path(gmr_root), workers=1)
    assert summary["status"] == "complete"
    assert summary["frames"] == 68
    before, after = MotionMatchingDatabase(), MotionMatchingDatabase()
    before.load_from_file(str(source / name))
    after.load_from_file(str(target / name))
    np.testing.assert_array_equal(before.range_starts, after.range_starts)
    np.testing.assert_array_equal(before.range_stops, after.range_stops)
    spec = make_t2_spec(model)
    np.testing.assert_array_equal(after.bone_parents, spec.bone_parents)
    poses = spec.to_mujoco_qpos(
        spec.decode_bones_to_qpos(after.bone_positions, after.bone_rotations)
    )
    wrist_columns = [
        spec.model.joint(n).qposadr[0] for n in locked_wrist_names(spec.model)
    ]
    np.testing.assert_allclose(poses[:, wrist_columns], 0.0, atol=1e-6)
    for side in ("left", "right"):
        assert spec.model.geom(side + "_g1_sphere_collision").id >= 0
    for index, (start, stop) in enumerate(zip(after.range_starts, after.range_stops)):
        work = target.with_name("target_work") / Path(name).stem / f"range_{index:02d}"
        phases = load_contact_phases(work / "phases.json")
        np.testing.assert_array_equal(
            after.contact_states[start:stop],
            foot_contact_labels(phases, stop - start, spec),
        )
        result = verify_trajectory_range(
            work,
            poses[start:stop],
            after.contact_states[start:stop],
            spec,
            summary["reports"][0]["range_reports"][index],
        )
        assert result["feasible"]

        assert {path.name for path in work.iterdir()} == {
            "scene.xml",
            "initialization.json",
            "phases.json",
            "trajectory.npz",
            "report.json",
            "worker.log",
        }
        if index == 0:
            penetrating = poses[start:stop].copy()
            penetrating[:, 2] -= 0.02
            with pytest.raises(
                ValueError, match="violates geometry/contact/hard constraints"
            ):
                verify_trajectory_range(
                    work,
                    penetrating,
                    after.contact_states[start:stop],
                    spec,
                    summary["reports"][0]["range_reports"][index],
                )

    # Accepted range artifacts are reused; the optimizer must not run again.
    results = sorted(target.with_name("target_work").rglob("trajectory.npz"))
    timestamps = [path.stat().st_mtime_ns for path in results]
    repeated = convert_trajectory_bank(source, target, model, Path(gmr_root), workers=1)
    assert repeated["status"] == "complete"
    assert [path.stat().st_mtime_ns for path in results] == timestamps
    assert not (target / "bank_request.json").exists()
    assert not list(target.glob("*.report.json"))
    with pytest.raises(ValueError, match="Inputs or settings changed"):
        convert_trajectory_bank(
            source, target, model, Path(gmr_root), iterations=79, workers=1
        )
