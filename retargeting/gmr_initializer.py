"""G1 body targets and GMR initialization for the locked-wrist T2 model."""

import importlib
import sys
import types
from pathlib import Path

import mujoco as mj
import numpy as np
from motion_matching.robots.base import locked_wrist_names
from scipy.spatial.transform import Rotation as R


def load_gmr(root: Path):
    # Import only upstream's solver and params, bypassing its GUI/torch exports.
    package = types.ModuleType("_php_headless_gmr")
    package.__path__ = [str(root.resolve() / "general_motion_retargeting")]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + ".motion_retarget")


def add_sphere_frames(names, positions, rotations):
    """G1 URDF palm's fixed transform, shared by both sphere-hand links."""
    names = list(names)
    extra_pos, extra_rot = [], []
    for side in ("left", "right"):
        index = names.index(side + "_wrist_yaw_link")
        r = R.from_quat(rotations[:, index], scalar_first=True)
        extra_pos.append(positions[:, index] + r.apply([0.029, -0.003, 0.0]))
        extra_rot.append((r * R.from_euler("y", 1.57)).as_quat(scalar_first=True))
    return (
        names + ["left_sphere_hand", "right_sphere_hand"],
        np.concatenate([positions, np.stack(extra_pos, axis=1)], axis=1),
        np.concatenate([rotations, np.stack(extra_rot, axis=1)], axis=1),
    )


def constrain_locked_wrists(solver):
    import mink

    names = locked_wrist_names(solver.model)
    if names:
        solver.ik_limits.append(
            mink.VelocityLimit(solver.model, {name: 0.0 for name in names})
        )


def make_config(source, target):
    """Calibrate rest orientations, including T2's lateral arm bind pose."""
    sd, td = mj.MjData(source), mj.MjData(target)
    mj.mj_forward(source, sd)
    mj.mj_forward(target, td)
    pairs = [
        ("waist_yaw_link", "pelvis", 100.0, 10.0),
        ("trunk", "torso_link", 20.0, 10.0),
    ]
    for side in ("left", "right"):
        pairs += [
            (f"{side}_hip_yaw_link", f"{side}_hip_yaw_link", 10.0, 1.0),
            (f"{side}_knee_pitch_link", f"{side}_knee_link", 40.0, 1.0),
            (f"{side}_ankle_roll_link", f"{side}_ankle_roll_link", 100.0, 10.0),
            # T2's fourth arm joint is the physical elbow flexion.
            (f"{side}_elbow_yaw_link", f"{side}_elbow_link", 10.0, 0.0),
            (f"{side}_hand_ee_link", f"{side}_sphere_hand", 100.0, 0.0),
        ]

    def pose(model, data, name):
        idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"Missing body {name}")
        return data.xpos[idx], R.from_quat(data.xquat[idx], scalar_first=True)

    sp, _ = pose(source, sd, "pelvis")
    tp, _ = pose(target, td, "waist_yaw_link")
    sf, _ = pose(source, sd, "left_ankle_roll_link")
    tf, _ = pose(target, td, "left_ankle_roll_link")
    scale = float((tp[2] - tf[2]) / (sp[2] - sf[2]))
    table = {}
    for target_name, source_name, pw, rw in pairs:
        if source_name.endswith("_sphere_hand"):
            _, sr = pose(
                source, sd, source_name.replace("_sphere_hand", "_wrist_yaw_link")
            )
            sr = sr * R.from_euler("y", 1.57)
        else:
            _, sr = pose(source, sd, source_name)
        _, tr = pose(target, td, target_name)
        offset = (sr.inv() * tr).as_quat(scalar_first=True).tolist()
        position_offset = [0.0, 0.0, 0.0]
        if target_name == "trunk":
            sbody, _ = pose(source, sd, source_name)
            tbody, _ = pose(target, td, target_name)
            position_offset = (
                tr.inv().apply((tbody - tp) - scale * (sbody - sp)).tolist()
            )
        table[target_name] = [source_name, pw, rw, position_offset, offset]
    return dict(
        human_height_assumption=1.0,
        human_root_name="pelvis",
        robot_root_name="waist_yaw_link",
        ground_height=0.0,
        human_scale_table={entry[0]: scale for entry in table.values()},
        use_ik_match_table1=True,
        use_ik_match_table2=False,
        ik_match_table1=table,
        ik_match_table2={},
    ), scale


def source_body_poses(source, source_qpos):
    """Evaluate the source joints and G1 sphere-hand frames in world space."""
    names = [source.body(int(b)).name for b in source.jnt_bodyid]
    data = mj.MjData(source)
    positions = []
    rotations = []
    for pose in source_qpos:
        data.qpos[:] = pose
        mj.mj_forward(source, data)
        positions.append(data.xpos[source.jnt_bodyid].copy())
        rotations.append(data.xquat[source.jnt_bodyid].copy())
    names, positions, rotations = add_sphere_frames(
        names, np.array(positions), np.array(rotations)
    )
    return names, positions, rotations


def initialize_gmr(
    model_path,
    gmr_root,
    config,
    config_path,
    source_qpos,
    names,
    positions,
    rotations,
    fps,
):
    """Return GMR poses and SQP body targets without altering source motion."""
    model = mj.MjModel.from_xml_path(str(model_path))
    gmr = load_gmr(gmr_root)
    gmr.ROBOT_XML_DICT["trajectory_seed"] = model_path
    gmr.IK_CONFIG_DICT["trajectory_seed"] = {"trajectory_seed": config_path}
    initializer = gmr.GeneralMotionRetargeting(
        "trajectory_seed", "trajectory_seed", verbose=False
    )
    initializer.model.opt.timestep = 1 / fps
    constrain_locked_wrists(initializer)
    pose = model.qpos0.copy()
    pose[3:7] = source_qpos[0, 3:7]
    td = mj.MjData(model)
    td.qpos[:] = pose
    mj.mj_forward(model, td)
    pose[:3] += source_qpos[0, :3] - td.body("waist_yaw_link").xpos
    initializer.configuration.update(pose)
    seed = []
    desired_pos = []
    desired_rot = []
    for t in range(len(source_qpos)):
        frame = {
            n: (
                positions[t, names.index(n)].copy(),
                rotations[t, names.index(n)].copy(),
            )
            for n in config["human_scale_table"]
        }
        for _ in range(30 if t == 0 else 1):
            before = initializer.configuration.data.qpos.copy()
            pose = initializer.retarget(frame)
            if np.max(abs(pose - before)) < 1e-5:
                break
        seed.append(pose.copy())
        ps = []
        rs = []
        for body, entry in config["ik_match_table1"].items():
            index = names.index(entry[0])
            rotation = R.from_quat(
                rotations[t, index], scalar_first=True
            ) * R.from_quat(entry[4], scalar_first=True)
            ps.append(positions[t, index] + rotation.apply(entry[3]))
            rs.append(rotation.as_quat(scalar_first=True))
        desired_pos.append(ps)
        desired_rot.append(rs)
    return np.array(seed), np.array(desired_pos), np.array(desired_rot)
