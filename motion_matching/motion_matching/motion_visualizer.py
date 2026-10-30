#!/usr/bin/env python3
"""Robot motion visualizer with tracking and retargeted motion playback.

Animate G1 and T2 using NPZ motions and BIN databases.

Usage:
    python motion_visualizer.py --motion-file motion.npz --terrain-file terrain.obj
    python motion_visualizer.py --motion-file path/to/directory/

Supported motion file format (FAR-tracking schema, produced by run.py):
- .npz with keys: fps, joint_pos, body_pos_w, body_quat_w, joint_vel,
  body_lin_vel_w, body_ang_vel_w, optional vel_cmd.
"""

from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import Literal
from threading import Lock

import numpy as np
import trimesh
import tyro
from scipy.spatial.transform import Rotation, Slerp

import viser
from viser.extras import ViserUrdf

from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.kinematics import quat
from motion_matching.metadata.constants import G1_JOINT_AXES, G1_BONE_LEFT_FOOT, G1_BONE_RIGHT_FOOT
from motion_matching.io.motion_io import save_edited_motion_npz
from motion_matching.robots import get_robot_spec
from motion_matching.scenarios.builtin import VEL_CMD_MAPPING
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY, resolve_paths
from motion_matching.utils import G1_URDF
from motion_matching.viz.scene import add_world_scene
from motion_matching.utils.math_utils import quat_to_scaled_angle_axis


# Reverse of the joint reordering applied by qpos_to_tracking — restores the
# URDF's actuated-joint order so ViserUrdf.update_cfg gets the right inputs.
# Mapping copied from tracking_motion_animator.py:108-115.
_TRACKING_JOINT_MAPPING = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
_REVERSE_JOINT_MAPPING = np.zeros(len(_TRACKING_JOINT_MAPPING), dtype=np.int64)
for _orig, _mapped in enumerate(_TRACKING_JOINT_MAPPING):
    _REVERSE_JOINT_MAPPING[_mapped] = _orig


def reconstruct_box_mesh(
    center: np.ndarray, size: np.ndarray, quat_wxyz: np.ndarray
) -> trimesh.Trimesh:
    """Build a transformed box from center, extents, and wxyz quaternion."""
    box = trimesh.creation.box(extents=size)
    transform = trimesh.transformations.quaternion_matrix(quat_wxyz)
    transform[:3, 3] = center
    box.apply_transform(transform)
    return box


def load_motion_data(
    motion_file: Path,
    *,
    robot_spec=None,
    range_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Load motion data in the FAR-tracking schema produced by run.py.

    G1 and T2 accept MuJoCo ``qpos``; T2 tracking uses joint/body names.
    Official G1 native tracking stores ``[xyz, wxyz, joints]`` in ``joint_pos``.
    G1 tracking stores joints in a custom policy order; the loader
    applies the reverse mapping so ``joint_data`` is in URDF-actuated order.
    A synthetic ``qpos_data`` (``[quat_wxyz, pos, joints]``) is reconstructed
    from ``body_pos_w[:, 0]`` / ``body_quat_w[:, 0]``.

    Returns:
        joint_data: (T, 29) joints in URDF order
        root_data:  (T, 7)  [x, y, z, qw, qx, qy, qz]
        qpos_data:  (T, 36) [qw, qx, qy, qz, x, y, z, joints]
        fps:        Frames per second
        vel_cmd_data: optional (T, D)
    """
    if motion_file.suffix == ".bin":
        db = MotionMatchingDatabase()
        db.load_from_file(str(motion_file))
        index = 0 if range_index is None else range_index
        start, stop = db.range_starts[index], db.range_stops[index]
        positions, rotations = db.bone_positions[start:stop], db.bone_rotations[start:stop]
        if robot_spec is not None:
            qpos = robot_spec.decode_bones_to_qpos(positions, rotations)
        else:
            angles = quat_to_scaled_angle_axis(rotations[:, -29:].reshape(-1, 4)).reshape(-1, 29, 3)
            joints = np.sum(angles * G1_JOINT_AXES, axis=-1)
            world_rotations, world_positions = quat.fk(rotations, positions, db.bone_parents)
            qpos = np.concatenate((world_rotations[:, 1], world_positions[:, 1], joints), axis=1)
        roots = np.concatenate((qpos[:, 4:7], qpos[:, :4]), axis=1)
        return qpos[:, 7:], roots, qpos, 60.0, None

    if motion_file.suffix != ".npz":
        raise ValueError(f"Unsupported file format: {motion_file.suffix}. Use .npz or .bin.")

    with np.load(motion_file, allow_pickle=False) as data:
        if robot_spec is None and "qpos" in data.files:
            qpos = data["qpos"]
            php_qpos = np.concatenate((qpos[:, 3:7], qpos[:, :3], qpos[:, 7:]), axis=1)
            fps = float(data["fps"].item())
            return qpos[:, 7:], qpos[:, :7], php_qpos, fps, None

        if robot_spec is not None:
            fps = float(data["fps"].item())
            if "qpos" in data.files:
                qpos_data = robot_spec.from_mujoco_qpos(data["qpos"])
            else:
                root = data["body_names"].tolist().index(data["root_body_name"].item())
                joint_names = data["joint_names"].tolist()
                joints = [joint_names.index(name) for name in robot_spec.joint_names]
                qpos_data = np.concatenate(
                    (data["body_quat_w"][:, root], data["body_pos_w"][:, root],
                     data["joint_pos"][:, joints]), axis=1,
                )
            root_data = np.concatenate((qpos_data[:, 4:7], qpos_data[:, :4]), axis=1)
            vel_cmd = data["vel_cmd"] if "vel_cmd" in data.files else None
            return qpos_data[:, 7:], root_data, qpos_data, fps, vel_cmd

        required = ("fps", "joint_pos", "body_pos_w", "body_quat_w")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise ValueError(
                f"NPZ file {motion_file} missing required keys {missing}; "
                "provide a generated tracking motion from motion_matching.run"
            )

        fps_val = data["fps"]
        if fps_val.size != 1:
            raise ValueError("fps must contain one positive frame rate")
        fps = float(fps_val.item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")

        joints = data["joint_pos"]
        if joints.ndim == 2 and joints.shape[1] == 36:
            qpos = np.concatenate((joints[:, 3:7], joints[:, :3], joints[:, 7:]), axis=1)
            vel_cmd = data["vel_cmd"] if "vel_cmd" in data.files else None
            return joints[:, 7:], joints[:, :7], qpos, fps, vel_cmd

        positions = data["body_pos_w"]
        quaternions = data["body_quat_w"]
        if joints.ndim != 2 or joints.shape[1] != 29 or len(joints) == 0:
            raise ValueError(f"joint_pos must have shape (T, 29) with T > 0, got {joints.shape}")
        num_timesteps = len(joints)
        if (positions.ndim != 3 or positions.shape[0] != num_timesteps
                or positions.shape[1] == 0 or positions.shape[2] != 3):
            raise ValueError("body_pos_w must have shape (T, B, 3) matching joint_pos, with B > 0")
        if quaternions.shape != (*positions.shape[:2], 4):
            raise ValueError("body_quat_w must have shape (T, B, 4) matching body_pos_w")
        if not all(np.isfinite(values).all() for values in (joints, positions, quaternions)):
            raise ValueError("Motion arrays must contain only finite values")
        if np.any(np.linalg.norm(quaternions[:, 0], axis=-1) == 0):
            raise ValueError("Root quaternions must be nonzero")

        joint_data = joints[:, _REVERSE_JOINT_MAPPING].astype(np.float32)
        root_pos = positions[:, 0, :].astype(np.float32)
        root_quat = quaternions[:, 0, :].astype(np.float32)
        root_data = np.concatenate([root_pos, root_quat], axis=1)
        qpos_data = np.concatenate([root_quat, root_pos, joint_data], axis=1).astype(np.float32)

        vel_cmd_data = data["vel_cmd"] if "vel_cmd" in data.files else None
        if vel_cmd_data is not None:
            if vel_cmd_data.ndim != 2 or len(vel_cmd_data) != num_timesteps:
                raise ValueError("vel_cmd must have shape (T, D) matching joint_pos")
            if not np.isfinite(vel_cmd_data).all():
                raise ValueError("vel_cmd must contain only finite values")

        print(f"Loaded NPZ motion data: {num_timesteps} frames @ {fps} FPS")
        print(f"Joints: {joint_data.shape[1]}  Bodies: {positions.shape[1]}")
        if vel_cmd_data is not None:
            print(f"Loaded vel_cmd data: {vel_cmd_data.shape}")

    return joint_data, root_data, qpos_data, fps, vel_cmd_data


def main(
    motion_file: list[Path],
    robot: Literal["g1", "t2"] | None = None,
    retargeted_motion_file: Path | None = None,
    urdf_path: Path | None = None,
    model_path: Path | None = None,
    terrain_file: Path | None = None,
    output_fps: int | None = None,
    loop: bool = True,
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    show_vel_cmd: bool = False,
) -> None:
    """View motion with Robot, Stage and Seq selectors in a shared sidebar.

    Args:
        motion_file: One or more NPZ/BIN files or directories (searched recursively).
        robot: Only include this robot. If omitted, include all detected robots.
        retargeted_motion_file: Optional second NPZ or directory of retargeted motions.
        urdf_path: Optional G1 URDF override.
        model_path: Optional T2 MJCF override.
        terrain_file: Optional terrain OBJ used when a sequence has no paired terrain.
        output_fps: Frame rate for saved edits; defaults to the selected motion's FPS.
        loop: Loop the selected sequence.
        load_meshes: Load visual meshes.
        load_collision_meshes: Load collision meshes.
        show_vel_cmd: Show velocity commands in a matplotlib window.
    """
    def paired_terrain_files(path: Path) -> list[Path]:
        stems = dict.fromkeys((path.stem.replace("_motion", "_terrain"), path.stem + "_terrain"))
        return [path.parent / f"{stem}{suffix}" for stem in stems for suffix in (".npy", ".obj")]

    def terrain_path(path: Path, range_index: int | None) -> Path | None:
        if path.suffix == ".bin":
            config = next((cfg for skill, cfg in SKILL_REGISTRY.items()
                           if resolve_paths(skill, cfg)[0] == path.name), {})
            terrains = config.get("terrains", [])
            if terrains:
                return path.parent / terrains[range_index]
            return next((p for p in (
                path.with_name(f"{path.stem}_range{range_index:02d}_terrain.npy"),
                path.with_name(f"{path.stem}_range{range_index:02d}_terrain.obj"),
            ) if p.is_file()), terrain_file)
        return next((p for p in paired_terrain_files(path) if p.is_file()), terrain_file)

    def has_terrain(path: Path, range_index: int | None) -> bool:
        terrain = terrain_path(path, range_index)
        if terrain is None or not terrain.is_file():
            return False
        if terrain.suffix == ".npy":
            return np.load(terrain, allow_pickle=False).size > 0
        return True

    catalog: dict[str, dict[str, dict[str, tuple[Path, int | None]]]] = {}
    sources = list(motion_file)
    if retargeted_motion_file is not None:
        sources.append(retargeted_motion_file)
    for source in sources:
        paths = sorted([*source.rglob("*.npz"), *source.rglob("*.bin")]) if source.is_dir() else [source]
        for path in paths:
            name = str(path.relative_to(source)) if source.is_dir() else path.name
            if path.suffix == ".bin":
                db = MotionMatchingDatabase()
                db.load_from_file(str(path))
                robot_name = "T2" if len(db.bone_parents) == 33 else "G1"
                stage = "Database"
                sequences = {
                    (f"{name} / range {index:02d}"
                     + ("" if has_terrain(path, index) else " [No terrain]")): (path, index)
                    for index in range(len(db.range_starts))
                }
            else:
                with np.load(path, allow_pickle=False) as data:
                    if "joint_pos" in data.files:
                        robot_name = "T2" if data["joint_pos"].shape[1] == 31 else "G1"
                        stage = "Motion matching"
                    elif "qpos" in data.files and data["qpos"].shape[1] in (36, 38):
                        robot_name = "T2" if data["qpos"].shape[1] == 38 else "G1"
                        stage = "Retargeted"
                    else:
                        continue
                    has_object = "object_qpos" in data.files and path.with_name(path.stem + "_object.obj").is_file()
                label = name if has_object or has_terrain(path, None) else name + " [No terrain]"
                sequences = {label: (path, None)}
            if robot is None or robot_name.lower() == robot:
                catalog.setdefault(robot_name, {}).setdefault(stage, {}).update(sequences)
    if not catalog:
        raise ValueError("No supported NPZ/BIN motion files found")

    for robot_name, stages in catalog.items():
        catalog[robot_name] = {
            stage: stages[stage] for stage in ("Retargeted", "Database", "Motion matching")
            if stage in stages
        }
    initial_robot = next(iter(catalog))
    initial_stage = next(iter(catalog[initial_robot]))
    server = viser.ViserServer()
    root_frame = add_world_scene(server)

    with server.gui.add_folder("Motion"):
        robot_select = server.gui.add_dropdown("Robot", options=list(catalog), initial_value=initial_robot)
        stage_select = server.gui.add_dropdown(
            "Stage", options=list(catalog[initial_robot]), initial_value=initial_stage,
        )
        seq_select = server.gui.add_dropdown(
            "Seq", options=list(catalog[initial_robot][initial_stage]),
        )
        autoplay = server.gui.add_checkbox("Autoplay Sequence", initial_value=False)
        delete = server.gui.add_button("Delete Current File")
    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("Play", initial_value=False)
        restart = server.gui.add_button("Restart")
        speed = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        save = server.gui.add_button("Save Motion")

    with server.gui.add_folder("Motion Info"):
        info_file = server.gui.add_text("File", initial_value="", disabled=True)
        info_frames = server.gui.add_text("Timesteps", initial_value="", disabled=True)
        info_joints = server.gui.add_text("Joints", initial_value="", disabled=True)
        info_duration = server.gui.add_text("Duration", initial_value="", disabled=True)
        info_fps = server.gui.add_text("FPS", initial_value="", disabled=True)
        info_command = server.gui.add_text("Velocity Command", initial_value="No Data", disabled=True)

        contact_label = server.gui.add_text("Contact (L/R)", initial_value="No Data", disabled=True)
        show_contact_markers = server.gui.add_checkbox("Show Contact Markers", initial_value=True)

    contact_markers = [None, None]
    root_sliders = []
    with server.gui.add_folder("Root Transform", expand_by_default=False):
        for label, lower, upper in (
            ("X Position", -5.0, 5.0), ("Y Position", -5.0, 5.0), ("Z Position", -2.0, 3.0),
            ("Roll", -np.pi, np.pi), ("Pitch", -np.pi, np.pi), ("Yaw", -np.pi, np.pi),
        ):
            root_sliders.append(server.gui.add_slider(label, min=lower, max=upper, step=0.01, initial_value=0.0))

    with server.gui.add_folder("Visibility", expand_by_default=False):
        show_meshes = server.gui.add_checkbox("Show meshes", initial_value=load_meshes, visible=load_meshes)
        show_collisions = server.gui.add_checkbox(
            "Show collision meshes", initial_value=load_collision_meshes, visible=load_collision_meshes,
        )

    with server.gui.add_folder("Terrain Controls") as terrain_folder:
        terrain_variant = server.gui.add_dropdown("Variant", options=["0"])
        rerandomize_terrain = server.gui.add_button("Re-randomize Terrain")
    terrain_folder.visible = False

    playback_lock = Lock()
    renderer, spec = None, None
    state = {"motion": None, "frame": 0.0, "contacts": None, "contact_positions": None}
    joint_folder = None
    joint_sliders = []
    terrain_handle = None
    terrain_boxes = None
    object_handle = None
    object_qpos = None


    def clear_terrain() -> None:
        nonlocal terrain_handle, terrain_boxes, object_handle, object_qpos
        if object_handle is not None:
            object_handle.remove()
            object_handle = None
        object_qpos = None
        if terrain_handle is not None:
            terrain_handle.remove()
            terrain_handle = None
        terrain_boxes = None
        terrain_folder.visible = False

    def apply_terrain_variant(index: int) -> None:
        nonlocal terrain_handle
        if terrain_handle is not None:
            terrain_handle.remove()
        mesh = trimesh.util.concatenate([
            reconstruct_box_mesh(box[:3], box[7:10], box[3:7]) for box in terrain_boxes[:, index]
        ])
        terrain_handle = server.scene.add_mesh_trimesh("/terrain", mesh=mesh)

    def load_terrain(path: Path, range_index: int | None) -> None:
        nonlocal terrain_handle, terrain_boxes, object_handle, object_qpos
        clear_terrain()
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "object_qpos" in data:
                    object_qpos = data["object_qpos"]
                    mesh = trimesh.load(path.with_name(path.stem + "_object.obj"), force="mesh")
                    object_handle = server.scene.add_mesh_trimesh("/tracked_object", mesh=mesh)
        terrain = terrain_path(path, range_index)
        if terrain is None:
            return
        if terrain.suffix == ".npy":
            terrain_boxes = np.load(terrain, allow_pickle=False)
            if not terrain_boxes.shape[0] or not terrain_boxes.shape[1]:
                return
            terrain_folder.visible = True
            variants = terrain_boxes.shape[1]
            terrain_variant.options = [str(index) for index in range(variants)]
            initial = int(np.random.randint(variants))
            terrain_variant.value = str(initial)
            rerandomize_terrain.visible = variants > 1
            apply_terrain_variant(initial)
        else:
            mesh = trimesh.load(terrain, force="mesh")
            terrain_handle = server.scene.add_mesh_trimesh("/terrain", mesh=mesh)

    def draw_pose(index: int) -> None:
        joints, roots, _, _, _ = state["motion"]
        with server.atomic():
            renderer.update_cfg(joints[index])
            root_frame.position = roots[index, :3]
            root_frame.wxyz = roots[index, 3:7]
            if object_handle is not None:
                object_handle.position = object_qpos[index, :3]
                object_handle.wxyz = object_qpos[index, 3:7]
            for slider, value in zip(joint_sliders, joints[index]):
                slider.value = float(value)
            euler = Rotation.from_quat(roots[index, [4, 5, 6, 3]]).as_euler("xyz")
            for slider, value in zip(root_sliders, np.concatenate((roots[index, :3], euler))):
                slider.value = float(value)
            contacts = state["contacts"]
            contact_label.value = (
                f"L: {int(contacts[index, 0])}  R: {int(contacts[index, 1])}"
                if contacts is not None else "No Data"
            )
            for side, name in enumerate(("left", "right")):
                visible = contacts is not None and show_contact_markers.value and bool(contacts[index, side])
                if visible:
                    # Same marker geometry and styling as play_database.py/run.py.
                    contact_markers[side] = server.scene.add_icosphere(
                        f"/contact_marker/{name}", position=state["contact_positions"][index, side],
                        radius=0.06, color=np.array([0, 100, 255]), subdivisions=2,
                    )
                elif contact_markers[side] is not None:
                    contact_markers[side].remove()
                    contact_markers[side] = None
            commands = state["motion"][4]
            info_command.value = (
                VEL_CMD_MAPPING.get(int(np.argmax(commands[index])), "Undefined")
                if commands is not None else "No Data"
            )

    def select_seq() -> None:
        play.value = False
        path, range_index = catalog[robot_select.value][stage_select.value][seq_select.value]
        motion = load_motion_data(path, robot_spec=spec, range_index=range_index)
        state.update(motion=motion, frame=0.0, contacts=None, contact_positions=None)
        if path.suffix == ".bin":
            db = MotionMatchingDatabase()
            db.load_from_file(str(path))
            start, stop = db.range_starts[range_index], db.range_stops[range_index]
            _, positions = quat.fk(db.bone_rotations[start:stop], db.bone_positions[start:stop], db.bone_parents)
            feet = [spec.bone_left_foot, spec.bone_right_foot] if spec is not None else [G1_BONE_LEFT_FOOT, G1_BONE_RIGHT_FOOT]
            state.update(contacts=db.contact_states[start:stop], contact_positions=positions[:, feet])
        show_contact_markers.disabled = state["contacts"] is None
        frame.max = len(motion[0]) - 1
        frame.value = 0
        for index, slider in enumerate(root_sliders[:3]):
            default_lower, default_upper = (-2.0, 3.0) if index == 2 else (-5.0, 5.0)
            slider.min = min(default_lower, float(motion[1][:, index].min()) - 1.0)
            slider.max = max(default_upper, float(motion[1][:, index].max()) + 1.0)
        info_file.value = path.name if range_index is None else f"{path.name} / range {range_index:02d}"
        info_frames.value = str(len(motion[0]))
        info_joints.value = str(motion[0].shape[1])
        info_duration.value = f"{len(motion[0]) / motion[3]:.2f} s"
        info_fps.value = f"{motion[3]:g}"
        load_terrain(path, range_index)
        draw_pose(0)
        print(f"Loaded {path}: {len(motion[0])} frames @ {motion[3]:g} FPS")

    def select_stage() -> None:
        paths = catalog[robot_select.value][stage_select.value]
        seq_select.options = list(paths)
        seq_select.value = next(iter(paths))
        select_seq()

    def select_robot() -> None:
        nonlocal renderer, spec, joint_folder, joint_sliders
        play.value = False
        if renderer is not None:
            renderer.remove()
        if joint_folder is not None:
            joint_folder.remove()
        spec = get_robot_spec("t2", model_path=model_path) if robot_select.value == "T2" else None
        if spec is not None:
            from motion_matching.viz.mujoco import ViserMujoco

            renderer = ViserMujoco(
                server, spec, load_meshes=load_meshes, load_collision_meshes=load_collision_meshes,
            )
        else:
            renderer = ViserUrdf(
                server, urdf_or_path=urdf_path or Path(G1_URDF), root_node_name="/robot_root",
                load_meshes=load_meshes, load_collision_meshes=load_collision_meshes,
                collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
            )
        renderer.show_visual = show_meshes.value
        renderer.show_collision = show_collisions.value
        joint_sliders = []
        joint_folder = server.gui.add_folder("Joint Values (DOF)", expand_by_default=False)
        limits = renderer.get_actuated_joint_limits()
        with joint_folder:
            for name in renderer.get_actuated_joint_names():
                lower, upper = limits[name]
                lower = -np.pi if lower is None else lower
                upper = np.pi if upper is None else upper
                slider = server.gui.add_slider(
                    name, min=lower, max=upper, step=0.001, initial_value=float(np.clip(0.0, lower, upper)),
                )
                slider.on_update(edit_joints)
                joint_sliders.append(slider)
        stage_select.options = list(catalog[robot_select.value])
        stage_select.value = stage_select.options[0]
        select_stage()

    def edit_joints(event) -> None:
        if event.client_id is None:
            return
        with playback_lock:
            play.value = False
            index = int(frame.value)
            joints, _, qpos, _, _ = state["motion"]
            joints[index] = [slider.value for slider in joint_sliders]
            qpos[index, 7:] = joints[index]
            draw_pose(index)

    def edit_root(event) -> None:
        if event.client_id is None:
            return
        with playback_lock:
            play.value = False
            index = int(frame.value)
            _, roots, qpos, _, _ = state["motion"]
            values = np.array([slider.value for slider in root_sliders])
            rotation = Rotation.from_euler("xyz", values[3:]).as_quat()[[3, 0, 1, 2]]
            roots[index] = np.concatenate((values[:3], rotation))
            qpos[index, :4], qpos[index, 4:7] = rotation, values[:3]
            draw_pose(index)

    for slider in root_sliders:
        slider.on_update(edit_root)

    @frame.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                play.value = False
                state["frame"] = float(frame.value)
                draw_pose(int(state["frame"]))

    @show_contact_markers.on_update
    def update_contact_markers(_) -> None:
        with playback_lock:
            draw_pose(int(frame.value))

    @show_meshes.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                renderer.show_visual = show_meshes.value

    @show_collisions.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                renderer.show_collision = show_collisions.value

    @terrain_variant.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                apply_terrain_variant(int(terrain_variant.value))

    @rerandomize_terrain.on_click
    def _(_event) -> None:
        with playback_lock:
            current, count = int(terrain_variant.value), terrain_boxes.shape[1]
            index = (current + int(np.random.randint(1, count))) % count
            terrain_variant.value = str(index)
            apply_terrain_variant(index)

    @robot_select.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                select_robot()

    @stage_select.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                select_stage()

    @seq_select.on_update
    def _(event) -> None:
        if event.client_id is not None:
            with playback_lock:
                select_seq()

    select_robot()

    @restart.on_click
    def _(_event) -> None:
        with playback_lock:
            state["frame"] = 0.0
            frame.value = 0
            draw_pose(0)
            play.value = True

    @save.on_click
    def _(_event) -> None:
        with playback_lock:
            path, range_index = catalog[robot_select.value][stage_select.value][seq_select.value]
            _, _, qpos, fps, commands = state["motion"]
            speed_value = float(speed.value)
            speed_suffix = f"{speed_value:.2f}".rstrip("0").rstrip(".").replace(".", "p")
            stem = path.stem if range_index is None else f"{path.stem}_range_{range_index:02d}"
            output_path = path.with_name(f"{stem}_speed_{speed_suffix}.npz")
            saved_fps = output_fps if output_fps is not None else fps
            tracking = save_edited_motion_npz(
                qpos, output_path, source_fps=fps, target_fps=saved_fps,
                speed=speed_value, vel_cmd=commands, robot_spec=spec,
            )
            if object_qpos is not None:
                source_times = np.arange(len(object_qpos))
                sample_times = np.linspace(0, len(object_qpos) - 1, len(tracking["joint_pos"]))
                positions = np.column_stack([
                    np.interp(sample_times, source_times, object_qpos[:, axis]) for axis in range(3)
                ])
                rotations = Slerp(source_times, Rotation.from_quat(
                    object_qpos[:, 3:7], scalar_first=True))(sample_times).as_quat(scalar_first=True)
                tracking["object_qpos"] = np.concatenate((positions, rotations), axis=1)
                np.savez(output_path, **tracking)
                shutil.copyfile(path.with_name(path.stem + "_object.obj"),
                                output_path.with_name(output_path.stem + "_object.obj"))
            print(f"Motion saved to {output_path} (speed={speed_value:.2f}, fps={saved_fps})")

    @delete.on_click
    def _(_event) -> None:
        with playback_lock:
            play.value = False
            robot_name, stage = robot_select.value, stage_select.value
            paths = catalog[robot_name][stage]
            current_index = list(paths).index(seq_select.value)
            path, _ = paths[seq_select.value]
            path.unlink()
            if path.suffix == ".npz":
                path.with_name(path.stem + "_object.obj").unlink(missing_ok=True)
                for terrain in paired_terrain_files(path):
                    if terrain.exists() or terrain.is_symlink():
                        terrain.unlink()
            for stages in catalog.values():
                for sequences in stages.values():
                    for name in [name for name, (source, _) in sequences.items() if source == path]:
                        del sequences[name]
                for empty_stage in [name for name, sequences in stages.items() if not sequences]:
                    del stages[empty_stage]
            for empty_robot in [name for name, stages in catalog.items() if not stages]:
                del catalog[empty_robot]
            print(f"Deleted {path}")
            if not catalog:
                state["motion"] = None
                clear_terrain()
                renderer.remove()
                joint_folder.remove()
                for control in (robot_select, stage_select, seq_select, play, restart, frame, save, delete, *root_sliders):
                    control.disabled = True
                info_file.value = "No file loaded"
                for info in (info_frames, info_joints, info_duration, info_fps):
                    info.value = "-"
                info_command.value = "No Data"
            elif robot_name not in catalog:
                robot_select.options = list(catalog)
                robot_select.value = robot_select.options[0]
                select_robot()
            else:
                stage_select.options = list(catalog[robot_name])
                stage_select.value = stage if stage in catalog[robot_name] else stage_select.options[0]
                seq_select.options = list(catalog[robot_name][stage_select.value])
                seq_select.value = seq_select.options[min(current_index, len(seq_select.options) - 1)]
                select_seq()

    fig_vel, ax_vel, vel_bars = None, None, None
    if show_vel_cmd:
        import matplotlib.pyplot as plt

        plt.ion()
        fig_vel, ax_vel = plt.subplots(figsize=(10, 4))

    while True:
        start = time.perf_counter()
        command = None
        with playback_lock:
            fps = 30.0
            if state["motion"] is not None:
                count, fps = len(state["motion"][0]), state["motion"][3]
                if play.value:
                    state["frame"] += speed.value
                    if state["frame"] >= count:
                        options = list(catalog[robot_select.value][stage_select.value])
                        next_index = options.index(seq_select.value) + 1
                        if autoplay.value and (next_index < len(options) or loop):
                            seq_select.value = options[next_index % len(options)]
                            select_seq()
                            play.value = True
                        elif loop:
                            state["frame"] %= count
                        else:
                            state["frame"] = count - 1
                            play.value = False
                    frame.value = int(state["frame"])
                    draw_pose(int(state["frame"]))
                else:
                    state["frame"] = float(frame.value)
                index = int(state["frame"])
                commands = state["motion"][4]
                if commands is not None:
                    command = commands[index].copy()
        if show_vel_cmd and plt.fignum_exists(fig_vel.number):
            if command is None:
                if vel_bars is not None:
                    for bar in vel_bars:
                        bar.set_height(0.0)
            elif vel_bars is None or len(vel_bars) != len(command):
                ax_vel.clear()
                ax_vel.set_title("Velocity Command (One-Hot)")
                ax_vel.set_xlabel("Command Index")
                ax_vel.set_ylabel("Activation")
                ax_vel.set_ylim(0, 1.1)
                vel_bars = ax_vel.bar(range(len(command)), command)
                ax_vel.set_xticks(range(len(command)))
                ax_vel.set_xticklabels(
                    [f"{i}:{VEL_CMD_MAPPING.get(i, str(i))}" for i in range(len(command))],
                    rotation=45, ha="right",
                )
                fig_vel.tight_layout()
            else:
                for bar, value in zip(vel_bars, command):
                    bar.set_height(value)
            fig_vel.canvas.draw_idle()
            fig_vel.canvas.flush_events()
        time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - start)))


if __name__ == "__main__":
    tyro.cli(main)
