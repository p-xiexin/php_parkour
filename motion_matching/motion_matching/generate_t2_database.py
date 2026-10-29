"""Build a T2 BIN with the official PHP database processing rules.

T2 changes only FK, static bind rotations, named mirror axes and body indices.
Resampling, Simulation bone smoothing, velocities and contacts follow the G1
builder in ``generate_database.py``.
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
from scipy import ndimage, signal
from scipy.interpolate import griddata

from motion_matching.kinematics import quat
from motion_matching.robots import RobotSpec, get_robot_spec


def _process_clip(qpos: np.ndarray, fps: float, spec: RobotSpec):
    positions, rotations = spec.encode_qpos_to_bones(qpos)
    positions, rotations = positions[:, 1:].copy(), rotations[:, 1:].copy()
    parents = spec.bone_parents[1:] - 1
    frame_ratio = 60. / fps
    if not np.abs(frame_ratio - 1.) < 1e-2:
        nframes, nbones = positions.shape[:2]
        times = np.linspace(0, nframes - 1, nframes)
        samples = np.linspace(0, nframes - 1, int(frame_ratio * nframes))
        positions = griddata(times, positions.reshape(nframes, -1), samples, method="cubic").reshape(-1, nbones, 3)
        rotations = griddata(times, rotations.reshape(nframes, -1), samples, method="cubic").reshape(-1, nbones, 4)
        rotations = quat.normalize(rotations)

    global_rotations, global_positions = quat.fk(lrot=rotations, lpos=positions, parents=parents)
    shoulders = [spec.bone_names.index(name) - 1 for name in
                 ("left_shoulder_pitch_link", "right_shoulder_pitch_link")]
    sim_position = np.array([1., 1., 0.]) * global_positions[:, shoulders].mean(axis=1, keepdims=True)
    sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode="interp")
    pelvis = spec.bone_hips - 1
    sim_direction = np.array([1., 1., 0.]) * quat.mul_vec(
        global_rotations[:, pelvis:pelvis + 1], np.array([1., 0., 0.]))
    sim_direction /= np.sqrt(np.sum(sim_direction ** 2, axis=-1))[..., None]
    window = 61 if len(sim_direction) > 61 else len(sim_direction)
    sim_direction = signal.savgol_filter(sim_direction, window, 3, axis=0, mode="interp")
    sim_direction /= np.sqrt(np.sum(sim_direction ** 2, axis=-1))[..., None]
    sim_rotation = quat.normalize(quat.between(np.array([1, 0, 0]), sim_direction))
    positions[:, 0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:, 0:1] - sim_position)
    rotations[:, 0:1] = quat.mul(quat.inv(sim_rotation), rotations[:, 0:1])
    positions = np.concatenate([sim_position, positions], axis=1)
    rotations = np.concatenate([sim_rotation, rotations], axis=1)

    velocities = np.empty_like(positions)
    velocities[1:-1] = (.5 * (positions[2:] - positions[1:-1]) * 60.
                          + .5 * (positions[1:-1] - positions[:-2]) * 60.)
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
    angular_velocities = np.zeros_like(positions)
    angular_velocities[1:-1] = (
        .5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[2:], rotations[1:-1]))) * 60.
        + .5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:-1], rotations[:-2]))) * 60.)
    angular_velocities[0] = angular_velocities[1] - (angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (angular_velocities[-2] - angular_velocities[-3])
    _, _, global_velocities, _ = quat.fk_vel(
        lrot=rotations, lpos=positions, lvel=velocities, lang=angular_velocities, parents=spec.bone_parents)
    contacts = np.linalg.norm(global_velocities[:, [spec.bone_left_foot, spec.bone_right_foot]], axis=-1) < .5
    for side in range(2):
        contacts[:, side] = ndimage.median_filter(contacts[:, side], size=6, mode="nearest")
    return positions, velocities, rotations, angular_velocities, contacts


def build_native_database(clips: list[dict], output_path: str | Path, *,
                          robot: str | RobotSpec = "t2", mirror: bool = True) -> dict:
    """Bake PHP-layout ``qpos`` clips with ``fps`` and an optional ``name``.

    Original and mirrored ranges are adjacent, as required by PHP's existing
    mirror tie breaking. The JSON sidecar records names and source ranges.
    """
    spec = get_robot_spec(robot) if isinstance(robot, str) else robot
    parts = [[], [], [], [], []]
    starts, stops, sources = [], [], []
    cursor = 0
    for index, clip in enumerate(clips):
        qpos = np.asarray(clip["qpos"])
        for mirrored in ([False, True] if mirror else [False]):
            pose = spec.mirror_qpos(qpos) if mirrored else qpos
            processed = _process_clip(pose, float(clip["fps"]), spec)
            for part, values in zip(parts, processed):
                part.append(values)
            starts.append(cursor)
            cursor += len(processed[0])
            stops.append(cursor)
            sources.append({"name": clip.get("name", f"clip_{index}"), "path": clip.get("path"),
                            "source_frames": len(qpos),
                            "source_fps": float(clip["fps"]), "mirrored": mirrored,
                            "range": [starts[-1], stops[-1]]})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.concatenate(part).astype(np.float32) for part in parts[:4]]
    contacts = np.concatenate(parts[4]).astype(np.uint8)
    with output.open("wb") as handle:
        for values in arrays:
            handle.write(struct.pack("II", values.shape[0], values.shape[1]) + values.tobytes())
        handle.write(struct.pack("I", spec.num_bones) + spec.bone_parents.tobytes())
        for values in (starts, stops):
            handle.write(struct.pack("I", len(values)) + np.asarray(values, dtype=np.int32).tobytes())
        handle.write(struct.pack("II", *contacts.shape) + contacts.tobytes())
    info = {"robot": spec.metadata(), "database_fps": 60, "num_frames": cursor,
            "range_starts": starts, "range_stops": stops, "sources": sources}
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Override the bundled T2 MJCF or PHP_T2_MJCF")
    parser.add_argument("--motion-file", type=Path, action="append", required=True)
    parser.add_argument("--layout", choices=("php", "mujoco"), default="php")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = get_robot_spec("t2", model_path=args.model)
    clips = []
    for path in args.motion_file:
        with np.load(path, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"])
            if args.layout == "mujoco":
                qpos = spec.from_mujoco_qpos(qpos)
            clips.append({"qpos": qpos, "fps": float(data["fps"].item()),
                          "name": path.stem, "path": str(path.resolve())})
    build_native_database(clips, args.output, robot=spec)


if __name__ == "__main__":
    main()
