"""IO and batch orchestration for GMR + SQP -> T2 databases.

Official G1 BINs use 60 Hz and do not contain a frame-rate field. The source
ranges, Simulation trajectory and terrain coordinates are preserved at scale 1.
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import mujoco as mj
import numpy as np
from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.kinematics import quat
from motion_matching.metadata.constants import G1_JOINT_AXES
from motion_matching.robots.base import locked_wrist_names
from motion_matching.robots.t2 import make_t2_spec
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY, resolve_paths
from motion_matching.utils import G1_MJCF
from motion_matching.utils.math_utils import quat_to_scaled_angle_axis
from scipy.spatial.transform import Rotation as R

from retargeting.gmr_initializer import (
    initialize_gmr,
    make_config,
    source_body_poses,
)
from retargeting.retarget_contacts import (
    ContactPhase,
    build_contact_phases,
    foot_contact_labels,
)
from retargeting.trajectory_optimization import (
    TrajectorySQP,
    WindowState,
    solve_trajectory_windows,
    trajectory_admissible,
    trajectory_metrics,
)

DATABASE_FPS = 60
DEFAULT_ITERATIONS = 80
DEFAULT_WINDOW_FRAMES = 1200


class RangeJob(TypedDict):
    id: str
    file: str
    skill: str
    range_index: int
    start: int
    stop: int
    output: str


def trajectory_runtime():
    """The pre-3.12 native GJK can contradict its own distance witnesses."""
    if tuple(map(int, mj.__version__.split(".")[:2])) < (3, 12):
        raise RuntimeError(
            "Scene retargeting requires MuJoCo >=3.12; "
            "use the isolated retarget environment"
        )
    if mj.mj_versionString() != mj.__version__:
        raise RuntimeError("MuJoCo Python and native library versions differ")
    return dict(
        mujoco=mj.__version__,
        mujoco_native=mj.mj_versionString(),
        numpy=np.__version__,
        scipy=importlib.import_module("scipy").__version__,
        clarabel=importlib.import_module("clarabel").__version__,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_contact_phases(path: Path) -> list[ContactPhase]:
    """Deserialize solve artifacts at the IO boundary."""
    phases = []
    for raw in json.loads(path.read_text()):
        for name in (
            "local_point",
            "anchor",
            "normal",
            "tangent_axes",
            "tangent_lower",
            "tangent_upper",
            "cap_axis_local",
        ):
            if raw[name] is not None:
                raw[name] = np.asarray(raw[name])
        phases.append(ContactPhase(**raw))
    return phases


def decode_g1_range(db, start, stop):
    """Decode the legacy G1 BIN layout directly to MuJoCo qpos."""
    positions = db.bone_positions[start:stop]
    rotations = db.bone_rotations[start:stop]
    angles = quat_to_scaled_angle_axis(rotations[:, -29:].reshape(-1, 4)).reshape(
        -1, 29, 3
    )
    joints = np.sum(angles * G1_JOINT_AXES, axis=-1)
    world_rotations, world_positions = quat.fk(rotations, positions, db.bone_parents)
    return np.concatenate(
        (world_positions[:, 1], world_rotations[:, 1], joints), axis=1
    )


def encode_range(qpos, source_db, start, stop, scale, spec, contacts):
    positions, rotations = spec.encode_qpos_to_bones(spec.from_mujoco_qpos(qpos))
    sim_pos = source_db.bone_positions[start:stop, 0].astype(float) * scale
    sim_rot = quat.normalize(source_db.bone_rotations[start:stop, 0].astype(float))
    # Keep original Simulation trajectory and range boundaries, without smoothing again.
    positions[:, 1] = quat.mul_vec(quat.inv(sim_rot), positions[:, 1] - sim_pos)
    rotations[:, 1] = quat.mul(quat.inv(sim_rot), rotations[:, 1])
    positions[:, 0], rotations[:, 0] = sim_pos, sim_rot
    rotations = quat.unroll(rotations)
    velocities = np.gradient(positions, 1.0 / DATABASE_FPS, axis=0)
    angular = np.zeros_like(positions)
    angular[1:] = (
        quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:], rotations[:-1])))
        * DATABASE_FPS
    )
    angular[0] = angular[1]
    contact = np.asarray(contacts, dtype=bool)
    if contact.shape != (stop - start, 2):
        raise ValueError("Contact annotations must match the complete range")
    return positions, velocities, rotations, angular, contact


def write_bin(path, parts, spec, starts, stops):
    temporary = path.with_suffix(".bin.tmp")
    with temporary.open("wb") as f:
        for array in parts[:4]:
            array = np.asarray(array, dtype=np.float32)
            f.write(struct.pack("II", *array.shape[:2]))
            f.write(array.tobytes())
        f.write(struct.pack("I", spec.num_bones))
        f.write(spec.bone_parents.astype(np.int32).tobytes())
        for array in (starts, stops):
            f.write(struct.pack("I", len(array)))
            f.write(np.asarray(array, dtype=np.int32).tobytes())
        contacts = np.asarray(parts[4], dtype=np.uint8)
        f.write(struct.pack("II", *contacts.shape))
        f.write(contacts.tobytes())
    temporary.replace(path)


def prepare_scene(spec, source_dir, output, skill, range_index):
    """Write the T2 model with the source range's paired finite box, if any."""
    tree = ET.parse(spec.model_path)
    option = tree.getroot().find("option")
    if option is None:
        option = ET.SubElement(tree.getroot(), "option")
    option.set("ccd_tolerance", "1e-9")
    compiler = tree.getroot().find("compiler")
    compiler.set(
        "meshdir",
        str((spec.model_path.parent / compiler.get("meshdir", ".")).resolve()),
    )
    surfaces = [(np.array([0.0, 0.0, 1.0]), 0.0, None, None, None)]
    if SKILL_REGISTRY[skill].get("terrains"):
        terrain = source_dir / SKILL_REGISTRY[skill]["terrains"][range_index]
        box = json.loads(terrain.with_suffix(".json").read_text())
        center = np.array(box["pos"])
        half = np.array(box["size"]) / 2
        rotation = R.from_quat(box["quat"], scalar_first=True).as_matrix()
        ET.SubElement(
            tree.getroot().find("worldbody"),
            "geom",
            name="scene_box",
            type="box",
            pos=" ".join(map(str, center)),
            quat=" ".join(map(str, box["quat"])),
            size=" ".join(map(str, half)),
            contype="1",
            conaffinity="1",
        )
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            tangent = rotation[:, other].T
            for sign in (-1, 1):
                normal = sign * rotation[:, axis]
                point = center + normal * half[axis]
                surfaces.append(
                    (
                        normal,
                        float(normal @ point),
                        tangent,
                        tangent @ center - half[other],
                        tangent @ center + half[other],
                    )
                )
    scene_path = output / "scene.xml"
    tree.write(scene_path)
    model = mj.MjModel.from_xml_path(str(scene_path))
    return model, surfaces


def load_window_state(path, initial):
    """Load a committed prefix after the caller checks the range report."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        poses = initial.copy()
        committed = int(saved["committed"])
        poses[:committed] = saved["qpos"]
        return WindowState(
            poses,
            saved["anchors"].copy(),
            committed,
            json.loads(str(saved["history"].item())),
        )


def save_window_state(path, state):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        qpos=state.poses[: state.committed],
        anchors=state.anchors,
        committed=state.committed,
        history=json.dumps(state.history),
    )
    temporary.replace(path)
    print(f"Committed {state.committed}/{len(state.poses)} frames", flush=True)


def _settings(model_path, gmr_root, iterations, window_frames):
    """Record the runtime and inputs needed to resume the same calculation."""
    return dict(
        algorithm="trajectory_sqp",
        model=str(model_path),
        model_sha256=digest(model_path),
        gmr_commit=subprocess.check_output(
            ["git", "-C", str(gmr_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        runtime_versions=trajectory_runtime(),
        ccd_tolerance=1e-9,
        iterations_limit=iterations,
        window_frames=window_frames,
    )


def _previous_report(path, request):
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("request") != request:
        raise ValueError("Inputs or settings changed; use a new output directory")
    return report


def _source_range(source, start, stop):
    db = MotionMatchingDatabase()
    db.load_from_file(str(source))
    ranges = np.flatnonzero((db.range_starts <= start) & (start < db.range_stops))
    if not len(ranges):
        raise ValueError("Start frame is outside the source ranges")
    index = int(ranges[0])
    stop = int(db.range_stops[index]) if stop is None else stop
    if stop > db.range_stops[index] or stop - start < 3:
        raise ValueError("Select at least three frames within one continuous range")
    return db, index, stop


def _prepare_problem(
    db, spec, source_dir, output, skill, range_index, start, stop, gmr_root
):
    """Build the unchanged GMR seed, contact phases and SQP tracking targets."""
    source = mj.MjModel.from_xml_path(str(G1_MJCF))
    source_qpos = decode_g1_range(db, start, stop)
    names, positions, rotations = source_body_poses(source, source_qpos)
    model, surfaces = prepare_scene(spec, source_dir, output, skill, range_index)
    config, _ = make_config(source, spec.model)
    config["human_scale_table"] = {name: 1.0 for name in config["human_scale_table"]}
    config_path = output / "initialization.json"
    write_json(config_path, config)
    seed, desired_pos, desired_rot = initialize_gmr(
        spec.model_path,
        gmr_root,
        config,
        config_path,
        source_qpos,
        names,
        positions,
        rotations,
        DATABASE_FPS,
    )
    phases = build_contact_phases(
        source,
        model,
        names,
        positions,
        rotations,
        db.contact_states[start:stop],
        surfaces,
        DATABASE_FPS,
    )
    (output / "phases.json").write_text(
        json.dumps(
            [asdict(p) for p in phases], default=lambda value: value.tolist(), indent=2
        )
        + "\n"
    )
    problem = TrajectorySQP(
        model,
        DATABASE_FPS,
        [model.body(n).id for n in config["ik_match_table1"]],
        desired_pos,
        desired_rot,
        phases,
        locked_wrist_names(model),
        position_weights=[
            entry[1] / 100 for entry in config["ik_match_table1"].values()
        ],
        orientation_weights=[
            entry[2] / 10 for entry in config["ik_match_table1"].values()
        ],
    )
    return problem, seed


def retarget_range(
    source_dir,
    output,
    model_path,
    gmr_root,
    skill,
    start=0,
    stop=None,
    iterations=DEFAULT_ITERATIONS,
    window_frames=DEFAULT_WINDOW_FRAMES,
):
    """Solve one range; report.json owns its inputs, progress and physical acceptance.

    A completed trajectory.npz holds the sole pose/anchor result and the targets
    needed for geometry verification. Checkpoints exist only while work is pending.
    """
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    if output == source_dir or output.is_relative_to(source_dir):
        raise ValueError("Output must be separate from the immutable source bank")
    spec = make_t2_spec(model_path)
    if len(locked_wrist_names(spec.model)) != 6:
        raise ValueError("Expected all six wrists locked")
    source_file = source_dir / resolve_paths(skill, SKILL_REGISTRY[skill])[0]
    db, range_index, stop = _source_range(source_file, start, stop)
    request = dict(
        _settings(spec.model_path, gmr_root, iterations, window_frames),
        source=str(source_file),
        source_sha256=digest(source_file),
        skill=skill,
        range_index=range_index,
        start=start,
        stop=stop,
    )
    report_path, result_path = output / "report.json", output / "trajectory.npz"
    previous = _previous_report(report_path, request)
    if (
        previous
        and previous["status"] == "kinematic_candidate"
        and result_path.exists()
    ):
        return previous
    output.mkdir(parents=True, exist_ok=True)
    report = dict(request=request, status="running")
    write_json(report_path, report)
    problem, seed = _prepare_problem(
        db, spec, source_dir, output, skill, range_index, start, stop, gmr_root
    )
    checkpoint = output / "window_checkpoint.npz"
    poses, anchors, history = solve_trajectory_windows(
        problem,
        seed,
        iterations,
        window_frames,
        state=load_window_state(checkpoint, seed) if previous else None,
        on_window=lambda state: save_window_state(checkpoint, state),
    )
    metrics = trajectory_metrics(problem, poses, anchors)
    accepted = trajectory_admissible(poses, metrics)
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        qpos=poses,
        anchors=anchors,
        fps=float(DATABASE_FPS),
        desired_positions=problem.target_positions,
        desired_rotations=problem.target_rotations,
    )
    temporary.replace(result_path)
    report.update(
        status="kinematic_candidate" if accepted else "rejected",
        metrics=metrics,
        history=history,
    )
    write_json(report_path, report)
    if not accepted:
        raise RuntimeError(
            "Trajectory violates contact/geometry constraints; see report.json"
        )
    checkpoint.unlink(missing_ok=True)
    return report


def _range_summary(report):
    request, history = report["request"], report["history"]
    return dict(
        **{key: request[key] for key in ("skill", "range_index", "start", "stop")},
        fps=DATABASE_FPS,
        status=report["status"],
        metrics=report["metrics"],
        iterations=len(history),
        solver_termination=history[-1].get("termination", "windowed")
        if history
        else "no_iterations",
    )


def _check_bin_roundtrip(path, poses, spec, starts, stops):
    """Check the serialized skeleton, root pose, joint limits and physical locks."""
    check = MotionMatchingDatabase()
    check.load_from_file(str(path))
    np.testing.assert_array_equal(check.range_starts, starts)
    np.testing.assert_array_equal(check.range_stops, stops)
    np.testing.assert_array_equal(check.bone_parents, spec.bone_parents)
    if check.bone_positions.shape != (len(poses), spec.num_bones, 3):
        raise ValueError("Output skeleton shape mismatch")
    reconstructed = spec.to_mujoco_qpos(
        spec.decode_bones_to_qpos(check.bone_positions, check.bone_rotations)
    )
    position_error = float(np.max(abs(reconstructed[:, :3] - poses[:, :3])))
    orientation_error = np.minimum(
        np.linalg.norm(reconstructed[:, 3:7] - poses[:, 3:7], axis=1),
        np.linalg.norm(reconstructed[:, 3:7] + poses[:, 3:7], axis=1),
    )
    if position_error > 1e-4 or orientation_error.max() > 1e-5:
        raise ValueError("BIN root pose round-trip failed")
    np.testing.assert_allclose(reconstructed[:, 7:], poses[:, 7:], atol=1e-5)
    limits = spec.model.jnt_range[spec.joint_ids]
    angles = poses[:, spec.joint_qpos_indices]
    limit_violation = float(
        np.maximum(np.maximum(limits[:, 0] - angles, angles - limits[:, 1]), 0.0).max()
    )
    wrists = locked_wrist_names(spec.model)
    columns = [spec.model.jnt_qposadr[spec.model.joint(n).id] for n in wrists]
    wrist_error = float(abs(poses[:, columns]).max())
    if limit_violation > 1e-5 or wrist_error > 1e-6:
        raise ValueError("Joint limits or wrist locks violated")
    return dict(
        roundtrip_root_error_m=position_error,
        max_joint_limit_violation_rad=limit_violation,
        max_wrist_angle_rad=wrist_error,
    )


def export_database(source, destination, jobs, results, request, spec):
    """Publish a BIN only after all its original ranges have passed."""
    db = MotionMatchingDatabase()
    db.load_from_file(str(source))
    parts = [[] for _ in range(5)]
    clips, reports = [], []
    for job in jobs:
        result = results[job["id"]]
        with np.load(
            Path(job["output"]) / "trajectory.npz", allow_pickle=False
        ) as saved:
            qpos = saved["qpos"]
        phases = load_contact_phases(Path(job["output"]) / "phases.json")
        contacts = foot_contact_labels(phases, len(qpos), spec)
        encoded = encode_range(
            qpos, db, job["start"], job["stop"], 1.0, spec, contacts=contacts
        )
        for target, values in zip(parts, encoded):
            target.append(values)
        clips.append(qpos)
        reports.append(_range_summary(result))
    arrays = [np.concatenate(part) for part in parts]
    poses = np.concatenate(clips)
    if len(poses) != len(db.bone_positions):
        raise ValueError(
            "Full source frame coverage is required before BIN publication"
        )
    # The temporary filename cannot be mistaken for a released database by the viewer.
    temporary = destination.with_suffix(".candidate")
    write_bin(temporary, arrays, spec, db.range_starts, db.range_stops)
    validation = _check_bin_roundtrip(
        temporary, poses, spec, db.range_starts, db.range_stops
    )
    report = dict(
        **validation,
        source=str(source),
        source_sha256=request["sources"][source.name],
        output_sha256=digest(temporary),
        frames=len(poses),
        fps=60,
        range_starts=db.range_starts.tolist(),
        range_stops=db.range_stops.tolist(),
        scale=1.0,
        locked_wrists=locked_wrist_names(spec.model),
        max_scene_penetration_m=max(r["metrics"]["penetration"] for r in reports),
        max_contact_error_m=max(r["metrics"]["contact"] for r in reports),
        max_support_speed_m_s=max(r["metrics"]["support_speed_max"] for r in reports),
        range_reports=reports,
        status="kinematic_candidate",
    )
    metadata = dict(
        robot=spec.metadata(),
        database_fps=60,
        num_frames=len(poses),
        range_starts=db.range_starts.tolist(),
        range_stops=db.range_stops.tolist(),
        source=str(source),
        scale=1.0,
        locked_wrists=locked_wrist_names(spec.model),
    )
    write_json(destination.with_suffix(".json"), metadata)
    temporary.replace(destination)
    return report


def _plan_jobs(source_files, work):
    jobs: list[RangeJob] = []
    jobs_by_file: dict[str, list[RangeJob]] = {}
    expected_frames = 0
    for source in source_files:
        candidates = [
            skill
            for skill, cfg in SKILL_REGISTRY.items()
            if resolve_paths(skill, cfg)[0] == source.name
        ]
        if not candidates:
            raise ValueError("No registered terrain mapping for " + source.name)
        skill = candidates[0]
        db = MotionMatchingDatabase()
        db.load_from_file(str(source))
        starts, stops = db.range_starts.astype(int), db.range_stops.astype(int)
        if (
            len(db.bone_parents) != 31
            or starts[0] != 0
            or stops[-1] != len(db.bone_positions)
            or not np.array_equal(starts[1:], stops[:-1])
        ):
            raise ValueError(
                "Source ranges must cover every frame exactly: " + source.name
            )
        terrain_list = SKILL_REGISTRY[skill].get("terrains", [])
        if terrain_list and len(terrain_list) != len(starts):
            raise ValueError("Terrain count must match source ranges: " + source.name)
        group = []
        for index, (start, stop) in enumerate(zip(starts, stops)):
            job: RangeJob = dict(
                id=f"{source.stem}/range_{index:02d}",
                file=source.name,
                skill=skill,
                range_index=index,
                start=int(start),
                stop=int(stop),
                output=str(work / source.stem / f"range_{index:02d}"),
            )
            jobs.append(job)
            group.append(job)
        jobs_by_file[source.name] = group
        expected_frames += len(db.bone_positions)
    return jobs, jobs_by_file, expected_frames


def _run_range(job: RangeJob, spec, source_dir, gmr_root, iterations, window_frames):
    """Each worker owns its report and resumes its own accepted result or checkpoint."""
    folder = Path(job["output"])
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "retargeting",
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(folder),
        "--model",
        str(spec.model_path),
        "--gmr-root",
        str(Path(gmr_root).resolve()),
        "--skill",
        job["skill"],
        "--start",
        str(job["start"]),
        "--stop",
        str(job["stop"]),
        "--iterations",
        str(iterations),
        "--window-frames",
        str(window_frames),
    ]
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    with (folder / "worker.log").open("a") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True
        )
    return json.loads((folder / "report.json").read_text())


def _copy_bank_assets(source_dir, output_dir, terrains):
    for relative in terrains:
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / relative, target)
    registry = {
        skill: dict(
            cfg, bin=resolve_paths(skill, cfg)[0], features=resolve_paths(skill, cfg)[1]
        )
        for skill, cfg in SKILL_REGISTRY.items()
    }
    write_json(output_dir / "registry.json", registry)


def convert_trajectory_bank(
    source_dir,
    output_dir,
    model_path,
    gmr_root,
    iterations=DEFAULT_ITERATIONS,
    workers=2,
    window_frames=DEFAULT_WINDOW_FRAMES,
):
    """Retarget every source range and export each BIN only when its ranges pass."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("Output bank must be separate from the immutable source bank")
    source_files = sorted(source_dir.glob("database_g1_*.bin"))
    if not source_files:
        raise ValueError("No G1 databases found")
    if workers < 1:
        raise ValueError("Worker count must be positive")
    spec = make_t2_spec(model_path)
    if len(locked_wrist_names(spec.model)) != 6:
        raise ValueError("Expected all six wrists locked")
    jobs, groups, expected_frames = _plan_jobs(
        source_files, output_dir.with_name(output_dir.name + "_work")
    )
    request = dict(
        _settings(spec.model_path, gmr_root, iterations, window_frames),
        source_dir=str(source_dir),
        sources={p.name: digest(p) for p in source_files},
        terrains={
            str(p.relative_to(source_dir)): digest(p)
            for p in sorted((source_dir / "terrain").rglob("*"))
            if p.is_file()
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".bank.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another bank conversion owns this output directory"
            ) from None
        report_path = output_dir / "bank_report.json"
        _previous_report(report_path, request)
        _copy_bank_assets(source_dir, output_dir, request["terrains"])
        reports, results, failures = {}, {}, {}
        began = time.monotonic()

        def summary(status):
            return dict(
                request=request,
                status=status,
                reports=list(reports.values()),
                databases=len(reports),
                frames=sum(r["frames"] for r in reports.values()),
                ranges=sum(len(r["range_starts"]) for r in reports.values()),
                expected_databases=len(source_files),
                expected_frames=expected_frames,
                expected_ranges=len(jobs),
                completed_ranges=len(results),
                failed_ranges=failures,
                elapsed_seconds=time.monotonic() - began,
            )

        write_json(report_path, summary("running"))
        ordered_jobs = sorted(jobs, key=lambda job: job["stop"] - job["start"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_range,
                    job,
                    spec,
                    source_dir,
                    gmr_root,
                    iterations,
                    window_frames,
                ): job
                for job in ordered_jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    results[job["id"]] = future.result()
                    group = groups[job["file"]]
                    if all(item["id"] in results for item in group):
                        reports[job["file"]] = export_database(
                            source_dir / job["file"],
                            output_dir / job["file"],
                            group,
                            results,
                            request,
                            spec,
                        )
                        for skill, cfg in SKILL_REGISTRY.items():
                            binary, feature = resolve_paths(skill, cfg)
                            if binary == job["file"]:
                                (output_dir / feature).unlink(missing_ok=True)
                        print("Published:", job["file"], flush=True)
                except Exception as error:
                    failures[job["id"]] = (
                        f"{error}; see {Path(job['output']) / 'worker.log'}"
                    )
                write_json(report_path, summary("running"))
        result = summary(
            "complete"
            if len(reports) == len(source_files) and not failures
            else "incomplete"
        )
        write_json(report_path, result)
        if result["status"] != "complete":
            raise RuntimeError(
                f"Bank incomplete: {len(reports)}/{len(source_files)} BINs; "
                "see bank_report.json"
            )
        return result
