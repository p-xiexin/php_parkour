#!/usr/bin/env python3
"""Verify GMR + SQP bank coverage, BIN round-trips and saved solve constraints."""

import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.robots.base import locked_wrist_names
from motion_matching.robots.t2 import make_t2_spec
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY

from retargeting import retarget_database as retarget
from retargeting import trajectory_optimization as solver
from retargeting.retarget_contacts import foot_contact_labels


def require(condition, message):
    """Validation must stay enabled under python -O."""
    if not condition:
        raise ValueError(message)


def _load_solve(work, spec):
    """Read targets and anchors from one result, or from an existing legacy bank."""
    model = mj.MjModel.from_xml_path(str(work / "scene.xml"))
    require(
        model.nq == spec.model.nq
        and [model.joint(i).name for i in range(model.njnt)]
        == [spec.model.joint(i).name for i in range(spec.model.njnt)],
        f"Scene joint layout mismatch: {work}",
    )
    phases = retarget.load_contact_phases(work / "phases.json")
    table = json.loads((work / "initialization.json").read_text())["ik_match_table1"]
    if (work / "trajectory.npz").exists():
        with np.load(work / "trajectory.npz", allow_pickle=False) as saved:
            positions, rotations = (
                saved["desired_positions"],
                saved["desired_rotations"],
            )
            anchors = saved["anchors"]
    else:
        # The accepted bank predates the consolidated result file; it stays read-only.
        with np.load(work / "problem.npz", allow_pickle=False) as saved:
            positions, rotations = (
                saved["desired_positions"],
                saved["desired_rotations"],
            )
        with np.load(work / "diagnostic_trajectory.npz", allow_pickle=False) as saved:
            anchors = saved["anchors"]
    problem = solver.TrajectorySQP(
        model,
        retarget.DATABASE_FPS,
        [model.body(name).id for name in table],
        positions,
        rotations,
        phases,
        locked_wrist_names(model),
        position_weights=[entry[1] / 100 for entry in table.values()],
        orientation_weights=[entry[2] / 10 for entry in table.values()],
    )
    return problem, anchors


def verify_trajectory_range(work, poses, contacts, spec, report):
    """Evaluate physical constraints on the decoded BIN, not on a duplicate NPZ."""
    label = f"{work.parent.name}/{work.name}"
    require(
        report["stop"] - report["start"] == len(poses)
        and report["fps"] == retarget.DATABASE_FPS,
        f"Range layout mismatch: {label}",
    )
    problem, anchors = _load_solve(work, spec)
    require(problem.frames == len(poses), f"Problem frame count mismatch: {label}")
    np.testing.assert_array_equal(
        contacts, foot_contact_labels(problem.phases, len(poses), spec)
    )
    metrics = solver.trajectory_metrics(problem, poses, anchors)
    require(
        problem.feasible(poses, anchors, metrics),
        f"Decoded BIN violates geometry/contact/hard constraints: {label}: {metrics}",
    )
    return dict(
        range_index=report["range_index"],
        start=report["start"],
        stop=report["stop"],
        frames=len(poses),
        feasible=True,
        metrics=metrics,
    )


def _verify_database(bank, report, spec, source, work_dir, geometry):
    name = Path(report["source"]).name
    require(
        retarget.digest(bank / name) == report["output_sha256"],
        f"BIN hash differs: {name}",
    )
    db = MotionMatchingDatabase()
    db.load_from_file(str(bank / name))
    require(len(db.bone_positions) == report["frames"], f"Frame count differs: {name}")
    np.testing.assert_array_equal(db.range_starts, report["range_starts"])
    np.testing.assert_array_equal(db.range_stops, report["range_stops"])
    np.testing.assert_array_equal(db.bone_parents, spec.bone_parents)
    for array in (
        db.bone_positions,
        db.bone_rotations,
        db.bone_velocities,
        db.bone_angular_velocities,
    ):
        require(np.isfinite(array).all(), f"Nonfinite BIN array: {name}")
    require(
        db.contact_states.shape == (report["frames"], 2),
        f"Wrong contact channel count: {name}",
    )
    poses = spec.to_mujoco_qpos(
        spec.decode_bones_to_qpos(db.bone_positions, db.bone_rotations)
    )
    joints = poses[:, spec.joint_qpos_indices]
    wrists = [spec.joint_names.index(name) for name in locked_wrist_names(spec.model)]
    require(
        float(np.max(abs(joints[:, wrists]))) <= 1e-6, f"Unlocked wrist in BIN: {name}"
    )
    limits = spec.model.jnt_range[spec.joint_ids]
    excess = np.maximum(np.maximum(limits[:, 0] - joints, joints - limits[:, 1]), 0.0)
    require(float(excess.max()) <= 1e-5, f"Joint limit exceeded: {name}")
    if source:
        original = Path(source) / name
        require(
            retarget.digest(original) == report["source_sha256"],
            f"G1 source hash differs: {name}",
        )
        before = MotionMatchingDatabase()
        before.load_from_file(str(original))
        np.testing.assert_array_equal(before.range_starts, db.range_starts)
        np.testing.assert_array_equal(before.range_stops, db.range_stops)
        require(
            len(before.bone_positions) == len(db.bone_positions),
            f"G1 frame count differs: {name}",
        )
    ranges = []
    if geometry:
        require(
            len(report["range_reports"]) == len(db.range_starts),
            f"Missing solve reports: {name}",
        )
        for index, (start, stop) in enumerate(zip(db.range_starts, db.range_stops)):
            row = report["range_reports"][index]
            require(
                row["range_index"] == index
                and row["start"] == int(start)
                and row["stop"] == int(stop),
                f"Range identity differs: {name}/{index}",
            )
            ranges.append(
                verify_trajectory_range(
                    work_dir / Path(name).stem / f"range_{index:02d}",
                    poses[start:stop],
                    db.contact_states[start:stop, :2],
                    spec,
                    row,
                )
            )
    return dict(
        name=name, frames=len(poses), ranges=len(db.range_starts), geometry=ranges
    )


def verify(bank, source=None, *, geometry=False, model=None, work_dir=None):
    """Check a complete official bank and optionally re-evaluate saved constraints."""
    bank = Path(bank).resolve()
    summary = json.loads((bank / "bank_report.json").read_text())
    settings = summary.get("request", summary)
    require(
        summary["status"] == "complete" and settings["algorithm"] == "trajectory_sqp",
        "Expected a complete GMR + SQP bank",
    )
    model = Path(model or settings["model"])
    if not model.is_absolute():
        model = bank / model
    require(retarget.digest(model) == settings["model_sha256"], "T2 model hash differs")
    spec = make_t2_spec(model)
    require(len(locked_wrist_names(spec.model)) == 6, "All six wrists must be locked")
    registry = json.loads((bank / "registry.json").read_text())
    require(set(registry) == set(SKILL_REGISTRY), "Incomplete skill registry")
    expected = {Path(report["source"]).name for report in summary["reports"]}
    require(
        len(expected) == len(summary["reports"]) == 19
        and expected == {p.name for p in bank.glob("*.bin")},
        "Expected exactly 19 distinct BIN files",
    )
    terrains = (
        settings["terrains"]
        if "request" in summary
        else json.loads((bank / "bank_request.json").read_text())["terrains"]
    )
    for relative, expected_hash in terrains.items():
        require(
            retarget.digest(bank / relative) == expected_hash,
            f"Terrain hash differs: {relative}",
        )
    work_dir = Path(work_dir) if work_dir else bank.with_name(bank.name + "_work")
    if geometry:
        retarget.trajectory_runtime()
    results = [
        _verify_database(bank, report, spec, source, work_dir, geometry)
        for report in summary["reports"]
    ]
    require(
        sum(r["frames"] for r in results) == summary["frames"]
        and sum(r["ranges"] for r in results) == summary["ranges"],
        "Bank totals differ",
    )
    return dict(
        status="passed",
        algorithm="trajectory_sqp",
        databases=len(results),
        frames=summary["frames"],
        ranges=summary["ranges"],
        model_sha256=settings["model_sha256"],
        gmr_commit=settings["gmr_commit"],
        wrist_and_limits_verified=True,
        geometry_rechecked=geometry,
        files=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="Recompute constraints from the saved SQP solve artifacts",
    )
    parser.add_argument(
        "--work-dir", type=Path, help="Saved solve artifacts; defaults to <bank>_work"
    )
    args = parser.parse_args()
    result = verify(
        args.bank,
        args.source,
        geometry=args.geometry,
        model=args.model,
        work_dir=args.work_dir,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    main()
