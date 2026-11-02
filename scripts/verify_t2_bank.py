"""Verify complete source coverage, native playback and matching of a T2 bank."""
import argparse
import json
from pathlib import Path

import numpy as np
from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.motion_visualizer import load_motion_data
from motion_matching.retarget_g1_to_t2 import digest, locked_wrist_names
from motion_matching.robots.t2 import make_t2_spec
from motion_matching.run import MotionMatchingSystem, generate_segment
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY


def require(condition, message):
    """Validation must stay enabled under python -O."""
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.bank / 'bank_report.json').read_text())
    spec = make_t2_spec(summary['model'])
    require(digest(spec.model_path) == summary['model_sha256'], 'Model hash mismatch')
    expected = sorted(p.name for p in args.source.glob('database_g1_*.bin'))
    actual = sorted(p.name for p in args.bank.glob('database_g1_*.bin'))
    require(expected == actual and len(actual) == 19, 'Full official 19-BIN coverage required')
    registry = json.loads((args.bank / 'registry.json').read_text())
    require(set(registry) == set(SKILL_REGISTRY), 'Incomplete skill registry')
    source_terrains = sorted(p.relative_to(args.source) for p in (args.source / 'terrain').rglob('*') if p.is_file())
    target_terrains = sorted(p.relative_to(args.bank) for p in (args.bank / 'terrain').rglob('*') if p.is_file())
    require(source_terrains == target_terrains, 'Terrain file coverage mismatch')
    require(len(source_terrains) == 64, 'Full official 64-file terrain coverage required')
    for relative in source_terrains:
        source_path, target_path = args.source / relative, args.bank / relative
        if relative.suffix == '.json':
            before, after = json.loads(source_path.read_text()), json.loads(target_path.read_text())
            for key in ('pos', 'size'):
                if key in before:
                    np.testing.assert_allclose(after[key], np.asarray(before[key]) * summary['scale'])
            if 'quat' in before:
                np.testing.assert_array_equal(after['quat'], before['quat'])
        elif relative.suffix == '.obj':
            def vertices(path):
                return np.array([[float(v) for v in line.split()[1:4]]
                                 for line in path.read_text().splitlines() if line.startswith('v ')])
            np.testing.assert_allclose(vertices(target_path), vertices(source_path) * summary['scale'])
    files = []
    wrist_columns = [i for i, n in enumerate(spec.joint_names) if n in locked_wrist_names(spec.model)]
    require(len(wrist_columns) == 6, 'All six wrist joints must be locked')
    for name in expected:
        source, target = args.source / name, args.bank / name
        report = json.loads(target.with_suffix('.report.json').read_text())
        for key in ('gmr_commit', 'model_sha256', 'adapter_sha256', 'batch_sha256'):
            require(report[key] == summary[key], f'Mixed bank revisions: {name}')
        require(digest(source) == report['source_sha256'], f'Source hash mismatch: {name}')
        require(digest(target) == report['output_sha256'], f'Output hash mismatch: {name}')
        before, after = MotionMatchingDatabase(), MotionMatchingDatabase()
        before.load_from_file(str(source)); after.load_from_file(str(target))
        np.testing.assert_array_equal(before.range_starts, after.range_starts)
        np.testing.assert_array_equal(before.range_stops, after.range_stops)
        require(len(before.bone_positions) == len(after.bone_positions), f'Frame count mismatch: {name}')
        np.testing.assert_array_equal(after.bone_parents, spec.bone_parents)
        for array in (after.bone_positions, after.bone_rotations, after.bone_velocities, after.bone_angular_velocities):
            require(np.isfinite(array).all(), f'Nonfinite database array: {name}')
        for i in range(len(after.range_starts)):
            joints, _, _, fps, _ = load_motion_data(target, robot_spec=spec, range_index=i)
            require(fps == 60, f'Unexpected frame rate: {name}')
            np.testing.assert_allclose(joints[:, wrist_columns], 0., atol=1e-6)
            limits = spec.model.jnt_range[spec.joint_ids]
            require(np.max(np.maximum(limits[:, 0]-joints, joints-limits[:, 1])) < 1e-5,
                    f'Joint limit exceeded: {name}')
        files.append(dict(name=name, frames=len(after.bone_positions), ranges=len(after.range_starts)))
    require(summary['databases'] == len(files)
            and summary['frames'] == sum(f['frames'] for f in files)
            and summary['ranges'] == sum(f['ranges'] for f in files), 'Bank summary totals mismatch')
    # Exercise the exact registry, including every traversal's search ranges and terrain.
    system = MotionMatchingSystem(robot_spec=spec, registry=registry, database_dir=args.bank)
    for db in [system.db, *system.traversal_dbs.values()]:
        require(np.isfinite(db.features).all(), 'Nonfinite motion matching features')
        require(db.bone_positions.shape[1] == 33, 'Unexpected matching skeleton')
    qpos, _ = generate_segment(system, speed=.8, turn_angle_degrees=0., duration_s=2., vel_cmd_idx=1)
    require(qpos.shape == (120, 38) and np.isfinite(qpos).all(), 'Invalid matching output')
    np.testing.assert_allclose(qpos[:, 7:][:, wrist_columns], 0., atol=1e-5)
    result = dict(databases=len(files), frames=sum(d['frames'] for d in files),
                  ranges=sum(d['ranges'] for d in files), files=files,
                  wrist_lock_verified=True, full_registry_loaded=True,
                  matching_smoke_frames=len(qpos), terrain_scale=summary['scale'])
    result['terrain_files_verified'] = len(source_terrains)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
