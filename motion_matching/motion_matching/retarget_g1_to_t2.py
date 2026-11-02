"""Retarget every frame/range in the official G1 BIN bank to T2 with GMR.

Legacy BIN filenames are retained for skill-registry compatibility. The source
bank is never modified. Use the output bank together with its model and registry.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import sys
import types
import json
from pathlib import Path
import struct
import subprocess
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_matching.core.database import MotionMatchingDatabase
from motion_matching.kinematics import quat
from motion_matching.robots.t2 import make_t2_spec
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY, resolve_paths
from motion_matching.utils import G1_MJCF


def load_gmr(root: Path):
    # Import only upstream's solver and params, bypassing its GUI/torch exports.
    package = types.ModuleType('_php_headless_gmr')
    package.__path__ = [str(root.resolve() / 'general_motion_retargeting')]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__ + '.motion_retarget')


def add_sphere_frames(names, positions, rotations):
    """G1 URDF palm's fixed transform, shared by both sphere-hand links."""
    names = list(names)
    extra_pos, extra_rot = [], []
    for side in ('left', 'right'):
        index = names.index(side + '_wrist_yaw_link')
        r = R.from_quat(rotations[:, index], scalar_first=True)
        extra_pos.append(positions[:, index] + r.apply([.029, -.003, 0.]))
        extra_rot.append((r * R.from_euler('y', 1.57)).as_quat(scalar_first=True))
    return (names + ['left_sphere_hand', 'right_sphere_hand'],
            np.concatenate([positions, np.stack(extra_pos, axis=1)], axis=1),
            np.concatenate([rotations, np.stack(extra_rot, axis=1)], axis=1))


def locked_wrist_names(model):
    """The variant declares physical locks as constant joint equalities."""
    return [model.joint(int(model.eq_obj1id[i])).name for i in range(model.neq)
            if model.eq_type[i] == mj.mjtEq.mjEQ_JOINT
            and model.eq_obj2id[i] == -1
            and '_wrist_' in model.joint(int(model.eq_obj1id[i])).name]


def constrain_locked_wrists(solver):
    import mink
    names = locked_wrist_names(solver.model)
    if names:
        solver.ik_limits.append(mink.VelocityLimit(solver.model, {name: 0. for name in names}))


def make_config(source, target):
    """Calibrate rest orientations, including T2's lateral arm bind pose."""
    sd, td = mj.MjData(source), mj.MjData(target)
    mj.mj_forward(source, sd)
    mj.mj_forward(target, td)
    pairs = [('waist_yaw_link', 'pelvis', 100., 10.), ('trunk', 'torso_link', 20., 10.)]
    for side in ('left', 'right'):
        pairs += [(f'{side}_hip_yaw_link', f'{side}_hip_yaw_link', 10., 1.),
                  (f'{side}_knee_pitch_link', f'{side}_knee_link', 40., 1.),
                  (f'{side}_ankle_roll_link', f'{side}_ankle_roll_link', 100., 10.),
                  # T2's fourth arm joint is the physical elbow flexion.
                  (f'{side}_elbow_yaw_link', f'{side}_elbow_link', 10., 0.),
                  (f'{side}_hand_ee_link', f'{side}_sphere_hand', 100., 0.)]
    def pose(model, data, name):
        idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f'Missing body {name}')
        return data.xpos[idx], R.from_quat(data.xquat[idx], scalar_first=True)
    sp, _ = pose(source, sd, 'pelvis')
    tp, _ = pose(target, td, 'waist_yaw_link')
    sf, _ = pose(source, sd, 'left_ankle_roll_link')
    tf, _ = pose(target, td, 'left_ankle_roll_link')
    scale = float((tp[2] - tf[2]) / (sp[2] - sf[2]))
    table = {}
    for target_name, source_name, pw, rw in pairs:
        if source_name.endswith('_sphere_hand'):
            _, sr = pose(source, sd, source_name.replace('_sphere_hand', '_wrist_yaw_link'))
            sr = sr * R.from_euler('y', 1.57)
        else:
            _, sr = pose(source, sd, source_name)
        _, tr = pose(target, td, target_name)
        offset = (sr.inv() * tr).as_quat(scalar_first=True).tolist()
        position_offset = [0., 0., 0.]
        if target_name == 'trunk':
            sbody, _ = pose(source, sd, source_name)
            tbody, _ = pose(target, td, target_name)
            position_offset = tr.inv().apply((tbody - tp) - scale * (sbody - sp)).tolist()
        table[target_name] = [source_name, pw, rw, position_offset, offset]
    return dict(human_height_assumption=1., human_root_name='pelvis',
                robot_root_name='waist_yaw_link', ground_height=0.,
                human_scale_table={entry[0]: scale for entry in table.values()},
                use_ik_match_table1=True, use_ik_match_table2=False,
                ik_match_table1=table, ik_match_table2={}), scale


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_range(qpos, source_db, start, stop, scale, spec):
    positions, rotations = spec.encode_qpos_to_bones(spec.from_mujoco_qpos(qpos))
    sim_pos = source_db.bone_positions[start:stop, 0].astype(float) * scale
    sim_rot = quat.normalize(source_db.bone_rotations[start:stop, 0].astype(float))
    # Keep original Simulation trajectory and range boundaries, without smoothing again.
    positions[:, 1] = quat.mul_vec(quat.inv(sim_rot), positions[:, 1] - sim_pos)
    rotations[:, 1] = quat.mul(quat.inv(sim_rot), rotations[:, 1])
    positions[:, 0], rotations[:, 0] = sim_pos, sim_rot
    rotations = quat.unroll(rotations)
    velocities = np.gradient(positions, 1./60., axis=0)
    angular = np.zeros_like(positions)
    angular[1:] = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations[1:], rotations[:-1]))) * 60.
    angular[0] = angular[1]
    _, _, gv, _ = quat.fk_vel(lrot=rotations, lpos=positions, lvel=velocities,
                              lang=angular, parents=spec.bone_parents)
    from scipy.ndimage import median_filter
    contact = np.linalg.norm(gv[:, [spec.bone_left_foot, spec.bone_right_foot]], axis=-1) < .5
    for side in range(2):
        contact[:, side] = median_filter(contact[:, side], size=6, mode='nearest')
    return positions, velocities, rotations, angular, contact


def write_bin(path, parts, spec, starts, stops):
    temporary = path.with_suffix('.bin.tmp')
    with temporary.open('wb') as f:
        for array in parts[:4]:
            array = np.asarray(array, dtype=np.float32)
            f.write(struct.pack('II', *array.shape[:2])); f.write(array.tobytes())
        f.write(struct.pack('I', spec.num_bones)); f.write(spec.bone_parents.astype(np.int32).tobytes())
        for array in (starts, stops):
            f.write(struct.pack('I', len(array))); f.write(np.asarray(array, dtype=np.int32).tobytes())
        contacts = np.asarray(parts[4], dtype=np.uint8)
        f.write(struct.pack('II', *contacts.shape)); f.write(contacts.tobytes())
    temporary.replace(path)


def convert_bank(source_dir, output_dir, model_path, gmr_root):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError('Output bank must be separate from immutable source bank')
    source_files = sorted(source_dir.glob('database_g1_*.bin'))
    if not source_files:
        raise ValueError(f'No G1 databases found under {source_dir}')
    spec = make_t2_spec(model_path)
    if len(locked_wrist_names(spec.model)) != 6 or any(
        mj.mj_name2id(spec.model, mj.mjtObj.mjOBJ_GEOM, side + '_g1_sphere_visual') < 0
        for side in ('left', 'right')
    ):
        raise ValueError('Expected the T2 sphere-hand model with all six wrists locked')
    output_dir.mkdir(parents=True, exist_ok=True)
    source_model = mj.MjModel.from_xml_path(str(G1_MJCF))
    target_data = mj.MjData(spec.model)
    # BIN skeleton is Simulation + pelvis + 29 hinge bodies. Use its FK directly;
    # projecting interpolated BIN rotations onto G1 hinges loses source motion.
    source_names = [source_model.body(int(b)).name for b in source_model.jnt_bodyid]
    config, scale = make_config(source_model, spec.model)
    config_path = output_dir / 'gmr_ik.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    gmr = load_gmr(gmr_root)
    gmr.ROBOT_XML_DICT['php_t2_bank'] = spec.model_path
    gmr.IK_CONFIG_DICT['php_g1_bank'] = {'php_t2_bank': config_path}
    solver = gmr.GeneralMotionRetargeting('php_g1_bank', 'php_t2_bank', verbose=False)
    solver.model.opt.timestep = 1./60.
    constrain_locked_wrists(solver)
    required = list(config['human_scale_table'])
    locked = locked_wrist_names(spec.model)
    wrist_indices = [int(spec.model.jnt_qposadr[spec.model.joint(n).id]) for n in locked]
    ground = spec.model.geom('ground').id
    revision = subprocess.check_output(['git','-C',str(gmr_root),'rev-parse','HEAD'],text=True).strip()
    # Preserve the report schema; adapter and batch now share one implementation.
    signature = dict(gmr_commit=revision, model_sha256=digest(spec.model_path),
                     adapter_sha256=digest(__file__),
                     batch_sha256=digest(__file__))
    reports = []
    for path in source_files:
        began = time.monotonic()
        destination = output_dir / path.name
        report_path = destination.with_suffix('.report.json')
        source_hash = digest(path)
        if report_path.exists() and destination.exists():
            previous = json.loads(report_path.read_text())
            if (all(previous.get(k) == v for k,v in signature.items())
                and previous.get('source_sha256') == source_hash
                and previous.get('output_sha256') == digest(destination)):
                reports.append(previous); print('Verified existing:',path.name,flush=True);continue
        db = MotionMatchingDatabase(); db.load_from_file(str(path))
        if len(db.bone_parents) != 31:
            raise ValueError('Source is not a G1 database: '+str(path))
        parts = [[] for _ in range(5)]; clips=[]; ground_pen=[]; self_pen=[]
        pos_error = {name: [] for name in config['ik_match_table1']}
        for range_index, (start,stop) in enumerate(zip(db.range_starts, db.range_stops)):
            start,stop=int(start),int(stop)
            world_rot, world_pos = quat.fk(db.bone_rotations[start:stop],
                                           db.bone_positions[start:stop], db.bone_parents)
            names, positions, rotations = add_sphere_frames(source_names, world_pos[:,1:], world_rot[:,1:])
            source_indices = [names.index(name) for name in required]
            # Reset GMR at each independent clip (including mirrored clips).
            initial = solver.model.qpos0.copy()
            initial[:3] = world_pos[0,1] * scale
            initial[3:7] = world_rot[0,1]
            solver.configuration.update(initial)
            clip=[]
            for source_pos, source_rot in zip(positions, rotations):
                frame={name:(source_pos[i].copy(),source_rot[i].copy()) for name,i in zip(required,source_indices)}
                result=solver.retarget(frame,offset_to_ground=False)
                if not np.isfinite(result).all():raise RuntimeError('Nonfinite IK output')
                if wrist_indices and np.max(np.abs(result[wrist_indices])) > 1e-4:
                    raise RuntimeError(f'Wrist lock constraint violated: {result[wrist_indices]}')
                # Remove solver roundoff from locked channels, which stay zero on playback.
                result[wrist_indices]=0.
                solver.configuration.update(result)
                clip.append(result)
                target_data.qpos[:]=result; mj.mj_forward(spec.model,target_data)
                for target,entry in config['ik_match_table1'].items():
                    pos_error[target].append(float(np.linalg.norm(target_data.body(target).xpos-solver.scaled_human_data[entry[0]][0])))
                gp,sp=0.,0.
                for contact in target_data.contact:
                    if ground in (contact.geom1,contact.geom2):gp=max(gp,-float(contact.dist))
                    else:sp=max(sp,-float(contact.dist))
                ground_pen.append(gp);self_pen.append(sp)
            clip=np.asarray(clip);clips.append(clip)
            processed=encode_range(clip,db,start,stop,scale,spec)
            for bank,values in zip(parts,processed):bank.append(values)
            print(f'{path.name}: range {range_index+1}/{len(db.range_starts)} ({stop-start} frames)',flush=True)
        arrays=[np.concatenate(p) for p in parts]
        write_bin(destination,arrays,spec,db.range_starts,db.range_stops)
        # Feature files from an earlier model/configuration must not survive a rebuild.
        for skill, cfg in SKILL_REGISTRY.items():
            binary, feature = resolve_paths(skill, cfg)
            if binary == path.name:
                (output_dir / feature).unlink(missing_ok=True)
        qpos=np.concatenate(clips)
        check=MotionMatchingDatabase();check.load_from_file(str(destination))
        np.testing.assert_array_equal(check.range_starts,db.range_starts)
        np.testing.assert_array_equal(check.range_stops,db.range_stops)
        if check.bone_positions.shape != (len(qpos),33,3):raise RuntimeError('Output skeleton shape mismatch')
        decoded=spec.decode_bones_to_qpos(check.bone_positions,check.bone_rotations)
        reconstructed=spec.to_mujoco_qpos(decoded)
        position_error=float(np.max(np.abs(reconstructed[:,:3]-qpos[:,:3])))
        if position_error>1e-4:raise RuntimeError('Database world position round-trip failed')
        np.testing.assert_allclose(reconstructed[:,7:],qpos[:,7:],atol=1e-5)
        orientation_error=np.minimum(np.linalg.norm(reconstructed[:,3:7]-qpos[:,3:7],axis=-1),
                                     np.linalg.norm(reconstructed[:,3:7]+qpos[:,3:7],axis=-1))
        if orientation_error.max()>1e-5:raise RuntimeError('Database root rotation round-trip failed')
        bounds=spec.model.jnt_range[spec.joint_ids];joints=qpos[:,spec.joint_qpos_indices]
        limit=float(np.max(np.maximum(bounds[:,0]-joints,joints-bounds[:,1]).clip(0)))
        if limit>1e-6:raise RuntimeError('Joint limit exceeded')
        report=dict(**signature,source=str(path),source_sha256=source_hash,
                    output_sha256=digest(destination),frames=len(qpos),fps=60,
                    range_starts=db.range_starts.tolist(),range_stops=db.range_stops.tolist(),
                    scale=scale,locked_wrists=locked,max_wrist_angle_rad=float(np.max(np.abs(qpos[:,wrist_indices]))),
                    max_joint_limit_violation_rad=limit,roundtrip_root_error_m=position_error,
                    mean_body_position_error_m={k:float(np.mean(v)) for k,v in pos_error.items()},
                    max_flat_ground_penetration_m=max(ground_pen),max_self_penetration_m=max(self_pen),
                    elapsed_seconds=time.monotonic()-began)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        metadata=dict(robot=spec.metadata(),database_fps=60,num_frames=len(qpos),
                      range_starts=db.range_starts.tolist(),range_stops=db.range_stops.tolist(),
                      source=str(path),scale=scale,locked_wrists=locked)
        destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        reports.append(report)
        print('Completed:',path.name,flush=True)
    # Scale OBJ vertices and box metadata with exactly the same world transform.
    for path in (source_dir/'terrain').rglob('*'):
        if not path.is_file():continue
        target=output_dir/path.relative_to(source_dir);target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.obj':
            lines=[]
            for line in path.read_text().splitlines():
                if line.startswith('v '):line='v '+' '.join(str(float(v)*scale) for v in line.split()[1:4])
                lines.append(line)
            target.write_text('\n'.join(lines)+'\n')
        elif path.suffix=='.json':
            d=json.loads(path.read_text())
            for key in ('pos','size'):
                if key in d:d[key]=(np.array(d[key])*scale).tolist()
            target.write_text(json.dumps(d,indent=2)+'\n')
        else:target.write_bytes(path.read_bytes())
    registry=json.loads(json.dumps(SKILL_REGISTRY))
    for name,cfg in registry.items():
        cfg['bin'],cfg['features']=resolve_paths(name,cfg)
        if 'height_range' in cfg:cfg['height_range']=[v*scale for v in cfg['height_range']]
        if 'start_ground_fit_distance' in cfg:cfg['start_ground_fit_distance']*=scale
    (output_dir/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    summary=dict(model=str(spec.model_path),**signature,databases=len(reports),
                 frames=sum(r['frames'] for r in reports),ranges=sum(len(r['range_starts']) for r in reports),
                 scale=scale,locked_wrists=locked,reports=reports,
                 limitations=['GMR kinematic retargeting; contacts and dynamics require further optimization.'])
    (output_dir/'bank_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Bank complete:',summary['databases'],'databases',summary['frames'],'frames',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--gmr-root',type=Path,required=True)
    args=parser.parse_args()
    convert_bank(args.source_dir,args.output_dir,args.model,args.gmr_root)


if __name__=='__main__':main()
