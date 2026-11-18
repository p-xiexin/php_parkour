"""Record matching choices and optimize the resulting T2 trajectory."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import BankContactSource
from .solver import generate_transition


class T2Postprocessor:
    """Record each matching step, then return a constrained PHP qpos array."""

    def __init__(self, spec, bank_dir, contact_bundle_dir, iterations=80):
        self.spec = spec
        self.bank_dir = Path(bank_dir)
        self.contact_bundle_dir = Path(contact_bundle_dir)
        self.iterations = iterations
        self.records, self._seeds = [], []

    def record(self, system, curr_db):
        if curr_db is system.db:
            source_frame, db_key = int(system.frame_idx), "locomotion"
        else:
            source_frame, db_key = int(system.traversal_curr_frame), system.motion_state
        state = system.inertializer
        rotation = (
            Rotation.from_quat(state.transition_dst_rotation, scalar_first=True)
            * Rotation.from_quat(state.transition_src_rotation, scalar_first=True).inv()
        )
        translation = np.asarray(state.transition_dst_position, dtype=float) - rotation.apply(
            state.transition_src_position
        )
        # Per-frame support-foot correction belongs only to the seed; it is
        # already absorbed into dst when a later matching transition occurs.
        translation[2] += float(system.curr_ground_height - curr_db.search_start_ground_height)
        seed = self.spec.decode_bones_to_qpos(system.bone_positions, system.bone_rotations)
        # Interpret legacy float32 FK rotations on SO(3), as for the targets.
        seed[:4] = Rotation.from_quat(seed[:4], scalar_first=True).as_quat(scalar_first=True)
        self.records.append(
            dict(
                db_key=db_key,
                source_frame=source_frame,
                transform_quat=rotation.as_quat(scalar_first=True),
                transform_pos=translation,
                motion_state=system.motion_state,
                stopped=bool(system.stopped),
                desired_velocity=np.asarray(system.desired_velocity, dtype=float).copy(),
            )
        )
        self._seeds.append(seed.copy())

    def process(self, reference_boxes):
        if len(self.records) < 4:
            raise ValueError("Whole-path generation requires at least four recorded frames")
        contacts = BankContactSource(self.bank_dir, self.contact_bundle_dir, self.spec).build(
            self.records, reference_boxes
        )
        target = contacts.target_qpos
        last = self.records[-1]
        stopping = np.linalg.norm(last["desired_velocity"]) == 0.0
        if stopping and (not last["stopped"] or last["motion_state"] != "locomotion"):
            raise ValueError("Final stop command has not completed in the recorded matching path")
        fixed = {frame: target[frame].copy() for frame in (0, 1, len(target) - 2, len(target) - 1)}
        if stopping:
            # Prescribe zero terminal velocity inside the solve.
            fixed[len(target) - 2] = target[-1].copy()
        output = generate_transition(
            self.spec,
            reference_boxes,
            target,
            initial_qpos=np.asarray(self._seeds),
            phases=contacts.phases,
            fixed_poses=fixed,
            fixed_anchors=contacts.fixed_anchors,
            iterations=self.iterations,
        )
        moving = np.linalg.norm([record["desired_velocity"] for record in self.records], axis=1) > 0
        distance = np.linalg.norm(np.diff(output[:, 4:6], axis=0), axis=1)
        if moving.any() and distance[moving[1:]].sum() <= 1e-6:
            raise ValueError("Generated path did not move during its motion commands")
        return output
