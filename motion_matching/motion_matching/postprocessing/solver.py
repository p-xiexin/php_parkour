"""Generate a T2 connection directly with trajectory and contact constraints.

The selected database motion supplies targets, not published output. A caller
provides its contact phases in the matched world scene and fixed boundary
samples. One optimization generates the complete connection; this module does
not search other clips, clip joints, shift the robot, or filter its result.
"""

import mujoco as mj
import numpy as np
import scipy.sparse as sp

from motion_matching.robots.base import locked_wrist_names
from retargeting.trajectory_optimization import TrajectorySQP, trajectory_metrics

from .validation import MotionConstraintVerifier


class _BoundaryTrajectory(TrajectorySQP):
    """Reuse the existing SQP, adding hard fixed samples at either boundary."""

    def __init__(self, *args, fixed_poses, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_poses = fixed_poses
        self.fixed_anchor_values = {i: self.phases[i].anchor.copy() for i in self.fixed_anchors}

    def linearize(self, poses, anchors, radius):
        P, c, A, lower, upper = super().linearize(poses, anchors, radius)
        count = len(self.fixed_poses) * self.nv
        if count:
            # Fixed poses are installed in the problem's initial state. Zero
            # tangent increments keep them fixed during every SQP line search.
            columns = np.concatenate(
                [frame * self.nv + np.arange(self.nv) for frame in self.fixed_poses]
            )
            fixed = sp.coo_matrix(
                (np.ones(count), (np.arange(count), columns)), shape=(count, A.shape[1])
            ).tocsc()
            A = sp.vstack([A, fixed], format="csc")
            lower, upper = np.r_[lower, np.zeros(count)], np.r_[upper, np.zeros(count)]
        return P, c, A, lower, upper

    def boundary_errors(self, poses, anchors):
        delta = np.zeros(self.nv)
        pose_error = 0.0
        for frame, expected in self.fixed_poses.items():
            mj.mj_differentiatePos(self.model, delta, 1.0, expected, poses[frame])
            pose_error = max(pose_error, float(np.max(abs(delta))))
        anchor_error = max(
            (
                float(np.max(abs(anchors[i] - value)))
                for i, value in self.fixed_anchor_values.items()
            ),
            default=0.0,
        )
        return pose_error, anchor_error

    def feasible(self, poses, anchors, metrics):
        return (
            super().feasible(poses, anchors, metrics)
            and max(self.boundary_errors(poses, anchors)) <= 1e-6
        )


def generate_transition(
    spec,
    reference_boxes,
    target_qpos,
    *,
    phases,
    fixed_poses,
    initial_qpos=None,
    fixed_anchors=(),
    fps=60,
    iterations=60,
    body_names=None,
):
    """Return constrained PHP qpos using source poses, contacts and fixed boundaries.

    ContactPhase objects use model body IDs and world anchors. fixed_poses maps
    frame indices to PHP poses; two fixed samples at each end preserve endpoint
    velocity. The optional initial guess does not change the target poses.
    """
    target = np.array(target_qpos, dtype=float, copy=True)
    initial = (
        target.copy() if initial_qpos is None else np.array(initial_qpos, dtype=float, copy=True)
    )
    verifier = MotionConstraintVerifier(spec, reference_boxes)
    model = verifier.model
    prescribed = {}
    for frame, pose in sorted(fixed_poses.items()):
        prescribed[frame] = spec.to_mujoco_qpos(pose)
        initial[frame] = pose
    if body_names is None:
        body_ids = list(dict.fromkeys(int(body) for body in model.jnt_bodyid))
    else:
        body_ids = [model.body(name).id for name in body_names]
    positions, rotations = [], []
    data = mj.MjData(model)
    for pose in spec.to_mujoco_qpos(target):
        data.qpos[:] = pose
        mj.mj_kinematics(model, data)
        positions.append(data.xpos[body_ids].copy())
        rotations.append(data.xquat[body_ids].copy())
    problem = _BoundaryTrajectory(
        model,
        fps,
        body_ids,
        positions,
        rotations,
        phases,
        locked_joint_names=locked_wrist_names(model),
        fixed_anchors=fixed_anchors,
        fixed_poses=prescribed,
    )
    solved, anchors, _ = problem.solve(spec.to_mujoco_qpos(initial), iterations=iterations)
    metrics = trajectory_metrics(problem, solved, anchors)
    output = spec.from_mujoco_qpos(solved)
    violations = verifier.check(output, fps=fps)
    if not problem.feasible(solved, anchors, metrics) or violations:
        raise RuntimeError(f"Constrained T2 trajectory failed: {metrics}; {violations}")
    return output
