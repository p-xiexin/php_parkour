"""Contact-constrained SQP and bounded-memory trajectory windows.

This module only operates on models and arrays. File IO and resume policy belong
in retarget_database; all objective weights and physical tolerances are unchanged.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import mujoco as mj
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

from retargeting.retarget_contacts import orient_native_contacts, support_gap


def _append_sparse_row(rows, cols, values, row, column, vector):
    nz = np.flatnonzero(abs(vector) > 1e-12)
    rows.extend([row] * len(nz))
    cols.extend(column + nz)
    values.extend(vector[nz])


class TrajectorySQP:
    """Optimize every frame and every phase anchor in one sparse problem.

    The input is a target robot, desired world-space body poses, and source
    contact observations. Initial poses are only a starting guess. Both contact
    anchors and the complete pose trajectory are decision variables.
    """

    def __init__(
        self,
        model,
        fps,
        body_ids,
        target_positions,
        target_rotations,
        phases,
        locked_joint_names=(),
        position_sigma=0.05,
        orientation_sigma=0.3,
        acceleration_sigma=80.0,
        position_weights=None,
        orientation_weights=None,
        fixed_prefix=0,
        fixed_anchors=(),
    ):
        self.model = model
        self.model.opt.ccd_tolerance = min(self.model.opt.ccd_tolerance, 1e-9)
        self.fixed_prefix = fixed_prefix
        self.fixed_anchors = set(fixed_anchors)
        if model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE or np.any(
            model.jnt_type[1:] != mj.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError("Expected one floating root followed by scalar hinges")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Frame rate must be positive")
        if len(target_positions) < 3:
            raise ValueError("Trajectory needs at least three frames")
        self.dt = 1.0 / fps
        self.body_ids = list(body_ids)
        self.target_positions = np.asarray(target_positions)
        self.target_rotations = np.asarray(target_rotations)
        self.phases = phases
        self.position_sigma = position_sigma
        self.orientation_sigma = orientation_sigma
        self.position_weights = (
            np.ones(len(body_ids))
            if position_weights is None
            else np.asarray(position_weights)
        )
        self.orientation_weights = (
            np.ones(len(body_ids))
            if orientation_weights is None
            else np.asarray(orientation_weights)
        )
        self.acceleration_sigma = acceleration_sigma
        self.penalty = 2e3
        self.locked = {model.joint(name).id for name in locked_joint_names}
        self.frames = len(target_positions)
        self.nv = model.nv
        self.pose_size = self.frames * self.nv
        self.anchor_size = 3 * len(phases)
        self.base_size = self.pose_size + self.anchor_size
        self.data = mj.MjData(model)
        self.frame_phases = [[] for _ in range(self.frames)]
        for index, phase in enumerate(phases):
            for frame in range(phase.start, phase.stop):
                self.frame_phases[frame].append(index)

    def forward(self, pose):
        self.data.qpos[:] = pose
        mj.mj_forward(self.model, self.data)

    def point(self, body, local):
        return self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ local

    def jacobian(self, body, point):
        jp, jr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        mj.mj_jac(self.model, self.data, jp, jr, point, body)
        return jp, jr

    def velocities(self, poses):
        result = np.zeros((len(poses) - 1, self.nv))
        for i in range(len(result)):
            mj.mj_differentiatePos(
                self.model, result[i], self.dt, poses[i], poses[i + 1]
            )
        return result

    def acceleration_jacobian(self, poses):
        """Differentiate neighbouring SO(3) logarithms in their own tangents."""
        velocity = self.velocities(poses)
        edges = []
        eps = 1e-6
        for t in range(self.frames - 1):
            blocks = [-np.eye(self.nv) / self.dt, np.eye(self.nv) / self.dt]
            for endpoint in range(2):
                for axis in range(3, 6):
                    a, b = poses[t].copy(), poses[t + 1].copy()
                    delta = np.zeros(self.nv)
                    delta[axis] = eps
                    mj.mj_integratePos(
                        self.model, a if endpoint == 0 else b, delta, 1.0
                    )
                    perturbed = np.zeros(self.nv)
                    mj.mj_differentiatePos(self.model, perturbed, self.dt, a, b)
                    blocks[endpoint][:, axis] = (perturbed - velocity[t]) / eps
            edges.append(blocks)
        rows = []
        cols = []
        values = []
        for t in range(self.frames - 2):
            blocks = [-edges[t][0], edges[t + 1][0] - edges[t][1], edges[t + 1][1]]
            for offset, block in enumerate(blocks):
                r, c = np.nonzero(abs(block) > 1e-10)
                rows.extend(t * self.nv + r)
                cols.extend((t + offset) * self.nv + c)
                values.extend(block[r, c] / (self.dt * self.acceleration_sigma))
        return sp.coo_matrix(
            (values, (rows, cols)), shape=((self.frames - 2) * self.nv, self.base_size)
        ).tocsr()

    def collisions(self, nearby=False):
        actual = [
            (
                int(c.geom1),
                int(c.geom2),
                c.pos.copy(),
                c.frame[:3].copy(),
                float(c.dist),
            )
            for c in self.data.contact
        ]
        actual = orient_native_contacts(self.model, self.data, actual)
        if not nearby:
            return actual
        existing = {tuple(sorted((c[0], c[1]))) for c in actual}
        margin = self.model.geom_margin.copy()
        try:
            self.model.geom_margin[:] = np.maximum(margin, 0.025)
            mj.mj_collision(self.model, self.data)
            pairs = {
                tuple(sorted((int(c.geom1), int(c.geom2)))) for c in self.data.contact
            }
        finally:
            self.model.geom_margin[:] = margin
            mj.mj_collision(self.model, self.data)
        for a, b in sorted(pairs - existing):
            witness = np.zeros(6)
            distance = float(
                mj.mj_geomDistance(self.model, self.data, a, b, 0.025, witness)
            )
            vector = witness[3:] - witness[:3]
            length = np.linalg.norm(vector)
            if distance >= 0.025:
                continue
            if distance > 1e-8 and length > 1e-12:
                normal = vector / length
            else:
                direction = self.data.geom_xpos[b] - self.data.geom_xpos[a]
                if self.model.geom_type[a] == mj.mjtGeom.mjGEOM_PLANE:
                    direction = self.data.geom_xmat[a].reshape(3, 3)[:, 2].copy()
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    raise RuntimeError(
                        "Undefined center separation for closed-contact query"
                    )
                direction /= norm
                original = self.data.geom_xpos[b].copy()
                found = False
                attempts = []
                try:
                    for probe in [1e-5, 1e-4, 1e-3]:
                        shift = probe * direction
                        self.data.geom_xpos[b] = original + shift
                        separated = float(
                            mj.mj_geomDistance(
                                self.model, self.data, a, b, 0.025, witness
                            )
                        )
                        vector = witness[3:] - witness[:3]
                        length = np.linalg.norm(vector)
                        self.data.geom_xpos[b] = original
                        if separated <= 1e-8 or length <= 1e-8:
                            attempts.append(
                                dict(
                                    probe=probe,
                                    separated=separated,
                                    witness_length=float(length),
                                )
                            )
                            continue
                        normal = vector / length
                        witness[3:] -= shift
                        distance, pa, pb = support_gap(
                            self.model, self.data, a, b, normal
                        )
                        witness_gap = float(normal @ (witness[3:] - witness[:3]))
                        error = abs(witness_gap - distance)
                        attempts.append(
                            dict(
                                probe=probe,
                                separated=separated,
                                witness_length=float(length),
                                witness_gap=witness_gap,
                                support_gap=distance,
                                certificate_error=error,
                            )
                        )
                        if error <= 1e-6:
                            found = True
                            break
                finally:
                    self.data.geom_xpos[b] = original
                if not found:
                    raise RuntimeError(
                        f"Failed certified closed-contact query: {a}, {b}: {attempts}"
                    )
            actual.append((a, b, 0.5 * (witness[:3] + witness[3:]), normal, distance))
        return actual

    def measure(self, poses, anchors):
        objective = 0.0
        penetration = []
        contact_errors = []
        contact_residuals = []
        previous_contact = {}
        support_speed_max = 0.0
        for t, pose in enumerate(poses):
            self.forward(pose)
            for k, body in enumerate(self.body_ids):
                dp = (
                    self.data.xpos[body] - self.target_positions[t, k]
                ) / self.position_sigma
                dr = (
                    R.from_quat(self.data.xquat[body], scalar_first=True)
                    * R.from_quat(self.target_rotations[t, k], scalar_first=True).inv()
                ).as_rotvec() / self.orientation_sigma
                objective += self.position_weights[k] * (
                    dp @ dp
                ) + self.orientation_weights[k] * (dr @ dr)
            pair_penetration = {}
            for a, b, point, normal, distance in self.collisions():
                key = (min(a, b), max(a, b))
                pair_penetration[key] = max(pair_penetration.get(key, 0.0), -distance)
            penetration.extend(pair_penetration.values())
            for index in self.frame_phases[t]:
                phase = self.phases[index]
                point = self.point(phase.body, phase.local_point)
                contact_errors.append(
                    np.linalg.norm(point - anchors[index] - phase.radius * phase.normal)
                )
                reference = previous_contact.get(
                    index, anchors[index] + phase.radius * phase.normal
                )
                contact_residuals.extend(abs(point - reference) / self.dt)
                if index in previous_contact:
                    support_speed_max = max(
                        support_speed_max,
                        float(
                            np.linalg.norm(point - previous_contact[index]) / self.dt
                        ),
                    )
                previous_contact[index] = point.copy()
                if phase.cap_axis_local is not None:
                    axis = (
                        self.data.xmat[phase.body].reshape(3, 3) @ phase.cap_axis_local
                    )
                    penetration.append(max(0.0, phase.radius * (phase.normal @ axis)))
        acc = np.diff(self.velocities(poses), axis=0) / (
            self.dt * self.acceleration_sigma
        )
        objective += np.sum(acc * acc)
        # An exact penalty is evaluated on nonlinear geometry for a single
        # trajectory-wide line search; no frame can silently freeze in place.
        merit = objective + self.penalty * (
            sum(penetration) / self.dt + sum(contact_residuals)
        )
        return dict(
            merit=float(merit),
            objective=float(objective),
            penetration=max(penetration, default=0.0),
            contact=max(contact_errors, default=0.0),
            support_speed_max=support_speed_max,
        )

    def _tracking_objective(self, poses):
        """Linearize body tracking and acceleration with their original weights."""
        residual_rows, residual_cols, residual_values, residual_rhs = [], [], [], []
        for t, pose in enumerate(poses):
            self.forward(pose)
            col = t * self.nv
            for k, body in enumerate(self.body_ids):
                point = self.data.xpos[body].copy()
                jp, jr = self.jacobian(body, point)
                error_pos = (point - self.target_positions[t, k]) / self.position_sigma
                rotation_error = (
                    R.from_quat(self.data.xquat[body], scalar_first=True)
                    * R.from_quat(self.target_rotations[t, k], scalar_first=True).inv()
                ).as_rotvec()
                theta = np.linalg.norm(rotation_error)
                x, y, z = rotation_error
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                coefficient = (
                    1 / 12
                    if theta < 1e-5
                    else (1 - 0.5 * theta / np.tan(0.5 * theta)) / (theta * theta)
                )
                jr = (np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)) @ jr
                error_rot = rotation_error / self.orientation_sigma
                for jac, error, weight in [
                    (jp / self.position_sigma, error_pos, self.position_weights[k]),
                    (
                        jr / self.orientation_sigma,
                        error_rot,
                        self.orientation_weights[k],
                    ),
                ]:
                    if weight == 0:
                        continue
                    for axis in range(3):
                        _append_sparse_row(
                            residual_rows,
                            residual_cols,
                            residual_values,
                            len(residual_rhs),
                            col,
                            np.sqrt(weight) * jac[axis],
                        )
                        residual_rhs.append(np.sqrt(weight) * error[axis])
        B = sp.coo_matrix(
            (residual_values, (residual_rows, residual_cols)),
            shape=(len(residual_rhs), self.base_size),
        ).tocsr()
        acc = np.diff(self.velocities(poses), axis=0) / (
            self.dt * self.acceleration_sigma
        )
        B = sp.vstack([B, self.acceleration_jacobian(poses)], format="csr")
        residual = np.r_[residual_rhs, acc.ravel()]
        H = 2 * (B.T @ B) + sp.eye(self.base_size) * 1e-5
        gradient = 2 * np.asarray(B.T @ residual).ravel()
        return H, gradient

    def _contact_constraints(self, poses, anchors):
        """Linearize sticking contacts, collision gaps and spherical-cap orientation."""
        contact_rows, contact_cols, contact_values, contact_rhs = [], [], [], []
        collision_rows, collision_cols, collision_values, collision_rhs = [], [], [], []
        collision_keys = []
        previous_contact = {}
        for t, pose in enumerate(poses):
            self.forward(pose)
            col = t * self.nv
            for a, b, point, normal, distance in self.collisions(nearby=True):
                ja, _ = self.jacobian(int(self.model.geom_bodyid[a]), point)
                jb, _ = self.jacobian(int(self.model.geom_bodyid[b]), point)
                _append_sparse_row(
                    collision_rows,
                    collision_cols,
                    collision_values,
                    len(collision_rhs),
                    col,
                    normal @ (jb - ja),
                )
                collision_rhs.append(-distance)
                collision_keys.append((t, min(a, b), max(a, b)))
            for index in self.frame_phases[t]:
                phase = self.phases[index]
                point = self.point(phase.body, phase.local_point)
                jac, _ = self.jacobian(phase.body, point)
                previous = previous_contact.get(index)
                for axis in range(3):
                    row = len(contact_rhs)
                    _append_sparse_row(
                        contact_rows,
                        contact_cols,
                        contact_values,
                        row,
                        col,
                        jac[axis] / self.dt,
                    )
                    if previous is None:
                        contact_rows.append(row)
                        contact_cols.append(self.pose_size + index * 3 + axis)
                        contact_values.append(-1.0 / self.dt)
                        contact_rhs.append(
                            (
                                anchors[index, axis]
                                + phase.radius * phase.normal[axis]
                                - point[axis]
                            )
                            / self.dt
                        )
                    else:
                        old_point, old_jac, old_column = previous
                        _append_sparse_row(
                            contact_rows,
                            contact_cols,
                            contact_values,
                            row,
                            old_column,
                            -old_jac[axis] / self.dt,
                        )
                        contact_rhs.append((old_point[axis] - point[axis]) / self.dt)
                # Chaining equality constraints is algebraically equivalent to
                # a shared fixed anchor, but preserves the temporal bandedness
                # instead of introducing a dense hub spanning the whole phase.
                previous_contact[index] = (point.copy(), jac.copy(), col)
                if phase.cap_axis_local is not None:
                    axis = (
                        self.data.xmat[phase.body].reshape(3, 3) @ phase.cap_axis_local
                    )
                    _, jr = self.jacobian(phase.body, point)
                    cross = np.array(
                        [
                            [0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0],
                        ]
                    )
                    jac_cap = phase.normal @ cross @ jr
                    _append_sparse_row(
                        collision_rows,
                        collision_cols,
                        collision_values,
                        len(collision_rhs),
                        col,
                        phase.radius * jac_cap,
                    )
                    collision_rhs.append(phase.radius * (phase.normal @ axis))
                    collision_keys.append(("cap", t, index))
        C = sp.coo_matrix(
            (contact_values, (contact_rows, contact_cols)),
            shape=(len(contact_rhs), self.base_size),
        ).tocsr()
        G = (
            sp.coo_matrix(
                (collision_values, (collision_rows, collision_cols)),
                shape=(len(collision_rhs), self.base_size),
            ).tocsr()
            / self.dt
        )
        collision_rhs = np.asarray(collision_rhs) / self.dt
        return C, contact_rhs, G, collision_rhs, collision_keys

    def _joint_and_anchor_bounds(self, poses, anchors):
        """Keep joint locks, fixed overlap states and finite-face anchor bounds hard."""
        inequality_rows, inequality_cols, inequality_values = [], [], []
        lower, upper = [], []
        for t, pose in enumerate(poses):
            col = t * self.nv
            for joint in range(self.model.njnt):
                if self.model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE:
                    continue
                qa = int(self.model.jnt_qposadr[joint])
                va = int(self.model.jnt_dofadr[joint])
                row = len(lower)
                inequality_rows.append(row)
                inequality_cols.append(col + va)
                inequality_values.append(1.0)
                lo, hi = self.model.jnt_range[joint]
                if not self.model.jnt_limited[joint]:
                    lo, hi = -np.inf, np.inf
                if joint in self.locked:
                    lo = hi = 0.0
                lower.append(lo - pose[qa])
                upper.append(hi - pose[qa])
        # Shared overlap states and continuing contact anchors are optimization
        # constraints, so adjacent windows cannot be stitched with a pose jump.
        for frame in range(self.fixed_prefix):
            for axis in range(self.nv):
                inequality_rows.append(len(lower))
                inequality_cols.append(frame * self.nv + axis)
                inequality_values.append(1.0)
                lower.append(0.0)
                upper.append(0.0)
        for index in self.fixed_anchors:
            for axis in range(3):
                inequality_rows.append(len(lower))
                inequality_cols.append(self.pose_size + index * 3 + axis)
                inequality_values.append(1.0)
                lower.append(0.0)
                upper.append(0.0)
        for index, phase in enumerate(self.phases):
            _append_sparse_row(
                inequality_rows,
                inequality_cols,
                inequality_values,
                len(lower),
                self.pose_size + 3 * index,
                phase.normal,
            )
            value = phase.plane_height - phase.normal @ anchors[index]
            lower.append(value)
            upper.append(value)
            if phase.tangent_axes is not None:
                for axis, lo, hi in zip(
                    phase.tangent_axes, phase.tangent_lower, phase.tangent_upper
                ):
                    _append_sparse_row(
                        inequality_rows,
                        inequality_cols,
                        inequality_values,
                        len(lower),
                        self.pose_size + 3 * index,
                        axis,
                    )
                    lower.append(lo - axis @ anchors[index])
                    upper.append(hi - axis @ anchors[index])

        limits = sp.coo_matrix(
            (inequality_values, (inequality_rows, inequality_cols)),
            shape=(len(lower), self.base_size),
        ).tocsr()
        return limits, lower, upper

    def linearize(self, poses, anchors, radius):
        """Assemble the elastic QP from tracking, contact and hard-bound blocks."""
        H, gradient = self._tracking_objective(poses)
        C, contact_rhs, G, collision_rhs, collision_keys = self._contact_constraints(
            poses, anchors
        )
        limits, lower, upper = self._joint_and_anchor_bounds(poses, anchors)
        nc, ng = C.shape[0], G.shape[0]
        groups = {}
        group_ids = []
        for key in collision_keys:
            if key not in groups:
                groups[key] = len(groups)
            group_ids.append(groups[key])
        ngroups = len(groups)
        collision_slack = sp.coo_matrix(
            (np.ones(ng), (np.arange(ng), group_ids)), shape=(ng, ngroups)
        ).tocsr()
        ns = 2 * nc + ngroups
        # Elastic constraints keep the initial QP feasible. Their residuals
        # remain explicit and block acceptance; they are never hidden in output.
        P = sp.block_diag([H, sp.eye(ns) * 1e-6], format="csc")
        linear = np.r_[gradient, np.full(ns, self.penalty)]
        constraints = [
            sp.hstack([limits, sp.csr_matrix((len(lower), ns))]),
            sp.hstack([C, sp.eye(nc), -sp.eye(nc), sp.csr_matrix((nc, ngroups))]),
            sp.hstack([G, sp.csr_matrix((ng, 2 * nc)), collision_slack]),
            sp.hstack([sp.csr_matrix((ns, self.base_size)), sp.eye(ns)]),
            sp.hstack([sp.eye(self.base_size), sp.csr_matrix((self.base_size, ns))]),
        ]
        trust = np.full(self.base_size, radius)
        for t in range(self.frames):
            trust[t * self.nv : t * self.nv + 3] = radius * 0.25
        trust[self.pose_size :] = radius * 0.25
        lb = np.r_[lower, contact_rhs, collision_rhs, np.zeros(ns), -trust]
        ub = np.r_[upper, contact_rhs, np.full(ng, np.inf), np.full(ns, np.inf), trust]
        return P, linear, sp.vstack(constraints, format="csc"), lb, ub

    def feasible(self, poses, anchors, metrics):
        """Check physical tolerances and the hard pose/anchor constraints."""
        if (
            not trajectory_admissible(poses, metrics)
            or anchors.shape != (len(self.phases), 3)
            or not np.isfinite(anchors).all()
            or not np.allclose(
                np.linalg.norm(poses[:, 3:7], axis=1), 1.0, atol=1e-6, rtol=0.0
            )
        ):
            return False
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE:
                continue
            values = poses[:, self.model.jnt_qposadr[joint]]
            if joint in self.locked and np.max(abs(values)) > 1e-6:
                return False
            if self.model.jnt_limited[joint]:
                lo, hi = self.model.jnt_range[joint]
                if np.any(values < lo - 1e-6) or np.any(values > hi + 1e-6):
                    return False
        for phase, anchor in zip(self.phases, anchors):
            if abs(phase.normal @ anchor - phase.plane_height) > 1e-6:
                return False
            if phase.tangent_axes is not None:
                uv = phase.tangent_axes @ anchor
                if np.any(uv < phase.tangent_lower - 1e-6) or np.any(
                    uv > phase.tangent_upper + 1e-6
                ):
                    return False
        return True

    def solve(self, initial, iterations=30, initial_anchors=None):
        from retargeting.trajectory_qp import solve_elastic_qp

        poses = np.array(initial, copy=True)
        anchors = (
            np.array([p.anchor for p in self.phases]).reshape(-1, 3)
            if initial_anchors is None
            else np.array(initial_anchors, copy=True)
        )
        # Anchors are auxiliary optimization variables, not prescribed motion.
        # Start free anchors in their convex support domain: otherwise a hard
        # face bound and the local trust box can have no intersection before
        # the first step. Convex line-search updates then remain in that domain.
        # Committed anchors and every pose/target are left unchanged.
        for index, phase in enumerate(self.phases):
            if index in self.fixed_anchors:
                continue
            anchors[index] += (
                phase.plane_height - phase.normal @ anchors[index]
            ) * phase.normal
            if phase.tangent_axes is not None:
                uv = phase.tangent_axes @ anchors[index]
                in_domain = np.minimum(
                    np.maximum(uv, phase.tangent_lower), phase.tangent_upper
                )
                anchors[index] += phase.tangent_axes.T @ (in_domain - uv)
        radius = 0.2
        history = []
        current = self.measure(poses, anchors)
        current_iteration = -1
        best = None
        termination = "iteration_limit"

        def remember_feasible():
            nonlocal best
            if self.feasible(poses, anchors, current) and (
                best is None or current["objective"] < best[2]
            ):
                best = (
                    poses.copy(),
                    anchors.copy(),
                    current["objective"],
                    current_iteration,
                )

        remember_feasible()
        for iteration in range(iterations):
            P, c, A, lo, hi = self.linearize(poses, anchors, radius)
            equality = np.isfinite(lo) & (lo == hi)
            low = np.isfinite(lo) & ~equality
            high = np.isfinite(hi) & ~equality
            canonical = sp.vstack([A[equality], -A[low], A[high]], format="csc")
            rhs = np.r_[lo[equality], -lo[low], hi[high]]
            solution, qp_report = solve_elastic_qp(
                P,
                c,
                canonical,
                rhs,
                int(equality.sum()),
                self.base_size,
                penalty=self.penalty,
            )
            if solution is None:
                print({"qp_failure": qp_report}, flush=True)
                if best is None:
                    raise RuntimeError(
                        "Trajectory QP did not produce a certified model step"
                    )
                termination = "qp_failure_with_feasible_incumbent"
                history.append(
                    dict(
                        iteration=iteration,
                        accepted=False,
                        radius=radius,
                        qp=qp_report,
                        **current,
                    )
                )
                break
            delta = solution[: self.pose_size].reshape(self.frames, self.nv)
            anchor_delta = solution[self.pose_size : self.base_size].reshape(-1, 3)
            accepted = False
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                trial = poses.copy()
                for t in range(self.frames):
                    mj.mj_integratePos(self.model, trial[t], delta[t], alpha)
                trial_anchors = anchors + alpha * anchor_delta
                evaluated = self.measure(trial, trial_anchors)
                if evaluated["merit"] < current["merit"]:
                    poses, anchors, current = trial, trial_anchors, evaluated
                    current_iteration = iteration
                    accepted = True
                    break
            remember_feasible()
            history.append(
                dict(
                    iteration=iteration,
                    accepted=accepted,
                    radius=radius,
                    qp=qp_report,
                    **current,
                )
            )
            print(history[-1], flush=True)
            # Exact-penalty continuation: elastic variables are a feasibility
            # mechanism, not permission to trade persistent penetration for fit.
            violation = max(current["penetration"], current["contact"])
            stationary = np.max(abs(delta)) < 0.02 or (
                len(history) > 2
                and abs(history[-1]["merit"] - history[-2]["merit"])
                < 1e-4 * max(1.0, current["merit"])
            )
            # Small point errors can still alternate quickly enough to violate
            # sticking velocity. Continue restoration for every physical
            # criterion used by feasible(), including that temporal residual.
            needs_restoration = violation > 5e-4 or current["support_speed_max"] > 0.01
            if accepted and needs_restoration and stationary and self.penalty < 2e6:
                self.penalty *= 5
                current = self.measure(poses, anchors)
                radius = max(radius, 0.1)
                print("Exact penalty", self.penalty, flush=True)
                continue
            if not accepted:
                radius *= 0.5
            elif alpha == 1.0:
                radius = min(0.3, radius * 1.2)
            if (
                violation < 5e-4
                and np.max(abs(delta)) < 1e-3
                and current["support_speed_max"] <= 0.01
                and self.feasible(poses, anchors, current)
            ):
                termination = "converged"
                break
            if radius < 1e-4:
                termination = "trust_region_limit"
                break
        recovered = False
        # Return the final feasible iterate normally. If refinement leaves the
        # feasible set, retain one whole previously certified trajectory; never
        # assemble frames from different iterates or relax acceptance criteria.
        if not self.feasible(poses, anchors, current) and best is not None:
            poses, anchors, _, current_iteration = best
            recovered = True
        if history:
            history[-1].update(
                termination=termination,
                returned_iteration=current_iteration,
                returned_feasible_incumbent=recovered
                or termination == "qp_failure_with_feasible_incumbent",
            )
        return poses, anchors, history


def trajectory_metrics(problem, poses, anchors):
    final = problem.measure(poses, anchors)
    speeds = []
    for phase in problem.phases:
        points = []
        for pose in poses[phase.start : phase.stop]:
            problem.forward(pose)
            points.append(problem.point(phase.body, phase.local_point).copy())
        if len(points) > 1:
            speeds.extend(np.linalg.norm(np.diff(points, axis=0), axis=1) / problem.dt)
    final["support_speed_p95"] = float(np.percentile(speeds, 95)) if speeds else None
    final["support_speed_max"] = float(max(speeds, default=0.0))
    acceleration = np.diff(problem.velocities(poses), axis=0) / problem.dt
    final["joint_acceleration_p95"] = float(np.percentile(abs(acceleration[:, 6:]), 95))
    final["root_displacement"] = np.linalg.norm(poses[-1, :3] - poses[0, :3]).item()
    return final


def trajectory_admissible(poses, metrics):
    return (
        np.isfinite(poses).all()
        and metrics["penetration"] <= 0.002
        and metrics["contact"] <= 0.001
        and metrics["support_speed_max"] <= 0.01
    )


@dataclass
class WindowState:
    """Committed poses and persistent anchors at a solver window boundary."""

    poses: np.ndarray
    anchors: np.ndarray
    committed: int
    history: list[dict]


def solve_trajectory_windows(
    problem,
    initial,
    iterations,
    window_frames,
    *,
    state: WindowState | None = None,
    on_window: Callable[[WindowState], None] | None = None,
):
    """Solve with two fixed overlap frames and shared support anchors.

    The optional state resumes a committed prefix; on_window lets the caller
    persist progress. Neither affects the numerical window problem.
    """
    if window_frames < 3:
        raise ValueError("A solver window needs at least three frames")
    if state is None:
        state = WindowState(
            np.array(initial, copy=True),
            np.array([p.anchor for p in problem.phases]).reshape(-1, 3),
            0,
            [],
        )
    poses, anchors = state.poses, state.anchors
    committed, history = state.committed, state.history
    if problem.frames <= window_frames:
        return problem.solve(initial, iterations=iterations, initial_anchors=anchors)
    while committed < problem.frames:
        lo = max(0, committed - 2)
        hi = min(problem.frames, lo + window_frames)
        # Keep at least one free frame if the last window is just the overlap.
        indices = [
            i for i, p in enumerate(problem.phases) if p.start < hi and p.stop > lo
        ]
        phases = [
            replace(
                problem.phases[i],
                start=max(problem.phases[i].start, lo) - lo,
                stop=min(problem.phases[i].stop, hi) - lo,
                anchor=anchors[i].copy(),
            )
            for i in indices
        ]
        fixed = [
            j for j, i in enumerate(indices) if problem.phases[i].start < committed
        ]
        local = TrajectorySQP(
            problem.model,
            1.0 / problem.dt,
            problem.body_ids,
            problem.target_positions[lo:hi],
            problem.target_rotations[lo:hi],
            phases,
            [problem.model.joint(i).name for i in problem.locked],
            position_sigma=problem.position_sigma,
            orientation_sigma=problem.orientation_sigma,
            acceleration_sigma=problem.acceleration_sigma,
            position_weights=problem.position_weights,
            orientation_weights=problem.orientation_weights,
            fixed_prefix=committed - lo,
            fixed_anchors=fixed,
        )
        result, points, trace = local.solve(poses[lo:hi], iterations=iterations)
        metrics = trajectory_metrics(local, result, points)
        if not trajectory_admissible(result, metrics):
            raise RuntimeError(
                f"Window {lo}:{hi} violates geometry/contact constraints: {metrics}"
            )
        # Do not overwrite even the numerically unchanged overlap samples.
        poses[committed:hi] = result[committed - lo :]
        anchors[indices] = points
        history.extend(dict(window_start=lo, window_stop=hi, **row) for row in trace)
        committed = hi
        state.committed = committed
        if on_window is not None:
            on_window(state)
    return poses, anchors, history
