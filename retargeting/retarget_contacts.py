"""Contact geometry, finite-surface correspondence and source contact inference."""

from dataclasses import dataclass

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R


def support_point(model, data, geom, normal):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local = rotation.T @ normal
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        count = model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count]
        point = vertices[np.argmax(vertices @ local)]
    elif kind == mj.mjtGeom.mjGEOM_BOX:
        point = size * np.sign(local)
    elif kind == mj.mjtGeom.mjGEOM_CYLINDER:
        point = np.r_[
            size[0] * local[:2] / max(np.linalg.norm(local[:2]), 1e-30),
            size[1] * np.sign(local[2]),
        ]
    elif kind == mj.mjtGeom.mjGEOM_SPHERE:
        point = size[0] * local
    elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
        point = size[0] * local + np.array([0.0, 0.0, size[1] * np.sign(local[2])])
    elif kind == mj.mjtGeom.mjGEOM_ELLIPSOID:
        point = size * size * local / max(np.linalg.norm(size * local), 1e-30)
    else:
        raise ValueError(f"Unsupported support geometry {kind}")
    return data.geom_xpos[geom] + rotation @ point


def support_gap(model, data, a, b, normal):
    if model.geom_type[a] == mj.mjtGeom.mjGEOM_PLANE:
        pb = support_point(model, data, b, -normal)
        gap = float(normal @ (pb - data.geom_xpos[a]))
        pa = pb - gap * normal
    else:
        pa = support_point(model, data, a, normal)
        pb = support_point(model, data, b, -normal)
        gap = float(normal @ (pb - pa))
    return gap, pa, pb


def orient_native_contacts(model, data, contacts):
    """Correct reversed normals only when the opposite support plane certifies them.

    Compare the deepest point of each parallel manifold group against the
    exact convex support gap. Shallow manifold points are not individual
    extremal witnesses. Retain every point and distance, including groups
    whose near-zero normal is not uniquely certified in either direction.
    """
    groups = {}
    for index, (a, b, point, normal, distance) in enumerate(contacts):
        key = (a, b, tuple(np.round(normal, 10)))
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        a, b, _, normal, _ = contacts[indices[0]]
        depth = min(contacts[i][-1] for i in indices)
        native_gap, _, _ = support_gap(model, data, a, b, normal)
        if abs(native_gap - depth) <= 1e-6:
            continue
        reverse_gap, _, _ = support_gap(model, data, a, b, -normal)
        if abs(reverse_gap - depth) <= 1e-6:
            for index in indices:
                ga, gb, point, old_normal, distance = contacts[index]
                contacts[index] = (ga, gb, point, -old_normal, distance)
    return contacts


@dataclass
class ContactPhase:
    body: int
    local_point: np.ndarray
    start: int
    stop: int
    anchor: np.ndarray
    normal: np.ndarray
    plane_height: float
    tangent_axes: np.ndarray | None = None
    tangent_lower: np.ndarray | None = None
    tangent_upper: np.ndarray | None = None
    radius: float = 0.0
    cap_axis_local: np.ndarray | None = None


def foot_probes(model, name):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    body = model.body(name).id
    candidates = []
    radii = []
    for gid in range(model.ngeom):
        descendant = int(model.geom_bodyid[gid])
        while descendant not in (0, body):
            descendant = int(model.body_parentid[descendant])
        if descendant != body or not (
            model.geom_contype[gid] or model.geom_conaffinity[gid]
        ):
            continue
        rotation = data.geom_xmat[gid].reshape(3, 3)
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_CAPSULE:
            candidates.extend(
                [
                    data.geom_xpos[gid]
                    + sign * rotation[:, 2] * model.geom_size[gid, 1]
                    for sign in (-1, 1)
                ]
            )
            radii.extend([model.geom_size[gid, 0]] * 2)
        elif model.geom_type[gid] == mj.mjtGeom.mjGEOM_SPHERE:
            candidates.append(data.geom_xpos[gid])
            radii.append(model.geom_size[gid, 0])
    if not candidates:
        raise ValueError(f"No contact support primitives for {name}")
    local = (np.array(candidates) - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    radii = np.array(radii)
    height = local[:, 2] - radii
    selected = height <= height.min() + 0.002
    local, radii = local[selected], radii[selected]
    normalized = (local[:, :2] - local[:, :2].mean(0)) / np.maximum(
        np.ptp(local[:, :2], axis=0), 1e-6
    )
    return [
        (local[index], radii[index])
        for direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for index in [np.argmax(normalized @ direction)]
    ]


def exposed_contact(model, body, preferred, normal, rotation):
    """Choose an exposed material primitive of the actual rigid contact body.

    A source sole landmark need not lie on the target support hull for a wall.
    Keep its correspondence only when it is exposed in the observed direction.
    """
    _, _, points, radii = _contact_features(model, body, preferred, 0.0)
    points, radii = points[1:], radii[1:]  # Only real geometry can be exposed.
    local_normal = rotation.T @ normal
    # Sole contact transfers a support patch, not a single extreme vertex of
    # a slightly tilted source pose. Toe/side contact uses the exposed hull.
    # The dominant face distinguishes these geometric modes without clip rules.
    if local_normal[2] > np.max(abs(local_normal[:2])):
        local_normal = np.array([0.0, 0.0, 1.0])
    support = points @ local_normal - radii
    exposed = np.flatnonzero(support <= support.min() + 1e-5)
    index = exposed[np.argmin(np.linalg.norm(points[exposed] - preferred, axis=1))]
    return points[index], float(radii[index])


def _contact_features(model, body, preferred, preferred_radius):
    """Collect candidate material points and their rigid-body geometry transforms."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    body_rotation = data.xmat[body].reshape(3, 3)
    body_position = data.xpos[body].copy()
    geometries, features, radii = [], [preferred.copy()], [float(preferred_radius)]
    for geom in range(model.ngeom):
        descendant = int(model.geom_bodyid[geom])
        while descendant not in (0, body):
            descendant = int(model.body_parentid[descendant])
        if descendant != body or not (
            model.geom_contype[geom] or model.geom_conaffinity[geom]
        ):
            continue
        gr = data.geom_xmat[geom].reshape(3, 3)
        gp = data.geom_xpos[geom].copy()
        geometries.append(
            (geom, body_rotation.T @ (gp - body_position), body_rotation.T @ gr)
        )
        kind = model.geom_type[geom]
        if kind == mj.mjtGeom.mjGEOM_SPHERE:
            world, radius = gp[None, :], model.geom_size[geom, 0]
        elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
            world = np.array(
                [gp + sign * gr[:, 2] * model.geom_size[geom, 1] for sign in (-1, 1)]
            )
            radius = model.geom_size[geom, 0]
        elif kind == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[start : start + count] @ gr.T + gp
            radius = 0.0
        else:
            raise ValueError(f"Unsupported material contact geometry {kind}")
        local = (world - body_position) @ body_rotation
        features.extend(local)
        radii.extend([float(radius)] * len(local))
    features, radii = np.array(features), np.array(radii)
    return data, geometries, features, radii


def _project_on_face(clearances, u0, lower, upper, tolerance):
    """Find a feasible point on the finite rectangle using the existing SLSQP search."""
    from scipy.optimize import minimize

    def cost(uv):
        delta = (uv - u0) / 0.025
        return float(delta @ delta)

    def gradient(uv):
        return 2 * (uv - u0) / (0.025**2)

    def project(start):
        result = minimize(
            cost,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints={
                "type": "ineq",
                "fun": lambda uv: (clearances(uv) + 1e-7) / 0.025,
            },
            options=dict(maxiter=40, ftol=1e-10, eps=1e-5),
        )
        uv = np.clip(result.x, lower, upper)
        if np.min(clearances(uv)) >= -tolerance:
            return uv
        return None

    original = np.clip(u0, lower, upper)
    feasible = []
    if np.min(clearances(original)) >= -tolerance:
        feasible.append(original)
    else:
        projected = project(original)
        if projected is not None:
            feasible.append(projected)
        # The finite rectangle has a fixed geometric boundary, not a
        # clip-specific search neighbourhood. Boundary starts recover
        # valid components when the initial penetrating query is flat.
        if not feasible:
            boundary = []
            for axis in range(2):
                for value in (lower[axis], upper[axis]):
                    uv = original.copy()
                    uv[axis] = value
                    boundary.append(uv)
            boundary.extend(
                np.array([x, y])
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
            )
            boundary.sort(key=cost)
            for uv in boundary:
                if np.min(clearances(uv)) >= -tolerance:
                    feasible.append(uv)
                    projected = project(uv)
                    if projected is not None:
                        feasible.append(projected)
                    break
    return min(feasible, key=cost) if feasible else None


def finite_contact_mapping(
    model,
    body,
    preferred,
    preferred_radius,
    normal,
    rotation,
    anchor,
    tangent_axes=None,
    tangent_lower=None,
    tangent_upper=None,
):
    """Choose a real material feature and a feasible anchor on finite terrain.

    Sole support remains a shared planar patch; its orientation is optimized
    by the trajectory solver. Other contacts retain the preferred landmark
    when possible, or use a nearby feature of the actual collision geometry.
    Material displacement and anchor displacement are both measured in metres.
    This only initializes a correspondence and anchor, never edits a trajectory.
    """
    preferred = np.asarray(preferred)
    normal = np.asarray(normal)
    rotation = np.asarray(rotation)
    anchor = np.asarray(anchor)
    local_normal = rotation.T @ normal
    if tangent_axes is None or local_normal[2] > np.max(abs(local_normal[:2])):
        point, radius = exposed_contact(model, body, preferred, normal, rotation)
        return point, radius, anchor.copy(), dict(mode="sole_or_infinite_face")

    axes = np.asarray(tangent_axes)
    lower, upper = np.asarray(tangent_lower), np.asarray(tangent_upper)
    data, geometries, features, radii = _contact_features(
        model, body, preferred, preferred_radius
    )
    terrain = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == 0
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    pairs = [
        (world, geom)
        for world in terrain
        for geom, _, _ in geometries
        if (
            (model.geom_contype[world] & model.geom_conaffinity[geom])
            or (model.geom_contype[geom] & model.geom_conaffinity[world])
        )
    ]
    if not pairs:
        raise ValueError("Finite contact mapping requires real static scene geometry")
    u0 = axes @ anchor
    material_error = np.sum(
        (
            features
            - radii[:, None] * local_normal
            - preferred
            + preferred_radius * local_normal
        )
        ** 2,
        axis=1,
    )
    order = np.argsort(material_error, kind="stable")
    best = None
    examined = 0
    queries = 0
    tolerance = 1e-6

    for index in order:
        if best is not None and material_error[index] > best[0]:
            break
        point, radius = features[index], radii[index]
        examined += 1
        cached_u, cached_distances = None, None

        def clearances(uv):
            nonlocal cached_u, cached_distances, queries
            uv = np.asarray(uv)
            if cached_u is not None and np.array_equal(uv, cached_u):
                return cached_distances
            placement = anchor + (uv - u0) @ axes + radius * normal - rotation @ point
            for geom, gp, gr in geometries:
                data.geom_xpos[geom] = placement + rotation @ gp
                data.geom_xmat[geom] = (rotation @ gr).ravel()
            cached_u = uv.copy()
            cached_distances = np.array(
                [
                    mj.mj_geomDistance(model, data, a, b, 0.1, np.zeros(6))
                    for a, b in pairs
                ]
            )
            queries += 1
            return cached_distances

        uv = _project_on_face(clearances, u0, lower, upper, tolerance)
        if uv is None:
            continue
        value = float(material_error[index] + np.sum((uv - u0) ** 2))
        if best is None or value < best[0]:
            best = (
                value,
                point.copy(),
                float(radius),
                anchor + (uv - u0) @ axes,
                float(np.min(clearances(uv))),
                float(material_error[index]),
            )
    if best is None:
        raise RuntimeError(
            "No visible material contact feature on the bounded scene face"
        )
    value, point, radius, selected_anchor, clearance, material = best
    if not np.isfinite(np.r_[point, radius, selected_anchor, value]).all():
        raise RuntimeError("Non-finite material correspondence")
    uv = axes @ selected_anchor
    if (
        abs(normal @ (selected_anchor - anchor)) > 1e-8
        or np.any(uv < lower - 1e-8)
        or np.any(uv > upper + 1e-8)
    ):
        raise RuntimeError("Selected contact anchor left its finite face")
    # Query the winning feature again independently of the last candidate's
    # cached projection closure; the search supplies an upper bound, not a
    # claim of a globally closest non-convex correspondence.
    placement = selected_anchor + radius * normal - rotation @ point
    for geom, gp, gr in geometries:
        data.geom_xpos[geom] = placement + rotation @ gp
        data.geom_xmat[geom] = (rotation @ gr).ravel()
    distances = np.array(
        [mj.mj_geomDistance(model, data, a, b, 0.1, np.zeros(6)) for a, b in pairs]
    )
    queries += 1
    clearance = float(np.min(distances))
    if not np.isfinite(distances).all() or clearance < -tolerance:
        raise RuntimeError("Selected contact feature failed final scene certificate")
    return (
        point,
        radius,
        selected_anchor,
        dict(
            mode="finite_scene",
            cost_m2=value,
            material_displacement_m=np.sqrt(material),
            anchor_displacement_m=float(np.linalg.norm(selected_anchor - anchor)),
            minimum_clearance_m=clearance,
            candidates_examined=examined,
            geometry_queries=queries,
        ),
    )


def contact_observations(
    points,
    surfaces,
    fps,
    radius=0.0,
    surface_normals=None,
    alignment_floor=0.0,
    alignment_penalty=1e4,
):
    """Per-material-point likelihoods for free or one finite contact face.

    A contact observation has Gaussian position and stationary-point velocity
    error. Finite-face geometry is the same anchor-plus-radius relation used
    by the optimizer, including tangential distance outside the face boundary.
    The free-state reference cost retains a finite preference for a supported
    patch when several nearby, stationary material points agree.
    """
    points = np.asarray(points)
    velocity = np.gradient(points, 1.0 / fps, axis=0)
    emission = np.full((len(points), len(surfaces) + 1), 2.0)
    projections = []
    for k, (normal, height, axes, lo, hi) in enumerate(surfaces):
        projected = points - (points @ normal - height)[:, None] * normal
        if axes is not None:
            uv = projected @ axes.T
            projected += (np.clip(uv, lo, hi) - uv) @ axes
        error = points - projected - radius * normal
        cost = 0.5 * np.sum((error / 0.025) ** 2, axis=1)
        cost += 0.5 * np.sum((velocity / 0.2) ** 2, axis=1)
        if surface_normals is not None:
            cost += alignment_penalty * np.maximum(
                alignment_floor - surface_normals @ normal, 0.0
            )
        emission[:, k + 1] = cost
        projections.append(projected)
    return emission, projections


def joint_contact_states(emissions, labels, fps, transition=0.06):
    """Exact joint Viterbi with a single OR contact-label factor per body.

    Pairwise transition costs are additive Potts costs on the material points.
    Successive min-plus transforms along each state axis avoid the dense
    Q**P-by-Q**P transition matrix without approximating the joint solution.
    """
    emissions = np.asarray(emissions)
    frames, points, choices = emissions.shape
    shape = (choices,) * points
    count = choices**points
    dtype = np.uint16 if count <= np.iinfo(np.uint16).max + 1 else np.uint32
    states = np.arange(count, dtype=dtype).reshape(shape)
    back = np.empty((frames - 1, count), dtype=dtype)

    def observation(frame):
        cost = np.zeros(shape)
        for point in range(points):
            dims = [1] * points
            dims[point] = choices
            cost += emissions[frame, point].reshape(dims)
        if labels is not None:
            # A foot-level label says at least one material point supports it.
            # It does not label every heel/toe point independently.
            cost += -1.0 if labels[frame] else 1.0
            cost[(0,) * points] += 2.0 if labels[frame] else -2.0
        return cost / fps

    scores = observation(0)
    for frame in range(1, frames):
        predecessor = states.copy()
        for axis in range(points):
            best = np.argmin(scores, axis=axis, keepdims=True)
            alternative = np.take_along_axis(scores, best, axis) + transition
            origin = np.take_along_axis(predecessor, best, axis)
            switch = alternative < scores
            scores = np.where(switch, alternative, scores)
            predecessor = np.where(switch, origin, predecessor)
        scores += observation(frame)
        back[frame - 1] = predecessor.ravel()
    path = np.empty(frames, dtype=dtype)
    path[-1] = np.argmin(scores)
    for frame in range(frames - 2, -1, -1):
        path[frame] = back[frame, path[frame + 1]]
    return np.array(np.unravel_index(path.astype(int), shape))


def contact_phases_from_states(
    states, projections, surfaces, body, local, radius=0.0, cap_axis=None
):
    phases = []
    boundaries = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        state = states[start]
        if state == 0:
            continue
        normal, height, axes, lo, hi = surfaces[state - 1]
        anchor = np.mean(projections[state - 1][start:stop], axis=0)
        phases.append(
            ContactPhase(
                body,
                local,
                int(start),
                int(stop),
                anchor,
                normal,
                height,
                axes,
                lo,
                hi,
                radius,
                cap_axis,
            )
        )
    return phases


def infer_phases(
    points,
    surfaces,
    target_body,
    local,
    labels,
    fps,
    radius=0.0,
    cap_axis=None,
    surface_normals=None,
    alignment_floor=0.0,
    source_radius=None,
    alignment_penalty=1e4,
):
    observation, projections = contact_observations(
        points,
        surfaces,
        fps,
        radius if source_radius is None else source_radius,
        surface_normals,
        alignment_floor,
        alignment_penalty,
    )
    states = joint_contact_states(observation[:, None, :], labels, fps)[0]
    return contact_phases_from_states(
        states, projections, surfaces, target_body, local, radius, cap_axis
    ), states


def infer_foot_phases(
    points,
    surfaces,
    target_body,
    target_locals,
    labels,
    fps,
    source_radii,
    target_radii,
    surface_normals,
):
    observations, projections = [], []
    for index in range(points.shape[1]):
        values, projected = contact_observations(
            points[:, index],
            surfaces,
            fps,
            source_radii[index],
            surface_normals,
            alignment_floor=0.5,
            alignment_penalty=2.0,
        )
        observations.append(values)
        projections.append(projected)
    states = joint_contact_states(np.stack(observations, axis=1), labels, fps)
    phases = []
    for index, local in enumerate(target_locals):
        phases.extend(
            contact_phases_from_states(
                states[index],
                projections[index],
                surfaces,
                target_body,
                local,
                target_radii[index],
            )
        )
    return phases, states


def build_contact_phases(
    source, model, names, positions, rotations, contact_labels, surfaces, fps
):
    """Map G1 support observations to T2 feet and spherical hand endpoints."""
    phases = []
    for side_index, side in enumerate(("left", "right")):
        foot = side + "_ankle_roll_link"
        body = model.body(foot).id
        index = names.index(foot)
        source_probes = foot_probes(source, foot)
        target_probes = foot_probes(model, foot)
        foot_rotation = R.from_quat(rotations[:, index], scalar_first=True)
        foot_points = np.stack(
            [
                positions[:, index] + foot_rotation.apply(local)
                for local, radius in source_probes
            ],
            axis=1,
        )
        found, _ = infer_foot_phases(
            foot_points,
            surfaces,
            body,
            [local for local, radius in target_probes],
            contact_labels[:, side_index],
            fps,
            [radius for local, radius in source_probes],
            [radius for local, radius in target_probes],
            foot_rotation.apply([0.0, 0.0, 1.0]),
        )
        for phase in found:
            observed = (
                R.from_quat(
                    rotations[phase.start : phase.stop, index], scalar_first=True
                )
                .mean()
                .as_matrix()
            )
            phase.local_point, phase.radius, phase.anchor, _ = finite_contact_mapping(
                model,
                body,
                phase.local_point,
                phase.radius,
                phase.normal,
                observed,
                phase.anchor,
                phase.tangent_axes,
                phase.tangent_lower,
                phase.tangent_upper,
            )
        phases.extend(found)
        name = side + "_sphere_hand"
        index = names.index(name)
        rotation = R.from_quat(rotations[:, index], scalar_first=True)
        points = positions[:, index] + rotation.apply([0.0, 0.0, 0.022])
        target_name = side + "_hand_ee_link"
        sign = 1 if side == "left" else -1
        found, _ = infer_phases(
            points,
            surfaces,
            model.body(target_name).id,
            np.array([0.0, sign * 0.022, 0.0]),
            None,
            fps,
            radius=0.04,
            cap_axis=np.array([0.0, sign, 0.0]),
            surface_normals=-rotation.apply([0.0, 0.0, 1.0]),
        )
        phases.extend(found)
    return phases


def foot_contact_labels(phases, frames, spec):
    """The optimizer's support phases are the authoritative contact labels."""
    contacts = np.zeros((frames, 2), dtype=np.uint8)
    feet = {
        spec.model.body(side + "_ankle_roll_link").id: i
        for i, side in enumerate(("left", "right"))
    }
    for phase in phases:
        if phase.body not in feet:
            continue
        a, b = int(phase.start), int(phase.stop)
        if not 0 <= a < b <= frames:
            raise ValueError("Invalid contact phase bounds")
        contacts[a:b, feet[phase.body]] = 1
    return contacts
