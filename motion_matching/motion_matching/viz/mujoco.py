"""Render the native T2 MJCF in the existing Viser motion viewer."""

import mujoco as mj
import numpy as np
import trimesh
import viser

from motion_matching.robots.base import RobotSpec


def _geom_mesh(model: mj.MjModel, geom: int) -> tuple[np.ndarray, np.ndarray]:
    kind = model.geom_type[geom]
    size = model.geom_size[geom]
    if kind == mj.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        vertex = model.mesh_vertadr[mesh]
        face = model.mesh_faceadr[mesh]
        return (model.mesh_vert[vertex:vertex + model.mesh_vertnum[mesh]],
                model.mesh_face[face:face + model.mesh_facenum[mesh]])
    if kind == mj.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind == mj.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif kind == mj.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    else:
        raise NotImplementedError(f"Unsupported T2 geometry type: {kind}")
    return mesh.vertices, mesh.faces


class ViserMujoco:
    """Render the T2 joint configuration under the viewer's robot root.

    The free root stays at the origin. The caller supplies its world pose
    through ``root_node_name``, as it does for the G1 URDF renderer.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        robot_spec: RobotSpec,
        root_node_name: str = "/robot_root",
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
    ) -> None:
        self.spec = robot_spec
        self.model = robot_spec.model
        self.data = mj.MjData(self.model)
        pose = np.zeros(7 + robot_spec.num_joints)
        pose[0] = 1.0
        self.data.qpos[:] = robot_spec.to_mujoco_qpos(pose)
        self._handles = []
        self._visual_handles = []
        self._collision_handles = []

        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] == 0:
                continue
            group = self.model.geom_group[geom]
            if group == 1 and self.model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH and load_meshes:
                rgba = self.model.geom_rgba[geom]
                name = "visual"
            elif group == 3 and load_collision_meshes:
                rgba = np.array([1.0, 0.0, 0.0, 0.5])
                name = "collision"
            else:
                continue
            vertices, faces = _geom_mesh(self.model, geom)
            handle = server.scene.add_mesh_simple(
                f"{root_node_name}/t2/{name}/{geom}",
                vertices=vertices,
                faces=faces,
                color=tuple((rgba[:3] * 255).astype(int)),
                opacity=float(rgba[3]),
            )
            self._handles.append((geom, handle))
            if name == "visual":
                self._visual_handles.append(handle)
            else:
                self._collision_handles.append(handle)
        self.update_cfg(np.zeros(robot_spec.num_joints))

    def get_actuated_joint_names(self) -> tuple[str, ...]:
        return tuple(self.spec.joint_names)

    def get_actuated_joint_limits(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(self.model.jnt_range[joint, 0]), float(self.model.jnt_range[joint, 1]))
            if self.model.jnt_limited[joint] else (-np.pi, np.pi)
            for name, joint in zip(self.spec.joint_names, self.spec.joint_ids)
        }

    @property
    def show_visual(self) -> bool:
        return any(handle.visible for handle in self._visual_handles)

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        for handle in self._visual_handles:
            handle.visible = visible

    @property
    def show_collision(self) -> bool:
        return any(handle.visible for handle in self._collision_handles)

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        for handle in self._collision_handles:
            handle.visible = visible

    def update_cfg(self, joint_angles: np.ndarray) -> None:
        self.data.qpos[self.spec.joint_qpos_indices] = joint_angles
        mj.mj_forward(self.model, self.data)
        for geom, handle in self._handles:
            rotation = np.empty(4)
            mj.mju_mat2Quat(rotation, self.data.geom_xmat[geom])
            handle.position = self.data.geom_xpos[geom].copy()
            handle.wxyz = rotation

    def remove(self) -> None:
        for _, handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._visual_handles.clear()
        self._collision_handles.clear()
