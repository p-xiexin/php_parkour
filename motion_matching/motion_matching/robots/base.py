"""Native MuJoCo skeleton and PHP qpos conversion for scalar-joint robots.

PHP qpos is ``[root_wxyz, root_xyz, joint_angles]``. The matching skeleton is
``[Simulation, free-root body, hinge bodies]`` and retains static bind rotations.
"""

from pathlib import Path

import mujoco as mj
import numpy as np

from motion_matching.kinematics import quat


class RobotSpec:
    def __init__(self, robot_id: str, model_path: str | Path, *, pelvis_body: str,
                 left_foot_body: str = "left_ankle_roll_link",
                 right_foot_body: str = "right_ankle_roll_link",
                 exclude_tracking_body_names: tuple[str, ...] = ()) -> None:
        self.robot_id = robot_id
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = mj.MjModel.from_xml_path(str(self.model_path))
        model = self.model
        free_ids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        if len(free_ids) != 1 or np.any(~np.isin(model.jnt_type, [mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_HINGE])):
            raise ValueError("Expected one free root and scalar hinge joints")
        self.root_joint_id = int(free_ids[0])
        self.root_qpos_address = int(model.jnt_qposadr[self.root_joint_id])
        self.root_body_id = int(model.jnt_bodyid[self.root_joint_id])
        self.joint_ids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)
        self.joint_names = [model.joint(int(i)).name for i in self.joint_ids]
        self.num_joints = len(self.joint_names)
        self.joint_qpos_indices = model.jnt_qposadr[self.joint_ids]
        self.joint_axes = model.jnt_axis[self.joint_ids].copy()
        hinge_bodies = model.jnt_bodyid[self.joint_ids]
        self.skeleton_body_ids = np.concatenate(([self.root_body_id], hinge_bodies))
        self.bone_names = ["Simulation"] + [model.body(int(i)).name for i in self.skeleton_body_ids]
        self.num_bones = len(self.bone_names)
        self.joint_bone_indices = np.arange(2, self.num_bones)
        body_to_bone = {int(body): i + 1 for i, body in enumerate(self.skeleton_body_ids)}
        parents = [-1, 0]
        for body in hinge_bodies:
            parent = int(model.body_parentid[body])
            while parent not in body_to_bone:
                if parent == 0:
                    raise ValueError("Hinge body is not descended from the free root")
                parent = int(model.body_parentid[parent])
            parents.append(body_to_bone[parent])
        self.bone_parents = np.asarray(parents, dtype=np.int32)
        self.bone_left_foot = self.bone_names.index(left_foot_body)
        self.bone_right_foot = self.bone_names.index(right_foot_body)
        self.bone_hips = self.bone_names.index(pelvis_body)
        self.body_ids = np.asarray([i for i in range(1, model.nbody)
                                    if model.body(i).name not in exclude_tracking_body_names])
        self.body_names = [model.body(int(i)).name for i in self.body_ids]

        zero = np.zeros(7 + self.num_joints)
        zero[0] = 1
        _, bind_rotations = self.encode_qpos_to_bones(zero)
        self.joint_rest_rotations = bind_rotations[self.joint_bone_indices]
        data = mj.MjData(model)
        data.qpos[:] = self.to_mujoco_qpos(zero)
        mj.mj_forward(model, data)
        axes_world = quat.mul_vec(data.xquat[hinge_bodies], self.joint_axes)
        permutation, signs = [], []
        for i, name in enumerate(self.joint_names):
            partner = ("right_" + name[5:] if name.startswith("left_") else
                       "left_" + name[6:] if name.startswith("right_") else name)
            j = self.joint_names.index(partner)
            reflected_axis = axes_world[j] * np.array([-1., 1., -1.])
            permutation.append(j)
            signs.append(np.sign(np.dot(axes_world[i], reflected_axis)))
        self.mirror_joint_indices = np.asarray(permutation)
        self.mirror_joint_signs = np.asarray(signs)

    def _validate_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim not in (1, 2) or qpos.shape[-1] != 7 + self.num_joints:
            raise ValueError(f"Expected PHP qpos (..., {7 + self.num_joints}), got {qpos.shape}")
        return qpos

    def to_mujoco_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = self._validate_qpos(qpos)
        result = np.broadcast_to(self.model.qpos0, (*qpos.shape[:-1], self.model.nq)).copy()
        adr = self.root_qpos_address
        result[..., adr:adr + 3] = qpos[..., 4:7]
        result[..., adr + 3:adr + 7] = qpos[..., :4]
        result[..., self.joint_qpos_indices] = qpos[..., 7:]
        return result

    def from_mujoco_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos)
        adr = self.root_qpos_address
        return np.concatenate((qpos[..., adr + 3:adr + 7], qpos[..., adr:adr + 3],
                               qpos[..., self.joint_qpos_indices]), axis=-1)

    def encode_qpos_to_bones(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode native FK with an identity Simulation bone; the builder bakes Sim0."""
        qpos = self._validate_qpos(qpos)
        single = qpos.ndim == 1
        qpos = np.atleast_2d(qpos)
        positions = np.zeros((len(qpos), self.num_bones, 3))
        rotations = np.zeros((len(qpos), self.num_bones, 4))
        rotations[..., 0] = 1
        data = mj.MjData(self.model)
        for i, pose in enumerate(self.to_mujoco_qpos(qpos)):
            data.qpos[:] = pose
            mj.mj_forward(self.model, data)
            gp, gr = data.xpos[self.skeleton_body_ids], data.xquat[self.skeleton_body_ids]
            positions[i, 1], rotations[i, 1] = gp[0], gr[0]
            for bone in range(2, self.num_bones):
                parent = self.bone_parents[bone]
                inverse = quat.inv(gr[parent - 1])
                positions[i, bone] = quat.mul_vec(inverse, gp[bone - 1] - gp[parent - 1])
                rotations[i, bone] = quat.mul(inverse, gr[bone - 1])
        rotations = quat.unroll(rotations)
        return (positions[0], rotations[0]) if single else (positions, rotations)

    def decode_bones_to_qpos(self, positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        single = positions.ndim == 2
        if single:
            positions, rotations = positions[None], rotations[None]
        gr, gp = quat.fk(lrot=rotations, lpos=positions, parents=self.bone_parents)
        hinge = quat.mul(quat.inv(self.joint_rest_rotations), rotations[:, self.joint_bone_indices])
        projected = np.sum(hinge[..., 1:] * self.joint_axes, axis=-1)
        angles = 2 * np.arctan2(projected, hinge[..., 0])
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        result = np.concatenate((gr[:, 1], gp[:, 1], angles), axis=-1)
        return result[0] if single else result

    def mirror_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = self._validate_qpos(qpos)
        result = qpos.copy()
        result[..., [1, 3, 5]] *= -1
        result[..., 7:] = qpos[..., 7:][..., self.mirror_joint_indices] * self.mirror_joint_signs
        return result

    def metadata(self) -> dict:
        return {"robot_id": self.robot_id, "model_path": str(self.model_path),
                "joint_names": self.joint_names, "bone_names": self.bone_names,
                "bone_parents": self.bone_parents.tolist(), "body_names": self.body_names,
                "bone_left_foot": self.bone_left_foot, "bone_right_foot": self.bone_right_foot,
                "bone_hips": self.bone_hips}
