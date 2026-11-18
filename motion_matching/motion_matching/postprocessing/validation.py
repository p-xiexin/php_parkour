"""T2 pose constraints and exported NPZ consistency in the matching scene."""

import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

THRESHOLDS = dict(
    joint_limit_rad=1e-5,
    wrist_angle_rad=1e-6,
    wrist_velocity_rad_s=1e-6,
    penetration_m=0.002,
    root_quaternion_norm=1e-6,
)


class MotionConstraintVerifier:
    """Check the locked-wrist T2 against its original model and reference boxes."""

    def __init__(self, spec, reference_boxes):
        self.spec = spec
        root = ET.parse(spec.model_path).getroot()
        compiler = root.find("compiler")
        compiler.set(
            "meshdir",
            str((spec.model_path.parent / compiler.get("meshdir", ".")).resolve()),
        )
        world = root.find("worldbody")
        for i, box in enumerate(reference_boxes):
            ET.SubElement(
                world,
                "geom",
                name=f"validation_box_{i}",
                type="box",
                pos=" ".join(map(str, box["pos"])),
                quat=" ".join(map(str, box["quat"])),
                size=" ".join(map(str, np.asarray(box["size"]) / 2)),
                contype="1",
                conaffinity="1",
            )
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        model = self.model
        model.opt.ccd_tolerance = min(model.opt.ccd_tolerance, 1e-9)
        self.wrist_columns = [i for i, name in enumerate(spec.joint_names) if "_wrist_" in name]
        self.data = mj.MjData(model)

    def check(self, qpos, fps):
        """Return violations for PHP poses [root_wxyz, root_xyz, joint_angles]."""
        php = np.asarray(qpos, dtype=float)
        if not np.isfinite(php).all():
            return ["nonfinite_qpos"]
        if abs(np.linalg.norm(php[:, :4], axis=1) - 1).max() > THRESHOLDS["root_quaternion_norm"]:
            return ["nonunit_root_quaternion"]

        violations = []
        joints = php[:, 7:]
        limits = self.model.jnt_range[self.spec.joint_ids]
        excess = np.maximum(np.maximum(limits[:, 0] - joints, joints - limits[:, 1]), 0.0)
        if excess.max() > THRESHOLDS["joint_limit_rad"]:
            violations.append("hard_joint_limits")
        if abs(joints[:, self.wrist_columns]).max() > THRESHOLDS["wrist_angle_rad"]:
            violations.append("locked_wrist_angle")
        speed = np.diff(joints[:, self.wrist_columns], axis=0) * fps
        if abs(speed).max(initial=0.0) > THRESHOLDS["wrist_velocity_rad_s"]:
            violations.append("locked_wrist_velocity")

        penetration = 0.0
        for pose in self.spec.to_mujoco_qpos(php):
            self.data.qpos[:] = pose
            mj.mj_kinematics(self.model, self.data)
            mj.mj_comPos(self.model, self.data)
            mj.mj_collision(self.model, self.data)
            for contact in self.data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if self.model.geom_bodyid[a] or self.model.geom_bodyid[b]:
                    penetration = max(penetration, float(-contact.dist))
        if penetration > THRESHOLDS["penetration_m"]:
            violations.append("collision_penetration")
        return violations

    def check_npz(self, path):
        """Check physical poses, stored body transforms and locked-wrist velocities."""
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        if (
            arrays["joint_names"].tolist() != self.spec.joint_names
            or arrays["body_names"].tolist() != self.spec.body_names
            or arrays["root_body_name"].item() != self.spec.body_names[0]
        ):
            raise ValueError("NPZ joint/body ordering differs from the robot specification")
        php = np.concatenate(
            (arrays["body_quat_w"][:, 0], arrays["body_pos_w"][:, 0], arrays["joint_pos"]), axis=1
        )
        violations = self.check(php, float(arrays["fps"].item()))
        if any(
            value.dtype.kind in "fciu" and not np.isfinite(value).all() for value in arrays.values()
        ):
            violations.append("nonfinite_npz_arrays")
            return violations

        position_matches = rotation_matches = True
        for frame, pose in enumerate(self.spec.to_mujoco_qpos(php)):
            self.data.qpos[:] = pose
            mj.mj_kinematics(self.model, self.data)
            positions = self.data.xpos[self.spec.body_ids]
            rotations = self.data.xquat[self.spec.body_ids]
            # NPZ transforms are float32; allow serialization roundoff.
            position_matches &= np.allclose(
                arrays["body_pos_w"][frame],
                positions,
                rtol=4 * np.finfo(np.float32).eps,
                atol=1e-6,
            )
            saved = arrays["body_quat_w"][frame]
            rotation_matches &= (
                np.minimum(
                    np.linalg.norm(saved - rotations, axis=1),
                    np.linalg.norm(saved + rotations, axis=1),
                ).max()
                <= 1e-6
            )
        if not position_matches:
            violations.append("body_position_mismatch")
        if not rotation_matches:
            violations.append("body_rotation_mismatch")
        speed = abs(arrays["joint_vel"][:, self.wrist_columns]).max(initial=0.0)
        if speed > THRESHOLDS["wrist_velocity_rad_s"]:
            violations.append("saved_locked_wrist_velocity")
        return violations
