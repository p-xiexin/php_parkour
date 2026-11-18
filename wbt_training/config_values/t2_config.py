"""T2 sphere-hand robot configurations with optional fixed wrists.

Joint and body orderings match the versioned MJCF used by the motion data.
PD gains retain the 10 Hz / damping-ratio 2 simulation baseline.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

from holosoma.config_values.robot import g1_29dof

T2_MODEL_PATH = Path(__file__).resolve().parents[2] / (
    "motion_matching/motion_matching/assets/robots/booster_t2/mjcf/"
    "t2_31dof_sphere_locked.xml"
)
T2_IK_ONLY_BODIES = {"left_hand_contact_link", "right_hand_contact_link"}
T2_LOCK_WRISTS = bool(int(os.environ.get("T2_LOCK_WRISTS", "0")))
T2_VELOCITY_LIMIT = 20.0
T2_ASSET_DIR = Path(
    os.environ.get("PHP_T2_ASSET_DIR", str(Path(__file__).resolve().parents[2] / ".local/t2_assets"))
)

# Both wrist modes retain the same body ordering.
T2_48BODY_NAMES = [
    "trunk",
    "head_yaw_link",
    "head_pitch_link",
    "trunk_center_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_wrist_roll_link",
    "left_hand_ee_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_hand_ee_link",
    "waist_pitch_link",
    "waist_roll_link",
    "waist_yaw_link",
    "waist_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_foot_heel_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_foot_sphere_3_link",
    "left_foot_sphere_4_link",
    "left_foot_sphere_5_link",
    "left_foot_sphere_1_link",
    "left_foot_sphere_2_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_foot_heel_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_foot_sphere_3_link",
    "right_foot_sphere_4_link",
    "right_foot_sphere_5_link",
    "right_foot_sphere_1_link",
    "right_foot_sphere_2_link",
]

T2_WRIST_JOINT_NAMES = [
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
]

T2_31DOF_JOINT_NAMES = [
    "aa_head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
T2_25DOF_JOINT_NAMES = [name for name in T2_31DOF_JOINT_NAMES if name not in T2_WRIST_JOINT_NAMES]

T2_JOINT_LIMITS = {
    "aa_head_yaw_joint": (-1.13446, 1.13446),
    "head_pitch_joint": (-0.349066, 0.436332),
    "left_shoulder_pitch_joint": (-3.05433, 1.74533),
    "left_shoulder_roll_joint": (-1.55334, 1.55334),
    "left_elbow_pitch_joint": (-2.0944, 2.0944),
    "left_elbow_yaw_joint": (-2.13803, 0.0),
    "left_wrist_pitch_joint": (-2.0944, 2.0944),
    "left_wrist_yaw_joint": (-0.783653, 0.71384),
    "left_wrist_roll_joint": (-1.39626, 0.959931),
    "right_shoulder_pitch_joint": (-3.05433, 1.74533),
    "right_shoulder_roll_joint": (-1.55334, 1.55334),
    "right_elbow_pitch_joint": (-2.0944, 2.0944),
    "right_elbow_yaw_joint": (0.0, 2.13803),
    "right_wrist_pitch_joint": (-2.0944, 2.0944),
    "right_wrist_yaw_joint": (-0.71384, 0.783653),
    "right_wrist_roll_joint": (-0.959931, 1.39626),
    "waist_pitch_joint": (-0.785398, 0.593412),
    "waist_roll_joint": (-0.392699, 0.392699),
    "waist_yaw_joint": (-2.44346, 2.44346),
    "left_hip_pitch_joint": (-2.96706, 2.26893),
    "left_hip_roll_joint": (-0.610865, 1.5708),
    "left_hip_yaw_joint": (-2.79253, 2.79253),
    "left_knee_pitch_joint": (0.0, 2.61799),
    "left_ankle_pitch_joint": (-0.6457718232379019, 0.33161255787892263),
    "left_ankle_roll_joint": (-0.349066, 0.349066),
    "right_hip_pitch_joint": (-2.96706, 2.26893),
    "right_hip_roll_joint": (-1.5708, 0.610865),
    "right_hip_yaw_joint": (-2.79253, 2.79253),
    "right_knee_pitch_joint": (0.0, 2.61799),
    "right_ankle_pitch_joint": (-0.6457718232379019, 0.33161255787892263),
    "right_ankle_roll_joint": (-0.349066, 0.349066),
}

T2_ARMATURE = {
    "aa_head_yaw_joint": 0.007209,
    "head_pitch_joint": 0.007209,
    "left_shoulder_pitch_joint": 0.017,
    "left_shoulder_roll_joint": 0.0137,
    "left_elbow_pitch_joint": 0.0137,
    "left_elbow_yaw_joint": 0.0137,
    "left_wrist_pitch_joint": 0.007209,
    "left_wrist_yaw_joint": 0.007209,
    "left_wrist_roll_joint": 0.007209,
    "right_shoulder_pitch_joint": 0.017,
    "right_shoulder_roll_joint": 0.0137,
    "right_elbow_pitch_joint": 0.0137,
    "right_elbow_yaw_joint": 0.0137,
    "right_wrist_pitch_joint": 0.007209,
    "right_wrist_yaw_joint": 0.007209,
    "right_wrist_roll_joint": 0.007209,
    "waist_pitch_joint": 0.0443514895,
    "waist_roll_joint": 0.08868658010000001,
    "waist_yaw_joint": 0.0281,
    "left_hip_pitch_joint": 0.0281,
    "left_hip_roll_joint": 0.0281,
    "left_hip_yaw_joint": 0.0281,
    "left_knee_pitch_joint": 0.0281,
    "left_ankle_pitch_joint": 0.0407491567,
    "left_ankle_roll_joint": 0.0204327143,
    "right_hip_pitch_joint": 0.0281,
    "right_hip_roll_joint": 0.0281,
    "right_hip_yaw_joint": 0.0281,
    "right_knee_pitch_joint": 0.0281,
    "right_ankle_pitch_joint": 0.0407491567,
    "right_ankle_roll_joint": 0.0204327143,
}

T2_EFFORT_LIMITS = {
    "aa_head_yaw_joint": 21.0,
    "head_pitch_joint": 21.0,
    "left_shoulder_pitch_joint": 70.0,
    "left_shoulder_roll_joint": 60.0,
    "left_elbow_pitch_joint": 60.0,
    "left_elbow_yaw_joint": 60.0,
    "left_wrist_pitch_joint": 21.0,
    "left_wrist_yaw_joint": 21.0,
    "left_wrist_roll_joint": 21.0,
    "right_shoulder_pitch_joint": 70.0,
    "right_shoulder_roll_joint": 60.0,
    "right_elbow_pitch_joint": 60.0,
    "right_elbow_yaw_joint": 60.0,
    "right_wrist_pitch_joint": 21.0,
    "right_wrist_yaw_joint": 21.0,
    "right_wrist_roll_joint": 21.0,
    "waist_pitch_joint": 60.0,
    "waist_roll_joint": 60.0,
    "waist_yaw_joint": 140.0,
    "left_hip_pitch_joint": 140.0,
    "left_hip_roll_joint": 140.0,
    "left_hip_yaw_joint": 140.0,
    "left_knee_pitch_joint": 140.0,
    "left_ankle_pitch_joint": 60.0,
    "left_ankle_roll_joint": 60.0,
    "right_hip_pitch_joint": 140.0,
    "right_hip_roll_joint": 140.0,
    "right_hip_yaw_joint": 140.0,
    "right_knee_pitch_joint": 140.0,
    "right_ankle_pitch_joint": 60.0,
    "right_ankle_roll_joint": 60.0,
}

_NATURAL_FREQ = 20.0 * math.pi
_DAMPING_RATIO = 2.0
_STIFFNESS = {name: armature * _NATURAL_FREQ**2 for name, armature in T2_ARMATURE.items()}
_DAMPING = {
    name: 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
    for name, armature in T2_ARMATURE.items()
}


def _build_t2_actuators(joint_names):
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        "joints": ImplicitActuatorCfg(
            joint_names_expr=joint_names,
            stiffness={name: _STIFFNESS[name] for name in joint_names},
            damping={name: _DAMPING[name] for name in joint_names},
            effort_limit_sim={name: T2_EFFORT_LIMITS[name] for name in joint_names},
            velocity_limit_sim=T2_VELOCITY_LIMIT,
            armature={name: T2_ARMATURE[name] for name in joint_names},
        )
    }


def build_t2_implicit_actuators():
    """Keep the 25-joint drive path used by existing locked-wrist checkpoints."""
    return _build_t2_actuators(T2_25DOF_JOINT_NAMES)


def build_t2_unlocked_implicit_actuators():
    return _build_t2_actuators(T2_31DOF_JOINT_NAMES)


_LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
]

_RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
]

_LOWER_BODY = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

_UPPER_BODY = [name for name in T2_25DOF_JOINT_NAMES if name not in _LOWER_BODY]
_DEFAULT_JOINT_ANGLES = {
    **dict.fromkeys(T2_25DOF_JOINT_NAMES, 0.0),
    "left_hip_pitch_joint": -0.2,
    "left_knee_pitch_joint": 0.4,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.2,
    "right_knee_pitch_joint": 0.4,
    "right_ankle_pitch_joint": -0.2,
}

t2_sphere_locked = replace(
    g1_29dof,
    num_bodies=len(T2_48BODY_NAMES),
    dof_obs_size=len(T2_25DOF_JOINT_NAMES),
    actions_dim=len(T2_25DOF_JOINT_NAMES),
    dof_names=T2_25DOF_JOINT_NAMES,
    body_names=T2_48BODY_NAMES,
    torso_name="trunk",
    knee_name="knee_pitch_link",
    key_bodies=["left_foot_sphere_5_link", "right_foot_sphere_5_link"],
    foot_body_name="ankle_roll_link",
    foot_height_name="foot_sphere_5_link",
    upper_dof_names=_UPPER_BODY,
    lower_dof_names=_LOWER_BODY,
    upper_left_arm_dof_names=_LEFT_ARM,
    upper_right_arm_dof_names=_RIGHT_ARM,
    arm_dof_names=_LEFT_ARM + _RIGHT_ARM,
    left_arm_dof_names=_LEFT_ARM,
    right_arm_dof_names=_RIGHT_ARM,
    left_ankle_dof_names=["left_ankle_pitch_joint", "left_ankle_roll_joint"],
    right_ankle_dof_names=["right_ankle_pitch_joint", "right_ankle_roll_joint"],
    knee_dof_names=["left_knee_pitch_joint", "right_knee_pitch_joint"],
    hips_dof_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
    ],
    waist_dof_names=["waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"],
    dof_pos_lower_limit_list=[T2_JOINT_LIMITS[name][0] for name in T2_25DOF_JOINT_NAMES],
    dof_pos_upper_limit_list=[T2_JOINT_LIMITS[name][1] for name in T2_25DOF_JOINT_NAMES],
    dof_vel_limit_list=[T2_VELOCITY_LIMIT] * len(T2_25DOF_JOINT_NAMES),
    dof_effort_limit_list=[T2_EFFORT_LIMITS[name] for name in T2_25DOF_JOINT_NAMES],
    dof_armature_list=[T2_ARMATURE[name] for name in T2_25DOF_JOINT_NAMES],
    dof_joint_friction_list=[0.0] * len(T2_25DOF_JOINT_NAMES),
    randomize_link_body_names=["trunk"],
    terminate_after_contacts_on=["trunk"],
    penalize_contacts_on=["trunk", "hip", "knee"],
    symmetry_joint_names=None,
    flip_sign_joint_names=None,
    init_state=replace(g1_29dof.init_state, pos=[0.0, 0.0, 1.0], default_joint_angles=_DEFAULT_JOINT_ANGLES),
    control=replace(
        g1_29dof.control,
        stiffness={name: _STIFFNESS[name] for name in T2_25DOF_JOINT_NAMES},
        damping={name: _DAMPING[name] for name in T2_25DOF_JOINT_NAMES},
        action_scale=1.0,
        action_scales_by_effort_limit_over_p_gain=False,
        control_mode="implicit_position_target",
        implicit_actuator_builder="wbt_training.config_values.t2_config:build_t2_implicit_actuators",
    ),
    asset=replace(
        g1_29dof.asset,
        asset_root=str(T2_ASSET_DIR),
        urdf_file="t2_sphere_locked.urdf",
        xml_file="t2_sphere_locked.xml",
        usd_file=None,
        robot_type="t2_sphere_locked",
        collapse_fixed_joints=False,
        replace_cylinder_with_capsule=False,
        enable_self_collisions=True,
    ),
    bridge=replace(g1_29dof.bridge, sdk_type="booster"),
)

t2_sphere_unlocked = replace(
    t2_sphere_locked,
    dof_obs_size=31,
    actions_dim=31,
    dof_names=T2_31DOF_JOINT_NAMES,
    upper_dof_names=[name for name in T2_31DOF_JOINT_NAMES if name not in _LOWER_BODY],
    upper_left_arm_dof_names=_LEFT_ARM + T2_WRIST_JOINT_NAMES[:3],
    upper_right_arm_dof_names=_RIGHT_ARM + T2_WRIST_JOINT_NAMES[3:],
    arm_dof_names=_LEFT_ARM + T2_WRIST_JOINT_NAMES[:3] + _RIGHT_ARM + T2_WRIST_JOINT_NAMES[3:],
    left_arm_dof_names=_LEFT_ARM + T2_WRIST_JOINT_NAMES[:3],
    right_arm_dof_names=_RIGHT_ARM + T2_WRIST_JOINT_NAMES[3:],
    dof_pos_lower_limit_list=[T2_JOINT_LIMITS[name][0] for name in T2_31DOF_JOINT_NAMES],
    dof_pos_upper_limit_list=[T2_JOINT_LIMITS[name][1] for name in T2_31DOF_JOINT_NAMES],
    dof_vel_limit_list=[T2_VELOCITY_LIMIT] * 31,
    dof_effort_limit_list=[T2_EFFORT_LIMITS[name] for name in T2_31DOF_JOINT_NAMES],
    dof_armature_list=[T2_ARMATURE[name] for name in T2_31DOF_JOINT_NAMES],
    dof_joint_friction_list=[0.0] * 31,
    init_state=replace(
        t2_sphere_locked.init_state,
        default_joint_angles={**_DEFAULT_JOINT_ANGLES, **dict.fromkeys(T2_WRIST_JOINT_NAMES, 0.0)},
    ),
    control=replace(
        t2_sphere_locked.control,
        stiffness=_STIFFNESS,
        damping=_DAMPING,
        implicit_actuator_builder="wbt_training.config_values.t2_config:build_t2_unlocked_implicit_actuators",
    ),
    asset=replace(
        t2_sphere_locked.asset,
        urdf_file="t2_sphere_unlocked.urdf",
        xml_file="t2_sphere_unlocked.xml",
        robot_type="t2_sphere_unlocked",
    ),
)

# Resolve only new-run defaults; saved configurations retain their own variant.
t2_robot = t2_sphere_locked if T2_LOCK_WRISTS else t2_sphere_unlocked
