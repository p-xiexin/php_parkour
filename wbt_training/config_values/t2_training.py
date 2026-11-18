"""T2 camera, reference anatomy and training-manager configurations."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg, MotionConfig
from holosoma.config_types.observation import (
    ObservationManagerCfg,
    ObsGroupCfg,
    ObsTermCfg,
)
from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.sensors.warp.camera_config.d435i_depth_config import RsD435iConfig

from wbt_training.config_values.depth_distillation import (
    WBT_DEPTH_PARAMS,
    WBT_DISTILL_GATE_PARAMS,
    wbt_distill_actor_obs,
    wbt_distill_critic_obs,
    wbt_distill_relax_termination_term,
    wbt_distill_teacher_obs,
)
from wbt_training.config_values.t2_config import T2_ASSET_DIR
from wbt_training.config_values.terrain import (
    WBT_MOTION_PARAMS,
    wbt_bad_tracking,
    wbt_motion_command,
    wbt_terrain_curriculum,
    wbt_terrain_reward_terms,
    wbt_timeout,
    wbt_undesired_contacts,
)


class T2RsD435iConfig(RsD435iConfig):
    """Virtual front camera; shared by training and MuJoCo deployment."""

    asset_meshes_root = str(T2_ASSET_DIR / "depth_meshes")
    base_link_frame = {"cam_front_depth": "trunk"}
    offset = {"cam_front_depth": {"offset_pos": (0.18, 0.0, 0.35), "offset_rot": (0.0, 27.0, 0.0)}}
    ray_cast_bodies = {
        "trunk": "trunk.obj",
        "head_yaw_link": "head_yaw_link.obj",
        "head_pitch_link": "head_pitch_link.obj",
        "left_shoulder_pitch_link": "left_shoulder_pitch_link.obj",
        "left_shoulder_roll_link": "left_shoulder_roll_link.obj",
        "left_elbow_pitch_link": "left_elbow_pitch_link.obj",
        "left_elbow_yaw_link": "left_elbow_yaw_link.obj",
        "left_wrist_pitch_link": "left_wrist_pitch_link.obj",
        "left_wrist_yaw_link": "left_wrist_yaw_link.obj",
        "left_hand_ee_link": "left_hand_ee_link.obj",
        "right_shoulder_pitch_link": "right_shoulder_pitch_link.obj",
        "right_shoulder_roll_link": "right_shoulder_roll_link.obj",
        "right_elbow_pitch_link": "right_elbow_pitch_link.obj",
        "right_elbow_yaw_link": "right_elbow_yaw_link.obj",
        "right_wrist_pitch_link": "right_wrist_pitch_link.obj",
        "right_wrist_yaw_link": "right_wrist_yaw_link.obj",
        "right_hand_ee_link": "right_hand_ee_link.obj",
        "waist_pitch_link": "waist_pitch_link.obj",
        "waist_roll_link": "waist_roll_link.obj",
        "waist_yaw_link": "waist_yaw_link.obj",
        "left_hip_pitch_link": "left_hip_pitch_link.obj",
        "left_hip_roll_link": "left_hip_roll_link.obj",
        "left_hip_yaw_link": "left_hip_yaw_link.obj",
        "left_knee_pitch_link": "left_knee_pitch_link.obj",
        "left_ankle_pitch_link": "left_ankle_pitch_link.obj",
        "left_ankle_roll_link": "left_ankle_roll_link.obj",
        "right_hip_pitch_link": "right_hip_pitch_link.obj",
        "right_hip_roll_link": "right_hip_roll_link.obj",
        "right_hip_yaw_link": "right_hip_yaw_link.obj",
        "right_knee_pitch_link": "right_knee_pitch_link.obj",
        "right_ankle_pitch_link": "right_ankle_pitch_link.obj",
        "right_ankle_roll_link": "right_ankle_roll_link.obj",
    }
    add_offpath_obstacle = False
    offpath_obstacle_meshes_root = None
    offpath_obstacle_bodies = {}


_TRACKED_BODIES = [
    "waist_yaw_link",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "trunk",
    "left_shoulder_roll_link",
    "left_elbow_yaw_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_yaw_link",
    "right_wrist_roll_link",
]
_ENDPOINTS = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
]

t2_motion_config = MotionConfig(
    **WBT_MOTION_PARAMS,
    motion_file="",  # Filled from the training registry by apply_terrain_preprocess.
    body_name_ref=["trunk"],
    body_names_to_track=_TRACKED_BODIES,
)
t2_terrain_command = replace(
    wbt_motion_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": t2_motion_config},
        ),
    },
)
t2_terrain_reward = RewardManagerCfg(
    terms={
        **wbt_terrain_reward_terms,
        "undesired_contacts": replace(
            wbt_undesired_contacts,
            params={
                **wbt_undesired_contacts.params,
                "undesired_contacts_body_names": r"^(?!.*foot_sphere_)(?!.*ankle_roll_link$)(?!.*wrist_)(?!.*hand_ee_link$).+$",
            },
        ),
    },
)
t2_terrain_termination = TerminationManagerCfg(
    terms={
        "timeout": wbt_timeout,
        "bad_tracking": replace(
            wbt_bad_tracking,
            params={
                **wbt_bad_tracking.params,
                "body_names_to_track": _TRACKED_BODIES,
                "bad_motion_body_pos_body_names": _ENDPOINTS,
            },
        ),
    },
)

t2_distill_command = replace(
    wbt_motion_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="wbt_training.config_values.motion_command:PhpMotionCommand",
            params={
                "motion_config": replace(t2_motion_config, use_adaptive_timesteps_sampler=False),
                **WBT_DISTILL_GATE_PARAMS,
            },
        ),
    },
)
t2_distill_observation = ObservationManagerCfg(
    groups={
        "actor_obs": wbt_distill_actor_obs,
        "critic_obs": wbt_distill_critic_obs,
        "teacher_obs": replace(
            wbt_distill_teacher_obs,
            terms={
                **wbt_distill_teacher_obs.terms,
                "which_motion": replace(
                    wbt_distill_teacher_obs.terms["which_motion"],
                    params={
                        **wbt_distill_teacher_obs.terms["which_motion"].params,
                        "bad_motion_body_pos_body_names": tuple(_ENDPOINTS),
                    },
                ),
            },
        ),
        "depth_camera": ObsGroupCfg(
            concatenate=False, enable_noise=False, history_length=1,
            terms={
                "depth_cam": ObsTermCfg(
                    func="holosoma.managers.observation.terms.depth:WarpDepthImageObsTerm",
                    scale=1.0, noise=0.0,
                    params={
                        **WBT_DEPTH_PARAMS,
                        "camera_sensor_cfg": "wbt_training.config_values.t2_training.T2RsD435iConfig",
                    },
                ),
            },
        ),
    },
)


t2_distill_curriculum = replace(
    wbt_terrain_curriculum,
    step_terms={
        **wbt_terrain_curriculum.step_terms,
        "relax_termination": wbt_distill_relax_termination_term,
    },
)
