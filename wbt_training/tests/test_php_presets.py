"""CPU regression checks for the public PHP presets and checkpoint paths."""

import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import tyro
from holosoma.config_values.logger import disabled, wandb
from holosoma.config_values.robot import g1_29dof as core_g1_robot
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt
from holosoma.managers.utils import resolve_callable
from holosoma.utils.helpers import get_class
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values import experiment, robot_config, t2_config

PHP_PRESETS = (
    "g1_wbt_terrain_ref",
    "g1_wbt_terrain_no_heightscan",
    "g1_wbt_terrain_ref_noscale",
    "g1_wbt_terrain_no_heightscan_noscale",
    "g1_wbt_terrain_ref_pelvis",
    "g1_wbt_terrain_no_heightscan_pelvis",
    "g1_wbt_terrain_warp_distill",
    "g1_wbt_terrain_warp_distill_finetune",
    "t2_wbt_terrain_ref",
    "t2_wbt_terrain_warp_distill",
    "t2_wbt_terrain_warp_distill_finetune",
)

PHP_CHECKPOINT_PATHS = (
    experiment.t2_wbt_terrain_ref.robot.control.implicit_actuator_builder,
    "wbt_training.config_values.t2_training.T2RsD435iConfig",
    "wbt_training.config_values.robot_config:build_g1_implicit_actuators",
    "wbt_training.config_values.terrain:height_scan",
    "wbt_training.config_values.terrain:apply_terrain_preprocess",
    "wbt_training.config_values.curriculum:RelaxTerminationThresholds",
    "wbt_training.config_values.motion_command:PhpMotionCommand",
    "wbt_training.config_values.motion_command:velocity_command",
    "wbt_training.config_values.motion_command:which_motion",
    "wbt_training.utils.student_teacher.PhpDistillation",
    "wbt_training.utils.student_teacher.PhpDistillationPPO",
)


def _php_paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _php_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _php_paths(item)
    elif isinstance(value, str) and value.startswith("wbt_training."):
        yield value


def test_package_import_is_inert_without_optional_dependencies():
    # -S excludes site packages, including Holosoma, torch, and the simulators.
    # Check sys.modules so silently caught ImportErrors cannot mask eager imports.
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            "-c",
            (
                "import sys; before = set(sys.modules); import wbt_training; "
                "assert set(sys.modules) - before == {'wbt_training'}"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_converter_works_with_only_numpy(tmp_path):
    # Ignore PYTHONPATH and site initialization. Expose this checkout and NumPy
    # explicitly, rejecting every other third-party import even when installed.
    script = dedent(
        """
        import sys
        from pathlib import Path

        sys.path[:0] = sys.argv[1:3]
        allowed = sys.stdlib_module_names | {"numpy", "wbt_training"}

        class NumpyOnlyImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] not in allowed:
                    raise ModuleNotFoundError(f"Dependency unavailable: {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, NumpyOnlyImports())

        import numpy as np
        from wbt_training.config_values.robot_config import (
            G1_29DOF_JOINT_NAMES,
            G1_32BODY_NAMES,
            ISAACLAB_TO_MUJOCO_DOF,
            build_g1_implicit_actuators,
        )
        from wbt_training.training_runs.motion_convert import convert

        assert len(G1_29DOF_JOINT_NAMES) == 29
        assert len(G1_32BODY_NAMES) == 32
        assert callable(build_g1_implicit_actuators)
        directory = Path(sys.argv[3])
        source = directory / "source.npz"
        target = directory / "converted.npz"
        joint_pos = np.arange(58).reshape(2, 29)
        joint_vel = joint_pos + 100
        body_pos = np.arange(180).reshape(2, 30, 3)
        body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (2, 30, 1))
        vel_cmd = np.eye(15)[:2]
        np.savez(
            source,
            fps=np.array([30]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros((2, 30, 3)),
            body_ang_vel_w=np.zeros((2, 30, 3)),
            vel_cmd=vel_cmd,
        )
        assert convert(str(source), str(target))
        with np.load(target, allow_pickle=False) as data:
            assert data["joint_pos"].shape == (2, 36)
            assert data["joint_vel"].shape == (2, 35)
            assert data["body_pos_w"].shape == (2, 32, 3)
            assert data["joint_names"].tolist() == G1_29DOF_JOINT_NAMES
            assert data["body_names"].tolist() == G1_32BODY_NAMES
            assert np.array_equal(data["joint_pos"][:, :3], body_pos[:, 0])
            assert np.array_equal(data["joint_pos"][:, 3:7], body_quat[:, 0])
            assert np.array_equal(data["joint_pos"][:, 7:], joint_pos[:, ISAACLAB_TO_MUJOCO_DOF])
            assert np.array_equal(data["joint_vel"][:, 6:], joint_vel[:, ISAACLAB_TO_MUJOCO_DOF])
            assert np.array_equal(data["body_pos_w"][:, 7], body_pos[:, 18])
            assert np.array_equal(data["body_pos_w"][:, 14], body_pos[:, 19])
            assert np.array_equal(data["vel_cmd"], vel_cmd)
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(Path(__file__).resolve().parents[2]),
            str(Path(np.__file__).resolve().parents[1]),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_registry_contains_all_php_presets():
    assert set(experiment.DEFAULTS) == set(PHP_PRESETS)
    assert experiment.g1_29dof_wbt is g1_29dof_wbt


@pytest.mark.parametrize("lock_wrists", [False, True])
def test_presets_do_not_parse_t2_assets(lock_wrists):
    script = dedent("""
        import os
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from unittest.mock import patch

        import tyro
        from holosoma.utils.tyro_utils import TYRO_CONIFG

        original_parse = ET.parse
        def without_t2(path, *args, **kwargs):
            if Path(path).name == "t2_31dof_sphere_locked.xml":
                raise FileNotFoundError("T2 assets are not installed")
            return original_parse(path, *args, **kwargs)

        with patch.object(ET, "parse", side_effect=without_t2):
            from wbt_training.config_values.experiment import AnnotatedExperimentConfig
            t2_dofs = 25 if os.environ["T2_LOCK_WRISTS"] == "1" else 31
            for preset, dofs in (
                ("g1-wbt-terrain-ref-noscale", 29),
                ("g1-wbt-terrain-warp-distill-finetune", 29),
                ("t2-wbt-terrain-ref", t2_dofs),
                ("t2-wbt-terrain-warp-distill-finetune", t2_dofs),
            ):
                cfg = tyro.cli(
                    AnnotatedExperimentConfig, args=["exp:" + preset], config=TYRO_CONIFG
                )
                assert cfg.robot.actions_dim == dofs
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        env={**os.environ, "T2_LOCK_WRISTS": str(int(lock_wrists))},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("robot", [t2_config.t2_sphere_locked, t2_config.t2_sphere_unlocked])
def test_t2_anatomy_and_wrist_action_space(robot):
    from wbt_training.config_values.t2_training import T2RsD435iConfig

    presets = {name: cfg for name, cfg in experiment.DEFAULTS.items() if name.startswith("t2_")}
    for cfg in presets.values():
        assert robot.actions_dim == robot.dof_obs_size == len(robot.dof_names)
        assert robot.num_bodies == len(robot.body_names) == 48
        assert sum("wrist" in name for name in robot.dof_names) == robot.actions_dim - 25
        motion = cfg.command.setup_terms["motion_command"].params["motion_config"]
        assert motion.motion_file == ""  # The registry supplies T2 data during preprocessing.
        assert set(motion.body_names_to_track) <= set(robot.body_names)
        assert motion.body_name_ref == ["trunk"]
        assert "left_elbow_yaw_link" in motion.body_names_to_track
        assert "left_elbow_pitch_link" not in motion.body_names_to_track
        assert set(robot.init_state.default_joint_angles) == set(robot.dof_names)
        for name in ("dof_pos_lower_limit_list", "dof_pos_upper_limit_list", "dof_vel_limit_list",
                     "dof_effort_limit_list", "dof_armature_list", "dof_joint_friction_list"):
            assert len(getattr(robot, name)) == robot.actions_dim
    camera = T2RsD435iConfig()
    assert set(camera.base_link_frame.values()) <= set(robot.body_names)
    assert set(camera.ray_cast_bodies) <= set(robot.body_names)
    assert {"left_hand_ee_link", "right_hand_ee_link"} <= set(camera.ray_cast_bodies)
    depth = presets["t2_wbt_terrain_warp_distill"].observation.groups["depth_camera"].terms["depth_cam"]
    assert depth.params["camera_sensor_cfg"] == "wbt_training.config_values.t2_training.T2RsD435iConfig"


@pytest.mark.parametrize("lock_wrists", [False, True])
def test_saved_t2_wrist_mode_survives_changed_default(tmp_path, lock_wrists):
    import torch

    robot = t2_config.t2_sphere_locked if lock_wrists else t2_config.t2_sphere_unlocked
    cfg = replace(experiment.t2_wbt_terrain_ref, robot=robot)
    path = tmp_path / "checkpoint.pt"
    torch.save({"experiment_config": cfg.to_serializable_dict()}, path)
    script = dedent("""
        import sys
        from dataclasses import asdict
        from holosoma.utils.eval_utils import CheckpointConfig
        from wbt_training.config_values import t2_config, experiment
        from wbt_training.utils.checkpoint import load_saved_experiment_config

        saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=sys.argv[1]))
        expected = t2_config.t2_sphere_locked if sys.argv[2] == "1" else t2_config.t2_sphere_unlocked
        assert experiment.t2_wbt_terrain_ref.robot.actions_dim != expected.actions_dim
        assert asdict(saved.robot) == asdict(expected)
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path), str(int(lock_wrists))],
        env={**os.environ, "T2_LOCK_WRISTS": str(int(not lock_wrists))},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_t2_registry_supplies_reference_motion(tmp_path, monkeypatch):
    from wbt_training.config_values.terrain import apply_terrain_preprocess
    from wbt_training.utils import multi_motion_helpers

    bundle, scratch = tmp_path / "motions", tmp_path / "combined"
    bundle.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(multi_motion_helpers, "_TMP_DIR", scratch)
    preset = experiment.t2_wbt_terrain_ref
    joints, bodies = preset.robot.dof_names, preset.robot.body_names
    np.savez(
        bundle / "walk_motion.npz",
        fps=np.array(50), joint_names=np.array(joints), body_names=np.array(bodies),
        joint_pos=np.tile([0., 0., 1., 1., 0., 0., 0.] + [0.] * len(joints), (2, 1)),
        joint_vel=np.zeros((2, len(joints) + 6)),
        body_pos_w=np.zeros((2, len(bodies), 3)),
        body_quat_w=np.tile([1., 0., 0., 0.], (2, len(bodies), 1)),
        body_lin_vel_w=np.zeros((2, len(bodies), 3)),
        body_ang_vel_w=np.zeros((2, len(bodies), 3)),
    )
    config = replace(
        preset, logger=disabled,
        training=replace(preset.training, registry_name=bundle.as_uri()),
    )
    processed = apply_terrain_preprocess(config, add_onpath_obstacle=False)
    motion = processed.command.setup_terms["motion_command"].params["motion_config"]
    with np.load(motion.motion_file) as data:
        np.testing.assert_array_equal(data["joint_names"], joints)
        assert data["joint_pos"].shape == (2, len(joints) + 7)
    assert motion.body_name_ref == ["trunk"]


def test_t2_preserves_default_php_training_recipe():
    presets = experiment.DEFAULTS
    for g1, t2 in (
        (experiment.g1_wbt_terrain_ref_noscale, presets["t2_wbt_terrain_ref"]),
        (experiment.g1_wbt_terrain_warp_distill, presets["t2_wbt_terrain_warp_distill"]),
        (experiment.g1_wbt_terrain_warp_distill_finetune,
         presets["t2_wbt_terrain_warp_distill_finetune"]),
    ):
        assert asdict(t2.algo) == asdict(g1.algo)
        assert asdict(t2.curriculum) == asdict(g1.curriculum)
        assert asdict(replace(t2.training, name=g1.training.name)) == asdict(g1.training)
        g1_motion = g1.command.setup_terms["motion_command"].params["motion_config"]
        t2_motion = t2.command.setup_terms["motion_command"].params["motion_config"]
        assert asdict(t2_motion.noise_to_initial_pose) == asdict(g1_motion.noise_to_initial_pose)
        if "teacher_obs" in t2.observation.groups:
            g1_routing = g1.observation.groups["teacher_obs"].terms["which_motion"]
            t2_routing = t2.observation.groups["teacher_obs"].terms["which_motion"]
            expected_params = dict(g1_routing.params)
            expected_params["bad_motion_body_pos_body_names"] = (
                "left_ankle_roll_link", "right_ankle_roll_link",
                "left_wrist_roll_link", "right_wrist_roll_link",
            )
            assert asdict(t2_routing) == asdict(replace(g1_routing, params=expected_params))
            assert t2_routing.func.endswith(":which_motion")
            assert set(t2_routing.params["bad_motion_body_pos_body_names"]) <= set(
                t2_motion.body_names_to_track
            )


@pytest.mark.parametrize("name", PHP_PRESETS)
def test_cli_selects_presets_with_wandb_logging_by_default(name):
    cfg = tyro.cli(
        experiment.AnnotatedExperimentConfig,
        args=[f"exp:{name.replace('_', '-')}"], config=TYRO_CONIFG,
    )
    assert cfg == experiment.DEFAULTS[name]
    assert cfg.logger == wandb
    motion_config = cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.resample_on_motion_end is True


@pytest.mark.parametrize("name", PHP_PRESETS)
def test_wandb_logging_can_be_disabled_explicitly(name):
    cfg = tyro.cli(
        experiment.AnnotatedExperimentConfig,
        args=[f"exp:{name.replace('_', '-')}", "logger:disabled"], config=TYRO_CONIFG,
    )
    assert cfg == replace(experiment.DEFAULTS[name], logger=disabled)


def test_serialized_presets_use_the_supported_php_checkpoint_paths():
    paths = {
        path for cfg in experiment.DEFAULTS.values()
        for path in _php_paths(asdict(cfg))
    }
    assert paths == set(PHP_CHECKPOINT_PATHS)


@pytest.mark.parametrize("path", PHP_CHECKPOINT_PATHS + (
    "wbt_training.config_values.motion_command:motion_expert_index",  # Old T2 checkpoints.
    "wbt_training.config_values.t2_config:build_t2_implicit_actuators",
    "wbt_training.config_values.t2_config:build_t2_unlocked_implicit_actuators",
))
def test_supported_checkpoint_paths_resolve(path):
    # Use the same resolvers as manager terms and algorithm construction.
    resolved = resolve_callable(path) if ":" in path else get_class(path)
    assert callable(resolved)


def test_converter_joint_and_body_orderings_are_unchanged():
    assert robot_config.G1_29DOF_JOINT_NAMES == core_g1_robot.dof_names
    assert robot_config.G1_32BODY_NAMES == core_g1_robot.body_names
    np.testing.assert_array_equal(
        robot_config.ISAACLAB_TO_MUJOCO_DOF,
        [
            0,
            3,
            6,
            9,
            13,
            17,
            1,
            4,
            7,
            10,
            14,
            18,
            2,
            5,
            8,
            11,
            15,
            19,
            21,
            23,
            25,
            27,
            12,
            16,
            20,
            22,
            24,
            26,
            28,
        ],
    )
