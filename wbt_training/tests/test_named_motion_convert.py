"""The fixed T2 motion layout retains all joints and adds the trunk root once."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wbt_training.config_values.t2_config import T2_48BODY_NAMES, T2_MODEL_PATH
from wbt_training.training_runs.motion_convert import convert
from wbt_training.utils import multi_motion_helpers


def test_t2_motion_root_and_idempotence(tmp_path, monkeypatch):
    source, output, again = [tmp_path / name for name in ("source.npz", "out.npz", "again.npz")]
    rng = np.random.default_rng(7)
    model = ET.parse(T2_MODEL_PATH).getroot()
    joints = [
        joint.get("name") for joint in model.findall("worldbody/body//joint")
        if joint.get("type") != "free"
    ]
    data = {
        "fps": np.array([50]),
        "joint_names": np.array(joints),
        "body_names": np.array(T2_48BODY_NAMES),
        "root_body_name": np.array("trunk"),
        "joint_pos": rng.normal(size=(4, 31)),
        "joint_vel": rng.normal(size=(4, 31)),
        "body_pos_w": rng.normal(size=(4, 48, 3)),
        "body_quat_w": np.tile([1., 0., 0., 0.], (4, 48, 1)),
        "body_lin_vel_w": rng.normal(size=(4, 48, 3)),
        "body_ang_vel_w": rng.normal(size=(4, 48, 3)),
        "vel_cmd": rng.normal(size=(4, 15)),
    }
    np.savez(source, **data)
    with pytest.raises(ValueError, match="G1 conversion"):
        convert(source, output, robot="g1")
    assert convert(source, output, robot="t2")
    assert convert(output, again, robot="t2")
    with np.load(output) as result, np.load(again) as repeated:
        np.testing.assert_array_equal(result["joint_pos"][:, :3], data["body_pos_w"][:, 0])
        np.testing.assert_array_equal(result["joint_pos"][:, 3:7], data["body_quat_w"][:, 0])
        np.testing.assert_array_equal(result["joint_pos"][:, 7:], data["joint_pos"])
        np.testing.assert_array_equal(result["joint_vel"][:, :3], data["body_lin_vel_w"][:, 0])
        np.testing.assert_array_equal(result["joint_vel"][:, 3:6], data["body_ang_vel_w"][:, 0])
        np.testing.assert_array_equal(result["joint_vel"][:, 6:], data["joint_vel"])
        for key in data:
            np.testing.assert_array_equal(result[key], repeated[key])
            if key not in ("joint_pos", "joint_vel"):
                np.testing.assert_array_equal(result[key], data[key])
    monkeypatch.setattr(multi_motion_helpers, "_TMP_DIR", tmp_path)
    combined = multi_motion_helpers.combine_motions([output, again])
    with np.load(combined) as result:
        assert result["root_body_name"].item() == "trunk"
        assert result["joint_pos"].shape == (8, 38)
        np.testing.assert_array_equal(np.flatnonzero(result["motion_ends"]), [3, 7])
    with np.load(again) as archive:
        different = dict(archive)
    different["root_body_name"] = np.array("waist_yaw_link")
    np.savez(again, **different)
    with pytest.raises(ValueError, match="root_body_name"):
        multi_motion_helpers.combine_motions([output, again])
    with pytest.raises(ValueError, match="trunk at body index 0"):
        convert(again, output, robot="t2")
    data["joint_pos"] = np.zeros((4, 3))
    np.savez(source, **data)
    with pytest.raises(ValueError, match="widths"):
        convert(source, output, robot="t2")
