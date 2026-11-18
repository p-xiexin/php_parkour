"""Cross-engine kinematics checks for both T2 wrist modes."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from wbt_training.config_values import t2_config
from wbt_training.config_values.t2_training import T2RsD435iConfig
from wbt_training.training_runs.t2_asset import (
    DEFAULT_MODEL,
    IK_ONLY_BODIES,
    export_mjcf,
    export_urdf,
)


def test_static_config_matches_versioned_model():
    source = ET.parse(DEFAULT_MODEL).getroot()
    locked = {joint.get("joint1") for joint in source.findall("equality/joint")}
    joints = [
        joint for joint in source.findall("worldbody/body//joint")
        if joint.get("type") != "free"
    ]
    bodies = source.findall("worldbody//body")
    assert t2_config.T2_31DOF_JOINT_NAMES == [joint.get("name") for joint in joints]
    assert set(t2_config.T2_WRIST_JOINT_NAMES) == locked
    assert t2_config.T2_25DOF_JOINT_NAMES == [
        joint.get("name") for joint in joints if joint.get("name") not in locked
    ]
    assert t2_config.T2_48BODY_NAMES == [
        body.get("name") for body in bodies if body.get("name") not in IK_ONLY_BODIES
    ]
    for joint in joints:
        name = joint.get("name")
        assert t2_config.T2_JOINT_LIMITS[name] == tuple(map(float, joint.get("range").split()))
        assert t2_config.T2_ARMATURE[name] == float(joint.get("armature"))
        assert t2_config.T2_EFFORT_LIMITS[name] == max(
            abs(float(value)) for value in joint.get("actuatorfrcrange").split()
        )
    assert T2RsD435iConfig.ray_cast_bodies == {
        body.get("name"): body.get("name") + ".obj"
        for body in bodies
        if any(geom.get("class") == "visual" and geom.get("mesh") for geom in body.findall("geom"))
    }


@pytest.mark.parametrize("lock_wrists", [False, True])
def test_urdf_matches_mjcf(tmp_path, lock_wrists):
    mj = pytest.importorskip("mujoco")
    yourdfpy = pytest.importorskip("yourdfpy")
    path = export_urdf(DEFAULT_MODEL, tmp_path / "robot.urdf", lock_wrists=lock_wrists)
    urdf = yourdfpy.URDF.load(path, load_meshes=False, build_scene_graph=True)
    model = mj.MjModel.from_xml_path(str(DEFAULT_MODEL))
    data = mj.MjData(model)
    dofs = 25 if lock_wrists else 31
    assert len(urdf.actuated_joint_names) == dofs
    assert sum("wrist" in name for name in urdf.actuated_joint_names) == (0 if lock_wrists else 6)
    deployed = mj.MjModel.from_xml_path(str(export_mjcf(
        DEFAULT_MODEL, tmp_path / "robot.xml", lock_wrists=lock_wrists
    )))
    deployed_data = mj.MjData(deployed)
    assert deployed.nu == dofs
    assert deployed.nq == dofs + 7
    assert deployed.neq == 0
    assert deployed.nbody == 49
    np.testing.assert_allclose(deployed.body_mass.sum(), model.body_mass.sum())
    rng = np.random.default_rng(42)
    for _ in range(10):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = 0
        values = {}
        for joint in range(1, model.njnt):
            name = model.joint(joint).name
            if lock_wrists and "wrist" in name:
                continue
            lo, hi = model.jnt_range[joint]
            value = rng.uniform(lo, hi)
            data.qpos[model.jnt_qposadr[joint]] = value
            values[name] = value
        urdf.update_cfg(values)
        mj.mj_kinematics(model, data)
        deployed_data.qpos[:7] = data.qpos[:7]
        for name, value in values.items():
            deployed_data.qpos[deployed.joint(name).qposadr] = value
        mj.mj_kinematics(deployed, deployed_data)
        for body in range(1, model.nbody):
            if model.body(body).name in IK_ONLY_BODIES:
                continue
            transform = urdf.get_transform(model.body(body).name)
            np.testing.assert_allclose(transform[:3, 3], data.xpos[body], atol=1e-9)
            np.testing.assert_allclose(transform[:3, :3], data.xmat[body].reshape(3, 3), atol=1e-9)
            deployed_body = deployed.body(model.body(body).name).id
            np.testing.assert_allclose(deployed_data.xpos[deployed_body], data.xpos[body], atol=1e-9)
            np.testing.assert_allclose(deployed_data.xmat[deployed_body], data.xmat[body], atol=1e-9)
