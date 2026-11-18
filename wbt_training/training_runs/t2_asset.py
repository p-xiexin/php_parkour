"""Export T2 URDF, MJCF and camera meshes for the selected wrist mode.

Geometry, inertial frames and joint limits come from the versioned MJCF.
The velocity cap is shared with the T2 training configuration.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wbt_training.config_values.t2_config import (
    T2_ASSET_DIR,
    T2_LOCK_WRISTS,
    T2_VELOCITY_LIMIT,
    T2_WRIST_JOINT_NAMES,
    t2_robot,
)
from wbt_training.config_values.t2_config import (
    T2_IK_ONLY_BODIES as IK_ONLY_BODIES,
)
from wbt_training.config_values.t2_config import (
    T2_MODEL_PATH as DEFAULT_MODEL,
)


def _numbers(value):
    return " ".join(f"{v:.12g}" for v in value)


def _origin(parent, source):
    if any(key in source.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Only explicit quaternion rotations are supported")
    quat = np.fromstring(source.get("quat", "1 0 0 0"), sep=" ")
    ET.SubElement(
        parent,
        "origin",
        xyz=source.get("pos", "0 0 0"),
        rpy=_numbers(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")),
    )


def export_urdf(model_path: Path, output_path: Path, *, lock_wrists: bool = False):
    source = ET.parse(model_path).getroot()
    mesh_dir = model_path.parent / source.find("compiler").get("meshdir", ".")
    meshes = {m.get("name"): m for m in source.findall("asset/mesh")}
    defaults = {
        d.get("class"): d.find("geom").attrib
        for d in source.findall(".//default")
        if d.find("geom") is not None
    }
    locked = set(T2_WRIST_JOINT_NAMES) if lock_wrists else set()
    robot = ET.Element("robot", name="t2_sphere_locked" if lock_wrists else "t2_sphere_unlocked")

    def body(element, parent=None):
        name = element.get("name")
        if name in IK_ONLY_BODIES:
            if len(element):
                raise ValueError(f"IK helper {name} unexpectedly has physical contents")
            return
        link = ET.SubElement(robot, "link", name=name)
        inertia = element.find("inertial")
        if inertia is not None:
            target = ET.SubElement(link, "inertial")
            _origin(target, inertia)
            ET.SubElement(target, "mass", value=inertia.get("mass"))
            diagonal = np.fromstring(inertia.get("diaginertia"), sep=" ")
            ET.SubElement(
                target,
                "inertia",
                ixx=str(diagonal[0]),
                iyy=str(diagonal[1]),
                izz=str(diagonal[2]),
                ixy="0",
                ixz="0",
                iyz="0",
            )
        for geom in element.findall("geom"):
            values = {**defaults.get(geom.get("class"), {}), **geom.attrib}
            collision = int(values.get("contype", "1")) or int(values.get("conaffinity", "1"))
            # Omit debug marker spheres, retain actual robot visuals and contacts.
            if not collision and values.get("type") != "mesh":
                continue
            target = ET.SubElement(link, "collision" if collision else "visual")
            _origin(target, geom)
            geometry = ET.SubElement(target, "geometry")
            kind = values.get("type", "sphere")
            size = np.fromstring(values.get("size", ""), sep=" ")
            if kind == "mesh":
                mesh = meshes[values["mesh"]]
                filename = os.path.relpath(
                    (mesh_dir / mesh.get("file")).resolve(), output_path.parent
                )
                ET.SubElement(geometry, "mesh", filename=filename, scale=mesh.get("scale", "1 1 1"))
            elif kind == "sphere":
                ET.SubElement(geometry, "sphere", radius=str(size[0]))
            elif kind == "box":
                ET.SubElement(geometry, "box", size=_numbers(2 * size))
            elif kind == "cylinder":
                ET.SubElement(geometry, "cylinder", radius=str(size[0]), length=str(2 * size[1]))
            else:
                raise ValueError(f"Unsupported T2 geometry: {kind}")
        if parent is not None:
            joints = element.findall("joint")
            if len(joints) > 1:
                raise ValueError(f"Multiple joints on {name}")
            joint = joints[0] if joints else None
            movable = joint is not None and joint.get("name") not in locked
            target = ET.SubElement(
                robot,
                "joint",
                name=joint.get("name") if joint is not None else name + "_fixed",
                type="revolute" if movable else "fixed",
            )
            _origin(target, element)
            ET.SubElement(target, "parent", link=parent)
            ET.SubElement(target, "child", link=name)
            if movable:
                if joint.get("pos", "0 0 0") != "0 0 0":
                    raise ValueError(f"Nonzero joint offset on {name}")
                ET.SubElement(target, "axis", xyz=joint.get("axis", "0 0 1"))
                lower, upper = joint.get("range").split()
                effort = max(abs(float(v)) for v in joint.get("actuatorfrcrange").split())
                ET.SubElement(
                    target,
                    "limit",
                    lower=lower,
                    upper=upper,
                    effort=str(effort),
                    velocity=str(T2_VELOCITY_LIMIT),
                )
        for child in element.findall("body"):
            body(child, name)

    body(source.find("worldbody/body"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def export_mjcf(model_path: Path, output_path: Path, *, lock_wrists: bool = False):
    """Keep or weld wrist joints and drives, matching the exported URDF."""
    source = ET.parse(model_path).getroot()
    compiler = source.find("compiler")
    mesh_dir = (model_path.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", os.path.relpath(mesh_dir, output_path.parent))
    locked = set(T2_WRIST_JOINT_NAMES) if lock_wrists else set()
    for parent in source.iter():
        for child in list(parent):
            if (
                (child.tag == "joint" and child.get("name") in locked)
                or child.get("joint") in locked
                or child.get("joint1") in T2_WRIST_JOINT_NAMES
                or (child.tag == "body" and child.get("name") in IK_ONLY_BODIES)
            ):
                parent.remove(child)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(source, space="  ")
    ET.ElementTree(source).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def export_depth_meshes(model_path: Path, output_dir: Path):
    """Bake each visual mesh into its link frame for Warp self-occlusion."""
    import trimesh

    source = ET.parse(model_path).getroot()
    mesh_dir = model_path.parent / source.find("compiler").get("meshdir", ".")
    assets = {m.get("name"): m for m in source.findall("asset/mesh")}
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for body in source.findall("worldbody//body"):
        parts = []
        for geom in body.findall("geom"):
            if geom.get("class") != "visual" or geom.get("mesh") is None:
                continue
            asset = assets[geom.get("mesh")]
            mesh = trimesh.load(mesh_dir / asset.get("file"), force="mesh", process=False)
            mesh.apply_scale(np.fromstring(asset.get("scale", "1 1 1"), sep=" "))
            quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            transform[:3, 3] = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
            mesh.apply_transform(transform)
            parts.append(mesh)
        if parts:
            name = body.get("name")
            target = output_dir / (name + ".obj")
            trimesh.util.concatenate(parts).export(target)
            result[name] = target.name
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=T2_ASSET_DIR / t2_robot.asset.urdf_file)
    args = parser.parse_args()
    print(export_urdf(DEFAULT_MODEL, args.output, lock_wrists=T2_LOCK_WRISTS))
    print(export_mjcf(DEFAULT_MODEL, args.output.with_suffix(".xml"), lock_wrists=T2_LOCK_WRISTS))
    meshes = export_depth_meshes(DEFAULT_MODEL, args.output.parent / "depth_meshes")
    print(f"Exported {len(meshes)} depth meshes")
