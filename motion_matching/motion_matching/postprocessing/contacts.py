"""Original-bank targets and persistent material contacts for trajectory generation.

The small contact bundle is independent of the retarget work directory. It
contains accepted source phases with their optimized anchors, including hands.
No contact is inferred from a blended trajectory, and no motion is selected here.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from motion_matching.core.database import MotionMatchingDatabase
from retargeting.retarget_contacts import ContactPhase

SURFACE_TOLERANCE = 2e-5  # World geometry originates in float32 MM transforms.


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def export_contact_bundle(bank_dir, work_dir, spec, output_dir):
    """Export accepted phases and optimized anchors from a retargeted bank.

    New solves store anchors in trajectory.npz. Existing released banks may
    still use diagnostic_trajectory.npz; both are read without rewriting them.
    """
    bank, work, output = map(lambda p: Path(p).resolve(), (bank_dir, work_dir, output_dir))
    summary = _read(bank / "bank_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    manifest = dict(
        version=1,
        databases={},
        ranges=0,
        phases=0,
    )
    try:
        for database in summary["reports"]:
            name = Path(database["source"]).name
            payload = dict(bin=name, ranges=[])
            for index, (start, stop) in enumerate(
                zip(database["range_starts"], database["range_stops"])
            ):
                folder = work / Path(name).stem / f"range_{index:02d}"
                phases = _read(folder / "phases.json")
                result = folder / "trajectory.npz"
                if not result.exists():
                    result = folder / "diagnostic_trajectory.npz"
                with np.load(result, allow_pickle=False) as saved:
                    anchors = saved["anchors"].copy()
                for phase, anchor in zip(phases, anchors):
                    phase["body"] = spec.model.body(phase["body"]).name
                    phase["anchor"] = anchor.tolist()
                payload["ranges"].append(dict(start=start, stop=stop, phases=phases))
                manifest["ranges"] += 1
                manifest["phases"] += len(phases)
            target = staging / (Path(name).stem + ".contacts.json")
            _write(target, payload)
            manifest["databases"][name] = dict(file=target.name)
        _write(staging / "manifest.json", manifest)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return manifest


@dataclass
class ContactInputs:
    target_qpos: np.ndarray
    phases: list
    fixed_anchors: tuple


def _contact(value, spec):
    values = dict(value)
    if isinstance(values["body"], str):
        values["body"] = spec.model.body(values["body"]).id
    for key in (
        "local_point",
        "anchor",
        "normal",
        "tangent_axes",
        "tangent_lower",
        "tangent_upper",
        "cap_axis_local",
    ):
        if values.get(key) is not None:
            values[key] = np.asarray(values[key], dtype=float).copy()
    return ContactPhase(**values)


def _material(phase):
    return (
        phase.body,
        tuple(np.round(phase.local_point, 10)),
        round(phase.radius, 10),
        None if phase.cap_axis_local is None else tuple(np.round(phase.cap_axis_local, 10)),
    )


def _same_plane(a, b):
    return (
        np.linalg.norm(a.normal - b.normal) <= SURFACE_TOLERANCE
        and abs(a.plane_height - b.plane_height) <= SURFACE_TOLERANCE
    )


def _contains(phase, point, tolerance=SURFACE_TOLERANCE):
    return abs(phase.normal @ point - phase.plane_height) <= tolerance and (
        phase.tangent_axes is None
        or (
            np.all(phase.tangent_axes @ point >= phase.tangent_lower - tolerance)
            and np.all(phase.tangent_axes @ point <= phase.tangent_upper + tolerance)
        )
    )


def _surfaces(boxes):
    # The robot model's original floor is z=0. Elevated source-ground contacts
    # must bind to one of the actual finite box faces below.
    surfaces = [
        (
            "ground",
            ContactPhase(0, np.zeros(3), 0, 1, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0),
        )
    ]
    for index, box in enumerate(boxes):
        pos, size, quaternion = (
            np.asarray(box[key], dtype=float) for key in ("pos", "size", "quat")
        )
        basis = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
        for axis in range(3):
            tangent = [i for i in range(3) if i != axis]
            axes = basis[:, tangent].T
            for sign in (-1, 1):
                normal = sign * basis[:, axis]
                center = pos + normal * size[axis] / 2
                surfaces.append(
                    (
                        f"box_{index}_axis_{axis}_{sign}",
                        ContactPhase(
                            0,
                            np.zeros(3),
                            0,
                            1,
                            center,
                            normal,
                            float(normal @ center),
                            axes,
                            axes @ center - size[tangent] / 2,
                            axes @ center + size[tangent] / 2,
                        ),
                    )
                )
    return surfaces


def _bind(phase, surfaces):
    candidates = [(name, face) for name, face in surfaces if _same_plane(phase, face)]
    if not candidates:
        raise ValueError(
            f"No real support face exists for {phase.anchor.tolist()} on plane "
            f"{phase.normal.tolist()} / {phase.plane_height}; supply a consistent reference scene"
        )

    def distance(item):
        face = item[1]
        if face.tangent_axes is None:
            return 0.0
        uv = face.tangent_axes @ phase.anchor
        return float(
            np.linalg.norm(
                np.maximum(np.maximum(face.tangent_lower - uv, uv - face.tangent_upper), 0.0)
            )
        )

    # A source anchor is an optimization initial guess, not an accepted output.
    # Associate a new stance with the nearest real co-planar face, retaining
    # both its original anchor value and the face's strict finite bounds. SQP
    # must move the anchor/poses into that domain; nothing is projected here.
    candidates.sort(key=distance)
    # Associate only with a face that overlaps the source patch. A closer
    # disjoint box must not hide another usable support on the same plane.
    for name, face in candidates:
        lower = None if face.tangent_lower is None else face.tangent_lower.copy()
        upper = None if face.tangent_upper is None else face.tangent_upper.copy()
        if phase.tangent_axes is not None and face.tangent_axes is not None:
            basis = phase.tangent_axes @ face.tangent_axes.T
            columns = np.argmax(abs(basis), axis=1)
            if not (
                len(set(columns)) == 2
                and np.allclose(abs(basis), np.eye(2)[columns], atol=SURFACE_TOLERANCE, rtol=0.0)
            ):
                continue
            for row, column in enumerate(columns):
                lo, hi = phase.tangent_lower[row], phase.tangent_upper[row]
                if basis[row, column] < 0:
                    lo, hi = -hi, -lo
                lower[column], upper[column] = (
                    max(lower[column], lo),
                    min(upper[column], hi),
                )
            if not np.all(lower <= upper + SURFACE_TOLERANCE):
                continue
            # Roundoff may produce an inverted zero-width edge intersection.
            upper = np.maximum(upper, lower)
        result = replace(
            phase,
            normal=face.normal.copy(),
            plane_height=face.plane_height,
            tangent_axes=None if face.tangent_axes is None else face.tangent_axes.copy(),
            tangent_lower=lower,
            tangent_upper=upper,
        )
        # A finite source patch on the original floor remains finite.
        if face.tangent_axes is None and phase.tangent_axes is not None:
            result = replace(
                result,
                tangent_axes=phase.tangent_axes.copy(),
                tangent_lower=phase.tangent_lower.copy(),
                tangent_upper=phase.tangent_upper.copy(),
            )
        return name, result

    raise ValueError("No compatible support face overlaps the source contact patch")


class BankContactSource:
    """Build full selected trajectories from a bank plus a compact contact bundle.

    Records contain ``db_key`` (a bank registry key),
    ``source_frame`` (global BIN index), ``transform_quat`` (wxyz), and
    ``transform_pos``. The transform acts on the original BIN world qpos and
    already includes platform elevation. Mirrored frames are already mirrored.
    """

    def __init__(self, bank_dir, bundle_dir, spec):
        self.bank, self.bundle = Path(bank_dir).resolve(), Path(bundle_dir).resolve()
        self.spec = spec
        self.manifest = _read(self.bundle / "manifest.json")
        self.registry = _read(self.bank / "registry.json")
        self._databases = {}

    def _database(self, record):
        key = record["db_key"]
        cfg = self.registry[key]
        name = cfg.get("bin", f"database_g1_{cfg.get('family', key)}.bin")
        if name not in self._databases:
            metadata = self.manifest["databases"][name]
            path = self.bundle / metadata["file"]
            payload = _read(path)
            db = MotionMatchingDatabase()
            db.load_from_file(str(self.bank / name))
            ranges = []
            for item in payload["ranges"]:
                phases = [_contact(value, self.spec) for value in item["phases"]]
                active = [[] for _ in range(item["stop"] - item["start"])]
                for i, phase in enumerate(phases):
                    for frame in range(phase.start, phase.stop):
                        active[frame].append(i)
                ranges.append((phases, active))
            self._databases[name] = (db, ranges)
        return self._databases[name]

    def build(self, records, reference_boxes):
        surfaces = _surfaces(reference_boxes)
        targets, phases, fixed = [], [], set()
        previous = {}
        for frame, record in enumerate(records):
            db, ranges = self._database(record)
            index = record["source_frame"]
            ri = int(np.searchsorted(db.range_starts, index, side="right") - 1)
            local = int(index - db.range_starts[ri])
            rotation_values, translation = (
                np.asarray(record["transform_quat"]),
                np.asarray(record["transform_pos"]),
            )
            rotation = Rotation.from_quat(rotation_values, scalar_first=True)
            raw = self.spec.decode_bones_to_qpos(db.bone_positions[index], db.bone_rotations[index])
            target = np.r_[
                (rotation * Rotation.from_quat(raw[:4], scalar_first=True)).as_quat(
                    scalar_first=True
                ),
                rotation.apply(raw[4:7]) + translation,
                raw[7:],
            ]
            targets.append(target)
            source_phases, active = ranges[ri]
            observations = []
            for phase_index in active[local]:
                phase = source_phases[phase_index]
                normal = rotation.apply(phase.normal)
                axes = None if phase.tangent_axes is None else rotation.apply(phase.tangent_axes)
                moved = replace(
                    phase,
                    start=frame,
                    stop=frame + 1,
                    anchor=rotation.apply(phase.anchor) + translation,
                    normal=normal,
                    plane_height=float(phase.plane_height + normal @ translation),
                    tangent_axes=axes,
                    tangent_lower=None
                    if axes is None
                    else phase.tangent_lower + axes @ translation,
                    tangent_upper=None
                    if axes is None
                    else phase.tangent_upper + axes @ translation,
                )
                observations.append(moved)
            current = {}
            for observation in observations:
                key = _material(observation)
                if key in current:
                    raise ValueError(f"Duplicate active material contact at output frame {frame}")
                old_index = previous.get(key)
                if old_index is not None and _same_plane(phases[old_index], observation):
                    # A stance remains on its existing physical face until the
                    # material point releases. A new clip's anchor cannot slide
                    # it to another box, even if that box is co-planar.
                    old = phases[old_index]
                    old.stop = frame + 1
                    current[key] = old_index
                    continue
                _, bound = _bind(observation, surfaces)
                current[key] = len(phases)
                if frame < 2:
                    fixed.add(len(phases))
                phases.append(bound)
            previous = current
        target_qpos = np.asarray(targets)
        # Check only immutable boundary/contact compatibility here; generation
        # remains the solver's job. Never alter phases to make endpoints pass.
        data = mj.MjData(self.spec.model)
        for frame in (0, 1, len(records) - 2, len(records) - 1):
            data.qpos[:] = self.spec.to_mujoco_qpos(target_qpos[frame])
            mj.mj_kinematics(self.spec.model, data)
            for i, phase in enumerate(phases):
                if not phase.start <= frame < phase.stop:
                    continue
                point = (
                    data.xpos[phase.body] + data.xmat[phase.body].reshape(3, 3) @ phase.local_point
                )
                anchor = point - phase.radius * phase.normal
                if not _contains(phase, anchor, tolerance=0.001):
                    raise ValueError(
                        f"Fixed endpoint frame {frame} contact {i} is outside its support face"
                    )
                if i in fixed:
                    if not (np.linalg.norm(anchor - phase.anchor) <= 0.001):
                        raise ValueError(
                            f"Fixed endpoint frame {frame} conflicts with inherited contact anchor {i}"
                        )
        return ContactInputs(target_qpos, phases, tuple(sorted(fixed)))
