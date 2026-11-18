"""Robot-specific parameters for the existing motion-matching algorithm."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionMatchingConfig:
    inertialize_blending_halflife: float
    search_time: float
    simulation_velocity_halflife: float
    simulation_rotation_halflife: float
    output_fps: int
    transition_cost: float
    ignore_range_end: int
    ignore_surrounding: int


def load_motion_matching_config(
    robot_id: str = "g1", path: str | Path | None = None
) -> MotionMatchingConfig:
    """Load the robot's parameters, then apply any explicit JSON overrides."""
    values = json.loads(Path(__file__).with_name(f"{robot_id}.json").read_text())
    if path is not None:
        values.update(json.loads(Path(path).read_text()))
    return MotionMatchingConfig(**values)
