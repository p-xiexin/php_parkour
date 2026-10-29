"""Native MuJoCo robot profiles; the existing G1 pipeline remains the default."""

from functools import lru_cache
from pathlib import Path

from motion_matching.robots.base import RobotSpec


@lru_cache
def get_robot_spec(robot: str = "t2", model_path: str | Path | None = None) -> RobotSpec:
    if robot == "t2":
        from motion_matching.robots.t2 import make_t2_spec
        return make_t2_spec(model_path)
    raise ValueError(f"Unknown robot profile: {robot!r}")


__all__ = ["RobotSpec", "get_robot_spec"]
