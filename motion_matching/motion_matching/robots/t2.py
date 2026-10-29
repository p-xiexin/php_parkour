"""Booster T2: native 31-joint fist-hand MJCF and physical pelvis mapping."""

import os
from pathlib import Path

from motion_matching.robots.base import RobotSpec
from motion_matching.utils import T2_MJCF


def make_t2_spec(model_path: str | Path | None = None) -> RobotSpec:
    model_path = model_path or os.environ.get("PHP_T2_MJCF") or T2_MJCF
    return RobotSpec("t2", model_path, pelvis_body="waist_yaw_link",
                     exclude_tracking_body_names=("left_hand_contact_link", "right_hand_contact_link"))
