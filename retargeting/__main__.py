"""Retarget official 60 Hz G1 BINs to T2 using GMR initialization and SQP."""

import argparse
from pathlib import Path

from motion_matching.scenarios.skill_registry import SKILL_REGISTRY
from motion_matching.utils import PKG_ROOT

from retargeting.retarget_database import (
    DEFAULT_ITERATIONS,
    DEFAULT_WINDOW_FRAMES,
    convert_trajectory_bank,
    retarget_range,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PKG_ROOT / "assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml",
    )
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument(
        "--skill",
        choices=list(SKILL_REGISTRY),
        help="Solve one range for inspection instead of the full bank",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.skill:
        retarget_range(
            args.source_dir,
            args.output_dir,
            args.model,
            args.gmr_root,
            args.skill,
            args.start,
            args.stop,
            args.iterations,
            args.window_frames,
        )
    else:
        convert_trajectory_bank(
            args.source_dir,
            args.output_dir,
            args.model,
            args.gmr_root,
            args.iterations,
            args.workers,
            args.window_frames,
        )


if __name__ == "__main__":
    main()
