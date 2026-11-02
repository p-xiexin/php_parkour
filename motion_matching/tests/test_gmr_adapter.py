"""Exercise real upstream GMR, its input conventions and native T2 playback."""
import os
from pathlib import Path

import numpy as np
import pytest

from motion_matching.motion_visualizer import load_motion_data
from motion_matching.robots.t2 import make_t2_spec


def test_sphere_bank_keeps_every_source_frame_and_range(tmp_path):
    root = os.environ.get('GMR_ROOT')
    bank = os.environ.get('G1_BANK_DIR')
    if not root or not bank:
        pytest.skip('Set GMR_ROOT and G1_BANK_DIR for full BIN conversion integration')
    import shutil
    from motion_matching.core.database import MotionMatchingDatabase
    from motion_matching.retarget_g1_to_t2 import convert_bank
    from motion_matching.retarget_g1_to_t2 import locked_wrist_names
    from motion_matching.utils import PKG_ROOT
    source = tmp_path / 'source'
    source.mkdir()
    filename = 'database_g1_jump_35.bin'
    shutil.copyfile(Path(bank) / filename, source / filename)
    model = PKG_ROOT / 'assets/robots/booster_t2/mjcf/t2_31dof_sphere_locked.xml'
    target = tmp_path / 'target'
    convert_bank(source, target, model, Path(root))
    before, after = MotionMatchingDatabase(), MotionMatchingDatabase()
    before.load_from_file(str(source / filename))
    after.load_from_file(str(target / filename))
    np.testing.assert_array_equal(before.range_starts, after.range_starts)
    np.testing.assert_array_equal(before.range_stops, after.range_stops)
    assert len(after.bone_positions) == len(before.bone_positions) == 68
    spec = make_t2_spec(model)
    assert len(locked_wrist_names(spec.model)) == 6
    for side in ('left', 'right'):
        assert spec.model.geom(side + '_g1_sphere_visual').id >= 0
        assert spec.model.geom(side + '_g1_sphere_collision').id >= 0
    for index in range(len(after.range_starts)):
        joints, _, php, _, _ = load_motion_data(target / filename, robot_spec=spec, range_index=index)
        for i, name in enumerate(spec.joint_names):
            if '_wrist_' in name:
                np.testing.assert_allclose(joints[:, i], 0., atol=1e-6)
        assert np.isfinite(php).all()


def test_empty_bank_is_rejected_without_creating_output(tmp_path):
    from motion_matching.retarget_g1_to_t2 import convert_bank
    source, target = tmp_path / 'empty', tmp_path / 'output'
    source.mkdir()
    with pytest.raises(ValueError, match='No G1 databases'):
        convert_bank(source, target, None, None)
    assert not target.exists()


def test_unlocked_fist_model_is_rejected(tmp_path):
    from motion_matching.retarget_g1_to_t2 import convert_bank
    from motion_matching.utils import PKG_ROOT
    source, target = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    (source / 'database_g1_example.bin').touch()
    model = PKG_ROOT / 'assets/robots/booster_t2/mjcf/t2_31dof_fc.xml'
    with pytest.raises(ValueError, match='all six wrists locked'):
        convert_bank(source, target, model, None)
    assert not target.exists()
