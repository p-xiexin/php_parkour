"""Zero-time transfer regressions from two recorded T2 matching transitions.

Only Simulation and trunk bones are needed: their old offsets expose both
root double-counting and noncommuting angular-velocity frame changes.
Fixtures are captured data; no motion database or external file is required.
"""

import numpy as np
import pytest
from motion_matching.utils.animation_state import (
    InertializerState,
    inertialize_pose_transition,
    inertialize_pose_update,
)
from scipy.spatial.transform import Rotation

_CASES = [
    {
        "frame": 164,
        "before": {
            "bone_offset_positions": [[5.50111178831969e-10, -2.642422392007404e-10, 0.0], [2.2035735458558747e-09, 5.891561127846501e-10, 1.4391807745539609e-09]],
            "bone_offset_velocities": [[-1.446188147014027e-08, 6.9471628449946365e-09, 0.0], [-5.8055935170386874e-08, -1.5535638553387798e-08, -3.7997317292592925e-08]],
            "bone_offset_rotations": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            "bone_offset_angular_velocities": [[0.0, 0.0, 2.7016085867378704e-19], [-1.8377579189335358e-14, 1.5550984165755427e-13, 1.7188522808309244e-14]],
        },
        "arguments": {
            "root_position": [2.603217840194702, -0.14692731201648712, 0.0],
            "root_velocity": [0.9682826399803162, -0.7691850662231445, 0.0],
            "root_rotation": [0.8622771501541138, 0.0, 0.0, -0.5064367651939392],
            "root_angular_velocity": [0.0, 0.0, -2.5184381008148193],
            "bone_src_positions": [[0.3084072768688202, -1.3976984024047852, 0.0], [-0.10630253702402115, 0.004252658225595951, 0.8375035524368286]],
            "bone_src_velocities": [[-0.5887869000434875, -1.0874497890472412, 0.0], [0.20624783635139465, -0.623227059841156, -0.35475635528564453]],
            "bone_src_rotations": [[-0.3352590799331665, -0.0, 0.0, 0.9421259760856628], [0.9661375284194946, 0.09934352338314056, 0.2381078004837036, 0.0037095812149345875]],
            "bone_src_angular_velocities": [[0.0, 0.0, -2.5184381008148193], [0.29537856578826904, 1.2542556524276733, -1.1252577304840088]],
            "bone_dst_positions": [[-5.9563140869140625, -2.598078966140747, 0.0], [-0.09699004143476486, 0.013515084981918335, 0.8297553658485413]],
            "bone_dst_velocities": [[-1.199641227722168, 0.542142391204834, 0.0], [0.2448965311050415, -0.3771931529045105, -0.4000353515148163]],
            "bone_dst_rotations": [[0.41168123483657837, 0.0, -0.0, 0.9113278985023499], [0.9735122323036194, 0.14783766865730286, 0.16910889744758606, -0.042663417756557465]],
            "bone_dst_angular_velocities": [[0.0, 0.0, -2.0003750324249268], [0.46389639377593994, 1.902111530303955, -0.5688729286193848]],
        },
        "legacy_after": {
            "bone_offset_positions": [[0.0, 0.0, 0.0], [-0.009312495589256287, -0.00926242582499981, 0.0077481865882873535]],
            "bone_offset_velocities": [[-0.08925218135118484, 0.014828688465058804, 0.0], [-0.038648754358291626, -0.2460339069366455, 0.045278966426849365]],
            "bone_offset_rotations": [[1.0, 0.0, 0.0, 0.0], [0.9953413009643555, -0.03533356636762619, 0.06363164633512497, 0.06323148310184479]],
            "bone_offset_angular_velocities": [[0.0, 0.0, -0.5180630683898926], [-0.1685178279876709, -0.6478558778762817, -0.556384801864624]],
        },
    },
    {
        "frame": 242,
        "before": {
            "bone_offset_positions": [[0.0010853188578039408, 0.0010220990516245365, 0.0], [-0.0011549534974619746, -0.00021418013784568757, 0.005305193830281496]],
            "bone_offset_velocities": [[-0.020823834463953972, -0.019159065559506416, 0.0], [0.026044685393571854, 0.004953238647431135, -0.10570754110813141]],
            "bone_offset_rotations": [[0.9999990463256836, 0.0, 0.0, 0.0013595829950645566], [0.999954104423523, -0.0047880117781460285, -0.0012032599188387394, -0.008211018517613411]],
            "bone_offset_angular_velocities": [[0.0, 0.0, -0.05206908658146858], [0.20725926756858826, 0.04511633887887001, 0.34280461072921753]],
        },
        "arguments": {
            "root_position": [2.4290361404418945, -2.1532654762268066, 0.0],
            "root_velocity": [-0.8156908750534058, -1.5559173822402954, 0.0],
            "root_rotation": [0.5981087684631348, 0.0, 0.0, -0.8014149069786072],
            "root_angular_velocity": [0.0, 0.0, -0.47175514698028564],
            "bone_src_positions": [[-2.9868717193603516, 0.09845716506242752, 0.0], [-0.062189504504203796, -0.012432031333446503, 0.8917734026908875]],
            "bone_src_velocities": [[1.503610610961914, 0.8559176921844482, 0.0], [-0.044059231877326965, 0.09538529813289642, 0.2287137359380722]],
            "bone_src_rotations": [[-0.9387539625167847, -0.0, -0.0, -0.34458819031715393], [0.997687041759491, 0.028272729367017746, 0.03708665445446968, 0.04945538192987442]],
            "bone_src_angular_velocities": [[0.0, 0.0, -0.41968604922294617], [-0.13545730710029602, 0.3534197509288788, -1.7904472351074219]],
            "bone_dst_positions": [[-4.287057876586914, 2.7438478469848633, 0.0], [-0.10608454793691635, -0.02359670214354992, 0.8924847841262817]],
            "bone_dst_velocities": [[-0.4924964904785156, -1.4770746231079102, 0.0], [0.021488256752490997, 0.19804464280605316, 0.21993491053581238]],
            "bone_dst_rotations": [[-0.7048282027244568, -0.0, 0.0, 0.7093780636787415], [0.9762617349624634, 0.02253369428217411, 0.2154182344675064, -0.00041434774175286293]],
            "bone_dst_angular_velocities": [[0.0, 0.0, -1.0221439599990845], [-0.5670649409294128, -0.13886263966560364, -1.0017828941345215]],
        },
        "legacy_after": {
            "bone_offset_positions": [[0.0010852813720703125, 0.0010221004486083984, 0.0], [0.04274009168148041, 0.010950490832328796, 0.004593789577484131]],
            "bone_offset_velocities": [[0.04767163470387459, -0.2934694290161133, 0.0], [-0.03950280323624611, -0.09770610928535461, -0.09692871570587158]],
            "bone_offset_rotations": [[0.9999990463256836, 0.0, 0.0, 0.0013595521450042725], [0.9827803373336792, 0.009554557502269745, -0.1809355765581131, 0.036249883472919464]],
            "bone_offset_angular_velocities": [[0.0, 0.0, 0.49831974506378174], [0.6388669013977051, 0.5373986959457397, -0.4458596706390381]],
        },
    },
]

def _load(case):
    state = InertializerState.create(2)
    for key, value in case["before"].items():
        getattr(state, key)[:] = np.asarray(value, dtype=np.float32)
    arguments = {key: np.asarray(value, dtype=np.float32)
                 for key, value in case["arguments"].items()}
    return state, arguments


@pytest.mark.parametrize("case", _CASES, ids=lambda case: f"frame_{case['frame']}")
def test_correct_transfer_preserves_rendered_pose_and_velocity(case):
    state, args = _load(case)
    offset_rotation = Rotation.from_quat(state.bone_offset_rotations, scalar_first=True)
    expected_position = args["bone_src_positions"] + state.bone_offset_positions
    expected_rotation = offset_rotation * Rotation.from_quat(args["bone_src_rotations"], scalar_first=True)
    expected_velocity = args["bone_src_velocities"] + state.bone_offset_velocities
    expected_angular = state.bone_offset_angular_velocities + offset_rotation.apply(
        args["bone_src_angular_velocities"])
    expected_position[0] = args["root_position"]
    expected_velocity[0] = args["root_velocity"]
    expected_angular[0] = args["root_angular_velocity"]
    expected_quaternions = expected_rotation.as_quat(scalar_first=True)
    expected_quaternions[0] = args["root_rotation"]

    inertialize_pose_transition(state=state, **args, correct_state_transfer=True)
    np.testing.assert_array_equal(state.bone_offset_positions[0], 0.)
    np.testing.assert_array_equal(state.bone_offset_rotations[0], [1., 0., 0., 0.])
    output = {key: np.zeros_like(args["bone_dst_" + key])
              for key in ("positions", "velocities", "rotations", "angular_velocities")}
    inertialize_pose_update(
        state=state, **{"bone_" + key: value for key, value in output.items()},
        **{"bone_input_" + key: args["bone_dst_" + key] for key in output},
        halflife=.05, dt=0.,
    )
    np.testing.assert_allclose(output["positions"], expected_position, atol=3e-7, rtol=0)
    rotation_error = (Rotation.from_quat(output["rotations"], scalar_first=True) *
                      Rotation.from_quat(expected_quaternions, scalar_first=True).inv()).magnitude()
    assert rotation_error.max() < 5e-7
    np.testing.assert_allclose(output["velocities"], expected_velocity, atol=1e-6, rtol=0)
    np.testing.assert_allclose(output["angular_velocities"], expected_angular, atol=1e-6, rtol=0)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: f"frame_{case['frame']}")
def test_default_transfer_matches_recorded_legacy_bits(case):
    state, args = _load(case)
    inertialize_pose_transition(state=state, **args)
    for key, expected in case["legacy_after"].items():
        np.testing.assert_array_equal(getattr(state, key), np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(state.transition_dst_position, args["root_position"])
    np.testing.assert_array_equal(state.transition_dst_rotation, args["root_rotation"])
    np.testing.assert_array_equal(state.transition_src_position, args["bone_dst_positions"][0])
    np.testing.assert_array_equal(state.transition_src_rotation, args["bone_dst_rotations"][0])
