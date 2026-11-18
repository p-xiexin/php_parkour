"""CPU regressions for observations after a source-clip teleport."""

from types import SimpleNamespace

import pytest
import torch
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.observation.terms.depth import WarpDepthImageObsTerm
from holosoma.managers.observation.terms.wbt import base_ang_vel


@pytest.mark.parametrize("latency", [3, 4])
def test_clip_reset_refreshes_only_affected_depth_histories(latency):
    term = object.__new__(WarpDepthImageObsTerm)
    term.num_envs = 4
    term.device = "cpu"
    term.buffer_len = 6
    old = torch.tensor([10., 20., 30., 40.]).view(4, 1, 1)
    fresh = old * 10
    term.depth_buffer = old[:, None].repeat(1, term.buffer_len, 1, 1)
    term._cached_output = old.clone()
    term.reset_episodes = torch.zeros(4, dtype=torch.bool)
    term.latency_frame_range = None
    term.latency_frame = latency
    term.command = SimpleNamespace(
        # Clip reset, ordinary step, clip reset, episode reset.
        motion_end_reset=torch.tensor([True, False, True, False]),
    )
    term._get_depth_images = lambda env: fresh
    term._process_depth_images = lambda images, env_ids: images
    term.reset(torch.tensor([3]))
    env = SimpleNamespace(device="cpu")

    # A final-observation read must neither consume reset flags nor advance time.
    torch.testing.assert_close(term(env, modify_history=False), old)
    observed = term(env)
    torch.testing.assert_close(observed[:, 0, 0], torch.tensor([100., 20., 300., 400.]))
    reset_ids = torch.tensor([0, 2, 3])
    torch.testing.assert_close(
        term.depth_buffer[reset_ids], fresh[reset_ids, None].expand(-1, 6, -1, -1)
    )
    term.command.motion_end_reset.zero_()
    for step in range(1, latency + 1):
        observed = term(env)
        assert observed[0, 0, 0] == 100  # Never sees the previous scene again.
        assert observed[1, 0, 0] == (20 if step < latency else 200)


def test_clip_reset_updates_base_frame_before_angular_velocity_observation():
    env = object.__new__(WholeBodyTrackingManager)
    old_quat = torch.tensor([[0., 0., 0., 1.]])
    env.base_quat = old_quat.clone()
    env.simulator = SimpleNamespace(
        base_quat=old_quat.clone(), robot_root_states=torch.zeros(1, 13)
    )
    half_sqrt = 2 ** -0.5
    new_quat = torch.tensor([[0., 0., half_sqrt, half_sqrt]])
    calls = []

    def teleport():
        calls.append("command")
        # Simulator refresh replaces the tensor rather than updating the old cache.
        env.simulator.base_quat = new_quat.clone()
        env.simulator.robot_root_states[:, 3:7] = new_quat
        env.simulator.robot_root_states[:, 10:13] = torch.tensor([[1., 0., 0.]])

    env.command_manager = SimpleNamespace(step=teleport)
    env.curriculum_manager = SimpleNamespace(step=lambda: calls.append("curriculum"))
    env.randomization_manager = SimpleNamespace(step=lambda: calls.append("randomization"))
    env._update_tasks_callback()

    assert calls == ["command", "curriculum", "randomization"]
    torch.testing.assert_close(env.base_quat, new_quat)
    # A +90 degree yaw maps world +X angular velocity to local -Y.
    torch.testing.assert_close(base_ang_vel(env), torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)
