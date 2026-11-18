"""Defaults stay independent; bank metadata owns source-frame intervals."""

import json
from dataclasses import asdict

from motion_matching.configs import load_motion_matching_config
from motion_matching.scenarios.skill_registry import SKILL_REGISTRY

from motion_matching import run
from retargeting.retarget_database import _copy_bank_assets


def test_defaults_and_partial_override_are_isolated(tmp_path):
    g1 = load_motion_matching_config()
    t2 = load_motion_matching_config("t2")
    assert (g1.inertialize_blending_halflife, g1.output_fps) == (0.1, 50)
    assert (t2.inertialize_blending_halflife, t2.output_fps) == (0.05, 60)
    common = {
        key: value
        for key, value in asdict(g1).items()
        if key not in ("inertialize_blending_halflife", "output_fps")
    }
    assert common == {key: asdict(t2)[key] for key in common}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps({"search_time": 0.2}))
    candidate = load_motion_matching_config("t2", path)
    assert candidate.search_time == 0.2
    assert candidate.inertialize_blending_halflife == 0.05
    assert candidate.output_fps == 60
    assert load_motion_matching_config("t2") == t2
    assert load_motion_matching_config("g1") == g1


def test_exported_bank_owns_search_intervals(tmp_path, monkeypatch):
    original = SKILL_REGISTRY["climb_76_down"]["search_ranges"].copy()
    _copy_bank_assets(tmp_path, tmp_path, [])
    registry = json.loads((tmp_path / "registry.json").read_text())
    expected = [[515, 521, 645], [1189, 1195, 1319]]
    assert registry["climb_76_down"]["search_ranges"] == expected
    assert SKILL_REGISTRY["climb_76_down"]["search_ranges"] == original
    # The loader consumes the bank's intervals, including a caller's metadata
    # change, without a second override layer in the robot parameters.
    registry["climb_76_down"]["search_ranges"] = [[520, 521, 645]]
    monkeypatch.setattr(run, "_load_db", lambda **kwargs: kwargs)
    _, traversal, _ = run.init_databases({"climb_76_down"}, registry=registry)
    assert traversal["climb_76_down"]["search_ranges"] == [[520, 521, 645]]
