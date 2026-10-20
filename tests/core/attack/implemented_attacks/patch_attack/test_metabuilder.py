"""Host ranking is local to each window and never changes selection metadata later."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.implemented_attacks.patch_attack import PatchAttack
from avsectester.attack.insertion.effect import InsertionAttackEffect
from avsectester.datasets import FrameReference
from avsectester.geometry.poses import ActorPose


def scenario_fixture(monkeypatch, scores, *, candidates=None, decision_steps=2, **options):
    frames = tuple(FrameReference("s", index, index, index * .1) for index in range(len(scores) + 1))
    rendered, reasons = [], []
    poses = {host: ActorPose(np.eye(4)) for per_frame in scores for host in per_frame}

    def render(request):
        assert request.effects == ()
        rendered.append(request.frame.position)
        return SimpleNamespace(view=SimpleNamespace(frame=request.frame.position, actors=poses,
                                                     ego=ActorPose(np.eye(4))))

    def evidence(effect, view):
        value = scores[view.frame].get(effect.insertion.placement.host)
        return SimpleNamespace(visibility=None if value is None else SimpleNamespace(fraction=value),
                               reason="outside the camera")

    monkeypatch.setattr(InsertionAttackEffect, "evidence", evidence)
    monkeypatch.setattr(InsertionAttackEffect, "apply", lambda *args: pytest.fail("selection is clean"))
    scenario = SimpleNamespace(scenario_id="s", frames=frames, seed=42, record_reason=reasons.append,
        services=SimpleNamespace(simulator=object(), session=lambda seed: nullcontext(
            SimpleNamespace(render=render))))
    attack = PatchAttack(decision_steps=decision_steps, candidate_hosts=candidates, **options)
    return attack.build_metadata(scenario), rendered, reasons


def test_top_hosts_are_ranked_independently_within_each_window(monkeypatch):
    scores = [dict(a=.9, b=.8, c=.7, d=.6, e=.5)] * 2
    scores += [dict(a=.2, b=.3, c=.9, d=.8, e=.7)] * 2
    selected, rendered, _ = scenario_fixture(monkeypatch, scores, candidates=["a", "b", "c", "d", "e"],
                                            hosts_per_window=3)
    assert rendered == [0, 1, 2, 3]
    assert [(segment.start.position, segment.stop.position, segment.roles["host"][0],
             segment.selection_rank) for segment in selected] == [
        (0, 2, "a", 0), (0, 2, "b", 1), (0, 2, "c", 2),
        (2, 4, "c", 0), (2, 4, "d", 1), (2, 4, "e", 2),
    ]
    assert [segment.selection_score for segment in selected] == [.9, .8, .7, .9, .8, .7]
    assert all(segment.placement.host == segment.roles["host"][0] for segment in selected)


def test_host_must_be_eligible_for_every_decision_and_ties_keep_candidate_order(monkeypatch):
    selected, _, reasons = scenario_fixture(monkeypatch,
        [dict(a=.9, b=.5, c=.5), dict(a=None, b=.5, c=.5)],
        candidates=["c", "b", "a"], hosts_per_window=3)
    assert [segment.roles["host"][0] for segment in selected] == ["c", "b"]
    assert any("not visible" in reason for reason in reasons)


def test_max_windows_limits_windows_without_dropping_top_hosts(monkeypatch):
    selected, _, _ = scenario_fixture(monkeypatch, [dict(a=.8, b=.7)] * 4,
                                      hosts_per_window=2, max_windows=1)
    assert len(selected) == 2
    assert all(segment.start.position == 0 for segment in selected)


def test_short_valid_prefix_does_not_hide_a_later_full_host_window(monkeypatch):
    selected, _, _ = scenario_fixture(monkeypatch,
        [dict(a=.8), dict(b=.9), dict(b=.8), dict(b=.7)], candidates=["a", "b"], decision_steps=2)
    assert [(segment.start.position, segment.stop.position) for segment in selected] == [(1, 3)]
