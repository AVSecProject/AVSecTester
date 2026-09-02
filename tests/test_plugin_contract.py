"""The SecurityPlugin contract: seams, check(), current_seam(), describe(), defense telemetry."""

from __future__ import annotations

import pytest
from avsectester.core.binding import IncompatiblePlugin
from avsectester.core.interfaces import AttackBase, DefenseBase, DefenseOutcome
from avsectester.core.threat_model import Knowledge, ThreatModel


class _Atk(AttackBase):
    category = "perception"
    seams = ("perception_out", "tracking_out")     # a multi-seam attack
    threat_model = ThreatModel(
        goal="g", knowledge=Knowledge.GRAYBOX, target="t", success_criteria="s"
    )

    def apply(self, data, ego_state=None, ctx=None, **kw):
        return (self.current_seam(ctx), data)       # echoes which seam fired


class _Def(DefenseBase):
    category = "input_sanitize"
    seams = ("perception_out",)

    def apply(self, data, ego_state=None, ctx=None, **kw):
        self.record_outcome(ctx, DefenseOutcome(seam="perception_out", dropped=[7], kept=len(data)))
        return data


def test_check_passes_when_stack_exposes_all_seams():
    _Atk().check(frozenset({"perception_out", "tracking_out"}))  # must not raise


def test_check_fails_on_unexposed_seam():
    with pytest.raises(IncompatiblePlugin, match="tracking_out"):
        _Atk().check(frozenset({"perception_out"}))  # no tracking_out


def test_current_seam_uses_ctx_then_falls_back():
    from types import SimpleNamespace

    a = _Atk()
    assert a.apply("x", ctx=SimpleNamespace(seam="tracking_out"))[0] == "tracking_out"
    assert a.apply("x", ctx=None)[0] == "perception_out"     # falls back to first declared seam


def test_describe_exposes_inventory_metadata():
    d = _Atk().describe()
    assert d["kind"] == "attack" and d["category"] == "perception"
    assert d["seams"] == ["perception_out", "tracking_out"]
    assert d["threat_model"]["knowledge"] == "graybox"


def test_defense_records_outcome_into_ctx():
    from types import SimpleNamespace

    ctx = SimpleNamespace(defense_outcomes=[])
    out = _Def().apply([1, 2, 3], ctx=ctx)
    assert out == [1, 2, 3]
    assert len(ctx.defense_outcomes) == 1
    assert ctx.defense_outcomes[0].dropped == [7]


def test_defense_outcome_noop_without_ctx():
    _Def().apply([1], ctx=None)  # must not raise


def test_lifecycle_defaults_are_noops():
    a = _Atk()
    a.reset()
    a.validate(None)  # type: ignore[arg-type]
