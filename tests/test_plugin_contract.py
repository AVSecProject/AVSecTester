"""The SecurityPlugin contract: binding resolution, describe(), defense telemetry."""

from __future__ import annotations

from avsectester.core.binding import BindingSpec
from avsectester.core.capability import Capability as Cap
from avsectester.core.capability import StackProfile
from avsectester.core.interfaces import AttackBase, DefenseBase, DefenseOutcome
from avsectester.core.threat_model import Knowledge, ThreatModel


class _Atk(AttackBase):
    category = "perception"
    bindings = (
        BindingSpec("perception_out", "detections", requires={Cap.NEURAL_PERCEPTION}, fidelity=2),
        BindingSpec("perception_input", "objects", requires={Cap.GT_PERCEPTION}, fidelity=1),
    )
    threat_model = ThreatModel(
        goal="g", knowledge=Knowledge.GRAYBOX, target="t", success_criteria="s"
    )

    def apply(self, data, **kw):
        return data


class _Def(DefenseBase):
    category = "input_sanitize"
    bindings = (BindingSpec("perception_out", "detections"),)

    def apply(self, data, ego_state=None, ctx=None, **kw):
        self.record_outcome(ctx, DefenseOutcome(seam="perception_out", dropped=[7], kept=len(data)))
        return data


def test_attack_resolves_binding_per_stack():
    a = _Atk()
    gt = StackProfile.of(["perception_input"], [Cap.GT_PERCEPTION])
    assert a.resolve_binding(gt).seam == "perception_input"
    assert a.bound_seam == "perception_input"
    neural = StackProfile.of(["perception_out"], [Cap.NEURAL_PERCEPTION])
    assert a.resolve_binding(neural).seam == "perception_out"


def test_describe_exposes_inventory_metadata():
    d = _Atk().describe()
    assert d["kind"] == "attack" and d["category"] == "perception"
    assert {b["seam"] for b in d["bindings"]} == {"perception_out", "perception_input"}
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
    a.setup(None)  # type: ignore[arg-type]
    a.reset()
    a.teardown()
