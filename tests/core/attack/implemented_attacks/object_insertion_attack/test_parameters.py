"""Attack construction rejects bad parameters before touching data or simulators."""

import pytest

from avsectester.attack.implemented_attacks.object_insertion_attack import ObjectInsertionAttack
from avsectester.attack.placement import WorldPlacement


@pytest.mark.parametrize("options,exception", [
    ({"decision_steps": True}, ValueError),
    ({"decision_steps": 0}, ValueError),
    ({"decision_steps": 1.5}, ValueError),
    ({"decision_steps": 2, "keep_tail": 1}, TypeError),
])
def test_bad_segment_options_fail_at_attack_construction(options, exception):
    with pytest.raises(exception):
        ObjectInsertionAttack(placement=WorldPlacement((4., 0., 0.)), **options)

@pytest.mark.parametrize("count", [0, True, -1, 1.5])
def test_count_limits_fail_during_construction(count):
    with pytest.raises(ValueError, match="max_segments"):
        ObjectInsertionAttack(decision_steps=2, placement=WorldPlacement((4., 0., 0.)), max_segments=count)

def test_object_missing_artifact_never_falls_back_to_original_asset():
    attack = ObjectInsertionAttack(decision_steps=1, placement=WorldPlacement((4., 0., 0.)))
    with pytest.raises(ValueError, match="explicit artifacts"):
        attack.insertion(None)


def test_object_assets_have_one_name_and_placement_is_required():
    import inspect
    from avsectester.attack.implemented_attacks.object_insertion_attack.assets import SignAsset

    assets = SignAsset()
    attack = ObjectInsertionAttack(decision_steps=1, assets=assets,
                                   placement=WorldPlacement((4., 0., 0.)))
    assert attack.assets is assets and not hasattr(attack, "asset")
    assert inspect.signature(ObjectInsertionAttack).parameters["placement"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="placement"):
        ObjectInsertionAttack(decision_steps=1)
