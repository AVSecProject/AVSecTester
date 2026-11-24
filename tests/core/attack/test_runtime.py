"""Online-phase records validate subscriptions, actions and callback output access."""
import pytest

from avsectester.attack import AVStackAttackAction, OnlineContext, ResultSubscription


def test_result_subscription():
    assert ResultSubscription("physical_state").kind == "physical_state"
    with pytest.raises(ValueError, match="Unknown attack result kind"):
        ResultSubscription("commands")


def test_record_output():
    context = OnlineContext(0, "observation", 0., 0)
    with pytest.raises(ValueError, match="nonempty"):
        context.record_output("", 1)
    with pytest.raises(RuntimeError, match="during a run"):
        context.record_output("decision", 1)


def test_avstack_attack_action():
    action = AVStackAttackAction("command", lambda value, context: value)
    assert action.apply(3, OnlineContext(0, "command", 0., 0)) == 3
    with pytest.raises(ValueError, match="nonempty"):
        AVStackAttackAction("", lambda value, context: value)
    with pytest.raises(TypeError, match="callable"):
        AVStackAttackAction("command", None)
