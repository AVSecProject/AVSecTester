"""Component names and attack discovery use one public resolution rule."""

import pytest

from avsectester.attack import Attack
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.experiment.type_registry import (
    available_attacks, resolve_type,
)
from tests.fakes import ConstantAVStack


def test_resolve_type():
    assert resolve_type("avsectester.attack.insertion.placement:WorldPlacement") is WorldPlacement
    assert resolve_type("tests.fakes:ConstantAVStack") is ConstantAVStack
    assert resolve_type("PatchAttack").__name__ == "PatchAttack"
    with pytest.raises(ValueError, match="Available attacks"):
        resolve_type("PatchAtack")


def test_available_attacks():
    names = available_attacks()
    assert {"PatchAttack", "ObjectInsertionAttack", "PhantomInjectionAttack", "SpeedFeedbackAttack",
            "StopSignPatchAttack"} <= set(names)
    assert names == tuple(sorted(names))
    assert all(issubclass(resolve_type(name), Attack) for name in names)
