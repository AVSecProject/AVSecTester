"""Component names and attack discovery use one public resolution rule."""

import pytest

from avsectester.attack import Attack
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.experiment.type_registry import (
    attack_directory_name, available_attacks, is_builtin_type_name, resolve_type,
)
from tests.fakes import ConstantAVStack


def test_resolve_type():
    assert resolve_type("avsectester.attack.insertion.placement:WorldPlacement") is WorldPlacement
    assert resolve_type("tests.fakes:ConstantAVStack") is ConstantAVStack
    assert resolve_type("PatchAttack").__name__ == "PatchAttack"
    with pytest.raises(ValueError, match="Available attacks"):
        resolve_type("PatchAtack")
    with pytest.raises(ValueError, match="package.module:name"):
        resolve_type("tests:fakes:ConstantAVStack")


@pytest.mark.parametrize("class_name, directory_name", [
    ("MyNNAttack", "my_nn_attack"), ("Attack2D", "attack2_d"), ("PatchAttack", "patch_attack"),
])
def test_attack_directory_name(class_name, directory_name):
    assert attack_directory_name(class_name) == directory_name


def test_available_attacks():
    names = available_attacks()
    assert {"PatchAttack", "ObjectInsertionAttack", "PhantomInjectionAttack", "SpeedFeedbackAttack"} <= set(names)
    assert names == tuple(sorted(names))
    assert all(issubclass(resolve_type(name), Attack) for name in names)


def test_is_builtin_type_name():
    assert is_builtin_type_name("CarlaDataset")
    assert not is_builtin_type_name("FasterRCNN")
    assert not is_builtin_type_name("tests.fakes:ConstantAVStack")
