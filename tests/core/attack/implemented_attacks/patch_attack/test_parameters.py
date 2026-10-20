"""Attack construction rejects bad parameters before touching data or simulators."""
from types import SimpleNamespace

import pytest

from avsectester.attack.implemented_attacks.patch_attack import PatchAttack


@pytest.mark.parametrize("options,exception", [
    ({"decision_steps": True}, ValueError),
    ({"decision_steps": 0}, ValueError),
    ({"decision_steps": 1.5}, ValueError),
    ({"decision_steps": 2, "keep_tail": 1}, TypeError),
])
def test_bad_segment_options_fail_at_attack_construction(options, exception):
    with pytest.raises(exception):
        PatchAttack(**options)

@pytest.mark.parametrize("count", [0, True, -1, 1.5])
def test_count_limits_fail_during_construction(count):
    with pytest.raises(ValueError, match="max_windows"):
        PatchAttack(decision_steps=2, max_windows=count)

@pytest.mark.parametrize("optimization", ["pgd", {}, {"algorithm": "pgd"}, object()])
def test_patch_requires_a_typed_optimization(optimization):
    with pytest.raises(TypeError, match="PatchOptimization"):
        PatchAttack(decision_steps=1, optimization=optimization)

@pytest.mark.parametrize("texture", ["image.png", {"image": "image.png"}, {"pattern": "typo"}])
def test_patch_does_not_guess_texture_specs(texture):
    with pytest.raises(TypeError, match="texture"):
        PatchAttack(decision_steps=1, texture=texture)

def test_patch_missing_artifact_never_falls_back_to_original_asset():
    attack = PatchAttack(decision_steps=1)
    with pytest.raises(ValueError, match="explicit artifacts"):
        attack.insertion(None, host="host")
    with pytest.raises(ValueError, match="explicit artifacts"):
        attack.create_runtime(None, None)

def test_no_metadata_skips_patch_optimizer_and_returns_original_assets(monkeypatch):
    from avsectester.attack.implemented_attacks.patch_attack.offline import PatchOptimization
    from avsectester.attack.optimization.whitebox import PGD
    from avsectester.attack.implemented_attacks.patch_attack import offline

    attack = PatchAttack(decision_steps=1, optimization=PatchOptimization(PGD(steps=1)))
    monkeypatch.setattr(offline, "optimize_patch", lambda *args: pytest.fail("no metadata to optimize"))
    assert attack.run_offline(SimpleNamespace(metadata=())) is attack.assets

@pytest.mark.parametrize("options", [
    {"candidate_hosts": "lead"}, {"candidate_hosts": []}, {"candidate_hosts": ["lead", "lead"]},
    {"hosts_per_window": 0}, {"hosts_per_window": True}, {"emissive": 1},
])
def test_patch_host_and_deployment_options_are_checked(options):
    with pytest.raises((ValueError, TypeError)):
        PatchAttack(decision_steps=1, **options)


def test_patch_mount_has_no_placeholder_actor_and_retains_local_options():
    from avsectester.attack.placement import AttachedPlacement

    attack = PatchAttack(decision_steps=1, anchor="top_center", offset_m=(.1, .2, .3))
    assert not hasattr(attack, "placement")
    assert attack.placement_for("first") == AttachedPlacement("first", "top_center", (.1, .2, .3))
    assert attack.placement_for("second") == AttachedPlacement("second", "top_center", (.1, .2, .3))


def test_patch_rejects_misspelled_mount_without_a_placeholder_host():
    with pytest.raises(ValueError, match="anchor"):
        PatchAttack(decision_steps=1, anchor="roof_top")
