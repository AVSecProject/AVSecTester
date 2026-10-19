"""Artifact reuse, ordered treatment assembly and independent adaptive execution."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from avsectester.attacks.adaptive import SpeedFeedbackAttack
from avsectester.closed_loop.avstack import AVStack
from avsectester.backend import run
from avsectester.interventions import prepare_from_config
from avsectester.closed_loop.datatypes import Control
from avsectester.preparation import (
    Intervention, PreparedIntervention, Treatment, prepare_intervention,
)
from avsectester.closed_loop.hooks import Hook, Plugin, Runtime
from avsectester.scenarios.source import ScenarioInstance
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer


def scene_instance():
    return ScenarioInstance(
        lambda: NuRecSimulator({}, renderer=StubRenderer(height=2, width=2)),
        None, {"scene": "example"},
    )


class ReportedSpeed(Intervention):
    def __init__(self, speed=4):
        self.speed = speed
        self.prepared = []

    def prepare(self, scenario):
        self.prepared.append(scenario)
        return {"speed": self.speed}

    def hooks(self, artifacts):
        class Handler(Plugin):
            def reset(self, context):
                self.seen = []

            def transform(self, observation, context):
                self.seen.append(context.step)
                return replace(observation, ego_speed=artifacts["speed"])

        return (Hook("observation", Handler()),)


def test_preparation_artifacts_are_reused_with_fresh_handlers_and_saved_provenance(tmp_path):
    scenario = scene_instance()
    method = ReportedSpeed()
    prepared = prepare_intervention(method, scenario, name="spoof")
    first, second = Treatment((prepared,)).runtime(), Treatment((prepared,)).runtime()
    assert method.prepared == [scenario]
    assert first.hooks[0].handler is not second.hooks[0].handler
    assert prepared.provenance == {"scenario": scenario.provenance}
    scenario.provenance["conditions"]["world"]["weather"] = "changed"
    assert "weather" not in prepared.provenance["scenario"]["conditions"]["world"]

    path = tmp_path / "artifacts" / "spoof.json"
    prepared.save(path)
    restored = PreparedIntervention.load(method, path)
    assert restored.name == "spoof" and restored.kind == "attack"
    assert restored.artifacts == {"speed": 4}
    assert restored.provenance == prepared.provenance
    assert restored.hooks()[0].handler is not first.hooks[0].handler
    supplied = prepare_intervention(method, scenario, artifacts=None)
    assert supplied.artifacts is None and method.prepared == [scenario]


def test_custom_artifact_codec_restores_arrays_and_rejects_unencoded_objects(tmp_path):
    class ArrayMethod(ReportedSpeed):
        def export_artifacts(self, artifacts):
            return {"array": artifacts.tolist()}

        def import_artifacts(self, data):
            return np.asarray(data["array"])

    data = np.array([[1, 2], [3, 4]])
    path = tmp_path / "array.json"
    method = ArrayMethod()
    prepared = prepare_intervention(method, scene_instance(), artifacts=data, kind="defense")
    prepared.save(path)
    restored = PreparedIntervention.load(method, path)
    np.testing.assert_array_equal(restored.artifacts, data)
    assert restored.kind == "defense"
    with pytest.raises(TypeError):
        prepare_intervention(ReportedSpeed(), scene_instance(), artifacts=data).save(path)
    # Failed serialization must not replace an existing usable artifact file.
    np.testing.assert_array_equal(PreparedIntervention.load(method, path).artifacts, data)


def test_invalid_preparation_metadata_is_rejected_before_preparing():
    method = ReportedSpeed()
    with pytest.raises(ValueError, match="position"):
        prepare_intervention(method, scene_instance(), position="invalid")
    assert method.prepared == []


def test_treatment_filters_preserve_definition_order_and_explicit_runtime_position():
    scenario = scene_instance()
    definitions = [
        prepare_intervention(ReportedSpeed(speed), scenario, kind=kind, name=name,
                             position=position)
        for speed, kind, name, position in [
            (1, "attack", "legacy", "before"),
            (2, "attack", "A1", "after"),
            (3, "defense", "D1", "after"),
            (4, "attack", "A2", "after"),
        ]
    ]
    base_handler = lambda value, context: value
    marker = object()
    base = Runtime([Hook("observation", base_handler)], case=marker, seed=99)
    treatment = Treatment(tuple(definitions))
    runtime = treatment.runtime(base)
    assert runtime.hooks[1].handler is base_handler
    assert runtime.case is marker and runtime.seed == 99
    assert [item.name for item in treatment.subset({"attack"}).interventions] == [
        "legacy", "A1", "A2",
    ]
    assert [item.name for item in treatment.interventions] == ["legacy", "A1", "D1", "A2"]
    assert len(treatment.runtime(kinds={"defense"}).hooks) == 1
    assert treatment.runtime(kinds=set()).hooks == ()
    assert base.hooks == (Hook("observation", base_handler),)

    class Driver(AVStack):
        def __call__(self, observation):
            # A2 follows D1. Regrouping defenses at the end would produce speed 3.
            assert observation.ego_speed == 4
            return Control()

    run(scenario.make_backend(), Driver(), 1, runtime=runtime)


def test_configuration_prepares_methods_once_and_adapts_legacy_convenience_fields():
    config = {
        "attacks": [{"stage": "perception", "hook": {"type": "PhantomInjection"}}],
        "patches": [{"target": "lead", "texture": {"pattern": "checkerboard"}}],
        "interventions": [{
            "factory": "avsectester.attacks.adaptive:SpeedFeedbackAttack",
            "options": {"target_speed": 4}, "name": "feedback", "kind": "defense",
            "artifacts": {"target_speed": 2, "gain": 0.5, "max_bias": 3},
        }],
    }
    treatment = prepare_from_config(config, scene_instance())
    first, second = treatment.runtime(), treatment.runtime()
    assert [hook.stage for hook in first.hooks] == [
        "perception.post", "simulator.setup", "observation",
    ]
    assert first.hooks[0].handler.config == {"type": "PhantomInjection"}
    assert first.hooks[1].handler.patches == config["patches"]
    assert first.hooks[1].handler is not second.hooks[1].handler
    assert first.hooks[2].handler.target_speed == 2
    assert [hook.stage for hook in treatment.runtime(kinds={"defense"}).hooks] == ["observation"]


def test_disabled_methods_and_raw_attack_hooks_are_not_constructed(monkeypatch):
    from avsectester import interventions

    method = ReportedSpeed()
    constructor = Mock(return_value=method)
    module = SimpleNamespace(build=constructor)
    monkeypatch.setattr(interventions, "import_module", lambda _: module)
    config = {
        "runtime": {"hooks": [{"stage": "command", "factory": "example:build"}]},
        "interventions": [{"factory": "example:build", "kind": "attack"}],
    }
    instance = scene_instance()
    runtime = interventions.configured_runtime(config, instance=instance, kinds={"defense"})
    assert runtime.hooks == ()
    constructor.assert_not_called()
    assert method.prepared == []
    config["interventions"][0]["kind"] = "defense"
    treatment = prepare_from_config(config, instance, kinds={"defense"})
    constructor.assert_called_once_with()
    assert method.prepared == [instance] and len(treatment.interventions) == 1


def test_configured_pair_reuses_preparation_and_preserves_selected_match(monkeypatch):
    from avsectester import interventions

    instance = replace(scene_instance(), match=object())
    method = ReportedSpeed()
    monkeypatch.setattr(interventions, "import_module", lambda _: SimpleNamespace(build=lambda: method))
    config = {"interventions": [{"factory": "example:build"}]}
    treatment = prepare_from_config(config, instance)
    clean, attacked = interventions.configured_runtime_pair(
        config, instance=instance, prepared=treatment,
    )
    assert method.prepared == [instance]
    assert clean.case is instance.match and attacked.case is instance.match
    assert clean.hooks == () and len(attacked.hooks) == 1
    override = Runtime(seed=31)
    _, overridden = interventions.configured_runtime_pair(
        config, instance=instance, prepared=treatment, attacked_runtime=override,
    )
    assert overridden is override and method.prepared == [instance]


@pytest.mark.parametrize("entry,match", [
    ({"factory": "avsectester.attacks.adaptive:SpeedFeedbackAttack", "options": {
        "target_speed": 4}, "unknown": True}, "Unknown intervention"),
    ({"kind": "attack"}, "requires a factory"),
    ({"factory": "avsectester.attacks.adaptive:SpeedFeedbackAttack", "options": {
        "target_speed": 4}, "kind": "invalid"}, "kind"),
])
def test_intervention_configuration_rejects_invalid_contracts(entry, match):
    with pytest.raises(ValueError, match=match):
        prepare_from_config({"interventions": [entry]}, scene_instance())


def test_adaptive_attack_uses_step_feedback_and_resets_between_runs():
    backend = NuRecSimulator({"ego0": {"speed": 2}}, renderer=StubRenderer(height=2, width=2))
    prepared = prepare_intervention(SpeedFeedbackAttack(5, gain=0.5), scene_instance())
    runtime = Treatment((prepared,)).runtime()
    handler = runtime.hooks[0].handler
    received = []

    class Driver(AVStack):
        def __call__(self, observation):
            received.append(observation.ego_speed)
            return Control()

    first = run(backend, Driver(), frames=3, runtime=runtime)
    assert received == [2, 3.5, 5]
    assert handler.feedback == [2, 2, 2]
    assert all(record.speed == 2 for record in first.records)
    run(backend, Driver(), frames=1, runtime=runtime)
    assert received[-1] == 2 and handler.feedback == [2]


def test_physical_panel_uses_selected_role_and_rejects_ambiguous_hosts(monkeypatch):
    from avsectester.attacks.patch.physical_patch import PhysicalPatch, PhysicalPatchPlugin

    actor = SimpleNamespace(id=10)
    backend = SimpleNamespace(native={"world": object(), "actors": {"lead": actor}},
                              setup_tick=Mock())
    apply = Mock()
    monkeypatch.setattr(PhysicalPatch, "apply", apply)
    plugin = PhysicalPatchPlugin([{"target": "hosts[0]"}])
    context = SimpleNamespace(simulator=backend, case=SimpleNamespace(binding_ids={
        "hosts": ("lead", "other"),
    }))
    plugin.transform(None, context)
    assert apply.call_args.args[1] is actor
    plugin.patches[0]["target"] = "hosts"
    with pytest.raises(ValueError, match="explicit index"):
        plugin.transform(None, context)
    plugin.patches[0]["target"] = 10
    plugin.transform(None, SimpleNamespace(simulator=backend, case=None))
    assert apply.call_count == 2
