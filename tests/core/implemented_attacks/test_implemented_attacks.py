"""Every supplied attack follows one public preparation and execution template."""

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.attack import OfflineContext
from avsectester.attack.insertion import PatchCompositor, PlaneAsset
from avsectester.rendering.textures import CheckerboardTexture
from avsectester.attack.insertion.placement import WorldPlacement
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.execution.services import ClosedLoopServices
from avsectester.experiment import load_configuration
from tests.fakes import MemorySimulator


ATTACKS = [
    pytest.param("PatchAttack", {
        "decision_steps": 2, "assets": CheckerboardTexture(size=8), "max_windows": 1,
        "compositor": PatchCompositor(),
    }, {"hosts_per_window": 0}, "default_driving_options", "memory_dataset", id="patch"),
    pytest.param("ObjectInsertionAttack", {
        "decision_steps": 2, "max_segments": 1,
        "assets": PlaneAsset(np.full((8, 8, 4), [240, 20, 20, 255], np.uint8), 1., 1.),
        "placement": WorldPlacement((7., 0., 1.)), "compositor": PatchCompositor(),
    }, {"min_visible_fraction": 2.}, "default_driving_options", "memory_dataset", id="object-insertion"),
    pytest.param("PhantomInjectionAttack", {"decision_steps": 2, "max_segments": 1},
                 {"score": 2.}, "native_driving_options", "memory_dataset", id="phantom-injection"),
    pytest.param("SpeedFeedbackAttack", {
        "decision_steps": 2, "target_speed": 4., "initial_bias": .5, "max_segments": 1,
    }, {"max_bias": -1.}, "default_driving_options", "memory_dataset", id="speed-feedback"),
    pytest.param("StopSignPatchAttack", {"approach_m": 3., "hold_steps": 1},
                 {"initial_patch": "red"}, "default_driving_options",
                 "stop_approach_dataset", id="stop-sign-patch"),
]


@pytest.mark.parametrize("attack_type,options,invalid,driving_options_fixture,dataset_fixture", ATTACKS)
def test_implemented_attack(attack_type, options, invalid, driving_options_fixture, dataset_fixture,
                            request, tmp_path):
    configuration = {
        "dataset": request.getfixturevalue(dataset_fixture), "simulator": MemorySimulator(),
        "avstack": {"type": "tests.fakes:ConstantAVStack", "options":
                    request.getfixturevalue(driving_options_fixture)},
        "attacks": {"attack": {"type": attack_type, "options": options}},
    }
    configured = load_configuration(configuration)
    attack = configured.attacks["attack"]
    with ClosedLoopServices(configured.dataset, configured.simulator, configured.avstack) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        assert len(metadata) == 1 and metadata[0].decision_steps == 2
        artifacts = attack.run_offline(OfflineContext(services, tuple(metadata)))
    history_driver = DrivingAdapter(lambda seed: configured.avstack.factory(seed), past_steps=1)
    with ClosedLoopServices(configured.dataset, configured.simulator, history_driver) as services:
        eligible = attack.build_metadata(services.scenario("scene", seed=0))
        assert eligible and all(segment.start.position >= 1 for segment in eligible)
    attack.save_artifacts(artifacts, tmp_path)
    restored = attack.load_artifacts(tmp_path)
    segment = metadata[0]
    logs = []
    for candidate in (artifacts, restored):
        with configured.simulator.create_runtime(seed=0) as simulator_runtime:
            logs.append(ClosedLoopRunner().run(simulator_runtime, configured.avstack.factory(0), RunRequest(
                segment.decision_steps, start=configured.dataset.scenario_start("scene", segment.start),
                attacks=(attack.create_runtime(segment, candidate),),
                recording=RecordingPolicy(("front",), ("raw", "model_input")),
            )))
    assert all(log.status == "completed" for log in logs)
    actions = [action for row in logs[0].steps for action in row.actions]
    actions.extend(action for event in logs[0].events if event.kind == "initialization"
                   for action in event.data.get("actions", ()))
    assert any(action.get("outcome") == "applied" for action in actions)
    for original, loaded in zip(logs[0].steps, logs[1].steps):
        np.testing.assert_array_equal(loaded.model_observation.sensor_data["front"],
                                      original.model_observation.sensor_data["front"])
        assert loaded.model_observation.ego_speed == original.model_observation.ego_speed
        assert loaded.executed_command == original.executed_command
        assert loaded.actions == original.actions
    configuration["attacks"]["attack"]["options"] = {**options, **invalid}
    with pytest.raises((ValueError, TypeError)):
        load_configuration(configuration)
