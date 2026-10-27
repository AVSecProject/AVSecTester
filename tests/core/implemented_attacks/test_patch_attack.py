"""Patch-only behavior: native mounting and service-backed optimization samples."""

import numpy as np
import pytest
from dataclasses import replace

from avsectester.attack import AttackSegment, OfflineContext, OnlineContext
from avsectester.attack.optimization import PGD, NES
from avsectester.implemented_attacks.patch_attack import PatchAttack
from avsectester.implemented_attacks.patch_attack.offline import PatchOptimization
from avsectester.attack.insertion import PatchCompositor
from avsectester.data_types import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor
from avsectester.execution.services import ClosedLoopServices, RenderRequest
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from tests.fakes import CarlaNativeBoundary, ConstantAVStack, MemoryDataset, MemorySimulator, actor_state


def test_create_runtime(monkeypatch):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = CarlaRecipe({"client": {"map_name": "Town03"},
        "ego": {"type": "CarlaMobileActor", "spawn": 0, "vehicle": "vehicle.a", "sensors": []},
        "lead": {"gap": 8., "vehicle": "vehicle.b"}}, 2, .1,
        {"ego": PrescribedMotion(constant_control=Control()),
         "lead": PrescribedMotion(constant_control=Control())})
    dataset = CarlaDataset({"scene": recipe})
    frames = dataset.frames("scene")
    attack = PatchAttack(decision_steps=2, assets=np.full((3, 4, 4), [0, 255, 0, 255], np.uint8))
    artifacts = np.full((3, 4, 4), [255, 0, 0, 255], np.uint8)
    segment = AttackSegment("scene", frames[0], frames[2], user_data={
        "host": "lead", "placement": {"anchor": "rear_center", "offset_m": [-.01, 0., 0.]},
        "deployment": "world",
    })
    with pytest.raises(ValueError, match="missing keys: host, placement. Required keys: host, placement"):
        attack.create_runtime(replace(segment, user_data={}), artifacts)
    deployed = []

    def inspect_native_panel(value, context):
        world = context.simulator_runtime.native["world"]
        panels = [actor for actor in world.get_actors() if actor.type_id.startswith("static.prop")]
        assert len(panels) == 1
        panel = panels[0]
        assert panel.is_alive
        assert panel.attached_to is context.simulator_runtime.native["actors"]["lead"]
        transform = panel.get_transform()
        np.testing.assert_allclose([transform.location.x, transform.location.y, transform.location.z],
                                   [-1.81, 0., .7])
        assert transform.rotation.pitch == pytest.approx(90.)
        np.testing.assert_array_equal(panel.textures["Diffuse"], artifacts)
        deployed.append(panel)
        return value

    with CarlaSimulator({"rate": 10., "synchronous": True, "seed": 0}, settle_iters=0).create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            2, start=dataset.scenario_start("scene", frames[0]),
            attacks=(attack.create_runtime(segment, artifacts),),
            hooks=HookSet([Hook("observation", inspect_native_panel)]),
        ))
    assert log.status == "completed"
    assert deployed and all(not panel.is_alive for panel in deployed)
    missing_host = replace(segment, user_data={**segment.user_data, "host": "absent"})
    with CarlaSimulator({"rate": 10.}, settle_iters=0).create_runtime() as runtime:
        with pytest.raises(ValueError, match="host 'absent'"):
            ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
                1, start=dataset.scenario_start("scene", frames[0]),
                attacks=(attack.create_runtime(missing_host, artifacts),),
            ))


@pytest.mark.parametrize("algorithm", [
    pytest.param(PGD(steps=2, batch=1), id="PGD"),
    pytest.param(NES(steps=2, population_size=4, learning_rate=2000., seed=0), id="NES"),
])
def test_run_offline(algorithm):
    dataset, simulator = MemoryDataset(), MemorySimulator()
    descriptor = ScorerDescriptor("brightness", {"image": "CHW float tensor [0,1]"},
                                   {"score": "scalar"}, {"score": "float"}, ("image",))
    received = []
    scores, targets = [], []

    def score(avstack, image, target):
        received.append(image.detach().cpu().numpy().copy())
        value = image.mean()
        scores.append(float(value.detach()))
        targets.append(target)
        return value

    driver = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, score)])
    attack = PatchAttack(decision_steps=2, max_windows=1,
                        compositor=PatchCompositor(),
                        optimization=PatchOptimization(algorithm, scorer_id="brightness",
                            device="cpu", size=(4, 4), brightness=(1., 1.), contrast=(1., 1.)))
    with ClosedLoopServices(dataset, simulator, driver) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        artifacts = attack.run_offline(OfflineContext(services, tuple(metadata)))
        segment = metadata[0]
        runtime = attack.create_runtime(segment, artifacts)
        online = OnlineContext(0, "setup", segment.start.timestamp, segment.start.position)
        try:
            actions = tuple(runtime.setup(online))
            actions += tuple(runtime.step(replace(online, stage="step")))
            with services.session(0) as session:
                rendered = session.render(RenderRequest(
                    segment.scenario_id, segment.start, "front", actions=actions))
            exported_score = np.asarray(rendered.observation.sensor_data["front"]).mean() / 255
        finally:
            runtime.stop(replace(online, stage="stop"))
    assert artifacts.shape == (4, 4, 4) and artifacts.dtype == np.uint8
    assert received and any(np.ptp(image) > 0 for image in received)
    assert exported_score < scores[0]
    assert all(target.box is not None for target in targets)
    assert all(target.box == targets[0].box for target in targets)
    np.testing.assert_array_equal(artifacts[..., 3], np.full((4, 4), 255, np.uint8))


def test_candidate_hosts_order():
    simulator = MemorySimulator(actors={
        "left": actor_state((10., 2., 0.)),
        "right": actor_state((10., -2., 0.)),
    })
    with ClosedLoopServices(MemoryDataset(), simulator) as services:
        scenario = services.scenario("scene", seed=0)
        first = PatchAttack(decision_steps=2, max_windows=1, hosts_per_window=2,
                            candidate_hosts=("right", "left"), compositor=PatchCompositor())
        second = PatchAttack(decision_steps=2, max_windows=1, hosts_per_window=2,
                             candidate_hosts=("left", "right"), compositor=PatchCompositor())
        right_first = first.build_metadata(scenario)
        left_first = second.build_metadata(scenario)
    assert [item.user_data["host"] for item in right_first] == ["right", "left"]
    assert [item.user_data["host"] for item in left_first] == ["left", "right"]
    assert [item.user_data["rank"] for item in right_first] == [0, 1]
