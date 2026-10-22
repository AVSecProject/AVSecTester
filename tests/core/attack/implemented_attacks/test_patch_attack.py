"""Patch-only behavior: native mounting and service-backed optimization samples."""

import numpy as np
import pytest
from dataclasses import replace

from avsectester.attack import AttackSegment, OfflineContext
from avsectester.attack.implemented_attacks.patch_attack import PatchAttack
from avsectester.attack.implemented_attacks.patch_attack.offline import PatchOptimization
from avsectester.attack.implemented_attacks.patch_attack.optimization_data import SessionDataSource
from avsectester.attack.insertion import PatchCompositor
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.services import ClosedLoopServices, DrivingAdapter, ScorerDescriptor
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from tests.fakes import CarlaNativeBoundary, ConstantAVStack, MemoryDataset, MemorySimulator


def test_world_mount_and_cleanup(monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
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
    with pytest.raises(ValueError, match="missing keys: placement. Required keys: host, placement"):
        attack.create_runtime(replace(segment, user_data={"host": "lead"}), artifacts)
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


def test_optimization_data_and_rendering():
    pytest.importorskip("torch")
    from avsectester.attack.optimization import PGD
    from avsectester.attack.implemented_attacks.patch_attack.offline import render_candidate

    dataset, simulator = MemoryDataset(), MemorySimulator()
    descriptor = ScorerDescriptor("brightness", {"image": "CHW float tensor [0,1]"},
                                   {"score": "scalar"}, {"score": "float"}, ("image",))
    driver = DrivingAdapter(lambda seed: ConstantAVStack(),
                            scorers=[(descriptor, lambda avstack, image, target: image.mean())])
    attack = PatchAttack(decision_steps=2, max_windows=1,
                        compositor=PatchCompositor())
    with ClosedLoopServices(dataset, simulator, driver) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        with services.session(0) as session:
            sample, = SessionDataSource(session, attack, metadata, size=(4, 4), device="cpu",
                brightness=(1., 1.), contrast=(1., 1.)).sample(1)
            assert sample.x.shape == (3, 80, 100)
            assert sample.context["mask"].any()
            assert sample.target.box[2] > sample.target.box[0]
            candidate = np.full((4, 4, 4), [240, 0, 0, 255], np.uint8)
            rendered = render_candidate(session, attack, metadata[0], sample.context["frame"],
                                         candidate, {}, sample.context["observation"])
            assert np.any(rendered.observation.sensor_data["front"][..., 0] == 240)
        attack.optimization = PatchOptimization(PGD(steps=2, batch=1), scorer_id="brightness",
            device="cpu", size=(4, 4), brightness=(1., 1.), contrast=(1., 1.))
        artifacts = attack.run_offline(OfflineContext(services, tuple(metadata)))
    assert artifacts.shape == (4, 4, 4) and artifacts.dtype == np.uint8
    np.testing.assert_array_equal(artifacts[..., 3], np.full((4, 4), 255, np.uint8))
