"""Patch-only behavior: native mounting and service-backed optimization samples."""

import numpy as np
import pytest
from dataclasses import replace

from avsectester.attack import AttackSegment, OfflineContext, OnlineContext
from avsectester.attack.optimization import PGD, NES
from avsectester.implemented_attacks.patch_attack import PatchAttack
from avsectester.implemented_attacks.patch_attack.offline import PatchOptimization
from avsectester.implemented_attacks.patch_attack.online import (
    CarlaPatchPanel, DEFAULT_PROP_MESH_PATH, MESH_PROP,
)
from avsectester.attack.insertion import PatchCompositor
from avsectester.data_types import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.avstacks.implemented.external_control import ExternalAVStack
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor
from avsectester.execution.services import ClosedLoopServices, RenderRequest
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.geometry.projection import project_box
from tests.fakes import CarlaNativeBoundary, ConstantAVStack, MemoryDataset, MemorySimulator, actor_state


@pytest.mark.parametrize(
    "options, expected_prop, expected_attributes, simulate_physics, avstack_type, simulator_options", [
    pytest.param({}, MESH_PROP, {"mesh_path": DEFAULT_PROP_MESH_PATH, "mass": "1"}, False,
                 ConstantAVStack, {}, id="default-mesh"),
    pytest.param({"prop": MESH_PROP, "prop_mesh_path": "/Game/User/Panel.Panel"}, MESH_PROP,
                 {"mesh_path": "/Game/User/Panel.Panel", "mass": "1"}, False,
                 ConstantAVStack, {}, id="custom-mesh"),
    pytest.param({"prop": "static.prop.streetbarrier"}, "static.prop.streetbarrier", {}, True,
                 ConstantAVStack, {}, id="blueprint"),
    pytest.param({}, MESH_PROP, {"mesh_path": DEFAULT_PROP_MESH_PATH, "mass": "1"}, False,
                 ExternalAVStack, {"ego_control": "ros2", "realtime_factor": 1000.},
                 id="external-world-patch"),
])
def test_create_runtime(monkeypatch, options, expected_prop, expected_attributes, simulate_physics,
                        avstack_type, simulator_options):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = CarlaRecipe({"client": {"map_name": "Town03"},
        "ego": {"type": "CarlaMobileActor", "spawn": 0, "vehicle": "vehicle.a", "sensors": [],
                "blueprint_attributes": {"role_name": "ego"}},
        "lead": {"gap": 8., "vehicle": "vehicle.b"}}, 2, .1,
        {"ego": PrescribedMotion(constant_control=Control()),
         "lead": PrescribedMotion(constant_control=Control())})
    dataset = CarlaDataset({"scene": recipe})
    frames = dataset.frames("scene")
    attack = PatchAttack(decision_steps=2, assets=np.full((3, 4, 4), [0, 255, 0, 255], np.uint8),
                        **options)
    assert attack.configuration_parameters()["prop_mesh_path"] == expected_attributes.get("mesh_path")
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
        assert panel.type_id == expected_prop
        assert panel.attributes == expected_attributes
        assert panel.simulate_physics == simulate_physics
        assert panel.attached_to is context.simulator_runtime.native["actors"]["lead"]
        transform = panel.get_transform()
        np.testing.assert_allclose([transform.location.x, transform.location.y, transform.location.z],
                                   [-1.81, 0., .7])
        assert transform.rotation.pitch == pytest.approx(90.)
        np.testing.assert_array_equal(panel.textures["Diffuse"], artifacts)
        deployed.append(panel)
        return value

    simulator = CarlaSimulator({"rate": 10., "synchronous": True, "seed": 0}, settle_iters=0,
                               **simulator_options)
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, avstack_type(), RunRequest(
            2, start=dataset.scenario_start("scene", frames[0]),
            attacks=(attack.create_runtime(segment, artifacts),),
            hooks=HookSet([Hook("simulator.step.pre", inspect_native_panel)]),
        ))
    assert log.status == "completed"
    assert deployed and all(not panel.is_alive for panel in deployed)
    missing_host = replace(segment, user_data={**segment.user_data, "host": "absent"})
    with simulator.create_runtime() as runtime:
        with pytest.raises(ValueError, match="host 'absent'"):
            ClosedLoopRunner().run(runtime, avstack_type(), RunRequest(
                1, start=dataset.scenario_start("scene", frames[0]),
                attacks=(attack.create_runtime(missing_host, artifacts),),
            ))


def test_prop_mesh_path_requires_mesh_blueprint():
    with pytest.raises(ValueError, match="prop_mesh_path is only supported"):
        PatchAttack(decision_steps=2, prop="static.prop.streetbarrier",
                    prop_mesh_path=DEFAULT_PROP_MESH_PATH)


def test_carla_patch_panel(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    panel = CarlaPatchPanel((0., 0., 1.), (0., 0., 0.),
                            np.full((3, 4, 4), 255, np.uint8), mesh_path=DEFAULT_PROP_MESH_PATH)
    with CarlaSimulator({"rate": 10.}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        native = runtime.native
        actors = panel.apply(native["world"], native["actors"]["ego"])
        try:
            assert len(actors) == 1 and actors[0].is_alive
            with pytest.raises(ValueError, match="requires a nonempty mesh_path"):
                replace(panel, mesh_path=None).apply(native["world"], native["actors"]["ego"])
            with pytest.raises(ValueError, match="mesh_path is only supported"):
                replace(panel, prop="static.prop.streetbarrier").apply(
                    native["world"], native["actors"]["ego"])
        finally:
            for actor in actors:
                actor.destroy()


@pytest.mark.parametrize("algorithm", [
    pytest.param(PGD(epochs=2, batch_size=1), id="PGD"),
    pytest.param(NES(steps=2, population_size=4, learning_rate=2000., seed=0), id="NES"),
])
def test_run_offline(algorithm):
    dataset, simulator = MemoryDataset(), MemorySimulator()
    descriptor = ScorerDescriptor("brightness", {"images": "BCHW float tensor [0,1]"},
                                   {"scores": "scalar"}, {"scores": "tensor"}, ("images",))
    received = []
    scores, targets = [], []

    def score(avstack, images, batch_targets):
        received.extend(images.detach().cpu().numpy().copy())
        values = images.mean(dim=(1, 2, 3))
        scores.extend(values.detach().cpu().tolist())
        targets.extend(batch_targets)
        return values

    driver = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, score)])
    attack = PatchAttack(decision_steps=1, max_windows=1,
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
            view = rendered.view
            host = view.actors[segment.user_data["host"]]
            expected_box = project_box(
                view.camera, view.cam_from_world, host.box_transform(), host.extent)
        finally:
            runtime.stop(replace(online, stage="stop"))
    assert artifacts.shape == (4, 4, 4) and artifacts.dtype == np.uint8
    assert received and any(np.ptp(image) > 0 for image in received)
    assert exported_score == pytest.approx(scores[-1], abs=2e-5)
    assert scores[-1] < scores[0]
    assert all(target.box is not None for target in targets)
    assert all(target.box == expected_box for target in targets)
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


def test_offline_candidate_frames():
    targets = set()

    def score(avstack, images, batch_targets):
        targets.update(value.box for value in batch_targets)
        return images.mean(dim=(1, 2, 3))

    descriptor = ScorerDescriptor(
        "brightness", {"images": "BCHW float tensor [0,1]"}, {"scores": "scalar"},
        {"scores": "tensor"}, ("images",),
    )
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), scorers=[(descriptor, score)])
    options = PatchOptimization(PGD(epochs=1, batch_size=2), scorer_id="brightness",
                                size=(4, 4), device="cpu")
    attack = PatchAttack(decision_steps=3, max_windows=1, compositor=PatchCompositor(),
                        optimization=options)
    dataset = MemoryDataset()
    with ClosedLoopServices(dataset, MemorySimulator(), adapter) as services:
        segment, = attack.build_metadata(services.scenario("scene", seed=0))
        attack.run_offline(OfflineContext(services, (segment,)))
        with services.session(0) as session:
            frames = dataset.frames("scene")[:4]
            boxes = []
            for frame in frames:
                view = session.render(RenderRequest("scene", frame, "front")).view
                host = view.actors[segment.user_data["host"]]
                boxes.append(project_box(
                    view.camera, view.cam_from_world, host.box_transform(), host.extent))
    assert len(set(boxes)) == 4
    assert targets == set(boxes[:3])
