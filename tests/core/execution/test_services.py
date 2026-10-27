"""Selection and preparation use isolated public service sessions."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.attack.insertion import InsertObject
from avsectester.attack.runtime import SensorAttackAction
from avsectester.data_types import Control, Observation, Unavailable
from avsectester.execution.frame_data_cache import FrameDataCache
from avsectester.execution.runner import RecordingPolicy
from avsectester.closed_loop.avstacks.driving_adapter import ComponentDescriptor, DrivingAdapter, ScorerDescriptor
from avsectester.execution.services import (
    CandidateRequest, ClosedLoopServices, RenderRequest, SensorRequest,
)
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.base import DeploymentPlan
from avsectester.rendering import ObjectVisibility
from avsectester.maps import ScenarioMap
from tests.fakes import (
    CarlaNativeBoundary, ConstantAVStack, HistoryAVStack, MemoryDataset, MemorySimulator, MinimalAttack,
)


def test_frame_data(carla_dataset, monkeypatch):
    boundary = CarlaNativeBoundary().install(monkeypatch, None)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    cache = FrameDataCache()
    frame = carla_dataset.frames("scene")[2]
    assert isinstance(carla_dataset.frame_data("scene", frame), Unavailable)
    with ClosedLoopServices(carla_dataset, simulator, frame_data_cache=cache) as services:
        with services.session(7) as session:
            generated = session.frame_data("scene", frame)
    expected = generated.ego.pose.copy()
    # Another service sharing the cache reuses the generated data without a CARLA client.
    monkeypatch.setattr(boundary, "build",
                        lambda *args, **kwargs: pytest.fail("frame data was generated again"))
    with ClosedLoopServices(carla_dataset, simulator, frame_data_cache=cache) as services:
        reused = services.scenario("scene", seed=7).frame_data(frame)
    np.testing.assert_allclose(reused.ego.pose, expected)


def test_scenario_map():
    scenario_map = ScenarioMap()
    dataset = MemoryDataset(maps={"scene": scenario_map})
    with ClosedLoopServices(dataset, MemorySimulator()) as services:
        assert services.scenario("scene", seed=0).scenario_map() is scenario_map


def test_read_sensors():
    data, simulator = MemoryDataset(), MemorySimulator()
    frames = data.frames("scene")
    environment = {"sensors": [{"type": "GaussianNoise", "camera": "front",
                                "options": {"severity": .2}}]}
    with ClosedLoopServices(data, simulator) as services:
        with services.session(7) as session:
            samples = [session.read_sensors(SensorRequest("scene", frame, ("front",), environment,
                                                        views=True))
                       for frame in frames[1:4]]
            assert [sample.observation.frame for sample in samples] == [1, 2, 3]
        for sample in samples:
            with services.session(7) as independent:
                replayed = independent.read_sensors(SensorRequest(
                    "scene", sample.frame, ("front",), environment))
                assert replayed.views == {}
            np.testing.assert_array_equal(sample.observation.sensor_data["front"],
                                          replayed.observation.sensor_data["front"])
            assert sample.provenance["seed"] == 7
            assert np.any(sample.views["front"].data != sample.observation.sensor_data["front"])
        with services.session(7) as session:
            request = SensorRequest("scene", frames[0], ("front",), views=True)
            captured = session.read_sensors(request)
            original = captured.views["front"].data.copy()
            captured.views["front"].data[:] = 0
            np.testing.assert_array_equal(session.read_sensors(request).views["front"].data, original)
            with pytest.raises(ValueError, match="Unavailable sensors"):
                session.read_sensors(SensorRequest("scene", frames[0], ("missing",)))
        with ClosedLoopServices(data, MemorySimulator(can_render=False)) as limited, limited.session(7) as session:
            with pytest.raises(NotImplementedError, match="The simulator does not provide sensor views"):
                session.read_sensors(SensorRequest("scene", frames[0], ("front",), views=True))


def test_render(make_object_in_front):
    class Fill(SensorAttackAction):
        sensor = "front"

        def apply(self, data, view):
            data[:] = 200
            return data

    data = MemoryDataset()
    frame = data.frames("scene")[2]
    with ClosedLoopServices(data, MemorySimulator()) as services, services.session(3) as session:
        rendered = session.render(RenderRequest("scene", frame, "front", actions=(Fill(),)))
        assert rendered.observation.frame == frame.position
        assert rendered.view.frame == frame.position
        assert np.all(rendered.observation.sensor_data["front"] == 200)
        assert np.any(rendered.view.data != 200)
        assert rendered.view.native == {}
        assert rendered.object_visibility == {}
        plain = session.render(RenderRequest("scene", frame, "front"))
        np.testing.assert_array_equal(plain.view.data, rendered.view.data)
        asset, world_from_object = make_object_in_front(plain.view)
        inserted = session.render(RenderRequest("scene", frame, "front", actions=(
            InsertObject("sign", asset, world_from_object),
        )))
        assert np.any(inserted.view.data != plain.view.data)
        assert inserted.view.depth is plain.view.depth is None
        visibility = inserted.object_visibility["sign"]["front"]
        assert isinstance(visibility, ObjectVisibility)
        assert visibility.visible_pixels > 0
        np.testing.assert_array_equal(inserted.observation.sensor_data["front"], inserted.view.data)
        unmodified = session.render(RenderRequest("scene", frame, "front"))
        np.testing.assert_array_equal(unmodified.view.data, plain.view.data)
        assert unmodified.object_visibility == {}
        assert inserted.object_visibility["sign"]["front"] == visibility


def test_rollout():
    data = MemoryDataset()
    frames = data.frames("scene")
    segment = AttackSegment("scene", frames[1], frames[3])
    request = CandidateRequest(segment, 7, 2, {}, RecordingPolicy(("front",)),
                               attack=MinimalAttack())
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(Control(throttle=.4)))
    with ClosedLoopServices(data, MemorySimulator(), adapter) as services, services.session(7) as session:
        first, second = session.rollout(request), session.rollout(request)
        for log in (first, second):
            assert log.status == "completed"
            assert log.steps[0].physical_pre.frame == 1
            assert log.steps[-1].physical_post.frame == 3
            assert [record.outputs[0].value["decision"] for record in log.steps] == [0, 1]
        np.testing.assert_allclose(first.steps[-1].physical_post.ego.pose,
                                   second.steps[-1].physical_post.ego.pose)
        with pytest.raises(ValueError, match="seed must match"):
            session.rollout(replace(request, seed=8))

    history_segment = AttackSegment("scene", frames[2], frames[4])
    history_request = CandidateRequest(history_segment, 7, 2, {}, RecordingPolicy(("front",)))
    history_adapter = DrivingAdapter(lambda seed: HistoryAVStack(), past_steps=2)
    with ClosedLoopServices(data, MemorySimulator(), history_adapter) as services:
        with services.session(7) as session:
            history_log = session.rollout(history_request)
    assert len(history_log.steps[0].components["past_observations"]) == 2
    initialization = next(event for event in history_log.events if event.kind == "initialization")
    assert initialization.data["past_steps"] == 2


def test_can_start_at():
    adapter = DrivingAdapter(lambda seed: HistoryAVStack(), past_steps=2)
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(capture_ticks=1), adapter,
                            deployment=DeploymentPlan(2)) as services:
        with services.session(7) as session:
            assert not session.can_start_at(0)
            assert not session.can_start_at(1)
            assert not session.can_start_at(2)
            assert session.can_start_at(3)
            with pytest.raises(ValueError, match="position must be a nonnegative integer"):
                session.can_start_at(-1)
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(capture_ticks=1),
                            deployment=DeploymentPlan(2)) as services, services.session(0) as session:
        assert session.can_start_at(0)
        assert not session.can_start_at(1)
        assert session.can_start_at(4)


def test_session():
    data = MemoryDataset()
    simulator = MemorySimulator(world_key=("memory", "services-world"))
    frame = data.frames("scene")[0]
    request = SensorRequest("scene", frame, ("front",))
    with ClosedLoopServices(data, simulator) as services:
        unused = services.session(1)
        # A session that has not acquired a world does not reserve it.
        with services.session(2) as active:
            assert active.read_sensors(request).frame == frame
            with pytest.raises(RuntimeError, match="already in use"):
                unused.read_sensors(request)
        assert unused.read_sensors(request).frame == frame
        unused.close()
        with pytest.raises(RuntimeError, match="closed"):
            unused.read_sensors(request)
    with pytest.raises(RuntimeError, match="closed"):
        services.session(3)


@pytest.fixture
def component_services():
    component = ComponentDescriptor("driver", {"observation": "Observation"},
                                    {"control": "Control"}, {"control": "Control"})
    scorer = ScorerDescriptor("mean", {"image": "numpy"}, {"score": "float"}, {"score": "float"})
    adapter = DrivingAdapter(
        lambda seed: ConstantAVStack(Control(throttle=.4)),
        components=((component, lambda avstack, observation: avstack(observation)),),
        scorers=((scorer, lambda avstack, image, target: float(np.mean(image) - target)),),
    )
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        yield services, component, scorer


def test_evaluate_component(component_services):
    services, component, _ = component_services
    assert services.capabilities.rollout
    assert services.capabilities.insert_objects
    with services.session(0) as session:
        assert session.components() == (component,)
        result = session.evaluate("driver", Observation(0., 0))
        assert result.value == Control(throttle=.4)
        with pytest.raises(ValueError, match="Unavailable component"):
            session.evaluate("unknown", Observation(0., 0))


def test_score(component_services):
    services, _, scorer = component_services
    with services.session(0) as session:
        assert session.scorers() == (scorer,)
        assert session.score("mean", np.full((2, 2), 5.), target=2.).value == 3.
        with pytest.raises(ValueError, match="does not support gradients"):
            session.score("mean", np.ones((2, 2)), target=0., require_grad=True)


def test_decision_time_step():
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), decision_time_step=.1)
    with pytest.raises(ValueError, match="AVStack decision_time_step"):
        ClosedLoopServices(MemoryDataset(), MemorySimulator(time_step=.2), adapter)


def test_record_reason():
    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        scenario = services.scenario("scene", seed=0)
        scenario.record_reason("candidate")
        scenario.record_reason("candidate")
        for index in range(102):
            scenario.record_reason(f"reason {index}")
        assert len(scenario.reasons) == 101
        assert scenario.reasons[0] == "candidate (count=2)"
        assert scenario.reasons[-1] == "Other selection reasons (count=3)"
        with pytest.raises(ValueError, match="nonempty"):
            scenario.record_reason("")


def test_candidate_request():
    dataset = MemoryDataset()
    frames = dataset.frames("scene")
    segment = AttackSegment("scene", frames[0], frames[2])
    request = CandidateRequest(segment, 0, 2, {}, RecordingPolicy())
    assert request.environment == {"world": {}, "sensors": []}
    with pytest.raises(ValueError, match="must equal its segment interval"):
        CandidateRequest(segment, 0, 1, {}, RecordingPolicy())
    with pytest.raises(ValueError, match="artifacts require an attack"):
        CandidateRequest(segment, 0, 2, {}, RecordingPolicy(), artifacts={"texture": 1})
