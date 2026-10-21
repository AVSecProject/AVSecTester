"""Selection and preparation use isolated public service sessions."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.attack.runtime import SensorAttackAction
from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.closed_loop.frame_data import FrameDataCache
from avsectester.closed_loop.runner import RecordingPolicy
from avsectester.closed_loop.services import (
    CandidateRequest, ClosedLoopServices, ComponentDescriptor, DrivingAdapter, RenderRequest,
    ScorerDescriptor, SensorRequest,
)
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets import Unavailable
from tests.fakes import CarlaNativeBoundary, ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def test_frame_data(carla_dataset, monkeypatch):
    boundary = CarlaNativeBoundary().install(monkeypatch)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    cache = FrameDataCache()
    frame = carla_dataset.frames("scene")[2]
    assert isinstance(carla_dataset.frame_data("scene", frame), Unavailable)
    with ClosedLoopServices(carla_dataset, simulator, frame_data_cache=cache) as services:
        with services.session(7) as session:
            generated = session.frame_data("scene", frame)
        selected = services.scenario("scene", seed=7).frame_data(frame)
    assert selected.frame == generated.frame == frame
    expected = generated.ego.pose.copy()
    np.testing.assert_allclose(selected.ego.pose, expected)
    generated.ego.pose[:] = 0
    np.testing.assert_allclose(selected.ego.pose, expected)
    # Another service sharing the cache reuses the generated data without a CARLA client.
    monkeypatch.setattr(boundary, "build",
                        lambda *args, **kwargs: pytest.fail("frame data was generated again"))
    with ClosedLoopServices(carla_dataset, simulator, frame_data_cache=cache) as services:
        reused = services.scenario("scene", seed=7).frame_data(frame)
    np.testing.assert_allclose(reused.ego.pose, expected)


def test_sensor_data():
    data, simulator = MemoryDataset(), MemorySimulator()
    frames = data.frames("scene")
    environment = {"sensors": [{"type": "GaussianNoise", "camera": "front",
                                "options": {"severity": .2}}]}
    with ClosedLoopServices(data, simulator) as services:
        with services.session(7) as session:
            samples = [session.sensor_data(SensorRequest("scene", frame, ("front",), environment))
                       for frame in frames[1:4]]
            assert [sample.observation.frame for sample in samples] == [1, 2, 3]
        for sample in samples:
            with services.session(7) as independent:
                replayed = independent.sensor_data(SensorRequest(
                    "scene", sample.frame, ("front",), environment))
            np.testing.assert_array_equal(sample.observation.sensor_data["front"],
                                          replayed.observation.sensor_data["front"])
            assert sample.provenance["seed"] == 7
        with services.session(7) as session:
            with pytest.raises(ValueError, match="Unavailable sensors"):
                session.sensor_data(SensorRequest("scene", frames[0], ("missing",)))


def test_render():
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
        plain = session.render(RenderRequest("scene", frame, "front"))
        np.testing.assert_array_equal(plain.view.data, rendered.view.data)
        with pytest.raises(ValueError, match="Unknown memory sensor: missing"):
            session.render(RenderRequest("scene", frame, "missing"))


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


def test_session():
    data = MemoryDataset()
    simulator = MemorySimulator(resource_key=("memory", "services-world"))
    frame = data.frames("scene")[0]
    request = SensorRequest("scene", frame, ("front",))
    with ClosedLoopServices(data, simulator) as services:
        unused = services.session(1)
        # A session that has not acquired a world does not reserve it.
        with services.session(2) as active:
            assert active.sensor_data(request).frame == frame
            with pytest.raises(RuntimeError, match="already in use"):
                unused.sensor_data(request)
        assert unused.sensor_data(request).frame == frame
        unused.close()
        with pytest.raises(RuntimeError, match="closed"):
            unused.sensor_data(request)
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


def test_evaluate(component_services):
    services, component, _ = component_services
    assert services.capabilities.rollout
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
