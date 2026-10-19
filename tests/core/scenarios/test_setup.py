"""Prepared configuration and shared conditions at the scenario/runtime boundary."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import numpy as np
import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.backend import run
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.closed_loop.datatypes import Control, Observation
from avsectester.closed_loop.hooks import Hook, Plugin, Runtime
from avsectester.scenarios import (
    ConfiguredScenarioSource,
    Dataset,
    DatasetFilter,
    FilterContext,
    InitialWindow,
    ScenarioInstance,
    ScenarioRequirement,
    scenario_from_config,
)
from avsectester.scenarios.filters import Constraint, FilterResult


class CameraBackend(WorldBackend):
    def __init__(self, frame0=0, config=None):
        self.frame0 = frame0
        self.config = config
        self.closed = False

    def reset(self):
        self.frame = self.frame0
        return self.observe()

    def step(self, control):
        self.frame += 1
        return self.observe()

    def observe(self):
        return Observation(
            self.frame * 0.1, self.frame,
            {"front": np.full((8, 10, 3), 100, np.uint8)},
        )

    def close(self):
        self.closed = True


class ImageRecorder(AVStack):
    def __init__(self):
        self.images = []

    def __call__(self, observation):
        self.images.append(observation.sensor_data["front"].copy())
        return Control()


def test_pass_through_has_no_preview_and_separates_configuration():
    received = []

    class WeatherBackend(CameraBackend):
        def configure_conditions(self, world):
            self.world = world

    original = {
        "ego": {"pipeline": {"type": "MyPipeline"}},
        "weather": {"fog_density": 20},
        "runtime": {"hooks": []}, "attacks": [], "patches": [],
        "frames": 2, "stack": {"type": "Driver"},
    }

    def factory(config):
        received.append(config)
        config["ego"]["pipeline"]["type"] = "backend copy"
        return WeatherBackend(config=config)

    source = ConfiguredScenarioSource(original, backend_factory=factory, provenance={"file": "scene.yaml"})
    cases = list(source.scenarios())
    assert len(cases) == 1 and not received
    case = cases[0]
    assert case.match is None and case.selection == {}
    assert set(case.config) == {"ego"}
    assert case.provenance["file"] == "scene.yaml"
    assert case.provenance["conditions"]["world"]["weather"] == {"fog_density": 20}
    first, second = case.make_backend(), case.make_backend()
    assert first is not second
    assert first.world == second.world == {"weather": {"fog_density": 20}}
    assert case.config["ego"]["pipeline"]["type"] == "MyPipeline"
    assert original["ego"]["pipeline"]["type"] == "MyPipeline"
    assert list(source.scenarios(limit=0)) == []
    with pytest.raises(ValueError, match="does not evaluate filters"):
        list(source.scenarios(ScenarioRequirement("empty")))


def test_unsupported_world_conditions_close_backend_and_replacements_do_not_layer_settings():
    backend = CameraBackend()
    case = ScenarioInstance(lambda: backend, conditions=ScenarioConditions(world={"weather": "WetCloudyNoon"}))
    with pytest.raises(ValueError, match="conditions"):
        case.make_backend()
    assert backend.closed

    applied = []

    class WeatherBackend(CameraBackend):
        def configure_conditions(self, world):
            applied.append(world)

    factory = lambda: WeatherBackend()
    original = ScenarioInstance(factory, conditions=ScenarioConditions(world={"weather": "ClearNoon"}))
    assert original.factory is factory
    original.make_backend()
    original.make_backend()
    assert applied == [{"weather": "ClearNoon"}] * 2
    updated = replace(original, conditions=ScenarioConditions(world={"weather": "WetCloudyNoon"}))
    updated.make_backend()
    replacement_backend = WeatherBackend()
    replacement_factory = lambda: replacement_backend
    changed_factory = replace(updated, factory=replacement_factory)
    assert changed_factory.factory is replacement_factory
    assert changed_factory.make_backend() is replacement_backend
    assert applied == [{"weather": "ClearNoon"}] * 2 + [{"weather": "WetCloudyNoon"}] * 2
    assert original.provenance["conditions"]["world"] == {"weather": "ClearNoon"}
    assert changed_factory.provenance["conditions"]["world"] == {"weather": "WetCloudyNoon"}
    for name in ("factory", "make_backend"):
        with pytest.raises(FrozenInstanceError):
            setattr(original, name, replacement_factory)


def test_condition_setup_failure_keeps_the_primary_error_and_reports_cleanup_failure():
    backend = CameraBackend()
    backend.close = Mock(side_effect=RuntimeError("cleanup failed"))
    instance = ScenarioInstance(lambda: backend, conditions=ScenarioConditions(world={"weather": "rain"}))
    with pytest.raises(ValueError, match="does not support world conditions") as caught:
        instance.make_backend()
    backend.close.assert_called_once()
    assert any("cleanup failed" in note for note in caught.value.__notes__)


def test_nurec_configuration_supports_actuator_and_trajectory_without_model_imports():
    from avsectester.closed_loop.simulators.nurec import KinematicBicycle, TrajectoryFollower

    for dynamics in (KinematicBicycle, TrajectoryFollower):
        case = scenario_from_config({
            "dt": 0.1,
            "backend": {
                "type": "NuRecSimulator", "renderer": {"type": "StubRenderer", "width": 10, "height": 8},
                "dynamics": {"type": dynamics.__name__}, "ego0": {"speed": 2},
            },
            "stack": {"type": "AlpamayoAVStack"},
        })
        backend = case.make_backend()
        try:
            assert isinstance(backend.dynamics, dynamics)
            observation = backend.reset()
            assert observation.sensor_data["camera_front"].shape == (8, 10, 3)
            assert observation.ego_speed == 2
        finally:
            backend.close()


def test_sensor_conditions_use_relative_steps_and_independent_randomness():
    conditions = ScenarioConditions.from_config({
        "seed": 17,
        "sensors": [{"type": "GaussianNoise", "severity": 0.2, "camera": "front"}],
    })

    def random_consumer(observation, context):
        context.rng.random(1000)
        return observation

    clean, attacked = ImageRecorder(), ImageRecorder()
    left = conditions.apply_runtime(Runtime(seed=1))
    right = conditions.apply_runtime(Runtime([Hook("observation", random_consumer)], seed=9))
    assert left.hooks[0].handler is not right.hooks[-1].handler
    run(CameraBackend(frame0=10), clean, 3, runtime=left)
    run(CameraBackend(frame0=800), attacked, 3, runtime=right)
    assert all(np.array_equal(a, b) for a, b in zip(clean.images, attacked.images))
    assert not np.array_equal(clean.images[0], clean.images[1])
    assert conditions.provenance()["sensors"][0]["seed"] == 17


def test_condition_order_is_explicit_and_custom_handlers_are_fresh():
    constructed = []

    class Offset(Plugin):
        def __init__(self):
            self.calls = 0
            constructed.append(self)

        def transform(self, observation, context):
            self.calls += 1
            observation.sensor_data["front"] += 10
            return observation

    def double(observation, context):
        observation.sensor_data["front"] *= 2
        return observation

    for order, expected in [("before", 220), ("after", 210)]:
        condition = ScenarioConditions(sensors=(SensorCondition(Offset, order=order),))
        for _ in range(2):
            stack = ImageRecorder()
            run(CameraBackend(), stack, 2, runtime=condition.apply_runtime(Runtime([Hook("observation", double)])))
            assert np.all(stack.images[0] == expected)
    assert len(constructed) == 4 and all(item.calls == 2 for item in constructed)


def test_condition_function_adapter_keeps_lifecycle_owner_and_condition_randomness():
    from avsectester.interventions import ObservationTransform

    owners = []

    def factory():
        state = {"reset": 0, "calls": 0, "feedback": [], "closed": False}
        owners.append(state)

        def transform(observation):
            state["calls"] += 1
            return observation

        def reset(context):
            state["reset"] += 1
            state["reset_random"] = int(context.rng.integers(100000))

        def after_step(context):
            state["feedback"].append(int(context.rng.integers(100000)))

        def close():
            state["closed"] = True

        transform.reset, transform.after_step, transform.close = reset, after_step, close
        return ObservationTransform(transform)

    def consume_attack_randomness(observation, context):
        context.rng.random(1000)
        return observation

    conditions = ScenarioConditions(seed=31, sensors=(SensorCondition(factory),))
    for seed in (1, 92):
        runtime = Runtime([Hook("observation", consume_attack_randomness)], seed=seed)
        run(CameraBackend(), ImageRecorder(), 2, runtime=conditions.apply_runtime(runtime))
    assert len(owners) == 2
    assert all(owner["reset"] == 1 and owner["calls"] == 2 and owner["closed"] for owner in owners)
    assert owners[0]["reset_random"] == owners[1]["reset_random"] == int(
        np.random.default_rng(31).integers(100000),
    )
    assert owners[0]["feedback"] == owners[1]["feedback"] == [
        int(np.random.default_rng((31, step, 1)).integers(100000)) for step in range(2)
    ]


def test_configured_filter_sees_weather_before_selection_and_backend_factory_gets_scene(scene):
    weather = {"fog_density": 35}
    seen, backends = [], []

    @contextmanager
    def candidate(config):
        seen.append(config)
        yield FilterContext(scene(), metadata={"replay_scenario": config})

    class WeatherBackend(CameraBackend):
        def configure_conditions(self, world):
            self.world = world

    def factory(config):
        backends.append(config)
        return WeatherBackend(config=config)

    case = scenario_from_config(
        {"ego": {}, "weather": weather, "runtime": {}, "selection": {"requirement": {"name": "empty"}}},
        candidate_provider=candidate, backend_factory=factory,
    )
    assert seen[0]["weather"] == weather
    assert "runtime" not in seen[0]
    assert not backends
    assert case.match is not None
    backend = case.make_backend()
    assert backend.world == {"weather": weather}
    assert set(backends[0]) == {"ego"}


def test_selection_conditions_preserve_labels_and_state_with_current_provider_data(scene):
    created = []

    class Counter(Plugin):
        def __init__(self):
            self.count = 0
            self.frames = []
            self.closed = False
            created.append(self)

        def transform(self, observation, context):
            self.count += 1
            self.frames.append(context.native["simulator"]["provider_frame"])
            assert context.ground_truth.frame == observation.frame
            observation.ego_speed = self.count
            return observation

        def close(self):
            self.closed = True

    class Frames(Dataset):
        def scenes(self):
            yield scene()

        def initial_sequence(self, origin, frames):
            return tuple(scene(frame=i) for i in range(frames))

        def make_backend(self, selected):
            return CameraBackend()

        def context(self, selected):
            return FilterContext(selected, native={
                "provider_frame": selected.frame,
                "observation": Observation(selected.t, selected.frame, ego_speed=0),
            })

    class ConditionedSpeed(Constraint):
        def evaluate(self, context):
            measurement = ScenarioConditions.selection_observation(context)
            assert context.scene.ego.speed == 0
            assert context.native["observation"].ego_speed == 0
            return FilterResult.from_bool(measurement.ego_speed == context.frame_index + 1, "counter")

    conditions = ScenarioConditions(sensors=(SensorCondition(Counter),))
    source = DatasetFilter(Frames(), conditions=conditions)
    cases = list(source.scenarios(ScenarioRequirement("window", constraints=[ConditionedSpeed()], window=InitialWindow(3))))
    assert len(cases) == 1
    assert len(created) == 1 and created[0].frames == [0, 1, 2] and created[0].closed
    with pytest.raises(ValueError, match="supplies no observation"):
        ScenarioConditions.selection_observation(FilterContext(scene()))


def test_selection_conditions_use_native_capture_copier_and_preserve_raw_measurements(scene):
    from avsectester.perturbations.image import AugmentationPipeline, Corruption
    from avsectester.closed_loop.simulators.carla import CarlaSimulator

    class NativeCapture:
        # Match the actual adapter's borrowed CARLA capture detection.
        __module__ = "carla.libcarla"
        raw_data = np.full((8, 10, 3), 100, np.uint8).tobytes()

        def __deepcopy__(self, memo):
            raise TypeError("Native CARLA captures cannot be deep-copied")

    class ImageWrapper:
        def __init__(self, capture, calibration):
            self.data, self.calibration = capture, calibration

        @property
        def rgb_image(self):
            if isinstance(self.data, np.ndarray):
                return self.data
            return np.frombuffer(self.data.raw_data, np.uint8).reshape(8, 10, 3)

    class Lift(Corruption):
        def apply(self, rgb, rng):
            return rgb + 10

    backend = CarlaSimulator({})
    backend.copy_observation = Mock(wraps=backend.copy_observation)
    current = scene()
    calibration = {"focal": 10.0}
    capture = NativeCapture()
    wrapper = ImageWrapper(capture, calibration)
    raw = Observation(current.t, current.frame, {"front": wrapper}, {"front": calibration})
    context = FilterContext(current, backend=backend, native={"observation": raw})

    with ScenarioConditions().selection_scope() as prepare:
        unchanged = prepare(context)
        assert prepare.session is None
        assert ScenarioConditions.selection_observation(unchanged) is raw
    backend.copy_observation.assert_not_called()

    conditions = ScenarioConditions(sensors=(SensorCondition.image(
        AugmentationPipeline([Lift()]), camera="front",
    ),))
    conditioned = conditions.selection_context(context)
    backend.copy_observation.assert_called_once_with(raw)
    output = ScenarioConditions.selection_observation(conditioned)
    assert np.all(output.sensor_data["front"].rgb_image == 110)
    assert np.all(raw.sensor_data["front"].rgb_image == 100)
    assert raw.sensor_data["front"].data is capture
    assert output.calibration["front"] is output.sensor_data["front"].calibration
    assert output.calibration["front"] is not calibration
    output.calibration["front"]["focal"] = 20.0
    assert calibration["focal"] == 10.0


@pytest.mark.parametrize("config", [
    {"seed": -1}, {"unknown": True}, {"sensors": [{"type": "GaussianNoise", "severity": 2}]},
    {"sensors": [{"type": "GaussianNoise", "stage": "perception.post"}]},
    {"sensors": [{"factory": "missing.factory:handler"}]},
])
def test_invalid_condition_configuration_fails_during_preparation(config):
    with pytest.raises(ValueError):
        ScenarioConditions.from_config(config)
