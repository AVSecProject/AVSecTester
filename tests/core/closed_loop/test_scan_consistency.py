"""Forward scans preserve the selected source state, including deployment/capture ticks."""

from copy import deepcopy

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Observation, WorldSnapshot
from avsectester.closed_loop.services import ClosedLoopServices, SensorRequest
from avsectester.closed_loop.simulator import DeploymentPlan, Simulator, SimulatorRuntime
from avsectester.datasets.base import (
    ActorState, Dataset, FrameData, FrameReference, ScenarioDescriptor, ScenarioStart, Unavailable,
)


class RecipeDataset(Dataset):
    """A recipe whose frame data must be materialized outside the dataset."""

    def __init__(self):
        self.refs = tuple(FrameReference("road", i, f"recipe-{i}", i * 0.1) for i in range(14))

    def scenarios(self):
        return iter((ScenarioDescriptor("road", "CarlaDataset"),))

    def frames(self, scenario_id):
        assert scenario_id == "road"
        return self.refs

    def frame_data(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return Unavailable("recipe_only")

    def native(self, scenario_id):
        assert scenario_id == "road"
        return {"time_step": 0.1, "recipe": "prescribed ego and traffic motion"}

    def scenario_start(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return ScenarioStart("CarlaDataset", scenario_id, frame, self.native(scenario_id))


class RecipeSimulator(Simulator):
    """CARLA-style captures are produced by a tick, rather than by observe()."""

    dataset_kinds = ("CarlaDataset",)
    capture_ticks = 1

    def __init__(self):
        self.runtimes = []

    @property
    def decision_time_step(self):
        return 0.1

    def create_runtime(self, seed=None):
        runtime = RecipeRuntime(self, seed)
        self.runtimes.append(runtime)
        return runtime

    @property
    def scans(self):
        # Materialization resets without Initialization; sensor scans reset with its evidence.
        return [runtime for runtime in self.runtimes if runtime.selected_reset]


class RecipeRuntime(SimulatorRuntime):
    def __init__(self, simulator, seed):
        super().__init__(simulator)
        self.seed = seed
        self.selected_reset = False
        self.closed = False
        self.events = []
        self.capture_count = 0

    def _initialize_world(self, start, initialization, environment):
        assert environment == {}  # SimulatorRuntime receives world settings only.
        self.selected_reset = initialization is not None
        self.position = 0
        self.native_ticks = 0
        self.ego_released = False
        self.sensor_capture = None
        self.events.append(("initialize", self.position))
        if initialization is not None:
            assert initialization.expected.frame == start.frame
            assert initialization.resolved_scenario == {"seed": self.seed}

    def _replay_position(self, start):
        return start.frame.position if start is not None else 0

    def _tick(self, kind, *, move):
        self.native_ticks += 1
        self.position += int(move)
        self.events.append((kind, self.position))
        # A sensor reports the state rendered at the last tick, not the live state at read time.
        self.sensor_capture = Observation(
            self.position * 0.1, self.position,
            {"front": np.array([[self.seed + self.position ** 2,
                                 100 - 2 * self.position, self.position]], dtype=float)},
            calibration={"front": {"focal_length": 90.0}},
            ego_speed=2 * self.position + 1.0,
        )

    def _replay_tick(self):
        assert not self.ego_released, "A scan must retain prescribed ego motion"
        self._tick("replay", move=True)

    def _hold_tick(self):
        self._tick("hold", move=False)

    def _before_selected_capture(self, position):
        if not self._generating_frame_data:
            self.ego_released = True

    def _check_initialization(self, initialization):
        if initialization is not None:
            assert initialization.expected.frame.position == self.position

    def _observe(self):
        assert self.sensor_capture is not None, "A capture requires a native sensor tick"
        self.capture_count += 1
        self.events.append(("capture", self.position))
        return deepcopy(self.sensor_capture)

    def _state(self):
        return WorldSnapshot(self.position * 0.1, self.position,
                             ego_speed=2 * self.position + 1.0)

    def _advance_native(self, control):
        raise AssertionError("A sensor scan must not run a driving step")

    def _capture_frame_data(self, frame):
        assert frame.position == self.position
        pose = np.eye(4)
        pose[0, 3] = self.seed + self.position ** 2
        return FrameData(frame, None, {"ego": ActorState(pose)})

    def _replayed_scenario(self):
        return {"seed": self.seed}

    def close(self):
        self.closed = True


def assert_selected_boundary(runtime, position, deployment_ticks):
    """The capture tick belongs to the window; it must not move the selected boundary."""
    first_capture = runtime.events.index(("capture", position))
    initialization = runtime.events[:first_capture]
    window = deployment_ticks + runtime.capture_ticks
    assert sum(kind == "hold" for kind, _ in initialization) == (window if position == 0 else 0)
    assert sum(kind == "replay" for kind, _ in initialization) == position
    assert not runtime.ego_released


@pytest.mark.parametrize("deployment_ticks", (0, 2))
@pytest.mark.parametrize("at_origin", (True, False))
def test_forward_scan_matches_independent_selected_origin_resets(deployment_ticks, at_origin):
    data, simulator = RecipeDataset(), RecipeSimulator()
    window = deployment_ticks + simulator.capture_ticks
    first = 0 if at_origin else window + 2
    positions = tuple(range(first, first + window + 3))
    with ClosedLoopServices(data, simulator, deployment=DeploymentPlan(deployment_ticks)) as services:
        with services.session(7) as session:
            scanned = {
                position: session.sensor_data(SensorRequest("road", data.refs[position], ("front",)))
                for position in positions
            }
            assert len(simulator.scans) == 1
            runtime = simulator.scans[0]
            assert_selected_boundary(runtime, first, deployment_ticks)
            assert runtime.capture_count == len(positions)
            assert runtime.position == positions[-1]
            assert runtime.native_ticks == (window if first == 0 else first) + len(positions) - 1
            assert not runtime.closed
        assert runtime.closed

        for position, sample in scanned.items():
            assert sample.frame == data.refs[position]
            assert sample.observation.t == pytest.approx(data.refs[position].timestamp)
            assert sample.observation.ego_speed == 2 * position + 1.0
            np.testing.assert_array_equal(
                sample.observation.sensor_data["front"], [[7 + position ** 2, 100 - 2 * position, position]],
            )
            # Boundaries inside a physical initialization window cannot be independently reset.
            if 0 < position < window:
                continue
            with services.session(7) as fresh:
                reset = fresh.sensor_data(SensorRequest("road", data.refs[position], ("front",)))
                assert_selected_boundary(simulator.scans[-1], position, deployment_ticks)
                assert simulator.scans[-1].capture_count == 1
            assert reset.frame == sample.frame
            assert reset.observation.t == sample.observation.t
            assert reset.observation.frame == sample.observation.frame
            assert reset.observation.ego_speed == sample.observation.ego_speed
            assert reset.observation.calibration == sample.observation.calibration
            np.testing.assert_array_equal(reset.observation.sensor_data["front"],
                                          sample.observation.sensor_data["front"])
    assert all(runtime.closed for runtime in simulator.runtimes)


def test_repeated_frame_does_not_tick_and_backward_read_rebuilds_the_scan():
    data, simulator = RecipeDataset(), RecipeSimulator()
    with ClosedLoopServices(data, simulator, deployment=DeploymentPlan(2)) as services:
        with services.session(7) as session:
            session.sensor_data(SensorRequest("road", data.refs[3], ("front",)))
            forward = session.sensor_data(SensorRequest("road", data.refs[5], ("front",)))
            runtime = simulator.scans[-1]
            ticks, captures = runtime.native_ticks, runtime.capture_count
            repeated = session.sensor_data(SensorRequest("road", data.refs[5], ("front",)))
            assert len(simulator.scans) == 1
            assert (runtime.native_ticks, runtime.capture_count) == (ticks, captures)
            np.testing.assert_array_equal(forward.observation.sensor_data["front"],
                                          repeated.observation.sensor_data["front"])

            backward = session.sensor_data(SensorRequest("road", data.refs[3], ("front",)))
            assert runtime.closed
            assert len(simulator.scans) == 2
            rebuilt = simulator.scans[-1]
            assert_selected_boundary(rebuilt, 3, 2)
            assert rebuilt.capture_count == 1
            np.testing.assert_array_equal(backward.observation.sensor_data["front"], [[16, 94, 3]])
            session.sensor_data(SensorRequest("road", data.refs[4], ("front",)))
            assert rebuilt.position == 4 and not rebuilt.ego_released
    assert all(runtime.closed for runtime in simulator.runtimes)
