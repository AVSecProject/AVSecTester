"""Portable attacks complete public preparation, execution and offline evaluation."""
import numpy as np
import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.simulator import SensorView, Simulator, SimulatorRuntime
from avsectester.datasets import (ActorState, Dataset, FrameData, FrameReference,
                                  ScenarioDescriptor, ScenarioStart)
from avsectester.geometry.cameras import PinholeCamera
from avsectester.geometry.poses import ActorPose

CAM_FROM_WORLD = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
CAMERA = PinholeCamera(np.array([[50, 0, 50], [0, 50, 40], [0, 0, 1.]]), 100, 80)


def host_state():
    pose = np.eye(4)
    pose[0, 3] = 10
    return ActorState(pose, (0., 0., 0.), "car", (4., 2., 2.))


class Data(Dataset):
    timeline = tuple(FrameReference("s", i, i, i * .1) for i in range(4))

    def scenarios(self):
        return iter((ScenarioDescriptor("s", "memory"),))

    def frames(self, scenario_id):
        return self.timeline

    def frame_data(self, scenario_id, frame):
        return FrameData(frame, actors={"lead": host_state()},
                         ego=ActorState(np.eye(4), (0., 0., 0.), "car", (4., 2., 2.)))

    def native(self, scenario_id):
        return {}

    def scenario_start(self, scenario_id, frame):
        return ScenarioStart("memory", scenario_id, frame, {})


class World(SimulatorRuntime):
    def _initialize_world(self, start, initialization, environment):
        self.i = start.frame.position

    def _observe(self):
        return Observation(self.i * .1, self.i,
                           {"front": np.full((80, 100, 3), 25, np.uint8)}, ego_speed=1.)

    def _state(self):
        return WorldSnapshot(self.i * .1, self.i, ego_speed=1., actors={"lead": host_state()},
                             ego=ActorState(np.eye(4), (1., 0., 0.), "car", (4., 2., 2.)))

    def _advance_native(self, command):
        self.i += 1

    def sensor_view(self, sensor):
        return SensorView(sensor, self._observe().sensor_data[sensor], CAMERA, CAM_FROM_WORLD,
                          {"lead": ActorPose(host_state().pose, (4., 2., 2.))},
                          ActorPose(np.eye(4), (4., 2., 2.)), self.i * .1, self.i)

    def close(self):
        pass


class Sim(Simulator):
    dataset_kinds = ("memory",)
    can_render = True

    def create_runtime(self, seed=None):
        return World(self)


class Driver(AVStack):
    @property
    def supported_stages(self):
        return frozenset({"perception.post"})

    def __call__(self, observation):
        self.current = self._emit("perception.post", [])
        return Control(throttle=min(observation.ego_speed / 10, 1.))

    def detections(self):
        return self.current

    def component_log(self):
        return {"perception": self.current}


@pytest.fixture
def memory_experiment():
    """An isolated in-memory scene with calibrated camera and a simple driving system."""
    from types import SimpleNamespace
    return SimpleNamespace(Data=Data, Sim=Sim, World=World, Driver=Driver,
                           host_state=host_state, CAMERA=CAMERA, CAM_FROM_WORLD=CAM_FROM_WORLD)
