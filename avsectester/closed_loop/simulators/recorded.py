"""Bounded replay of recorded frames. Commands never change recorded trajectories."""

from __future__ import annotations

from copy import deepcopy

from avsectester.closed_loop.simulator import Simulator
from avsectester.datasets.base import ScenarioStart
from avsectester.closed_loop.datatypes import Observation, WorldSnapshot


class RecordedReplaySimulator(Simulator):
    dataset_kinds = ("NuScenesDataset",)
    def __init__(self, start=None):
        super().__init__()
        self._configured_start = deepcopy(start)
        self._frames = ()
        self._index = 0

    def _initialize_world(self, start, initialization, environment):
        if environment:
            raise ValueError("Recorded replay does not support environment overrides")
        start = start if start is not None else self._configured_start
        if not isinstance(start, ScenarioStart) or start.kind != "NuScenesDataset":
            raise ValueError("Recorded replay requires a NuScenesDataset ScenarioStart")
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to recorded data")
        frames = deepcopy(start.payload["frame_data"])
        if not frames or frames[0].frame != start.frame:
            raise ValueError("Recorded frames do not begin at the requested start frame")
        for i, data in enumerate(frames):
            if (data.frame.scenario_id != start.scenario_id
                    or data.frame.position != start.frame.position + i
                    or data.scene is None):
                raise ValueError("Recorded replay requires contiguous frames")
        self._frames = frames
        self.initialization_report = {"kind": "recorded_replay", "scenario": start.scenario_id,
                                      "start_position": start.frame.position}
        self._index = 0

    def _observe(self):
        import numpy as np
        from PIL import Image

        data = self._frames[self._index]
        sensors = {}
        for name, location in data.sensor_data.items():
            with Image.open(location.uri) as image:
                sensors[name] = np.array(image.convert("RGB"))
        scene = data.scene
        return Observation(data.frame.timestamp, data.frame.position,
                           sensor_data=sensors, calibration=deepcopy(scene.cameras),
                           vehicle_state=deepcopy(data.actors.get("ego")),
                           ego_speed=scene.ego.speed)

    def ground_truth(self):
        if not self._frames:
            raise RuntimeError("Initialize recorded replay before requesting state")
        data = self._frames[self._index]
        return WorldSnapshot(data.frame.timestamp, data.frame.position,
                             deepcopy(data.actors.get("ego")), data.scene.ego.speed,
                             {k: deepcopy(v) for k, v in data.actors.items() if k != "ego"})

    def _advance_native(self, control):
        self._replay_tick()

    def _replay_tick(self):
        # Recorded motion is the prescribed motion; commands never change it.
        if self._index + 1 >= len(self._frames):
            raise ValueError("Recorded step exceeds terminal recorded support")
        self._index += 1

    def _capture_frame_data(self, frame):
        data = self._frames[self._index]
        if data.frame != frame:
            raise ValueError("Recorded replay is not at the requested frame")
        return deepcopy(data)

    def close(self):
        self._capture = None
        self._world_initialized = False
        self._frames = ()
