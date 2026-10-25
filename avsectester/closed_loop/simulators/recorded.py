"""Bounded open-loop replay of recorded frames. Commands never change recorded trajectories.

Recorded datasets without a reconstruction (nuScenes) use it so their sensor data reaches the
closed loop through a simulator, like every other dataset. It is not a closed-loop simulation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView, Simulator, SimulatorRuntime
from avsectester.datasets.base import ScenarioStart
from avsectester.data_types import Observation, WorldSnapshot
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.validation import require_nonnegative_number


class RecordedReplaySimulator(Simulator):
    """Open-loop replay with native camera names or explicitly configured output aliases.

    ``camera_aliases`` maps dataset-native names to observation and view names. The
    configured ``start`` is used by resets without an explicit scenario start.
    """

    dataset_kinds = ("NuScenesDataset",)
    can_render = True
    can_replay = True
    can_insert_objects = True

    def __init__(self, start: ScenarioStart | None = None, *,
                 camera_aliases: Mapping[str, str] = {}, occlusion_tolerance_m: float = 0.1) -> None:
        require_nonnegative_number(occlusion_tolerance_m, "occlusion_tolerance_m")
        self.start = deepcopy(start)
        self.occlusion_tolerance_m = occlusion_tolerance_m
        if not isinstance(camera_aliases, Mapping) or any(
            not isinstance(native, str) or not native or not isinstance(alias, str) or not alias
            for native, alias in camera_aliases.items()
        ):
            raise ValueError("camera_aliases must map nonempty native camera names to nonempty aliases")
        if len(set(camera_aliases.values())) != len(camera_aliases):
            raise ValueError("camera_aliases must use unique output names")
        self.camera_aliases = dict(camera_aliases)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {**super().identity, "camera_aliases": dict(self.camera_aliases)}

    def create_runtime(self, seed: int | None = None) -> RecordedReplaySimulatorRuntime:
        """A new replay; recorded frames are fixed, so ``seed`` is unused."""
        return RecordedReplaySimulatorRuntime(self)


class RecordedReplaySimulatorRuntime(SimulatorRuntime):
    """One replay of recorded frames."""

    def __init__(self, simulator: RecordedReplaySimulator) -> None:
        super().__init__(simulator)
        self._configured_start = deepcopy(simulator.start)
        self.camera_aliases = dict(simulator.camera_aliases)
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
                    or data.frame.position != start.frame.position + i):
                raise ValueError("Recorded replay requires contiguous frames")
            if data.ego is None:
                raise ValueError("Recorded replay requires an ego state at every frame")
        for data in frames:
            outputs = [self.camera_aliases.get(name, name) for name in data.sensor_references]
            if len(set(outputs)) != len(outputs):
                raise ValueError("camera_aliases produce duplicate recorded sensor names")
        self._frames = tuple(
            replace(data, actors={name: actor for name, actor in data.actors.items() if name != "ego"})
            for data in frames
        )
        self.initialization_report = {"kind": "recorded_replay", "scenario": start.scenario_id,
                                      "start_position": start.frame.position}
        self._index = 0

    def _observe(self):
        from PIL import Image

        data = self._frames[self._index]
        sensors = {}
        for name, location in data.sensor_references.items():
            with Image.open(location.uri) as image:
                sensors[self.camera_aliases.get(name, name)] = np.array(image.convert("RGB"))

        calibration = {self.camera_aliases.get(name, name): deepcopy(dict(location.metadata))
                       for name, location in data.sensor_references.items()}
        speed = float(np.linalg.norm(data.ego.velocity)) if data.ego.velocity is not None else 0.0
        return Observation(data.frame.timestamp, data.frame.position,
                           sensor_data=sensors, calibration=calibration,
                           vehicle_state=deepcopy(data.ego), ego_speed=speed)

    def _state(self):
        if not self._frames:
            raise RuntimeError("Initialize recorded replay before requesting state")
        data = self._frames[self._index]

        speed = float(np.linalg.norm(data.ego.velocity)) if data.ego.velocity is not None else 0.0
        return WorldSnapshot(data.frame.timestamp, data.frame.position, deepcopy(data.ego), speed,
                             deepcopy(data.actors),
                             ego=deepcopy(data.ego))

    @property
    def sensors(self) -> frozenset[str]:
        return self.camera_sensors

    @property
    def camera_sensors(self) -> frozenset[str]:
        if not self._world_initialized:
            return frozenset()
        return frozenset(self.camera_aliases.get(name, name)
                         for name, reference in self._frames[self._index].sensor_references.items()
                         if "camera" in reference.metadata)

    def sensor_geometry(self, sensor: str) -> SensorGeometry:
        if not self._world_initialized:
            raise RuntimeError("Initialize recorded replay before requesting camera geometry")
        data = self._frames[self._index]
        native = next((name for name in data.sensor_references
                       if self.camera_aliases.get(name, name) == sensor), None)
        if native is None and sensor in data.sensor_references:
            native = sensor
        if native is None:
            raise ValueError(f"Unknown recorded sensor {sensor!r}")
        metadata = data.sensor_references[native].metadata
        if "camera" not in metadata or "ego_from_sensor" not in metadata:
            raise ValueError(f"Recorded sensor {sensor!r} has no camera calibration")
        return SensorGeometry(self.camera_aliases.get(native, native), deepcopy(metadata["camera"]),
                              np.linalg.inv(np.asarray(data.ego.pose) @ metadata["ego_from_sensor"]))

    def _sensor_view(self, sensor: str) -> SensorView:
        """Calibrated recorded RGB frame and cuboids in the recorded world."""
        from avsectester.data_types import read_camera_rgb

        geometry = self.sensor_geometry(sensor)
        data = self._frames[self._index]
        sensor = geometry.sensor
        if self._capture is None or sensor not in self._capture.sensor_data:
            raise ValueError(f"Recorded sensor {sensor!r} has no RGB capture")
        captured = read_camera_rgb(self._capture, sensor)
        if captured is None:
            raise ValueError(f"Recorded sensor {sensor!r} has no RGB capture")
        actors = data.actors
        return SensorView(sensor, captured[1], geometry.camera, geometry.cam_from_world,
                          actors, data.ego, data.frame.timestamp, data.frame.position,
                          sensor_from_world=BODY_FROM_OPTICAL @ geometry.cam_from_world,
                          occlusion_tolerance_m=self.simulator.occlusion_tolerance_m)

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
        self._frames = ()
        super().close()
