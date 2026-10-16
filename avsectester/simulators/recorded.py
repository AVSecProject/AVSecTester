"""Bounded replay of recorded source states. Commands never change recorded trajectories."""

from __future__ import annotations

from copy import deepcopy

from avsectester.capture import CaptureBackend
from avsectester.datasets.base import SourceData
from avsectester.plane import Observation, WorldSnapshot


class RecordedSequenceBackend(CaptureBackend):
    source_kinds = ("nuscenes",)
    def __init__(self, source=None):
        super().__init__()
        self._configured_source = deepcopy(source)
        self._references = ()
        self._index = 0

    def _initialize_world(self, source, initialization, environment):
        if environment:
            raise ValueError("Recorded replay does not support environment overrides")
        source = source if source is not None else self._configured_source
        if not isinstance(source, SourceData) or source.kind != "nuscenes":
            raise ValueError("Recorded replay requires nuscenes SourceData")
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to recorded data")
        references = deepcopy(source.payload["references"])
        if not references or references[0].frame != source.start:
            raise ValueError("Recorded sequence does not start at the requested source boundary")
        for i, reference in enumerate(references):
            if (reference.frame.scenario_id != source.scenario_id
                    or reference.frame.position != source.start.position + i
                    or reference.scene is None):
                raise ValueError("Recorded replay requires contiguous source states")
        self._references = references
        self.initialization_report = {"kind": "recorded_replay", "source": source.scenario_id,
                                      "start_position": source.start.position}
        self._index = 0

    def _observe(self):
        import numpy as np
        from PIL import Image

        reference = self._references[self._index]
        sensors = {}
        for name, location in reference.sensors.items():
            with Image.open(location.uri) as image:
                sensors[name] = np.array(image.convert("RGB"))
        scene = reference.scene
        return Observation(reference.frame.timestamp, reference.frame.position,
                           sensor_data=sensors, calibration=deepcopy(scene.cameras),
                           vehicle_state=deepcopy(reference.actors.get("ego")),
                           ego_speed=scene.ego.speed)

    def ground_truth(self):
        if not self._references:
            raise RuntimeError("Initialize recorded replay before requesting state")
        reference = self._references[self._index]
        return WorldSnapshot(reference.frame.timestamp, reference.frame.position,
                             deepcopy(reference.actors.get("ego")), reference.scene.ego.speed)

    def _advance_native(self, control):
        if self._index + 1 >= len(self._references):
            raise ValueError("Recorded step exceeds terminal source support")
        self._index += 1

    def close(self):
        self._capture = None
        self._world_initialized = False
        self._references = ()
