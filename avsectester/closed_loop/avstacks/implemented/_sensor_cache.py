"""Driver-owned references to the most recently delivered acquisition of each sensor."""
from __future__ import annotations

from dataclasses import replace

from avsectester.data_types import DeliveredAcquisition, Observation


class SensorCache:
    def __init__(self) -> None:
        self._latest: dict[str, DeliveredAcquisition] = {}

    def clear(self) -> None:
        self._latest.clear()

    def update(self, observation: Observation) -> None:
        for sensor in observation.sensor_data:
            self._latest[sensor] = observation.delivered(sensor)[-1]
        # A due frame without payload leaves its sensor unavailable until the next delivery.
        for sensor in observation.deliveries.keys() - observation.sensor_data.keys():
            self._latest.pop(sensor, None)

    def resolve(self, observation: Observation) -> Observation:
        """Build a private adapter view without changing the framework's incremental input."""
        if self._latest.keys() <= observation.sensor_data.keys():
            return observation
        missing = {sensor: sample for sensor, sample in self._latest.items()
                   if sensor not in observation.sensor_data}
        return replace(observation,
            sensor_data={**{sensor: sample.data for sensor, sample in missing.items()},
                         **observation.sensor_data},
            calibration={**{sensor: sample.calibration for sensor, sample in missing.items()
                            if sample.calibration is not None}, **observation.calibration},
            deliveries={**{sensor: (sample,) for sensor, sample in missing.items()},
                        **observation.deliveries})
