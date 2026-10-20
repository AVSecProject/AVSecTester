"""Actual native sensor objects, with construction and payload access supplied by each case."""

from copy import deepcopy
from operator import itemgetter

import numpy as np
import pytest


def whole_payload(data):
    return data


SENSOR_CASES = [
    pytest.param({
        "sensor": "ImageData", "calibration": "CameraCalibration",
        "data": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "calibration_args": (np.eye(3, 4), (2, 3, 3)), "calibration_options": {"channel_order": "rgb"},
    }, id="image"),
    pytest.param({
        "sensor": "LidarData", "calibration": "LidarCalibration",
        "data": np.arange(12, dtype=float).reshape(3, 4),
        "sensor_options": {"flipy": True}, "attributes": {"flipy": True},
    }, id="lidar"),
    pytest.param({
        "sensor": "GpsData", "calibration": "GpsCalibration", "data": np.array([1., 2., 3.]),
        "sensor_options": {"levar": np.zeros(3)},
    }, id="gps"),
    pytest.param({
        "sensor": "ImuData", "calibration": "ImuCalibration", "data": {"dt": .05, "dv": np.ones(3)},
        "payload_array": itemgetter("dv"),
    }, id="imu"),
]


@pytest.fixture(params=SENSOR_CASES)
def native_sensor_case(request):
    """``(make_sensor, payload_array, attributes)`` for one native sensor type.

    ``make_sensor(reference)`` builds the sensor mounted at ``reference``; ``payload_array``
    selects the array inside its data; ``attributes`` are native fields a copy must keep.
    """
    pytest.importorskip("avstack.config")
    from avstack import calibration, sensors

    case = request.param

    def make_sensor(reference):
        mounted = getattr(calibration, case["calibration"])(
            reference, *case.get("calibration_args", ()), **case.get("calibration_options", {}))
        return getattr(sensors, case["sensor"])(.5, 10, deepcopy(case["data"]), mounted, case["sensor"],
                                                **case.get("sensor_options", {}))

    return make_sensor, case.get("payload_array", whole_payload), case.get("attributes", {})
