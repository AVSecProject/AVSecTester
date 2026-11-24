"""Camera observations and learned-model boundaries shared by driving-system tests."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.data_types import DeliveredAcquisition, Observation, TimedStateEstimate
from tests import fakes


@pytest.fixture
def alpamayo_model_boundary(monkeypatch):
    base = pytest.importorskip("alpasim_driver.models.base")
    pytest.importorskip("alpasim_grpc.v0.egodriver_pb2")
    pytest.importorskip("alpasim_driver.navigation")
    return fakes.alpamayo_model_boundary(base, monkeypatch)


@pytest.fixture
def camera_detector_boundary(monkeypatch):
    pytest.importorskip("avstack.config")
    api = pytest.importorskip("mmdet.apis")
    return fakes.camera_detector_boundary(api, monkeypatch)


@pytest.fixture
def camera_observation():
    return lambda: Observation(0., 0, {"front": np.zeros((32, 32, 3), np.uint8)})


def synthetic_policy_observations(count=16, start_time=-1.5, time_step=.1, speed=4.,
                                  acceleration=0., yaw=0., route=None):
    """A small synthetic recording supplied to policy tests as actual observed history."""
    observations = []
    for index in range(count):
        elapsed = index * time_step
        time = start_time + elapsed
        distance = speed * elapsed + .5 * acceleration * elapsed ** 2
        state = PlanarEgoState(x=distance * float(np.cos(yaw)), y=distance * float(np.sin(yaw)),
                        yaw=yaw, speed=speed + acceleration * elapsed, t=time,
                        acceleration=acceleration)
        observations.append(Observation(
            time, index, {"front": np.full((4, 6, 3), index, np.uint8)},
            vehicle_state=state, ego_speed=state.speed, route=deepcopy(route)))
    # The first policy decision receives all recorded context in one incremental delivery.
    origin = min(15, len(observations) - 1)
    initial = observations[origin]
    states = tuple(TimedStateEstimate(round(item.t * 1e6), item.t,
                    item.vehicle_state, item.ego_speed) for item in observations[:origin + 1])
    samples = tuple(DeliveredAcquisition(state.time_us, item.t, item.frame,
                   item.sensor_data["front"], item.calibration.get("front"), state,
                   item.frame - origin) for item, state in zip(observations[:origin + 1], states))
    observations[origin] = replace(initial, deliveries={"front": samples}, state_estimates=states)
    return tuple(observations)


@pytest.fixture
def alpamayo_observations():
    return synthetic_policy_observations


@pytest.fixture
def policy_observation(alpamayo_observations):
    return lambda: alpamayo_observations()[-1]


@pytest.fixture
def rule_based_detector_model():
    stop = [0.] * 80
    stop[11] = 1.
    car = [0.] * 80
    car[2] = 1.
    return fakes.yolox_detection_model([
        [50., 40., 4., 4., .9, *stop],
        [10., 20., 4., 4., .95, *car],
    ])


@pytest.fixture
def rule_based_observation():
    from avsectester.geometry import CameraCalibration, PinholeCamera
    from avsectester.geometry.cameras import BODY_FROM_OPTICAL

    return lambda: Observation(
        0., 0, {"front": np.full((80, 100, 3), 30, np.uint8)},
        {"front": CameraCalibration(PinholeCamera(
            np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]), 100, 80),
            BODY_FROM_OPTICAL)},
        PlanarEgoState(speed=10.), 10.,
    )


@pytest.fixture
def rule_based_detector(rule_based_detector_model):
    from avsectester.closed_loop.avstacks.implemented.modules import YOLOXDetector

    return YOLOXDetector("synthetic-local-weights", device="cpu", model=rule_based_detector_model)
