"""Real avstack sensor/reference objects remain independent of simulator state."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("avstack.config")
pytest.importorskip("carla")

from avcarla.geometry import CarlaReferenceFrame
from avstack.geometry import GlobalOrigin3D

from avsectester.data_types import Control, Observation, StateEstimate
from avsectester.closed_loop.simulators.carla import CarlaSimulator


def test_copy_observation(native_sensor_case, make_ego):
    make_sensor, payload_array, attributes = native_sensor_case
    body = CarlaReferenceFrame(GlobalOrigin3D, location=(1, 2, 0))
    sensor_reference = CarlaReferenceFrame(body, location=(0, 0, 1))
    sensor = make_sensor(sensor_reference)
    calibration = sensor.calibration
    original = payload_array(sensor.data).copy()
    state = make_ego(xyz=(1, 2, 0), speed=4)
    observation = Observation(.5, 10, {"sensor": sensor}, {"sensor": calibration}, state, 4)

    with CarlaSimulator({}).create_runtime() as runtime:
        copied = runtime.copy_observation(observation)
    copied_sensor = copied.sensor_data["sensor"]
    assert copied_sensor.calibration is copied.calibration["sensor"]
    assert copied_sensor.reference.reference is not body
    copied_sensor.reference.reference.x[:] = 90
    copied.vehicle_state.position.x[:] = -90
    payload_array(copied_sensor.data)[:] = 99
    np.testing.assert_array_equal(payload_array(sensor.data), original)
    np.testing.assert_array_equal(body.x, [1, 2, 0])
    np.testing.assert_array_equal(state.position.x, [1, 2, 0])
    for name, expected in attributes.items():
        assert getattr(copied_sensor, name) == expected
    replacement = deepcopy(copied.calibration["sensor"])
    copied.set_calibration("sensor", replacement)
    assert copied_sensor.calibration is copied.calibration["sensor"] is replacement
    assert copied_sensor.reference is replacement.reference
    assert sensor.calibration is calibration


def test_delivered_native_state_interpolation(monkeypatch, make_ego):
    from avsectester.closed_loop.hooks import Hook, HookSet
    from avsectester.closed_loop.timing import SensorTiming, TimingConfiguration
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
    from avsectester.execution.runner import ClosedLoopRunner, RunRequest
    from tests.fakes import CarlaNativeBoundary, ConstantAVStack

    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = CarlaRecipe({"client": {"map_name": "Town03_Opt"}, "ego": {
        "type": "CarlaMobileActor", "spawn": 0, "vehicle": "vehicle.a"}}, 4, .1,
        {"ego": PrescribedMotion(constant_control=Control())})
    dataset = CarlaDataset({"scene": recipe})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    localized, decisions = [], []

    def localize(value, context):
        frame = context.physical_state.frame
        origin_frame = localized[0][0] if localized else frame
        elapsed = (frame - origin_frame) / 100.
        state = make_ego(xyz=(100. * elapsed, 20. * elapsed, 0.),
                         yaw=4. * elapsed, speed=2. + 20. * elapsed,
                         t=context.observation.t)
        localized.append((frame, state))
        return StateEstimate(state, 2. + 20. * elapsed)

    def remember(value, context):
        decisions.append(value)
        return value

    simulator = CarlaSimulator({"rate": 100}, decision_period_us=100000, settle_iters=0)
    timing = TimingConfiguration(sensors={"front": SensorTiming(30000)},
                                 pose_reporting_interval_us=50000)
    hooks = HookSet([Hook("localization.post", localize), Hook("observation", remember)])
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(),
                                     RunRequest(2, start=start, timing=timing, hooks=hooks))
    assert log.status == "completed"
    origin = decisions[0].delivered("front")[0].time_us
    samples = decisions[1].delivered("front")
    assert [item.time_us - origin for item in samples] == [30000, 60000, 90000]
    for item in samples:
        elapsed = (item.time_us - origin) / 1e6
        state = item.state.vehicle_state
        np.testing.assert_allclose(state.position.x, [100. * elapsed, 20. * elapsed, 0.])
        before = np.floor(elapsed / .05) * .05
        after = before + .05
        fraction = (elapsed - before) / .05
        first_velocity = (2. + 20. * before) * np.array([np.cos(4. * before), np.sin(4. * before), 0.])
        last_velocity = (2. + 20. * after) * np.array([np.cos(4. * after), np.sin(4. * after), 0.])
        np.testing.assert_allclose(state.velocity.x,
                                   (1. - fraction) * first_velocity + fraction * last_velocity)
        np.testing.assert_allclose(state.attitude.q.components,
                                   make_ego(yaw=4. * elapsed).attitude.q.components, atol=1e-12)
        assert state.t == item.state.t == item.time_us / 1e6
        assert type(state) is type(localized[0][1])
    np.testing.assert_array_equal(localized[0][1].position.x, [0., 0., 0.])
    assert decisions[0].delivered("front")[0].state.vehicle_state is decisions[0].vehicle_state
