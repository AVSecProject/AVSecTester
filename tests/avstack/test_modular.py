"""The real modular pipeline: component capture, native hooks and body-frame detections."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("avstack.config")

from avsectester.closed_loop.avstacks.modular import ModularAVStack
from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.closed_loop.hooks import Hook, HookSet
from tests.fakes import MemorySimulator


def test_component_log(pipeline_config, native_observation):
    avstack = ModularAVStack(pipeline_config)
    avstack.instrument()
    command = None
    snapshots = []
    for frame in range(8):
        observation = native_observation(frame, [(20. - frame * 1.5, 0., 0.)])
        if frame == 0:
            avstack.reset(observation)
        command = avstack(observation)
        snapshots.append(avstack.component_log())
    assert set(snapshots[-1]) == {"perception", "tracking", "planning", "control"}
    assert len(snapshots[0]["tracking"]) == 0 and len(snapshots[-1]["tracking"]) == 1
    assert snapshots[0]["planning"].top()[1].target_speed == 6.
    assert snapshots[-1]["planning"].top()[1].target_speed == 0.
    assert command.brake > 0 and snapshots[-1]["control"].brake == command.brake
    initial_position = snapshots[0]["perception"][0].position.x.copy()
    avstack(native_observation(8, []))
    np.testing.assert_allclose(snapshots[0]["perception"][0].position.x, initial_position)
    avstack.reset(native_observation())
    assert all(output is None for output in avstack.component_log().values())


def test_hooks(pipeline_config, native_observation, monkeypatch):
    from avstack.config import HOOKS
    from avstack.datastructs import DataContainer

    class HideDetections:
        def __call__(self, output):
            return (DataContainer(output.frame, output.timestamp, [], output.source_identifier,
                                  source_reference=output.source_reference),)

    monkeypatch.setitem(HOOKS.module_dict, "HideDetectionsForTest", HideDetections)
    avstack = ModularAVStack(pipeline_config)
    avstack.instrument(("perception", "control"))
    native_hook = NativeHook({"type": "HideDetectionsForTest"})
    hooks = HookSet([
        Hook("perception.post", native_hook),
        Hook("control.post", lambda value, context: SimpleNamespace(throttle=0., steer=.2, brake=.8)),
    ])
    with MemorySimulator().create_runtime() as simulator_runtime:
        with hooks.session(simulator_runtime, avstack).activate():
            simulator_runtime.reset()
            observation = native_observation()
            avstack.reset(observation)
            command = avstack(observation)
    assert command.brake == .8 and command.steer == .2
    assert len(avstack.component_log()["perception"]) == 0
    assert avstack.component_log()["control"].brake == .8
    ordinary = avstack(native_observation(1))
    assert ordinary.steer != .2 and len(avstack.detections()) == 1


def test_detections(pipeline_config, native_observation, make_ego, make_detections):
    true_ego = make_ego(xyz=(20., 10., 0.), yaw=np.pi / 3)
    observation = native_observation(ego=true_ego, xyzs=[(5., 0., 0.)], mount=(1., 2., 1.))
    false_localization = make_ego(xyz=(300., -400., 40.), yaw=-np.pi / 2)
    avstack = ModularAVStack(pipeline_config)
    avstack.reset(observation)
    avstack(replace(observation, vehicle_state=false_localization))
    (detection,) = avstack.detections()
    np.testing.assert_allclose(detection.position, [6., 2., 1.75], atol=1e-6)
    assert detection.extent == (4., 1.8, 1.5)
    assert detection.yaw == pytest.approx(0., abs=1e-6)
    assert detection.category == "Car" and detection.score == pytest.approx(.8)
    # A world frame is not a sensor mounting frame. Driving still works without canonical boxes.
    from avstack.geometry import GlobalOrigin3D
    unsupported = replace(observation, sensor_data={"lidar": make_detections([(5., 0., 0.)])},
                          calibration={"lidar": SimpleNamespace(reference=GlobalOrigin3D)})
    assert avstack(unsupported) is not None
    assert avstack.detections() is None
