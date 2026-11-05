"""The driving-system interface owns its typed observation pipeline."""

from dataclasses import replace
import math

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.base import (
    CommandCompatibilityError, ObservationPipeline, validate_command_compatibility,
)
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState
from avsectester.data_types import StateEstimate
from tests.fakes import ConstantAVStack, DeceleratingAVStack, MemorySimulator


def test_observation_pipeline():
    avstack = ConstantAVStack()
    assert isinstance(avstack.observation_pipeline, ObservationPipeline)
    pipeline = ObservationPipeline()
    avstack.observation_pipeline = pipeline
    assert avstack.observation_pipeline is pipeline
    with pytest.raises(TypeError, match="observation_pipeline must be ObservationPipeline"):
        avstack.observation_pipeline = object()


def test_observation_pipeline_process():
    stages = []

    def change(value, context):
        stages.append(context.stage)
        return replace(value, ego_speed=value.ego_speed + 1.)

    def localizer(value, context):
        stages.append(context.stage)
        return StateEstimate(value.vehicle_state, value.ego_speed + 2.)

    hooks = HookSet([Hook(stage, change) for stage in
                     ("sensors.post", "localization.pre", "localization.post", "observation")])
    pipeline = ObservationPipeline(localizer)
    avstack = ConstantAVStack()
    avstack.observation_pipeline = pipeline
    with MemorySimulator().create_runtime() as runtime:
        runtime.reset()
        observation = runtime.observe()
        original_speed = observation.ego_speed
        with hooks.session(runtime, avstack).activate() as dispatcher:
            result = pipeline.process(observation, dispatcher)
        assert result.ego_speed == observation.ego_speed + 6.
        assert stages == ["sensors.post", "localization.pre", "localization",
                          "localization.post", "observation"]
        assert observation.ego_speed == original_speed
    with pytest.raises(TypeError, match="localizer"):
        ObservationPipeline(object())


def test_observation_pipeline_route_follows_localization():
    def world(points, state):
        cosine, sine = math.cos(state.yaw), math.sin(state.yaw)
        return points[:, :2] @ np.array([[cosine, -sine], [sine, cosine]]).T + [state.x, state.y]

    def shifted(value, context):
        return StateEstimate(replace(value.vehicle_state, x=value.vehicle_state.x + 1.,
                                     yaw=value.vehicle_state.yaw + math.pi / 2), value.ego_speed)

    route = np.array([[2., 0., 0.], [5., 1., 0.]])
    avstack = ConstantAVStack()
    with MemorySimulator().create_runtime() as runtime:
        runtime.reset()
        observation = replace(runtime.observe(), route=route.copy(),
                              vehicle_state=PlanarEgoState(x=3., y=-2., yaw=.4))
        for hook, moved in ((shifted, True), (lambda value, context: value, False)):
            with HookSet([Hook("localization.post", hook)]).session(runtime, avstack).activate() as dispatcher:
                result = avstack.observation_pipeline.process(observation, dispatcher)
            np.testing.assert_allclose(world(result.route, result.vehicle_state),
                                       world(route, observation.vehicle_state), atol=1e-12)
            assert np.array_equal(result.route, route) != moved
            np.testing.assert_array_equal(result.route[:, 2], 0.)
        missing = HookSet([Hook("localization.post", lambda value, context: StateEstimate(None, 0.))])
        with missing.session(runtime, avstack).activate() as dispatcher:
            with pytest.raises(ValueError, match="planar x, y and yaw"):
                avstack.observation_pipeline.process(observation, dispatcher)


def test_validate_command_compatibility():
    with MemorySimulator().create_runtime() as runtime:
        validate_command_compatibility(runtime, ConstantAVStack())
        with pytest.raises(CommandCompatibilityError, match="trajectory"):
            validate_command_compatibility(runtime, DeceleratingAVStack())
        hooks = HookSet([Hook("command", lambda value, context: value)])
        validate_command_compatibility(runtime, DeceleratingAVStack(), hooks)
