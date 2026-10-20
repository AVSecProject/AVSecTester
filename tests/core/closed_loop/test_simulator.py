"""The base simulator runtime's public capture, replay and ownership contract."""

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulator import DeploymentPlan, can_deploy_at
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator


def test_reset():
    data = MemoryDataset()
    frames = data.frames("scene")
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset(data.scenario_start("scene", frames[3]))
        assert simulator_runtime.observe().frame == 3
        assert simulator_runtime.state().t == pytest.approx(frames[3].timestamp)
        simulator_runtime.reset()
        assert simulator_runtime.observe().frame == 0
        assert simulator_runtime.clock.t == 0.
        with pytest.raises(ValueError, match="does not support"):
            simulator_runtime.reset(world={"sensors": []})


def test_step():
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError, match="Reset"):
            simulator_runtime.step(Control())
        simulator_runtime.reset()
        before = simulator_runtime.clock
        simulator_runtime.step(Control(throttle=.5))
        assert simulator_runtime.clock.frame == before.frame + 1
        assert simulator_runtime.clock.t == pytest.approx(before.t + .1)
        assert simulator_runtime.observe().frame == simulator_runtime.clock.frame
        with pytest.raises(ValueError, match="Simulator does not support trajectory commands"):
            simulator_runtime.step(Control(trajectory=np.zeros((2, 3))))


def test_observe_and_state():
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError, match="Reset"):
            simulator_runtime.observe()
        simulator_runtime.reset()
        first = simulator_runtime.observe()
        original = first.sensor_data["front"].copy()
        first.sensor_data["front"][:] = 0
        np.testing.assert_array_equal(simulator_runtime.observe().sensor_data["front"], original)
        state = simulator_runtime.state()
        state.ego.pose[0, 3] = 1000.
        assert simulator_runtime.state().ego.pose[0, 3] != 1000.
        assert simulator_runtime.clock.frame == first.frame


def test_setup_tick():
    data = MemoryDataset()
    deployment = DeploymentPlan(2)
    starts = (0, 4)

    class Deploy:
        setup_ticks = deployment.ticks

        def __call__(self, value, context):
            context.simulator_runtime.setup_tick()

    for position in starts:
        with MemorySimulator(capture_ticks=1).create_runtime() as simulator_runtime:
            hooks = HookSet([Hook("simulator.setup", Deploy())])
            with hooks.session(simulator_runtime, ConstantAVStack()).activate():
                simulator_runtime.reset(data.scenario_start("scene", data.frames("scene")[position]),
                                        deployment=deployment)
            assert simulator_runtime.observe().frame == position
            assert simulator_runtime.clock.t == pytest.approx(position * .1)
            with pytest.raises(RuntimeError, match="simulator.setup"):
                simulator_runtime.setup_tick()
            assert can_deploy_at(position, deployment, 1)
            with pytest.raises(ValueError, match="deployment window"):
                simulator_runtime.reset(data.scenario_start("scene", data.frames("scene")[1]),
                                        deployment=deployment)


def test_generate_frame_data():
    data = MemoryDataset()
    frames = data.frames("scene")[:4]
    with MemorySimulator().create_runtime() as simulator_runtime:
        sequence = simulator_runtime.generate_frame_data(data.scenario_start("scene", frames[0]), frames)
        assert tuple(record.frame for record in sequence.frames) == tuple(frames)
        for record in sequence.frames:
            simulator_runtime.reset(data.scenario_start("scene", record.frame))
            np.testing.assert_allclose(record.ego.pose, simulator_runtime.state().ego.pose)
        with pytest.raises(ValueError, match="consecutive"):
            simulator_runtime.generate_frame_data(data.scenario_start("scene", frames[0]), frames[::2])


def test_exclusive():
    simulator = MemorySimulator(resource_key=("memory", "shared-world"))
    with simulator.create_runtime() as first, simulator.create_runtime() as second:
        with first.exclusive():
            with pytest.raises(RuntimeError, match="already"):
                with second.exclusive():
                    pytest.fail("two independent users acquired one world")
        with second.exclusive():
            second.reset()
            assert second.observe().frame == 0
