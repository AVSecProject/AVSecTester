"""Insertion actions expose evidence and own their world-object cleanup."""
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import AttackRuntime
from avsectester.attack.insertion import (
    InsertObject, Insertion, InsertionAttackAction, MoveObject, PlaneAsset,
)
from avsectester.attack.insertion.placement import Orientation, WorldPlacement
from avsectester.closed_loop.simulators.base import SensorView
from avsectester.data_types import ActorState
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.rendering import ObjectAsset, PlacedObject, PlaneSurface
from tests.fakes import ConstantAVStack, MemorySimulator


def test_insertion_attack_action(camera, optical_from_world):
    image = np.full((80, 100, 3), [20, 60, 80], np.uint8)
    texture = np.full((16, 16, 4), [220, 0, 0, 255], np.uint8)
    target = Insertion("target", PlaneAsset(texture, 4., 2.), WorldPlacement((10., 0., 0.)),
                       Orientation("fixed_world", (0., 0., 180.)))
    texture[:, :8, 3] = 0
    surface = PlaneSurface(np.array([[0., -1., 1.], [0., 1., 1.],
                                    [0., 1., 0.], [0., -1., 0.]]), texture)
    pose = np.eye(4)
    pose[:3, 3] = (5., 0., -.5)
    placed = PlacedObject("screen", ObjectAsset.from_surfaces((surface,), "screen"), pose)
    view = SensorView("front", image, camera, optical_from_world,
                      {"unbounded": ActorState(np.eye(4))}, ActorState(np.eye(4)), 0., 0,
                      objects={"screen": placed})
    action = InsertionAttackAction("front", target, active_for_us=100000)
    assert action.active_for_us == 100000
    rendered = action.apply(image, view)
    assert np.any(rendered[:, 50:] != image[:, 50:])
    np.testing.assert_array_equal(rendered[:, :50], image[:, :50])
    assert action.receipt_details() == {
        "rendered": True, "reason": "", "occlusion_source": "actor_cuboids",
        "ignored_occluders": ("unbounded",),
    }
    action.apply(image, replace(view, depth=np.full((80, 100), np.inf)))
    assert action.receipt_details()["ignored_occluders"] == ()


def test_insert_object(make_object_in_front):
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, world_from_object = make_object_in_front(simulator_runtime.sensor_view("front"))
        action = InsertObject("sign", asset, world_from_object)
        with ExitStack() as resources:
            action.apply(simulator_runtime, resources)
            assert "sign" in simulator_runtime.state().objects
            details = action.receipt_details()
            assert set(details) == {"object_id", "asset_category", "world_from_object", "velocity"}
            assert details["object_id"] == "sign" and details["asset_category"] == asset.category
            np.testing.assert_array_equal(details["world_from_object"], world_from_object)
            assert details["velocity"] == (0., 0., 0.)
        assert simulator_runtime.state().objects == {}
        with ExitStack() as resources:
            action.apply(simulator_runtime, resources)
            simulator_runtime.remove_object("sign")
            assert "sign" not in simulator_runtime.state().objects
        assert "sign" not in simulator_runtime.state().objects
        with pytest.raises(ValueError, match="velocity must contain three finite"):
            InsertObject("sign", asset, world_from_object, velocity=(0., np.nan, 0.))


def test_move_object(make_object_in_front):
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        asset, pose = make_object_in_front(simulator_runtime.sensor_view("front"))
        target = pose.copy()
        target[0, 3] += 1.

        class MovingSign(AttackRuntime):
            def setup(self, context):
                return (InsertObject("sign", asset, pose, velocity=(2., 0., 0.)),)

            def step(self, context):
                return (MoveObject("sign", target),)

        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(),
                                     RunRequest(1, attacks=(MovingSign(),)))

    step, = log.steps
    np.testing.assert_array_equal(step.physical_pre.objects["sign"].velocity, (2., 0., 0.))
    np.testing.assert_array_equal(step.physical_post.objects["sign"].pose, target)
    initialization = next(event for event in log.events if event.kind == "initialization")
    inserted, = [action for action in initialization.data["actions"] if action["kind"] == "receipt"]
    assert inserted["outcome"] == "applied"
    assert inserted["details"]["velocity"] == (2., 0., 0.)
    moved, = [action for action in step.actions if action["kind"] == "receipt"]
    assert moved["outcome"] == "applied"
    assert moved["details"]["object_id"] == "sign"
    np.testing.assert_array_equal(moved["details"]["world_from_object"], target)
