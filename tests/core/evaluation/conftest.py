"""Factories shared by offline evaluation tests, without importing other test modules."""

import math

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.closed_loop.datatypes import Control, WorldSnapshot
from avsectester.datasets import ActorState, FrameReference
from avsectester.evaluation import ComparisonGroup, TreatmentPair
from avsectester.runlog import RunLog, StepRecord


@pytest.fixture
def actor_factory():
    def actor(x=0, y=0, *, z=0, yaw=0, extent=(4, 2, 2), category="car", box_center=(0, 0, 0)):
        pose = np.eye(4)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
        pose[:3, 3] = [x, y, z]
        return ActorState(pose, (0, 0, 0), category, extent, box_center)
    return actor


@pytest.fixture
def log_factory(actor_factory):
    def log(velocities, brake=None, *, shift=0, actors=None, detections=()):
        brake = [0] * len(velocities) if brake is None else brake
        steps = []
        for index, (speed, braking) in enumerate(zip(velocities, brake)):
            pre = WorldSnapshot(index * .1, index, ego_speed=speed,
                                ego=actor_factory(index + shift), actors=actors or {})
            post = WorldSnapshot((index + 1) * .1, index + 1, ego_speed=speed,
                                 ego=actor_factory(index + 1 + shift), actors=actors or {})
            steps.append(StepRecord(index, physical_pre=pre, physical_post=post,
                                    executed_command=Control(throttle=0 if braking else .5, brake=braking),
                                    detections=detections))
        return RunLog(metadata={"avstack_type": "TestDriver"}, steps=steps, status="completed")
    return log


@pytest.fixture
def group_factory():
    def group(runs, roles=None):
        segment = AttackSegment("scene", FrameReference("scene", 0, 0, 0),
                                FrameReference("scene", 4, 4, .4), roles=roles or {})
        return ComparisonGroup("attack", segment, 0, runs)
    return group


@pytest.fixture
def pair_factory():
    def pair(first, second):
        return TreatmentPair("first", "second", first, second)
    return pair
