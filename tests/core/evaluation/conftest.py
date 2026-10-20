"""Small recorded worlds used by metric, report and visualization tests."""

import math

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.closed_loop.datatypes import Control, WorldSnapshot
from avsectester.datasets import ActorState, FrameReference
from avsectester.evaluation import ComparisonGroup, TreatmentPair
from avsectester.runlog import RunLog, StepRecord


@pytest.fixture
def make_actor():
    def actor(x=0., y=0., *, z=0., yaw=0., extent=(4., 2., 2.), category="car"):
        pose = np.eye(4)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
        pose[:3, 3] = [x, y, z]
        return ActorState(pose, (0., 0., 0.), category, extent)
    return actor


@pytest.fixture
def make_log(make_actor):
    def log(velocities, brake=None, *, shift=0., actors=None, detections=()):
        brake = [0.] * len(velocities) if brake is None else brake
        steps = []
        for index, (speed, braking) in enumerate(zip(velocities, brake)):
            pre = WorldSnapshot(index * .1, index, ego_speed=speed,
                                ego=make_actor(index + shift), actors=actors or {})
            post = WorldSnapshot((index + 1) * .1, index + 1, ego_speed=speed,
                                 ego=make_actor(index + 1 + shift), actors=actors or {})
            steps.append(StepRecord(
                index, physical_pre=pre, physical_post=post, detections=detections,
                executed_command=Control(throttle=0. if braking else .5, brake=braking),
            ))
        return RunLog(metadata={"avstack_type": "ConstantAVStack"}, steps=steps, status="completed")
    return log


@pytest.fixture
def segment():
    return AttackSegment("scene", FrameReference("scene", 0, 0, 0.),
                         FrameReference("scene", 3, 3, .3), roles={"host": ("lead",)})


@pytest.fixture
def make_group(segment):
    return lambda runs: ComparisonGroup("attack", segment, 0, runs)


@pytest.fixture
def make_pair():
    return lambda first, second: TreatmentPair("clean", "attacked", first, second)
