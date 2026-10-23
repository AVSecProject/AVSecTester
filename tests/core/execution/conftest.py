"""Synthetic longitudinal recordings and optional controller cases for closed-loop tests."""

import json
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController, TrajectoryFollower
from avsectester.datasets.nurec import NuRecDataset


@pytest.fixture
def nurec_longitudinal_recording(nurec_usdz_path, tmp_path):
    path = tmp_path / "longitudinal.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source:
        metadata = json.loads(source.read("rig_trajectories.json"))
        recording = metadata["rig_trajectories"][0]
        camera = next(iter(metadata["camera_calibrations"]))
        timestamps, transforms = [], []
        for position in range(31):
            timestamp = position * 100000
            pose = np.eye(4)
            pose[:3, 3] = [position, 0., .3]
            timestamps.append([timestamp, timestamp + 1])
            transforms.append([pose.tolist(), pose.tolist()])
        recording["cameras_frame_timestamps_us"][camera] = timestamps
        recording["cameras_frame_T_rig_worlds"][camera] = transforms
        with zipfile.ZipFile(path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json"
                                else source.read(name))
    return NuRecDataset([path], time_step=.1)


@pytest.fixture
def trajectory_follower_deceleration():
    return TrajectoryFollower(), 1., .02


@pytest.fixture
def alpasim_controller_deceleration():
    pytest.importorskip("alpasim_controller")
    return AlpaSimController(), 2., .3


@pytest.fixture(params=["trajectory_follower_deceleration", "alpasim_controller_deceleration"])
def deceleration_case(request):
    return request.getfixturevalue(request.param)
