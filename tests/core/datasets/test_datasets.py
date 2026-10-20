"""Dataset components share one public timeline contract; interpolation stays numerical."""
from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.datasets import FrameData, Unavailable
from avsectester.datasets.carla import CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.poses import rotation_matrix


@pytest.fixture(params=["carla_dataset", "nurec_dataset", "nuscenes_dataset"])
def dataset_case(request):
    return request.getfixturevalue(request.param)


def test_datasets(dataset_case):
    dataset = dataset_case
    scenario = next(dataset.scenarios())
    frames = dataset.frames(scenario.id)
    assert [frame.position for frame in frames] == list(range(len(frames)))
    assert all(right.timestamp > left.timestamp for left, right in zip(frames, frames[1:]))
    assert dataset.decision_window(scenario.id, frames[0], len(frames) - 1) == frames
    start = dataset.scenario_start(scenario.id, frames[1])
    assert start.frame == frames[1] and start.scenario_id == scenario.id
    with pytest.raises(ValueError, match="terminal frame"):
        dataset.decision_window(scenario.id, frames[-1], 1)
    with pytest.raises(ValueError, match="scenario's frames"):
        dataset.scenario_start(scenario.id, replace(frames[1], timestamp=99.))


@pytest.mark.parametrize("dataset_fixture", ["nurec_dataset", "nuscenes_dataset"])
def test_recorded_frame_data(dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    scenario = next(dataset.scenarios())
    frame = dataset.frames(scenario.id)[1]
    data = dataset.frame_data(scenario.id, frame)
    assert isinstance(data, FrameData)
    assert data.frame == frame and data.ego is not None
    assert "host" in data.actors and "ego" not in data.actors
    assert data.actors["host"].extent == (4., 2., 2.)


def test_carla_frame_data(carla_dataset):
    frame = carla_dataset.frames("scene")[1]
    data = carla_dataset.frame_data("scene", frame)
    assert isinstance(data, Unavailable) and data.reason == "recipe_only"


def test_nurec_interpolation(nurec_usdz_path):
    dataset = NuRecDataset([nurec_usdz_path], time_step=.05)
    frames = dataset.frames("scene")
    for position, timestamp in ((0, 0.), (5, .25), (10, .5)):
        frame = dataset.frame_data("scene", frames[position])
        np.testing.assert_allclose(frame.ego.pose[:3, 3], [2 * timestamp, 0., .3])
        np.testing.assert_allclose(frame.actors["host"].pose[:3, 3], [8 + 4 * timestamp, 0., 1.])
        np.testing.assert_allclose(frame.actors["host"].pose[:3, :3],
                                   rotation_matrix((0., 0., 360 * timestamp)), atol=1e-12)
        assert ("brief" in frame.actors) == (timestamp == .25)
    ego = dataset.frame_data("scene", frames[5]).ego
    assert ego.extent == (4., 2., 1.5) and ego.box_center == (.2, 0., .7)
    # Returned annotations and arrays remain independent of the immutable recording.
    data = dataset.frame_data("scene", frames[5])
    data.actors["host"].pose[:] = 0
    metadata = dataset.raw_metadata(nurec_usdz_path)
    metadata["sequence_tracks.json"].clear()
    np.testing.assert_allclose(dataset.frame_data("scene", frames[5]).actors["host"].pose[0, 3], 9.)


def test_carla_recipe():
    recipe = CarlaRecipe({"ego": {}}, 2, .1,
        {"ego": PrescribedMotion(constant_control=Control(throttle=.4))})
    assert [command.throttle for command in recipe.motion["ego"].controls] == [.4, .4]
    recipe.motion["ego"].controls[0].throttle = .1
    assert recipe.motion["ego"].controls[1].throttle == .4
    with pytest.raises(ValueError, match="exactly these actors"):
        CarlaRecipe({"ego": {}, "lead": {"gap": 8}}, 2, .1, recipe.motion)
    with pytest.raises(ValueError, match="runtime settings"):
        CarlaRecipe({"ego": {}, "client": {"seed": 1}}, 2, .1, recipe.motion)
