"""Tiny datasets, recorded worlds and geometry shared by public API tests."""
import json
import math
import zipfile

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.datasets import FrameReference
from avsectester.data_types import ActorState, Control, WorldSnapshot
from avsectester.runlog import RunLog, StepRecord



@pytest.fixture
def nurec_map_variants(nurec_usdz_path, tmp_path):
    """The same reconstruction with a light placeholder or only implicit wait lines."""
    import io
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    with zipfile.ZipFile(nurec_usdz_path) as archive:
        line_rows = parquet.read_table(io.BytesIO(archive.read("clipgt/wait_line.parquet"))).to_pylist()
    for row in line_rows:
        row["wait_line"]["is_implicit"] = True
    replacements = {
        "placeholder_lights": ("traffic_light", [{"key": {"map_id": None}, "traffic_light": {
            "center": None, "dimensions": None, "orientation": None, "category": None}}]),
        "implicit_only_lines": ("wait_line", line_rows),
    }
    paths = {}
    for variant, (layer, rows) in replacements.items():
        path = tmp_path / f"{variant}.usdz"
        member_name = f"clipgt/{layer}.parquet"
        output = arrow.BufferOutputStream()
        parquet.write_table(arrow.Table.from_pylist(rows), output)
        with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(path, "w") as destination:
            for member in source.infolist():
                if member.filename != member_name:
                    destination.writestr(member.filename, source.read(member.filename))
            destination.writestr(member_name, output.getvalue().to_pybytes())
        paths[variant] = str(path)
    return paths


@pytest.fixture
def nuscenes_root(tmp_path):
    """Three linked samples read by the real nuScenes SDK, including a moving host."""
    pytest.importorskip("nuscenes")
    from PIL import Image
    root = tmp_path / "nuscenes"
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    tables = {name: [] for name in (
        "category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor",
        "ego_pose", "log", "scene", "sample", "sample_data", "sample_annotation", "map")}
    tables["category"] = [{"token": "car-category", "name": "vehicle.car", "description": "car"}]
    tables["sensor"] = [{"token": "front-sensor", "channel": "CAM_FRONT", "modality": "camera"}]
    tables["calibrated_sensor"] = [{"token": "front-calibration", "sensor_token": "front-sensor",
        "translation": [0., 0., 1.5], "rotation": [.5, -.5, .5, -.5],
        "camera_intrinsic": [[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]}]
    tables["instance"] = [{"token": "host", "category_token": "car-category",
        "nbr_annotations": 3, "first_annotation_token": "host-0", "last_annotation_token": "host-2"}]
    tables["log"] = [{"token": "log", "logfile": "synthetic", "vehicle": "test", "date_captured": "2020-01-01",
                       "location": "boston-seaport"}]
    tables["scene"] = [{"token": "scene", "name": "synthetic-road", "description": "tiny road",
        "log_token": "log", "nbr_samples": 3, "first_sample_token": "sample-0", "last_sample_token": "sample-2"}]
    for position in range(3):
        previous = lambda prefix: f"{prefix}-{position - 1}" if position else ""
        following = lambda prefix: f"{prefix}-{position + 1}" if position < 2 else ""
        tables["sample"].append({"token": f"sample-{position}", "scene_token": "scene",
            "timestamp": position * 100000, "prev": previous("sample"), "next": following("sample")})
        filename = f"front-{position}.png"
        Image.fromarray(np.full((80, 100, 3), 50 + position, np.uint8)).save(root / filename)
        tables["sample_data"].append({"token": f"image-{position}", "sample_token": f"sample-{position}",
            "ego_pose_token": f"ego-{position}", "calibrated_sensor_token": "front-calibration",
            "timestamp": position * 100000, "fileformat": "png", "is_key_frame": True,
            "filename": filename, "width": 100, "height": 80,
            "prev": previous("image"), "next": following("image")})
        tables["ego_pose"].append({"token": f"ego-{position}", "timestamp": position * 100000,
                                   "translation": [position * .1, 0., 0.], "rotation": [1., 0., 0., 0.]})
        tables["sample_annotation"].append({"token": f"host-{position}", "sample_token": f"sample-{position}",
            "instance_token": "host", "attribute_tokens": [], "visibility_token": "",
            "translation": [8. + position * .2, 0., 1.], "size": [2., 4., 2.], "rotation": [1., 0., 0., 0.],
            "num_lidar_pts": 1, "num_radar_pts": 0, "prev": previous("host"), "next": following("host")})
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(root / "map.png")
    tables["map"] = [{"token": "map", "filename": "map.png", "log_tokens": ["log"]}]
    for name, records in tables.items():
        (version / f"{name}.json").write_text(json.dumps(records))
    return str(root)


@pytest.fixture
def nuscenes_dataset(nuscenes_root):
    from avsectester.datasets.nuscenes import NuScenesDataset
    return NuScenesDataset(nuscenes_root, version="v1.0-mini")


@pytest.fixture
def carla_dataset():
    from avsectester.data_types import Control
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion

    recipe = CarlaRecipe({"client": {"map_name": "Town03_Opt"}, "ego": {
        "type": "CarlaMobileActor", "spawn": 0, "vehicle": "vehicle.a"}}, 4, .1,
        {"ego": PrescribedMotion(constant_control=Control(throttle=.2))})
    return CarlaDataset({"scene": recipe})


@pytest.fixture
def nurec_dataset(nurec_usdz_path):
    from avsectester.datasets.nurec import NuRecDataset
    return NuRecDataset([nurec_usdz_path], time_step=.1)


@pytest.fixture
def assert_camera_views_equal():
    def compare(first, second):
        assert first.sensor == second.sensor
        assert first.native == second.native == {}
        np.testing.assert_array_equal(first.data, second.data)
        np.testing.assert_equal(first.depth, second.depth)
        np.testing.assert_allclose(first.cam_from_world, second.cam_from_world)
        np.testing.assert_allclose(first.sensor_from_world, second.sensor_from_world)
        assert type(first.camera) is type(second.camera)
        assert (first.camera.width, first.camera.height) == (second.camera.width, second.camera.height)
        pixels = np.array([[.5, .5], [first.camera.width / 2, first.camera.height / 2]])
        np.testing.assert_allclose(first.camera.unproject(pixels), second.camera.unproject(pixels))
        assert first.actors.keys() == second.actors.keys()
        for identity in first.actors:
            np.testing.assert_allclose(first.actors[identity].box_transform(), second.actors[identity].box_transform())
            assert first.actors[identity].extent == second.actors[identity].extent
        np.testing.assert_allclose(first.ego.box_transform(), second.ego.box_transform())
        assert first.ego.extent == second.ego.extent
    return compare


@pytest.fixture
def make_object_in_front():
    """An opaque rectangular face two metres along the selected camera's optical axis."""
    from avsectester.rendering import ObjectAsset, PlaneSurface

    def make(view):
        surface = PlaneSurface(np.array([[0., -1., 2.], [0., 1., 2.],
                                         [0., 1., 0.], [0., -1., 0.]]),
                               np.full((8, 8, 4), [230, 10, 90, 255], np.uint8))
        asset = ObjectAsset((surface,), (.1, 2., 2.), "stop_sign")
        optical_from_object = np.array([[0., 1., 0., 0.], [0., 0., -1., 1.],
                                        [-1., 0., 0., 2.], [0., 0., 0., 1.]])
        return asset, np.linalg.inv(view.cam_from_world) @ optical_from_object
    return make


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
                         FrameReference("scene", 3, 3, .3))
