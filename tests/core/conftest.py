"""Tiny on-disk NuRec and nuScenes records shared by public dataset and attack tests."""
import json
import zipfile

import numpy as np
import pytest


@pytest.fixture
def nurec_usdz_path(tmp_path):
    camera = "camera_front_wide_120fov@scene"
    transforms = []
    for position in range(3):
        pose = np.eye(4)
        pose[:3, 3] = [position * .5, 0., .3]
        transforms.append([pose.tolist(), pose.tolist()])
    rig = {
        "camera_calibrations": {camera: {
            "camera_model": {"parameters": {"resolution": [16, 12],
                "principal_point": [8, 6], "angle_to_pixeldist_poly": [0, 8], "max_angle": 1.5}},
            "T_sensor_rig": [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]],
        }},
        "rig_trajectories": [{"sequence_id": "scene", "rig_bbox": {
            "dim": [4., 2., 1.5], "centroid": [.2, 0., .7]},
            "cameras_frame_timestamps_us": {camera: [[0, 1], [250000, 250001], [500000, 500001]]},
            "cameras_frame_T_rig_worlds": {camera: transforms}}],
    }
    tracks = {"scene": {"tracks_data": {
        "tracks_id": ["host", "brief"], "tracks_label_class": ["automobile", "pedestrian"],
        "tracks_timestamps_us": [[0, 500000], [200000, 300000]],
        "tracks_poses": [[[8, 0, 1, 0, 0, 0, 1], [10, 0, 1, 0, 0, 1, 0]],
                         [[5, 2, 1, 0, 0, 0, 1], [5, 2, 1, 0, 0, 0, 1]]]},
        "cuboidtracks_data": {"cuboids_dims": [[4, 2, 2], [1, 1, 2]]}}}
    path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps(rig))
        archive.writestr("sequence_tracks.json", json.dumps(tracks))
    return str(path)


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
    from avsectester.closed_loop.datatypes import Control
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion

    recipe = CarlaRecipe({"client": {"map_name": "Town03_Opt"}, "ego": {
        "type": "CarlaMobileActor", "spawn": 0, "vehicle": "vehicle.a"}}, 4, .1,
        {"ego": PrescribedMotion(constant_control=Control(throttle=.2))})
    return CarlaDataset({"scene": recipe})


@pytest.fixture
def nurec_dataset(nurec_usdz_path):
    from avsectester.datasets.nurec import NuRecDataset
    return NuRecDataset([nurec_usdz_path], time_step=.1)
