"""Synthetic recorded data shared by core tests in several component folders."""

import json
import zipfile

import numpy as np
import pytest


@pytest.fixture
def nurec_usdz_path(tmp_path):
    """Write a three-frame NuRec scene archive (rig, camera, two tracks) and return its path."""
    camera = "camera_front_wide_120fov@test"
    optical_to_rig = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]]
    timestamps = [[0, 1], [250000, 250001], [500000, 500001]]
    identity = np.eye(4).tolist()
    rig = {
        "camera_calibrations": {
            camera: {
                "camera_model": {
                    "parameters": {
                        "resolution": [640, 480],
                        "principal_point": [320, 240],
                        "angle_to_pixeldist_poly": [0, 200],
                        "max_angle": 1.5,
                    }
                },
                "T_sensor_rig": optical_to_rig,
            },
        },
        "rig_trajectories": [
            {
                "sequence_id": "test",
                "cameras_frame_timestamps_us": {camera: timestamps},
                "cameras_frame_T_rig_worlds": {camera: [[identity, identity]] * 3},
            }
        ],
        "custom_annotation": {"weather": "cloudy"},
    }
    tracks = {
        "test": {
            "tracks_data": {
                "tracks_id": ["host", "obstacle"],
                "tracks_label_class": ["automobile", "protruding_object"],
                "tracks_timestamps_us": [[0, 500000], [0, 500000]],
                "tracks_poses": [
                    [[10, 0, 1, 0, 0, 0, 1], [12, 0, 1, 0, 0, 1, 0]],
                    [[15, 1, 1, 0, 0, 0, 1], [15, 1, 1, 0, 0, 0, -1]],
                ],
                "tracks_flags": ["CONTROLLABLE", "NONE"],
            },
            "cuboidtracks_data": {"cuboids_dims": [[4, 2, 2], [1, 1, 1]]},
        }
    }
    path = tmp_path / "scene.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps(rig))
        archive.writestr("sequence_tracks.json", json.dumps(tracks))
        archive.writestr("extra.json", '{"user_data": true}')
    return str(path)
