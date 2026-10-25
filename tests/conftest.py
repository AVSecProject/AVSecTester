"""First-party test data and explicit opt-in for simulator execution."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from avsectester.datasets.nurec import NuRecDataset


def pytest_addoption(parser):
    parser.addoption(
        "--run-carla", action="store_true", help="Run tests against a live CARLA server"
    )
    parser.addoption(
        "--carla-config",
        default=str(Path(__file__).resolve().parents[1] / "configs/carla_phantom.yaml"),
        help="Scenario YAML for the live CARLA test",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-carla"):
        skip = pytest.mark.skip(reason="Live simulator test: opt in with --run-carla")
        for item in items:
            if item.get_closest_marker("carla") is not None:
                item.add_marker(skip)


@pytest.fixture
def make_detections():
    import numpy as np
    from avstack.datastructs import DataContainer
    from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position
    from avstack.modules.perception.detections import BoxDetection

    def make(xyzs=(), *, reference=GlobalOrigin3D, frame=0):
        detections = [
            BoxDetection(
                data=Box3D(
                    Position(np.array(xyz, dtype=float), reference),
                    Attitude(np.quaternion(1), reference),
                    [1.5, 1.8, 4.0],
                    where_is_t="bottom",
                ),
                noise=np.ones(6),
                source_identifier="test-lidar",
                reference=reference,
                obj_type="Car",
                score=0.8,
            )
            for xyz in xyzs
        ]
        return DataContainer(
            frame, frame * 0.05, detections, "test-lidar", source_reference=reference
        )

    return make


@pytest.fixture
def make_ego():
    import numpy as np
    from avstack.environment.objects import VehicleState
    from avstack.geometry import Attitude, Box3D, GlobalOrigin3D, Position, Velocity
    from avstack.geometry import transformations as tforms

    def make(*, xyz=(0, 0, 0), yaw=0.0, speed=5.0, t=0.0):
        ref = GlobalOrigin3D
        position = Position(np.array(xyz, dtype=float), ref)
        attitude = Attitude(tforms.transform_orientation([0, 0, yaw], "euler", "quat"), ref)
        velocity = Velocity(speed * np.array([np.cos(yaw), np.sin(yaw), 0.0]), ref)
        ego = VehicleState(obj_type="car", ID=0)
        ego.set(
            t,
            position,
            Box3D(position, attitude, [1.5, 1.8, 4.0]),
            velocity=velocity,
            attitude=attitude,
        )
        return ego

    return make


@pytest.fixture
def pipeline_config():
    return dict(
        type="ModularDrivingPipeline",
        perception=dict(type="Passthrough3DObjectDetector"),
        tracking=dict(type="BasicBoxTracker3D"),
        planning=dict(type="ForwardCollisionPlanner", target_speed=6.0),
        control=dict(
            type="VehiclePIDController",
            args_lateral=dict(K_P=1.0, K_D=0.0, K_I=0.0),
            args_longitudinal=dict(K_P=0.5, K_D=0.0, K_I=0.0),
        ),
    )


@pytest.fixture
def native_observation(make_ego, make_detections):
    """Construct an actual avstack capture with an explicit sensor-to-body mounting frame."""
    def make(frame=0, xyzs=((20., 0., 0.),), *, ego=None, mount=(0., 0., 0.)):
        import numpy as np
        from avstack.geometry import ReferenceFrame
        from avsectester.data_types import Observation

        ego = make_ego(t=frame * .05) if ego is None else ego
        sensor = ReferenceFrame(np.asarray(mount, dtype=float), np.quaternion(1), ego.as_reference())
        return Observation(frame * .05, frame, {
            "lidar": make_detections(xyzs, reference=sensor, frame=frame),
        }, vehicle_state=ego, ego_speed=float(np.linalg.norm(ego.velocity.x)))

    return make


@pytest.fixture
def nurec_usdz_path(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    camera = "camera_front_wide_120fov@scene"
    transforms = []
    for position in range(3):
        pose = np.eye(4)
        pose[:3, 3] = [position * .5, 0., .3]
        transforms.append([pose.tolist(), pose.tolist()])
    rig = {
        "camera_calibrations": {camera: {
            "camera_model": {"parameters": {"resolution": [16, 12],
                "principal_point": [8, 6], "angle_to_pixeldist_poly": [0, 8],
                "pixeldist_to_angle_poly": [0, .125], "max_angle": 1.5}},
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
    def points(coordinates):
        return [dict(zip(("x", "y", "z"), point)) for point in coordinates]

    def lane(center_y, vehicle_types, use_types=(), speed_limit="25"):
        return {"left_rail": points([(-2, center_y + 2, 0), (4, center_y + 2, 0)]),
                "right_rail": points([(-2, center_y - 2, 0), (4, center_y - 2, 0)]),
                "speed_limit": speed_limit, "vehicle_types": vehicle_types, "use_types": use_types}

    def control(category, y):
        return {"center": {"x": 3., "y": y, "z": 1.5},
                "dimensions": {"x": .1, "y": .8, "z": 1.2},
                "orientation": {"x": 0., "y": 0., "z": 1., "w": 0.}, "category": category}

    layers = {
        "lane": [("lane", lane(0, ["CAR", "BICYCLE"])),
                 ("signal-lane", lane(6, ["CAR"])), ("bike-lane", lane(-6, ["BICYCLE"], speed_limit="0")),
                 ("ambiguous-lane", lane(12, ["CAR"])),
                 ("shoulder-lane", lane(18, ["CAR"], ["SHOULDER_LANE"])),
                 ("parking-lane", lane(-12, ["CAR"], ["PARKING_LANE"])),
                 ("bus-lane", lane(-18, ["BUS"]))],
        "traffic_sign": [("stop", control("TRAFFIC_SIGN_REGULATORY_R1_STOP", 2.)),
            ("no-stopping", control("TRAFFIC_SIGN_REGULATORY_R8_NO_STOPPING_ANYTIME", -10.)),
            ("yield", control("TRAFFIC_SIGN_REGULATORY_R1_YIELD", -4.)),
            ("speed-limit", control("TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT", 2.))],
        "traffic_light": [("light", control("traffic_light", 8.)),
                          ("second-light", control("traffic_light", 14.))],
        "wait_line": [("stop-line", {"category": "STOP", "is_implicit": False,
            "location": points([(3, -2, 0), (3, 2, 0)])}),
            ("signal-line", {"category": "STOP", "is_implicit": False,
                             "location": points([(3, 4, 0), (3, 8, 0)])}),
            ("yield-line", {"category": "STOP", "is_implicit": False,
                            "location": points([(3, -8, 0), (3, -4, 0)])}),
            ("ambiguous-line", {"category": "STOP", "is_implicit": False,
                                "location": points([(3, 10, 0), (3, 14, 0)])}),
            ("unknown-line", {"category": "STOP", "is_implicit": False,
                              "location": points([(3, -14, 0), (3, -10, 0)])}),
            ("unbound-line", {"category": "STOP", "is_implicit": False,
                              "location": points([(2, -14, 0), (2, -10, 0)])}),
            ("implicit-line", {"category": "STOP", "is_implicit": True,
                               "location": points([(2, -2, 0), (2, 2, 0)])}),
            ("unspecified-line", {"category": "STOP", "is_implicit": None,
                                  "location": points([(1, -2, 0), (1, 2, 0)])})],
        "drivable_space": [(None, {"location": None})],
    }
    associations = [
        {"key": {"map_id": "lane", "kind": "SIGN_TO_LANE"},
         "association": {"subjects": ["lane"], "objects": ["stop"]}},
        {"key": {"map_id": "lane", "kind": "WAIT_LINE_TO_LANE"},
         "association": {"subjects": ["lane"], "objects": ["stop-line"]}},
        {"key": {"map_id": "stop-line", "kind": "WAIT_LINE_TO_SIGN"},
         "association": {"subjects": ["stop-line"], "objects": ["stop"]}},
        {"key": {"map_id": "lane", "kind": "RIGHT_LANE"},
         "association": {"subjects": ["lane"], "objects": ["-1"]}},
    ]
    for kind, lane_id, other_id in (
        ("SIGN_TO_LANE", "parking-lane", "no-stopping"), ("SIGN_TO_LANE", "lane", "speed-limit"),
        ("SIGN_TO_LANE", "bike-lane", "yield"), ("LIGHT_TO_LANE", "signal-lane", "light"),
        ("LIGHT_TO_LANE", "ambiguous-lane", "light"),
        ("LIGHT_TO_LANE", "ambiguous-lane", "second-light"),
        ("WAIT_LINE_TO_LANE", "signal-lane", "signal-line"),
        ("WAIT_LINE_TO_LANE", "bike-lane", "yield-line"),
        ("WAIT_LINE_TO_LANE", "ambiguous-lane", "ambiguous-line"),
        ("WAIT_LINE_TO_LANE", "parking-lane", "unknown-line"),
        ("WAIT_LINE_TO_LANE", "lane", "implicit-line"),
        ("WAIT_LINE_TO_LANE", "lane", "unspecified-line"),
    ):
        associations.append({"key": {"map_id": lane_id, "kind": kind},
                             "association": {"subjects": [lane_id], "objects": [other_id]}})

    def mesh_ply(vertices, faces):
        header = (f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        return header + "\n".join(" ".join(map(str, point)) for point in vertices) + "\n" + "\n".join(
            "3 " + " ".join(map(str, face)) for face in faces) + "\n"

    cube_vertices = [(4, -1, 0), (6, -1, 0), (6, 1, 0), (4, 1, 0),
                     (4, -1, 2), (6, -1, 2), (6, 1, 2), (4, 1, 2)]
    cube_faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
                  (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                  (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    ground_vertices = [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rig_trajectories.json", json.dumps(rig))
        archive.writestr("sequence_tracks.json", json.dumps(tracks))
        archive.writestr("mesh.ply", mesh_ply(cube_vertices, cube_faces))
        archive.writestr("mesh_ground.ply", mesh_ply(ground_vertices, [(0, 1, 2), (0, 2, 3)]))
        archive.writestr("map.xodr", '<OpenDRIVE><road><type><speed max="25" unit="mph"/>'
                        '</type></road></OpenDRIVE>')
        for layer, values in layers.items():
            rows = [{"key": {"clip_id": "scene", "label_class_id": "test", "map_id": identity},
                     layer: data, "version": 0} for identity, data in values]
            output = arrow.BufferOutputStream()
            parquet.write_table(arrow.Table.from_pylist(rows), output)
            archive.writestr(f"clipgt/{layer}.parquet", output.getvalue().to_pybytes())
        output = arrow.BufferOutputStream()
        parquet.write_table(arrow.Table.from_pylist(associations), output)
        archive.writestr("clipgt/association.parquet", output.getvalue().to_pybytes())
    return str(path)


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
