"""First-party test data and explicit opt-in for simulator execution."""

from pathlib import Path

import pytest


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
