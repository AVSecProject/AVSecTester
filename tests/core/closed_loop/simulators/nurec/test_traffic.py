"""CATK session protocol through real protobuf RPC and a deterministic native service."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, TrafficSession, TrafficTrack
from avsectester.geometry.poses import make_pose, rotation_matrix
from tests.fakes import TrafficServiceBoundary


@pytest.fixture
def traffic_service():
    pytest.importorskip("alpasim_grpc.v0.traffic_pb2")
    with TrafficServiceBoundary(scene_ids=("clipgt-scene",)) as service:
        yield service


@pytest.fixture
def traffic_inputs():
    pose = make_pose(rotation_matrix((0., 0., 30.)), (2., 3., 1.))
    points = ((0, pose), (200000, pose))
    return dict(scene_id="clipgt-scene", random_seed=23, start_us=0, end_us=200000,
                ego_extent=(4., 2., 1.5), ego_trajectory=points,
                tracks=(TrafficTrack("vehicle 1", points, (5., 2.5, 2.), "automobile", True),
                        TrafficTrack("host", points, (4., 2., 2.), "automobile", False)))


def test_alpasim_traffic():
    traffic = AlpaSimTraffic()
    assert traffic.endpoint == "127.0.0.1:6200"
    assert traffic.history_steps == 16
    with pytest.raises(ValueError, match="rpc_timeout_s"):
        AlpaSimTraffic(rpc_timeout_s=0.)
    with pytest.raises(ValueError, match="history_steps"):
        AlpaSimTraffic(history_steps=0)


def test_traffic_track():
    track = TrafficTrack("car", ((0, np.eye(4)),), (4., 2., 1.), "automobile", False)
    assert track.track_id == "car" and track.extent == (4., 2., 1.)
    with pytest.raises(ValueError, match="three positive"):
        TrafficTrack("car", track.trajectory, (4., 0., 1.), "automobile", False)


def test_get_metadata(traffic_service):
    import grpc

    traffic = AlpaSimTraffic(traffic_service.endpoint)
    assert traffic.get_metadata().minimum_history_length_us == 1_600_000
    traffic_service.rpc_errors["get_metadata"] = (grpc.StatusCode.UNAVAILABLE, "native service unavailable")
    with pytest.raises(RuntimeError, match="UNAVAILABLE.*native service unavailable"):
        traffic.get_metadata()
    traffic_service.rpc_errors.clear()
    traffic_service.delays["get_metadata"] = .1
    with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
        traffic.get_metadata(timeout_s=.005)


def test_get_available_scenes(traffic_service):
    import grpc

    traffic = AlpaSimTraffic(traffic_service.endpoint)
    assert traffic.get_available_scenes() == ("clipgt-scene",)
    traffic_service.delays["get_available_scenes"] = .1
    with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
        traffic.get_available_scenes(timeout_s=.005)
    traffic_service.delays.clear()
    traffic_service.rpc_errors["get_available_scenes"] = (grpc.StatusCode.PERMISSION_DENIED, "not authorized")
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED.*not authorized"):
        traffic.get_available_scenes()


def test_served_scenes(traffic_service):
    import grpc

    traffic = AlpaSimTraffic(traffic_service.endpoint)
    assert traffic.served_scenes() == ("clipgt-scene",)
    for code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
        traffic_service.rpc_errors["get_available_scenes"] = (code, "endpoint not ready")
        assert traffic.served_scenes() is None
    traffic_service.rpc_errors.clear()
    traffic_service.delays["get_available_scenes"] = .1
    assert traffic.served_scenes(timeout_s=.005) is None
    traffic_service.delays.clear()
    traffic_service.rpc_errors["get_available_scenes"] = (grpc.StatusCode.PERMISSION_DENIED, "not authorized")
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED.*not authorized"):
        traffic.served_scenes()
    with pytest.raises(ValueError, match="timeout_s"):
        traffic.served_scenes(timeout_s=0.)


def test_open_session(traffic_service, traffic_inputs, monkeypatch):
    import grpc
    from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto

    traffic_service.capture_channels(monkeypatch)
    traffic = AlpaSimTraffic(traffic_service.endpoint)
    session = traffic.open_session(**traffic_inputs)
    assert isinstance(session, TrafficSession)
    try:
        request = traffic_service.sessions[-1]
        assert request.session_uuid == session.uuid
        assert request.scene_id == "clipgt-scene" and request.random_seed == 23
        assert request.handover_time_us == session.handover_time_us == 100000
        ego, vehicle, host = request.logged_object_trajectories
        assert ego.object_id == "EGO"
        assert (ego.aabb.size_x, ego.aabb.size_y, ego.aabb.size_z) == (4., 2., 1.5)
        assert [point.timestamp_us for point in ego.trajectory.poses] == [0, 200000]
        point = ego.trajectory.poses[0].pose
        assert (point.vec.x, point.vec.y, point.vec.z) == (2., 3., 1.)
        assert point.quat.w == pytest.approx(np.cos(np.deg2rad(15)))
        assert point.quat.z == pytest.approx(np.sin(np.deg2rad(15)))
        assert vehicle.is_static and vehicle.label_class == "automobile"
        assert not host.is_static
        assert session.track_ids == ("vehicle 1", "host")
    finally:
        session.close()

    pose = np.eye(4)
    final = pose.copy()
    final[0, 3] = 31.
    cropped = TrafficTrack("cropped", ((-100000, pose), (3000000, final)), (4., 2., 1.), "car", False)
    outside = TrafficTrack("outside", ((3000000, pose), (4000000, pose)), (4., 2., 1.), "car", True)
    args = {**traffic_inputs, "start_us": 1500000, "end_us": 2000000, "tracks": (cropped, outside)}
    session = traffic.open_session(**args)
    try:
        assert session.track_ids == ("cropped",)
        assert session.handover_time_us == 1600000
        sent = traffic_service.sessions[-1].logged_object_trajectories[1].trajectory.poses
        assert [point.timestamp_us for point in sent] == [0, 2000000]
        np.testing.assert_allclose(pose_from_proto(sent[0].pose)[:3, 3], [1., 0., 0.])
        np.testing.assert_allclose(pose_from_proto(sent[-1].pose)[:3, 3], [21., 0., 0.])
    finally:
        session.close()
    with pytest.raises(RuntimeError, match="does not serve"):
        traffic.open_session(**{**traffic_inputs, "scene_id": "missing"})
    traffic_service.minimum_history_length_us = 1
    with pytest.raises(RuntimeError, match="minimum_history_length_us=1"):
        traffic.open_session(**traffic_inputs)
    traffic_service.minimum_history_length_us = 1600000
    duplicate = TrafficTrack("bad", ((0, pose), (0, pose)), (4., 2., 1.), "car", False)
    with pytest.raises(ValueError, match="strictly increasing"):
        traffic.open_session(**{**traffic_inputs, "tracks": (duplicate,)})
    traffic_service.rpc_errors["start_session"] = (grpc.StatusCode.INVALID_ARGUMENT, "bad scene")
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT.*bad scene"):
        traffic.open_session(**traffic_inputs)
    assert all(channel.closed for channel in traffic_service.channels)


def test_record_override(traffic_service, traffic_inputs):
    from alpasim_grpc.v0 import traffic_pb2
    from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto

    traffic_service.simulate_response = lambda request: traffic_pb2.TrafficReturn()
    session = AlpaSimTraffic(traffic_service.endpoint).open_session(**traffic_inputs)
    pose = make_pose(rotation_matrix((0., 0., 25.)), (70., 0., 1.))
    velocity = np.array([1., -.5, .2])
    try:
        with pytest.raises(ValueError, match="Unknown"):
            session.record_override("unknown", 1600000, pose, velocity)
        session.record_override("host", 1600000, pose, velocity)
        # A correction at the same instant replaces t0, without a second backfill.
        corrected = pose.copy()
        corrected[0, 3] += 1.
        session.record_override("host", 1600000, corrected, velocity)
        first = corrected.copy()
        first[:3, 3] += velocity * .1
        session.simulate(1700000, np.eye(4), {"host": first})
        track = traffic_service.requests[-1].object_trajectory_updates[1].trajectory.poses
        assert [point.timestamp_us for point in track] == list(range(100000, 1700001, 100000))
        for sample in track[:15]:
            expected = pose.copy()
            expected[:3, 3] += velocity * ((sample.timestamp_us - 1600000) / 1e6)
            np.testing.assert_allclose(pose_from_proto(sample.pose), expected)
        np.testing.assert_allclose(pose_from_proto(track[15].pose), corrected)
        np.testing.assert_allclose(pose_from_proto(track[16].pose), first)
        session.record_override("host", 1700000, first, (0., 0., 0.))
        session.simulate(1800000, np.eye(4), {"host": first})
        updated = traffic_service.requests[-1].object_trajectory_updates[1].trajectory.poses
        assert [point.timestamp_us for point in updated] == list(range(100000, 1800001, 100000))
        np.testing.assert_allclose(pose_from_proto(updated[-1].pose), first)
    finally:
        session.close()


def test_simulate(traffic_service, traffic_inputs, monkeypatch):
    from alpasim_grpc.v0 import common_pb2 as common, traffic_pb2 as traffic
    from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto

    args = {**traffic_inputs, "tracks": (*traffic_inputs["tracks"],
            TrafficTrack("other", traffic_inputs["ego_trajectory"], (4., 2., 1.), "car", False))}
    session = AlpaSimTraffic(traffic_service.endpoint).open_session(**args)
    pose = make_pose(rotation_matrix((0., 0., 90.)), (7., 8., 9.))
    def response(request):
        return traffic.TrafficReturn(object_trajectory_updates=[
            traffic.ObjectTrajectoryUpdate(object_id=identity, trajectory=common.Trajectory(poses=[
                common.PoseAtTime(timestamp_us=request.time_query_us, pose=common.Pose(
                    vec=common.Vec3(x=7., y=8., z=9.), quat=common.Quat(w=np.sqrt(.5), z=np.sqrt(.5))))]))
            for identity in ("host", "vehicle 1", "other", "unknown", "EGO")])
    traffic_service.simulate_response = response
    session.record_override("other", 1600000, pose, (0., 0., 0.))
    try:
        result = session.simulate(1700000, pose, {"other": pose})
        assert tuple(result) == ("host",)
        np.testing.assert_allclose(result["host"], pose, atol=1e-6)
        request = traffic_service.requests[-1]
        assert request.time_query_us == 1700000 and request.session_uuid == session.uuid
        assert [item.object_id for item in request.object_trajectory_updates] == ["EGO", "other"]
        ego, other = request.object_trajectory_updates
        assert [point.timestamp_us for point in ego.trajectory.poses] == [1700000]
        np.testing.assert_allclose(pose_from_proto(ego.trajectory.poses[0].pose), pose, atol=1e-12)
        assert [point.timestamp_us for point in other.trajectory.poses] == list(range(100000, 1700001, 100000))
    finally:
        session.close()

    import grpc

    traffic_service.capture_channels(monkeypatch)
    client = AlpaSimTraffic(traffic_service.endpoint)
    session = client.open_session(**traffic_inputs)
    with pytest.raises(ValueError, match="override keys"):
        session.simulate(200000, np.eye(4), {"host": np.eye(4)})
    assert traffic_service.closed[-1].session_uuid == session.uuid
    assert all(channel.closed for channel in traffic_service.channels)
    session = client.open_session(**traffic_inputs)
    traffic_service.simulate_response = lambda request: traffic.TrafficReturn(
        object_trajectory_updates=[traffic.ObjectTrajectoryUpdate(object_id="host",
            trajectory=common.Trajectory(poses=[common.PoseAtTime(timestamp_us=request.time_query_us+1,
                pose=common.Pose(quat=common.Quat(w=1.)))]))])
    with pytest.raises(RuntimeError, match="has no pose at"):
        session.simulate(200000, np.eye(4), {})
    assert traffic_service.closed[-1].session_uuid == session.uuid
    assert all(channel.closed for channel in traffic_service.channels)
    session = client.open_session(**traffic_inputs)
    traffic_service.rpc_errors["simulate"] = (grpc.StatusCode.UNAVAILABLE, "traffic unavailable")
    with pytest.raises(RuntimeError, match="UNAVAILABLE.*traffic unavailable"):
        session.simulate(200000, np.eye(4), {})
    assert traffic_service.closed[-1].session_uuid == session.uuid
    session = client.open_session(**traffic_inputs)
    traffic_service.rpc_errors["simulate"] = (grpc.StatusCode.INTERNAL, "predictor failed")
    traffic_service.rpc_errors["close_session"] = (grpc.StatusCode.UNAVAILABLE, "cleanup failed")
    with pytest.raises(RuntimeError, match="INTERNAL.*predictor failed") as caught:
        session.simulate(200000, np.eye(4), {})
    assert "cleanup failed" in caught.value.__notes__[0]
    assert all(channel.closed for channel in traffic_service.channels)


def test_close(traffic_service, traffic_inputs, monkeypatch):
    import grpc

    traffic_service.capture_channels(monkeypatch)
    traffic = AlpaSimTraffic(traffic_service.endpoint)
    session = traffic.open_session(**traffic_inputs)
    session.close()
    session.close()
    assert traffic_service.closed[-1].session_uuid == session.uuid
    with pytest.raises(RuntimeError, match="not open"):
        session.simulate(100000, np.eye(4), {})
    session = traffic.open_session(**traffic_inputs)
    traffic_service.rpc_errors["close_session"] = (grpc.StatusCode.UNAVAILABLE, "disconnected")
    with pytest.raises(RuntimeError, match="UNAVAILABLE.*disconnected"):
        session.close()
    session.close()
    with pytest.raises(RuntimeError, match="not open"):
        session.simulate(100000, np.eye(4), {})
    assert all(channel.closed for channel in traffic_service.channels)
