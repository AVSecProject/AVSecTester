"""ROS field conversion through ordinary and structured payloads."""

from array import array

import numpy as np
import pytest

from avsectester.closed_loop.ros_interception.ros_messages import (
    OCCUPANCY_GRID, POINT_CLOUD, to_fields, to_message,
)


def test_point_cloud_fields_preserve_layout_and_recompute_modified_dimensions():
    sensors = pytest.importorskip("sensor_msgs.msg")
    layout = [sensors.PointField(name="x", offset=0, datatype=7, count=1),
              sensors.PointField(name="ring", offset=4, datatype=4, count=2)]
    data = bytearray(36)
    source = np.ndarray((2, 2), dtype=np.dtype([("x", ">f4"), ("ring", ">u2", 2)]),
                        buffer=data, strides=(20, 8))
    source["x"] = [[10., 20.], [30., 40.]]
    source["ring"] = (1, 2)
    cloud = sensors.PointCloud2(height=2, width=2, point_step=8, row_step=20, is_bigendian=True,
                                fields=layout, data=array("B", data))
    fields = to_fields(cloud, POINT_CLOUD)
    np.testing.assert_equal(fields["data"], source)

    fields["data"] = fields["data"][:, 1]
    new = to_message(fields, sensors.PointCloud2, POINT_CLOUD)
    assert (new.height, new.width, new.point_step, new.row_step) == (1, 2, 8, 16)
    assert new.fields == layout
    np.testing.assert_equal(np.frombuffer(new.data, dtype=source.dtype), source[:, 1])


def test_occupancy_grid_fields_preserve_int8_and_map_dimensions():
    navigation = pytest.importorskip("nav_msgs.msg")
    source = navigation.OccupancyGrid(info=navigation.MapMetaData(height=2, width=2),
                                      data=array("b", [-1, 0, 1, 100]))
    fields = to_fields(source, OCCUPANCY_GRID)
    assert fields["data"].dtype == np.int8
    edited = fields["data"].reshape(2, 2)
    edited[1, 1] = 75
    new = to_message({**fields, "data": edited}, navigation.OccupancyGrid, OCCUPANCY_GRID)
    np.testing.assert_equal(np.asarray(new.data), np.array([-1, 0, 1, 75], dtype=np.int8))
    with pytest.raises(ValueError, match="matching its height and width"):
        to_message({**fields, "data": edited[0]}, navigation.OccupancyGrid, OCCUPANCY_GRID)


def test_fixed_numeric_message_sequence_round_trips_trajectory():
    planning = pytest.importorskip("autoware_planning_msgs.msg")
    serialization = pytest.importorskip("rclpy.serialization")
    type_name = "autoware_planning_msgs/msg/Trajectory"
    trajectory = planning.Trajectory()
    trajectory.header.frame_id = "map"
    trajectory.points = [planning.TrajectoryPoint(longitudinal_velocity_mps=2.0) for _ in range(3)]
    trajectory.points[1].pose.position.y = -1.5
    trajectory.points[2].time_from_start.sec = 1

    fields = to_fields(trajectory, type_name)
    points = fields["points"]
    assert points.shape == (3,)
    assert points["pose"]["position"]["y"].tolist() == [0., -1.5, 0.]
    assert to_fields(planning.Trajectory(), type_name)["points"].dtype == points.dtype

    replacement = to_message(fields, planning.Trajectory, type_name)
    assert serialization.serialize_message(replacement) == serialization.serialize_message(trajectory)
    points["longitudinal_velocity_mps"] = 0.
    replacement = to_message(fields, planning.Trajectory, type_name)
    decoded = serialization.deserialize_message(serialization.serialize_message(replacement),
                                               planning.Trajectory)
    assert [point.longitudinal_velocity_mps for point in decoded.points] == [0.] * 3
    assert decoded.points[1].pose.position.y == -1.5

    for invalid in ([{"longitudinal_velocity_mps": 0.}], np.zeros(3, dtype=np.float32),
                    points.reshape(3, 1)):
        with pytest.raises(ValueError, match=r"\.points requires a one-dimensional array with dtype"):
            to_message({**fields, "points": invalid}, planning.Trajectory, type_name)
    with pytest.raises(ValueError, match=r"Unknown ROS field .*\.header\.other"):
        to_message({**fields, "header": {"other": 5}}, planning.Trajectory, type_name)


def test_variable_message_members_keep_wrappers_around_fixed_records():
    perception = pytest.importorskip("autoware_perception_msgs.msg")
    geometry = pytest.importorskip("geometry_msgs.msg")
    serialization = pytest.importorskip("rclpy.serialization")
    type_name = "autoware_perception_msgs/msg/PredictedObjects"
    path = perception.PredictedPath(path=[
        geometry.Pose(position=geometry.Point(x=1.)),
        geometry.Pose(position=geometry.Point(x=3., y=2.)),
    ])
    kinematics = perception.PredictedObjectKinematics(predicted_paths=[path])
    objects = perception.PredictedObjects(objects=[
        perception.PredictedObject(existence_probability=.5, kinematics=kinematics)])
    fields = to_fields(objects, type_name)

    assert isinstance(fields["objects"], list)
    item, = fields["objects"]
    assert item["existence_probability"] == .5
    trajectory, = item["kinematics"]["predicted_paths"]
    points = trajectory["path"]
    assert points["position"]["x"].tolist() == [1., 3.]
    replacement = to_message(fields, perception.PredictedObjects, type_name)
    assert serialization.serialize_message(replacement) == serialization.serialize_message(objects)

    points["position"]["x"] += 5.
    modified = to_message(fields, perception.PredictedObjects, type_name)
    decoded = serialization.deserialize_message(serialization.serialize_message(modified),
                                               perception.PredictedObjects)
    result = decoded.objects[0].kinematics.predicted_paths[0].path
    assert [(pose.position.x, pose.position.y) for pose in result] == [(6., 0.), (8., 2.)]
    with pytest.raises(ValueError, match=r"\.objects requires a sequence of mappings"):
        to_message({**fields, "objects": [None]}, perception.PredictedObjects, type_name)


def test_byte_bearing_messages_keep_mapping_records():
    diagnostics = pytest.importorskip("tier4_system_msgs.msg")
    serialization = pytest.importorskip("rclpy.serialization")
    type_name = "tier4_system_msgs/msg/DiagGraphStatus"
    graph = diagnostics.DiagGraphStatus(nodes=[
        diagnostics.DiagNodeStatus(level=b"\x01", input_level=b"\x02", latch_level=b"\x03",
                                   is_dependent=True),
    ])
    fields = to_fields(graph, type_name)
    assert isinstance(fields["nodes"], list)
    assert fields["nodes"][0]["level"] == b"\x01"
    restored = to_message(fields, diagnostics.DiagGraphStatus, type_name)
    assert serialization.serialize_message(restored) == serialization.serialize_message(graph)


def test_boolean_and_string_arrays_follow_declared_types():
    parameters = pytest.importorskip("rcl_interfaces.msg")
    serialization = pytest.importorskip("rclpy.serialization")
    type_name = "rcl_interfaces/msg/ParameterValue"
    for values in ([], [True, False, True]):
        parameter = parameters.ParameterValue(bool_array_value=values, string_array_value=["x"])
        fields = to_fields(parameter, type_name)
        assert fields["bool_array_value"].dtype == np.bool_
        assert fields["bool_array_value"].tolist() == values
        assert fields["string_array_value"] == ["x"]
        restored = to_message(fields, parameters.ParameterValue, type_name)
        assert restored.bool_array_value == values
        assert serialization.serialize_message(restored) == serialization.serialize_message(parameter)


def test_fixed_array_fields_pack_inside_structured_records():
    shapes = pytest.importorskip("shape_msgs.msg")
    geometry = pytest.importorskip("geometry_msgs.msg")
    serialization = pytest.importorskip("rclpy.serialization")
    type_name = "shape_msgs/msg/Mesh"
    mesh = shapes.Mesh(triangles=[shapes.MeshTriangle(vertex_indices=[1, 2, 0])],
                       vertices=[geometry.Point(x=1.), geometry.Point(y=2.), geometry.Point(z=3.)])
    fields = to_fields(mesh, type_name)
    assert fields["triangles"]["vertex_indices"].shape == (1, 3)
    np.testing.assert_array_equal(fields["triangles"]["vertex_indices"], [[1, 2, 0]])
    restored = to_message(fields, shapes.Mesh, type_name)
    assert serialization.serialize_message(restored) == serialization.serialize_message(mesh)
