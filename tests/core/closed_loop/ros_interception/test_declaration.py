"""ROS stages, forwarding and observation declarations."""

import pytest

from avsectester.closed_loop.ros_interception.declaration import (
    InterceptedTopic, InterceptionDeclaration, PoseProjection,
)
from avsectester.closed_loop.simulators.messages import MessageStage


def test_interception_declaration():
    pose = InterceptedTopic("/pose", "nav_msgs/msg/Odometry", ("/ekf",), "gnss")
    companion = InterceptedTopic("/companion", "geometry_msgs/msg/Pose", ("/carla",), passthrough=True)
    projection = PoseProjection(pose.name, "/ekf", "map", "base_link")
    declaration = InterceptionDeclaration(
        (pose, companion), projection, observed_topics=(companion.name, "/clock"),
        observed_sensors=((companion.name, "gnss"),))
    assert declaration.stages == {
        "/pose": MessageStage(sensor="gnss"),
        "/clock": MessageStage(modifiable=False),
        "/companion": MessageStage(sensor="gnss", modifiable=False),
    }
    forwarding, observation = declaration.relay_configurations(startup_timeout_s=2.)
    assert forwarding["role"] == "forwarding" and observation["role"] == "observation"
    assert [item["name"] for item in forwarding["topics"]] == [pose.name, companion.name]
    assert forwarding["topics"][1]["passthrough"] is True
    assert forwarding["observed_topics"] == [] and forwarding["projection"]["pose_topic"] == pose.name
    assert observation["topics"] == [] and observation["projection"] is None
    assert observation["observed_topics"] == [companion.name, "/clock"]
    assert forwarding["startup_timeout_s"] == observation["startup_timeout_s"] == 2.
    assert InterceptionDeclaration((companion,)).stages == {}
    with pytest.raises(ValueError, match="cannot be both modified and observed: /pose"):
        InterceptionDeclaration((pose,), observed_topics=(pose.name,))
