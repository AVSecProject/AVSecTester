import pytest

from avsectester.closed_loop.simulators.messages import require_absolute_topic


def test_absolute_topic_name():
    for valid in ("/scan", "/carla/_ego/imu_42"):
        require_absolute_topic(valid)
    for invalid in ("scan", "/", "/carla//imu", "/carla/3imu", "/imu/", "/imu-raw"):
        with pytest.raises(ValueError, match="absolute ROS topic name"):
            require_absolute_topic(invalid)
