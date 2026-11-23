"""Autoware 1.9.0 output boundaries and their image launch definitions."""
from collections.abc import Sequence
from dataclasses import dataclass

from avsectester.closed_loop.ros_interception.declaration import (
    InterceptedTopic, InterceptionDeclaration, PoseProjection, private_topic,
)
from avsectester.closed_loop.simulators.messages import require_absolute_topic


INTERCEPTION_VERSION = 3
OPERATION_MODE_TOPIC = "/system/operation_mode/state"

_CARLA = "/_CREATED_BY_BARE_DDS_APP_/_CREATED_BY_BARE_DDS_APP_"
_KIT = "awsim_sensor_kit_launch"
_LOCALIZATION = "tier4_localization_launch"
_FUSION = "pose_twist_fusion_filter/pose_twist_fusion_filter.launch.xml"
_PERCEPTION = "tier4_perception_launch"
_CONTROL = "tier4_control_launch"
_CONTROL_LAUNCH = "control.launch.xml"
_MODE_MANAGER = "autoware_operation_mode_transition_manager"
_GNSS_COVARIANCE = "/sensing/gnss/pose_with_covariance"
_GNSS_POSE = "/sensing/gnss/pose"
_GNSS_SENSOR = "gnss"
_POSE_TOPIC = "/localization/kinematic_state"
_TF_TOPIC = "/tf"
_EKF = "/localization/pose_twist_fusion_filter/ekf_localizer"


@dataclass(frozen=True)
class _Rewrite:
    path: str
    original: str
    replacement: str

    def for_topic(self, topic: str) -> dict[str, str]:
        return {"path": self.path, "topic": topic,
                "original": self.original.replace("@TOPIC@", topic),
                "replacement": self.replacement.replace("@ORIGINAL@", topic).replace(
                    "@TOPIC@", private_topic(topic))}


def _location(package: str, launch_file: str) -> str:
    return f"{package}/share/{package}/launch/{launch_file}"


def _attribute(package: str, launch_file: str, tag: str, name: str, *,
               source: str = "@TOPIC@", attribute: str = "value", tail: str = "/>") -> _Rewrite:
    head = f'<{tag} name="{name}" {attribute}="'
    return _Rewrite(_location(package, launch_file), f'{head}{source}"{tail}',
                    f'{head}@TOPIC@"{tail}')


def _remap(package: str, launch_file: str, anchor: str, *, indent: str) -> _Rewrite:
    target = '<remap from="@ORIGINAL@" to="@TOPIC@"/>'
    return _Rewrite(_location(package, launch_file), anchor,
                    f"{anchor}\n{indent}{target}")


@dataclass(frozen=True)
class _Topic:
    name: str
    type: str
    publishers: tuple[str, ...]
    sensor: str | None = None
    rewrite: _Rewrite | None = None
    argument: str | None = None
    passthrough: bool = False

    def declaration(self) -> InterceptedTopic:
        return InterceptedTopic(self.name, self.type, self.publishers, self.sensor, self.passthrough)


_CLOUD = "sensor_msgs/msg/PointCloud2"
_IMU = "sensor_msgs/msg/Imu"
_TRAJECTORY = "autoware_planning_msgs/msg/Trajectory"

_TOPICS = {
    "sensors": (
        _Topic("/sensing/lidar/top/pointcloud_raw_ex", _CLOUD, (_CARLA,), sensor="lidar_top"),
        _Topic("/sensing/imu/tamagawa/imu_raw", _IMU, (_CARLA,), sensor="imu"),
        _Topic(_GNSS_COVARIANCE, "geometry_msgs/msg/PoseWithCovarianceStamped",
               (_CARLA, "/sensing/gnss_poser"), sensor=_GNSS_SENSOR,
               rewrite=_attribute(_KIT, "sensing.launch.xml", "arg", "output_topic_gnss_pose_cov")),
        _Topic(_GNSS_POSE, "geometry_msgs/msg/Pose", (_CARLA,), sensor=_GNSS_SENSOR, passthrough=True),
    ),
    "sensing": (
        _Topic("/sensing/lidar/concatenated/pointcloud", _CLOUD,
               ("/sensing/lidar/concatenated/pointcloud_relay_ring_to_concat",),
               rewrite=_attribute(_KIT, "lidar.launch.xml", "param", "output_topic")),
        _Topic("/sensing/imu/imu_data", _IMU, ("/sensing/imu/imu_corrector",),
               rewrite=_attribute(_KIT, "imu.launch.xml", "arg", "output_topic", source="imu_data")),
        _Topic("/sensing/vehicle_velocity_converter/twist_with_covariance",
               "geometry_msgs/msg/TwistWithCovarianceStamped", ("/sensing/vehicle_velocity_converter",),
               rewrite=_attribute(_KIT, "sensing.launch.xml", "arg", "output_twist_with_covariance")),
    ),
    "localization": (
        _Topic(_POSE_TOPIC, "nav_msgs/msg/Odometry", ("/localization/pose_twist_fusion_filter/stop_filter",),
               rewrite=_attribute(_LOCALIZATION, _FUSION, "arg", "output_odom_name")),
        _Topic("/localization/acceleration", "geometry_msgs/msg/AccelWithCovarianceStamped",
               ("/localization/pose_twist_fusion_filter/twist2accel",),
               rewrite=_attribute(_LOCALIZATION, _FUSION, "arg", "out_accel")),
    ),
    "perception": (
        _Topic("/perception/object_recognition/objects", "autoware_perception_msgs/msg/PredictedObjects",
               ("/perception/object_recognition/prediction/map_based_prediction",),
               rewrite=_attribute(_PERCEPTION, "object_recognition/prediction/prediction.launch.xml",
                                  "arg", "output_topic")),
        _Topic("/perception/obstacle_segmentation/pointcloud", _CLOUD,
               ("/perception/obstacle_segmentation/occupancy_grid_based_outlier_filter",),
               rewrite=_Rewrite(_location(_PERCEPTION, "obstacle_segmentation/ground_segmentation/ground_segmentation.launch.py"),
                                'self.output_topic = "@TOPIC@"', 'self.output_topic = "@TOPIC@"')),
        _Topic("/perception/occupancy_grid_map/map", "nav_msgs/msg/OccupancyGrid",
               ("/perception/occupancy_grid_map/occupancy_grid_map_node",),
               rewrite=_attribute(_PERCEPTION, "perception.launch.xml", "arg", "output")),
        _Topic("/perception/traffic_light_recognition/traffic_signals",
               "autoware_perception_msgs/msg/TrafficLightGroupArray",
               ("/perception/traffic_light_recognition/crosswalk_traffic_light_estimator",),
               rewrite=_attribute(_PERCEPTION, "traffic_light_recognition/traffic_light.launch.xml",
                                  "arg", "output/traffic_signals", attribute="default",
                                  tail=' description="final output topic as traffic light recognition"/>')),
    ),
    "planning": (
        _Topic("/planning/trajectory", _TRAJECTORY, ("/planning/planning_validator",),
               rewrite=_attribute("tier4_planning_launch", "planning.launch.xml", "arg", "output_trajectory")),
    ),
    "control": (
        _Topic("/control/command/control_cmd", "autoware_control_msgs/msg/Control", ("/control/vehicle_cmd_gate",),
               argument="output_control_cmd"),
        _Topic("/control/trajectory_follower/lateral/predicted_trajectory", _TRAJECTORY,
               ("/control/trajectory_follower/controller_node_exe",),
               rewrite=_Rewrite(_location(_CONTROL, _CONTROL_LAUNCH),
                                '<remap from="~/output/predicted_trajectory" to="lateral/predicted_trajectory"/>',
                                '<remap from="~/output/predicted_trajectory" to="@TOPIC@"/>')),
        _Topic(OPERATION_MODE_TOPIC, "autoware_adapi_v1_msgs/msg/OperationModeState", (f"/control/{_MODE_MANAGER}",),
               rewrite=_remap(_CONTROL, _CONTROL_LAUNCH,
                              f'name="{_MODE_MANAGER}"\n        >', indent="          ")),
    ),
}

_TF_REWRITE = _remap(
    "autoware_ekf_localizer", "ekf_localizer.launch.xml",
    '<node pkg="autoware_ekf_localizer" exec="autoware_ekf_localizer_node" output="both">',
    indent="    ",
)

_TOPIC_BY_NAME = {topic.name: topic for group in _TOPICS.values() for topic in group}
_MODIFIABLE_TOPICS = frozenset(topic.name for topic in _TOPIC_BY_NAME.values() if not topic.passthrough)


def _canonical_topics(names: Sequence[str], option: str) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(f"{option} must be a sequence of topic names")
    for name in names:
        require_absolute_topic(name)
    return tuple(sorted(set(names)))


def validate_topics(
    modified_topics: Sequence[str], observed_topics: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize topic selection. Only modified topics must be in the pinned table."""
    modified = _canonical_topics(modified_topics, "modified_topics")
    observed = _canonical_topics(observed_topics, "observed_topics")
    unknown = sorted(set(modified) - _MODIFIABLE_TOPICS)
    if unknown:
        choices = ", ".join(sorted(_MODIFIABLE_TOPICS))
        raise ValueError(f"Unknown modified topics: {', '.join(unknown)}. Choose from: {choices}")
    return modified, observed


def interception_declaration(
    modified_topics: Sequence[str], observed_topics: Sequence[str] = (),
) -> InterceptionDeclaration | None:
    """Declare the selected stages and any companion routes required by modified outputs."""
    if not modified_topics and not observed_topics:
        return None
    projection = (PoseProjection(_POSE_TOPIC, _EKF, "map", "base_link")
                  if _POSE_TOPIC in modified_topics else None)
    routed = list(modified_topics)
    if _GNSS_COVARIANCE in modified_topics:
        routed.append(_GNSS_POSE)
    intercepted = tuple(_TOPIC_BY_NAME[name].declaration() for name in sorted(routed))
    observed_sensors = tuple((name, sensor) for name in observed_topics
                             if (topic := _TOPIC_BY_NAME.get(name)) is not None
                             and (sensor := topic.sensor) is not None)
    return InterceptionDeclaration(intercepted, projection, tuple(observed_topics), observed_sensors)


def launch_changes(modified_topics: Sequence[str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Return launch changes for the explicitly modified topics and required projection."""
    edits = []
    arguments = {}
    for name in modified_topics:
        topic = _TOPIC_BY_NAME[name]
        if topic.rewrite is not None:
            edits.append(topic.rewrite.for_topic(topic.name))
        if topic.argument is not None:
            arguments[topic.argument] = private_topic(topic.name)
    if _POSE_TOPIC in modified_topics:
        edits.append(_TF_REWRITE.for_topic(_TF_TOPIC))
    return edits, arguments
