"""Bridge JSON lines on standard input and output to Autoware's ROS 2 interfaces.

The host owns sequencing, deadlines and readiness checks.
"""
import json
import math
from queue import Empty, Queue
import sys
from threading import Thread
from typing import Any

from autoware_adapi_v1_msgs.msg import (
    LocalizationInitializationState,
    MrmState,
    OperationModeState,
    RouteState,
)
from autoware_adapi_v1_msgs.srv import ChangeOperationMode, InitializeLocalization
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def _enum(value: int | bytes) -> int:
    return value if isinstance(value, int) else ord(value)


def _read_commands(inbox: Queue[dict[str, Any] | None]) -> None:
    try:
        for line in sys.stdin:
            inbox.put(json.loads(line))
    finally:
        inbox.put(None)


class _Adapter(Node):
    def __init__(self) -> None:
        super().__init__("avsectester_autoware_adapter")
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OperationModeState, "/api/operation_mode/state", self._mode,
                                 state_qos)
        self.create_subscription(LocalizationInitializationState,
                                 "/api/localization/initialization_state", self._localization,
                                 state_qos)
        self.create_subscription(RouteState, "/api/routing/state", self._route, state_qos)
        self.create_subscription(MrmState, "/api/fail_safe/mrm_state", self._mrm, state_qos)
        self.create_subscription(Odometry, "/localization/kinematic_state", self._pose,
                                 qos_profile_sensor_data)
        self.create_subscription(DiagnosticArray, "/diagnostics", self._diagnostics, 10)
        self._lidar = self.create_subscription(PointCloud2, "/sensing/lidar/top/pointcloud_raw_ex",
                                               self._pointcloud, qos_profile_sensor_data, raw=True)
        self._lidar_seen = False
        self._goal = self.create_publisher(PoseStamped, "/planning/mission_planning/goal", 10)
        self._initialize = self.create_client(InitializeLocalization, "/api/localization/initialize")
        self._autonomous = self.create_client(ChangeOperationMode,
                                              "/api/operation_mode/change_to_autonomous")
        self._discovery = None

    def _mode(self, message: OperationModeState) -> None:
        _emit("mode", mode=_enum(message.mode),
              available=bool(message.is_autonomous_mode_available),
              in_transition=bool(message.is_in_transition))

    def _localization(self, message: LocalizationInitializationState) -> None:
        _emit("localization", state=_enum(message.state))

    def _route(self, message: RouteState) -> None:
        _emit("route", state=_enum(message.state))

    def _mrm(self, message: MrmState) -> None:
        _emit("mrm", state=_enum(message.state), behavior=_enum(message.behavior))

    def _pose(self, message: Odometry) -> None:
        pose = message.pose.pose
        rotation = pose.orientation
        # The official gate derives planar yaw from the quaternion's z and w.
        yaw = math.atan2(2 * rotation.w * rotation.z, 1 - 2 * rotation.z ** 2)
        if all(math.isfinite(value) for value in (pose.position.x, pose.position.y, yaw)):
            _emit("pose", x=pose.position.x, y=pose.position.y, yaw_rad=yaw)

    def _diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            level = _enum(status.level)
            if level >= 2 and "distortion" in status.name:
                _emit("distortion", name=status.name, level=level,
                      message=status.message)

    def _pointcloud(self, message: bytes) -> None:
        if not self._lidar_seen:
            self._lidar_seen = True
            _emit("lidar")

    def discovery(self) -> None:
        if self._lidar is not None and self._lidar_seen:
            self.destroy_subscription(self._lidar)
            self._lidar = None
        signature = (self._initialize.service_is_ready(), self._autonomous.service_is_ready(),
                     bool(self._goal.get_subscription_count()))
        if signature != self._discovery:
            self._discovery = signature
            _emit("discovery", initialize=signature[0], autonomous=signature[1],
                  goal_subscriber=signature[2])

    def command(self, message: dict[str, Any]) -> None:
        request_id, call = message["id"], message["call"]
        if call == "goal":
            if not self._goal.get_subscription_count():
                _emit("result", id=request_id, success=False, message="Goal subscriber disconnected")
                return
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = message["args"]["x"]
            goal.pose.position.y = message["args"]["y"]
            heading = math.radians(message["args"]["yaw_deg"])
            goal.pose.orientation.z = round(math.sin(heading / 2), 6)
            goal.pose.orientation.w = round(math.cos(heading / 2), 6)
            self._goal.publish(goal)
            _emit("result", id=request_id, success=True)
            return
        if call == "initialize":
            client, request = self._initialize, InitializeLocalization.Request()
        elif call == "engage":
            client, request = self._autonomous, ChangeOperationMode.Request()
        else:
            raise ValueError(f"Unknown adapter call: {call}")
        if not client.service_is_ready():
            _emit("result", id=request_id, success=False, message=f"{call} service unavailable")
            return
        future = client.call_async(request)

        def complete(done: Any) -> None:
            try:
                status = done.result().status
            except Exception as error:
                _emit("result", id=request_id, success=False, message=str(error))
            else:
                _emit("result", id=request_id, success=bool(status.success),
                      code=int(status.code), message=status.message)

        future.add_done_callback(complete)


def main() -> None:
    rclpy.init()
    node = _Adapter()
    inbox: Queue[dict[str, Any] | None] = Queue()
    Thread(target=_read_commands, args=(inbox,), daemon=True).start()
    _emit("connected")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.discovery()
            try:
                command = inbox.get_nowait()
            except Empty:
                continue
            if command is None:
                break
            node.command(command)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
