"""Own an Autoware process group and run ADAPI readiness checks for each CARLA run."""
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import selectors
import subprocess
import sys
from threading import Event, Thread
import time
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np

from avsectester.closed_loop.docker_containers import launch_container, stop_container
from avsectester.closed_loop.simulators.carla.external_driving_system import (
    ExternalDrivingContext,
    ExternalDrivingSession,
)
from avsectester.validation import (
    require_nonnegative_integer,
    require_nonnegative_number,
    require_positive_number,
)

from .container.apply_carla_overrides import OVERRIDE_VERSION


_DEFAULT_IMAGE = ("ghcr.io/autowarefoundation/autoware@sha256:"
                  "1db2988caa5dc5bae41548a8d556d0b57f55eb4dd63685426d94fd9c7a786034")
_DEFAULT_LAUNCH = {
    "vehicle_model": "sample_vehicle",
    "sensor_model": "awsim_sensor_kit",
    "perception_mode": "lidar",
}
_FIXED_LAUNCH = {"simulator_type": "carla", "launch_simulator_interface": "false", "rviz": "false"}
_MOUNTED_ADAPTER = "/avsectester-adapter"
_MOUNTED_RUN = "/avsectester-run"
_MOUNTED_PUBLISHER = "/avsectester-publisher.py"
_SETUP = (
    'for s in /opt/autoware/setup.bash /opt/ros/*/setup.bash; do '
    '[ -f "$s" ] && source "$s" && break; done; exec "$@"'
)
_DOCKER_TIMEOUT_S = 10.0
_LIDAR_TIMEOUT_S = 20.0
_LOCALIZATION_ATTEMPTS = 6
_LOCALIZATION_RETRY_S = 8.0
# A goal published as DDS discovery completes can be lost. Republishing before engagement
# only replaces the route with the same one.
_GOAL_RETRY_S = 5.0
_REAR_AXLE_OFFSET_M = 1.425
_LOCALIZATION_INITIALIZED = 3
_ROUTE_SET = 2
_ROUTE_ARRIVED = 3
_MODE_AUTONOMOUS = 2


class AutowareDrivingSystem:
    """Host a local Autoware image as a separate container for each CARLA run.

    The assets, DDS and work directories must be visible under the same paths to the
    experiment and Docker host. The writable model directory is a Docker host path.
    """

    def __init__(
        self, *, gpu: int, ros_domain_id: int, autoware_data_directory: str | Path,
        assets_directory: str | Path, dds_directory: str | Path, work_directory: str | Path,
        image: str = _DEFAULT_IMAGE,
        launch_arguments: Mapping[str, str | int | float | bool] | None = None,
        startup_timeout_s: float = 300.0, engage_timeout_s: float = 10.0,
        localization_tolerance_m: float = 0.5, localization_tolerance_deg: float = 5.0,
        distortion_check_s: float = 12.0,
    ) -> None:
        """Configure the Autoware image, host mounts, launch arguments and readiness checks."""
        require_nonnegative_integer(gpu, "gpu")
        require_nonnegative_integer(ros_domain_id, "ros_domain_id")
        if ros_domain_id > 232:
            raise ValueError("ros_domain_id must be between 0 and 232")
        require_positive_number(startup_timeout_s, "startup_timeout_s")
        require_positive_number(engage_timeout_s, "engage_timeout_s")
        require_nonnegative_number(localization_tolerance_m, "localization_tolerance_m")
        require_nonnegative_number(localization_tolerance_deg, "localization_tolerance_deg")
        require_nonnegative_number(distortion_check_s, "distortion_check_s")
        if not isinstance(image, str) or not image.strip() or image.startswith("-"):
            raise ValueError("image must be a nonempty Docker image reference")
        arguments = dict(_DEFAULT_LAUNCH)
        for key, value in (launch_arguments or {}).items():
            if key in _FIXED_LAUNCH or key == "map_path":
                raise ValueError(f"Autoware launch argument {key!r} is fixed")
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid Autoware launch argument: {key!r}")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Autoware launch argument {key!r} must be a scalar")
            arguments[key] = str(value).lower() if isinstance(value, bool) else str(value)
        self.image, self.gpu, self.ros_domain_id = image, gpu, ros_domain_id
        # The daemon, rather than the experiment container, resolves the host-only model mount.
        self.autoware_data_directory = Path(autoware_data_directory).absolute()
        self.assets_directory = Path(assets_directory).resolve()
        self.dds_directory = Path(dds_directory).resolve()
        self.work_directory = Path(work_directory).resolve()
        self.launch_arguments = MappingProxyType(arguments)
        self.startup_timeout_s, self.engage_timeout_s = startup_timeout_s, engage_timeout_s
        self.localization_tolerance_m = localization_tolerance_m
        self.localization_tolerance_deg = localization_tolerance_deg
        self.distortion_check_s = distortion_check_s

    @property
    def identity(self) -> dict[str, Any]:
        """Return result-affecting Autoware settings for preparation compatibility."""
        return {
            "image": self.image,
            "launch_arguments": dict(self.launch_arguments),
            "override_version": OVERRIDE_VERSION,
            "startup_timeout_s": self.startup_timeout_s,
            "engage_timeout_s": self.engage_timeout_s,
            "localization_tolerance_m": self.localization_tolerance_m,
            "localization_tolerance_deg": self.localization_tolerance_deg,
            "distortion_check_s": self.distortion_check_s,
        }

    def start(self, context: ExternalDrivingContext) -> ExternalDrivingSession:
        """Start one Autoware session asynchronously and return its readiness interface."""
        town = context.map_name.rsplit("/", 1)[-1].removesuffix("_Opt")
        if town != "Town10HD":
            raise ValueError(f"Autoware has no map for {town!r}. "
                             "Generate the Autoware map with CARLA map_tools first.")
        return _AutowareSession(self, context, town)


def _exec(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *arguments], text=True, capture_output=True, check=False,
                          timeout=_DOCKER_TIMEOUT_S)


def _require_success(result: subprocess.CompletedProcess[str], message: str) -> None:
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{message}: {detail or 'Docker command failed'}")


def _mount(source: Path, destination: str, *, writable: bool = False) -> list[str]:
    if "," in str(source):
        raise ValueError(f"Docker bind mount paths must not contain commas: {source}")
    mode = "" if writable else ",readonly"
    return ["--mount", f"type=bind,source={source},target={destination}{mode}"]


def _tail(path: Path, count: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        position = stream.seek(0, os.SEEK_END)
        suffix = b""
        while position and suffix.count(b"\n") <= count:
            size = min(position, 8192)
            position -= size
            stream.seek(position)
            suffix = stream.read(size) + suffix
    return "\n".join(suffix.decode(errors="replace").splitlines()[-count:])


class _AutowareProcessGroup:
    """Own one Autoware container and its processes. A worker thread runs their I/O."""

    def __init__(self, system: AutowareDrivingSystem, context: ExternalDrivingContext,
                 town: str) -> None:
        self.system, self.context, self.town = system, context, town
        self.name = f"avsectester-autoware-{os.getuid()}-{uuid.uuid4().hex[:8]}"
        self.directory = system.work_directory / self.name
        self.logs = {key: self.directory / filename for key, filename in {
            "overrides": "overrides.log", "traffic_source": "traffic-source.log",
            "autoware": "autoware.log", "traffic_publisher": "traffic-publisher.log",
            "adapter": "adapter.log", "adapter_events": "adapter-events.jsonl",
            "last_logs": "last-logs.txt",
        }.items()}
        self.events: Queue[dict[str, Any]] = Queue()
        self.commands: Queue[dict[str, Any]] = Queue()
        self.close_requested = Event()
        self.cleanup_error: Exception | None = None
        self.recent_logs = ""
        self._created_directory = False
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._identity: str | None = None
        self._thread = Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _launch(self) -> None:
        system = self.system
        self.directory.mkdir(parents=True, exist_ok=False)
        self._created_directory = True
        map_directory = system.assets_directory / "map_tools" / "maps" / self.town
        source = system.assets_directory / "run" / "carla_traffic_light_state_source.py"
        publisher = system.assets_directory / "run" / "autoware_traffic_light_state_publisher.py"
        lanelet = map_directory / "lanelet2_map.osm"
        for asset in (source, publisher, lanelet, map_directory / "pointcloud_map.pcd",
                      map_directory / "map_projector_info.yaml",
                      system.dds_directory / "cyclonedds.xml"):
            if not asset.is_file():
                raise FileNotFoundError(f"Missing Autoware input: {asset}")
        _require_success(_exec(["image", "inspect", system.image]),
                         f"Autoware image {system.image} must be present locally")
        if self.close_requested.is_set():
            return
        mounted_map = f"/maps/{self.town}"
        adapter_directory = Path(__file__).resolve().parent / "container"
        command = [
            "docker", "run", "--pull=never", "-d", "--name", self.name,
            "--network", "host", "--gpus", f"device={system.gpu}",
            "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            "-e", "CYCLONEDDS_URI=file:///dds/cyclonedds.xml",
            "-e", f"ROS_DOMAIN_ID={system.ros_domain_id}",
            *_mount(system.dds_directory, "/dds"),
            *_mount(map_directory, mounted_map),
            *_mount(publisher, _MOUNTED_PUBLISHER),
            *_mount(adapter_directory, _MOUNTED_ADAPTER),
            *_mount(self.directory, _MOUNTED_RUN),
            *_mount(system.autoware_data_directory, "/root/autoware_data", writable=True),
            "--entrypoint", "bash", system.image, "-c", "sleep infinity",
        ]
        self._identity = launch_container(command, name=self.name)
        self.events.put({"event": "milestone", "name": "container_started"})
        if self.close_requested.is_set():
            return
        override = _exec(["exec", self._identity, "python3",
                          f"{_MOUNTED_ADAPTER}/apply_carla_overrides.py"])
        self.logs["overrides"].write_text(override.stdout + override.stderr)
        _require_success(override, "Applying CARLA configuration overrides failed")
        self.events.put({"event": "milestone", "name": "overrides_applied"})
        if self.close_requested.is_set():
            return

        signal_file = "traffic_light_states.json"
        self._process("traffic_source", [sys.executable, "-I", "-u", str(source),
                      "--host", self.context.host, "--port", str(self.context.port),
                      "--map", str(lanelet), "--output", str(self.directory / signal_file)])
        launch = {**system.launch_arguments, **_FIXED_LAUNCH, "map_path": mounted_map}
        self._process("autoware", self._container_command([
            "ros2", "launch", "autoware_launch", "e2e_simulator.launch.xml",
            *(f"{key}:={value}" for key, value in launch.items()),
        ]))
        self._process("traffic_publisher", self._container_command([
            "python3", "-u", _MOUNTED_PUBLISHER, "--input", f"{_MOUNTED_RUN}/{signal_file}",
        ]))
        self._process("adapter", self._container_command([
            "python3", "-u", f"{_MOUNTED_ADAPTER}/adapter.py",
        ], interactive=True), interactive=True)

    def _container_command(self, command: list[str], *, interactive: bool = False) -> list[str]:
        identity = self._identity
        assert identity is not None
        args = ["docker", "exec"]
        if interactive:
            args += ["-i", "-e", "RCUTILS_LOGGING_USE_STDOUT=0"]
        return [*args, identity, "bash", "-c", _SETUP, "avsectester-autoware", *command]

    def _process(self, name: str, command: list[str], *, interactive: bool = False) -> None:
        with self.logs[name].open("wb") as log:
            self._processes[name] = subprocess.Popen(
                command, stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
                stdout=subprocess.PIPE if interactive else log,
                stderr=log if interactive else subprocess.STDOUT,
            )
        self.events.put({"event": "milestone", "name": f"{name}_started"})

    def _tail_logs(self, count: int) -> str:
        return "\n".join(f"{self.logs[name].name}:\n{content}"
                         for name in ("overrides", "autoware", "traffic_source",
                                      "traffic_publisher", "adapter")
                         if (content := _tail(self.logs[name], count)))

    def _check_processes(self) -> None:
        for name, process in self._processes.items():
            status = process.poll()
            if status is not None:
                raise RuntimeError(f"{name} process exited with status {status}")

    def _communicate(self) -> None:
        adapter = self._processes["adapter"]
        assert adapter.stdin is not None and adapter.stdout is not None
        identity = self._identity
        assert identity is not None
        pending = b""
        inspected = time.monotonic()
        with selectors.DefaultSelector() as selector, self.logs["adapter_events"].open("wb") as log:
            selector.register(adapter.stdout, selectors.EVENT_READ)
            while not self.close_requested.is_set():
                self._check_processes()
                if time.monotonic() >= inspected + 1:
                    state = _exec(["inspect", "--format", "{{.State.Running}}", identity])
                    if state.returncode or state.stdout.strip() != "true":
                        raise RuntimeError(f"Autoware container {self.name} exited")
                    self.recent_logs = self._tail_logs(12)
                    inspected = time.monotonic()
                try:
                    command = self.commands.get_nowait()
                except Empty:
                    pass
                else:
                    adapter.stdin.write((json.dumps(command, allow_nan=False) + "\n").encode())
                    adapter.stdin.flush()
                if not selector.select(.05):
                    continue
                data = os.read(adapter.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("Autoware adapter closed its event stream")
                log.write(data)
                log.flush()
                lines = (pending + data).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    self.events.put(json.loads(line))

    def _cleanup(self) -> None:
        errors: list[str] = []
        identity = self._identity
        if self._created_directory:
            try:
                tails = self._tail_logs(200)
                if identity:
                    try:
                        daemon_logs = _exec(["logs", "--tail", "200", identity])
                    except (OSError, subprocess.TimeoutExpired) as error:
                        tails += f"\nUnable to collect Docker logs: {error}"
                    else:
                        tails += f"\nDocker logs:\n{daemon_logs.stdout}{daemon_logs.stderr}"
                self.logs["last_logs"].write_text(tails)
            except Exception as error:
                errors.append(f"Saving Autoware logs: {error}")
        if identity is not None:
            try:
                state = _exec(["inspect", identity])
                if not state.returncode or "No such" not in state.stderr:
                    stop_container(identity)
            except Exception as error:
                errors.append(f"Removing Autoware container {identity}: {error}")
        for name, process in self._processes.items():
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            except Exception as error:
                errors.append(f"Stopping {name}: {error}")
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.stdout is not None:
                    process.stdout.close()
        if errors:
            self.cleanup_error = RuntimeError("\n".join(errors))

    def _run(self) -> None:
        try:
            self._launch()
            if not self.close_requested.is_set():
                self._communicate()
        except Exception as error:
            if not self.close_requested.is_set():
                if self._created_directory:
                    try:
                        self.recent_logs = self._tail_logs(12)
                    except OSError as log_error:
                        self.recent_logs = f"Unable to read process logs: {log_error}"
                self.events.put({"event": "failure", "message": str(error)})
        finally:
            self._cleanup()

    def close(self) -> None:
        self.close_requested.set()
        self._thread.join()
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _AutowareSession:
    """Evaluate readiness and engagement from nonblocking adapter events."""

    def __init__(self, system: AutowareDrivingSystem, context: ExternalDrivingContext,
                 town: str) -> None:
        self._system, self._context = system, context
        self._started = time.monotonic()
        self._phase = "startup"
        self._sim_time: float | None = None
        self._timeline: list[dict[str, Any]] = []
        self._pending: dict[int, str] = {}
        self._request_id = 0
        self._localization_attempts = 0
        self._retry_at = self._started
        self._adapter_connected: float | None = None
        self._lidar = False
        self._discovery: dict[str, Any] = {}
        self._pose: dict[str, Any] | None = None
        self._localization_error: dict[str, float] | None = None
        self._watch_started = self._started
        self._goal_accepted = False
        self._goal_attempts = 0
        self._goal_sent_at = self._started
        self._state: dict[str, Any] = {
            "mode": None, "autonomous_available": None, "in_transition": None,
            "route_state": None, "localization_state": None, "mrm_state": None, "mrm_behavior": None,
        }
        self._errors: list[str] = []
        self._engage_request: dict[str, Any] | None = None
        self._engage_ack: dict[str, Any] | None = None
        self._engage_confirmation: dict[str, Any] | None = None
        self._arrived: dict[str, Any] | None = None
        self._ticks_since_engage = 0
        self._stopped = False
        self._process_group = _AutowareProcessGroup(system, context, town)

    def _time(self) -> dict[str, Any]:
        return {"wall_time": datetime.now(timezone.utc).isoformat(),
                "wall_elapsed_s": time.monotonic() - self._started, "sim_time_s": self._sim_time}

    def _record(self, name: str, **detail: Any) -> None:
        self._timeline.append({"event": name, **self._time(), **detail})

    def _next(self, milestone: str, phase: str) -> None:
        self._record(milestone)
        self._phase = phase

    def _send(self, call: str, **args: Any) -> None:
        self._request_id += 1
        self._pending[self._request_id] = call
        self._process_group.commands.put({"id": self._request_id, "call": call, "args": args})

    def _fail(self, reason: str) -> None:
        self._errors.append(reason)

    def _result(self, event: dict[str, Any]) -> None:
        call = self._pending.pop(event["id"], None)
        if call is None:
            self._fail(f"Unexpected Autoware adapter response: {event}")
            return
        if call == "initialize":
            self._record("localization_response", attempt=self._localization_attempts,
                         success=event["success"], message=event.get("message", ""),
                         code=event.get("code"))
            if event["success"]:
                self._phase = "wait for localization INITIALIZED"
            elif self._localization_attempts == _LOCALIZATION_ATTEMPTS:
                self._fail(f"Localization rejected all {_LOCALIZATION_ATTEMPTS} requests: "
                           f"{event.get('message', '')}")
            else:
                self._retry_at = time.monotonic() + _LOCALIZATION_RETRY_S
        elif call == "goal":
            if event["success"]:
                self._goal_accepted = True
            else:
                self._fail(f"Goal rejected: {event.get('message', '')}")
        else:
            if event["success"]:
                self._engage_ack = self._time()
            else:
                self._fail(f"Autonomous mode request rejected: {event.get('message', '')}")

    def _drain(self) -> None:
        for _ in range(512):
            try:
                event = self._process_group.events.get_nowait()
            except Empty:
                return
            kind = event["event"]
            if kind == "failure":
                self._fail(event["message"])
            elif kind == "milestone":
                self._record(event["name"])
            elif kind == "connected":
                self._adapter_connected = time.monotonic()
                self._next("adapter_connected", "wait for ADAPI")
            elif kind == "discovery":
                self._discovery = event
            elif kind == "mode":
                self._state.update(mode=event["mode"], autonomous_available=event["available"],
                                   in_transition=event["in_transition"])
            elif kind == "localization":
                self._state["localization_state"] = event["state"]
            elif kind == "route":
                self._state["route_state"] = event["state"]
                if event["state"] == _ROUTE_ARRIVED and self._arrived is None:
                    self._arrived = self._time()
            elif kind == "mrm":
                self._state.update(mrm_state=event["state"], mrm_behavior=event["behavior"])
            elif kind == "pose":
                self._pose = event
            elif kind == "lidar":
                self._lidar = True
                self._record("lidar_received")
            elif kind == "distortion" and self._phase == "check pointcloud distortion":
                self._fail(f"Pointcloud distortion diagnostic {event['name']}: {event['message']}")
            elif kind == "result":
                self._result(event)

    def _raise_failure(self) -> None:
        if self._errors:
            logs = self._process_group.recent_logs or "No process output collected yet."
            raise RuntimeError(
                f"Autoware {self._phase}: {self._errors[0]}\nLast ADAPI state: "
                f"{json.dumps(self._state, sort_keys=True)}\nLast process logs:\n{logs}\n"
                f"Run logs: {self._process_group.directory}"
            )

    def _pose_gate(self, truth: np.ndarray) -> bool:
        pose = self._pose
        if pose is None:
            return False
        yaw = math.atan2(truth[1, 0], truth[0, 0])
        x = truth[0, 3] - _REAR_AXLE_OFFSET_M * math.cos(yaw)
        y = truth[1, 3] - _REAR_AXLE_OFFSET_M * math.sin(yaw)
        distance = math.hypot(pose["x"] - x, pose["y"] - y)
        angle = abs(math.degrees((pose["yaw_rad"] - yaw + math.pi) % math.tau - math.pi))
        self._localization_error = {"distance_m": distance, "yaw_error_deg": angle}
        return (distance <= self._system.localization_tolerance_m
                and angle <= self._system.localization_tolerance_deg)

    def _readiness(self, truth: np.ndarray, now: float) -> None:
        if self._phase == "wait for ADAPI":
            if self._discovery.get("autonomous"):
                self._next("adapi_ready", "initialize localization")
        elif self._phase == "initialize localization":
            if not self._pending and now >= self._retry_at and self._discovery.get("initialize"):
                self._localization_attempts += 1
                self._send("initialize")
                self._record("localization_request", attempt=self._localization_attempts)
        elif self._phase == "wait for localization INITIALIZED":
            if self._state["localization_state"] == _LOCALIZATION_INITIALIZED:
                self._next("localization_initialized", "validate localization")
        elif self._phase == "validate localization":
            if self._pose_gate(truth):
                self._record("localization_check_passed", **(self._localization_error or {}))
                self._watch_started = now
                self._phase = "check pointcloud distortion"
        elif self._phase == "check pointcloud distortion":
            if now >= self._watch_started + self._system.distortion_check_s:
                self._next("distortion_check_passed", "wait for goal subscriber")
        elif self._phase == "wait for goal subscriber":
            if self._discovery.get("goal_subscriber"):
                self._request_goal(now)
                self._phase = "wait for route SET"
        elif self._phase == "wait for route SET":
            if self._goal_accepted and self._state["route_state"] == _ROUTE_SET:
                self._next("route_set", "wait for autonomous availability")
            elif "goal" not in self._pending.values() and now >= self._goal_sent_at + _GOAL_RETRY_S:
                self._request_goal(now)
        elif self._phase == "wait for autonomous availability":
            if self._lidar and self._state["autonomous_available"]:
                self._next("autonomous_available", "ready")

    def _request_goal(self, now: float) -> None:
        self._goal_attempts += 1
        self._goal_sent_at = now
        self._send("goal", **self._context.ego_goal)
        self._record("goal_requested", attempt=self._goal_attempts)

    def poll(self, ego_pose: np.ndarray, sim_time: float) -> bool:
        self._sim_time = sim_time
        self._drain()
        self._raise_failure()
        if self._phase == "ready":
            return True
        now = time.monotonic()
        startup_expired = now >= self._started + self._system.startup_timeout_s
        if (not self._lidar and self._adapter_connected is not None
                and (startup_expired or now >= self._adapter_connected + _LIDAR_TIMEOUT_S)):
            self._fail("No /sensing/lidar/top/pointcloud_raw_ex messages. "
                       "Check the ROS domain ID and DDS network.")
        elif startup_expired:
            if self._phase == "wait for route SET":
                self._fail(f"Goal rejected: route did not become SET after {self._goal_attempts} "
                           "goal requests before startup timed out")
            else:
                detail = (f". Last localization error: {self._localization_error}"
                          if self._phase == "validate localization" else "")
                self._fail(f"Startup timed out after {self._system.startup_timeout_s:g} s{detail}")
        self._raise_failure()
        self._readiness(ego_pose, now)
        return self._phase == "ready"

    def engage(self) -> None:
        self._drain()
        self._raise_failure()
        if self._phase != "ready":
            raise RuntimeError("Autoware must be ready before requesting autonomous mode")
        self._phase = "engage autonomous mode"
        self._engage_request = self._time()
        self._send("engage")

    def check(self) -> None:
        if self._engage_request is None:
            self._drain()
            self._raise_failure()
            return
        self._ticks_since_engage += 1
        elapsed = self._ticks_since_engage / self._context.tick_hz
        origin = self._engage_request["sim_time_s"]
        assert origin is not None
        self._sim_time = origin + elapsed
        self._drain()
        self._raise_failure()
        if self._engage_confirmation is not None:
            return
        if (self._engage_ack is not None and self._state["mode"] == _MODE_AUTONOMOUS
                and not self._state["in_transition"]):
            self._engage_confirmation = self._time()
            self._phase = "autonomous run"
        elif elapsed >= self._system.engage_timeout_s:
            self._fail("Autoware did not enter AUTONOMOUS before the simulation-time deadline")
            self._raise_failure()

    def telemetry(self) -> dict[str, Any]:
        return dict(self._state)

    def report(self) -> dict[str, Any]:
        return {
            "container_name": self._process_group.name,
            "readiness": [dict(entry) for entry in self._timeline],
            "engage_request": self._engage_request and dict(self._engage_request),
            "engage_response": self._engage_ack and dict(self._engage_ack),
            "engage_confirmation": self._engage_confirmation and dict(self._engage_confirmation),
            "arrived": self._arrived and dict(self._arrived),
            "logs": {key: str(path) for key, path in self._process_group.logs.items()},
        }

    def stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self._process_group.close()
