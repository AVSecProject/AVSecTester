"""Public Autoware session behavior using Docker and ROS process boundaries."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla import ExternalDrivingContext
from avsectester.closed_loop.simulators.carla.autoware import AutowareDrivingSystem
from avsectester.closed_loop.simulators.carla.autoware.driving_system import DEFAULT_AUTOWARE_IMAGE
from avsectester.closed_loop.simulators.carla.autoware.container.apply_interception import apply_rewrites
from tests.fakes import ControllableClock, docker_process_boundary, wait_until


CONTEXT = ExternalDrivingContext("127.0.0.1", 24880, "Town10HD_Opt",
                                 {"x": -1.16, "y": -28.37, "yaw_deg": -.16}, 20)


def autoware_process_boundary(tmp_path, monkeypatch):
    """Fake the Docker daemon, traffic source process and JSON-speaking ROS adapter."""
    extension = tmp_path / "autoware_docker.py"
    extension.write_text(r'''
import json, os, sys
autoware_mode = os.environ.get("DOCKER_AUTOWARE_MODE", "ready")
if args[:2] == ["image", "inspect"]:
    if autoware_mode == "missing-image":
        print("No such image", file=sys.stderr)
        raise SystemExit(1)
    print("id-local-image")
    raise SystemExit(0)
if args[0] == "exec":
    import threading
    import time

    if args[-1].endswith("apply_carla_overrides.py"):
        print("Applied CARLA overrides")
        raise SystemExit(0)
    if any(arg.endswith("/apply_interception.py") for arg in args):
        print("Applied ROS interception rewrites")
        raise SystemExit(0)
    interactive = args[1] == "-i"
    identity_index = 1
    while args[identity_index] in {"-i", "-w", "-e"}:
        identity_index += 1 if args[identity_index] == "-i" else 2
    path = locate(args[identity_index])
    if "avsectester.closed_loop.ros_interception.relay" in args:
        import socket
        from avsectester.closed_loop.ros_interception.wire import send_frame
        argv = next(argv for line in calls.read_text().splitlines()
                    if (argv := json.loads(line))[0] == "run"
                    and argv[argv.index("--name") + 1] == path.name)
        mount = next(arg for arg in argv if ",target=/avsectester-run," in arg)
        directory = os.open(mount.split("source=", 1)[1].split(",target=", 1)[0], os.O_DIRECTORY)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(f"/proc/self/fd/{directory}/interception.sock")
                send_frame(connection, {"kind": "ready"})
                while (current := read(path)) and current["running"]:
                    time.sleep(.05)
        finally:
            os.close(directory)
        raise SystemExit(0)
    if not interactive:
        print("Started an Autoware process", flush=True)
        while (current := read(path)) and current["running"]:
            time.sleep(.05)
        raise SystemExit(0)

    def event(kind, **fields):
        print(json.dumps({"event": kind, **fields}), flush=True)

    def mode_event(mode, available, *, in_transition=False, raw_mode=None,
                   raw_available=None, raw_in_transition=None):
        if "--raw-operation-mode-topic" in args:
            raw = {"mode": mode if raw_mode is None else raw_mode,
                   "stamp": {"sec": 20, "nanosec": 0},
                   "is_autonomous_mode_available": available if raw_available is None else raw_available,
                   "is_in_transition": in_transition if raw_in_transition is None else raw_in_transition}
            event("raw_mode", mode=raw["mode"], in_transition=raw["is_in_transition"], raw=raw)
        event("mode", mode=mode, available=available, in_transition=in_transition)

    event("connected")
    if autoware_mode != "no-lidar":
        event("lidar")
    event("discovery", initialize=True, autonomous=autoware_mode != "no-adapi",
          goal_subscriber=True)
    mode_event(1, False)
    event("mrm", state=1, behavior=1)
    retries = 0

    def publish_distortion():
        while True:
            event("distortion", name="distortion_corrector", message="missing point time")
            time.sleep(.02)

    for line in sys.stdin:
        command = json.loads(line)
        kind = command["call"]
        request_id = command["id"]
        if kind == "initialize":
            retries += 1
            if autoware_mode == "retry" and retries == 1:
                event("result", id=request_id, success=False, message="GNSS fix unavailable")
                continue
            event("result", id=request_id, success=True)
            event("localization", state=3)
            event("pose", x=-1.425, y=0.0, yaw_rad=0.0)
            if autoware_mode == "distortion":
                threading.Thread(target=publish_distortion, daemon=True).start()
        elif kind == "goal":
            accepted = autoware_mode != "goal-rejected"
            event("result", id=request_id, success=accepted,
                  message="" if accepted else "no route")
            if accepted:
                event("route", state=2)
                if autoware_mode == "interception-mode-disagreement":
                    mode_event(1, True, raw_mode=0, raw_available=False, raw_in_transition=True)
                else:
                    mode_event(1, True)
        elif kind == "engage":
            accepted = autoware_mode != "engage-rejected"
            event("result", id=request_id, success=accepted,
                  message="" if accepted else "mode transition rejected")
            if accepted and autoware_mode != "engage-timeout":
                if autoware_mode == "interception-mode-disagreement":
                    mode_event(1, True, in_transition=True, raw_mode=2, raw_in_transition=False)
                else:
                    mode_event(2, True)
                time.sleep(.05)
                event("route", state=3)
        else:
            raise ValueError(kind)
    raise SystemExit(0)
''')
    state, calls = docker_process_boundary(tmp_path, monkeypatch, extension=extension)

    assets = tmp_path / "assets"
    official = assets / "run"
    official.mkdir(parents=True)
    (official / "carla_traffic_light_state_source.py").write_text(
        "import time\nwhile True:\n    time.sleep(1)\n")
    (official / "autoware_traffic_light_state_publisher.py").touch()
    town = assets / "map_tools/maps/Town10HD"
    town.mkdir(parents=True)
    for filename in ("lanelet2_map.osm", "pointcloud_map.pcd", "map_projector_info.yaml"):
        (town / filename).touch()
    dds = tmp_path / "dds"
    dds.mkdir()
    (dds / "cyclonedds.xml").touch()
    data = tmp_path / "autoware_data"
    data.mkdir()
    options = {"assets_directory": assets, "dds_directory": dds,
               "autoware_data_directory": data, "work_directory": tmp_path / "runs"}
    return state, calls, options


@pytest.fixture()
def boundary(tmp_path, monkeypatch):
    return autoware_process_boundary(tmp_path, monkeypatch)


def _ready(session):
    wait_until(lambda: session.poll(np.eye(4), 20.0))


def _drive_until_arrived(session):
    def arrived():
        session.check()
        return session.report()["arrived"] is not None

    wait_until(arrived)


def test_autoware_start(boundary, monkeypatch):
    state, calls, paths = boundary
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, distortion_check_s=0,
                                   launch_arguments={"vehicle_model": "my_vehicle"}, **paths)
    assert system.identity["image"] == DEFAULT_AUTOWARE_IMAGE
    with pytest.raises(ValueError, match="CARLA map_tools"):
        system.start(ExternalDrivingContext("127.0.0.1", 24880, "Town03", CONTEXT.ego_goal, 20))
    clock = ControllableClock()
    clock.install(monkeypatch, AutowareDrivingSystem)
    monkeypatch.setenv("DOCKER_AUTOWARE_MODE", "retry")
    session = system.start(CONTEXT)
    try:
        def first_request_rejected():
            session.poll(np.eye(4), 20.)
            return any(event["event"] == "localization_response" and not event["success"]
                       for event in session.report()["readiness"])

        wait_until(first_request_rejected)
        clock.advance(60.)
        _ready(session)
        report = session.report()
        assert [event["attempt"] for event in report["readiness"]
                if event["event"] == "localization_request"] == [1, 2]
        conditions = ["adapi_ready", "localization_initialized", "localization_check_passed",
                      "distortion_check_passed", "goal_requested", "route_set", "autonomous_available"]
        assert [event["event"] for event in report["readiness"]
                if event["event"] in conditions] == conditions
        assert session.telemetry()["route_state"] == 2

        docker = [json.loads(line) for line in calls.read_text().splitlines()]
        launch = next(argv for argv in docker if argv[0] == "run")
        assert "--pull=never" in launch
        assert launch[launch.index("--gpus") + 1] == "device=4"
        assert "ROS_DOMAIN_ID=94" in launch
        assert launch[launch.index("--name") + 1] == report["container_name"]
        assert any("target=/root/autoware_data" in argument for argument in launch)
        assert any("target=/maps/Town10HD,readonly" in argument for argument in launch)
        stack = next(argv for argv in docker if "e2e_simulator.launch.xml" in argv)
        assert {"vehicle_model:=my_vehicle", "sensor_model:=awsim_sensor_kit",
                "perception_mode:=lidar", "rviz:=false", "simulator_type:=carla",
                "launch_simulator_interface:=false", "map_path:=/maps/Town10HD"} <= set(stack)
    finally:
        session.stop()
    assert not list(state.iterdir())
    assert "Applied CARLA overrides" in Path(report["logs"]["last_logs"]).read_text()
    session.stop()


def test_autoware_reports_readiness_failure(boundary, monkeypatch):
    state, _, paths = boundary
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, startup_timeout_s=1.5,
                                   distortion_check_s=.5, **paths)
    for mode, error in (
        ("no-adapi", "wait for ADAPI: Startup timed out"),
        ("distortion", "Pointcloud distortion diagnostic"),
        ("no-lidar", "Check the ROS domain ID and DDS network"),
    ):
        monkeypatch.setenv("DOCKER_AUTOWARE_MODE", mode)
        session = system.start(CONTEXT)
        try:
            with pytest.raises(RuntimeError, match=error) as failure:
                _ready(session)
            assert "Last ADAPI state:" in str(failure.value)
            assert "Last process logs:" in str(failure.value)
        finally:
            session.stop()
        assert not list(state.iterdir())


def test_autoware_intercepted_operation_mode(boundary, monkeypatch):
    state, calls, paths = boundary
    topics = ["/system/operation_mode/state", "/control/command/control_cmd"]
    system = AutowareDrivingSystem(
        gpu=0, ros_domain_id=94, distortion_check_s=0,
        modified_topics=topics, **paths)
    assert system.identity["modified_topics"] == sorted(topics)
    monkeypatch.setenv("DOCKER_AUTOWARE_MODE", "interception-mode-disagreement")
    session = system.start(CONTEXT)
    try:
        _ready(session)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        stack = next(argv for argv in commands if "e2e_simulator.launch.xml" in argv)
        assert "output_control_cmd:=/avsectester/intercepted/control/command/control_cmd" in stack
        adapter = next(argv for argv in commands if "/avsectester-adapter/adapter.py" in argv)
        assert adapter[-2:] == ["--raw-operation-mode-topic",
                                "/avsectester/intercepted/system/operation_mode/state"]
        ready = session.telemetry()
        assert ready["autonomous_available"] is True
        assert ready["operation_mode_state"]["is_autonomous_mode_available"] is False
        assert (ready["mode"], ready["in_transition"]) == (1, False)
        session.engage()
        _drive_until_arrived(session)
        running = session.telemetry()
        assert (running["mode"], running["in_transition"]) == (2, False)
        assert running["operation_mode_state"]["mode"] == 2
        assert running["operation_mode_state"]["stamp"] == {"sec": 20, "nanosec": 0}
        assert session.report()["engage_confirmation"] is not None
    finally:
        session.stop()
    assert not list(state.iterdir())


def test_autoware_topic_selection(boundary):
    _, _, paths = boundary
    options = {"gpu": 0, "ros_domain_id": 94, **paths}
    default = AutowareDrivingSystem(**options)
    assert default.interception is None
    assert {"modified_topics", "observed_topics", "interception_version"}.isdisjoint(default.identity)
    with pytest.raises(ValueError, match="Unknown modified topics: /arbitrary.*Choose from"):
        AutowareDrivingSystem(modified_topics=["/arbitrary"], **options)
    with pytest.raises(ValueError, match="launch argument 'map_path' is fixed"):
        AutowareDrivingSystem(launch_arguments={"map_path": "/x"}, **options)

    covariance = "/sensing/gnss/pose_with_covariance"
    pose = "/sensing/gnss/pose"
    lidar = "/sensing/lidar/top/pointcloud_raw_ex"
    observed = ["/z_external", "/localization/kinematic_state", pose, lidar]
    system = AutowareDrivingSystem(modified_topics=[covariance], observed_topics=observed, **options)
    assert system.identity["observed_topics"] == sorted(observed)
    assert system.interception is not None
    routed = {topic.name: topic for topic in system.interception.topics}
    assert set(routed) == {covariance, pose}
    assert len(routed[covariance].publishers) == 2
    assert routed[covariance].sensor == "gnss"
    assert routed[pose].passthrough
    assert set(system.interception.observed_sensors) == {(pose, "gnss"), (lidar, "lidar_top")}


def test_autoware_interception_processes(boundary):
    state, calls, paths = boundary
    modified = "/localization/kinematic_state"
    observed = "/system/operation_mode/state"
    system = AutowareDrivingSystem(
        gpu=0, ros_domain_id=94, distortion_check_s=0,
        modified_topics=[modified], observed_topics=[observed], **paths)
    session = system.start(CONTEXT)
    try:
        _ready(session)
        report = session.report()
        directory = Path(report["logs"]["adapter"]).parent
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        base_uri = "CYCLONEDDS_URI=file:///dds/cyclonedds.xml"
        assert base_uri in next(argv for argv in commands if argv[0] == "run")
        relays = {Path(argv[argv.index("--configuration") + 1]).stem: argv
                  for argv in commands if "avsectester.closed_loop.ros_interception.relay" in argv}
        assert set(relays) == {"interception-forwarding", "interception-observation"}
        assert base_uri + ",<General><AllowMulticast>spdp</AllowMulticast></General>" in relays[
            "interception-forwarding"]
        assert not any(arg.startswith("CYCLONEDDS_URI=") for arg in relays["interception-observation"])
        adapter = next(argv for argv in commands if "/avsectester-adapter/adapter.py" in argv)
        assert "--raw-operation-mode-topic" not in adapter
        assert any("/avsectester-adapter/apply_interception.py" in argv for argv in commands)
        for role, argv in relays.items():
            assert argv[argv.index("--socket") + 1] == "/avsectester-run/interception.sock"
            assert (directory / (role + ".json")).is_file()
    finally:
        session.stop()
    assert not (directory / "interception.sock").exists()
    assert not list(state.iterdir())


def test_autoware_observation_only(boundary):
    state, calls, paths = boundary
    system = AutowareDrivingSystem(
        gpu=0, ros_domain_id=94, distortion_check_s=0,
        modified_topics=[], observed_topics=["/control/command/control_cmd"], **paths,
    )
    session = system.start(CONTEXT)
    try:
        _ready(session)
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        relay, = (argv for argv in commands if "avsectester.closed_loop.ros_interception.relay" in argv)
        configuration = relay[relay.index("--configuration") + 1]
        assert Path(configuration).stem == "interception-observation"
        assert not any(arg.startswith("CYCLONEDDS_URI=") for arg in relay)
        assert not any(arg.endswith("/apply_interception.py") for argv in commands for arg in argv)
        launch = next(argv for argv in commands if "e2e_simulator.launch.xml" in argv)
        assert not any(arg.startswith("output_control_cmd:=") for arg in launch)
    finally:
        session.stop()
    assert not list(state.iterdir())


def test_apply_interception_rewrites(tmp_path):
    path = tmp_path / "launch.xml"
    source = ('<remap from="~/output" to="/system/operation_mode/state"/>\n'
              '<remap from="~/input" to="/system/operation_mode/state"/>\n')
    path.write_text(source)
    spec = tmp_path / "interception.json"
    edit = {
        "path": path.name, "topic": "/system/operation_mode/state",
        "original": 'from="~/output" to="/system/operation_mode/state"',
        "replacement": 'from="~/output" to="/avsectester/intercepted/system/operation_mode/state"',
    }
    spec.write_text(json.dumps([edit, {**edit, "topic": "/missing", "original": "different"}]))
    with pytest.raises(ValueError, match="/missing: expected one producer definition"):
        apply_rewrites(spec, tmp_path)
    assert path.read_text() == source
    spec.write_text(json.dumps([edit]))
    apply_rewrites(spec, tmp_path)
    assert path.read_text() == source.replace(edit["original"], edit["replacement"], 1)


def test_autoware_engage(boundary, monkeypatch):
    state, _, paths = boundary
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, distortion_check_s=0, **paths)
    session = system.start(CONTEXT)
    try:
        _ready(session)
        session.engage()
        _drive_until_arrived(session)
        report = session.report()
        assert report["engage_request"]["sim_time_s"] == 20.0
        assert report["engage_confirmation"]["sim_time_s"] > 20.0
        assert report["arrived"]["sim_time_s"] >= report["engage_confirmation"]["sim_time_s"]
        assert session.telemetry() == {
            "mode": 2, "autonomous_available": True, "in_transition": False,
            "localization_state": 3, "route_state": 3, "mrm_state": 1, "mrm_behavior": 1,
        }
    finally:
        session.stop()

    monkeypatch.setenv("DOCKER_AUTOWARE_MODE", "engage-rejected")
    session = system.start(CONTEXT)
    try:
        _ready(session)
        session.engage()
        with pytest.raises(RuntimeError, match="Autonomous mode request rejected"):
            _drive_until_arrived(session)
    finally:
        session.stop()
    assert not list(state.iterdir())

    monkeypatch.setenv("DOCKER_AUTOWARE_MODE", "engage-timeout")
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, distortion_check_s=0,
                                   engage_timeout_s=.2, **paths)
    session = system.start(CONTEXT)
    try:
        _ready(session)
        session.engage()
        for _ in range(3):
            session.check()
        with pytest.raises(RuntimeError, match="simulation-time deadline"):
            session.check()
    finally:
        session.stop()


def test_autoware_check_detects_exit(boundary):
    state, _, paths = boundary
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, distortion_check_s=0, **paths)
    session = system.start(CONTEXT)
    try:
        _ready(session)
        session.engage()
        _drive_until_arrived(session)
        subprocess.run(["docker", "stop", session.report()["container_name"]], check=True)
        with pytest.raises(RuntimeError, match="exited"):
            wait_until(session.check)
    finally:
        session.stop()
    assert not list(state.iterdir())
