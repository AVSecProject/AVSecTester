"""Public Autoware session behavior using Docker and ROS process boundaries."""
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla import ExternalDrivingContext
from avsectester.closed_loop.simulators.carla.autoware import AutowareDrivingSystem
from avsectester.closed_loop.simulators.carla.autoware.container.apply_interception import apply_rewrites
from avsectester.closed_loop.simulators.messages import MessageStage
from tests.fakes import autoware_process_boundary


CONTEXT = ExternalDrivingContext("127.0.0.1", 24880, "Town10HD_Opt",
                                 {"x": -1.16, "y": -28.37, "yaw_deg": -.16}, 20)


@pytest.fixture()
def boundary(tmp_path, monkeypatch):
    return autoware_process_boundary(tmp_path, monkeypatch)


def _ready(session, timeout=12):
    pose = np.eye(4)
    deadline = time.monotonic() + timeout
    while not session.poll(pose, 20.0):
        if time.monotonic() >= deadline:
            pytest.fail("Autoware fixture did not become ready")
        time.sleep(.01)


def _drive_until_arrived(session):
    deadline = time.monotonic() + 3
    while session.report()["arrived"] is None:
        session.check()
        if time.monotonic() >= deadline:
            pytest.fail("Autoware fixture did not complete its route")
        time.sleep(.01)


def test_autoware_start(boundary, monkeypatch):
    state, calls, paths = boundary
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, distortion_check_s=0,
                                   launch_arguments={"vehicle_model": "my_vehicle"}, **paths)
    with pytest.raises(ValueError, match="launch argument 'map_path' is fixed"):
        AutowareDrivingSystem(gpu=4, ros_domain_id=94, launch_arguments={"map_path": "/x"},
                              **paths)
    with pytest.raises(ValueError, match="CARLA map_tools"):
        system.start(ExternalDrivingContext("127.0.0.1", 24880, "Town03", CONTEXT.ego_goal, 20))
    monkeypatch.setenv("DOCKER_AUTOWARE_MODE", "retry")
    session = system.start(CONTEXT)
    try:
        # One localization retry and one goal retry wait their full intervals.
        _ready(session, timeout=20)
        report = session.report()
        milestones = [event["event"] for event in report["readiness"]]
        conditions = ["adapi_ready", "localization_request", "localization_response",
                      "localization_request", "localization_response", "localization_initialized",
                      "localization_check_passed", "distortion_check_passed", "goal_requested",
                      "goal_requested", "route_set", "autonomous_available"]
        assert [item for item in milestones if item in conditions] == conditions
        for name, interval_s in (("localization_request", 8.0), ("goal_requested", 5.0)):
            requests = [event for event in report["readiness"] if event["event"] == name]
            assert requests[1]["wall_elapsed_s"] - requests[0]["wall_elapsed_s"] >= interval_s
        assert session.telemetry()["route_state"] == 2

        docker = [json.loads(line) for line in calls.read_text().splitlines()]
        launch = next(argv for argv in docker if argv[0] == "run")
        assert "--pull=never" in launch
        assert launch[launch.index("--gpus") + 1] == "device=4"
        assert "ROS_DOMAIN_ID=94" in launch
        assert launch[launch.index("--name") + 1] == report["container_name"]
        assert launch.count("--mount") == 6
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
    system = AutowareDrivingSystem(gpu=4, ros_domain_id=94, startup_timeout_s=2,
                                   distortion_check_s=.5, **paths)
    failures = (
        ("missing-image", "must be present locally"),
        ("no-adapi", "wait for ADAPI: Startup timed out"),
        ("init-rejected", "initialize localization: Startup timed out"),
        ("uninitialized", "wait for localization INITIALIZED: Startup timed out"),
        ("bad-pose", "Last localization error"),
        ("distortion", "Pointcloud distortion diagnostic"),
        ("no-goal-subscriber", "wait for goal subscriber: Startup timed out"),
        ("goal-rejected", "Goal rejected: no route"),
        ("no-route", "Goal rejected: route did not become SET"),
        ("unavailable", "wait for autonomous availability: Startup timed out"),
        ("no-lidar", "Check the ROS domain ID and DDS network"),
    )
    for mode, error in failures:
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
    with pytest.raises(ValueError, match="cannot be both modified and observed: /planning/trajectory"):
        AutowareDrivingSystem(modified_topics=["/planning/trajectory"],
                              observed_topics=["/planning/trajectory"], **options)

    covariance = "/sensing/gnss/pose_with_covariance"
    pose = "/sensing/gnss/pose"
    lidar = "/sensing/lidar/top/pointcloud_raw_ex"
    observed = ["/z_external", "/localization/kinematic_state", pose, lidar]
    system = AutowareDrivingSystem(modified_topics=[covariance], observed_topics=observed, **options)
    assert system.identity["observed_topics"] == sorted(observed)
    assert system.interception is not None
    assert system.interception.projection is None
    assert system.interception.stages == {
        covariance: MessageStage("gnss"),
        pose: MessageStage("gnss", modifiable=False),
        lidar: MessageStage("lidar_top", modifiable=False),
        "/z_external": MessageStage(modifiable=False),
        "/localization/kinematic_state": MessageStage(modifiable=False),
    }
    routed = {topic.name: topic for topic in system.interception.topics}
    assert set(routed) == {covariance, pose}
    assert len(routed[covariance].publishers) == 2
    assert routed[pose].passthrough


@pytest.mark.parametrize(("modified", "observed", "roles", "apply_overrides"), [
    (["/localization/kinematic_state"], [], {"forwarding"}, True),
    ([], ["/system/operation_mode/state"], {"observation"}, False),
    (["/localization/kinematic_state"], ["/system/operation_mode/state"],
     {"forwarding", "observation"}, True),
])
def test_autoware_interception_processes(boundary, modified, observed, roles, apply_overrides):
    state, calls, paths = boundary
    system = AutowareDrivingSystem(
        gpu=0, ros_domain_id=94, distortion_check_s=0,
        modified_topics=modified, observed_topics=observed, **paths)
    session = system.start(CONTEXT)
    try:
        _ready(session)
        channel = session.interception_channel
        assert channel is not None and channel.relay_count == len(roles)
        report = session.report()
        directory = Path(report["logs"]["adapter"]).parent
        commands = [json.loads(line) for line in calls.read_text().splitlines()]
        base_uri = "CYCLONEDDS_URI=file:///dds/cyclonedds.xml"
        assert base_uri in next(argv for argv in commands if argv[0] == "run")
        relays = {Path(argv[argv.index("--configuration") + 1]).stem.removeprefix("interception-"): argv
                  for argv in commands if "avsectester.closed_loop.ros_interception.relay" in argv}
        assert relays.keys() == roles
        expected = {
            "forwarding": (modified, [], "/localization/kinematic_state",
                           [base_uri + ",<General><AllowMulticast>spdp</AllowMulticast></General>"]),
            "observation": ([], observed, None, []),
        }
        for role, relay in relays.items():
            routed, listened, pose_topic, environment = expected[role]
            assert relay[relay.index("-w") + 1] == "/avsectester-src"
            assert relay[relay.index("--socket") + 1] == "/avsectester-run/interception.sock"
            assert [arg for arg in relay if arg.startswith("CYCLONEDDS_URI=")] == environment
            assert Path(report["logs"][f"interception_{role}"]).name == f"interception-{role}.log"
            specification = json.loads((directory / f"interception-{role}.json").read_text())
            assert specification["role"] == role
            assert [item["name"] for item in specification["topics"]] == routed
            assert specification["observed_topics"] == listened
            assert (specification["projection"] or {}).get("pose_topic") == pose_topic

        adapter = next(argv for argv in commands if "/avsectester-adapter/adapter.py" in argv)
        assert "--raw-operation-mode-topic" not in adapter
        rewrites_applied = any("/avsectester-adapter/apply_interception.py" in argv for argv in commands)
        assert rewrites_applied == apply_overrides
    finally:
        session.stop()
    assert not (directory / "interception.sock").exists()
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
        deadline = time.monotonic() + 3
        with pytest.raises(RuntimeError, match="exited"):
            while time.monotonic() < deadline:
                session.check()
                time.sleep(.01)
    finally:
        session.stop()
    assert not list(state.iterdir())
