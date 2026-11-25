"""Simulator components share one calibrated capture and world-object contract."""

from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from avsectester.closed_loop.simulators.base import Initialization, UnsupportedStartError
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.trajectory_tracking import TrajectoryTracker
from avsectester.closed_loop.simulators.carla_nurec import CarlaNuRecSimulator
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.closed_loop.timing import SensorTiming, TimingConfiguration
from avsectester.execution.services import ClosedLoopServices, RenderRequest, SensorRequest
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.data_types import Control
from tests.fakes import CarlaNativeBoundary, CarlaVector, NuRecRpcBoundary


@pytest.fixture
def carla_case(carla_dataset, monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime(seed=3) as runtime:
        sequence = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames)
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(sequence.at(frames[2]), sequence.resolved_scenario, sequence.provenance)
    initial = carla_dataset.scenario_start("scene", frames[0])
    initial_state = Initialization(sequence.at(frames[0]), sequence.resolved_scenario, sequence.provenance)
    return SimpleNamespace(dataset=carla_dataset, simulator=simulator, start=start,
                           initialization=initialization, shape=(12, 16, 3),
                           initial=initial, initial_state=initial_state)


@pytest.fixture
def nurec_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    # Shared simulator contracts compare inserted objects with geometric composites.
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), object_occlusion="geometry", time_step=.1)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    initial = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    return SimpleNamespace(dataset=nurec_dataset, simulator=simulator, start=start,
                           initialization=None, shape=(12, 16, 3), initial=initial, initial_state=None)


@pytest.fixture
def carla_nurec_case(nurec_dataset, monkeypatch):
    boundary = CarlaNativeBoundary().install(monkeypatch)
    boundary.install_opendrive_parser(monkeypatch)
    NuRecRpcBoundary().install(monkeypatch)
    simulator = CarlaNuRecSimulator(NuRecRenderer(cameras=["front"]), client={"rate": 20},
                                    object_occlusion="geometry", time_step=.1,
                                    min_track_duration_s=0., settle_s=.1)
    frames = nurec_dataset.frames("scene")
    start = nurec_dataset.scenario_start("scene", frames[1])
    initial = nurec_dataset.scenario_start("scene", frames[0])
    return SimpleNamespace(dataset=nurec_dataset, simulator=simulator, start=start, boundary=boundary,
                           initialization=None, shape=(12, 16, 3), initial=initial, initial_state=None)


@pytest.fixture
def recorded_case(nuscenes_dataset):
    simulator = RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"})
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[1])
    initial = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    return SimpleNamespace(dataset=nuscenes_dataset, simulator=simulator, start=start,
                           initialization=None, shape=(80, 100, 3), initial=initial, initial_state=None)


@pytest.fixture(params=["carla_case", "nurec_case", "recorded_case", "carla_nurec_case"])
def simulator_case(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def assert_inserted_world_object(make_object_in_front):
    """The inserted face follows the simulator's composition, depth and projection geometry."""
    from avsectester.data_types import Control
    from avsectester.geometry.projection import project_box
    from avsectester.rendering import PlacedObject, composite_objects

    def check(runtime, start=None, initialization=None):
        runtime.reset(start, initialization)
        asset, pose = make_object_in_front(runtime.sensor_view("front"))
        runtime.step(Control())
        clean_first = runtime.sensor_view("front")
        runtime.step(Control())
        clean_second = runtime.sensor_view("front")
        runtime.reset(start, initialization)
        runtime.insert_object("sign", asset, pose)
        runtime.step(Control())
        view = runtime.sensor_view("front")
        expected = composite_objects(
            clean_first.data, clean_first.camera, clean_first.cam_from_world,
            (PlacedObject("sign", asset, pose),), scene_depth=clean_first.depth,
            occluders=clean_first.actors, occlusion_tolerance_m=clean_first.occlusion_tolerance_m,
        )
        np.testing.assert_array_equal(view.data, expected.image)
        planned = clean_first.composite_plan((PlacedObject("sign", asset, pose),))
        np.testing.assert_array_equal(planned.apply(clean_first.data).image, view.data)
        np.testing.assert_array_equal(view.composition.background, clean_first.data)
        np.testing.assert_array_equal(view.composition.plan.visible_mask, planned.visible_mask)
        assert runtime.object_visibility("sign")["front"] == expected.object_visibility["sign"]
        changed = np.any(view.data != clean_first.data, axis=2)
        assert np.any(changed)
        world_from_box = pose.copy()
        world_from_box[:3, 3] += pose[:3, :3] @ np.array([0., 0., 1.])
        x0, y0, x1, y1 = project_box(view.camera, view.cam_from_world, world_from_box, asset.extent)
        ys, xs = np.nonzero(changed)
        assert xs.min() >= np.floor(x0) and xs.max() < np.ceil(x1)
        assert ys.min() >= np.floor(y0) and ys.max() < np.ceil(y1)
        runtime.remove_object("sign")
        runtime.step(Control())
        np.testing.assert_array_equal(runtime.sensor_view("front").data, clean_second.data)
        return clean_first, view, expected
    return check


@pytest.fixture
def assert_camera_views_equal():
    def compare(first, second):
        assert first.sensor == second.sensor
        assert first.native == second.native == {}
        np.testing.assert_array_equal(first.data, second.data)
        np.testing.assert_equal(first.depth, second.depth)
        np.testing.assert_allclose(first.cam_from_world, second.cam_from_world)
        np.testing.assert_allclose(first.sensor_from_world, second.sensor_from_world)
        assert type(first.camera) is type(second.camera)
        assert (first.camera.width, first.camera.height) == (second.camera.width, second.camera.height)
        pixels = np.array([[.5, .5], [first.camera.width / 2, first.camera.height / 2]])
        np.testing.assert_allclose(first.camera.unproject(pixels), second.camera.unproject(pixels))
        assert first.actors.keys() == second.actors.keys()
        for identity in first.actors:
            np.testing.assert_allclose(first.actors[identity].box_transform(), second.actors[identity].box_transform())
            assert first.actors[identity].extent == second.actors[identity].extent
        np.testing.assert_allclose(first.ego.box_transform(), second.ego.box_transform())
        assert first.ego.extent == second.ego.extent
    return compare


def test_simulator_contract(simulator_case):
    case = simulator_case
    simulator, start, initialization, expected_shape = case.simulator, case.start, case.initialization, case.shape
    with simulator.create_runtime(seed=3) as runtime:
        assert runtime.can_replay is simulator.can_replay
        runtime.reset(start, initialization)
        original = runtime.observe()
        state = runtime.state()
        assert state.ego is not None and "ego" not in state.actors
        view = runtime.sensor_view("front")
        assert runtime.camera_sensors <= runtime.sensors
        assert view.sensor in runtime.sensors
        assert view.data.shape == expected_shape
        assert view.time == original.t and view.frame == original.frame
        assert view.cam_from_world.shape == (4, 4)
        assert view.sensor_from_world.shape == (4, 4)
        np.testing.assert_allclose(view.sensor_from_world, BODY_FROM_OPTICAL @ view.cam_from_world)
        assert view.camera.project(np.array([[0., 0., 4.]])).shape == (1, 2)
        runtime.step(Control(throttle=.1))
        advanced = runtime.observe()
        assert advanced.frame == state.frame + 1
        assert advanced.t - state.t == pytest.approx(.1)
        assert runtime.state().frame == advanced.frame
        with pytest.raises(ValueError):
            runtime.sensor_view("missing-camera")
    with pytest.raises(RuntimeError, match="Reset"):
        runtime.observe()


def test_read_sensors_views(simulator_case, assert_camera_views_equal):
    """Fresh service sessions must expose the same capture through read_sensors and render."""
    dataset, simulator, start = simulator_case.dataset, simulator_case.simulator, simulator_case.start
    frames = dataset.frames(start.scenario_id)[:3]
    with ClosedLoopServices(dataset, simulator) as services:
        with services.session(3) as session:
            samples = [session.read_sensors(SensorRequest(start.scenario_id, frame, ("front",),
                                                        views=True)) for frame in frames]
        for sample in samples:
            with services.session(3) as session:
                rendered = session.render(RenderRequest(start.scenario_id, sample.frame, "front"))
            assert_camera_views_equal(sample.views["front"], rendered.view)


def test_carla_nurec_reset_and_native_step(carla_nurec_case):
    case = carla_nurec_case
    with case.simulator.create_runtime() as runtime:
        runtime.reset(case.start)
        world = case.boundary.native_client.get_world()
        initial = runtime.state().ego.pose
        np.testing.assert_allclose(initial, case.start.payload["ego"].pose)
        assert runtime.initialization_report["proxy_count"] == 1
        assert runtime.initialization_report["missing_proxies"] == []
        assert world.get_settings().deterministic_collisions
        assert runtime.initialization_report["ego_geometry"]["rear_axle_x"] == pytest.approx(-1.3)
        trajectory = Control(trajectory=[((0., 0., 0.), (1., 0., 0., 0.), 0),
                                         ((1., .3, 0.), (1., 0., 0., 0.), 100000)])
        runtime.advance_interval(100000, trajectory, report_times_us=(50000,))
        runtime.capture_at(50000)
        middle = runtime.state().ego.pose
        runtime.capture_at(100000)
        assert initial[0, 3] < middle[0, 3] < runtime.state().ego.pose[0, 3]
        proxy = next(actor for actor in world.get_actors()
                     if actor.type_id.startswith("vehicle.") and actor.is_alive and not actor.simulate_physics)
        ego = next(actor for actor in world.get_actors()
                   if actor.type_id.startswith("vehicle.") and actor.is_alive and actor.simulate_physics)
        assert ego.control.steer < 0. and ego.control.throttle > 0.
        assert proxy.get_transform().location.x > 7.
    assert not any(actor.is_alive for actor in (proxy, ego))
    assert not world.get_settings().synchronous_mode


def test_carla_nurec_force_gt_handoff(carla_nurec_case):
    case = carla_nurec_case
    with case.simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=100000))
        runtime.reset(case.start, closed_loop_run=True)
        world = case.boundary.native_client.get_world()
        runtime.advance_interval(100000, Control(throttle=1.), force_gt=True)
        runtime.capture_at(100000)
        time_us = round(case.start.frame.timestamp * 1e6) + 100000
        recorded = case.dataset.ego_state(case.start.scenario_id, time_us)
        np.testing.assert_allclose(runtime.state().ego.pose, recorded.pose)
        assert not any(actor.simulate_physics for actor in world.get_actors()
                       if actor.is_alive and actor.type_id.startswith("vehicle."))
        runtime.advance_interval(200000, Control(throttle=1.), coerce_dynamic_state=True)
        runtime.capture_at(200000)
        ego = next(actor for actor in world.get_actors()
                   if actor.is_alive and actor.type_id.startswith("vehicle.") and actor.simulate_physics)
        assert ego.velocity.x > recorded.velocity[0]


def test_carla_nurec_brakes_fully_for_empty_trajectory(carla_nurec_case):
    case = carla_nurec_case
    simulator = CarlaNuRecSimulator(
        case.simulator.renderer, client={"rate": 20}, time_step=.1,
        object_occlusion="geometry", tracker=TrajectoryTracker(hold_brake=.25),
    )
    with simulator.create_runtime() as runtime:
        runtime.reset(case.start)
        runtime.step(Control(trajectory=[]))
        world = case.boundary.native_client.get_world()
        ego = next(actor for actor in world.get_actors()
                   if actor.is_alive and actor.type_id.startswith("vehicle.") and actor.simulate_physics)
        assert ego.control.brake == 1. and ego.control.throttle == 0.


def test_carla_nurec_reports_proxy_body_contacts_per_interval(carla_nurec_case):
    case = carla_nurec_case
    with case.simulator.create_runtime() as runtime:
        runtime.reset(case.start)
        state = runtime.state()
        (track, actor), = state.actors.items()
        runtime.set_actor_state(track, replace(actor, pose=state.ego.pose, velocity=(0., 0., 0.)))
        runtime.advance_interval(100000, Control(brake=1.))
        runtime.capture_at(100000)
        assert runtime.interval_telemetry() == {"collisions": [track]}
        ahead = state.ego.pose.copy()
        ahead[:3, 3] += 50. * ahead[:3, 0]
        runtime.set_actor_state(track, replace(actor, pose=ahead, velocity=(0., 0., 0.)))
        runtime.advance_interval(200000, Control(brake=1.))
        runtime.capture_at(200000)
        assert runtime.interval_telemetry() == {"collisions": []}


def test_carla_nurec_reports_reverse_motion_as_nonnegative_speed(nurec_dataset, monkeypatch):
    def roll_back():
        # A collision rebound leaves the physical ego moving backward after the tick.
        for vehicle in boundary.native_client.world.get_actors():
            if vehicle.is_alive and vehicle.type_id.startswith("vehicle.") and vehicle.simulate_physics:
                forward = np.asarray(vehicle.get_transform().get_matrix())[:3, 0]
                vehicle.velocity = CarlaVector(*(-2. * forward))

    boundary = CarlaNativeBoundary(tick_callback=roll_back).install(monkeypatch)
    boundary.install_opendrive_parser(monkeypatch)
    NuRecRpcBoundary().install(monkeypatch)
    simulator = CarlaNuRecSimulator(NuRecRenderer(cameras=["front"]), client={"rate": 20},
                                    time_step=.1, min_track_duration_s=0., settle_s=.1)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        runtime.step(Control(throttle=.1))
        state = runtime.state()
        assert state.vehicle_state.speed == pytest.approx(-2.)
        assert state.ego_speed == pytest.approx(2.)
        assert runtime.observe().ego_speed == pytest.approx(2.)


def test_carla_nurec_retains_slower_camera_pose_for_trajectory(carla_nurec_case):
    case = carla_nurec_case
    with case.simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(frame_interval_us=500000)}))
        runtime.reset(case.initial)
        for time_us in range(100000, 500000, 100000):
            runtime.advance_interval(time_us, Control(throttle=.1))
            runtime.capture_at(time_us, ())
        trajectory = Control(trajectory=[((0., 0., 0.), (1., 0., 0., 0.), 0),
                                         ((2., 0., 0.), (1., 0., 0., 0.), 500000)])
        runtime.step(trajectory)
        assert runtime.clock.time_us == 500000


@pytest.mark.parametrize("bad_origin,error,match", [
    (True, None, "geoReference"),
    (False, "parser refused this map", "parser refused this map"),
])
def test_carla_nurec_rejects_unusable_maps(carla_nurec_case, tmp_path, monkeypatch,
                                         bad_origin, error, match):
    case = carla_nurec_case
    archive_path = tmp_path / "bad-map.usdz"
    with ZipFile(case.start.payload["usdz_path"]) as source, ZipFile(archive_path, "w") as result:
        for name in source.namelist():
            data = source.read(name)
            if name == "map.xodr":
                if bad_origin:
                    data = data.replace(b"+=alt_0=0", b"")
                else:
                    data += b"\n<!-- rejected parser input -->\n"
            result.writestr(name, data)
    case.boundary.install_opendrive_parser(monkeypatch, error)
    start = replace(case.start, payload={**case.start.payload, "usdz_path": archive_path})
    with pytest.raises(UnsupportedStartError, match=match):
        case.simulator.validate_start(start)


def test_carla_nurec_reports_map_check_failures(carla_nurec_case, tmp_path, monkeypatch):
    case = carla_nurec_case
    archive_path = tmp_path / "unchecked-map.usdz"
    with ZipFile(case.start.payload["usdz_path"]) as source, ZipFile(archive_path, "w") as result:
        for name in source.namelist():
            data = source.read(name)
            result.writestr(name, data + b"\n<!-- unchecked parser input -->\n" if name == "map.xodr" else data)
    case.boundary.install_opendrive_parser(
        monkeypatch, "ModuleNotFoundError: No module named 'carla'", returncode=3)
    start = replace(case.start, payload={**case.start.payload, "usdz_path": archive_path})
    with pytest.raises(RuntimeError, match="CARLA Python client"):
        case.simulator.validate_start(start)


def test_carla_nurec_reuses_validated_archive_map_for_reset(carla_nurec_case, monkeypatch):
    case = carla_nurec_case
    reads = []
    read = ZipFile.read

    def read_archive(archive, name, *args, **kwargs):
        if name == "map.xodr":
            reads.append(archive.filename)
        return read(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "read", read_archive)
    case.simulator.validate_start(case.start)
    with case.simulator.create_runtime() as runtime:
        runtime.reset(case.start)
    assert reads == [str(case.start.payload["usdz_path"])]


@pytest.mark.parametrize("simulator_case", ["carla_case", "recorded_case"], indirect=True)
def test_insert_object(simulator_case, assert_inserted_world_object):
    simulator, initial, initial_state = simulator_case.simulator, simulator_case.initial, simulator_case.initial_state
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, _expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is None and inserted.depth is None


def test_insert_object_with_depth(nurec_case, assert_inserted_world_object):
    simulator, initial, initial_state = nurec_case.simulator, nurec_case.initial, nurec_case.initial_state
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is not None
        np.testing.assert_array_equal(inserted.depth, expected.depth)


@pytest.fixture
def nurec_alias_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    return nurec_dataset, "front", NuRecSimulator, {
        "renderer": NuRecRenderer(cameras=["front"]), "time_step": .1,
        "depth": "none", "physics_update_mode": "none",
    }


@pytest.fixture
def recorded_alias_case(nuscenes_dataset):
    return nuscenes_dataset, "CAM_FRONT", RecordedReplaySimulator, {}


@pytest.fixture(params=(
    pytest.param("nurec_alias_case", id="nurec"),
    pytest.param("recorded_alias_case", id="recorded"),
))
def alias_case(request):
    return request.getfixturevalue(request.param)


def test_camera_aliases(alias_case):
    dataset, native, simulator_type, options = alias_case
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    with simulator_type(**options).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == {native}
        native_view = runtime.sensor_view(native)
    aliases = {native: "forward"}
    simulator = simulator_type(**options, camera_aliases=aliases)
    assert simulator.identity["camera_aliases"] == aliases
    aliases[native] = "changed"
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        observation = runtime.observe()
        assert runtime.camera_sensors == {"forward"}
        assert set(observation.sensor_data) == set(observation.calibration) == {"forward"}
        assert runtime.sensor_view(native).sensor == "forward"
        assert runtime.sensor_geometry(native).sensor == "forward"
        view = runtime.sensor_view("forward")
        np.testing.assert_array_equal(view.data, native_view.data)
        np.testing.assert_allclose(view.cam_from_world, native_view.cam_from_world)
        runtime.step(Control())
        assert runtime.sensor_view("forward").sensor == "forward"
