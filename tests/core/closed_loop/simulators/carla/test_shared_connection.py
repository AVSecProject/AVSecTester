"""Native CARLA connection, geometry measurements and rear-axle control."""

from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla.conversions import carla_transform
from avsectester.closed_loop.simulators.carla.ego_vehicle import CarlaEgoVehicle
from avsectester.closed_loop.simulators.carla.vehicles import (
    BlueprintGeometry, best_fit_blueprint, probe_blueprints,
)
from avsectester.closed_loop.simulators.carla.world import (
    CarlaWorldSession, NativeCommands, validated_client_settings,
)
from avsectester.geometry.poses import make_pose, mirror_y_transform, rotation_matrix
from tests.fakes import CarlaNativeBoundary, CarlaVector


@pytest.fixture
def session(monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    settings, _ = validated_client_settings({"rate": 10, "max_substeps": 8})
    native = CarlaWorldSession.connect(settings, 2.0)
    try:
        yield native
    finally:
        native.close()


def test_configure_carla_world(session):
    reported = session.configure(publish_tf=False, deterministic_collisions=True)
    assert reported == {"synchronous_mode": True, "fixed_delta_seconds": .1,
                        "substepping": True, "max_substep_delta_time": .01, "max_substeps": 8,
                        "publish_tf": False, "deterministic_collisions": True}
    assert session.world.get_settings().deterministic_collisions
    assert not session.world.get_publish_tf()
    session.world.world_settings_overrides["max_substeps"] = 4
    with pytest.raises(RuntimeError, match="max_substeps differs.*expected 8, got 4"):
        session.configure()


def test_carla_session_closes_only_owned_actors(session):
    session.configure()
    blueprint = session.blueprint("vehicle.a")
    transform = carla_transform(np.eye(4))
    foreign = session.world.spawn_actor(blueprint, transform)
    owned = session.spawn(blueprint, transform)
    assert owned is not None
    session.close()
    assert foreign.is_alive
    assert not owned.is_alive
    assert not session.world.get_settings().synchronous_mode


def test_carla_session_caches_level_top_until_world_replacement(session):
    session.world.level_boxes.extend([
        SimpleNamespace(location=CarlaVector(z=4.), extent=CarlaVector(z=15.)),
        SimpleNamespace(location=CarlaVector(z=8.), extent=CarlaVector(z=2.)),
    ])
    assert session.level_top_z() == 19.
    session.world.level_boxes.clear()
    assert session.level_top_z() == 19.

    session.load_opendrive_world("<OpenDRIVE/>")
    assert session.level_top_z() == 0.


def test_carla_session_reuses_only_a_clean_world_generated_from_the_same_map(monkeypatch):
    boundary = CarlaNativeBoundary().install(monkeypatch)
    settings, _ = validated_client_settings({})

    def load(xodr, foreign_vehicle=False):
        session = CarlaWorldSession.connect(settings, 2.)
        try:
            world = session.load_opendrive_world(xodr)
            session.spawn(session.blueprint("vehicle.a"), carla_transform(np.eye(4)))
            if foreign_vehicle:
                world.spawn_actor(session.blueprint("vehicle.a"), carla_transform(np.eye(4)))
            return world
        finally:
            session.close()

    first = load("<OpenDRIVE name='a'/>")
    assert load("<OpenDRIVE name='a'/>") is first
    other_map = load("<OpenDRIVE name='b'/>")
    assert other_map is not first
    assert load("<OpenDRIVE name='b'/>", foreign_vehicle=True) is other_map
    after_foreign = load("<OpenDRIVE name='b'/>")
    assert after_foreign is not other_map
    replaced = boundary.native_client.generate_opendrive_world("<OpenDRIVE name='b'/>", None)
    assert load("<OpenDRIVE name='b'/>") not in (after_foreign, replaced)


def test_native_commands_aggregate_errors(session):
    import carla

    actor = session.spawn(session.blueprint("vehicle.a"), carla_transform(np.eye(4)))
    assert actor is not None
    commands = NativeCommands()
    commands.queue(carla.command.ApplyTargetVelocity(actor.id, carla.Vector3D(1., 0., 0.)))
    commands.queue(carla.command.SetSimulatePhysics(actor.id, False))
    session.client.command_error = "command rejected"
    try:
        with pytest.raises(RuntimeError, match="CARLA native command failed: command rejected; command rejected"):
            commands.flush(session.client)
        commands.flush(session.client)
    finally:
        session.client.command_error = None


def test_probe_carla_blueprint_measures_rear_axle(session):
    session.configure()
    vehicle, = probe_blueprints(session, "vehicle.a")
    assert vehicle.id == "vehicle.a"
    assert vehicle.dimensions == (4., 2., 1.4)
    assert vehicle.box_location == (.2, 0., .7)
    assert vehicle.rear_axle_x == pytest.approx(-1.3)
    assert vehicle.wheelbase == pytest.approx(2.6)
    assert vehicle.max_steer_angle == 45.


def test_best_fit_carla_blueprint_uses_squared_dimension_error():
    compact = BlueprintGeometry("vehicle.small", (4., 2., 1.4), (0., 0., 0.))
    wide = BlueprintGeometry("vehicle.wide", (4., 3., 1.4), (0., 0., 0.))
    assert best_fit_blueprint((wide, compact), (3.9, 2.1, 1.4)) == compact


def test_carla_ego_spawn_and_read_rear_axle_frame(session):
    session.configure()
    geometry = BlueprintGeometry("vehicle.a", (4., 2., 1.4), (.2, 0., .7), -1.3, 2.6, 45.)
    rig = make_pose(rotation_matrix((0., 0., 90.)), (5., 7., 1.))
    ego = CarlaEgoVehicle.spawn(session, geometry, rig)
    session.tick()
    actor_pose = mirror_y_transform(ego.actor.get_transform().get_matrix())
    np.testing.assert_allclose(actor_pose[:3, 3], (5., 8.3, 1.))
    np.testing.assert_allclose(ego.read().world_from_rig, rig, atol=1e-12)

    ego.set_kinematic(rig)
    session.tick()
    assert not ego.actor.simulate_physics
    ego.enable_physics((0., 2., .1))
    session.tick()
    ego.actor.angular_velocity = CarlaVector(0., 0., 90.)
    ego.actor.acceleration = CarlaVector(2., -3., 4.)
    result = ego.read()
    assert ego.actor.simulate_physics
    np.testing.assert_allclose(result.velocity, (0., 2., .1))
    np.testing.assert_allclose(result.acceleration, (2., 3., 4.))
    assert result.yaw_rate == pytest.approx(-np.pi / 2)
