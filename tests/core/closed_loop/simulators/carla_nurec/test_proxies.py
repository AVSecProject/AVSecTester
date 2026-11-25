"""Recorded road users become kinematic actors in the native CARLA boundary."""

import numpy as np

from avsectester.closed_loop.simulators.carla.vehicles import BlueprintGeometry
from avsectester.closed_loop.simulators.carla.world import CarlaWorldSession, validated_client_settings
from avsectester.closed_loop.simulators.carla_nurec.alignment import SceneAlignment
from avsectester.closed_loop.simulators.carla_nurec.proxies import RecordedTrackProxies
from avsectester.data_types import ActorState
from avsectester.geometry.geodesy import enu_from_ecef
from tests.fakes import CarlaNativeBoundary


def test_recorded_track_proxies_select_move_and_remove_actors(monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    settings, _ = validated_client_settings({})
    session = CarlaWorldSession.connect(settings, 2.)
    origin = (0., 0., 0.)
    alignment = SceneAlignment.from_scene(origin, np.linalg.inv(enu_from_ecef(*origin)))
    car = BlueprintGeometry("vehicle.a", (4., 2., 1.4), (.2, 0., .7))
    walker = BlueprintGeometry("walker.pedestrian.0001", (.5, .5, 1.8), (0., 0., .9))
    proxies = RecordedTrackProxies(session, alignment, (car,), (walker,))

    def actor(x, category, extent):
        pose = np.eye(4)
        pose[0, 3] = x
        return ActorState(pose, category=category, extent=extent)

    try:
        session.configure()
        proxies.sync({"road-car": actor(4., "vehicle", car.dimensions),
                      "cyclist": actor(8., "rider", car.dimensions),
                      "walker": actor(20., "person", walker.dimensions),
                      "animal": actor(32., "animal", walker.dimensions)})
        session.tick()
        live = [item for item in session.world.get_actors()
                if item.type_id.startswith(("vehicle.", "walker.")) and item.is_alive]
        assert [item.type_id for item in live] == ["vehicle.a", "vehicle.a", "walker.pedestrian.0001"]
        assert all(not item.simulate_physics for item in live)
        assert proxies.count == 3 and not proxies.missing
        proxies.sync({"walker": actor(21., "person", walker.dimensions)})
        session.tick()
        assert live[-1].get_transform().location.x == 21.
        assert not any(item.is_alive for item in live[:2])
        proxies.destroy_all()
        session.tick()
        assert not live[-1].is_alive and proxies.count == 0
    finally:
        session.close()


def test_recorded_track_proxy_moves_from_staging_to_pose_inside_static_box(monkeypatch):
    # The static prop extends from 0 to 18 m, covering the recorded pose and a 10 m lift.
    CarlaNativeBoundary(static_boxes=[((7., -4., 9.), (2., 2., 9.))]).install(monkeypatch)
    settings, _ = validated_client_settings({})
    session = CarlaWorldSession.connect(settings, 2.)
    car = BlueprintGeometry("vehicle.a", (4., 2., 1.4), (.2, 0., .7))
    proxies = RecordedTrackProxies(session, SceneAlignment(np.eye(4)), (car,), ())
    pose = np.eye(4)
    pose[:3, 3] = (7., 4., 2.)
    track = ActorState(pose, category="vehicle", extent=car.dimensions, box_center=car.box_location)

    try:
        session.configure()
        proxies.sync({"parked-car": track})
        session.tick()
        assert not proxies.missing
        proxy, = [actor for actor in session.world.get_actors() if actor.type_id == "vehicle.a"]
        assert not proxy.simulate_physics
        location = proxy.get_transform().location
        np.testing.assert_allclose([location.x, location.y, location.z], [7., -4., 2.])
    finally:
        session.close()
