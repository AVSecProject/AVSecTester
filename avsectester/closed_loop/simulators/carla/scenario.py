"""Resolve CARLA scenario choices and manage native actor spawning and cleanup."""
from __future__ import annotations

import logging
import secrets
from copy import deepcopy
from typing import Any

import numpy as np

from avsectester.geometry.poses import mirror_y_transform

from .sensors import _set_blueprint_attributes, _verify_blueprint_attributes


def _loadable_map_name(name):
    """Use a server map name rather than the asset path reported by native snapshots."""
    return name.rsplit("/", 1)[-1] if isinstance(name, str) else name


def resolve_scenario_seed(scenario: dict) -> dict:
    """Copy a scene and supply a missing client seed without contacting CARLA."""
    resolved = deepcopy(scenario)
    if resolved["client"].get("seed") is None:
        resolved["client"]["seed"] = secrets.randbelow(2**31)
    return resolved


def prepare_scenario(scenario: dict) -> dict:
    """Resolve scene choices for preparation on the configured CARLA server.

    Explicit actor settings are preserved. The returned configuration contains no live
    simulator objects and can be reused after each world reset.
    """
    from avcarla.config import CARLA

    resolved = resolve_scenario_seed(scenario)
    config = resolved["client"]
    if "map_name" in config:
        config["map_name"] = _loadable_map_name(config["map_name"])
    if config.get("traffic_manager_seed") is None:
        config["traffic_manager_seed"] = config["seed"]
    if config.get("reset_world") is False:
        raise ValueError("CARLA scenario preparation requires reset_world")
    config["reset_world"] = True
    config.setdefault("strict_spawn", False)
    client = CARLA.build(deepcopy(config))
    try:
        config.update(client.scene_settings())
        if "map_name" in config:
            config["map_name"] = _loadable_map_name(config["map_name"])
        rng = np.random.RandomState(config["seed"])
        blueprints = list(client.world.get_blueprint_library().filter("vehicle"))
        sorted_blueprints = sorted(blueprints, key=lambda bp: bp.id)
        actors = [resolved["ego"], *resolved.get("npcs", [])]
        # Reserve explicit indices before choosing any random spawn, including later NPCs.
        used = {actor["spawn"] for actor in actors if isinstance(actor.get("spawn"), int)}
        for actor in actors:
            field = "vehicle" if actor is resolved["ego"] else "npc_type"
            vehicle = actor.get(field, "random")
            if vehicle in ("random", "randint", "vehicle", "random-vehicle"):
                actor[field] = sorted_blueprints[int(rng.randint(len(sorted_blueprints)))].id
            elif isinstance(vehicle, int):
                actor[field] = blueprints[vehicle].id
            if actor.get("spawn") in ("random", "randint"):
                available = [i for i in range(len(client.spawn_points)) if i not in used]
                if not available:
                    raise ValueError("Not enough unoccupied spawn points for the scenario")
                actor["spawn"] = available[int(rng.randint(len(available)))]
                used.add(actor["spawn"])
            if isinstance(actor.get("destination"), str) and actor["destination"] in (
                "random",
                "randint",
            ):
                actor["destination"] = int(rng.randint(len(client.spawn_points)))
            for sensor in actor.get("sensors", []):
                # Stable source IDs also prevent IDs from drifting across repeated runs.
                if sensor.get("source_ID") is None:
                    sensor["source_ID"] = 0
                if sensor.get("type") == "CarlaLidar" and sensor.get("noise_seed") is None:
                    sensor["noise_seed"] = int(rng.randint(2**31))
        resolved["npcs"] = actors[1:]
    finally:
        client.close()
    return resolved


def _destroy_npc(npc):
    try:
        npc.destroy()
    except Exception:
        logging.getLogger(__name__).warning("Could not destroy NPC", exc_info=True)


def _destroy_actor(actor):
    """Best-effort teardown for a raw CARLA actor such as the synthesized lead vehicle."""
    try:
        actor.destroy()
    except Exception:
        logging.getLogger(__name__).warning("Could not destroy CARLA actor", exc_info=True)


def _spawn_ros_rig(world, ego, specifications, resources, *, record_poses=False):
    """Spawn the native ROS hierarchy independently of framework acquisitions."""
    if not specifications:
        return {}
    import carla

    blueprint_library = world.get_blueprint_library()
    parents = {"ego": ego}
    poses = {"ego": np.eye(4)} if record_poses else {}
    for spec in specifications:
        blueprint = blueprint_library.find(spec["blueprint"])
        attrs = spec.get("blueprint_attributes", {})
        _set_blueprint_attributes(blueprint, attrs)
        pose = spec["transform"]
        transform = carla.Transform(carla.Location(**pose["location"]),
                                    carla.Rotation(**pose["rotation"]))
        actor = world.spawn_actor(blueprint, transform, attach_to=parents[spec["parent"]])
        resources.callback(_destroy_actor, actor)
        _verify_blueprint_attributes(actor, attrs, f"ROS rig actor {spec['name']!r}")
        if spec.get("ros", False):
            actor.enable_for_ros()
        parents[spec["name"]] = actor
        if record_poses:
            poses[spec["name"]] = poses[spec["parent"]] @ mirror_y_transform(np.asarray(transform.get_matrix()))
    return poses


def _spawn_config(spec: dict, actor: Any, vehicle_field: str) -> dict:
    """Capture the successful spawn before driving, in native CARLA world coordinates."""
    resolved = deepcopy(spec)
    transform = actor.spawn_transform
    resolved[vehicle_field] = actor.actor.type_id
    # Store the final transform, including spawn offsets and any retry displacement.
    # Replay bypasses those offsets rather than applying them a second time.
    resolved["spawn_transform"] = {
        "location": {axis: getattr(transform.location, axis) for axis in ("x", "y", "z")},
        "rotation": {axis: getattr(transform.rotation, axis) for axis in ("pitch", "yaw", "roll")},
    }
    return resolved
