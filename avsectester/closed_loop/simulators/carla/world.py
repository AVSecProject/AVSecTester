"""Configure native CARLA worlds and batch commands at tick boundaries."""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from copy import deepcopy
from fnmatch import fnmatch
from hashlib import sha256
from typing import Any

from numpy.typing import ArrayLike

from avsectester.closed_loop.timing import seconds_to_microseconds
from avsectester.datasets.carla import CARLA_CLIENT_RUNTIME_SETTINGS, CARLA_CLIENT_SUBSTEP_SETTINGS
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_integer,
    require_positive_number,
)

from .conversions import carla_transform


_OPENDRIVE_GENERATION = dict(
    vertex_distance=2.0, max_road_length=500.0, wall_height=0.0,
    additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True,
)
# Worlds this process generated, by endpoint: the episode ID and the OpenDRIVE digest.
_GENERATED_WORLDS: dict[tuple[str, str, int], tuple[int, bytes]] = {}
_RUNTIME_ACTORS = ("vehicle.*", "walker.*", "sensor.*", "controller.*")


def validated_client_settings(
    client: dict, allowed: Collection[str] = CARLA_CLIENT_RUNTIME_SETTINGS,
) -> tuple[dict, int]:
    """Copy and validate CARLA runtime settings, returning the base tick in microseconds."""
    unknown = set(client).difference(allowed)
    if unknown:
        raise ValueError(f"CARLA client only accepts runtime settings: {sorted(unknown)}")
    settings = deepcopy(client)
    settings.setdefault("rate", 20.0)
    settings.setdefault("synchronous", True)
    rate = settings["rate"]
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError("CARLA client rate must be positive and finite")
    if settings["synchronous"] is not True:
        raise ValueError("CARLA requires synchronous operation")
    if "substepping" in client and not isinstance(client["substepping"], bool):
        raise ValueError("client.substepping must be a boolean")
    if "max_substep_delta_time" in client:
        require_positive_number(client["max_substep_delta_time"], "client.max_substep_delta_time")
    if "max_substeps" in client:
        require_positive_integer(client["max_substeps"], "client.max_substeps")
    for name in ("seed", "traffic_manager_seed"):
        if settings.get(name) is not None:
            require_nonnegative_integer(settings[name], f"client.{name}")
    for name, default in (("connect_port", 2000), ("traffic_manager_port", 8000)):
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ValueError(f"client.{name} must be an integer in [1,65535]")
    try:
        tick = seconds_to_microseconds(1 / float(rate))
    except ValueError as error:
        raise ValueError("CARLA client.rate must produce an integer-microsecond base tick") from error
    return settings, tick


def ticks_for_duration(seconds: float, rate: float, name: str) -> int:
    """Require the named duration to be a nonnegative whole number of base ticks."""
    require_nonnegative_number(seconds, name)
    ticks = seconds * float(rate)
    if not math.isfinite(ticks) or not math.isclose(ticks, round(ticks), rel_tol=0., abs_tol=1e-9):
        raise ValueError(f"{name} must be a whole number of CARLA base ticks")
    return round(ticks)


def _physics_settings(client: Mapping[str, Any]) -> dict[str, Any]:
    return {"synchronous_mode": client["synchronous"], "fixed_delta_seconds": 1. / client["rate"],
            **{name: client[name] for name in CARLA_CLIENT_SUBSTEP_SETTINGS if name in client}}


def configure_world_settings(
    world: Any, client: Mapping[str, Any], *, publish_tf: bool | None = None,
    deterministic_collisions: bool | None = None,
) -> dict[str, Any]:
    """Check the active physics settings and apply and verify explicit TF or collision options."""
    requested = _physics_settings(client)
    native = world.get_settings()
    if deterministic_collisions is not None:
        requested["deterministic_collisions"] = deterministic_collisions
        if native.deterministic_collisions != deterministic_collisions:
            native.deterministic_collisions = deterministic_collisions
            world.apply_settings(native)
            native = world.get_settings()
    actual = {name: getattr(native, name) for name in (
        "synchronous_mode", "fixed_delta_seconds", *CARLA_CLIENT_SUBSTEP_SETTINGS)}
    if deterministic_collisions is not None:
        actual["deterministic_collisions"] = native.deterministic_collisions
    if publish_tf is not None:
        requested["publish_tf"] = publish_tf
        world.set_publish_tf(publish_tf)
        actual["publish_tf"] = world.get_publish_tf()
    for name, value in requested.items():
        observed = actual[name]
        matches = observed == value
        if name in {"fixed_delta_seconds", "max_substep_delta_time"} and observed is not None:
            # The SDK can return these settings at native floating-point precision.
            matches = math.isclose(observed, value, rel_tol=1e-7)
        if not matches:
            raise RuntimeError(
                f"CARLA world setting {name} differs from the requested value: "
                f"expected {value!r}, got {observed!r}")
    return actual


class NativeCommands:
    """Queue native commands and apply a batch without advancing the world."""

    def __init__(self) -> None:
        self._commands: list[Any] = []

    def queue(self, command: Any) -> None:
        """Append a command in native execution order."""
        self._commands.append(command)

    def flush(self, client: Any) -> None:
        """Apply and consume pending commands, reporting all native response errors together."""
        if not self._commands:
            return
        commands, self._commands = self._commands, []
        responses = client.apply_batch_sync(commands, False)
        failures = [response.error for response in responses if response.has_error()]
        if failures:
            raise RuntimeError(f"CARLA native command failed: {'; '.join(failures)}")

    def clear(self) -> None:
        """Discard pending commands without applying them."""
        self._commands.clear()


def carla_world_key(host: str, port: int) -> tuple[str, str, int]:
    """Return the resource key for a native CARLA endpoint, normalizing loopback addresses."""
    if host in {"localhost", "127.0.0.1", "::1"}:
        host = "localhost"
    return ("carla", host, port)


class CarlaWorldSession:
    """Own a native CARLA world attachment and the actors created through it."""

    def __init__(self, client: Any, settings: Mapping[str, Any]) -> None:
        self.client = client
        self.world = client.get_world()
        self._settings = settings
        self._commands = NativeCommands()
        self._actors: dict[int, Any] = {}
        self._pending_destroy: set[int] = set()
        self._restore_async = False
        self._level_top_z: float | None = None

    @classmethod
    def connect(cls, settings: Mapping[str, Any], timeout_s: float) -> CarlaWorldSession:
        """Connect with validated runtime settings and apply the native RPC timeout."""
        import carla

        client = carla.Client(settings.get("connect_ip", "127.0.0.1"), settings.get("connect_port", 2000))
        client.set_timeout(timeout_s)
        return cls(client, settings)

    def load_opendrive_world(self, xodr: str) -> Any:
        """Activate an OpenDRIVE world built with the standard reconstructed-scene parameters.

        The current world is kept when this process generated it from the same OpenDRIVE text,
        its episode is unchanged and it holds no vehicles, walkers, sensors or controllers.
        Otherwise CARLA generates a new world.
        """
        import carla

        if self._actors:
            raise RuntimeError("Close the CARLA world session before replacing a world with owned actors")
        key = carla_world_key(self._settings.get("connect_ip", "127.0.0.1"),
                              self._settings.get("connect_port", 2000))
        digest = sha256(xodr.encode("utf-8")).digest()
        current = self.client.get_world()
        if _GENERATED_WORLDS.get(key) == (current.id, digest) and not any(
                actor.is_alive for actor in current.get_actors()
                if any(fnmatch(actor.type_id, pattern) for pattern in _RUNTIME_ACTORS)):
            self.world = current
        else:
            _GENERATED_WORLDS.pop(key, None)
            self.world = self.client.generate_opendrive_world(
                xodr, carla.OpendriveGenerationParameters(**_OPENDRIVE_GENERATION))
            _GENERATED_WORLDS[key] = (self.world.id, digest)
        self._level_top_z = None
        return self.world

    def level_top_z(self) -> float:
        """Return the highest level bounding-box top in meters, or zero for an empty level."""
        import carla

        if self._level_top_z is None:
            boxes = self.world.get_level_bbs(carla.CityObjectLabel.Any)
            self._level_top_z = max((box.location.z + box.extent.z for box in boxes), default=0.)
        return self._level_top_z

    def configure(
        self, *, publish_tf: bool | None = None, deterministic_collisions: bool | None = None,
    ) -> dict[str, Any]:
        """Apply synchronous physics settings and return the verified native values."""
        native = self.world.get_settings()
        for name, value in _physics_settings(self._settings).items():
            setattr(native, name, value)
        self._restore_async = True
        self.world.apply_settings(native)
        return configure_world_settings(self.world, self._settings, publish_tf=publish_tf,
                                        deterministic_collisions=deterministic_collisions)

    def queue(self, command: Any) -> None:
        """Queue a native command to run before the next tick."""
        self._commands.queue(command)

    def tick(self) -> int:
        """Flush queued commands and advance the world by one native tick."""
        self._commands.flush(self.client)
        for identity in self._pending_destroy:
            self._actors.pop(identity)
        self._pending_destroy.clear()
        return self.world.tick()

    def blueprint(self, identity: str) -> Any:
        """Find the native blueprint with the given identifier in the active world."""
        return self.world.get_blueprint_library().find(identity)

    def spawn(self, blueprint: Any, transform: Any) -> Any | None:
        """Try to spawn and register an actor, returning None if the native spawn fails."""
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self._actors[actor.id] = actor
        return actor

    def place(self, actor: Any, world_from_actor: ArrayLike) -> None:
        """Queue a pose in the right-handed CARLA world coordinates."""
        import carla

        self.queue(carla.command.ApplyTransform(actor.id, carla_transform(world_from_actor)))

    def place_kinematic(self, actor: Any, world_from_actor: ArrayLike) -> None:
        """Queue physics disabling and pose placement for a kinematic actor."""
        import carla

        self.queue(carla.command.SetSimulatePhysics(actor.id, False))
        self.place(actor, world_from_actor)

    def destroy(self, actor: Any) -> None:
        """Queue destruction of an owned actor at the next command boundary."""
        import carla

        identity = actor.id
        if self._actors.get(identity) is not actor:
            raise ValueError(f"CARLA world session does not own actor {identity}")
        if identity not in self._pending_destroy:
            self.queue(carla.command.DestroyActor(identity))
            self._pending_destroy.add(identity)

    def close(self) -> None:
        """Discard pending motion, destroy owned actors, and restore asynchronous ticking."""
        import carla

        self._commands.clear()
        try:
            for identity, actor in self._actors.items():
                if actor.is_alive:
                    self._commands.queue(carla.command.DestroyActor(identity))
            self._commands.flush(self.client)
        finally:
            self._actors.clear()
            self._pending_destroy.clear()
            if self._restore_async:
                self._restore_async = False
                native = self.world.get_settings()
                native.synchronous_mode = False
                native.fixed_delta_seconds = None
                self.world.apply_settings(native)


__all__ = [
    'CarlaWorldSession',
    'NativeCommands',
    'carla_world_key',
    'configure_world_settings',
    'ticks_for_duration',
    'validated_client_settings',
]
