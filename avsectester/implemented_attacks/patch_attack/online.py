"""Deploy calibrated image patches or native CARLA panels from explicit artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import numpy as np

from avsectester.attack.runtime import AttackRuntime, WorldAttackAction
from avsectester.rendering.textures import load_rgba_image
from avsectester.attack.insertion.action import InsertionAttackAction
from avsectester.attack.insertion.placement import AttachedPlacement
from avsectester.geometry.poses import rotation_matrix
from avsectester.closed_loop.simulators.carla.conversions import rotation_to_carla

DEFAULT_PROP = "static.prop.ironplank"


def to_carla_texture(rgba: np.ndarray) -> Any:
    """Convert an ``(H, W, 4)`` uint8 RGBA array into a ``carla.TextureColor`` (standard RGBA)."""
    import carla

    h, w = rgba.shape[:2]
    tex = carla.TextureColor(w, h)
    for y in range(h):
        for x in range(w):
            r, g, b, a = (int(v) for v in rgba[y, x])
            tex.set(x, y, carla.Color(r, g, b, a))
    return tex


def _matte_material() -> Any:
    """A matte, non-metallic, non-emissive material map so the patch is lit by the scene like a real
    poster (packs AO=1, Roughness=1, Metallic=0, Emissive=0 into AO_Roughness_Metallic_Emissive)."""
    import carla

    tex = carla.TextureFloatColor(4, 4)
    for y in range(4):
        for x in range(4):
            tex.set(x, y, carla.FloatColor(1.0, 1.0, 0.0, 0.0))
    return tex


@dataclass
class CarlaPatchPanel:
    """A native CARLA panel owned by one run, with an explicit actor-local transform."""

    location: tuple
    rotation: tuple
    texture: np.ndarray
    prop: str = DEFAULT_PROP
    emissive: bool = True

    def apply(
        self, world: Any, target_actor: Any, *,
        tick: Callable[[], Any] | None = None,
        on_spawn: Callable[[Any], None] | None = None,
    ) -> list:
        """Spawn the panel on ``target_actor`` in ``world`` and paint it. Return spawned actors."""
        import carla

        bl = world.get_blueprint_library()
        hits = bl.filter(self.prop)
        if not hits:
            raise ValueError(f"patch prop {self.prop!r} not in the CARLA blueprint library")
        rel = carla.Transform(
            carla.Location(x=self.location[0], y=self.location[1], z=self.location[2]),
            carla.Rotation(pitch=self.rotation[0], yaw=self.rotation[1], roll=self.rotation[2]),
        )
        names_before = set(world.get_names_of_all_objects())
        prop = world.spawn_actor(
            hits[0], rel, attach_to=target_actor, attachment_type=carla.AttachmentType.Rigid
        )
        if on_spawn is not None:
            on_spawn(prop)  # register ownership before any later deployment operation can fail
        (tick or world.tick)()  # let the prop register as a nameable StaticMeshActor
        new_names = list(set(world.get_names_of_all_objects()) - names_before)
        if not new_names:
            prop.destroy()
            raise RuntimeError("spawned patch prop did not register a paintable object name")
        texture = to_carla_texture(load_rgba_image(self.texture))
        matte = _matte_material()
        painted = []
        for name in new_names:
            try:
                world.apply_color_texture_to_object(name, carla.MaterialParameter.Diffuse, texture)
                if self.emissive:  # self-illuminate (unlit) so scene lighting does not dim the texture
                    world.apply_color_texture_to_object(name, carla.MaterialParameter.Emissive, texture)
                else:  # explicit scene-lit mode: brightness and shadows follow the environment
                    world.apply_float_color_texture_to_object(
                        name, carla.MaterialParameter.AO_Roughness_Metallic_Emissive, matte
                    )
                painted.append(name)
            except RuntimeError:
                pass  # not every new name is the prop mesh; paint the one(s) that take it
        if not painted:
            prop.destroy()
            raise RuntimeError(f"could not paint the patch onto any of {new_names}")
        return [prop]


class PatchAttackRuntime(AttackRuntime):
    """Deploy one patch in a fresh run, through world setup or a camera action."""

    def __init__(self, attack, texture, physical, segment):
        self.attack, self.texture, self.physical = attack, texture, physical
        self.insertion = attack.insertion(texture, placement=attack.selected_placement(segment))
        self.setup_ticks = 1 if physical else 0

    def setup(self, context):
        return (PatchWorldAttackAction(self.attack, self.texture, self.insertion),) if self.physical else ()

    def step(self, context):
        return (() if self.physical else (InsertionAttackAction(
            self.attack.sensor, self.insertion, compositor=self.attack.compositor),))


@dataclass
class PatchWorldAttackAction(WorldAttackAction):
    attack: Any
    texture: np.ndarray
    insertion: Any
    setup_ticks: int = 1

    def apply(self, simulator_runtime, resources):
        native = simulator_runtime.native
        actors, world = native.get("actors", {}), native.get("world")
        placement = self.insertion.placement
        if not isinstance(placement, AttachedPlacement):
            raise TypeError("Native CARLA patches require AttachedPlacement")
        host_id = placement.host
        if world is None or host_id not in actors:
            raise ValueError(f"Physical patch requires a CARLA world and host {host_id!r}")
        host = simulator_runtime.state().actors[host_id]
        if host.extent is None:
            raise ValueError(f"Patch host {host_id!r} has no bounding-box extent")
        if self.insertion.orientation.mode != "follow_host":
            raise ValueError("CARLA physical patches require follow_host orientation")
        # Compose entirely in actor-local coordinates. The actor origin is not the box
        # centre; selecting and optimizing the plane use this same local box offset.
        box_rotation = rotation_matrix(host.box_rotation)
        position = np.asarray(host.box_center) + box_rotation @ placement.local_position(host.extent)
        rotation = box_rotation @ rotation_matrix(self.insertion.orientation.rotation_deg)
        flip = np.diag([1., -1., 1.])
        # The prop's native XY plane is mounted vertically. This fixed installation
        # correction maps the canonical +X-front plane onto that native mesh.
        correction = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        native_rotation = flip @ rotation @ flip @ correction
        panel = CarlaPatchPanel(location=tuple(flip @ position),
            rotation=rotation_to_carla(native_rotation), texture=self.texture,
            prop=self.attack.prop, emissive=self.attack.emissive)
        panel.apply(world, actors[host_id], tick=simulator_runtime.setup_tick,
                    on_spawn=lambda created: resources.callback(_destroy_panel, created))


def _destroy_panel(actor):
    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass


__all__ = [
    'CarlaPatchPanel',
]
