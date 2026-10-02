"""Physical adversarial patch — a *world-level* attack for CARLA.

Unlike the two existing attack seams (``perturb(Observation)`` in :func:`avsectester.backend.run`,
and avstack ``HOOKS`` hooks on a pipeline stage), a physical patch modifies the **world itself**: a
textured panel is rigidly attached to a target vehicle (e.g. the rear of the lead car) so the ego's
camera renders it in-scene, with correct perspective, lighting and occlusion. It is therefore applied
by the backend at ``reset`` (see :class:`avsectester.scenario.CarlaBackend`), not by the run loop.

Mechanism (validated against CARLA 0.9.15):
  * spawn a flat prop (``static.prop.ironplank``) with ``attach_to=<target>, AttachmentType.Rigid``;
  * a freshly spawned prop registers a new ``StaticMeshActor_<id>`` name in
    ``world.get_names_of_all_objects()`` — that name (not the actor id/type) is what the texture API
    paints;
  * paint it with ``world.apply_color_texture_to_object(name, MaterialParameter.Diffuse, texture)``.

The pixel generation (:func:`checkerboard_rgba`, :func:`image_rgba`) is plain numpy/PIL and unit-tested
without CARLA; only :func:`to_carla_texture` and :meth:`PhysicalPatch.apply` touch ``carla`` (imported
lazily, so this module imports fine in the base env).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The validated flat patch prop + its pose on a car rear (relative to the target actor).
DEFAULT_PROP = "static.prop.ironplank"
DEFAULT_LOCATION = (-2.3, 0.0, 1.0)  # x back, y lateral, z up (metres) in the target's frame
DEFAULT_ROTATION = (90.0, 0.0, 0.0)  # pitch, yaw, roll (deg) — stands the plank vertical, facing aft


def checkerboard_rgba(size: int = 256, squares: int = 8,
                      color_a: tuple = (255, 0, 0), color_b: tuple = (255, 255, 255)) -> np.ndarray:
    """A red/white (default) checkerboard as an ``(size, size, 4)`` uint8 RGBA array."""
    cell = max(1, size // squares)
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[..., 3] = 255
    xs = (np.arange(size) // cell)[:, None]
    ys = (np.arange(size) // cell)[None, :]
    mask = (xs + ys) % 2 == 0
    img[mask, :3] = color_a
    img[~mask, :3] = color_b
    return img


def image_rgba(path: str, size: int = 256) -> np.ndarray:
    """Load an image file as an ``(size, size, 4)`` uint8 RGBA array (for a real adversarial patch)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA").resize((size, size))
    return np.asarray(im, dtype=np.uint8)


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
class PhysicalPatch:
    """A textured panel attached to a target actor's rear, painted with an arbitrary image.

    ``texture`` is either ``{"pattern": "checkerboard", ...}`` (the default) or
    ``{"image": "<path>", "size": N}``. By default (``emissive=False``) the panel is given a **matte**
    material so it is **lit by the scene** — brightness and shadows track the environment, the
    realistic look. ``emissive=True`` instead paints the texture into the Emissive channel so it is
    self-lit (unlit by the scene) — an unrealistic mode that lets an optimized adversarial texture
    survive rendering unchanged. ``apply`` spawns and paints the panel and returns the actors.
    """

    prop: str = DEFAULT_PROP
    location: tuple = DEFAULT_LOCATION
    rotation: tuple = DEFAULT_ROTATION
    texture: dict = field(default_factory=lambda: {"pattern": "checkerboard", "size": 256})
    emissive: bool = False

    def _rgba(self) -> np.ndarray:
        spec = dict(self.texture)
        size = int(spec.get("size", 256))
        if spec.get("image"):
            return image_rgba(spec["image"], size=size)
        return checkerboard_rgba(
            size=size,
            squares=int(spec.get("squares", 8)),
            color_a=tuple(spec.get("color_a", (255, 0, 0))),
            color_b=tuple(spec.get("color_b", (255, 255, 255))),
        )

    def apply(self, world: Any, target_actor: Any) -> list:
        """Spawn the panel on ``target_actor`` in ``world`` and paint it; return spawned actors."""
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
        world.tick()  # let the prop register as a nameable StaticMeshActor
        new_names = list(set(world.get_names_of_all_objects()) - names_before)
        if not new_names:
            prop.destroy()
            raise RuntimeError("spawned patch prop did not register a paintable object name")
        texture = to_carla_texture(self._rgba())
        matte = _matte_material()
        painted = []
        for name in new_names:
            try:
                world.apply_color_texture_to_object(name, carla.MaterialParameter.Diffuse, texture)
                if self.emissive:  # self-illuminate (unlit) so scene lighting does not dim the texture
                    world.apply_color_texture_to_object(name, carla.MaterialParameter.Emissive, texture)
                else:  # realistic default: a matte surface lit by the scene (shadows/brightness track it)
                    world.apply_float_color_texture_to_object(
                        name, carla.MaterialParameter.AO_Roughness_Metallic_Emissive, matte
                    )
                painted.append(name)
            except RuntimeError:
                pass  # not every new name is the prop mesh; paint the one(s) that take it
        if not painted:
            prop.destroy()
            raise RuntimeError(f"could not paint the patch onto any of {new_names}")
        self.painted_objects = painted
        return [prop]


def build_patch(cfg: dict | PhysicalPatch) -> PhysicalPatch:
    """Build a :class:`PhysicalPatch` from a scenario-config dict (or pass one through)."""
    if isinstance(cfg, PhysicalPatch):
        return cfg
    cfg = dict(cfg)
    cfg.pop("type", None)
    cfg.pop("target", None)  # target selection is the backend's concern, not the patch's
    cfg.pop("gap", None)
    cfg.pop("lead_vehicle", None)
    return PhysicalPatch(**cfg)
