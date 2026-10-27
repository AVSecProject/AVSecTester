"""Stop-sign replica faces, patch masks, initial textures and map-aligned panel geometry.

The replica is the canonical MUTCD R1-1 face. A patch replaces only the masked pixels of that
face, in face-texture coordinates, so the legend stays where the face design puts it.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from avsectester.maps import TrafficControl
from avsectester.rendering import STOP_SIGN_FACE, ObjectAsset, load_rgba_image
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_number,
)


DEFAULT_MASK_PATH = Path(__file__).with_name("masks") / "rp2_bars.png"
INITIAL_PATCHES = ("none", "white", "black", "gaussian_noise")
MUTCD_STOP_SIGN_SIZES_M = (0.762, 0.914, 1.219)


def replica_face() -> np.ndarray:
    """The canonical uint8 RGBA stop-sign face, transparent outside the octagon."""
    return load_rgba_image(STOP_SIGN_FACE)


def read_patch_mask(path: str | Path) -> np.ndarray:
    """Read a nonempty binary mask that lies inside the replica face's opaque octagon."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.size == 0 or not np.isin(pixels, (0, 255)).all() or not (pixels == 255).any():
        raise ValueError("Patch mask must be nonempty and binary (0 or 255)")
    mask = np.array(pixels == 255, copy=True)
    face = replica_face()
    if mask.shape != face.shape[:2]:
        raise ValueError(f"Patch mask must match the replica face resolution {face.shape[:2]}")
    if (mask & (face[..., 3] == 0)).any():
        raise ValueError("Patch mask must lie inside the replica face's opaque pixels")
    return mask


def initial_patch_texture(
    kind: str, mask: np.ndarray, *, noise_std: float, noise_seed: int,
) -> np.ndarray:
    """Return the replica face with its masked pixels white, black or Gaussian noise.

    ``none`` returns the unpatched replica face. Noise samples independent RGB values from
    ``clip(0.5 + noise_std * N(0, 1), 0, 1)``. Alpha and unmasked pixels are always the face's.
    """
    if kind not in INITIAL_PATCHES:
        raise ValueError("initial_patch must be none, white, black or gaussian_noise")
    require_nonnegative_number(noise_std, "noise_std")
    require_nonnegative_integer(noise_seed, "noise_seed")
    texture = replica_face()
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != texture.shape[:2] or not mask.any():
        raise ValueError("Patch mask must be a nonempty boolean canvas at the face resolution")
    if kind == "white":
        texture[..., :3][mask] = 255
    elif kind == "black":
        texture[..., :3][mask] = 0
    elif kind == "gaussian_noise":
        rng = np.random.default_rng(noise_seed)
        colors = np.rint(np.clip(.5 + noise_std * rng.standard_normal((*mask.shape, 3)),
                                0., 1.) * 255).astype(np.uint8)
        texture[..., :3][mask] = colors[mask]
    return texture


def validate_replica_texture(texture: object, mask: np.ndarray) -> np.ndarray:
    """Require a uint8 RGBA replica face that differs from the canonical face only in the mask."""
    if not isinstance(texture, np.ndarray):
        raise TypeError("Replica artifacts must be a uint8 RGBA array")
    face = replica_face()
    if texture.dtype != np.uint8 or texture.shape != face.shape:
        raise ValueError(f"Replica artifacts must be uint8 RGBA with shape {face.shape}")
    if not np.array_equal(texture[..., 3], face[..., 3]):
        raise ValueError("Replica artifacts must keep the canonical face alpha")
    if not np.array_equal(texture[..., :3][~mask], face[..., :3][~mask]):
        raise ValueError("Replica artifacts may change colours only inside the patch mask")
    return texture.copy()


def _check_control(control: TrafficControl) -> None:
    if not isinstance(control, TrafficControl):
        raise TypeError("control must be a TrafficControl")
    if control.kind != "stop_sign" or control.pose_reference != "face_center":
        raise ValueError("A replica requires a stop_sign control with pose_reference='face_center'")
    require_positive_number(control.extent[1], "sign width")
    require_positive_number(control.extent[2], "sign height")


def replica_asset(
    control: TrafficControl, texture: np.ndarray, width_scale: float, face_size: str,
) -> ObjectAsset:
    """Build a post-free face panel with a selected width and the texture aspect ratio.

    ``mutcd_standard`` selects the MUTCD R1-1 size closest to the mapped sign height;
    ``map_width`` keeps the mapped width. Both multiply that width by ``width_scale``.

    The canonical face is the panel's nominal texture, so a harmonizer estimates scene
    lighting from the unpatched replica and applies it unchanged to every patch.
    """
    _check_control(control)
    require_positive_number(width_scale, "width_scale")
    if face_size == "mutcd_standard":
        face_width = min(MUTCD_STOP_SIGN_SIZES_M, key=lambda size: abs(size - control.extent[2]))
    elif face_size == "map_width":
        face_width = control.extent[1]
    else:
        raise ValueError("face_size must be mutcd_standard or map_width")
    width_m = face_width * width_scale
    asset = ObjectAsset.from_panel(
        texture, category="stop_sign", width_m=width_m,
        mount_height_m=0., post_count=0, nominal_texture=replica_face(),
    )
    if face_size == "map_width" and not np.isclose(
        asset.extent[2], control.extent[2], rtol=1e-6, atol=1e-6,
    ):
        warnings.warn(
            f"Stop sign {control.id}: map height {control.extent[2]:.6g} m differs from "
            f"texture-derived panel height {asset.extent[2]:.6g} m; "
            f"using scaled width {width_m:.6g} m "
            f"(map width {control.extent[1]:.6g} m, width_scale {width_scale:.6g})",
            UserWarning, stacklevel=2,
        )
    return asset


def replica_pose(
    control: TrafficControl, offset_m: float = .10, *, panel_height_m: float,
    face_orientation: str,
) -> np.ndarray:
    """Place a panel in front of the mapped face center using the selected orientation.

    ``upright`` projects the mapped +X normal onto the horizontal plane and uses world
    +Z as its up axis; ``map`` preserves the mapped rotation. Offset the face center along
    the selected +X, then lower the ObjectAsset origin by half the panel height along
    the selected +Z. The mapped face center itself is never corrected.
    """
    _check_control(control)
    require_nonnegative_number(offset_m, "offset_m")
    require_positive_number(panel_height_m, "panel_height_m")
    pose = np.array(control.pose, copy=True)
    if face_orientation == "upright":
        normal = np.array([pose[0, 0], pose[1, 0], 0.])
        horizontal_norm = np.linalg.norm(normal)
        if horizontal_norm < 1e-6:
            raise ValueError("upright face_orientation requires a horizontal normal of at least 1e-6")
        normal /= horizontal_norm
        up = np.array([0., 0., 1.])
        pose[:3, :3] = np.column_stack((normal, np.cross(up, normal), up))
    elif face_orientation != "map":
        raise ValueError("face_orientation must be upright or map")
    rotation = pose[:3, :3]
    pose[:3, 3] += offset_m * rotation[:, 0] - panel_height_m / 2 * rotation[:, 2]
    return pose


def texture_from_colors(original_rgba: np.ndarray, mask: np.ndarray,
                        colors_chw_numpy: np.ndarray) -> np.ndarray:
    """Export masked RGB colors in [0, 1], preserving all unmasked pixels and alpha."""
    original = np.asarray(original_rgba)
    colors = np.asarray(colors_chw_numpy)
    mask = np.asarray(mask)
    if (original.ndim != 3 or original.shape[2] != 4 or original.dtype != np.uint8
            or mask.shape != original.shape[:2] or mask.dtype != bool
            or colors.shape != (3, *original.shape[:2]) or not np.isfinite(colors).all()):
        raise ValueError("Texture export needs uint8 RGBA, a matching bool mask and finite RGB CHW colours")
    result = original.copy()
    pixels = np.rint(np.clip(np.moveaxis(colors, 0, -1), 0., 1.) * 255).astype(np.uint8)
    result[..., :3][mask] = pixels[mask]
    return result


