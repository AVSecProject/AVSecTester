"""Binary patch canvases, initial textures and map-aligned panel geometry."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from avsectester.maps import TrafficControl
from avsectester.rendering import ObjectAsset
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_number,
)


DEFAULT_MASK_PATH = Path(__file__).with_name("masks") / "rp2_bars.png"


def read_patch_mask(path: str | Path) -> np.ndarray:
    """Read a nonempty binary mask as an independent boolean canvas."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    if pixels.size == 0 or not np.isin(pixels, (0, 255)).all() or not (pixels == 255).any():
        raise ValueError("Patch mask must be nonempty and binary (0 or 255)")
    return np.array(pixels == 255, copy=True)


def initial_patch_texture(
    kind: str, mask: np.ndarray, *, noise_std: float, noise_seed: int,
) -> np.ndarray:
    """Make masked white, black or independently sampled Gaussian RGB pixels."""
    if kind not in ("white", "black", "gaussian_noise"):
        raise ValueError("initial_patch must be white, black or gaussian_noise")
    require_nonnegative_number(noise_std, "noise_std")
    require_nonnegative_integer(noise_seed, "noise_seed")
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool or not mask.any():
        raise ValueError("Patch mask must be a nonempty boolean canvas")
    texture = np.zeros((*mask.shape, 4), dtype=np.uint8)
    texture[..., 3] = mask.astype(np.uint8) * 255
    if kind == "white":
        texture[..., :3][mask] = 255
    elif kind == "gaussian_noise":
        rng = np.random.default_rng(noise_seed)
        colors = np.rint(np.clip(.5 + noise_std * rng.standard_normal((*mask.shape, 3)),
                                0., 1.) * 255).astype(np.uint8)
        texture[..., :3][mask] = colors[mask]
    return texture


def validate_patch_texture(texture: object, mask: np.ndarray) -> np.ndarray:
    """Require uint8 RGBA artifacts with exactly the configured mask alpha."""
    if not isinstance(texture, np.ndarray):
        raise TypeError("Patch artifacts must be a uint8 RGBA array")
    shape = (*mask.shape, 4)
    if texture.dtype != np.uint8 or texture.shape != shape:
        raise ValueError(f"Patch artifacts must be uint8 RGBA with shape {shape}")
    if not np.array_equal(texture[..., 3], mask.astype(np.uint8) * 255):
        raise ValueError("Patch artifacts alpha must equal the configured mask times 255")
    return texture.copy()


def _check_control(control: TrafficControl) -> None:
    if not isinstance(control, TrafficControl):
        raise TypeError("control must be a TrafficControl")
    if control.kind != "stop_sign" or control.pose_reference != "face_center":
        raise ValueError("A patch requires a stop_sign control with pose_reference='face_center'")
    require_positive_number(control.extent[1], "sign width")
    require_positive_number(control.extent[2], "sign height")


def patch_asset(control: TrafficControl, texture: np.ndarray, width_scale: float) -> ObjectAsset:
    """Build a transparent panel using the map width and canvas aspect ratio."""
    _check_control(control)
    require_positive_number(width_scale, "width_scale")
    width_m = control.extent[1] * width_scale
    asset = ObjectAsset.from_panel(
        texture, category="stop_sign_patch", width_m=width_m,
        mount_height_m=0., post_count=0,
    )
    if not np.isclose(asset.extent[2], control.extent[2], rtol=1e-6, atol=1e-6):
        warnings.warn(
            f"Stop sign {control.id}: map height {control.extent[2]:.6g} m differs from "
            f"texture-derived panel height {asset.extent[2]:.6g} m; "
            f"using scaled width {width_m:.6g} m "
            f"(map width {control.extent[1]:.6g} m, width_scale {width_scale:.6g})",
            UserWarning, stacklevel=2,
        )
    return asset


def patch_pose(
    control: TrafficControl, offset_m: float = .10, *, panel_height_m: float,
) -> np.ndarray:
    """Place the panel centre along the mapped face's +X-facing normal.

    The returned pose has the bottom-centred origin used by ObjectAsset. Subtract
    half the actual panel height along local +Z, including tilted signs.
    """
    _check_control(control)
    require_nonnegative_number(offset_m, "offset_m")
    require_positive_number(panel_height_m, "panel_height_m")
    pose = np.array(control.pose, copy=True)
    rotation = pose[:3, :3]
    pose[:3, 3] += offset_m * rotation[:, 0] - panel_height_m / 2 * rotation[:, 2]
    return pose


def texture_from_colors(original_rgba: np.ndarray, mask: np.ndarray,
                        colors_chw_numpy: np.ndarray) -> np.ndarray:
    """Export masked RGB colours in [0, 1], preserving all unmasked pixels and alpha."""
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
