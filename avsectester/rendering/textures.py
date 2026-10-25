"""Texture descriptions and independent RGBA image loading for rendering assets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from avsectester.validation import require_positive_integer

# Bundled public-domain texture shared by attacks; sources are listed in assets/SOURCES.md.
STOP_SIGN_FACE = Path(__file__).resolve().with_name("assets") / "mutcd_r1-1_stop.png"


class Texture(Protocol):
    """A user-provided texture that returns an RGB or RGBA uint8 image."""

    def rgba(self) -> np.ndarray: ...


def load_rgba_image(value: np.ndarray | str | Path | Texture) -> np.ndarray:
    """Load one RGB/RGBA image as an independent uint8 RGBA array.

    Texture objects expose ``rgba()``. Paths are interpreted by the caller; configuration
    loading resolves them relative to the configuration file before constructing a texture.
    """
    if callable(getattr(value, "rgba", None)):
        value = value.rgba()
    if isinstance(value, (str, Path)):
        from PIL import Image
        with Image.open(value) as image:
            array = np.array(image.convert("RGBA"))
    elif isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
    else:
        raise TypeError("texture must be an image path, uint8 image array or texture object")
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4) or 0 in array.shape:
        raise ValueError("texture must be a nonempty uint8 RGB or RGBA image")
    if array.shape[2] == 3:
        array = np.concatenate((array, np.full((*array.shape[:2], 1), 255, np.uint8)), axis=2)
    return array


@dataclass(frozen=True)
class CheckerboardTexture:
    """An opaque square checkerboard with an explicit pixel size and cell count."""

    size: int = 256
    cells: int = 8
    colors: tuple = ((255, 0, 0), (255, 255, 255))

    def __post_init__(self) -> None:
        for name in ("size", "cells"):
            require_positive_integer(getattr(self, name), name)
        if self.cells > self.size:
            raise ValueError("cells cannot exceed the texture size")
        colors = np.asarray(self.colors)
        if (colors.shape != (2, 3) or not np.issubdtype(colors.dtype, np.integer)
                or ((colors < 0) | (colors > 255)).any()):
            raise ValueError("colors must contain two RGB integer colors in [0, 255]")

    def rgba(self) -> np.ndarray:
        positions = np.arange(self.size) * self.cells // self.size
        pattern = (positions[:, None] + positions[None, :]) % 2
        rgb = np.asarray(self.colors, dtype=np.uint8)[pattern]
        return np.concatenate((rgb, np.full((self.size, self.size, 1), 255, np.uint8)), axis=2)
