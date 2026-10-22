"""Shared image loading for textured rendering assets."""
from pathlib import Path

import numpy as np


def load_rgba_image(value: np.ndarray | str | Path | object) -> np.ndarray:
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

