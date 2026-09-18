"""Physical (CARLA-render) threat model: δ is a texture PAINTED on the car, then rendered.

``apply`` is the crux of the physical attack — instead of a digital paste, it paints δ onto the
physical patch and returns the frame CARLA *renders*, so the attack optimizes exactly what the camera
sees. It is therefore non-differentiable and pairs with a gradient-free attack (NES).

The CARLA coupling is injected as two callables so this module stays free of carla/torch:
  * ``paint(rgba)``  applies an ``(H,W,4)`` uint8 texture to the physical patch,
  * ``render()``     ticks and returns the ego camera frame as ``(H,W,3)`` uint8.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .interface import AdvSample, Perturbation


def to_rgba_uint8(delta: np.ndarray) -> np.ndarray:
    """(3, H, W) float in [0,1] -> (H, W, 4) uint8 RGBA (opaque)."""
    rgb = (np.clip(delta, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    alpha = np.full((*rgb.shape[:2], 1), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


class CarlaRenderPerturbation(Perturbation):
    """A texture painted on the physical patch and rendered by CARLA (non-differentiable)."""

    def __init__(
        self,
        paint: Callable[[np.ndarray], None],
        render: Callable[[], np.ndarray],
        size: tuple[int, int] = (64, 64),
        init_texture: np.ndarray | None = None,
        seed: int = 0,
    ):
        self.paint = paint
        self.render = render
        self.hp, self.wp = size
        self.init_texture = init_texture
        self._rng = np.random.default_rng(seed)

    def init(self) -> np.ndarray:
        if self.init_texture is not None:  # warm start (e.g. the digital-surrogate texture)
            return np.clip(self.init_texture, 0, 1).astype(np.float32).copy()
        return self._rng.random((3, self.hp, self.wp)).astype(np.float32)

    def apply(self, sample: AdvSample, delta: np.ndarray) -> np.ndarray:
        self.paint(to_rgba_uint8(delta))  # physically repaint the patch
        return self.render()  # the CARLA-RENDERED frame

    def project(self, delta: np.ndarray) -> np.ndarray:
        return np.clip(delta, 0, 1)

    def export(self, delta: np.ndarray) -> np.ndarray:
        return to_rgba_uint8(delta)
