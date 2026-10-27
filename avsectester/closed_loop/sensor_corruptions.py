"""Image-space sensor corruptions used by experiment environments.

Operators map uint8 RGB frames through a caller-owned random generator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Corruption(ABC):
    """One image-space corruption. ``apply(rgb, rng) -> rgb`` maps an ``(H,W,3)`` uint8 RGB frame to a
    corrupted one; ``severity`` in ``[0, 1]`` scales the corruption strength (0 ≈ identity)."""

    category: str = "sensor"
    'Environment corruption category used to group related sensor effects.'

    def __init__(self, severity: float = 0.5) -> None:
        self.severity = float(np.clip(severity, 0.0, 1.0))

    @property
    def name(self) -> str:
        """Configuration name identifying this corruption."""
        return f"{type(self).__name__}@{self.severity:.2f}"

    @abstractmethod
    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Transform one sensor payload using the supplied random generator."""
        ...


def _f(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32)


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# --- weather ----------------------------------------------------------------------------------------

class Fog(Corruption):
    """Atmospheric haze: blend toward a bright airlight and flatten contrast (uniform, depth-free)."""

    category = "weather"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        t = 1.0 - 0.82 * self.severity          # transmission (1 = clear)
        airlight = 235.0
        out = t * _f(rgb) + (1.0 - t) * airlight
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * (1.0 - 0.25 * self.severity) + mean   # slight contrast loss
        return _u8(out)


class Rain(Corruption):
    """Darken, then add motion-blurred bright streaks (procedural)."""

    category = "weather"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        h, w = rgb.shape[:2]
        out = _f(rgb) * (1.0 - 0.25 * self.severity)               # overcast darkening
        layer = np.zeros((h, w), np.float32)
        n = int(self.severity * (h * w) / 400)
        xs = rng.integers(0, w, n)
        ys = rng.integers(0, h, n)
        length = 1 + int(self.severity * 18)
        for x, y in zip(xs, ys):
            layer[y:min(y + length, h), x] = rng.uniform(140, 255)
        k = np.zeros((length, length), np.float32)                 # near-vertical motion blur
        k[:, length // 2] = 1.0 / length
        layer = cv2.filter2D(layer, -1, k)
        out += layer[..., None]
        return _u8(out)


class Snow(Corruption):
    """Bright speckles plus a slight brighten and blur (procedural)."""

    category = "weather"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        h, w = rgb.shape[:2]
        out = _f(rgb) * (1.0 + 0.12 * self.severity) + 8.0 * self.severity
        flakes = rng.random((h, w)) < (0.02 * self.severity)
        speck = np.zeros((h, w), np.float32)
        speck[flakes] = rng.uniform(200, 255, int(flakes.sum()))
        speck = cv2.GaussianBlur(speck, (0, 0), 0.6 + self.severity)
        out += speck[..., None]
        return _u8(out)


# --- lighting ---------------------------------------------------------------------------------------

class Brightness(Corruption):
    """Multiplicative brightening (over-exposure / bright sun)."""

    category = "lighting"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return _u8(_f(rgb) * (1.0 + 0.8 * self.severity))


class LowLight(Corruption):
    """Night / underexposure: darken and add read noise (a dark sensor is a noisy sensor)."""

    category = "lighting"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = _f(rgb) * (1.0 - 0.75 * self.severity)
        out += rng.normal(0.0, 6.0 + 18.0 * self.severity, rgb.shape)
        return _u8(out)


class Contrast(Corruption):
    """Reduce contrast toward the frame mean (flat, washed-out light)."""

    category = "lighting"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mean = _f(rgb).mean(axis=(0, 1), keepdims=True)
        return _u8((_f(rgb) - mean) * (1.0 - 0.7 * self.severity) + mean)


class Gamma(Corruption):
    """Non-linear tone shift (darker mid-tones), as from a different exposure curve."""

    category = "lighting"
    'Environment corruption category used to group related sensor effects.'

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        g = 1.0 + 1.6 * self.severity
        return _u8(255.0 * (_f(rgb) / 255.0) ** g)


class ColorTemp(Corruption):
    """Color-cast shift — warm (default, sodium street-lights / sunset) or cool."""

    category = "lighting"
    'Environment corruption category used to group related sensor effects.'

    def __init__(self, severity: float = 0.5, warm: bool = True) -> None:
        super().__init__(severity)
        self.warm = warm

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = _f(rgb)
        k = 0.45 * self.severity
        r, b = (1 + k, 1 - k) if self.warm else (1 - k, 1 + k)
        out[..., 0] *= r
        out[..., 2] *= b
        return _u8(out)


# --- sensor / optics --------------------------------------------------------------------------------

class GaussianNoise(Corruption):
    """Additive Gaussian read noise."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return _u8(_f(rgb) + rng.normal(0.0, 60.0 * self.severity, rgb.shape))


class ShotNoise(Corruption):
    """Poisson (photon shot) noise — signal-dependent, as in low light."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = max(2.0, 60.0 * (1.0 - self.severity))   # fewer effective photons -> more noise
        return _u8(np.asarray(rng.poisson(_f(rgb) / 255.0 * scale)) / scale * 255.0)


class MotionBlur(Corruption):
    """Directional blur from ego/scene motion (random angle, length scales with severity)."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        length = 1 + int(self.severity * 20)
        if length < 2:
            return rgb
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0 / length
        angle = float(rng.uniform(0, 180))
        m = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0)
        k = cv2.warpAffine(k, m, (length, length))
        k /= max(k.sum(), 1e-6)
        return _u8(cv2.filter2D(_f(rgb), -1, k))


class DefocusBlur(Corruption):
    """Out-of-focus blur (Gaussian, radius scales with severity)."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        sigma = 0.4 + 4.0 * self.severity
        return _u8(cv2.GaussianBlur(_f(rgb), (0, 0), sigma))


class JPEGCompression(Corruption):
    """Lossy-compression block artefacts (encoder quality drops with severity)."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import cv2

        quality = round(100 - 92 * self.severity)
        bgr = rgb[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, max(quality, 5)])
        if not ok:
            return rgb
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


class ChromaticAberration(Corruption):
    """Lateral color fringing — shift the red and blue channels apart."""

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = round(self.severity * 8)
        if s == 0:
            return rgb
        out = rgb.copy()
        out[:, :, 0] = np.roll(rgb[:, :, 0], s, axis=1)
        out[:, :, 2] = np.roll(rgb[:, :, 2], -s, axis=1)
        return out


CORRUPTIONS: dict[str, type[Corruption]] = {
    c.__name__: c for c in (
        Fog, Rain, Snow,
        Brightness, LowLight, Contrast, Gamma, ColorTemp,
        GaussianNoise, ShotNoise, MotionBlur, DefocusBlur, JPEGCompression, ChromaticAberration,
    )
}
'Mapping from configuration names to the supplied sensor corruption classes.'


ALBUMENTATIONS_TRANSFORMS = frozenset({
    "Fog", "Rain", "Snow", "GaussNoise", "ISONoise", "MotionBlur", "DefocusBlur",
    "JPEGCompression",
})


class AlbumentationsCorruption(Corruption):
    """Named Albumentations transform with deterministic, severity-scaled parameters.

    Install the ``augment`` extra. Supported transform names are Fog, Rain, Snow,
    GaussNoise, ISONoise, MotionBlur, DefocusBlur and JPEGCompression.
    """

    def __init__(self, transform: str, severity: float = 0.5) -> None:
        super().__init__(severity)
        builders = _albumentations_builders()
        if transform not in builders:
            raise ValueError(f"Unknown Albumentations transform: {transform!r}")
        self.transform = transform
        self.category, self._build = builders[transform]

    @property
    def name(self) -> str:
        return f"AlbumentationsCorruption[{self.transform}]@{self.severity:.2f}"

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import albumentations as A

        transform = A.Compose([self._build(self.severity)], seed=int(rng.integers(0, 2**31 - 1)))
        return transform(image=np.ascontiguousarray(rgb))["image"]


def _odd(n: float) -> int:
    n = max(3, round(n))
    return n if n % 2 == 1 else n + 1


def _albumentations_builders() -> dict[str, tuple[str, Any]]:
    """``name -> (category, build(severity))`` for the Albumentations-backed operators (imports A so the
    builder lambdas capture it). Severity in ``[0, 1]`` maps to each transform's native parameters.
    Parameter names target classic albumentations 1.4.x (this environment uses numpy<2)."""
    import albumentations as A

    return {
        "Fog": ("weather", lambda s: A.RandomFog(
            fog_coef_range=(min(0.2 + 0.6 * s, 0.95),) * 2, alpha_coef=0.08 + 0.06 * s, p=1.0)),
        "Rain": ("weather", lambda s: A.RandomRain(
            drop_length=int(5 + 15 * s), drop_width=1, blur_value=_odd(1 + 6 * s),
            brightness_coefficient=max(0.5, 1.0 - 0.4 * s), p=1.0)),
        "Snow": ("weather", lambda s: A.RandomSnow(
            snow_point_range=(min(0.1 + 0.4 * s, 0.9),) * 2, brightness_coeff=1.5 + 1.0 * s, p=1.0)),
        "GaussNoise": ("sensor", lambda s: A.GaussNoise(std_range=(min(0.05 + 0.35 * s, 0.99),) * 2, p=1.0)),
        "ISONoise": ("sensor", lambda s: A.ISONoise(
            color_shift=(0.01 + 0.05 * s,) * 2, intensity=(0.1 + 0.6 * s,) * 2, p=1.0)),
        "MotionBlur": ("sensor", lambda s: A.MotionBlur(blur_limit=(_odd(3 + s * 12),) * 2, p=1.0)),
        "DefocusBlur": ("sensor", lambda s: A.Defocus(
            radius=(max(1, round(1 + s * 8)),) * 2, alias_blur=(0.1, 0.1), p=1.0)),
        "JPEGCompression": ("sensor", lambda s: A.ImageCompression(
            quality_range=(int(max(5, 100 - 90 * s)),) * 2, p=1.0)),
    }


CORRUPTIONS["AlbumentationsCorruption"] = AlbumentationsCorruption


__all__ = [
    'CORRUPTIONS',
    'Corruption',
    'Fog',
    'Rain',
    'Snow',
    'Brightness',
    'LowLight',
    'Contrast',
    'Gamma',
    'ColorTemp',
    'GaussianNoise',
    'ShotNoise',
    'MotionBlur',
    'DefocusBlur',
    'JPEGCompression',
    'ChromaticAberration',
    'AlbumentationsCorruption',
]
