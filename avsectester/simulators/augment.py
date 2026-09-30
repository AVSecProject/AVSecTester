"""Scenario augmentation — controlled corruptions of the camera stream, for attack-robustness testing.

An attack is only *robust* if it survives the nuisance variation a deployed AV sees: weather, lighting,
and sensor/optics degradation. This module provides that variation as image-space :class:`Corruption`
operators, lifted to the same ``perturb: Observation -> Observation`` seam an attack uses
(:func:`avsectester.backend.run`). So an augmentation composes with an attack — ``compose(attack, aug)``
inserts the patch and *then* corrupts the frame — and the two can be diffed under one condition to see
whether the attack still fires (see ``AUGMENT_DESIGN.md``).

Backend-agnostic: a :class:`Corruption` transforms an ``(H, W, 3)`` uint8 RGB array, so this works on
CARLA, in-process NuRec, and dataset replay (nuScenes/nuRec recorded frames) alike. CARLA additionally
supports *world-level* weather (native ``carla.WeatherParameters`` at reset) — see
:mod:`avsectester.simulators.carla`.

Every operator takes ``severity`` in ``[0, 1]`` (0 ≈ identity) and a seeded ``numpy.random.Generator``,
so runs are reproducible and the *same* corruption can be applied to the clean and attacked run of a pair
(:class:`AugmentationPipeline` seeds from ``(seed, frame)``). ``cv2`` is imported lazily.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from avsectester.plane import Observation

# ----------------------------------------------------------------------------------------------------
# Corruption operators
# ----------------------------------------------------------------------------------------------------


class Corruption(ABC):
    """One image-space corruption. ``apply(rgb, rng) -> rgb`` maps an ``(H,W,3)`` uint8 RGB frame to a
    corrupted one; ``severity`` in ``[0, 1]`` scales the effect (0 ≈ identity)."""

    category: str = "sensor"

    def __init__(self, severity: float = 0.5) -> None:
        self.severity = float(np.clip(severity, 0.0, 1.0))

    @property
    def name(self) -> str:
        return f"{type(self).__name__}@{self.severity:.2f}"

    @abstractmethod
    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...


def _f(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32)


def _u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


# --- weather ----------------------------------------------------------------------------------------

class Fog(Corruption):
    """Atmospheric haze: blend toward a bright airlight and flatten contrast (uniform, depth-free)."""

    category = "weather"

    def apply(self, rgb, rng):
        t = 1.0 - 0.82 * self.severity          # transmission (1 = clear)
        airlight = 235.0
        out = t * _f(rgb) + (1.0 - t) * airlight
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * (1.0 - 0.25 * self.severity) + mean   # slight contrast loss
        return _u8(out)


class Rain(Corruption):
    """Darken, then add motion-blurred bright streaks (procedural)."""

    category = "weather"

    def apply(self, rgb, rng):
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

    def apply(self, rgb, rng):
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

    def apply(self, rgb, rng):
        return _u8(_f(rgb) * (1.0 + 0.8 * self.severity))


class LowLight(Corruption):
    """Night / underexposure: darken and add read noise (a dark sensor is a noisy sensor)."""

    category = "lighting"

    def apply(self, rgb, rng):
        out = _f(rgb) * (1.0 - 0.75 * self.severity)
        out += rng.normal(0.0, 6.0 + 18.0 * self.severity, rgb.shape)
        return _u8(out)


class Contrast(Corruption):
    """Reduce contrast toward the frame mean (flat, washed-out light)."""

    category = "lighting"

    def apply(self, rgb, rng):
        mean = _f(rgb).mean(axis=(0, 1), keepdims=True)
        return _u8((_f(rgb) - mean) * (1.0 - 0.7 * self.severity) + mean)


class Gamma(Corruption):
    """Non-linear tone shift (darker mid-tones), as from a different exposure curve."""

    category = "lighting"

    def apply(self, rgb, rng):
        g = 1.0 + 1.6 * self.severity
        return _u8(255.0 * (_f(rgb) / 255.0) ** g)


class ColorTemp(Corruption):
    """Colour-cast shift — warm (default, sodium street-lights / sunset) or cool."""

    category = "lighting"

    def __init__(self, severity: float = 0.5, warm: bool = True) -> None:
        super().__init__(severity)
        self.warm = warm

    def apply(self, rgb, rng):
        out = _f(rgb)
        k = 0.45 * self.severity
        r, b = (1 + k, 1 - k) if self.warm else (1 - k, 1 + k)
        out[..., 0] *= r
        out[..., 2] *= b
        return _u8(out)


# --- sensor / optics --------------------------------------------------------------------------------

class GaussianNoise(Corruption):
    """Additive Gaussian read noise."""

    def apply(self, rgb, rng):
        return _u8(_f(rgb) + rng.normal(0.0, 60.0 * self.severity, rgb.shape))


class ShotNoise(Corruption):
    """Poisson (photon shot) noise — signal-dependent, as in low light."""

    def apply(self, rgb, rng):
        scale = max(2.0, 60.0 * (1.0 - self.severity))   # fewer effective photons -> more noise
        return _u8(rng.poisson(_f(rgb) / 255.0 * scale) / scale * 255.0)


class MotionBlur(Corruption):
    """Directional blur from ego/scene motion (random angle, length scales with severity)."""

    def apply(self, rgb, rng):
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

    def apply(self, rgb, rng):
        import cv2

        sigma = 0.4 + 4.0 * self.severity
        return _u8(cv2.GaussianBlur(_f(rgb), (0, 0), sigma))


class JPEGCompression(Corruption):
    """Lossy-compression block artefacts (encoder quality drops with severity)."""

    def apply(self, rgb, rng):
        import cv2

        quality = round(100 - 92 * self.severity)
        bgr = rgb[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, max(quality, 5)])
        if not ok:
            return rgb
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


class ChromaticAberration(Corruption):
    """Lateral colour fringing — shift the red and blue channels apart."""

    def apply(self, rgb, rng):
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


def common_corruptions(severity: float = 0.5) -> list[AugmentationPipeline]:
    """A standard robustness suite — one single-corruption pipeline per operator at ``severity`` (the AV
    analogue of ImageNet-C common corruptions, plus driving weather). Sweep ``severity`` for a curve."""
    return [AugmentationPipeline([cls(severity)]) for cls in CORRUPTIONS.values()]


# ----------------------------------------------------------------------------------------------------
# Pipeline + lift to the perturb seam
# ----------------------------------------------------------------------------------------------------


class AugmentationPipeline:
    """A seeded chain of :class:`Corruption`s. ``apply(rgb, frame)`` runs them in order under an RNG
    seeded from ``(seed, frame)`` — deterministic in the frame index, so the clean and attacked runs of a
    pair (sharing ``seed``) get a pixel-identical corruption and ``metric.impact`` measures only the attack.
    """

    def __init__(self, corruptions: Iterable[Corruption], seed: int = 0) -> None:
        self.corruptions = list(corruptions)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return "+".join(c.name for c in self.corruptions) or "identity"

    def apply(self, rgb: np.ndarray, frame: int = 0) -> np.ndarray:
        rng = np.random.default_rng((self.seed, int(frame)))
        out = np.asarray(rgb)[:, :, :3].astype(np.uint8)
        for corruption in self.corruptions:
            out = corruption.apply(out, rng)
        return out


def read_camera_rgb(observation: Observation, camera: str | None = None) -> tuple[str, np.ndarray] | None:
    """``(sensor_id, (H,W,3) uint8 RGB)`` for a camera payload, or None. Duck-typed over the canonical
    ndarray payload (NuRec/dataset) and an avstack ``ImageData`` (``.rgb_image``) — the read side of the
    sensor-plane seam, so an augmentation needs no backend-specific view."""
    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    payload = data[key]
    rgb = getattr(payload, "rgb_image", None)
    if rgb is None and isinstance(payload, np.ndarray) and payload.ndim == 3 and payload.shape[2] >= 3:
        rgb = payload
    if rgb is None:
        return None
    return key, np.asarray(rgb)[:, :, :3].astype(np.uint8)


def write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation:
    """Return a copy of ``observation`` with sensor ``key``'s image replaced by ``rgb``. An ndarray
    payload is replaced; an ``ImageData``-like payload is shallow-copied with only its pixel buffer
    swapped (the write side of the seam, shared by attacks and augmentations)."""
    payload = observation.sensor_data[key]
    new_rgb = np.asarray(rgb).astype(np.uint8)
    if isinstance(payload, np.ndarray):
        new_payload: Any = new_rgb
    else:
        new_payload = copy.copy(payload)
        new_payload.data = new_rgb
    new_data = dict(observation.sensor_data)
    new_data[key] = new_payload
    return replace(observation, sensor_data=new_data)


def sensor_augmentation(pipeline: AugmentationPipeline | Iterable[Corruption],
                        camera: str | None = None) -> Callable[[Observation], Observation]:
    """Lift an :class:`AugmentationPipeline` (or a list of corruptions) to a ``perturb(obs) -> obs`` that
    corrupts the camera image in place on the Observation. Returns the frame unchanged when there is no
    readable camera. Backend-agnostic (uses :func:`read_camera_rgb` / :func:`write_camera_rgb`)."""
    pipe = pipeline if isinstance(pipeline, AugmentationPipeline) else AugmentationPipeline(pipeline)

    def _perturb(observation: Observation) -> Observation:
        got = read_camera_rgb(observation, camera)
        if got is None:
            return observation
        key, rgb = got
        return write_camera_rgb(observation, key, pipe.apply(rgb, observation.frame))

    return _perturb


def compose(*perturbs: Callable[[Observation], Observation] | None) -> Callable[[Observation], Observation]:
    """Compose perturbs left-to-right: ``compose(attack, augment)(obs) == augment(attack(obs))`` — the
    attack inserts the patch, then the augmentation corrupts the already-attacked frame. ``None`` entries
    are skipped, so ``compose(attack, None)`` is just the attack."""

    def _perturb(observation: Observation) -> Observation:
        for p in perturbs:
            if p is not None:
                observation = p(observation)
        return observation

    return _perturb
