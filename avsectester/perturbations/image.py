"""Scenario augmentation — controlled corruptions of the camera stream, for attack-robustness testing.

An attack is only *robust* if it survives the nuisance variation a deployed AV sees: weather, lighting,
and sensor/optics degradation. This module provides that variation as image-space :class:`Corruption`
operators, lifted to the same ``perturb: Observation -> Observation`` seam an attack uses
(:func:`avsectester.backend.run`). So an augmentation composes with an attack — ``compose(attack, aug)``
inserts the patch and *then* corrupts the frame — and the two can be diffed under one condition to see
whether the attack still fires (see ``docs/guides/conditions.md`` and ``docs/guides/evaluation.md``).

Backend-agnostic: a :class:`Corruption` transforms an ``(H, W, 3)`` uint8 RGB array, so this works on
CARLA, in-process NuRec, and dataset replay (nuScenes/nuRec recorded frames) alike. CARLA additionally
supports *world-level* weather (native ``carla.WeatherParameters`` at reset) — see
:mod:`avsectester.closed_loop.simulators.carla`.

Every operator takes ``severity`` in ``[0, 1]`` (0 ≈ identity) and a seeded ``numpy.random.Generator``,
so runs are reproducible and the *same* corruption can be applied to the clean and attacked run of a pair
(:class:`AugmentationPipeline` seeds from ``(seed, frame)``). ``cv2`` is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import numpy as np


from avsectester.closed_loop import datatypes
from avsectester.closed_loop.datatypes import Observation

# ----------------------------------------------------------------------------------------------------
# Corruption operators
# ----------------------------------------------------------------------------------------------------


# These operator aliases keep the existing robustness/CLI and deferred concrete-attack
# entry points working; new environment configurations use closed_loop.sensor_transforms.
from avsectester.closed_loop.sensor_transforms import (
    Corruption as Corruption,
    Fog as Fog,
    Rain as Rain,
    Snow as Snow,
    Brightness as Brightness,
    LowLight as LowLight,
    Contrast as Contrast,
    Gamma as Gamma,
    ColorTemp as ColorTemp,
    GaussianNoise as GaussianNoise,
    ShotNoise as ShotNoise,
    MotionBlur as MotionBlur,
    DefocusBlur as DefocusBlur,
    JPEGCompression as JPEGCompression,
    ChromaticAberration as ChromaticAberration,
    CORRUPTIONS as CORRUPTIONS,
)


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


# ----------------------------------------------------------------------------------------------------
# Optional Albumentations backend
# ----------------------------------------------------------------------------------------------------
# Battle-tested implementations (Albumentations, ~15k stars, paper) for the weather/sensor operators,
# behind the same :class:`Corruption` interface. Optional dependency (`augment` extra) — the hand-rolled
# operators above are the zero-dependency fallback.
#
# Backend: **classic albumentations (>=1.4,<2, MIT)**, installed via the `augment` extra. It imports as
# ``import albumentations as A``. This stack is numpy<2 (torch 2.1 / OpenMMLab-official), so we use classic
# albumentations rather than AlbumentationsX/2.x (which require numpy>=2). Same transforms + API; the
# builder parameter names below target the 1.4.x API (e.g. ``blur_limit``, ``radius``, ``color_shift``).

def albumentations_available() -> bool:
    """True if the (compatible) Albumentations backend can be imported."""
    try:
        import albumentations  # noqa: F401
    except ImportError:
        return False
    return True


class AlbumentationsCorruption(Corruption):
    """Wrap an Albumentations transform as a :class:`Corruption`.

    ``build(severity) -> A.BasicTransform`` maps our ``severity`` to the transform's own parameters.
    Determinism (so the clean and attacked run of a pair match) comes from seeding ``A.Compose`` with a
    value drawn from *our* rng — the :class:`AugmentationPipeline`'s ``(seed, frame)`` seeding therefore
    still controls the realisation. ``category`` and ``label`` are for reporting only."""

    def __init__(self, build: Callable[[float], Any], severity: float = 0.5,
                 category: str = "sensor", label: str = "albumentations") -> None:
        super().__init__(severity)
        self._build = build
        self.category = category
        self._label = label

    @property
    def name(self) -> str:
        return f"alb:{self._label}@{self.severity:.2f}"

    def apply(self, rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import albumentations as A

        seed = int(rng.integers(0, 2**31 - 1))
        transform = A.Compose([self._build(self.severity)], seed=seed)
        return transform(image=np.ascontiguousarray(rgb))["image"]


def _odd(n: float) -> int:
    n = max(3, round(n))
    return n if n % 2 == 1 else n + 1


def _albumentations_builders() -> dict[str, tuple[str, Callable[[float], Any]]]:
    """``name -> (category, build(severity))`` for the Albumentations-backed operators (imports A so the
    builder lambdas capture it). Severity in ``[0, 1]`` maps to each transform's native parameters.
    Parameter names target classic albumentations 1.4.x (this stack is numpy<2)."""
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


def albumentations_corruptions(severity: float = 0.5) -> list[AugmentationPipeline]:
    """Albumentations-backed robustness suite — one single-corruption pipeline per operator at
    ``severity`` (the parallel of :func:`common_corruptions`, using Albumentations implementations).
    Raises if the backend is not installed."""
    if not albumentations_available():
        raise RuntimeError("Albumentations backend not installed; `pip install 'albumentations>=1.4,<2'`")
    return [AugmentationPipeline([AlbumentationsCorruption(build, severity, cat, label=name)])
            for name, (cat, build) in _albumentations_builders().items()]




def sensor_augmentation(pipeline: AugmentationPipeline | Iterable[Corruption],
                        camera: str | None = None) -> Callable[[Observation], Observation]:
    """Lift an :class:`AugmentationPipeline` (or a list of corruptions) to a ``perturb(obs) -> obs`` that
    corrupts the camera image in place on the Observation. Returns the frame unchanged when there is no
    readable camera. Backend-agnostic (uses :mod:`avsectester.closed_loop.datatypes` camera helpers)."""
    pipe = pipeline if isinstance(pipeline, AugmentationPipeline) else AugmentationPipeline(pipeline)

    def _perturb(observation: Observation) -> Observation:
        got = datatypes.read_camera_rgb(observation, camera)
        if got is None:
            return observation
        key, rgb = got
        return datatypes.write_camera_rgb(observation, key, pipe.apply(rgb, observation.frame))

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



def frame_perturbation(
    insert: Callable[[Observation, Any], Any], camera: str | None = None, base: Callable[[Observation], Any] | None = None
) -> Callable[[Observation], Observation]:
    """Return ``perturb(obs) -> obs`` that rewrites one camera's frame with ``insert(obs, rgb) -> rgb``.

    The sensor-plane attack seam for backends whose camera payload is a raw RGB ndarray (NuRec): the
    AV stack itself perceives the inserted object. ``carla.insertion_perturbation`` is the
    CARLA ``ImageData`` analogue. A visualization callback alone does not change model input.
    """
    from avsectester.closed_loop.simulators.viz import camera_view

    def _perturb(observation: Observation) -> Observation:
        data = observation.sensor_data
        if not data:
            return observation
        key = camera if camera in data else next(iter(data))
        rgb = camera_view(observation, key) if base is None else base(observation)
        if rgb is None:
            return observation
        new_data = dict(data)
        new_data[key] = insert(observation, rgb)
        return replace(observation, sensor_data=new_data)

    return _perturb
