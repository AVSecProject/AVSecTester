"""Patch-specific sampling, preparation and optional candidate rollout validation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from avsectester.attack.optimization import GradientOptimizer, HideObject, SessionScorer
from avsectester.data_types import read_camera_rgb
from avsectester.rendering.textures import Texture
from avsectester.execution.runner import RecordingPolicy
from avsectester.execution.services import CandidateRequest, RenderRequest
from .optimization_data import SessionDataSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchOptimization:
    """Typed patch optimization settings; the algorithm owns its optimization seed.

    The closed-loop seed and environment come from ``OfflineContext``. ``init_texture`` is
    an explicit NES warm start. ``target_box`` overrides the default projected host box.
    """

    algorithm: GradientOptimizer
    scorer_id: str | None = None
    size: tuple[int, int] = (64, 64)
    device: str | None = None
    target_box: tuple[float, float, float, float] | None = None
    brightness: tuple[float, float] = (0.8, 1.2)
    contrast: tuple[float, float] = (0.8, 1.2)
    validate_rollout: bool = False
    init_texture: np.ndarray | str | Path | Texture | None = None

    def __post_init__(self) -> None:
        from avsectester.attack.optimization import PGD
        from avsectester.attack.optimization import NES
        if not isinstance(self.algorithm, (PGD, NES)):
            raise TypeError("algorithm must be a PGD or NES optimization algorithm")
        if (len(self.size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                      for v in self.size)):
            raise ValueError("size must contain two positive integers")
        object.__setattr__(self, "size", tuple(self.size))
        if self.device is None:
            object.__setattr__(self, "device", "cuda:0" if isinstance(self.algorithm, PGD) else "cpu")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty device string")
        if self.scorer_id is not None and (not isinstance(self.scorer_id, str) or not self.scorer_id):
            raise ValueError("scorer_id must be a nonempty string or None")
        for name in ("brightness", "contrast"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or not np.isfinite(values).all() or not 0 <= values[0] <= values[1]:
                raise ValueError(f"{name} must contain ordered finite nonnegative bounds")
            object.__setattr__(self, name, values)
        if self.target_box is not None:
            box = tuple(float(v) for v in self.target_box)
            if (len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0]
                    or box[3] <= box[1]):
                raise ValueError("target_box must be a finite nonempty xyxy box")
            object.__setattr__(self, "target_box", box)
        if not isinstance(self.validate_rollout, bool):
            raise TypeError("validate_rollout must be a bool")
        if self.init_texture is not None:
            if isinstance(self.algorithm, PGD):
                raise ValueError("init_texture is supported only by NES, not PGD")
            from avsectester.rendering.textures import load_rgba_image
            object.__setattr__(self, "init_texture", load_rgba_image(self.init_texture))


def render_candidate(session, attack, segment, frame, texture, environment, observation=None):
    """Render through the attack's online factory, and always stop that candidate runtime."""
    from avsectester.attack.runtime import OnlineContext
    from avsectester.execution.services import ensure_fresh

    runtime = ensure_fresh(attack.create_runtime(segment, texture))
    online = OnlineContext(frame.position - segment.start.position, "setup", frame.timestamp,
                           frame.position, observation=observation)
    failure = None
    try:
        actions = tuple(runtime.setup(online))
        online = replace(online, stage="step")
        actions += tuple(runtime.step(online))
        return session.render(RenderRequest(segment.scenario_id, frame, attack.sensor,
                                             actions=actions, environment=environment))
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            runtime.stop(online)
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"Candidate attack cleanup also failed: {error!r}")


def optimize_patch(attack, context, options: PatchOptimization):
    """Optimize selected metadata without selecting or validating its validity again.

    Image insertion uses calibrated ray sampling. Physical insertion samples the actual
    self-emissive prop with native renders; the sampled native footprint is cached per origin.
    Optional rollout validation consumes the resulting logs and fails on incomplete runs.
    """
    from avsectester.attack.optimization import PGD
    from avsectester.attack.optimization import PatchPerturbation

    if not isinstance(options, PatchOptimization):
        raise TypeError("optimization must be a PatchOptimization instance")
    if not context.metadata:
        return attack.assets
    algorithm = options.algorithm
    white_box = isinstance(algorithm, PGD)
    scorer_id = options.scorer_id or ("mmdet.rpn_objectness" if white_box else "mmdet.inference")
    seed = algorithm.seed
    from avsectester.rendering.textures import load_rgba_image
    import cv2
    alpha = cv2.resize(load_rgba_image(attack.assets)[..., 3],
                       (options.size[1], options.size[0]), interpolation=cv2.INTER_NEAREST)

    def export_texture(delta):
        rgba = to_rgba_uint8(delta)
        rgba[..., 3] = alpha
        return rgba
    with context.services.session(context.seed) as session:
        descriptors = {item.id: item for item in session.scorers()}
        if scorer_id not in descriptors:
            raise ValueError(f"Unavailable patch optimization scorer: {scorer_id}")
        if white_box and not descriptors[scorer_id].differentiable_inputs:
            raise ValueError(f"PGD requires a differentiable scorer: {scorer_id}")
        data = SessionDataSource(session, attack, context.metadata, size=options.size,
                                 device=options.device, target_box=options.target_box, seed=seed,
                                 brightness=options.brightness, contrast=options.contrast,
                                 at_segment_start=not white_box, environment=context.environment,
                                 require_patch_grid=white_box)
        scorer = SessionScorer(session, scorer_id, require_grad=white_box,
                               input_format=descriptors[scorer_id].inputs.get("images"),
                               device=options.device)
        if white_box:
            perturbation = PatchPerturbation(size=options.size, device=options.device, seed=seed,
                                              alpha=alpha)
        else:
            class RenderPerturbation:
                def init(self):
                    if options.init_texture is not None:
                        import cv2
                        height, width = options.size
                        rgb = cv2.resize(options.init_texture[..., :3], (width, height))
                        return rgb.transpose(2, 0, 1).astype(np.float32) / 255
                    return np.random.default_rng(seed).random((3, *options.size)).astype(np.float32)

                def apply(self, sample, delta):
                    rendered = render_candidate(session, attack, sample.context["segment"],
                                                sample.context["frame"], export_texture(delta),
                                                context.environment,
                                                sample.context["observation"])
                    captured = read_camera_rgb(rendered.observation, rendered.view.sensor)
                    if captured is None or captured[0] != rendered.view.sensor:
                        raise ValueError("Rendered candidate has no selected RGB camera")
                    return captured[1].copy()

                def project(self, delta):
                    return np.clip(delta, 0, 1)

                def export(self, delta):
                    return export_texture(delta)

            perturbation = RenderPerturbation()
        result = algorithm.run(data, perturbation, scorer, HideObject())
        if result.export is None:
            raise ValueError("Patch optimizer did not export an artifact")
        if options.validate_rollout:
            for segment in context.metadata:
                log = session.rollout(CandidateRequest(segment, context.seed, segment.decision_steps,
                                      context.environment, RecordingPolicy(), attack=attack,
                                      artifacts=result.export))
                if log.status != "completed" or len(log.steps) != segment.decision_steps:
                    raise RuntimeError(f"Optimized patch rollout did not complete: {log.status}")
                final_speed = log.steps[-1].physical_post.ego_speed if log.steps else None
                logger.info("Optimized patch rollout: scenario=%s steps=%d final_speed=%s",
                            segment.scenario_id, len(log.steps), final_speed)
        return result.export


def to_rgba_uint8(delta: np.ndarray) -> np.ndarray:
    """(3, H, W) float in [0,1] -> (H, W, 4) uint8 RGBA (opaque)."""
    rgb = (np.clip(delta, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    alpha = np.full((*rgb.shape[:2], 1), 255, np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def run_offline(attack, context):
    if not context.metadata or attack.optimization is None:
        return attack.assets
    if any(segment.user_data.get("deployment") == "world" for segment in context.metadata) and not attack.emissive:
        raise ValueError("Native CARLA patch optimization requires emissive=True")
    return optimize_patch(attack, context, attack.optimization)


__all__ = [
    'PatchOptimization',
]
