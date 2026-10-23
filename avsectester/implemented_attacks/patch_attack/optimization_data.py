"""Fresh optimization samples obtained through the closed-loop service boundary."""
from __future__ import annotations

import random
from copy import deepcopy
import hashlib

import numpy as np

from avsectester.data_types import read_camera_rgb
from avsectester.execution.services import RenderRequest
from avsectester.attack.optimization import AdversarialSample, DataSource
from avsectester.data_types import TargetSpec
from avsectester.geometry import projection


def native_sampling_grid(black, white):
    """Measure a native prop's visible footprint from two self-emissive captures.

    Captures share their frame, environment, alpha and deployment. A local black/white
    response distinguishes the prop from unchanged background colors, including vegetation.
    The digital surrogate stretches the texture over the footprint's image rectangle and
    preserves the measured occlusion mask; it does not recover the native mesh's UV map.
    """
    low, high = (np.asarray(image, dtype=np.float32) for image in (black, white))
    if (low.shape != high.shape or low.ndim != 3 or low.shape[-1] != 3
            or not np.isfinite(low).all() or not np.isfinite(high).all()):
        raise ValueError("Native patch probes must have equal finite RGB image dimensions")
    response = high - low
    active = (response > 8).all(axis=-1)
    if not active.any():
        raise ValueError("Native patch probes contain no stable visible texture pixels")
    ys, xs = np.nonzero(active)
    left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    grid = np.full((*active.shape, 2), 2., dtype=np.float32)
    grid[active, 0] = ((xs + .5 - left) / (right - left)) * 2 - 1
    grid[active, 1] = ((ys + .5 - top) / (bottom - top)) * 2 - 1
    return grid, active[None].astype(np.float32)


class SessionDataSource(DataSource):
    """Read one capture per image sample and cache required native-prop probe captures.

    Physical probes are cached within this data source only; its attack, session seed,
    environment, alpha, placement and prop mesh remain fixed for that optimization.
    """

    def __init__(self, session, attack, metadata, *, size, device, target_box=None,
                 brightness=(0.8, 1.2), contrast=(0.8, 1.2), seed=0, at_segment_start=False,
                 environment=None, require_patch_grid=True):
        if not metadata:
            raise ValueError("Patch optimization requires selected segments")
        self.session, self.attack, self.metadata = session, attack, tuple(metadata)
        self.size, self.device, self.target_box = tuple(size), device, target_box
        self.brightness, self.contrast = brightness, contrast
        self.at_segment_start = at_segment_start
        self.require_patch_grid = require_patch_grid
        self.environment = deepcopy({} if environment is None else environment)
        self._rng = random.Random(seed)
        self._native_samples = {}

    def _physical_capture(self, segment, frame):
        from .offline import render_candidate
        from avsectester.rendering.textures import load_rgba_image

        if not self.attack.emissive:
            raise ValueError("Native patch optimization requires self-emissive deployment")
        # This instance owns immutable attack/environment settings, so no cache crosses attacks.
        dimensions = tuple(getattr(self.attack, name, None) for name in
                           ("prop", "width_m", "height_m", "emissive", "sensor", "orientation"))
        alpha_source = load_rgba_image(self.attack.assets)[..., 3]
        identity = repr((segment.scenario_id, frame.position, segment.user_data,
                         getattr(self.session, "seed", None), self.environment, dimensions, self.size))
        key = (identity, hashlib.sha256(alpha_source.tobytes()).digest())
        if key not in self._native_samples:
            height, width = self.size
            alpha = alpha_source
            import cv2
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
            black = np.zeros((height, width, 4), np.uint8)
            black[..., 3] = alpha
            white = black.copy()
            white[..., :3] = 255
            textures = [black, white]
            captures = [render_candidate(self.session, self.attack, segment, frame, texture,
                                         self.environment) for texture in textures]
            rgbs = []
            for capture in captures:
                item = read_camera_rgb(capture.observation, capture.view.sensor)
                if item is None:
                    raise ValueError("Native patch probe has no selected RGB capture")
                rgbs.append(item[1].copy())
            grid, mask = native_sampling_grid(rgbs[0], rgbs[1])
            self._native_samples[key] = captures[-1], grid, mask
        return self._native_samples[key]

    def sample(self, batch):
        import torch
        from avsectester.attack.insertion.action import InsertionAttackAction

        result = []
        for _ in range(batch):
            segment = self._rng.choice(self.metadata)
            frames = self.session.services.dataset.frames(segment.scenario_id)
            physical = segment.user_data.get("deployment") == "world"
            candidates = frames[segment.start.position:segment.stop.position]
            if physical:
                candidates = tuple(frame for frame in candidates
                                   if self.session.can_start_at(frame.position))
            if self.at_segment_start:
                frame = segment.start
                if physical and not self.session.can_start_at(frame.position):
                    raise ValueError("Selected patch start cannot preserve the deployment window")
            elif candidates:
                frame = self._rng.choice(candidates)
            else:
                raise ValueError("Patch segment has no frame that preserves the deployment window")
            if physical and self.require_patch_grid:
                capture, grid, mask = self._physical_capture(segment, frame)
            else:
                capture = self.session.render(RenderRequest(segment.scenario_id, frame,
                                            self.attack.sensor, environment=self.environment))
                grid, mask = None, None
                if self.require_patch_grid:
                    view = capture.view
                    insertion = self.attack.insertion(self.attack.assets,
                        placement=self.attack.selected_placement(segment))
                    action = InsertionAttackAction(self.attack.sensor, insertion)
                    resolved = action.resolve(view)
                    surfaces = resolved.planes()
                    if len(surfaces) != 1:
                        raise ValueError("Patch optimization requires exactly one planar surface")
                    if view.camera is None or view.cam_from_world is None:
                        raise ValueError("Optimization requires a calibrated camera view")
                    if view.data.shape[:2] != (view.camera.height, view.camera.width):
                        raise ValueError("Optimization camera calibration and image dimensions differ")
                    grid, mask = projection.plane_sampling_grid(
                        view.camera, view.cam_from_world, surfaces[0].corners)
                    evidence = action.evidence(view, resolved)
                    if evidence.visibility is None:
                        raise ValueError(f"Patch optimization visibility is unavailable: {evidence.reason}")
                    mask *= np.asarray(evidence.visible_mask, dtype=np.float32)[None]
            view = capture.view
            captured = read_camera_rgb(capture.observation, view.sensor)
            if captured is None or captured[0] != view.sensor:
                raise ValueError("Optimization requires the selected RGB camera")
            if mask is not None and not mask.any():
                raise ValueError("Patch has no visible pixels in the optimization camera")
            box = self.target_box
            if box is None:
                host = segment.user_data.get("host")
                if not isinstance(host, str) or not host:
                    raise ValueError("Default patch objective requires one selected host")
                if host not in view.actors:
                    raise ValueError(f"Optimization host {host!r} is missing from the selected view")
                actor = view.actors[host]
                if actor.extent is None:
                    raise ValueError(f"Optimization host {host!r} bounding-box extent is unavailable")
                if view.camera is None or view.cam_from_world is None:
                    raise ValueError("Optimization requires a calibrated camera view")
                box = projection.project_box(view.camera, view.cam_from_world, actor.box_transform(), actor.extent)
                if box is None:
                    raise ValueError("Optimization host box is behind or outside the camera image")
            result.append(AdversarialSample(
                x=torch.as_tensor(captured[1].copy(), device=self.device).permute(2, 0, 1).float() / 255,
                target=TargetSpec(tuple(float(v) for v in box)),
                context={"sampling_grid": grid, "mask": mask, "frame": frame, "segment": segment,
                         "observation": capture.observation,
                         "brightness": self._rng.uniform(*self.brightness),
                         "contrast": self._rng.uniform(*self.contrast)},
            ))
        return result
