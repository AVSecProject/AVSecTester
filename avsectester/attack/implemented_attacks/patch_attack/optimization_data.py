"""Fresh optimization samples obtained through the closed-loop service boundary."""
from __future__ import annotations

import random
from copy import deepcopy
import hashlib

import numpy as np

from avsectester.closed_loop.datatypes import read_camera_rgb
from avsectester.closed_loop.services import RenderRequest
from avsectester.attack.optimization import AdversarialSample, DataSource, TargetSpec


def plane_sampling_grid(view, corners):
    """Return an image-sized texture grid and geometric mask for a calibrated plane.

    Pixel centres are unprojected through the actual lens, then intersected with the plane.
    Grid coordinates use ``grid_sample(align_corners=False)``: texture edges are -1/+1.
    This works for both pinhole and f-theta cameras; four projected corners cannot describe
    the curved footprint of a wide-angle lens.
    """
    from avsectester.attack.insertion.geometry import NEAR_Z, camera_ray_mask
    from avsectester.geometry.cameras import transform

    corners = transform(view.cam_from_world, np.asarray(corners, dtype=float))
    origin = corners[0]
    u, v = corners[1] - origin, corners[3] - origin
    normal = np.cross(u, v)
    basis = np.column_stack((u, v))
    gram = basis.T @ basis
    if not np.isfinite(corners).all() or np.linalg.det(gram) <= 1e-18:
        raise ValueError("Optimization needs a finite nondegenerate patch plane")
    height, width = view.data.shape[:2]
    if (height, width) != (view.camera.height, view.camera.width):
        raise ValueError("Optimization camera calibration and image dimensions differ")
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
    rays = np.asarray(view.camera.unproject(pixels), dtype=float)
    denominator = rays @ normal
    distance = np.divide(origin @ normal, denominator,
                         out=np.full(len(rays), np.inf), where=np.abs(denominator) > 1e-12)
    valid = camera_ray_mask(view.camera, rays) & np.isfinite(distance) & (distance > 0)
    valid &= distance * rays[:, 2] > NEAR_Z
    offsets = rays * np.where(valid, distance, 0.)[:, None] - origin
    coordinates = offsets @ basis @ np.linalg.inv(gram)
    valid &= np.isfinite(coordinates).all(axis=1)
    valid &= ((coordinates >= 0.) & (coordinates <= 1.)).all(axis=1)
    # Invalid rays stay out of the texture. In particular NaNs must never reach autograd.
    grid = np.where(valid[:, None], coordinates * 2. - 1., 2.)
    return (grid.reshape(height, width, 2).astype(np.float32),
            valid.reshape(1, height, width).astype(np.float32))


def projected_host_box(view, host_id):
    """Project the selected host's full box into the same camera as the scored image."""
    from itertools import product
    from avsectester.geometry.cameras import transform

    if host_id not in view.actors:
        raise ValueError(f"Optimization host {host_id!r} is missing from the selected view")
    host = view.actors[host_id]
    if host.extent is None:
        raise ValueError(f"Optimization host {host_id!r} bounding-box extent is unavailable")
    corners = np.asarray(tuple(product((-0.5, 0.5), repeat=3))) * host.extent
    camera_points = transform(view.cam_from_world @ host.transform, corners)
    if (camera_points[:, 2] <= 0).any():
        raise ValueError("Optimization host box crosses the camera plane")
    pixels = np.asarray(view.camera.project(camera_points))
    if pixels.shape != (8, 2) or not np.isfinite(pixels).all():
        raise ValueError("Optimization host box has an undefined camera projection")
    lower = np.maximum(pixels.min(axis=0), (0, 0))
    upper = np.minimum(pixels.max(axis=0), (view.camera.width, view.camera.height))
    if (upper <= lower).any():
        raise ValueError("Optimization host box is outside the camera image")
    return tuple(float(v) for v in (*lower, *upper))


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
        from avsectester.attack.insertion.assets import load_rgba_image

        if not self.attack.emissive:
            raise ValueError("Native patch optimization requires self-emissive deployment")
        # This instance owns immutable attack/environment settings, so no cache crosses attacks.
        dimensions = tuple(getattr(self.attack, name, None) for name in
                           ("prop", "width_m", "height_m", "emissive", "sensor", "orientation"))
        alpha_source = load_rgba_image(self.attack.assets)[..., 3]
        identity = repr((segment.scenario_id, frame.position, segment.roles, segment.placement,
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
                                   if self.session.can_deploy_at(frame.position))
            if self.at_segment_start:
                frame = segment.start
                if physical and not self.session.can_deploy_at(frame.position):
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
                    hosts = segment.roles.get("host", ())
                    insertion = self.attack.insertion(self.attack.assets,
                        host=hosts[0] if hosts else None, placement=segment.placement)
                    action = InsertionAttackAction(self.attack.sensor, insertion)
                    resolved = action.resolve(view)
                    surfaces = resolved.planes()
                    if len(surfaces) != 1:
                        raise ValueError("Patch optimization requires exactly one planar surface")
                    grid, mask = plane_sampling_grid(view, surfaces[0].corners)
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
                hosts = segment.roles.get("host", ())
                if len(hosts) != 1:
                    raise ValueError("Default patch objective requires one selected host")
                box = projected_host_box(view, hosts[0])
            result.append(AdversarialSample(
                x=torch.as_tensor(captured[1].copy(), device=self.device).permute(2, 0, 1).float() / 255,
                target=TargetSpec(tuple(float(v) for v in box)),
                context={"sampling_grid": grid, "mask": mask, "frame": frame, "segment": segment,
                         "observation": capture.observation,
                         "brightness": self._rng.uniform(*self.brightness),
                         "contrast": self._rng.uniform(*self.contrast)},
            ))
        return result
