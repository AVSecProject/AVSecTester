"""Realistic 2D patch insertion — a **simulation** concern (how an inserted patch renders), shared by
all backends. Not attack logic: warping a texture onto a surface and harmonizing it to the scene is
modelling how the object would appear in the render; the *attack* only chooses the payload (which
texture) and the target (where) — see :mod:`avsectester.attacks.patch.physical_patch`.

The patch is composited **on the rendered frame**, never baked into the world/reconstruction:

    project the target surface quad -> perspective-warp the patch onto it -> harmonize to fit.

Each backend supplies only the clean frame + the target's image-space quad (from its pose/geometry via
a per-backend adapter, e.g. ``carla.lead_rear_quad`` / ``detector_quad``); the warp + harmonization
are shared here. Harmonization is pluggable:
  * ``ClassicHarmonizer`` — OpenCV Poisson blend + Reinhard color transfer; in-process, reliable, no
    heavy deps, texture-preserving (keeps the patch's gradients, matches color/lighting to the scene).
  * ``PCTNetHarmonizer`` — a learned harmonizer (libcom's PCTNet) run **in-process**: PCTNet needs only
    torch/torchvision/numpy/einops, so we load just its module from the ``third_party/libcom`` submodule
    (bypassing libcom's diffusers-laden package ``__init__``) and run it in this process.

cv2/torch/skimage are imported lazily, so this module imports without them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from avsectester.plane import Observation
from avsectester.insertion import Insertion
from avsectester.rendering.cameras import Camera, project_to_pixels, cam_coords, carla_cam_coords
from avsectester.rendering.types import InsertionGeometry, GeometryProvider, EvidenceProvider
from avsectester.simulators.viz import View, camera_view

from avsectester.rendering.harmonizers import Harmonizer, ClassicHarmonizer, PCTNetHarmonizer


# ---------------------------------------------------------------------------------------------------
# Geometry: projection (backend adapters convert world -> camera coords, then call these)
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# Warp + composite
# ---------------------------------------------------------------------------------------------------
def warp_patch(frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray):
    """Perspective-warp ``patch_rgba`` onto the image quad ``dst_quad`` (4x2, TL,TR,BR,BL); composite.

    Returns ``(composite_rgb uint8, mask uint8 HxW)`` — the patch alpha-blended over the frame.
    """
    import cv2

    h, w = frame_rgb.shape[:2]
    ph, pw = patch_rgba.shape[:2]
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, np.asarray(dst_quad, dtype=np.float32))
    warped = cv2.warpPerspective(patch_rgba, homography, (w, h), flags=cv2.INTER_LINEAR)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    comp = frame_rgb.astype(np.float32) * (1 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    mask = (warped[:, :, 3] > 127).astype(np.uint8) * 255
    return comp.astype(np.uint8), mask


def render_plane(
    frame_rgb: np.ndarray,
    camera: Any,
    cam_from_world: np.ndarray,
    corners_world: np.ndarray,
    texture_rgba: np.ndarray,
    soften: float = 0.0,
    margin: int = 2,
):
    """Render a textured planar rectangle anchored in the **world** into the frame; composite it.

    The world-anchored counterpart of :func:`warp_patch`: instead of an image quad, the target is a
    3-D rectangle ``corners_world`` (4x3, TL,TR,BR,BL as seen from the front) and a real ``camera``
    (:mod:`avsectester.rendering.cameras` — pinhole or f-theta fisheye). Every pixel in the
    rectangle's image footprint is cast as a ray and intersected with the plane, so the result is exact
    for a distorting lens (a 4-corner homography is not). The texture is pre-filtered to its on-screen
    size (no aliasing on distant objects); ``soften`` (px sigma) optionally blurs it to match a soft
    render. Returns ``(composite_rgb uint8, mask uint8 HxW)`` like :func:`warp_patch`, or the frame and
    an empty mask when the plane is behind the camera / out of view.
    """
    import cv2
    from avsectester.insertion import PlaneSurface
    from avsectester.rendering.geometry import prepare_surface, sample_image

    if frame_rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError("Frame dimensions do not match camera calibration")
    sampler = prepare_surface(PlaneSurface(corners_world, texture_rgba), camera, cam_from_world)
    sampled, _ = sample_image(sampler, camera, margin=margin)
    sampled = sampled.astype(np.float32)
    if soften > 0:
        bounds = sampler.bounds or (0, 0, camera.width, camera.height)
        padding = margin - 2 if sampler.bounds is not None else 0
        x0, y0 = max(0, bounds[0] - padding), max(0, bounds[1] - padding)
        x1, y1 = min(camera.width, bounds[2] + padding), min(camera.height, bounds[3] + padding)
        if x1 > x0 and y1 > y0:
            sampled[y0:y1, x0:x1] = cv2.GaussianBlur(sampled[y0:y1, x0:x1], (0, 0), soften)
    alpha = sampled[:, :, 3:4].astype(np.float32) / 255
    composite = frame_rgb.astype(np.float32) * (1 - alpha) + sampled[:, :, :3] * alpha
    mask = (sampled[:, :, 3] > 127).astype(np.uint8) * 255
    return composite.astype(np.uint8), mask


# ---------------------------------------------------------------------------------------------------
# The compositor
# ---------------------------------------------------------------------------------------------------
class PatchCompositor:
    """Warp a patch onto an image-space quad and harmonize it — backend-agnostic."""

    def __init__(self, harmonizer: Harmonizer | None = None) -> None:
        self.harmonizer = harmonizer or ClassicHarmonizer()

    def apply(
        self, frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray
    ) -> np.ndarray:
        """Insert ``patch_rgba`` at image quad ``dst_quad`` in ``frame_rgb``; return the patched frame."""
        composite, mask = warp_patch(frame_rgb, dst_quad, patch_rgba)
        return self.harmonizer(composite, mask, frame_rgb)

    def apply_planes(
        self,
        frame_rgb: np.ndarray,
        camera: Any,
        cam_from_world: np.ndarray,
        planes: list,
        soften: float = 0.0,
    ) -> np.ndarray:
        """Render world-anchored textured ``planes`` — ``[(corners_world 4x3, texture_rgba), ...]``,
        drawn in order (later ones on top) — with :func:`render_plane`, then harmonize them into the
        frame together as one object. Returns the clean frame when nothing is in view."""
        # TODO: use scene depth or an occlusion mask to hide insertions behind foreground objects.
        comp, union = frame_rgb, np.zeros(frame_rgb.shape[:2], np.uint8)
        for corners, texture in planes:
            comp, mask = render_plane(comp, camera, cam_from_world, corners, texture, soften=soften)
            union |= mask
        return self.harmonizer(comp, union, frame_rgb) if union.any() else frame_rgb


def render_resolved(frame, camera, cam_from_world, resolved, evidence=None, *, compositor=None):
    """Composite a resolved insertion with per-pixel surface depth and visibility.

    ``evidence`` is the estimator result for this same pose and camera. Its visible
    mask prevents the insertion from painting over foreground geometry. Without
    evidence, only the insertion's own surfaces occlude one another. An optional
    compositor supplies harmonization, applied within the visible region only.

    Returns ``(RGB image, uint8 mask)``. Unknown visibility is an explicit error,
    because treating missing evidence as visible would silently change the scene.
    """
    from avsectester.rendering.geometry import insertion_image

    shape = frame.shape[:2]
    if shape != (camera.height, camera.width):
        raise ValueError("Frame dimensions do not match camera calibration")
    if evidence is not None:
        if evidence.visibility is None:
            raise ValueError(f"Insertion visibility is unknown: {evidence.reason}")
        if evidence.visible_mask.shape != shape:
            raise ValueError("Visibility mask dimensions do not match the image")
    if evidence is not None and getattr(evidence, "sampled_rgba", None) is not None:
        rgba, depth = evidence.sampled_rgba, evidence.target_depth
    else:
        rgba, depth = insertion_image(resolved, camera, cam_from_world)
    active = np.isfinite(depth)
    if evidence is not None:
        active &= evidence.visible_mask
    output = frame.copy()
    alpha = rgba[active, 3:4].astype(np.float32) / 255
    output[active] = (frame[active] * (1 - alpha) + rgba[active, :3] * alpha).astype(np.uint8)
    union = active.astype(np.uint8) * 255
    if compositor is not None and active.any():
        harmonized = compositor.harmonizer(output, union, frame)
        output[active] = harmonized[active]
    return output, union


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 image points into (TL, TR, BR, BL) — the warp's quad order.

    Geometric ordering (by pixel position, not physical-corner identity) so the warped patch stays
    upright and unmirrored regardless of the target's world orientation relative to the camera.
    """
    p = np.asarray(pts, dtype=np.float64)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]  # x - y
    return np.stack([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])


def box_to_quad(
    box,
    width_frac: float = 0.6,
    height_frac: float = 0.5,
    v_center: float = 0.5,
    yaw: float = 0.0,
    aspect: float | None = None,
) -> np.ndarray:
    """Approximate a target plane's image quad (TL,TR,BR,BL) from a 2-D detection ``box`` [x1,y1,x2,y2].

    For a surface seen roughly head-on (e.g. a lead vehicle's rear), the plane's image quad is a
    centered sub-rectangle of the detection box — this is the *planar-warp* target, no depth needed:
    :func:`warp_patch` then homography-maps the patch onto it (every pixel, exact for a plane).
    ``width_frac`` / ``height_frac`` size the patch within the box, ``v_center`` places it vertically
    (0=top, 1=bottom), and ``yaw`` (radians) foreshortens one side into a trapezoid for an oblique view.
    ``aspect`` (height/width, in pixels) overrides ``height_frac`` to keep the patch's shape, e.g. 1.0
    for a square sign face.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, y1 + v_center * bh
    hw = 0.5 * width_frac * bw
    hh = hw * aspect if aspect is not None else 0.5 * height_frac * bh
    # yaw>0 pushes the right edge back (narrower) -> a trapezoid, approximating an oblique plane
    l, r = 1.0 + math.sin(yaw), 1.0 - math.sin(yaw)
    return np.array(
        [
            [cx - hw, cy - hh * l],
            [cx + hw, cy - hh * r],
            [cx + hw, cy + hh * r],
            [cx - hw, cy + hh * l],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------------------------------
# View wrappers — plug the insertion into the standard viz pipeline (backend-agnostic)
# ---------------------------------------------------------------------------------------------------
def detector_quad(
    detect: Callable[[Any], Any],
    base: View = camera_view,
    width_frac: float = 0.7,
    height_frac: float = 0.55,
    v_center: float = 0.5,
    yaw: float = 0.0,
    central: float = 0.25,
    aspect: float | None = None,
    pick: str = "largest",
) -> Callable[[Observation], Any]:
    """Return ``quad_of(observation) -> (4,2) | None``: the rear-face quad of the lead vehicle,
    approximated from a 2-D detector box (image-space, no depth). Runs ``detect(rgb) -> [(xyxy, score,
    label)]`` on the ``base`` view, picks the largest box near the image centre (the lead), and turns it
    into a planar-warp quad via :func:`box_to_quad`. Feeds :func:`composite_view` — the NuRec/real-
    imagery analog of the geometric ``carla.lead_rear_quad``.

    ``pick="largest"`` (default) takes the biggest box within ``central`` of the image centre, which can
    hop to a large car in the next lane as it overtakes; ``pick="lane"`` takes only boxes spanning the
    image's centre column (the ego lane) and, of those, the nearest (lowest bottom edge)."""

    def _quad_of(observation: Observation) -> Any:
        rgb = base(observation)
        if rgb is None:
            return None
        w = rgb.shape[1]
        # TODO: associate vehicle identities across frames; per-frame ranking can switch targets.
        best = None
        for box, _score, _label in detect(rgb):
            cx = (box[0] + box[2]) / 2.0
            if pick == "lane":
                key = box[3]  # nearest in-lane vehicle: lowest bottom edge
                ok = box[0] < w / 2 < box[2]
            else:
                key = (box[2] - box[0]) * (box[3] - box[1])
                ok = abs(cx - w / 2) < central * w
            if ok and (best is None or key > best[1]):
                best = (box, key)
        if best is None:
            return None
        return box_to_quad(
            best[0],
            width_frac=width_frac,
            height_frac=height_frac,
            v_center=v_center,
            yaw=yaw,
            aspect=aspect,
        )

    return _quad_of


def hold_quad(
    quad_of: Callable[[Observation], Any], frames: int = 3
) -> Callable[[Observation], Any]:
    """Wrap ``quad_of`` to reuse the last quad for up to ``frames`` consecutive misses, so a single
    missed detection does not make the inserted object blink out of the sequence.
    This holds image coordinates only; it does not associate vehicle identities."""
    state = {"quad": None, "age": 0}

    def _quad_of(observation: Observation) -> Any:
        quad = quad_of(observation)
        if quad is not None:
            state["quad"], state["age"] = quad, 0
            return quad
        state["age"] += 1
        return state["quad"] if state["age"] <= frames else None

    return _quad_of


def composite_view(
    compositor: Any,
    patch_rgba: Any,
    quad_of: Callable[[Observation], Any],
    base: View = camera_view,
) -> View:
    """Wrap a camera ``base`` view to insert a harmonized patch onto the rendered frame.

    The mirror of :func:`avsectester.simulators.viz.detections_view`, at the pixel-insertion layer:
    ``quad_of(observation)`` gives the target surface as a ``(4, 2)`` image-space quad (TL, TR, BR, BL)
    — the backend-specific projection is injected (e.g. :func:`avsectester.simulators.carla.lead_rear_quad`
    or :func:`detector_quad`) — and ``compositor`` (a :class:`PatchCompositor`) warps + harmonizes
    ``patch_rgba`` onto it. Returns the clean frame unchanged when ``quad_of`` yields None."""

    def _view(observation: Observation) -> Any:
        rgb = base(observation)
        if rgb is None:
            return None
        quad = quad_of(observation)
        return rgb if quad is None else compositor.apply(rgb, quad, patch_rgba)

    return _view


def frame_perturbation(
    insert: Callable[[Observation, Any], Any], camera: str | None = None, base: View | None = None
) -> Callable[[Observation], Observation]:
    """Return ``perturb(obs) -> obs`` that rewrites one camera's frame with ``insert(obs, rgb) -> rgb``.

    The sensor-plane attack seam for backends whose camera payload is a raw RGB ndarray (NuRec): the
    AV stack itself perceives the inserted object (unlike :func:`composite_view`, which only paints the
    visualization). ``carla.camera_patch_perturbation`` is the CARLA ``ImageData`` analogue.
    """
    from dataclasses import replace

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


class InsertionRenderer:
    """Resolve bound objects and render them through the same geometry used by selection.

    ``geometry(observation)`` returns :class:`InsertionGeometry`. Tuple returns remain supported. An optional
    ``evidence_provider(observation, resolved, geometry)`` returns visibility evidence by insertion
    ID. No filters run here. Use ``frame_perturbation`` or a backend adapter to update model inputs.
    """

    def __init__(
        self,
        insertions: Sequence[Insertion],
        camera: Camera,
        geometry: GeometryProvider,
        compositor: PatchCompositor | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ):
        self.insertions = tuple(insertions)
        if len({item.id for item in self.insertions}) != len(self.insertions):
            raise ValueError("Insertion IDs must be unique")
        self.camera = camera
        self.geometry = geometry
        self.compositor = compositor
        self.evidence_provider = evidence_provider
        self.resolved = ()
        self.evidence = {}

    def __call__(self, observation, rgb):
        from avsectester.insertion import resolve_insertion

        state = self.geometry(observation)
        if not isinstance(state, InsertionGeometry):
            state = InsertionGeometry(*state)
        self.resolved = tuple(
            resolve_insertion(item, state.actors, state.victim) for item in self.insertions
        )
        self.evidence = (
            self.evidence_provider(observation, self.resolved, state)
            if self.evidence_provider is not None
            else {}
        )
        result = rgb
        for item in self.resolved:
            result, _ = render_resolved(
                result,
                self.camera,
                state.cam_from_world,
                item,
                self.evidence[item.id] if self.evidence_provider else None,
                compositor=self.compositor,
            )
        return result
