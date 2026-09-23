"""Standard-warp + harmonization patch compositor — one backend-agnostic 2D insert for CARLA & NuRec.

The patch is composited **on the rendered frame**, never baked into the world/reconstruction:

    project the target surface quad -> perspective-warp the patch onto it -> harmonize to fit.

Each backend supplies only the clean frame + camera intrinsics + the target's image-space quad (from
its pose/geometry); the warp + harmonization are shared here. Harmonization is pluggable:
  * ``ClassicHarmonizer`` — OpenCV Poisson blend + Reinhard color transfer; in-process, reliable, no
    heavy deps, texture-preserving (keeps the patch's gradients, matches color/lighting to the scene).
  * ``PCTNetHarmonizer`` — a learned harmonizer (libcom's PCTNet) run **in-process**: PCTNet needs only
    torch/torchvision/numpy/einops, so we load just its module from the ``third_party/libcom`` submodule
    (bypassing libcom's diffusers-laden package ``__init__``) and run it in this process.

cv2/torch/skimage are imported lazily, so this module imports without them.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------
# Geometry: projection (backend adapters convert world -> camera coords, then call these)
# ---------------------------------------------------------------------------------------------------
def project_to_pixels(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Standard pinhole projection: camera-frame points (x right, y down, z forward) -> (N,2) px."""
    uv = (K @ np.asarray(pts_cam, dtype=np.float64).T).T
    return uv[:, :2] / uv[:, 2:3]


def cam_coords(pts_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    """Standard extrinsic: world points -> camera frame (x right, y down, z fwd). NuRec/pinhole."""
    pts = np.asarray(pts_world, dtype=np.float64)
    homog = np.c_[pts, np.ones(len(pts))]
    return (np.asarray(world_to_cam, dtype=np.float64) @ homog.T).T[:, :3]


def carla_cam_coords(pts_world: np.ndarray, cam_inverse_matrix: np.ndarray) -> np.ndarray:
    """CARLA UE convention: world -> camera then reorder to standard (x right, y down, z fwd).

    ``cam_inverse_matrix`` = ``camera.get_transform().get_inverse_matrix()``; UE camera axes are
    (x fwd, y right, z up), so the standard camera frame is ``[y, -z, x]``.
    """
    pc = cam_coords(pts_world, cam_inverse_matrix)  # UE camera coords (x fwd, y right, z up)
    return np.stack([pc[:, 1], -pc[:, 2], pc[:, 0]], axis=1)


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
    alpha = (warped[:, :, 3:4].astype(np.float32) / 255.0)
    comp = frame_rgb.astype(np.float32) * (1 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    mask = (warped[:, :, 3] > 127).astype(np.uint8) * 255
    return comp.astype(np.uint8), mask


# ---------------------------------------------------------------------------------------------------
# Harmonizers
# ---------------------------------------------------------------------------------------------------
class Harmonizer(ABC):
    """Adjust the pasted foreground to fit the background. Must preserve the patch's texture."""

    @abstractmethod
    def __call__(self, composite_rgb: np.ndarray, mask: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
        ...


class ClassicHarmonizer(Harmonizer):
    """Reinhard color transfer (match the patch's Lab mean/std to the scene) + Poisson seamless blend.

    Reliable, in-process, no model weights; texture-preserving (Poisson keeps the patch's gradients
    while shifting color/brightness to the background).
    """

    def __init__(self, color_transfer: bool = True, poisson: bool = True) -> None:
        self.color_transfer = color_transfer
        self.poisson = poisson

    def __call__(self, composite_rgb, mask, background_rgb):
        import cv2

        out = composite_rgb.copy()
        m = mask > 0
        if self.color_transfer and m.any():
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
            bg_lab = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            # dilate the mask to sample the surrounding scene for the target statistics
            ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
            ring &= ~m
            if ring.any():
                for c in range(3):
                    fmu, fsd = lab[m, c].mean(), lab[m, c].std() + 1e-5
                    bmu, bsd = bg_lab[ring, c].mean(), bg_lab[ring, c].std() + 1e-5
                    lab[m, c] = (lab[m, c] - fmu) * (bsd / fsd) + bmu
                out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        if self.poisson and m.any():
            ys, xs = np.where(m)
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            out = cv2.seamlessClone(out, background_rgb, mask, center, cv2.NORMAL_CLONE)
        return out


class PCTNetHarmonizer(Harmonizer):
    """Learned harmonization via libcom's **PCTNet**, run **in-process** (no subprocess, no separate env).

    PCTNet is a self-contained color-transform CNN needing only torch / torchvision / numpy / einops —
    all compatible with the avstack stack (torch 1.13) — so we load just the ``pct_net`` module from the
    ``third_party/libcom`` submodule, bypassing libcom's package ``__init__`` (which eagerly imports
    diffusers-based models). PCTNet gives a milder, texture-preserving harmonization than the classic
    Poisson blend — better for keeping an adversarial pattern intact. Falls back to
    :class:`ClassicHarmonizer` on any error (or set ``strict`` to raise instead).
    """

    _SUBMODULE = "third_party/libcom/libcom/image_harmonization"

    def __init__(self, device: int = 0, weights: str | None = None, strict: bool = False) -> None:
        self.device_id = device
        self.weights = weights
        self.strict = strict
        self._net = None
        self._dev = None
        self._fallback = ClassicHarmonizer()

    def _load(self):
        if self._net is not None:
            return self._net
        import importlib.util
        import sys
        import types
        from pathlib import Path

        import torch

        root = Path(__file__).resolve().parents[2] / self._SUBMODULE
        src = root / "source"
        # dummy parent packages so pct_net's absolute imports resolve WITHOUT running libcom/__init__
        # (which imports diffusers/pytorch-lightning models that conflict with the avstack stack).
        for name in ("libcom", "libcom.image_harmonization", "libcom.image_harmonization.source"):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = []
                sys.modules[name] = mod

        def _load_file(name, path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        _load_file("libcom.image_harmonization.source.functions", src / "functions.py")
        pct = _load_file("libcom.image_harmonization.source.pct_net", src / "pct_net.py")
        weights = Path(self.weights) if self.weights else self._resolve_weights(root)
        self._dev = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        net = pct.PCTNet()
        net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self._net = net.to(self._dev).eval()
        log.info("PCTNet harmonizer loaded in-process (%s, %s)", weights.name, self._dev)
        return self._net

    def _resolve_weights(self, root):
        from pathlib import Path

        cand = root / "pretrained_models" / "PCTNet.pth"
        if cand.exists():
            return cand
        from huggingface_hub import hf_hub_download  # downloaded once, then cached in the submodule
        cand.parent.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download("BCMIZB/Libcom_pretrained_models", "PCTNet.pth",
                                    local_dir=str(cand.parent)))

    def __call__(self, composite_rgb, mask, background_rgb):
        try:
            import cv2
            import torch

            net = self._load()
            img = np.ascontiguousarray(composite_rgb, dtype=np.uint8)  # RGB HxWx3
            m = (np.asarray(mask) > 0).astype(np.uint8) * 255
            img_lr, mask_lr = cv2.resize(img, (256, 256)), cv2.resize(m, (256, 256))

            def _t(a):
                return torch.from_numpy(a).float().div(255).permute(2, 0, 1).to(self._dev)

            def _tm(a):
                return torch.from_numpy(a).float().div(255)[None].to(self._dev)

            with torch.no_grad():  # PCTNet.forward adds the batch dim itself; pass (C,H,W)
                out = net(_t(img_lr), _t(img), _tm(mask_lr), _tm(m))
            res = torch.clamp(255.0 * out.squeeze(0).permute(1, 2, 0), 0, 255).cpu().numpy()
            log.info("harmonized frame with in-process PCTNet")
            return res.astype(np.uint8)
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"PCTNet harmonization failed: {exc}") from exc
            log.warning("PCTNet failed (%s) -> classic fallback", str(exc)[-200:])
            return self._fallback(composite_rgb, mask, background_rgb)


# ---------------------------------------------------------------------------------------------------
# The compositor
# ---------------------------------------------------------------------------------------------------
class PatchCompositor:
    """Warp a patch onto an image-space quad and harmonize it — backend-agnostic."""

    def __init__(self, harmonizer: Harmonizer | None = None) -> None:
        self.harmonizer = harmonizer or ClassicHarmonizer()

    def apply(self, frame_rgb: np.ndarray, dst_quad: np.ndarray, patch_rgba: np.ndarray) -> np.ndarray:
        """Insert ``patch_rgba`` at image quad ``dst_quad`` in ``frame_rgb``; return the patched frame."""
        composite, mask = warp_patch(frame_rgb, dst_quad, patch_rgba)
        return self.harmonizer(composite, mask, frame_rgb)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 image points into (TL, TR, BR, BL) — the warp's quad order.

    Geometric ordering (by pixel position, not physical-corner identity) so the warped patch stays
    upright and unmirrored regardless of the target's world orientation relative to the camera.
    """
    p = np.asarray(pts, dtype=np.float64)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]  # x - y
    return np.stack([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])


def box_to_quad(box, width_frac: float = 0.6, height_frac: float = 0.5,
                v_center: float = 0.5, yaw: float = 0.0) -> np.ndarray:
    """Approximate a target plane's image quad (TL,TR,BR,BL) from a 2-D detection ``box`` [x1,y1,x2,y2].

    For a surface seen roughly head-on (e.g. a lead vehicle's rear), the plane's image quad is a
    centered sub-rectangle of the detection box — this is the *planar-warp* target, no depth needed:
    :func:`warp_patch` then homography-maps the patch onto it (every pixel, exact for a plane).
    ``width_frac`` / ``height_frac`` size the patch within the box, ``v_center`` places it vertically
    (0=top, 1=bottom), and ``yaw`` (radians) foreshortens one side into a trapezoid for an oblique view.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, y1 + v_center * bh
    hw, hh = 0.5 * width_frac * bw, 0.5 * height_frac * bh
    # yaw>0 pushes the right edge back (narrower) -> a trapezoid, approximating an oblique plane
    l, r = 1.0 + math.sin(yaw), 1.0 - math.sin(yaw)
    return np.array([[cx - hw, cy - hh * l], [cx + hw, cy - hh * r],
                     [cx + hw, cy + hh * r], [cx - hw, cy + hh * l]], dtype=np.float64)
