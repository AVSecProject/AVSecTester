"""Standard-warp + harmonization patch compositor — one backend-agnostic 2D insert for CARLA & NuRec.

The patch is composited **on the rendered frame**, never baked into the world/reconstruction:

    project the target surface quad -> perspective-warp the patch onto it -> harmonize to fit.

Each backend supplies only the clean frame + camera intrinsics + the target's image-space quad (from
its pose/geometry); the warp + harmonization are shared here. Harmonization is pluggable:
  * ``ClassicHarmonizer`` — OpenCV Poisson blend + Reinhard color transfer; in-process, reliable, no
    heavy deps, texture-preserving (keeps the patch's gradients, matches color/lighting to the scene).
  * ``LibcomHarmonizer`` — a learned harmonizer (libcom) run **out-of-process** in an isolated conda
    env, because libcom's deps (mmdet 3.2 / mmpose / diffusers) conflict with the avstack stack.

cv2/skimage are imported lazily, so this module imports without them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------
# Geometry: intrinsics + projection (backend adapters convert world -> camera coords, then call these)
# ---------------------------------------------------------------------------------------------------
def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K from horizontal FOV (CARLA/NuRec RGB cameras)."""
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


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


class LibcomHarmonizer(Harmonizer):
    """Learned harmonization via libcom, run out-of-process in an isolated conda env.

    libcom's deps (mmdet 3.2 / mmpose / diffusers) conflict with the avstack stack, so it lives in its
    own env and we shell out: write composite+mask to temp PNGs, run ``scripts/libcom_harmonize.py`` in
    ``env``, read back the harmonized PNG. Falls back to ``ClassicHarmonizer`` if the call fails.

    Set the env up once with ``scripts/setup_libcom_env.sh`` (installs the AVSecProject/libcom fork,
    which fixes the upstream packaging + HF-download bugs). PCTNet gives a milder, texture-preserving
    harmonization than the classic Poisson blend — better for keeping an adversarial pattern intact.
    """

    def __init__(self, env: str = "libcom", model: str = "PCTNet", strict: bool = False) -> None:
        self.env = env
        self.model = model
        self.strict = strict  # True -> raise instead of silently falling back (for validation)
        self._fallback = ClassicHarmonizer()
        self._proc = None  # persistent worker (model stays resident across frames)
        self._tmp = None
        self._n = 0

    def _worker(self):
        """Lazily start (or restart) the resident libcom server; returns the live process."""
        import subprocess
        import tempfile
        from pathlib import Path

        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        script = Path(__file__).resolve().parents[2] / "scripts" / "libcom_harmonize.py"
        # -u + --no-capture-output so READY/OK flush straight through conda's wrapper
        self._proc = subprocess.Popen(
            ["conda", "run", "--no-capture-output", "-n", self.env, "python", "-u", str(script),
             "--serve", "--model", self.model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._tmp = self._tmp or tempfile.mkdtemp(prefix="libcom_hz_")
        for _ in range(2000):  # skip library banner noise on stdout until the READY sentinel
            line = self._proc.stdout.readline()
            if line == "":  # EOF: the worker died before signaling ready
                raise RuntimeError("libcom worker exited before READY")
            if line.strip() == "READY":
                log.info("libcom %s worker ready (env=%s)", self.model, self.env)
                return self._proc
        raise RuntimeError("libcom worker never signaled READY")

    def __call__(self, composite_rgb, mask, background_rgb):
        from pathlib import Path

        from PIL import Image

        try:
            proc = self._worker()
            self._n += 1
            d = Path(self._tmp)
            comp_p, mask_p, out_p = d / f"c{self._n}.png", d / f"m{self._n}.png", d / f"o{self._n}.png"
            Image.fromarray(composite_rgb).save(comp_p)
            Image.fromarray(mask).save(mask_p)
            proc.stdin.write(f"{comp_p}\t{mask_p}\t{out_p}\n")
            proc.stdin.flush()
            resp = ""
            for _ in range(2000):  # skip any per-inference stdout noise until OK / ERR
                line = proc.stdout.readline()
                if line == "":
                    raise RuntimeError("libcom worker died mid-request")
                resp = line.strip()
                if resp == "OK" or resp.startswith("ERR"):
                    break
            if resp != "OK":
                raise RuntimeError(f"libcom worker: {resp!r}")
            out = np.asarray(Image.open(out_p).convert("RGB"))
            log.info("harmonized frame with libcom %s", self.model)
            return out
        except Exception as exc:
            if self.strict:  # validation mode: never hide a failure behind the classic blend
                raise RuntimeError(f"libcom {self.model} harmonization failed: {exc}") from exc
            log.warning("libcom %s failed (%s) -> classic fallback", self.model, str(exc)[-300:])
            return self._fallback(composite_rgb, mask, background_rgb)

    def close(self) -> None:
        """Shut the resident worker down (also happens on GC)."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort teardown
                self._proc.kill()
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - best-effort teardown during GC; nothing to log to
            pass


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


def rear_face_quad(center: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4 world-space corners (TL, TR, BR, BL) of a planar patch given its center + right/up half-edges."""
    c, r, u = (np.asarray(v, dtype=np.float64) for v in (center, right, up))
    return np.stack([c - r + u, c + r + u, c + r - u, c - r - u])
