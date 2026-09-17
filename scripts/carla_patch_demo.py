#!/usr/bin/env python
"""Physical adversarial-patch demo (CARLA): render a patch on the lead car's rear, clean vs attacked.

Drives the ego (a simple cruise stack) in ``configs/carla_patch_scenario.yaml`` twice through the
framework's ``run`` loop — once clean, once with the patch attached to the lead car by
``CarlaBackend`` at reset — and saves the ego camera frames from each to ``./tmp/carla_patch/{clean,
patched}/``. This proves the physical patch renders in-scene (the substrate); feeding the camera into
a detector so the patch changes perception/driving is the next slice.

Run in the `avsec` conda env against a CARLA 0.9.15 server on :2000 (GPU 2):
    conda run -n avsec python scripts/carla_patch_demo.py [--frames 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from avsectester.backend import AVStack
from avsectester.plane import Control
from avsectester.scenario import CarlaBackend

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "carla_patch"


class CruiseStack(AVStack):
    """A trivial AV stack: hold a gentle constant throttle (enough to roll, not to reach the lead)."""

    def __init__(self, throttle: float = 0.3) -> None:
        self.throttle = throttle

    def __call__(self, obs) -> Control:
        return Control(throttle=self.throttle)


def _rgb(frame) -> np.ndarray | None:
    """Best-effort extract an (H, W, 3) uint8 array from a camera Observation payload."""
    for attr in ("rgb_image", "data"):
        v = getattr(frame, attr, None)
        if v is not None and hasattr(v, "shape"):
            arr = np.asarray(v)
            return arr[..., :3] if arr.ndim == 3 else arr
    return np.asarray(frame)[..., :3] if hasattr(frame, "shape") else None


def _save_frames(backend, stack, frames, out_dir):
    """Drive `frames` steps, saving each camera view to out_dir; return the driving Trace."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    obs = backend.reset()
    stack.reset(obs)
    saved = 0
    for i in range(frames):
        control = stack(obs)
        for payload in (obs.sensor_data or {}).values():
            arr = _rgb(payload)
            if arr is not None and arr.ndim == 3:
                Image.fromarray(arr.astype(np.uint8)).save(out_dir / f"frame_{i:04d}.png")
                saved += 1
                break
        obs = backend.step(control)
    print(f"  saved {saved} camera frames -> {out_dir}")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "carla_patch_scenario.yaml"))
    ap.add_argument("--frames", type=int, default=None, help="override frames from the config")
    args = ap.parse_args()

    scenario = yaml.safe_load(Path(args.config).read_text())
    frames = args.frames or scenario.get("frames", 20)
    patches = scenario.get("patches")

    print(f"[demo] physical-patch CARLA demo, {frames} frames/run")
    print("[clean]   no patch on the lead car")
    backend = CarlaBackend(scenario, patches=None)
    try:
        _save_frames(backend, CruiseStack(), frames, OUT / "clean")
    finally:
        backend.close()

    print("[patched] checkerboard patch attached to the lead car's rear")
    backend = CarlaBackend(scenario, patches=patches)
    try:
        _save_frames(backend, CruiseStack(), frames, OUT / "patched")
    finally:
        backend.close()

    # report a coarse difference on the last frame (the patch should change the pixels)
    from PIL import Image

    last = f"frame_{frames - 1:04d}.png"
    ca, pa = OUT / "clean" / last, OUT / "patched" / last
    if ca.exists() and pa.exists():
        c = np.asarray(Image.open(ca), np.float32)
        p = np.asarray(Image.open(pa), np.float32)
        print(f"[diff] mean |clean-patched| on {last}: {np.abs(c - p).mean():.2f}/255")
    print(f"[output] frames under {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
