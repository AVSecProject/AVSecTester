"""Visualize the scenario-augmentation operators on a real camera frame — reusing ``simulators.viz``.

Applies each :class:`Corruption` (at a fixed severity) to one real nuRec ``.mp4`` frame and tiles the
results into a labelled :func:`filmstrip`, so the weather/lighting/sensor corruptions used for attack-
robustness testing can be eyeballed. Writes ``tmp/compare/augmentations.png``.

Run: ``python scripts/visualize_augmentations.py [--severity 0.6]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from avsectester.simulators.augment import CORRUPTIONS
from avsectester.simulators.viz import filmstrip, save_image

OUT = Path("tmp/compare")
NUREC = Path("/workspace/hdd/datasets/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.01_release")
_UUID = "023b7fcc-671c-40e3-9bd2-c66b0b073fbc"


def _label(rgb, text):
    import cv2
    import numpy as np
    out = np.ascontiguousarray(rgb[:, :, ::-1])          # to BGR for cv2 text
    cv2.rectangle(out, (0, 0), (len(text) * 12 + 10, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out[:, :, ::-1]


def main(severity: float) -> None:
    import cv2
    import numpy as np

    mp4 = str(NUREC / _UUID / "camera_front_wide_120fov.mp4")
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 178)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {mp4}")
    frame = cv2.resize(bgr[:, :, ::-1], (640, 360))       # downscale for a compact contact sheet
    rng = np.random.default_rng(0)

    tiles = [_label(frame, "clean")]
    for name, cls in CORRUPTIONS.items():
        tiles.append(_label(cls(severity).apply(frame, rng), f"{cls.category}: {name}"))

    out = OUT / "augmentations.png"
    save_image(filmstrip(tiles, cols=3), out)
    print(f"saved {out} ({len(tiles)} tiles at severity {severity})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=float, default=0.6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    main(args.severity)
