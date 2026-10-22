"""Write a labelled sensor-transform comparison for a supplied image or video frame."""

from __future__ import annotations

import argparse
from pathlib import Path

from avsectester.closed_loop.sensor_corruptions import CORRUPTIONS, AlbumentationsCorruption
from avsectester.evaluation.visualization import filmstrip, save_image


def _label(rgb, text):
    from PIL import Image, ImageDraw
    import numpy as np

    result = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, len(text) * 7 + 10), 22), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(result)


def render_comparison(frame, severity: float, implementation: str, seed: int = 0):
    import numpy as np

    tiles = [_label(frame, "clean")]
    if implementation == "albumentations":
        operators = [AlbumentationsCorruption(name, severity) for name in (
            "Fog", "Rain", "Snow", "GaussNoise", "ISONoise", "MotionBlur", "DefocusBlur", "JPEGCompression",
        )]
    else:
        operators = [cls(severity) for cls in CORRUPTIONS.values() if cls is not AlbumentationsCorruption]
    for index, operator in enumerate(operators):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        tiles.append(_label(operator.apply(frame, rng), f"{operator.category}: {operator.name}"))
    return filmstrip(tiles, cols=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image or recorded video")
    parser.add_argument("--frame", type=int, default=0, help="Recorded video frame index")
    parser.add_argument("--severity", type=float, default=.6)
    parser.add_argument("--implementation", choices=("native", "albumentations"), default="native")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("tmp/compare/augmentations.png"))
    args = parser.parse_args()
    import cv2

    frame = cv2.imread(str(args.image))
    if frame is None:
        video = cv2.VideoCapture(str(args.image))
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ok, frame = video.read()
        finally:
            video.release()
        if not ok:
            raise SystemExit(f"Cannot read frame {args.frame} from {args.image}")
    rgb = cv2.resize(frame[:, :, ::-1], (640, 360))
    save_image(render_comparison(rgb, args.severity, args.implementation, args.seed), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
