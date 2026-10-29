"""Real YOLOX COCO inference and input gradients on the canonical replica face."""

import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from avsectester.closed_loop.avstacks.implemented.modules import YOLOXDetector
from avsectester.data_types import TargetSpec
from avsectester.rendering import STOP_SIGN_FACE, load_rgba_image


def test_yolox_detector_gpu():
    weights = Path(os.environ.get("AVSECTESTER_YOLOX_WEIGHTS", "models/yolox_m.pth"))
    if not torch.cuda.is_available():
        pytest.skip("YOLOX live inference requires CUDA")
    if importlib.util.find_spec("yolox") is None:
        pytest.skip("YOLOX live inference requires the yolox package in the nurec image")
    if not weights.is_file():
        pytest.skip(f"YOLOX weights are required at {weights}; set AVSECTESTER_YOLOX_WEIGHTS")
    face = load_rgba_image(STOP_SIGN_FACE)
    height = 200
    width = round(face.shape[1] * height / face.shape[0])
    face = np.asarray(Image.fromarray(face).resize((width, height), Image.Resampling.BILINEAR))
    rgb = np.full((1080, 1920, 3), 114, dtype=np.uint8)
    left, top = (1920 - width) // 2, (1080 - height) // 2
    alpha = face[..., 3:4].astype(np.float32) / 255
    region = rgb[top:top + height, left:left + width]
    region[:] = np.rint(face[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    target = TargetSpec((left, top, left + width, top + height), "stop sign")
    detector = YOLOXDetector(weights)
    try:
        detections = detector.detect(rgb)
        matching = []
        for box, score, label in detections:
            intersection = np.maximum(0, np.minimum(box[2:], target.box[2:])
                                      - np.maximum(box[:2], target.box[:2])).prod()
            union = np.maximum(0, box[2:] - box[:2]).prod() + width * height - intersection
            if label == "stop sign" and intersection / union >= .3:
                matching.append(score)
        assert matching and max(matching) >= .5
        image = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda:0", torch.float32) / 255
        image.requires_grad_(True)
        score = detector.target_scores(image.unsqueeze(0), [target])[0]
        assert score.item() == pytest.approx(max(matching), abs=1e-6)
        gradient, = torch.autograd.grad(score, image)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max().item() > 0
    finally:
        detector.close()
