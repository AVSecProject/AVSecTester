"""Shared glue for the demo scripts — so no demo re-defines a stack or a detector.

Imported by the patch demos (``from demo_common import ...``); keeps the CARLA-trained 2D detector and
the trivial driving stacks in one place instead of copy-pasted across scripts.
"""

from __future__ import annotations

from avsectester.backend import AVStack
from avsectester.plane import Control


class CruiseStack(AVStack):
    """Hold a constant throttle so the ego rolls forward over a sequence (no perception)."""

    def __init__(self, throttle: float = 0.4) -> None:
        self.throttle = throttle

    def reset(self, observation) -> None:
        pass

    def __call__(self, observation) -> Control:
        return Control(throttle=self.throttle)


def build_detector(gpu: int):
    """Return ``detect(rgb) -> [(xyxy, score, 'car')]`` using the CARLA-trained 2D detector.

    Picks the single best plausible lead-car box (drops near-full-frame false positives). Heavy
    avstack/mmdet imports happen here, so importing this module stays cheap until it is called.
    """
    import avstack.modules.perception.object2dfv  # noqa: F401 - registers the detector
    from avstack.config import MODELS
    from mmdet.apis import inference_detector

    det = MODELS.build({"type": "MMDetObjectDetector2D", "model": "fasterrcnn",
                        "dataset": "carla-vehicle", "gpu": gpu, "threshold": 0.3})

    def detect(rgb):
        h, w = rgb.shape[:2]
        inst = inference_detector(det.model, rgb[:, :, ::-1]).pred_instances
        boxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        best = None  # the single best plausible lead-car box
        for b, sc in zip(boxes, scores):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < 0.6 * w * h and (best is None or sc > best[1]):
                best = (b, float(sc), "car")
        return [best] if best else []

    return detect


def plausible_detector(detect, area_max: float = 0.30, max_height_frac: float = 0.6):
    """Gate a detector to boxes a real planner would trust: not implausibly large or too tall.

    Adversarial patches can spawn a phantom full-scene box; gating it out is standard AV hygiene and
    keeps a hiding demo about the genuine object-hiding, not a false positive.
    """

    def _detect(rgb):
        h, w = rgb.shape[:2]
        out = []
        for box, score, label in detect(rgb):
            bh = box[3] - box[1]
            area = (box[2] - box[0]) * bh / (w * h)
            if bh <= max_height_frac * h and area <= area_max:
                out.append((box, score, label))
        return out

    return _detect
