"""Shared demo stacks, detector factories and explicit insertion payload construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from avsectester.attacks.object_insertion.person_poster import billboard, load_cutout, poster_rgba, standee
from avsectester.attacks.object_insertion.sign_spoof import SignAsset, sign_rgba
from avsectester.insertion import (
    ActorPose, AttachedPlacement, Insertion, Orientation, PlaneAsset, WorldPlacement,
)
from avsectester.rendering.harmonizers import ClassicHarmonizer, Harmonizer, PCTNetHarmonizer
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


COCO_VEHICLES = {3: "vehicle", 6: "vehicle", 8: "vehicle"}  # car / bus / truck
COCO_PERSON, COCO_TRAFFIC_LIGHT, COCO_STOP_SIGN = 1, 10, 13


def build_coco_detector(gpu: int = 0, threshold: float = 0.5, labels: dict | None = None):
    """Return ``detect(rgb) -> [(xyxy, score, name)]`` using a COCO-pretrained detector.

    For real / neural-reconstruction imagery (NuRec/Alpamayo) where the CARLA-trained detector does not
    apply. Keeps car/bus/truck boxes (COCO labels 3/6/8) as ``'vehicle'`` by default; pass ``labels``
    (``{coco_id: name}``) to keep other classes, e.g. ``{13: 'stop sign'}``. torchvision is imported lazily.
    """
    labels = COCO_VEHICLES if labels is None else labels
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).eval().to(dev)

    def detect(rgb):
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(dev)
        with torch.no_grad():
            out = model([t])[0]
        res = []
        for b, lab, sc in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(),
                              out["scores"].cpu().numpy()):
            if int(lab) in labels and float(sc) >= threshold:
                res.append((b, float(sc), labels[int(lab)]))
        return res

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


OBJECTS = {
    "stop": {"x": 28.0, "y": -6.5, "label": (COCO_STOP_SIGN, "stop sign")},
    "standee": {"x": 25.0, "y": -3.2, "label": (COCO_PERSON, "person")},
    "billboard": {"x": 28.0, "y": -7.0, "label": (COCO_PERSON, "person")},
    "trafficlights": {"x": 26.0, "y": -6.0, "label": (COCO_TRAFFIC_LIGHT, "traffic light")},
}


class NoHarmonizer(Harmonizer):
    """Plain alpha compositing, without appearance harmonization."""

    def __call__(self, composite_rgb, mask, background_rgb):
        return composite_rgb


def make_harmonizer(name: str, gpu: int) -> Harmonizer:
    """Load the requested method. Model failures remain errors in experiments."""
    return {
        "none": NoHarmonizer,
        "classic": ClassicHarmonizer,
        "chroma": lambda: ClassicHarmonizer(preserve_chroma=True, blend="feather"),
        "libcom": lambda: PCTNetHarmonizer(device=gpu, strict=True),
    }[name]()


def world_object(asset, x: float, y: float, *, yaw: float = 0, ground_z: float = 0) -> Insertion:
    """Place a local sign asset facing back along -x when ``yaw=0`` (radians)."""
    return Insertion("object", asset, WorldPlacement((x, y, ground_z)),
                     Orientation("fixed_world", (0, 0, 180 + math.degrees(yaw))))


@dataclass(frozen=True)
class ObjectPayload:
    roadside: Insertion
    vehicle_texture: np.ndarray


def build_object(args) -> ObjectPayload:
    """Build independent payload and placement data for the requested object."""
    spec = OBJECTS[args.object]
    x, y = spec["x"] if args.x is None else args.x, spec["y"] if args.y is None else args.y
    if args.object == "stop":
        face = sign_rgba()
        asset = SignAsset(face, 0.9 if args.size is None else args.size, 1.5 if args.mount is None else args.mount)
    elif args.object == "trafficlights":
        from avsectester.attacks.object_insertion.traffic_light import roadside_rig, traffic_lights_rgba

        face = traffic_lights_rgba(n=3, lit="red")
        asset = roadside_rig(face, width=2.4 if args.size is None else args.size,
                             mount_height=2.2 if args.mount is None else args.mount)
    else:
        if not args.asset:
            raise ValueError(f"--object {args.object} requires --asset <person cut-out PNG>")
        person = load_cutout(args.asset)
        face = poster_rgba(person)
        asset = (standee(person, height=1.75 if args.size is None else args.size) if args.object == "standee" else
                 billboard(person, width=1.4 if args.size is None else args.size,
                           mount_height=0.6 if args.mount is None else args.mount))
    return ObjectPayload(world_object(asset, x, y, yaw=args.yaw, ground_z=args.ground_z), face)


def rear_insertion(texture, host: str, pose: ActorPose, *, width_frac: float,
                   height_frac: float | None = None, offset_m=(0, 0, 0),
                   orientation: Orientation | None = None) -> Insertion:
    """Install a plane on a host's rear, sized in metres from its full cuboid dimensions."""
    width = pose.extent[1] * width_frac
    height = (pose.extent[2] * height_frac if height_frac is not None else
              width * texture.shape[0] / texture.shape[1])
    return Insertion("rear", PlaneAsset(texture, width, height),
                     AttachedPlacement(host, "rear_center", tuple(offset_m)),
                     orientation or Orientation("follow_host", (0, 0, 180)))


def nurec_source(usdz: str, scene_id: str | None):
    """Load metadata and reject a renderer/metadata scene mismatch before driving."""
    from avsectester.scenarios.datasets.nurec import NuRecDataset

    dataset = NuRecDataset([usdz], keyframe=0)
    scene = next(dataset.scenes())
    recorded_id = scene.source["scene_id"]
    if scene_id is not None and scene_id not in recorded_id:
        raise ValueError(f"--scene {scene_id!r} does not match metadata scene {recorded_id!r}")
    return dataset, scene


def nurec_rear_insertion(payload, scene, host: str) -> Insertion:
    """Bind once to an explicit recorded vehicle. Never rerank detections during driving."""
    objects = {obj.track_id: obj for obj in scene.objects}
    if host not in objects or objects[host].category != "vehicle":
        raise ValueError(f"Host {host!r} must be a vehicle present at the initial frame")
    obj = objects[host]
    aspect = payload.vehicle_texture.shape[0] / payload.vehicle_texture.shape[1]
    width_frac = 0.6 if aspect < 0.95 else (0.4 if aspect == 1.0 else 0.3)
    return rear_insertion(payload.vehicle_texture, host, ActorPose(np.eye(4), obj.extent),
                          width_frac=width_frac, offset_m=(-0.03, 0, 0.05 * obj.extent[2]))


def carla_rear_perturbation(backend, texture, compositor, *, width_frac=1.0, height_frac=0.95):
    """Create the attachment after reset, when the configured lead has been spawned."""
    from avsectester.simulators.carla import insertion_perturbation

    perturb = None

    def apply(observation):
        nonlocal perturb
        if perturb is None:
            pose = backend.selection_context().actors["lead"]
            insertion = rear_insertion(texture, "lead", pose,
                                       width_frac=width_frac, height_frac=height_frac)
            perturb = insertion_perturbation(backend, [insertion], compositor=compositor)
        return perturb(observation)

    return apply
