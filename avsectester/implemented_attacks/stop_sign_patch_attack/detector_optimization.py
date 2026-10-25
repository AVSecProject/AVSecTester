"""Optimize the patch pixels of a replica stop-sign face against a differentiable detector."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import traceback
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import torch
from PIL import Image

from avsectester.attack import OfflineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import AdversarialSample, DataSource, HideObject, Perturbation
from avsectester.data_types import TargetSpec
from avsectester.execution.services import ClosedLoopSession, RenderRequest
from avsectester.geometry.projection import project_box
from avsectester.validation import require_nonnegative_integer, require_positive_integer

from .assets import replica_face, texture_from_colors
from .metabuilder import control_from_user_data, route_progress

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack
    from .offline import DetectorPatchOptimization

OBJECT_ID = "stop_sign_replica"


class TextureDataSource(DataSource):
    """Sample already calibrated training captures; held-out captures are never passed here."""

    def __init__(self, samples: Sequence[AdversarialSample], *, seed: int = 0) -> None:
        require_nonnegative_integer(seed, "seed")
        if not samples or any(not isinstance(sample, AdversarialSample) for sample in samples):
            raise ValueError("TextureDataSource requires nonempty AdversarialSample inputs")
        self.samples = tuple(samples)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int) -> list[AdversarialSample]:
        require_positive_integer(batch, "batch")
        return [self.samples[index] for index in self.rng.integers(len(self.samples), size=batch)]


class MaskedPatchPerturbation(Perturbation):
    """Optimize RGB only in the mask and compose it through each capture's frozen visibility.

    The variable is a CHW texture. The original alpha, object geometry and pixels outside the
    mask are fixed. ``apply`` returns a camera image; the NumPy perturbation in rollout_optimization.py
    instead returns a physical texture to its rollout scorer.
    """

    def __init__(self, original_rgba: np.ndarray, mask: np.ndarray, *, device: str = "cpu") -> None:
        original = np.asarray(original_rgba)
        mask = np.asarray(mask)
        texture_from_colors(original, mask, np.zeros((3, *original.shape[:2])))
        if not mask.any():
            raise ValueError("Patch mask must contain at least one pixel")
        self.original_rgba = original.copy()
        self.mask = torch.as_tensor(mask.copy(), dtype=torch.bool, device=device)[None]
        self.original_rgb = torch.as_tensor(original[..., :3].copy(), device=device,
                                            dtype=torch.float32).permute(2, 0, 1) / 255

    def init(self) -> torch.Tensor:
        return self.original_rgb.clone().detach().requires_grad_(True)

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.where(self.mask, delta.clamp(0., 1.), self.original_rgb)

    def apply(self, sample: AdversarialSample, delta: torch.Tensor) -> torch.Tensor:
        texture = self.project(delta)
        return sample.context["plan"].composite(
            sample.x.to(texture.device), {(OBJECT_ID, 0): texture})

    def export(self, delta: torch.Tensor) -> np.ndarray:
        return texture_from_colors(self.original_rgba, self.mask[0].cpu().numpy(),
                                   self.project(delta).detach().cpu().numpy())


def _colors(texture: np.ndarray) -> torch.Tensor:
    """Straight RGB colours of a uint8 RGBA texture as a CHW tensor in [0, 1]."""
    return torch.as_tensor(texture[..., :3].copy(), dtype=torch.float32).permute(2, 0, 1) / 255


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collect_samples(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> tuple[list[AdversarialSample], list[AdversarialSample]]:
    """Calibrate exact detectable ranges and sample disjoint, time-spaced captures.

    The d_max search covers recording start through the recorded stop, on the entering route
    and facing the selected sign. Frames are searched by descending distance; the first frame
    whose unpatched replica meets the attack's detection threshold is therefore the farthest
    detectable eligible frame, without a monotonicity assumption. The configured visibility minimum excludes
    ineligible frames explicitly. d_min uses the recorded speed at the selected segment
    start, squared and divided by twice the configured braking deceleration.

    Captures use the framework-provided context.services directly. Native
    geometry or service errors stop collection with a saved reproduction; they are never scores.
    """
    options = cast("DetectorPatchOptimization", attack.optimization)
    if not context.metadata:
        raise ValueError("Detector patch optimization requires at least one training segment")
    if context.environment.get("sensors"):
        raise ValueError("White-box composition samples require an environment without sensor corruptions")
    destination = Path(options.output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = context.services.dataset
    cohort = context.metadata[:options.max_training_segments]
    replica_colors = _colors(replica_face())
    calibration: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    train, heldout = [], []
    for segment in cohort:
        scene = segment.scenario_id
        control = control_from_user_data(segment.user_data)
        asset, pose = attack.replica_placement(segment, original)
        frames = tuple(dataset.frames(scene))
        positions = np.asarray(segment.user_data["route_points"], dtype=float)
        stop = int(segment.user_data["recorded_stop_frame"])
        distances = np.linalg.norm(positions - control.pose[:3, 3], axis=1)
        start_state = dataset.frame_data(scene, segment.start).ego
        speed = float(np.linalg.norm(start_state.velocity))
        d_min = speed ** 2 / (2 * options.braking_deceleration_mps2)
        record: dict[str, Any] = {"scene": scene, "selected_start": segment.start.position,
                                  "approach_speed_mps": speed, "d_min_m": d_min, "d_max_m": None,
                                  "search_first_frame": 0, "search_last_frame": stop, "probes": []}
        calibration.append(record)
        cache: dict[int, AdversarialSample] = {}

        def capture(session: ClosedLoopSession, index: int, stage: str) -> AdversarialSample | None:
            if index in cache:
                return cache[index]
            observation = session.render(RenderRequest(scene, frames[index], options.camera_id,
                actions=[InsertObject(OBJECT_ID, asset, pose, harmonizer=attack.harmonizer)],
                environment=context.environment))
            evidence = observation.object_visibility.get(OBJECT_ID, {}).get(options.camera_id)
            visible_pixels = 0 if evidence is None else evidence.visible_pixels
            note = {"frame": index, "timestamp_s": frames[index].timestamp,
                    "distance_m": float(distances[index]), "stage": stage, "visible_pixels": visible_pixels}
            record["probes"].append(note)
            _write_json(destination / "calibration.json", calibration)
            if visible_pixels < options.min_visible_pixels:
                note.update(status="skipped", reason=f"fewer than {options.min_visible_pixels} visible pixels",
                            occlusion_source=None if evidence is None else evidence.occlusion_source)
                return None
            composition = observation.view.composition
            if composition is None:
                raise ValueError("Visible inserted patch has no composition")
            box = project_box(observation.view.camera, observation.view.cam_from_world,
                              control.pose, control.extent)
            if box is None:
                note.update(status="skipped", reason="sign face projects outside the camera",
                            occlusion_source=evidence.occlusion_source)
                return None
            background = torch.as_tensor(composition.background.copy(), dtype=torch.float32).permute(2, 0, 1) / 255
            note.update(status="captured", target_box=list(box), occlusion_source=evidence.occlusion_source)
            return AdversarialSample(background, TargetSpec(box, "stop sign"),
                                     {"plan": composition.plan, "record": note, "scene": scene})

        try:
            progress = route_progress(positions, positions)
            before = progress <= float(segment.user_data["line_route_distance_m"])
            domain = [index for index in range(stop + 1) if before[index]
                      and dataset.frame_data(scene, frames[index]).ego.pose[:3, 0] @ control.pose[:3, 0] < -.5]
            with context.services.session(context.seed) as session:
                for index in sorted(domain, key=lambda value: (-distances[value], value)):
                    sample = capture(session, index, "d_max_search")
                    if sample is not None:
                        with torch.no_grad():
                            replica_image = sample.context["plan"].composite(
                                sample.x, {(OBJECT_ID, 0): replica_colors})
                            score = float(options.detector.target_score(replica_image, sample.target))
                        sample.context["record"]["replica_score"] = score
                        if score >= options.detection_threshold:
                            record["d_max_m"] = float(distances[index])
                            cache[index] = sample
                            break
                    _write_json(destination / "calibration.json", calibration)
                d_max = record["d_max_m"]
                if d_max is None or d_max < d_min:
                    raise ValueError(f"No detectable stopping-distance interval for {scene}: {d_min}, {d_max}")
                eligible = [index for index in domain if d_min <= distances[index] <= d_max]
                if len(eligible) < 4:
                    raise ValueError(f"Too few eligible recorded frames for {scene}")
                timestamps = np.asarray([frames[index].timestamp for index in eligible])
                sample_times = np.linspace(timestamps[0], timestamps[-1],
                                           min(options.samples_per_scene, len(eligible)))
                chosen = np.unique([int(np.abs(timestamps - value).argmin()) for value in sample_times])
                scene_samples: dict[str, list[AdversarialSample]] = {"train": [], "heldout": []}
                for order, position in enumerate(chosen):
                    index = eligible[int(position)]
                    sample = capture(session, index, "time_uniform_sample")
                    if sample is None:
                        continue
                    split = "train" if order % 2 == 0 else "heldout"
                    entry = {"scene": scene, "split": split,
                             **sample.context["record"]}
                    manifest.append(entry)
                    sample.context["record"] = entry
                    scene_samples[split].append(sample)
                if any(not values for values in scene_samples.values()):
                    raise ValueError(f"Visibility leaves no disjoint train/held-out samples for {scene}")
                train.extend(scene_samples["train"])
                heldout.extend(scene_samples["heldout"])
                record.update(status="calibrated", train_frames=[s.context["record"]["frame"] for s in scene_samples["train"]],
                              heldout_frames=[s.context["record"]["frame"] for s in scene_samples["heldout"]])
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            _write_json(destination / "calibration.json", calibration)
            _write_csv(destination / "sample_manifest.csv", manifest)
            raise
        _write_json(destination / "calibration.json", calibration)
        _write_csv(destination / "sample_manifest.csv", manifest)
    return train, heldout


def optimize_patch(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> np.ndarray:
    """Run framework PGD/HideObject; score held-out frames with and without the replica.

    ``clean`` is the reconstructed sign without insertion, ``replica`` the unpatched replica,
    ``initial`` the initial patch and ``optimized`` the PGD result.
    """
    options = cast("DetectorPatchOptimization", attack.optimization)
    destination = Path(options.output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    mask = attack.mask
    train, heldout = collect_samples(attack, context, original)
    perturbation = MaskedPatchPerturbation(original, mask, device=getattr(options.detector, "device", "cpu"))
    initial = perturbation.export(perturbation.init())
    result = options.optimizer.run(TextureDataSource(train, seed=context.seed), perturbation,
                                  options.detector, HideObject())
    optimized = attack.replica_texture(result.export)
    for name, texture in [("initial_patch", initial), ("optimized_texture", optimized)]:
        Image.fromarray(texture).save(destination / (name + ".png"))
    scores = [{"iteration": index, "stage": "before_update", "objective": value}
              for index, value in enumerate(result.history)]
    scores.append({"iteration": len(result.history), "stage": "final_fresh_batch", "objective": result.final_score})
    _write_csv(destination / "pgd_scores.csv", scores)
    measured = []
    with torch.no_grad():
        for sample in heldout:
            for name, candidate in [("clean", None), ("replica", replica_face()), ("initial", initial),
                                    ("optimized", optimized)]:
                image = sample.x
                if candidate is not None:
                    image = sample.context["plan"].composite(sample.x, {(OBJECT_ID, 0): _colors(candidate)})
                score = float(options.detector.target_score(image, sample.target))
                entry = sample.context["record"]
                measured.append({"scene": sample.context["scene"],
                    "frame": entry["frame"], "distance_m": entry["distance_m"], "texture": name,
                    "score": score, "detected": score >= options.detection_threshold})
    _write_csv(destination / "heldout_scores.csv", measured)
    grouped = defaultdict(list)
    for row in measured:
        lower = int(row["distance_m"] // 10) * 10
        grouped[row["texture"], lower].append(row)
    rates = [{"texture": key[0], "distance_bin_m": f"[{key[1]},{key[1]+10})",
              "frames": len(rows), "detections": sum(row["detected"] for row in rows),
              "detection_rate": sum(row["detected"] for row in rows) / len(rows)}
             for key, rows in sorted(grouped.items())]
    _write_csv(destination / "detection_rate_vs_distance.csv", rates)
    # Draw without changing matplotlib's global backend, as the framework's visualization does.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure()
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(range(len(result.history)), result.history, marker=".", label="pre-update batch objective")
    axis.scatter([len(result.history)], [result.final_score], label="final fresh batch")
    axis.set(xlabel="PGD iteration", ylabel="Stop-sign confidence objective")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "pgd_curve.png", dpi=150)
    figure.clear()
    _write_json(destination / "optimization.json", {"train_frames": len(train), "heldout_frames": len(heldout),
        "history": result.history, "final_score": result.final_score,
        "max_training_segments": options.max_training_segments,
        "training_segments": len(context.metadata[:options.max_training_segments]),
        "mask_pixels": int(mask.sum()), "offset_m": attack.offset_m, "width_scale": attack.width_scale,
        "detector_stop_sign_label": getattr(options.detector, "stop_sign_label", None)})
    return optimized
