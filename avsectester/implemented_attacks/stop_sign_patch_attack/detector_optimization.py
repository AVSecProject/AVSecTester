"""Optimize the patch pixels of a replica stop-sign face against a differentiable detector."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import torch

from avsectester.attack import OfflineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import (
    AdversarialSample, DataSource, HideBelowThreshold, Perturbation, SessionScorer,
)
from avsectester.data_types import TargetSpec
from avsectester.execution.services import ClosedLoopSession, RenderRequest
from avsectester.geometry.projection import project_box
from avsectester.validation import require_nonnegative_integer, require_positive_integer

from .assets import replica_face, texture_from_colors
from .metabuilder import control_from_user_data, route_progress
from .online import REPLICA_OBJECT_ID

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack
    from .offline import DetectorPatchOptimization


class TextureDataSource(DataSource):
    """Sample already calibrated training captures uniformly at random."""

    def __init__(self, samples: Sequence[AdversarialSample], *, seed: int = 0) -> None:
        require_nonnegative_integer(seed, "seed")
        if not samples or any(not isinstance(sample, AdversarialSample) for sample in samples):
            raise ValueError("TextureDataSource requires nonempty AdversarialSample inputs")
        self.samples = tuple(samples)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int) -> list[AdversarialSample]:
        require_positive_integer(batch, "batch")
        return [self.samples[index] for index in self.rng.integers(len(self.samples), size=batch)]


def _masked_total_variation(colors: torch.Tensor, mask: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean Charbonnier RGB distance over vertical and horizontal neighbor pairs inside ``mask``."""
    vertical = mask[1:, :] & mask[:-1, :]
    horizontal = mask[:, 1:] & mask[:, :-1]
    distances = [
        (colors[:, 1:, :] - colors[:, :-1, :]).square().sum(0)[vertical],
        (colors[:, :, 1:] - colors[:, :, :-1]).square().sum(0)[horizontal],
    ]
    squared = torch.cat(distances)
    return (squared + eps ** 2).sqrt().mean() - eps


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
            sample.x.to(texture.device), {(REPLICA_OBJECT_ID, 0): texture})

    def export(self, delta: torch.Tensor) -> np.ndarray:
        return texture_from_colors(self.original_rgba, self.mask[0].cpu().numpy(),
                                   self.project(delta).detach().cpu().numpy())

    def smoothness(self, delta: torch.Tensor) -> torch.Tensor:
        """Mean Charbonnier color distance between adjacent texels that are both in the mask.

        Each vertical or horizontal pair contributes ``sqrt(||c_p - c_q||^2 + eps^2) - eps`` over
        RGB. Pairs across the patch boundary are excluded: that edge is the physical sticker
        outline, not patch content. Fixed texels outside the mask contribute nothing.
        """
        return _masked_total_variation(self.project(delta), self.mask[0])


def _colors(texture: np.ndarray) -> torch.Tensor:
    """Straight RGB colors of a uint8 RGBA texture as a CHW tensor in [0, 1]."""
    return torch.as_tensor(texture[..., :3].copy(), dtype=torch.float32).permute(2, 0, 1) / 255


_CALIBRATION_COLUMNS = {
    "scene": "Dataset scenario identifier.",
    "selected_start": "Selected training-segment start position (frames, zero-based).",
    "approach_speed_mps": "Recorded ego speed at the selected segment start (m/s).",
    "d_min_m": "Route distance to the stop line inside which braking_deceleration_mps2 can no longer "
               "stop the approach speed before the line: speed squared over twice the deceleration (m).",
    "d_max_m": "Farthest eligible route distance to the stop line where the unpatched replica reaches "
               "the detection threshold (m).",
    "search_first_frame": "First recording frame considered for range calibration (frames, zero-based).",
    "search_last_frame": "Last recording frame considered, at the recorded stop (frames, zero-based).",
    "status": "Training-segment calibration status: calibrating, calibrated or failed.",
    "error": "Exception type and message when calibration failed; otherwise unavailable.",
}
_PROBE_COLUMNS = {
    "scene": "Dataset scenario identifier.",
    "frame": "Recorded capture frame position (frames, zero-based).",
    "timestamp_s": "Recorded capture timestamp (s).",
    "distance_m": "Recorded ego-to-sign face-centre distance in 3D (m).",
    "line_distance_m": "Recorded ego route distance to the stop line; negative past the line (m).",
    "stage": "Capture purpose: d_max_search or sample.",
    "visible_pixels": "Visible replica face pixels in the selected camera (pixels).",
    "status": "Capture status: captured or skipped; unavailable while the capture is in progress.",
    "reason": "Reason the capture was skipped; otherwise unavailable.",
    "occlusion_source": "Simulator visibility source used for the replica face.",
    "target_box_left_px": "Projected map sign-face target-box left coordinate (pixels).",
    "target_box_top_px": "Projected map sign-face target-box top coordinate (pixels).",
    "target_box_right_px": "Projected map sign-face target-box right coordinate (pixels).",
    "target_box_bottom_px": "Projected map sign-face target-box bottom coordinate (pixels).",
    "replica_score": "Unpatched-replica stop-sign confidence during d_max search (ratio, 0–1).",
}
_TEXTURES = ("clean", "replica", "initial", "optimized")
_TEXTURE_MEANINGS = (
    "clean = reconstructed sign without a replica; replica = unpatched replica; "
    "initial = replica with the initial patch; optimized = replica with the optimized patch."
)


def _capture_row(scene: str, note: Mapping[str, Any]) -> dict[str, Any]:
    """Write named capture columns explicitly, including absent projection evidence."""
    box = note.get("target_box") or [None] * 4
    return {
        "scene": scene, "frame": note.get("frame"), "timestamp_s": note.get("timestamp_s"),
        "distance_m": note.get("distance_m"), "line_distance_m": note.get("line_distance_m"),
        "stage": note.get("stage"),
        "visible_pixels": note.get("visible_pixels"), "status": note.get("status"),
        "reason": note.get("reason"), "occlusion_source": note.get("occlusion_source"),
        "target_box_left_px": box[0], "target_box_top_px": box[1],
        "target_box_right_px": box[2], "target_box_bottom_px": box[3],
        "replica_score": note.get("replica_score"),
    }


def _distance_bin(sample: AdversarialSample) -> int:
    """Lower bound of the 10 m route-distance-to-line bin of a training capture."""
    return int(sample.context["record"]["line_distance_m"] // 10) * 10


def _report_progress(report: Any, progress: list[dict[str, Any]], lowers: list[int],
                     decision_threshold: float, interval: int) -> None:
    """Write the training-set scores recorded during PGD as a table and step series."""
    steps = [row["step"] for row in progress]
    bins = {f"detection_rate_{lower}_{lower + 10}m": f"[{lower},{lower + 10}) m" for lower in lowers}
    report.set_table(
        "training_progress", progress,
        description=f"The current patch scored on every training frame every {interval} PGD steps and "
        f"after the last step. Detection means confidence ≥ decision_threshold ({decision_threshold}).",
        columns={
            "step": "Completed PGD updates (steps).",
            "hinge_loss": "Mean hinge objective over all training frames (ratio, 0–1).",
            "detection_rate": "Fraction of training frames detected (ratio, 0–1).",
            "max_confidence": "Highest stop-sign confidence over the training frames (ratio, 0–1).",
            **{name: f"Fraction detected among training frames {interval_m} from the stop line (ratio, 0–1)."
               for name, interval_m in bins.items()},
        },
    )
    report.set_series(
        "training_hinge_loss", [row["hinge_loss"] for row in progress], x=steps, x_label="PGD iteration",
        y_label="Mean hinge loss", y_unit="ratio (0–1)",
        description="Hinge objective averaged over every training frame, not one batch.",
    )
    report.set_series(
        "training_detection_rate", [row["detection_rate"] for row in progress], x=steps,
        x_label="PGD iteration", y_label="Detected fraction", y_unit="ratio (0–1)",
        description="Fraction of all training frames still detected by the driving system's detector.",
    )
    for name, interval_m in bins.items():
        report.set_series(
            f"training_{name}", [row[name] for row in progress], x=steps, x_label="PGD iteration",
            y_label="Detected fraction", y_unit="ratio (0–1)",
            description=f"Fraction of training frames {interval_m} from the stop line still detected.",
        )


def collect_samples(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray, *,
    session: ClosedLoopSession, scorer: SessionScorer, sensor_id: str, decision_threshold: float,
) -> list[AdversarialSample]:
    """Calibrate the detectable range that decides a stop and sample captures inside it.

    Distances are route distances from ego to the stop line. The attack must suppress every
    detection until the driving system can no longer stop before the line: d_min is the recorded
    speed at the segment start squared over twice ``braking_deceleration_mps2``, the driving
    system's strongest braking. Frames closer than d_min need no suppression. The d_max search
    covers recording start through the recorded stop, on the entering route and facing the sign,
    in descending line distance; the first frame whose unpatched replica meets the scorer's
    decision threshold is therefore the farthest detectable one, without a monotonicity
    assumption. The configured visibility minimum excludes ineligible frames explicitly.
    ``samples_per_scene="all"`` uses every eligible frame; an integer selects that many time-uniform
    frames.
    Every selected frame is a PGD training sample.

    Captures use the framework-provided context.services directly. Native
    geometry or service errors stop collection with a saved reproduction; they are never scores.
    """
    options = cast("DetectorPatchOptimization", attack.optimization)
    if not context.metadata:
        raise ValueError("Detector patch optimization requires at least one training segment")
    if context.environment.get("sensors"):
        raise ValueError("White-box composition samples require an environment without sensor corruptions")
    dataset = context.services.dataset
    cohort = context.metadata[:options.max_training_segments]
    replica_colors = _colors(replica_face())
    calibration: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    train: list[AdversarialSample] = []

    def update_report() -> None:
        rows = [{column: record.get(column) for column in _CALIBRATION_COLUMNS}
                for record in calibration]
        context.report.set_table(
            "calibration", rows,
            description=f"Detectable-range calibration, with one row per training segment. Camera: {sensor_id}; "
            f"decision_threshold: {decision_threshold}.",
            columns=_CALIBRATION_COLUMNS if rows else {},
        )
        probes = [_capture_row(record["scene"], probe)
                  for record in calibration for probe in record["probes"]]
        context.report.set_table(
            "calibration_probes", probes,
            description=f"One row per attempted capture from camera {sensor_id} during range calibration and sampling.",
            columns=_PROBE_COLUMNS if probes else {},
        )
        samples = [_capture_row(row["scene"], row) for row in manifest]
        context.report.set_table(
            "sample_manifest", samples,
            description="Recorded captures used for PGD updates, from every training segment.",
            columns=_PROBE_COLUMNS if samples else {},
        )

    for segment in cohort:
        scene = segment.scenario_id
        control = control_from_user_data(segment.user_data)
        asset, pose = attack.replica_placement(segment, original)
        frames = tuple(dataset.frames(scene))
        positions = np.asarray(segment.user_data["route_points"], dtype=float)
        stop = int(segment.user_data["recorded_stop_frame"])
        distances = np.linalg.norm(positions - control.pose[:3, 3], axis=1)
        progress = route_progress(positions, positions)
        line_distances = float(segment.user_data["line_route_distance_m"]) - progress
        start_state = dataset.frame_data(scene, segment.start).ego
        speed = float(np.linalg.norm(start_state.velocity))
        d_min = speed ** 2 / (2 * options.braking_deceleration_mps2)
        record: dict[str, Any] = {"scene": scene, "selected_start": segment.start.position,
                                  "approach_speed_mps": speed, "d_min_m": d_min, "d_max_m": None,
                                  "search_first_frame": 0, "search_last_frame": stop, "status": "calibrating", "error": None, "probes": []}
        calibration.append(record)
        update_report()
        cache: dict[int, AdversarialSample] = {}

        def capture(session: ClosedLoopSession, index: int, stage: str) -> AdversarialSample | None:
            if index in cache:
                return cache[index]
            observation = session.render(RenderRequest(scene, frames[index], sensor_id,
                actions=[InsertObject(REPLICA_OBJECT_ID, asset, pose, harmonizer=attack.harmonizer)],
                environment=context.environment))
            evidence = observation.object_visibility.get(REPLICA_OBJECT_ID, {}).get(sensor_id)
            visible_pixels = 0 if evidence is None else evidence.visible_pixels
            note = {"frame": index, "timestamp_s": frames[index].timestamp,
                    "distance_m": float(distances[index]), "line_distance_m": float(line_distances[index]),
                    "stage": stage, "visible_pixels": visible_pixels}
            record["probes"].append(note)
            update_report()
            if visible_pixels < options.min_visible_pixels:
                note.update(status="skipped", reason=f"fewer than {options.min_visible_pixels} visible pixels",
                            occlusion_source=None if evidence is None else evidence.occlusion_source)
                update_report()
                return None
            composition = observation.view.composition
            if composition is None:
                raise ValueError("Visible inserted patch has no composition")
            box = project_box(observation.view.camera, observation.view.cam_from_world,
                              control.pose, control.extent)
            if box is None:
                note.update(status="skipped", reason="sign face projects outside the camera",
                            occlusion_source=evidence.occlusion_source)
                update_report()
                return None
            background = torch.as_tensor(composition.background.copy(), dtype=torch.float32).permute(2, 0, 1) / 255
            note.update(status="captured", target_box=list(box), occlusion_source=evidence.occlusion_source)
            update_report()
            return AdversarialSample(background, TargetSpec(box, "stop sign"),
                                     {"plan": composition.plan, "record": note, "scene": scene})

        try:
            domain = [index for index in range(stop + 1) if line_distances[index] >= 0
                      and dataset.frame_data(scene, frames[index]).ego.pose[:3, 0] @ control.pose[:3, 0] < -.5]
            for index in sorted(domain, key=lambda value: (-line_distances[value], value)):
                sample = capture(session, index, "d_max_search")
                if sample is not None:
                    with torch.no_grad():
                        replica_image = sample.context["plan"].composite(
                            sample.x, {(REPLICA_OBJECT_ID, 0): replica_colors})
                        score = float(scorer.target_score(replica_image, sample.target))
                    sample.context["record"]["replica_score"] = score
                    if score >= decision_threshold:
                        record["d_max_m"] = float(line_distances[index])
                        cache[index] = sample
                        break
                update_report()
            d_max = record["d_max_m"]
            if d_max is None or d_max < d_min:
                raise ValueError(f"No detectable stopping-distance interval for {scene}: {d_min}, {d_max}")
            eligible = [index for index in domain if d_min <= line_distances[index] <= d_max]
            if len(eligible) < 4:
                raise ValueError(f"Too few eligible recorded frames for {scene}")
            if options.samples_per_scene == "all":
                chosen = np.arange(len(eligible))
            else:
                timestamps = np.asarray([frames[index].timestamp for index in eligible])
                sample_times = np.linspace(timestamps[0], timestamps[-1],
                                           min(options.samples_per_scene, len(eligible)))
                chosen = np.unique([int(np.abs(timestamps - value).argmin()) for value in sample_times])
            scene_samples = []
            for position in chosen:
                sample = capture(session, eligible[int(position)], "sample")
                if sample is None:
                    continue
                entry = {"scene": scene, **sample.context["record"]}
                manifest.append(entry)
                sample.context["record"] = entry
                scene_samples.append(sample)
            if not scene_samples:
                raise ValueError(f"Visibility leaves no training sample for {scene}")
            train.extend(scene_samples)
            record.update(status="calibrated")
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            update_report()
            raise
        update_report()
    return train


def optimize_patch(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> np.ndarray:
    """Run framework PGD with HideBelowThreshold and score every training frame per texture.

    The driving system stops after one detection, so the loss counts only scores above the
    decision threshold minus ``confidence_margin``. ``clean`` is the reconstructed sign without
    insertion, ``replica`` the unpatched replica, ``initial`` the initial patch and ``optimized``
    the PGD result.
    """
    options = cast("DetectorPatchOptimization", attack.optimization)
    with context.services.session(context.seed) as session:
        descriptions = tuple(session.scorers())
        if options.scorer_id is None:
            candidates = [item for item in descriptions if "image" in item.differentiable_inputs]
            if len(candidates) != 1:
                raise ValueError("Detector patch optimization requires exactly one differentiable image scorer; "
                                 "set scorer_id explicitly when multiple scorers are available")
            descriptor = candidates[0]
        else:
            descriptor = next((item for item in descriptions if item.id == options.scorer_id), None)
            if descriptor is None:
                raise ValueError(f"Unavailable detector scorer: {options.scorer_id!r}")
        if "image" not in descriptor.differentiable_inputs:
            raise ValueError(f"Scorer {descriptor.id!r} must support image gradients")
        if descriptor.sensor_id is None or descriptor.decision_threshold is None:
            raise ValueError(f"Scorer {descriptor.id!r} must declare sensor_id and decision_threshold")
        sensor_id = descriptor.sensor_id
        decision_threshold = descriptor.decision_threshold
        scorer = SessionScorer(session, descriptor.id, require_grad=True,
                               input_format=descriptor.inputs.get("image"), device=options.device)
        mask = attack.mask
        train = collect_samples(
            attack, context, original, session=session, scorer=scorer,
            sensor_id=sensor_id, decision_threshold=decision_threshold,
        )
        perturbation = MaskedPatchPerturbation(original, mask, device=options.device)
        initial = perturbation.export(perturbation.init())
        objective = HideBelowThreshold(decision_threshold, options.confidence_margin)
        context.report.set_value(
            "train_frames", len(train), unit="frames",
            description="Captured frames used for PGD updates.",
        )
        context.report.set_value(
            "training_segments", len(context.metadata[:options.max_training_segments]), unit="segments",
            description="Training segments used, after max_training_segments.",
        )
        context.report.set_value(
            "mask_pixels", int(mask.sum()), unit="pixels",
            description="Texture pixels the patch may change.",
        )
        lowers = sorted({_distance_bin(sample) for sample in train})
        progress: list[dict[str, Any]] = []

        def monitor(step: int, delta: torch.Tensor) -> None:
            if step % options.progress_interval and step != options.optimizer.steps:
                return
            with torch.no_grad():
                scores = [float(scorer.target_score(perturbation.apply(sample, delta), sample.target))
                          for sample in train]
            row: dict[str, Any] = {
                "step": step,
                "hinge_loss": float(np.mean([float(objective.loss(score)) for score in scores])),
                "detection_rate": sum(score >= decision_threshold for score in scores) / len(scores),
                "max_confidence": max(scores),
            }
            for lower in lowers:
                members = [score for sample, score in zip(train, scores) if _distance_bin(sample) == lower]
                row[f"detection_rate_{lower}_{lower + 10}m"] = (
                    sum(score >= decision_threshold for score in members) / len(members))
            progress.append(row)
            _report_progress(context.report, progress, lowers, decision_threshold, options.progress_interval)

        result = options.optimizer.run(TextureDataSource(train, seed=context.seed), perturbation,
                                      scorer, objective, monitor=monitor)
        optimized = attack.replica_texture(result.export)
        context.report.set_image(
            "optimized_texture", optimized, description="Replica face with the optimized patch.",
        )
        context.report.set_value(
            "patch_smoothness", float(perturbation.smoothness(result.delta)), unit="RGB distance (0–1)",
            description="Mean Charbonnier color distance between adjacent patch texels; PGD tv_weight "
            "weights this term. Lower is smoother.",
        )
        context.report.set_series(
            "pgd_objective", result.history, x_label="PGD iteration",
            y_label="Mean hinge loss", y_unit="ratio (0–1)",
            description=f"Batch mean of max(0, confidence − (decision_threshold − confidence_margin)) "
            f"before each update; decision_threshold {decision_threshold}, confidence_margin "
            f"{options.confidence_margin}. Zero means every batch frame is below the target.",
        )
        measured = []
        with torch.no_grad():
            for sample in train:
                entry = sample.context["record"]
                row = {"scene": sample.context["scene"], "frame": entry["frame"],
                       "line_distance_m": entry["line_distance_m"]}
                for name, candidate in [("clean", None), ("replica", replica_face()), ("initial", initial),
                                        ("optimized", optimized)]:
                    image = sample.x
                    if candidate is not None:
                        image = sample.context["plan"].composite(sample.x, {(REPLICA_OBJECT_ID, 0): _colors(candidate)})
                    row[f"confidence_{name}"] = float(scorer.target_score(image, sample.target))
                measured.append(row)
        detected = sum(row["confidence_optimized"] >= decision_threshold for row in measured)
        context.report.set_value(
            "detected_frame_fraction", detected / len(measured), unit="ratio (0–1)",
            description=f"Fraction of training frames whose optimized-patch confidence reaches "
            f"decision_threshold ({decision_threshold}). One detection makes the driving system stop, "
            "so the attack needs zero.",
        )
        context.report.set_value(
            "max_confidence", max(row["confidence_optimized"] for row in measured), unit="ratio (0–1)",
            description="Highest optimized-patch confidence over the training frames.",
        )
        context.report.set_table(
            "texture_summary", [
                {"texture": name, "frames": len(measured),
                 "mean_confidence": float(np.mean([row[f"confidence_{name}"] for row in measured])),
                 "max_confidence": max(row[f"confidence_{name}"] for row in measured),
                 "detection_rate": sum(row[f"confidence_{name}"] >= decision_threshold for row in measured)
                 / len(measured)}
                for name in _TEXTURES
            ],
            description="Detector confidence and detection rate across the training frames. " + _TEXTURE_MEANINGS,
            columns={
                "texture": "Texture variant: clean, replica, initial or optimized, as defined above.",
                "frames": "Training frames scored for this texture (frames).",
                "mean_confidence": "Mean stop-sign confidence across training frames (ratio, 0–1).",
                "max_confidence": "Highest stop-sign confidence across training frames (ratio, 0–1).",
                "detection_rate": f"Fraction with confidence ≥ decision_threshold ({decision_threshold}) (ratio, 0–1).",
            },
        )
        context.report.set_table(
            "frame_scores", measured,
            description="One row per training frame, comparing detector confidence for all four texture variants. " +
            _TEXTURE_MEANINGS,
            columns={
                "scene": "Dataset scenario identifier.",
                "frame": "Recorded capture frame position (frames, zero-based).",
                "line_distance_m": "Recorded ego route distance to the stop line (m).",
                **{f"confidence_{name}": f"Detector stop-sign confidence for {name} (ratio, 0–1)."
                   for name in _TEXTURES},
            },
        )
        grouped = defaultdict(list)
        for sample, row in zip(train, measured):
            grouped[_distance_bin(sample)].append(row)
        rates = [
            {"line_distance_bin_m": f"[{lower},{lower + 10})", "frames": len(rows),
             **{f"detection_rate_{name}": sum(row[f"confidence_{name}"] >= decision_threshold for row in rows)
                / len(rows) for name in _TEXTURES}}
            for lower, rows in sorted(grouped.items())
        ]
        context.report.set_table(
            "detection_rate_vs_distance", rates,
            description=f"Fraction of training frames with confidence ≥ decision_threshold "
            f"({decision_threshold}), for each texture in 10 m bins of route distance to the stop line.",
            columns={
                "line_distance_bin_m": "Route distance to the stop line, inclusive lower and exclusive upper bound (m).",
                "frames": "Training frames in this distance bin (frames).",
                **{f"detection_rate_{name}": f"Detection rate for {name} (ratio, 0–1)." for name in _TEXTURES},
            },
        )
        return optimized
