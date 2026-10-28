"""Optimize the patch pixels of a replica stop-sign face against a differentiable detector."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import torch

from avsectester.attack import OfflineContext
from avsectester.attack.insertion import InsertObject
from avsectester.attack.optimization import AdversarialSample, DataSource, HideObject, Perturbation, SessionScorer
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
            sample.x.to(texture.device), {(REPLICA_OBJECT_ID, 0): texture})

    def export(self, delta: torch.Tensor) -> np.ndarray:
        return texture_from_colors(self.original_rgba, self.mask[0].cpu().numpy(),
                                   self.project(delta).detach().cpu().numpy())


def _colors(texture: np.ndarray) -> torch.Tensor:
    """Straight RGB colors of a uint8 RGBA texture as a CHW tensor in [0, 1]."""
    return torch.as_tensor(texture[..., :3].copy(), dtype=torch.float32).permute(2, 0, 1) / 255


_CALIBRATION_COLUMNS = {
    "scene": "Dataset scenario identifier.",
    "selected_start": "Selected training-segment start position (frames, zero-based).",
    "approach_speed_mps": "Recorded ego speed at the selected segment start (m/s).",
    "d_min_m": "Approach speed squared divided by twice the braking deceleration (m).",
    "d_max_m": "Farthest eligible distance where the unpatched replica reaches the detection threshold (m).",
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
    "stage": "Capture purpose: d_max_search or time_uniform_sample.",
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
        "distance_m": note.get("distance_m"), "stage": note.get("stage"),
        "visible_pixels": note.get("visible_pixels"), "status": note.get("status"),
        "reason": note.get("reason"), "occlusion_source": note.get("occlusion_source"),
        "target_box_left_px": box[0], "target_box_top_px": box[1],
        "target_box_right_px": box[2], "target_box_bottom_px": box[3],
        "replica_score": note.get("replica_score"),
    }


def collect_samples(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray, *,
    session: ClosedLoopSession, scorer: SessionScorer, sensor_id: str, decision_threshold: float,
) -> tuple[list[AdversarialSample], list[AdversarialSample]]:
    """Calibrate exact detectable ranges and sample disjoint, time-spaced captures.

    The d_max search covers recording start through the recorded stop, on the entering route
    and facing the selected sign. Frames are searched by descending distance; the first frame
    whose unpatched replica meets the driving scorer's decision threshold is therefore the farthest
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
    dataset = context.services.dataset
    cohort = context.metadata[:options.max_training_segments]
    replica_colors = _colors(replica_face())
    calibration: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    train, heldout = [], []

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
            description=f"One row per attempted capture from camera {sensor_id} during range calibration and time-uniform sampling.",
            columns=_PROBE_COLUMNS if probes else {},
        )
        samples = [{**_capture_row(row["scene"], row), "split": row["split"]} for row in manifest]
        context.report.set_table(
            "sample_manifest", samples,
            description="Time-spaced recorded captures, alternating between PGD training and held-out scoring.",
            columns={**_PROBE_COLUMNS, "split": "Capture role: train for updates or heldout for scoring only."}
            if samples else {},
        )

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
                    "distance_m": float(distances[index]), "stage": stage, "visible_pixels": visible_pixels}
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
            progress = route_progress(positions, positions)
            before = progress <= float(segment.user_data["line_route_distance_m"])
            domain = [index for index in range(stop + 1) if before[index]
                      and dataset.frame_data(scene, frames[index]).ego.pose[:3, 0] @ control.pose[:3, 0] < -.5]
            for index in sorted(domain, key=lambda value: (-distances[value], value)):
                sample = capture(session, index, "d_max_search")
                if sample is not None:
                    with torch.no_grad():
                        replica_image = sample.context["plan"].composite(
                            sample.x, {(REPLICA_OBJECT_ID, 0): replica_colors})
                        score = float(scorer.target_score(replica_image, sample.target))
                    sample.context["record"]["replica_score"] = score
                    if score >= decision_threshold:
                        record["d_max_m"] = float(distances[index])
                        cache[index] = sample
                        break
                update_report()
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
            record.update(status="calibrated")
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            update_report()
            raise
        update_report()
    return train, heldout


def optimize_patch(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> np.ndarray:
    """Run framework PGD/HideObject; score held-out frames with and without the replica.

    ``clean`` is the reconstructed sign without insertion, ``replica`` the unpatched replica,
    ``initial`` the initial patch and ``optimized`` the PGD result.
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
        train, heldout = collect_samples(
            attack, context, original, session=session, scorer=scorer,
            sensor_id=sensor_id, decision_threshold=decision_threshold,
        )
        perturbation = MaskedPatchPerturbation(original, mask, device=options.device)
        initial = perturbation.export(perturbation.init())
        context.report.set_value(
            "train_frames", len(train), unit="frames",
            description="Captured frames used for PGD updates.",
        )
        context.report.set_value(
            "heldout_frames", len(heldout), unit="frames",
            description="Captured frames held out from updates and used only for scoring.",
        )
        context.report.set_value(
            "training_segments", len(context.metadata[:options.max_training_segments]), unit="segments",
            description="Training segments used, after max_training_segments.",
        )
        context.report.set_value(
            "mask_pixels", int(mask.sum()), unit="pixels",
            description="Texture pixels the patch may change.",
        )
        result = options.optimizer.run(TextureDataSource(train, seed=context.seed), perturbation,
                                      scorer, HideObject())
        optimized = attack.replica_texture(result.export)
        context.report.set_image(
            "optimized_texture", optimized, description="Replica face with the optimized patch.",
        )
        context.report.set_value(
            "final_confidence", float(result.final_score), unit="ratio (0–1)",
            description="Mean detector stop-sign confidence on a fresh training batch after the last PGD update. "
            "Lower is better for the attack.",
        )
        context.report.set_series(
            "pgd_confidence", result.history, x_label="PGD iteration",
            y_label="Mean stop-sign confidence", y_unit="ratio (0–1)",
            description="Batch mean detector confidence before each update (HideObject loss).",
        )
        measured = []
        with torch.no_grad():
            for sample in heldout:
                entry = sample.context["record"]
                row = {"scene": sample.context["scene"], "frame": entry["frame"], "distance_m": entry["distance_m"]}
                for name, candidate in [("clean", None), ("replica", replica_face()), ("initial", initial),
                                        ("optimized", optimized)]:
                    image = sample.x
                    if candidate is not None:
                        image = sample.context["plan"].composite(sample.x, {(REPLICA_OBJECT_ID, 0): _colors(candidate)})
                    row[f"confidence_{name}"] = float(scorer.target_score(image, sample.target))
                measured.append(row)
        context.report.set_table(
            "heldout_summary", [
                {"texture": name, "frames": len(measured),
                 "mean_confidence": float(np.mean([row[f"confidence_{name}"] for row in measured])),
                 "detection_rate": sum(row[f"confidence_{name}"] >= decision_threshold for row in measured)
                 / len(measured)}
                for name in _TEXTURES
            ],
            description="Detector confidence and detection rate across frames held out from PGD updates. " + _TEXTURE_MEANINGS,
            columns={
                "texture": "Texture variant: clean, replica, initial or optimized, as defined above.",
                "frames": "Held-out frames used for this texture (frames).",
                "mean_confidence": "Mean stop-sign confidence across held-out frames (ratio, 0–1).",
                "detection_rate": f"Fraction with confidence ≥ decision_threshold ({decision_threshold}) (ratio, 0–1).",
            },
        )
        context.report.set_table(
            "heldout_scores", measured,
            description="One row per held-out frame, comparing detector confidence for all four texture variants. " +
            _TEXTURE_MEANINGS,
            columns={
                "scene": "Dataset scenario identifier.",
                "frame": "Recorded capture frame position (frames, zero-based).",
                "distance_m": "Recorded ego-to-sign face-centre distance in 3D (m).",
                **{f"confidence_{name}": f"Detector stop-sign confidence for {name} (ratio, 0–1)."
                   for name in _TEXTURES},
            },
        )
        grouped = defaultdict(list)
        for row in measured:
            grouped[int(row["distance_m"] // 10) * 10].append(row)
        rates = [
            {"distance_bin_m": f"[{lower},{lower + 10})", "frames": len(rows),
             **{f"detection_rate_{name}": sum(row[f"confidence_{name}"] >= decision_threshold for row in rows)
                / len(rows) for name in _TEXTURES}}
            for lower, rows in sorted(grouped.items())
        ]
        context.report.set_table(
            "detection_rate_vs_distance", rates,
            description=f"Fraction of held-out frames with confidence ≥ decision_threshold "
            f"({decision_threshold}), for each texture in 10 m distance bins.",
            columns={
                "distance_bin_m": "Ego-to-sign distance interval, inclusive lower and exclusive upper bound (m).",
                "frames": "Held-out frames in this distance bin (frames).",
                **{f"detection_rate_{name}": f"Detection rate for {name} (ratio, 0–1)." for name in _TEXTURES},
            },
        )
        return optimized
