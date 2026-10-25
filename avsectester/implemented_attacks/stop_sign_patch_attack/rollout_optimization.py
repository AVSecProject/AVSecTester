"""Optimize the patch pixels of a replica stop-sign face with one-decision rollout queries."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import json
from numbers import Integral
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from PIL import Image

from avsectester.attack import AttackSegment, OfflineContext
from avsectester.attack.optimization import (
    AdversarialSample, DataSource, OptimizationObjective, Perturbation, Scorer,
)
from avsectester.data_types import TargetSpec
from avsectester.datasets.base import FrameReference
from avsectester.execution.runner import RecordingPolicy
from avsectester.execution.services import CandidateRequest, ClosedLoopSession
from avsectester.validation import require_positive_integer, require_positive_number

from .assets import texture_from_colors

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack
    from .offline import RolloutPatchOptimization


def fixed_start_segments(
    segment: AttackSegment, frames: Sequence[FrameReference], can_start_at: Callable[[int], bool],
    distances_m: Sequence[float] = (40., 30., 20.),
) -> tuple[AttackSegment, ...]:
    """Choose distinct eligible one-decision starts nearest the requested sign distances.

    Search only the saved selected approach, before its recorded stop. Positions and the sign
    centre come from the unchanged selection metadata; no route or placement is adjusted.
    """
    positions = np.asarray(segment.user_data["route_points"], dtype=float)
    sign = np.asarray(segment.user_data["sign_pose"], dtype=float)
    if positions.shape != (len(frames), 3) or sign.shape != (4, 4):
        raise ValueError("Saved route and sign pose must match the dataset timeline")
    if any(not np.isfinite(value) or value <= 0 for value in distances_m):
        raise ValueError("Requested sign distances must be finite and positive")
    last = min(int(segment.user_data["recorded_stop_frame"]), segment.stop.position - 1, len(frames) - 2)
    domain = [index for index in range(segment.start.position, last + 1)
              if can_start_at(index) and (positions[index] - sign[:3, 3]) @ sign[:3, 0] > 0]
    distances = np.linalg.norm(positions - sign[:3, 3], axis=1)
    result = []
    for distance in distances_m:
        if not domain:
            raise ValueError("Not enough distinct eligible start frames in the selected approach")
        index = min(domain, key=lambda value: (abs(distances[value] - distance), value))
        domain.remove(index)
        result.append(AttackSegment(segment.scenario_id, frames[index], frames[index + 1], segment.user_data))
    return tuple(result)


def planned_forward_distance(trajectory: Sequence, horizon_s: float = 3.) -> float:
    """Interpolate command X progress from its first timestamp, in metres.

    Waypoints contain position, orientation and an integer microsecond timestamp. Their
    actual time axis must increase strictly and cover the requested horizon; no output
    frequency, policy component log or extrapolation is used.
    """
    require_positive_number(horizon_s, "horizon_s")
    if trajectory is None or len(trajectory) < 2:
        raise ValueError("Produced trajectory must contain at least two timed waypoints")
    positions = np.asarray([point[0] for point in trajectory], dtype=float)
    if positions.shape != (len(trajectory), 3) or not np.isfinite(positions).all():
        raise ValueError("Produced trajectory positions must be finite XYZ values")
    stamps = [point[2] for point in trajectory]
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in stamps):
        raise ValueError("Produced trajectory timestamps must be integer microseconds")
    times = np.asarray(stamps, dtype=np.int64)
    relative = times - times[0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Produced trajectory timestamps must increase strictly")
    target = horizon_s * 1e6
    if relative[-1] < target:
        raise ValueError("Produced trajectory does not cover the requested horizon")
    return float(np.interp(target, relative, positions[:, 0]) - positions[0, 0])


class MaskedTexturePerturbation(Perturbation):
    """A NumPy RGB colour variable whose applied value is the physical uint8 RGBA texture.

    Geometry and alpha stay fixed. This differs from white-box image composition: NES passes
    this exported texture to the rollout scorer, which deploys it as the replica face.
    """

    def __init__(self, original_rgba: np.ndarray, mask: np.ndarray) -> None:
        self.original_rgba = np.asarray(original_rgba).copy()
        self.mask = np.asarray(mask).copy()
        self.original_rgb = np.moveaxis(self.original_rgba[..., :3], -1, 0).astype(np.float32) / 255
        texture_from_colors(self.original_rgba, self.mask, self.original_rgb)
        if not self.mask.any():
            raise ValueError("Patch mask must contain at least one pixel")

    def init(self) -> np.ndarray:
        return self.original_rgb.copy()

    def project(self, delta: np.ndarray) -> np.ndarray:
        value = np.asarray(delta, dtype=np.float32)
        if value.shape != self.original_rgb.shape or not np.isfinite(value).all():
            raise ValueError("Texture colours must be finite RGB CHW values at the original resolution")
        return np.where(self.mask[None], np.clip(value, 0., 1.), self.original_rgb)

    def apply(self, sample: AdversarialSample, delta: np.ndarray) -> np.ndarray:
        return self.export(delta)

    def export(self, delta: np.ndarray) -> np.ndarray:
        return texture_from_colors(self.original_rgba, self.mask, self.project(delta))


class MaximizeScore(OptimizationObjective):
    """Turn positive forward metres into a loss minimized by the optimizer."""

    def loss(self, score: Any) -> Any:
        return -score


class SingleTextureSource(DataSource):
    """Supply the sample required by NES's generic interface.

    The framework interface currently requires an AdversarialSample and target box even
    for rollout scoring; removing that restriction belongs in the framework. The box
    spans the texture and is unused by rollouts. The perturbation exports physical RGBA
    textures without performing object detection.
    """

    def __init__(self, original: np.ndarray) -> None:
        self.original = np.asarray(original).copy()
        height, width = self.original.shape[:2]
        self.target = TargetSpec((0, 0, width, height), "stop sign")

    def sample(self, batch: int) -> list[AdversarialSample]:
        require_positive_integer(batch, "batch")
        return [AdversarialSample(self.original, self.target) for _ in range(batch)]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ["evaluation", "start_index", "scenario_id", "start_position",
               "forward_m", "status", "error", "run_log"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class RolloutForwardScorer(Scorer):
    """Mean planned forward metres across fixed one-decision starts.

    ``image`` in the generic Scorer interface is a physical RGBA texture here; ``target``
    is unused. Records remain in memory. The caller owns CSV persistence, while the
    configured optimizer determines the number of evaluations and rollout queries.
    """

    differentiable = False

    def __init__(
        self, session: ClosedLoopSession, segments: Sequence[AttackSegment], attack: StopSignPatchAttack,
        *, seed: int, environment: Mapping[str, Any], horizon_s: float,
        rollout_directory: str | Path,
    ) -> None:
        if not segments or any(segment.decision_steps != 1 for segment in segments):
            raise ValueError("RolloutForwardScorer requires nonempty one-decision segments")
        require_positive_number(horizon_s, "horizon_s")
        self.session = session
        self.segments = tuple(segments)
        self.attack = attack
        self.seed = seed
        self.environment = dict(environment)
        self.horizon_s = horizon_s
        self.rollout_directory = Path(rollout_directory)
        self.records: list[dict[str, Any]] = []
        self.evaluations = 0

    def target_score(self, image: np.ndarray, target: TargetSpec) -> float:
        evaluation = self.evaluations
        self.evaluations += 1
        forwards = []
        for start_index, segment in enumerate(self.segments):
            output = self.rollout_directory / f"{len(self.records):04d}"
            row: dict[str, Any] = {
                "evaluation": evaluation, "start_index": start_index,
                "scenario_id": segment.scenario_id, "start_position": segment.start.position,
                "forward_m": None, "status": "incomplete", "error": "", "run_log": str(output),
            }
            self.records.append(row)
            try:
                log = self.session.rollout(CandidateRequest(
                    segment, self.seed, 1, self.environment, RecordingPolicy(representations=()),
                    attack=self.attack, artifacts=image, output_path=output,
                ))
                if log.status != "completed" or len(log.steps) != 1:
                    raise ValueError("Candidate must complete exactly one logged decision")
                forward = planned_forward_distance(log.steps[0].produced_command.trajectory, self.horizon_s)
                forwards.append(forward)
                row.update(status="completed", forward_m=forward)
            except BaseException as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
        return float(np.mean(forwards))


def optimize_patch_by_rollouts(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> np.ndarray:
    """Run the supplied optimizer over fixed starts and persist its rollout evidence.

    For NES, the query count is the number of starts times
    ``1 + steps * (population_size + 1)``; this orchestration imposes no separate budget.
    """
    options = cast("RolloutPatchOptimization", attack.optimization)
    mask = attack.mask
    if not context.metadata:
        raise ValueError("Rollout patch optimization requires at least one training segment")
    output = Path(options.output_directory)
    rollout_directory = output / "rollouts"
    if rollout_directory.exists() and any(rollout_directory.iterdir()):
        raise ValueError("Black-box rollout directory must be empty")
    rollout_directory.mkdir(parents=True, exist_ok=True)
    perturbation = MaskedTexturePerturbation(original, mask)
    scorer = None
    try:
        with context.services.session(context.seed) as session:
            starts = tuple(
                start
                for segment in context.metadata[:options.max_training_segments]
                for start in fixed_start_segments(
                    segment, context.services.dataset.frames(segment.scenario_id),
                    session.can_start_at, options.start_distances_m,
                )
            )
            scorer = RolloutForwardScorer(
                session, starts, attack, seed=context.seed, environment=context.environment,
                horizon_s=options.horizon_s, rollout_directory=rollout_directory,
            )
            result = options.optimizer.run(SingleTextureSource(original), perturbation, scorer, MaximizeScore())
    finally:
        _write_csv(output / "rollouts.csv", scorer.records if scorer is not None else [])
    Image.fromarray(perturbation.export(perturbation.init())).save(output / "initial_patch.png")
    optimized = attack.replica_texture(result.export)
    Image.fromarray(optimized).save(output / "optimized_texture.png")
    _write_json(output / "optimization.json", {
        "history": result.history, "final_score": result.final_score,
        "max_training_segments": options.max_training_segments,
        "training_segments": len(context.metadata[:options.max_training_segments]),
        "start_distances_m": list(options.start_distances_m), "horizon_s": options.horizon_s,
        "starts": [{"scenario_id": start.scenario_id, "start_position": start.start.position,
                    "stop_position": start.stop.position, "timestamp": start.start.timestamp}
                   for start in starts],
    })
    return optimized
