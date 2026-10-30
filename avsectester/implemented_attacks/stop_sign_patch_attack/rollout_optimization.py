"""Optimize stop-sign textures by delaying the first planned stop command."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING, cast

import numpy as np

from avsectester.attack import AttackSegment, OfflineContext
from avsectester.attack.optimization import (
    AdversarialSample, DataSource, OptimizationObjective, Perturbation, Scorer,
)
from avsectester.data_types import TargetSpec
from avsectester.execution.runner import RecordingPolicy
from avsectester.execution.services import CandidateRequest, ClosedLoopSession
from avsectester.runlog import StepRecord
from avsectester.validation import require_positive_integer, require_positive_number

from .assets import initial_patch_texture, texture_from_colors, upsample_cells

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack
    from .offline import RolloutPatchOptimization


def planned_min_speed(trajectory: Sequence) -> float:
    """Return the minimum horizontal speed between consecutive timed waypoints."""
    if trajectory is None or len(trajectory) < 2:
        raise ValueError("Produced trajectory must contain at least two timed waypoints")
    positions = np.asarray([point[0] for point in trajectory], dtype=float)
    if positions.shape != (len(trajectory), 3) or not np.isfinite(positions).all():
        raise ValueError("Produced trajectory positions must be finite XYZ values")
    stamps = [point[2] for point in trajectory]
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in stamps):
        raise ValueError("Produced trajectory timestamps must be integer microseconds")
    elapsed = np.diff(np.asarray(stamps, dtype=np.int64))
    if np.any(elapsed <= 0):
        raise ValueError("Produced trajectory timestamps must increase strictly")
    return float(np.min(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
                        / (elapsed / 1e6)))


def _before_sign(state: Any, sign: np.ndarray) -> bool:
    """Whether the ego origin is still on the approach side of the sign face."""
    position = np.asarray(state.ego.pose[:3, 3], dtype=float)
    return bool((position - sign[:3, 3]) @ sign[:3, 0] > 0)


def _sign_distance(state: Any, sign: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state.ego.pose[:2, 3], dtype=float) - sign[:2, 3]))


def _issues_stop_command(step: StepRecord, stop_speed_mps: float) -> bool:
    return planned_min_speed(step.produced_command.trajectory) < stop_speed_mps


def stop_command_outcome(
    steps: Sequence[StepRecord], sign_pose: np.ndarray, stop_speed_mps: float,
) -> tuple[str, float, int | None]:
    """Return the rollout outcome, its scored distance and the first stop-command index.

    ``stop_command`` scores the horizontal ego-origin distance to the sign at the first stop
    command issued before passing the sign. ``passed_sign`` scores 0. ``segment_ended`` means the
    decisions ran out first and scores the remaining distance after the last decision.
    """
    require_positive_number(stop_speed_mps, "stop_speed_mps")
    sign = np.asarray(sign_pose, dtype=float)
    if sign.shape != (4, 4) or not np.isfinite(sign).all():
        raise ValueError("sign_pose must be a finite 4 by 4 transform")
    if not steps:
        raise ValueError("Stop-command distance requires at least one logged decision")
    for step in steps:
        if not _before_sign(step.physical_pre, sign):
            return "passed_sign", 0., None
        if _issues_stop_command(step, stop_speed_mps):
            return "stop_command", _sign_distance(step.physical_pre, sign), step.step
    final = steps[-1].physical_post
    if not _before_sign(final, sign):
        return "passed_sign", 0., None
    return "segment_ended", _sign_distance(final, sign), None


class LowFrequencyTexturePerturbation(Perturbation):
    """Search bicubic RGB grid offsets while preserving texture alpha and unmasked pixels."""

    def __init__(self, original_rgba: np.ndarray, mask: np.ndarray, cell: int = 32) -> None:
        require_positive_integer(cell, "cell")
        self.original_rgba = np.asarray(original_rgba).copy()
        self.mask = np.asarray(mask).copy()
        self.original_rgb = np.moveaxis(self.original_rgba[..., :3], -1, 0).astype(np.float32) / 255
        texture_from_colors(self.original_rgba, self.mask, self.original_rgb)  # validates both inputs
        if not self.mask.any():
            raise ValueError("Patch mask must contain at least one pixel")
        self.grid_shape = tuple(-(-size // cell) for size in self.mask.shape)
        self.active = np.zeros(self.grid_shape, dtype=bool)
        for index in np.ndindex(self.grid_shape):
            basis = np.zeros(self.grid_shape, dtype=np.float32)
            basis[index] = 1.
            self.active[index] = np.any(upsample_cells(basis, self.mask.shape)[self.mask] != 0)

    def init(self) -> np.ndarray:
        return np.zeros((3, *self.grid_shape), dtype=np.float32)

    def project(self, delta: np.ndarray) -> np.ndarray:
        value = np.asarray(delta, dtype=np.float32)
        if value.shape != (3, *self.grid_shape) or not np.isfinite(value).all():
            raise ValueError("Texture offsets must be finite RGB values at the grid resolution")
        return np.where(self.active[None], np.clip(value, -1., 1.), 0.)

    def apply(self, sample: AdversarialSample, delta: np.ndarray) -> np.ndarray:
        return self.export(delta)

    def export(self, delta: np.ndarray) -> np.ndarray:
        offsets = np.stack([upsample_cells(channel, self.mask.shape)
                            for channel in self.project(delta)])
        return texture_from_colors(self.original_rgba, self.mask, self.original_rgb + offsets)


class MinimizeScore(OptimizationObjective):
    """Use stop-command distance directly as the minimized loss."""

    def loss(self, score: Any) -> Any:
        return score


class SingleTextureSource(DataSource):
    """Contain one physical texture sample for closed-loop scoring.

    The framework requires an AdversarialSample and target box even though closed-loop
    scoring does not use that box. The target spans the texture.
    """

    def __init__(self, original: np.ndarray) -> None:
        self.original = np.asarray(original).copy()
        height, width = self.original.shape[:2]
        self.target = TargetSpec((0, 0, width, height), "stop sign")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> AdversarialSample:
        if index != 0:
            raise IndexError(index)
        return AdversarialSample(self.original, self.target)


def _run_candidate(
    session: ClosedLoopSession, attack: StopSignPatchAttack, segment: AttackSegment,
    texture: np.ndarray, *, seed: int, environment: Mapping[str, Any], stop_speed_mps: float,
    output: Path | None, report_root: Path | None, row: dict[str, Any],
) -> float:
    sign = np.asarray(segment.user_data["sign_pose"], dtype=float)

    def terminate_when(step: StepRecord) -> bool:
        return (not _before_sign(step.physical_pre, sign)
                or _issues_stop_command(step, stop_speed_mps)
                or not _before_sign(step.physical_post, sign))

    row.update(
        scenario_id=segment.scenario_id, start_position=segment.start.position,
        stop_position=segment.stop.position, outcome=None, stop_command_distance_m=None,
        stop_command_step=None, decision_steps=0, status="incomplete", error="",
        run_log=None if output is None or report_root is None else
        output.relative_to(report_root).as_posix(),
    )
    started = perf_counter()
    try:
        log = session.rollout(CandidateRequest(
            segment, seed, segment.decision_steps, environment, RecordingPolicy(representations=()),
            attack=attack, artifacts=texture, output_path=output, terminate_when=terminate_when,
        ))
        row["decision_steps"] = len(log.steps)
        if log.status != "completed" or not log.steps:
            raise ValueError("Candidate must complete at least one logged decision")
        outcome, distance, step = stop_command_outcome(log.steps, sign, stop_speed_mps)
        row.update(status="completed", outcome=outcome, stop_command_distance_m=distance,
                   stop_command_step=step)
        return distance
    except BaseException as error:
        failed_log = getattr(error, "run_log", None)
        if failed_log is not None:
            row["decision_steps"] = len(failed_log.steps)
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        row["elapsed_s"] = perf_counter() - started


class StopCommandScorer(Scorer):
    """Return each texture's mean first stop-command distance across training segments."""

    differentiable = False

    def __init__(
        self, session: ClosedLoopSession, segments: Sequence[AttackSegment], attack: StopSignPatchAttack,
        *, seed: int, environment: Mapping[str, Any], stop_speed_mps: float,
        rollout_directory: str | Path | None = None,
    ) -> None:
        if not segments:
            raise ValueError("StopCommandScorer requires nonempty training segments")
        require_positive_number(stop_speed_mps, "stop_speed_mps")
        self.session, self.segments, self.attack = session, tuple(segments), attack
        self.seed, self.environment, self.stop_speed_mps = seed, dict(environment), stop_speed_mps
        self.rollout_directory = None if rollout_directory is None else Path(rollout_directory)
        self.records: list[dict[str, Any]] = []
        self.queries = 0

    def target_scores(self, images: Sequence[np.ndarray], targets: Sequence[TargetSpec]) -> np.ndarray:
        """Record one closed loop per training segment for each input texture in batch order."""
        if len(images) != len(targets):
            raise ValueError("images and targets must have the same batch length")
        scores = []
        for image in images:
            query_index = self.queries
            self.queries += 1
            distances = []
            for index, segment in enumerate(self.segments):
                output = (None if self.rollout_directory is None else
                          self.rollout_directory / f"{len(self.records):04d}")
                row = {"query_index": query_index, "segment_index": index}
                self.records.append(row)
                distances.append(_run_candidate(
                    self.session, self.attack, segment, image, seed=self.seed,
                    environment=self.environment, stop_speed_mps=self.stop_speed_mps,
                    output=output,
                    report_root=None if self.rollout_directory is None else self.rollout_directory.parent,
                    row=row,
                ))
            scores.append(float(np.mean(distances)))
        return np.asarray(scores, dtype=float)


_QUERY_COLUMNS = {
    "scenario_id": "Dataset scenario identifier.",
    "start_position": "Selected start frame position (zero-based).",
    "stop_position": "Selected terminal frame position (zero-based).",
    "outcome": "How the rollout ended: stop_command, passed_sign or segment_ended.",
    "stop_command_distance_m": ("Scored horizontal ego-origin distance to the sign face: at the first stop "
                                "command, 0 after passing without one, or remaining at the segment end (m)."),
    "stop_command_step": "First stop-command decision index (zero-based), or unavailable.",
    "decision_steps": "Actual number of logged decisions.",
    "status": "Closed-loop query status.",
    "error": "Exception type and message on failure, empty on success.",
    "run_log": ("RunLog directory under eligibility_rollouts/ or optimization_rollouts/, relative to "
                "attacks/<attack>/offline_report/files. Unavailable for in-memory reports."),
    "elapsed_s": "Wall-clock duration of the closed-loop query (s).",
}


def optimize_patch_by_rollouts(
    attack: StopSignPatchAttack, context: OfflineContext, original: np.ndarray,
) -> np.ndarray:
    """Check training-segment eligibility with the clean texture, then minimize the retained
    segments' mean first stop-command distance.
    """
    options = cast("RolloutPatchOptimization", attack.optimization)
    if not context.metadata:
        raise ValueError("Rollout patch optimization requires at least one training segment")
    directory = context.report.directory
    eligibility: list[dict[str, Any]] = []
    selected = []
    scorer = None
    try:
        with context.services.session(context.seed) as session:
            clean = initial_patch_texture(
                "none", attack.mask, noise_std=attack.noise_std, noise_seed=attack.noise_seed,
                noise_cell_texels=attack.noise_cell_texels,
            )
            for index, segment in enumerate(context.metadata):
                sign = np.asarray(segment.user_data["sign_pose"])
                start = np.asarray(segment.user_data["route_points"])[segment.start.position]
                row: dict[str, Any] = {
                    "segment_index": index,
                    "start_distance_m": float(np.linalg.norm(start[:2] - sign[:2, 3])),
                    "retained": False, "reason": "",
                }
                eligibility.append(row)
                _run_candidate(
                    session, attack, segment, clean, seed=context.seed,
                    environment=context.environment, stop_speed_mps=options.stop_speed_mps,
                    output=(None if directory is None else
                            directory / "eligibility_rollouts" / f"{index:04d}"),
                    report_root=directory, row=row,
                )
                if row["outcome"] == "passed_sign":
                    row["reason"] = "No stop command before passing the sign"
                elif row["outcome"] == "segment_ended":
                    row["reason"] = "Segment ends before a stop command or passing the sign"
                elif row["stop_command_step"] < options.min_stop_command_step:
                    row["reason"] = "Stop command occurs before min_stop_command_step"
                else:
                    row.update(retained=True, reason="Retained")
                    selected.append(segment)
                if (options.max_training_segments is not None
                        and len(selected) >= options.max_training_segments):
                    break
            if not selected:
                raise ValueError("No training segments have an eligible stop command")
            context.report.set_value(
                "training_segments", len(selected), unit="segments",
                description="Eligible training segments retained for optimization.",
            )
            perturbation = LowFrequencyTexturePerturbation(
                original, attack.mask, attack.noise_cell_texels)
            scorer = StopCommandScorer(
                session, selected, attack, seed=context.seed, environment=context.environment,
                stop_speed_mps=options.stop_speed_mps,
                rollout_directory=None if directory is None else directory / "optimization_rollouts",
            )
            result = options.optimizer.run(
                SingleTextureSource(original), perturbation, scorer, MinimizeScore())
    finally:
        context.report.set_table(
            "training_segment_eligibility", eligibility,
            description="Clean-texture stop commands and training-segment eligibility.",
            columns={} if not eligibility else {
                "segment_index": "Segment index in the supplied training metadata (zero-based).",
                "start_distance_m": "Horizontal recorded ego-origin distance to the sign at the start (m).",
                "retained": "Whether the segment is retained for optimization.",
                "reason": "Reason for retaining or excluding the segment.",
                **_QUERY_COLUMNS,
            },
        )
        records = [] if scorer is None else scorer.records
        context.report.set_table(
            "optimization_rollouts", records,
            description="Texture queries in query order, including failed closed loops.",
            columns={} if not records else {
                "query_index": ("Texture query index in query order (zero-based). "
                                "Zero is the initial texture."),
                "segment_index": "Retained training segment index (zero-based).",
                **_QUERY_COLUMNS,
            },
        )
        context.report.set_value(
            "optimization_rollout_count", len(records), unit="rollouts",
            description="Optimization closed loops, excluding training-segment eligibility rollouts.",
        )
    optimized = attack.replica_texture(result.export)
    context.report.set_image(
        "optimized_texture", optimized, description="Replica face with the optimized patch.",
    )
    context.report.set_value(
        "final_stop_command_distance_m", float(result.final_score), unit="m",
        description="Mean stop-command distance of the final texture. Smaller distances delay stopping.",
    )
    context.report.set_series(
        "stop_command_distance_m", result.history,
        x_label="NES step (0 = initial texture)", y_unit="m",
        description="Mean stop-command distance after each NES step and for the initial texture.",
    )
    return optimized
