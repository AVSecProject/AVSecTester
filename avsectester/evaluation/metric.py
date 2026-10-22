"""Metrics computed from recorded logs, without live simulators or driving models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Literal, TypeAlias, TypeGuard

import numpy as np

from avsectester.attack import AttackSegment
from avsectester.data_types import Unavailable
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog
from avsectester.geometry.boxes import oriented_boxes_overlap
from avsectester.validation import (
    require_nonnegative_number, require_positive_integer, require_positive_number,
)


@dataclass(frozen=True)
class UnavailableMetricValue:
    """A missing scalar and its reason; value_type preserves unavailable boolean fields."""

    reason: str
    value_type: Literal["number", "boolean"] = "number"

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("An unavailable metric value needs a reason")
        if self.value_type not in ("number", "boolean"):
            raise ValueError("An unavailable value_type must be number or boolean")


MetricScalar: TypeAlias = int | float | bool | None | UnavailableMetricValue
FieldMeasurements: TypeAlias = Mapping[str, MetricScalar]
TreatmentMeasurements: TypeAlias = Mapping[str, FieldMeasurements]


@dataclass(frozen=True)
class ComparisonGroup:
    """Completed treatments sharing one attack segment and seed, plus its saved scenario map.

    ``scenario_map`` comes from the experiment result; metric evaluation never queries datasets.
    """

    attack_name: str
    segment: AttackSegment
    seed: int
    runs: Mapping[str, RunLog]
    scenario_map: ScenarioMap | Unavailable = field(
        default_factory=lambda: Unavailable("No saved scenario map is available"),
    )


@dataclass(frozen=True)
class TreatmentPair:
    """A completed pair in configuration order; differences are defined as B minus A."""

    treatment_a: str
    treatment_b: str
    log_a: RunLog
    log_b: RunLog


class Metric(ABC):
    """Evaluate recorded data; return scalars or UnavailableMetricValue values carrying their reasons.

    Run metrics accept ComparisonGroup and return a field mapping for each treatment.
    PairMetric implementations accept TreatmentPair and return one field mapping per pair.
    Evaluation must not start simulators or run driving-system components.
    """

    @abstractmethod
    def evaluate(self, data: Any) -> Mapping[str, Any]: ...


class PairMetric(Metric):
    """A metric evaluated once for each completed pair, rather than for individual runs."""

    @abstractmethod
    def evaluate(self, pair: TreatmentPair) -> FieldMeasurements: ...


def speeds(log: RunLog) -> list[float] | None:
    """Physical speed after each decision, or None when any decision has no physical state."""
    if not log.steps:
        return None
    values = []
    for step in log.steps:
        state = step.physical_post
        if state is None or state.ego_speed is None:
            return None
        value = float(state.ego_speed)
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def executed_brakes(log: RunLog) -> list[float] | None:
    """Executed actuator brake commands; trajectory commands do not provide this evidence."""
    if not log.steps or any(step.executed_command is None for step in log.steps):
        return None
    if any(step.executed_command.trajectory is not None for step in log.steps):
        return None
    return [float(step.executed_command.brake) for step in log.steps]


def brake_steps(log: RunLog) -> int | None:
    values = executed_brakes(log)
    if values is None:
        return None
    return sum(value > 0 and step.executed_command.throttle == 0
               for value, step in zip(values, log.steps))


def _recorded_pose_is_available(actor: Any) -> TypeGuard[Any]:
    if actor is None or getattr(actor, "pose", None) is None:
        return False
    pose = np.asarray(actor.pose, dtype=float)
    return pose.shape == (4, 4) and bool(np.isfinite(pose).all())


class DrivingMetrics(Metric):
    """Physical speed, actuator braking, actor distances and top-down box collisions."""

    def __init__(self, stop_speed: float = 0.5) -> None:
        require_nonnegative_number(stop_speed, "stop_speed")
        self.stop_speed = stop_speed

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        result: dict[str, dict[str, MetricScalar]] = {}
        for name, log in group.runs.items():
            values: dict[str, MetricScalar] = {}
            physical_speeds = speeds(log)
            if physical_speeds is None:
                reason = "Physical post-state speeds are missing or the run is empty"
                values.update(dict.fromkeys(("final_speed", "peak_speed", "mean_speed"), UnavailableMetricValue(reason)))
                values["stopped"] = UnavailableMetricValue(reason, "boolean")
            else:
                values.update(final_speed=physical_speeds[-1], peak_speed=max(physical_speeds),
                              mean_speed=float(np.mean(physical_speeds)), stopped=physical_speeds[-1] <= self.stop_speed)
            count = brake_steps(log)
            values["brake_steps"] = count if count is not None else UnavailableMetricValue(
                "Executed actuator commands are unavailable")
            distances = []
            missing_positions, missing_extents, collision = False, False, False
            states = [state for step in log.steps for state in (step.physical_pre, step.physical_post)]
            if not states:
                missing_positions = True
            for state in states:
                ego = state.ego if state is not None else None
                if not _recorded_pose_is_available(ego):
                    missing_positions = True
                    continue
                for actor in state.actors.values():
                    if not _recorded_pose_is_available(actor):
                        missing_positions = True
                        continue
                    ego_box, actor_box = ego.box_transform(), actor.box_transform()
                    assert ego_box is not None and actor_box is not None
                    origin, target = ego_box[:3, 3], actor_box[:3, 3]
                    distance = float(np.linalg.norm(target - origin))
                    distances.append(distance)
                    overlaps = (None if ego.extent is None or actor.extent is None else
                                oriented_boxes_overlap(ego_box, ego.extent, actor_box, actor.extent))
                    missing_extents |= overlaps is None
                    collision |= overlaps is True
            values["min_distance"] = min(distances) if distances and not missing_positions else UnavailableMetricValue(
                "Ego/actor poses are missing or no other actor was recorded")
            values["collision"] = True if collision else (
                UnavailableMetricValue("Ego/actor poses or bounding-box extents are missing", "boolean")
                if missing_positions or missing_extents else False)
            result[name] = values
        return result


class TrajectoryDeviation(PairMetric):
    """Symmetric distance between true ego positions at matching decision indices (metres)."""

    def evaluate(self, pair: TreatmentPair) -> FieldMeasurements:
        fields = ("mean_deviation", "max_deviation", "final_deviation")
        if ([step.step for step in pair.log_a.steps] != [step.step for step in pair.log_b.steps]
                or not pair.log_a.steps):
            return dict.fromkeys(fields, UnavailableMetricValue("Ego poses are missing or decision indices differ"))
        differences = []
        for first, second in zip(pair.log_a.steps, pair.log_b.steps):
            first_ego = first.physical_post.ego if first.physical_post is not None else None
            second_ego = second.physical_post.ego if second.physical_post is not None else None
            if not (_recorded_pose_is_available(first_ego) and _recorded_pose_is_available(second_ego)):
                return dict.fromkeys(fields, UnavailableMetricValue("Physical ego poses are unavailable"))
            differences.append(float(np.linalg.norm(np.asarray(first_ego.pose)[:3, 3] - np.asarray(second_ego.pose)[:3, 3])))
        return dict(zip(fields, (float(np.mean(differences)), max(differences), differences[-1])))


class DetectionMetrics(Metric):
    """Greedy nearest-centre BEV matching in the true ego frame at capture time.

    Truths and predictions use the same BEV max_range; None disables the range limit.
    A nonempty category_map enables class-aware matching. Unlabelled detections then match
    no category. There is no countable true-negative population.
    """

    def __init__(self, match_distance: float = 2.0, max_range: float | None = 50.0,
                 category_map: Mapping[str, str] | None = None) -> None:
        require_positive_number(match_distance, "match_distance")
        if max_range is not None:
            require_positive_number(max_range, "max_range")
        self.match_distance, self.max_range = match_distance, max_range
        self.category_map = dict(category_map or {})

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        result: dict[str, dict[str, MetricScalar]] = {}
        for name, log in group.runs.items():
            if any(step.detections is None for step in log.steps) or not log.steps:
                avstack_type = log.metadata.get("avstack_type", "unknown AVStack")
                raise ValueError(f"DetectionMetrics requires detections from driving system {avstack_type!r} "
                                 f"(treatment {name!r})")
            tp, fp, fn = 0, 0, 0
            unavailable = False
            for step in log.steps:
                world = step.physical_pre
                ego = world.ego if world is not None else None
                if not _recorded_pose_is_available(ego):
                    unavailable = True
                    break
                pose = np.asarray(ego.pose, dtype=float)
                truths = []
                for identity, actor in world.actors.items():
                    if not _recorded_pose_is_available(actor):
                        unavailable = True
                        break
                    box = actor.box_transform()
                    position = pose[:3, :3].T @ (box[:3, 3] - pose[:3, 3])
                    if self.max_range is None or np.linalg.norm(position[:2]) <= self.max_range:
                        truths.append((identity, actor, position))
                if unavailable:
                    break
                detections = step.detections
                assert detections is not None
                predictions = [detection for detection in detections
                               if self.max_range is None or np.linalg.norm(detection.position[:2]) <= self.max_range]
                candidates = []
                for index, detection in enumerate(predictions):
                    for truth_index, (_, actor, position) in enumerate(truths):
                        if self.category_map:
                            if detection.category is None:
                                continue
                            predicted = self.category_map.get(detection.category, detection.category)
                            expected = self.category_map.get(actor.category, actor.category)
                            if predicted != expected:
                                continue
                        distance = float(np.linalg.norm(np.asarray(detection.position)[:2] - position[:2]))
                        if distance <= self.match_distance:
                            candidates.append((distance, index, truth_index))
                matched_predictions: set[int] = set()
                matched_truths: set[int] = set()
                for _, index, truth_index in sorted(candidates):
                    if index not in matched_predictions and truth_index not in matched_truths:
                        matched_predictions.add(index)
                        matched_truths.add(truth_index)
                tp += len(matched_truths)
                fp += len(predictions) - len(matched_predictions)
                fn += len(truths) - len(matched_truths)
            fields = ["tp", "fp", "fn", "precision", "recall", "f1"]
            if unavailable:
                result[name] = dict.fromkeys(fields, UnavailableMetricValue(
                    "Physical ego/actor ground-truth poses are unavailable"))
                continue
            values: dict[str, MetricScalar] = {"tp": tp, "fp": fp, "fn": fn}
            for field_name, numerator, denominator in (("precision", tp, tp + fp), ("recall", tp, tp + fn),
                                                  ("f1", 2 * tp, 2 * tp + fp + fn)):
                values[field_name] = numerator / denominator if denominator else UnavailableMetricValue(
                    "No detections or ground-truth opportunities define this ratio")
            result[name] = values
        return result


class StopImpact(PairMetric):
    """Directional stopping/braking comparisons, evaluated in both A→B and B→A directions."""

    def __init__(self, stop_speed: float = 0.5, reference_speed: float = 1.0,
                 min_overdrive_steps: int = 3) -> None:
        require_nonnegative_number(stop_speed, "stop_speed")
        require_nonnegative_number(reference_speed, "reference_speed")
        require_positive_integer(min_overdrive_steps, "min_overdrive_steps")
        self.stop_speed, self.reference_speed = stop_speed, reference_speed
        self.min_overdrive_steps = min_overdrive_steps

    def evaluate(self, pair: TreatmentPair) -> FieldMeasurements:
        result: dict[str, MetricScalar] = {}
        for direction, reference_log, compared_log in (("a_to_b", pair.log_a, pair.log_b),
                                                 ("b_to_a", pair.log_b, pair.log_a)):
            result.update({f"{direction}.{field}": value
                           for field, value in self._direction(reference_log, compared_log).items()})
        return result

    def _direction(self, reference_log: RunLog, compared_log: RunLog) -> FieldMeasurements:
        booleans = ("reference_drove", "induced_braking", "induced_stop", "suppressed_stop")
        reference_speeds, compared_speeds = speeds(reference_log), speeds(compared_log)
        if reference_speeds is None or compared_speeds is None:
            reason = "Reference or compared run physical speeds are unavailable"
        elif [step.step for step in reference_log.steps] != [step.step for step in compared_log.steps]:
            reason = "Reference and compared run decision indices differ"
        else:
            reason = None
        if reason is not None:
            return dict.fromkeys(booleans, UnavailableMetricValue(reason, "boolean")) | {
                "overdrive_steps": UnavailableMetricValue(reason)}
        assert reference_speeds is not None and compared_speeds is not None
        reference_brakes, braking = brake_steps(reference_log), brake_steps(compared_log)
        overdrive = sum(first <= self.stop_speed and second > self.reference_speed
                        for first, second in zip(reference_speeds, compared_speeds))
        return {
            "reference_drove": max(reference_speeds) >= self.reference_speed,
            "induced_braking": braking > reference_brakes if braking is not None and reference_brakes is not None
            else UnavailableMetricValue("Reference or compared run executed actuator commands are unavailable", "boolean"),
            "induced_stop": compared_speeds[-1] <= self.stop_speed < reference_speeds[-1],
            "suppressed_stop": reference_speeds[-1] <= self.stop_speed and overdrive >= self.min_overdrive_steps,
            "overdrive_steps": overdrive,
        }
