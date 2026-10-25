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
    """Completed treatments sharing one segment unit and seed, plus its saved scenario map.

    ``scenario_map`` comes from the experiment result; metric evaluation never queries datasets.
    """

    segment: AttackSegment
    seed: int
    runs: Mapping[str, RunLog]
    scenario_map: ScenarioMap | Unavailable = field(
        default_factory=lambda: Unavailable("No saved scenario map is available"),
    )
    treatment_attacks: Mapping[str, str | None] = field(default_factory=dict)


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


def signed_stop_line_distances(
    positions: np.ndarray, line_points: np.ndarray, *, endpoint_tolerance_m: float = 3.,
) -> tuple[np.ndarray, tuple[int, float] | None]:
    """Horizontal distances to an endpoint-extended stop polyline and first crossing.

    The first and last segments extend outwards by endpoint_tolerance_m. Samples
    through the crossing segment's start are positive; later samples are negative.
    With no crossing all distances are positive. Heights are ignored.
    """
    require_nonnegative_number(endpoint_tolerance_m, "endpoint_tolerance_m")
    positions = np.asarray(positions, dtype=float)
    line = np.array(line_points, dtype=float, copy=True)
    if (positions.ndim != 2 or positions.shape[1:] != (3,)
            or not np.isfinite(positions).all()):
        raise ValueError("positions must be a finite (N, 3) array")
    if (line.ndim != 2 or line.shape[1:] != (3,) or len(line) < 2
            or not np.isfinite(line).all()):
        raise ValueError("line_points must be a finite (M, 3) array with M >= 2")
    points, line = positions[:, :2], line[:, :2]
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    nonzero = np.flatnonzero(lengths > 0)
    if not len(nonzero):
        raise ValueError("The stop line needs a nonzero horizontal segment")
    first, last = int(nonzero[0]), int(nonzero[-1])
    first_direction = (line[first + 1] - line[first]) / lengths[first]
    last_direction = (line[last + 1] - line[last]) / lengths[last]
    line[:first + 1] -= endpoint_tolerance_m * first_direction
    line[last + 1:] += endpoint_tolerance_m * last_direction
    starts, vectors = line[:-1], np.diff(line, axis=0)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared > 0
    starts, vectors, squared = starts[valid], vectors[valid], squared[valid]
    relative = points[:, None, :] - starts
    fractions = np.clip(np.einsum("nsi,si->ns", relative, vectors) / squared, 0., 1.)
    nearest = starts + fractions[:, :, None] * vectors
    distances = np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=1)
    crossing = None
    for index, (begin, end) in enumerate(zip(points[:-1], points[1:])):
        fractions_on_path = [
            fraction for start, vector in zip(starts, vectors)
            if (fraction := _segment_intersection_fraction(begin, end - begin, start, vector)) is not None
        ]
        if fractions_on_path:
            crossing = (index, min(fractions_on_path))
            distances[index + 1:] *= -1.
            break
    return distances, crossing


def _segment_intersection_fraction(
    origin: np.ndarray, direction: np.ndarray, line_origin: np.ndarray, line_direction: np.ndarray,
) -> float | None:
    """First intersection parameter on a closed horizontal trajectory segment."""
    def cross(first: np.ndarray, second: np.ndarray) -> float:
        return float(first[0] * second[1] - first[1] * second[0])

    offset = line_origin - origin
    determinant = cross(direction, line_direction)
    tolerance = 1e-10
    if abs(determinant) > tolerance:
        fraction = cross(offset, line_direction) / determinant
        along_line = cross(offset, direction) / determinant
        if -tolerance <= fraction <= 1. + tolerance and -tolerance <= along_line <= 1. + tolerance:
            return float(np.clip(fraction, 0., 1.))
        return None
    if abs(cross(offset, line_direction)) > tolerance:
        return None
    squared_length = float(direction @ direction)
    if squared_length <= tolerance:
        along_line = float((origin - line_origin) @ line_direction / (line_direction @ line_direction))
        return 0. if -tolerance <= along_line <= 1. + tolerance else None
    ends = (float(offset @ direction / squared_length),
            float((offset + line_direction) @ direction / squared_length))
    beginning, ending = max(0., min(ends)), min(1., max(ends))
    return beginning if beginning <= ending + tolerance else None


_STOP_LINE_FIELDS = (
    "stopped_before_line", "crossed_without_stop", "line_crossed", "speed_at_line",
    "min_speed_before_line", "max_deceleration", "time_to_stop",
    "min_vehicle_distance_after_line", "initial_distance_to_line", "final_distance_to_line",
)
_STOP_LINE_BOOLEAN_FIELDS = frozenset({"stopped_before_line", "crossed_without_stop", "line_crossed"})


def _unavailable_stop_line_measurements(reason: str) -> dict[str, MetricScalar]:
    return {name: UnavailableMetricValue(reason, "boolean" if name in _STOP_LINE_BOOLEAN_FIELDS else "number")
            for name in _STOP_LINE_FIELDS}


class StopLineMetrics(Metric):
    """Physical stopping and crossing relative to the selected saved map stop line.

    The segment must store its line ID in user_data["stop_line_id"]; geometry comes
    from ComparisonGroup.scenario_map.stop_lines. The initial position is assumed
    to be before the stop line. Distances use the horizontal polyline with endpoint
    extensions, rather than recorded route arc positions.
    """

    def __init__(self, stop_speed: float = .5, endpoint_tolerance_m: float = 3.) -> None:
        require_nonnegative_number(stop_speed, "stop_speed")
        require_nonnegative_number(endpoint_tolerance_m, "endpoint_tolerance_m")
        self.stop_speed = stop_speed
        self.endpoint_tolerance_m = endpoint_tolerance_m

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        if isinstance(group.scenario_map, Unavailable):
            reason = f"The saved scenario map is unavailable: {group.scenario_map.reason}"
        elif "stop_line_id" not in group.segment.user_data:
            reason = "The attack segment has no stop_line_id in user_data"
        else:
            line_id = group.segment.user_data["stop_line_id"]
            try:
                line = group.scenario_map.stop_lines[line_id]
            except (KeyError, TypeError):
                reason = f"Stop line {line_id!r} is absent from the saved scenario map"
            else:
                return {name: self._measure(log, line.points) for name, log in group.runs.items()}
        return {name: _unavailable_stop_line_measurements(reason) for name in group.runs}

    def _measure(self, log: RunLog, line_points: np.ndarray) -> dict[str, MetricScalar]:
        try:
            snapshots = ([log.steps[0].physical_pre] + [step.physical_post for step in log.steps]
                         if log.steps else [event.data["physical_state"] for event in log.events
                                            if event.kind == "initialization"])
            if not snapshots or any(state is None or state.ego is None for state in snapshots):
                raise ValueError("Physical ego samples are unavailable")
            times = np.array([state.t for state in snapshots], dtype=float)
            positions = np.stack([state.ego.pose[:3, 3] for state in snapshots])
            velocities = np.array([state.ego_speed for state in snapshots], dtype=float)
            if (not np.isfinite(times).all() or not np.isfinite(velocities).all()
                    or (velocities < 0).any() or (np.diff(times) <= 0).any()):
                raise ValueError("Physical samples need finite increasing times and nonnegative speeds")
            distances, crossing = signed_stop_line_distances(
                positions, line_points, endpoint_tolerance_m=self.endpoint_tolerance_m,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            return _unavailable_stop_line_measurements(
                f"Physical stop-line evidence is unavailable: {type(error).__name__}: {error}")
        before = slice(None, crossing[0] + 1) if crossing is not None else slice(None)
        stopped_before = bool(np.any(velocities[before] < self.stop_speed))
        stop_indices = np.flatnonzero(velocities < self.stop_speed)
        measurements: dict[str, MetricScalar] = {
            "stopped_before_line": stopped_before,
            "crossed_without_stop": crossing is not None and not stopped_before,
            "line_crossed": crossing is not None,
            "speed_at_line": (float(velocities[crossing[0]] + crossing[1]
                                    * (velocities[crossing[0] + 1] - velocities[crossing[0]]))
                              if crossing is not None else UnavailableMetricValue(
                                  "The physical trace has no crossing of the selected stop line")),
            "min_speed_before_line": float(velocities[before].min()),
            "max_deceleration": (float(max(0., np.max(-np.diff(velocities) / np.diff(times))))
                                 if len(times) > 1 else UnavailableMetricValue(
                                     "At least two physical samples are needed for deceleration")),
            "time_to_stop": (float(times[stop_indices[0]] - times[0]) if len(stop_indices)
                             else UnavailableMetricValue("No sampled speed is below the stop threshold")),
            "initial_distance_to_line": float(distances[0]),
            "final_distance_to_line": float(distances[-1]),
        }
        try:
            nearest = []
            for state in snapshots[crossing[0] + 1:] if crossing is not None else []:
                origin = state.ego.box_transform()[:3, 3]
                for actor in state.actors.values():
                    if actor.category == "vehicle":
                        center = actor.box_transform()[:3, 3]
                        if not np.isfinite(origin).all() or not np.isfinite(center).all():
                            raise ValueError("Vehicle box centres must be finite")
                        nearest.append(float(np.linalg.norm(center - origin)))
            measurements["min_vehicle_distance_after_line"] = (min(nearest) if nearest
                else UnavailableMetricValue("No post-line vehicle centre evidence is available"))
        except (TypeError, ValueError, AttributeError) as error:
            measurements["min_vehicle_distance_after_line"] = UnavailableMetricValue(
                f"Vehicle centre evidence is unavailable: {type(error).__name__}: {error}")
        return measurements


class ReasoningPhraseRate(Metric):
    """Case-insensitive phrase counts and rate over all recorded decision texts.

    A missing text leaves the rate unavailable, while retaining recorded text and
    phrase counts. Empty strings are available text. Components may be native
    objects in memory or mappings after RunLog loading.
    """

    def __init__(self, phrase: str, component: str = "policy", field: str = "reasoning_text") -> None:
        if not isinstance(phrase, str) or not phrase:
            raise ValueError("phrase must be a nonempty string")
        if not isinstance(component, str) or not component or not isinstance(field, str) or not field:
            raise ValueError("component and field must be nonempty strings")
        self.phrase, self.component, self.field = phrase, component, field

    def evaluate(self, group: ComparisonGroup) -> TreatmentMeasurements:
        results: dict[str, dict[str, MetricScalar]] = {}
        for name, log in group.runs.items():
            texts = []
            for step in log.steps:
                component = step.components.get(self.component)
                text = (component.get(self.field) if isinstance(component, Mapping)
                        else getattr(component, self.field, None))
                texts.append(text if isinstance(text, str) else None)
            available = sum(text is not None for text in texts)
            mentions = sum(self.phrase.casefold() in text.casefold() for text in texts if text is not None)
            results[name] = {
                "phrase_steps": mentions,
                "reasoning_steps": available,
                "phrase_rate": (mentions / len(texts) if texts and available == len(texts)
                                else UnavailableMetricValue(
                                    f"Text {self.component}.{self.field} is missing for one or more decisions, "
                                    "or there are no decisions")),
            }
        return results
