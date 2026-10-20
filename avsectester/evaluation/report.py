"""Offline experiment evaluation and pairwise reports derived only from RunLogs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import csv
import inspect
import itertools
import json
import math
from numbers import Real
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np

from .metric import (
    ComparisonGroup, DetectionMetrics, DrivingMetrics, Metric, PairMetric,
    TrajectoryDeviation, TreatmentPair, UnavailableMetricValue, executed_brakes, speeds,
)

if TYPE_CHECKING:
    from avsectester.experiment.config import ExperimentConfiguration
    from avsectester.experiment.models import ExperimentResult


_SUMMARY_FIELDS = (
    "DrivingMetrics.final_speed", "DrivingMetrics.brake_steps", "DrivingMetrics.stopped",
    "DrivingMetrics.collision", "DetectionMetrics.precision", "DetectionMetrics.recall",
)


@dataclass(frozen=True)
class EvaluationConfiguration:
    """Metrics for the whole experiment; None chooses one data-aware default metric set."""

    metrics: tuple[Metric, ...] | None = None

    def __post_init__(self) -> None:
        if self.metrics is not None:
            values = tuple(self.metrics)
            if any(not isinstance(metric, Metric) for metric in values):
                raise TypeError("evaluation.metrics must contain Metric instances")
            if len({type(metric) for metric in values}) != len(values):
                raise ValueError("evaluation.metrics must not repeat a metric type")
            object.__setattr__(self, "metrics", values)


@dataclass
class EvaluationReport:
    """Per-run and per-pair measurements, missing reasons and population statistics."""

    groups: list[dict[str, Any]] = field(default_factory=list)
    treatment_summaries: list[dict[str, Any]] = field(default_factory=list)
    pair_summaries: list[dict[str, Any]] = field(default_factory=list)
    selections: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metric_errors: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible report data, independent of live experiment objects."""
        return _json_data({
            "groups": self.groups, "treatment_summaries": self.treatment_summaries,
            "pair_summaries": self.pair_summaries, "selections": self.selections,
            "metrics": self.metrics, "notes": self.notes, "metric_errors": self.metric_errors,
            "statistics": {"std_ddof": 0, "groups": "completed runs or completed treatment pairs",
                           "deltas": "treatment B minus treatment A; booleans use 0 and 1"},
        })

    def format_summary(self) -> str:
        """Format treatment tables, mean pairwise-difference matrices and report explanations."""
        lines = []
        attacks = list(dict.fromkeys(row["attack_name"] for row in self.treatment_summaries))
        for attack in attacks:
            rows = [row for row in self.treatment_summaries if row["attack_name"] == attack]
            fields = [name for name in _SUMMARY_FIELDS if any(name in row["metrics"] for row in rows)]
            kinds = {name: next(row["metrics"][name]["kind"] for row in rows if name in row["metrics"])
                     for name in fields}
            headers = ["treatment", "completed/total", "failed"] + [
                f"{name.split('.')[-1]} ({'ratio' if kinds[name] == 'proportion' else 'mean'})"
                for name in fields]
            table = [headers]
            for row in rows:
                values = []
                for name in fields:
                    measurement = row["metrics"].get(name, {})
                    value = measurement.get("true_fraction") if kinds[name] == "proportion" else measurement.get("mean")
                    values.append(_format_number(value))
                table.append([row["treatment"], f"{row['groups']}/{row['total_groups']}",
                              str(row["failed_runs"]), *values])
            lines.append(f"Attack: {attack}")
            lines.extend(_table(table))
            treatments = [row["treatment"] for row in rows]
            for field_name in ("DrivingMetrics.final_speed", "DrivingMetrics.brake_steps"):
                pairs = [row for row in self.pair_summaries if row["attack_name"] == attack]
                if any(field_name in row["deltas"] for row in pairs):
                    lines.append(f"Mean difference B - A: {field_name} (row A, column B)")
                    matrix = [["A \\ B", *treatments]]
                    for first in treatments:
                        cells = []
                        for second in treatments:
                            if first == second:
                                cells.append("—")
                                continue
                            match = next((row for row in pairs if {row["treatment_a"], row["treatment_b"]}
                                          == {first, second}), None)
                            sample = match["deltas"].get(field_name, {}) if match is not None else {}
                            value = sample.get("mean")
                            if value is not None and match is not None and match["treatment_a"] != first:
                                value = -value
                            cells.append(_format_number(value))
                        matrix.append([first, *cells])
                    lines.extend(_table(matrix))
        lines.extend(self.notes)
        for error in self.metric_errors:
            lines.append(f"Evaluation error ({error['metric']}): {error['error']}")
        errors = sum(item["status"] == "error" for item in self.selections)
        if errors:
            lines.append(f"Selection errors: {errors}")
        return "\n".join(lines) if lines else "No treatment runs were recorded."

    def save(self, directory: str | Path) -> Path:
        """Write report.json and runs.csv; metric names prefix columns to protect identities."""
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "report.json").write_text(
            json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        fields = list(dict.fromkeys(name for group in self.groups for row in group["treatments"].values()
                                    for name in row["metrics"]))
        with (destination / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "attack_name", "segment_index", "scenario_id", "seed", "treatment", "status", "error", *fields,
            ])
            writer.writeheader()
            for group in self.groups:
                for name, row in group["treatments"].items():
                    writer.writerow({key: group[key] for key in (
                        "attack_name", "segment_index", "scenario_id", "seed",
                    )} | {"treatment": name, "status": row["status"], "error": row["error"] or ""}
                                    | _json_data(row["metrics"]))
        self.path = destination
        return destination


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return ["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in rows]


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{0.0 if value == 0 else value:.3g}"


def _json_data(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, np.ndarray):
        return _json_data(value.tolist())
    if isinstance(value, Mapping):
        return {str(name): _json_data(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _measurements(values: Any, metric: Metric, label: str) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{type(metric).__name__} must return scalar mappings")
    measurements: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    kinds: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("Metric field names must be nonempty strings")
        field_name = f"{label}.{name}"
        if isinstance(value, UnavailableMetricValue):
            reasons[field_name] = value.reason
            kinds[field_name] = value.value_type
            value = None
        else:
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, bool) and not _numeric(value):
                raise TypeError(f"Metric {field_name} is not a finite scalar or None")
            kinds[field_name] = "boolean" if isinstance(value, bool) else "number"
            if value is None:
                reasons[field_name] = f"{type(metric).__name__}: required recorded data is unavailable"
        measurements[field_name] = value
    return measurements, reasons, kinds


def _statistics(values: Sequence[Any], kind: str) -> dict[str, Any]:
    samples = [value for value in values if value is not None]
    if kind == "boolean":
        return {"kind": "proportion", "count": len(samples),
                "true_fraction": sum(samples) / len(samples) if samples else None}
    return {"kind": "number", "count": len(samples),
            "mean": float(np.mean(samples)) if samples else None,
            "std": float(np.std(samples, ddof=0)) if samples else None, "std_ddof": 0}


def _summarize_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    fields = list(dict.fromkeys(name for row in rows for name in (row[key] or {})))
    result = {}
    for name in fields:
        kinds = {row.get("metric_types", {}).get(name) for row in rows} - {None}
        samples = [(row[key] or {}).get(name) for row in rows]
        present = [value for value in samples if value is not None]
        boolean = (all(isinstance(value, bool) for value in present) if present else "boolean" in kinds)
        kind = "boolean" if boolean and key == "metrics" else "number"
        result[name] = _statistics(samples, kind)
    return result


def _aggregate(report: EvaluationReport) -> None:
    treatments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in report.groups:
        for treatment, row in group["treatments"].items():
            treatments[group["attack_name"], treatment].append(row)
        for pair in group["comparisons"]:
            pairs[group["attack_name"], pair["treatment_a"], pair["treatment_b"]].append(pair)
    report.treatment_summaries = [
        {"attack_name": attack_name, "treatment": treatment,
         "groups": sum(row["status"] == "completed" for row in rows), "total_groups": len(rows),
         "failed_runs": sum(row["status"] != "completed" for row in rows),
         "metrics": _summarize_rows(rows, "metrics")}
        for (attack_name, treatment), rows in treatments.items()
    ]
    report.pair_summaries = [
        {"attack_name": attack_name, "treatment_a": first, "treatment_b": second,
         "groups": sum(row["status"] == "completed" for row in rows), "total_groups": len(rows),
         "deltas": _summarize_rows(rows, "deltas"), "metrics": _summarize_rows(rows, "metrics")}
        for (attack_name, first, second), rows in pairs.items()
    ]


def _metric_specification(metric: Metric) -> dict[str, Any]:
    options = {}
    unavailable = []
    for name, parameter in inspect.signature(type(metric).__init__).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(metric, name):
            options[name] = _json_data(getattr(metric, name))
        else:
            unavailable.append(name)
    specification: dict[str, Any] = {
        "type": f"{type(metric).__module__}:{type(metric).__qualname__}", "options": options}
    if unavailable:
        specification["unavailable_parameters"] = unavailable
    # Invalid user parameter metadata is reported without losing run-status files.
    json.dumps(specification, allow_nan=False)
    return specification


def _default_metrics(result: ExperimentResult) -> tuple[tuple[Metric, ...], list[str]]:
    metrics: tuple[Metric, ...] = (DrivingMetrics(), TrajectoryDeviation())
    complete = [run.log for run in result.runs if run.log.status == "completed"]
    if complete and all(log.steps and all(step.detections is not None for step in log.steps) for log in complete):
        return metrics + (DetectionMetrics(),), []
    reason = ("no completed run provides recorded evidence" if not complete
              else "completed run logs do not all provide detections")
    return metrics, [f"DetectionMetrics omitted: {reason}."]


def _error(report: EvaluationReport, metric: str, error: Exception, **context: Any) -> None:
    report.metric_errors.append({"metric": metric, "error": f"{type(error).__name__}: {error}", **context})


def evaluate(result: ExperimentResult, configuration: ExperimentConfiguration) -> EvaluationReport:
    """Evaluate completed runs and every treatment pair, retaining statuses on metric failure.

    The default metric set is chosen once for the whole experiment. Treatments follow their
    configuration order. Each unordered pair has run-metric differences B minus A and one
    evaluation of each PairMetric; directional metrics report both directions themselves.
    """
    report = EvaluationReport()
    if configuration.evaluation.metrics is None:
        metrics, report.notes = _default_metrics(result)
    else:
        metrics = configuration.evaluation.metrics
    for metric in metrics:
        try:
            report.metrics.append(_metric_specification(metric))
        except Exception as error:
            report.metrics.append({"type": f"{type(metric).__module__}:{type(metric).__qualname__}",
                                   "options": None})
            _error(report, type(metric).__name__, error, phase="parameter description")
    labels = {type(metric): (type(metric).__name__
                            if sum(type(item).__name__ == type(metric).__name__ for item in metrics) == 1
                            else f"{type(metric).__module__}:{type(metric).__qualname__}")
              for metric in metrics}
    grouped: dict[tuple[str, int, int], list[Any]] = defaultdict(list)
    for run in result.runs:
        grouped[run.group].append(run)
    order = {item.name: index for index, item in enumerate(configuration.treatments)}
    for (attack_name, segment_index, seed), original in grouped.items():
        runs = sorted(original, key=lambda run: order[run.treatment])
        complete = {run.treatment: run.log for run in runs if run.log.status == "completed"}
        group = ComparisonGroup(attack_name, runs[0].segment, seed, complete)
        context = {"attack_name": attack_name, "segment_index": segment_index, "seed": seed}
        treatments = {
            run.treatment: {"status": run.log.status, "error": run.error,
                            "metrics": {}, "reasons": {}, "metric_types": {}}
            for run in runs
        }
        for metric in metrics:
            if isinstance(metric, PairMetric) or not complete:
                continue
            try:
                measurements = metric.evaluate(group)
                if not isinstance(measurements, Mapping):
                    raise TypeError(f"{type(metric).__name__}.evaluate must return a mapping")
                updates = {}
                for treatment, values in measurements.items():
                    if treatment not in complete:
                        raise ValueError(f"Metric returned unknown or incomplete treatment {treatment!r}")
                    updates[treatment] = _measurements(values, metric, labels[type(metric)])
                for treatment, (values, reasons, kinds) in updates.items():
                    row = treatments[treatment]
                    row["metrics"].update(values)
                    row["reasons"].update(reasons)
                    row["metric_types"].update(kinds)
            except Exception as error:
                _error(report, type(metric).__name__, error, **context)
        comparisons = []
        for first, second in itertools.combinations(treatments, 2):
            first_row, second_row = treatments[first], treatments[second]
            completed = first in complete and second in complete
            pair_row: dict[str, Any] = {
                "treatment_a": first, "treatment_b": second,
                "status": "completed" if completed else "unavailable",
                "reason": None if completed else f"Pair requires completed runs: {first}={first_row['status']}, "
                f"{second}={second_row['status']}",
                "deltas": {}, "metrics": {} if completed else None, "reasons": {}, "metric_types": {},
            }
            for name in dict.fromkeys([*first_row["metrics"], *second_row["metrics"]]):
                second_value, first_value = second_row["metrics"].get(name), first_row["metrics"].get(name)
                value = (float(second_value) - float(first_value)
                         if completed and second_value is not None and first_value is not None else None)
                pair_row["deltas"][name] = value
                if value is None:
                    pair_row["reasons"][f"delta.{name}"] = pair_row["reason"] or "One run measurement is unavailable"
            if completed:
                pair = TreatmentPair(first, second, complete[first], complete[second])
                for metric in metrics:
                    if not isinstance(metric, PairMetric):
                        continue
                    try:
                        values, reasons, kinds = _measurements(metric.evaluate(pair), metric, labels[type(metric)])
                        pair_row["metrics"].update(values)
                        pair_row["reasons"].update(reasons)
                        pair_row["metric_types"].update(kinds)
                    except Exception as error:
                        _error(report, type(metric).__name__, error, treatment_a=first,
                               treatment_b=second, **context)
            comparisons.append(pair_row)
        report.groups.append({**context, "scenario_id": group.segment.scenario_id,
                              "treatments": treatments, "comparisons": comparisons})
    _aggregate(report)
    report.selections = [item.as_dict() for item in result.selections]
    if configuration.output is not None:
        destination = Path(configuration.output) / "evaluation"
        # Save statuses first: plotting errors must not erase the useful partial report.
        report.save(destination)
        plots = destination / "plots"
        if plots.exists():
            shutil.rmtree(plots)
        plots.mkdir()
        for (attack_name, segment_index, seed), runs in grouped.items():
            try:
                plot_group({run.treatment: run.log for run in sorted(runs, key=lambda run: order[run.treatment])
                            if run.log.status == "completed"},
                           plots / f"{attack_name}-segment-{segment_index:05d}-seed-{seed}.png",
                           title=f"{attack_name}: segment {segment_index}, seed {seed}")
            except Exception as error:
                _error(report, "plot_group", error, attack_name=attack_name,
                       segment_index=segment_index, seed=seed)
        report.save(destination)
    return report


def plot_group(runs: Mapping[str, Any], path: str | Path, *, title: str | None = None) -> Path:
    """Plot physical speeds and executed brakes without changing matplotlib's global backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6))
    FigureCanvasAgg(figure)
    speed_axis, brake_axis = figure.subplots(2, 1, sharex=True)
    if title:
        figure.suptitle(title)
    for name, log in runs.items():
        positions = [step.step for step in log.steps]
        physical_speeds, braking = speeds(log), executed_brakes(log)
        if physical_speeds is not None:
            speed_axis.plot(positions, physical_speeds, label=name)
        if braking is not None:
            brake_axis.plot(positions, braking, label=name)
    speed_axis.set_ylabel("ego speed (m/s)")
    brake_axis.set_ylabel("executed brake")
    brake_axis.set_xlabel("decision step")
    for axis in (speed_axis, brake_axis):
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    figure.clear()
    return destination
