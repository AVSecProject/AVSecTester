"""Evaluate ordered recorded run groups independently of experiment configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import inspect
import itertools
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from avsectester.attack import AttackSegment
from avsectester.data_types import Unavailable
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog

from .metric import (
    ComparisonGroup, Metric, MetricDescription, MetricField, PairMetric, TreatmentPair, UnavailableMetricValue,
)
from .report import EvaluationReport, _json_data


@dataclass(frozen=True)
class RunGroup:
    """Ordered treatment logs sharing a segment and seed, including unsuccessful runs.

    ``logs`` preserves the caller's treatment order. ``errors`` supplies execution or
    reading failures separately from the logs. RunGroup contains every run;
    ComparisonGroup, passed to metrics, contains completed runs only. Maps are saved evidence, never queried
    from a dataset during evaluation.
    """

    segment_index: int
    segment: AttackSegment
    seed: int
    logs: Mapping[str, RunLog]
    errors: Mapping[str, str] = field(default_factory=dict)
    scenario_map: ScenarioMap | Unavailable = field(
        default_factory=lambda: Unavailable("No saved scenario map is available"),
    )
    treatment_attacks: Mapping[str, str | None] = field(default_factory=dict)
    part: str | None = None


def evaluate_groups(
    groups: Sequence[RunGroup], metrics: Sequence[Metric], *, notes: Sequence[str] = (),
) -> EvaluationReport:
    """Evaluate run metrics and each treatment pair in the supplied group and log order.

    Failed and incomplete logs remain in the report with their errors; metrics receive
    completed logs only. Pair differences are B minus A, and each pair metric is evaluated
    once per completed unordered pair. Metric failures are recorded without discarding
    other measurements. The caller chooses metrics and owns all output files.
    """
    report = EvaluationReport(notes=list(notes))
    labels = {type(metric): (type(metric).__name__
                            if sum(type(item).__name__ == type(metric).__name__ for item in metrics) == 1
                            else f"{type(metric).__module__}:{type(metric).__qualname__}")
              for metric in metrics}
    for metric in metrics:
        label = labels[type(metric)]
        try:
            report.metrics.append(_metric_specification(metric, label))
        except Exception as error:
            report.metrics.append({
                "type": f"{type(metric).__module__}:{type(metric).__qualname__}", "options": None,
                "label": label, "summary": "", "fields": {},
                "kind": "pair" if isinstance(metric, PairMetric) else "run",
            })
            report.record_error(label, error, phase="metric description")
    for source in groups:
        complete = {name: log for name, log in source.logs.items() if log.status == "completed"}
        group = ComparisonGroup(source.segment, source.seed, complete,
                                source.scenario_map, source.treatment_attacks)
        context = {"segment_index": source.segment_index, "seed": source.seed, "part": source.part,
                   "treatment_attacks": {name: source.treatment_attacks.get(name) for name in source.logs}}
        treatments: dict[str, dict[str, Any]] = {
            name: {"status": log.status, "error": source.errors.get(name),
                   "metrics": {}, "reasons": {}, "metric_types": {}}
            for name, log in source.logs.items()
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
                report.record_error(type(metric).__name__, error, **context)
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
                        report.record_error(type(metric).__name__, error, treatment_a=first,
                                            treatment_b=second, **context)
            comparisons.append(pair_row)
        report.groups.append({**context, "scenario_id": group.segment.scenario_id,
                              "treatments": treatments, "comparisons": comparisons})
    report.summarize()
    return report


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


def _metric_specification(metric: Metric, label: str) -> dict[str, Any]:
    options = {}
    unavailable = []
    for name, parameter in inspect.signature(type(metric).__init__).parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(metric, name):
            options[name] = _json_data(getattr(metric, name))
        else:
            unavailable.append(name)
    description = metric.describe()
    if not isinstance(description, MetricDescription):
        raise TypeError("Metric.describe must return MetricDescription")
    if any(not isinstance(name, str) or not name or not isinstance(value, MetricField)
           for name, value in description.fields.items()):
        raise TypeError("MetricDescription.fields must map nonempty field names to MetricField")
    specification: dict[str, Any] = {
        "type": f"{type(metric).__module__}:{type(metric).__qualname__}", "options": options,
        "label": label, "summary": description.summary,
        "kind": "pair" if isinstance(metric, PairMetric) else "run",
        "fields": {name: {"description": value.description, "unit": value.unit}
                   for name, value in description.fields.items()},
    }
    if unavailable:
        specification["unavailable_parameters"] = unavailable
    # Invalid user parameter metadata is reported without losing run-status files.
    json.dumps(specification, allow_nan=False)
    return specification


__all__ = [
    'RunGroup',
    'evaluate_groups',
]
