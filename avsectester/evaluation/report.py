"""Offline report records, formatting and statistics for recorded measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .metric import Metric, UnavailableMetricValue, _require_unique_metric_types


@dataclass(frozen=True)
class EvaluationConfiguration:
    """Metrics for the whole experiment. None chooses one data-aware default metric set."""

    metrics: tuple[Metric, ...] | None = None

    def __post_init__(self) -> None:
        if self.metrics is not None:
            values = tuple(self.metrics)
            if any(not isinstance(metric, Metric) for metric in values):
                raise TypeError("evaluation.metrics must contain Metric instances")
            _require_unique_metric_types(values)
            object.__setattr__(self, "metrics", values)


@dataclass
class EvaluationReport:
    """Per-run and per-pair measurements, missing reasons and population statistics.

    With ``repeats`` above one, group rows carry their repeat number, ``repeat_pairs`` holds
    pair metrics between repeats of one treatment, and pair summaries add mean differences.
    """

    groups: list[dict[str, Any]] = field(default_factory=list)
    treatment_summaries: list[dict[str, Any]] = field(default_factory=list)
    pair_summaries: list[dict[str, Any]] = field(default_factory=list)
    selections: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metric_errors: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None
    experiment: dict[str, Any] = field(default_factory=dict)
    repeats: int = 1
    repeat_pairs: list[dict[str, Any]] = field(default_factory=list)

    def summarize(self) -> None:
        """Rebuild treatment and pair population statistics from the recorded group rows."""
        treatments: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
        pairs: dict[tuple[str | None, str, str], list[dict[str, Any]]] = defaultdict(list)
        for group in self.groups:
            for treatment, row in group["treatments"].items():
                treatments[group["part"], treatment].append(row)
            for pair in group["comparisons"]:
                pairs[group["part"], pair["treatment_a"], pair["treatment_b"]].append(pair)
        self.treatment_summaries = [
            {"part": part, "treatment": treatment,
             "groups": sum(row["status"] == "completed" for row in rows), "total_groups": len(rows),
             "failed_runs": sum(row["status"] != "completed" for row in rows),
             "metrics": _summarize_rows(rows, "metrics")}
            for (part, treatment), rows in treatments.items()
        ]
        self.pair_summaries = [
            {"part": part, "treatment_a": first, "treatment_b": second,
             "groups": sum(row["status"] == "completed" for row in rows), "total_groups": len(rows),
             "deltas": _summarize_rows(rows, "deltas"), "metrics": _summarize_rows(rows, "metrics")}
            for (part, first, second), rows in pairs.items()
        ]
        if self.repeats > 1:
            metrics = {(row["part"], row["treatment"]): row["metrics"]
                       for row in self.treatment_summaries}
            declared_fields = [
                f"{specification['label']}.{name}"
                for specification in self.metrics if specification.get("kind") == "run"
                for name in specification.get("fields", {})
            ]
            for row in self.pair_summaries:
                row.update(_mean_differences(row, metrics, declared_fields))
            repeat_pairs: dict[tuple[str | None, str], list[dict[str, Any]]] = {
                identity: [] for identity in treatments}
            for row in self.repeat_pairs:
                repeat_pairs[row["part"], row["treatment"]].append(row)
            # Same-treatment rows summarize pair metrics between repeats. They have no deltas.
            self.pair_summaries.extend(
                {"part": part, "treatment_a": treatment, "treatment_b": treatment,
                 "groups": len(rows), "total_groups": len(rows), "deltas": {},
                 "metrics": _summarize_rows(rows, "metrics")}
                for (part, treatment), rows in repeat_pairs.items()
            )

    def record_error(self, source: str, error: Exception, **context: Any) -> None:
        """Record an evaluation failure and its group or pair context without losing results."""
        self.metric_errors.append({
            "metric": source, "error": f"{type(error).__name__}: {error}", **context,
        })

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible report data, independent of live experiment objects."""
        statistics = {"std_ddof": 0, "groups": "completed runs or completed treatment pairs",
                      "deltas": "treatment B minus treatment A. Booleans use 0 and 1."}
        repeated: dict[str, Any] = {}
        if self.repeats > 1:
            repeated = {"repeats": self.repeats, "repeat_pairs": self.repeat_pairs}
            statistics |= {
                "mean_differences": "mean of treatment B's completed runs minus mean of treatment A's. "
                                    "Booleans use true fractions.",
                "repeat_pairs": "pair metrics between completed repeats of one treatment in a part, segment "
                                "and seed. Same-treatment pair summaries count these pairs as groups.",
            }
        return _json_data({
            "groups": self.groups, "treatment_summaries": self.treatment_summaries,
            "pair_summaries": self.pair_summaries, "selections": self.selections,
            "metrics": self.metrics, "notes": self.notes, "metric_errors": self.metric_errors,
            "experiment": self.experiment, **repeated, "statistics": statistics,
        })

    def format_summary(self) -> str:
        """Format every metric class by part and treatment, followed by report explanations."""
        lines = []
        parts = list(dict.fromkeys(row["part"] for row in self.treatment_summaries))
        for part in parts:
            rows = [row for row in self.treatment_summaries if row["part"] == part]
            names = list(dict.fromkeys(name for row in rows for name in row["metrics"]))
            classes: dict[str, list[tuple[str, str, str | None]]] = {}
            for specification in self.metrics:
                label = specification.get("label")
                if not label:
                    continue
                fields = [(name, name[len(label) + 1:],
                           specification.get("fields", {}).get(name[len(label) + 1:], {}).get("unit"))
                          for name in names if name.startswith(label + ".")]
                if fields:
                    classes[label] = fields
            matched = {name for fields in classes.values() for name, _, _ in fields}
            other: list[tuple[str, str, str | None]] = [(name, name, None) for name in names if name not in matched]
            if other:
                classes["Metrics"] = other
            if part is not None:
                lines.append(f"Part: {part}")
            for label, fields in classes.items():
                lines.append(label)
                table = [["treatment", "completed/total", "failed",
                          *[f"{field} ({unit})" if unit else field for _, field, unit in fields]]]
                for row in rows:
                    table.append([row["treatment"], f"{row['groups']}/{row['total_groups']}",
                                  str(row["failed_runs"]),
                                  *[format_statistic(row["metrics"].get(name, {})) for name, _, _ in fields]])
                lines.extend(_table(table))
            if not classes:
                lines.extend(_table([["treatment", "completed/total", "failed"], *[
                    [row["treatment"], f"{row['groups']}/{row['total_groups']}",
                     str(row["failed_runs"])] for row in rows]]))
        lines.extend(self.notes)
        for error in self.metric_errors:
            lines.append(f"Evaluation error ({error['metric']}): {error['error']}")
        errors = sum(item["status"] == "error" for item in self.selections)
        if errors:
            lines.append(f"Selection errors: {errors}")
        return "\n".join(lines) if lines else "No treatment runs were recorded."

    def save(self, directory: str | Path) -> Path:
        """Write evaluation report.json and the accompanying run, pair and summary CSVs.

        Metric names prefix fields to protect identities. CSV values retain full precision.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "report.json").write_text(
            json.dumps(self.as_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        fields = list(dict.fromkeys(name for group in self.groups for row in group["treatments"].values()
                                    for name in row["metrics"]))
        repeated = self.repeats > 1
        group_columns = ["part", "segment_index", "scenario_id", "seed"]
        run_columns = group_columns + (["repeat"] if repeated else [])
        with (destination / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                *run_columns, "treatment", "attack_name", "status", "error", *fields,
            ])
            writer.writeheader()
            for group in self.groups:
                for name, row in group["treatments"].items():
                    writer.writerow({key: group.get(key) for key in run_columns}
                                    | {"treatment": name, "attack_name": group["treatment_attacks"].get(name),
                         "status": row["status"], "error": row["error"] or ""}
                                    | _json_data(row["metrics"]))
        statistic_columns = [
            "field", "kind", "count", "mean", "std", "true_fraction",
            "min", "median", "max", "true_count",
        ]
        treatment_columns = [
            "part", "treatment", "groups", "total_groups", "failed_runs",
        ]
        with (destination / "treatment_summary.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=treatment_columns + statistic_columns)
            writer.writeheader()
            for summary in self.treatment_summaries:
                identity = {name: summary[name] for name in treatment_columns}
                for name, statistic in summary["metrics"].items():
                    writer.writerow(identity | {"field": name} | _csv_statistic(statistic))
        pair_columns = [
            "part", "treatment_a", "treatment_b", "groups", "total_groups",
        ]
        with (destination / "pair_summary.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=pair_columns + ["measurement"] + statistic_columns
                + (["reason"] if repeated else []),
            )
            writer.writeheader()
            for summary in self.pair_summaries:
                identity = {name: summary[name] for name in pair_columns}
                for key, measurement in (("deltas", "paired_difference"),
                                         ("metrics", "pair_metric")):
                    for name, statistic in summary[key].items():
                        writer.writerow(identity | {"measurement": measurement, "field": name}
                                        | _csv_statistic(statistic))
                # Same-treatment rows have no mean differences.
                for name, difference in summary.get("mean_differences", {}).items():
                    writer.writerow(identity | {
                        "measurement": "mean_difference", "field": name, "kind": "number",
                        "mean": difference, "reason": summary["mean_difference_reasons"].get(name),
                    })
        repeat_path = destination / "repeat_pair.csv"
        if repeated:
            identities = group_columns + ["treatment", "repeat_a", "repeat_b"]
            pair_fields = list(dict.fromkeys(name for row in self.repeat_pairs for name in row["metrics"]))
            with repeat_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=identities + pair_fields)
                writer.writeheader()
                for row in self.repeat_pairs:
                    writer.writerow({key: row[key] for key in identities} | _json_data(row["metrics"]))
        else:
            repeat_path.unlink(missing_ok=True)
        self.path = destination
        return destination


def _csv_statistic(statistic: Mapping[str, Any]) -> dict[str, Any]:
    proportion = statistic["kind"] == "proportion"
    return _json_data({
        "kind": statistic["kind"], "count": statistic["count"],
        "mean": None if proportion else statistic.get("mean"),
        "std": None if proportion else statistic.get("std"),
        "true_fraction": statistic.get("true_fraction") if proportion else None,
        "min": statistic.get("min"), "median": statistic.get("median"), "max": statistic.get("max"),
        "true_count": statistic.get("true_count") if proportion else None,
    })


def _mean_differences(pair: Mapping[str, Any],
                      treatment_metrics: Mapping[tuple[str | None, str], dict[str, Any]],
                      declared_fields: Sequence[str]) -> dict[str, Any]:
    first, second = pair["treatment_a"], pair["treatment_b"]
    a, b = (treatment_metrics[pair["part"], name] for name in (first, second))
    differences: dict[str, float | None] = {}
    reasons = {}
    for name in dict.fromkeys([*a, *b, *declared_fields]):
        left, right = (entries.get(name, {}) for entries in (a, b))
        value_a = left.get("true_fraction" if left.get("kind") == "proportion" else "mean")
        value_b = right.get("true_fraction" if right.get("kind") == "proportion" else "mean")
        if value_a is not None and value_b is not None:
            differences[name] = value_b - value_a
            continue
        differences[name] = None
        missing = [treatment for treatment, value in ((first, value_a), (second, value_b)) if value is None]
        reasons[name] = f"No completed measurement for {' or '.join(missing)}"
    return {"mean_differences": differences, "mean_difference_reasons": reasons}


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    expanded = []
    for row in rows:
        cells = [value.splitlines() or [""] for value in row]
        expanded.extend([[cell[line] if line < len(cell) else "" for cell in cells]
                         for line in range(max(map(len, cells)))])
    widths = [max(len(row[index]) for row in expanded) for index in range(len(expanded[0]))]
    return ["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
            for row in expanded]


def format_number(value: Any) -> str:
    """Format report scalars without changing JSON or CSV precision.

    Display booleans as yes or no and real values with four decimal places. Missing and
    nonfinite values use an em dash. Integers display without decimal places.
    """
    if value is None or isinstance(value, UnavailableMetricValue):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        formatted = f"{number:.4f}"
        return "0.0000" if formatted == "-0.0000" else formatted
    return str(value)


def format_statistic(statistic: Mapping[str, Any]) -> str:
    """Format population statistics with available-group counts below the measurement.

    One numeric observation shows only its value. Larger samples add the standard deviation
    and, when recorded, the median and range.
    """
    value = (statistic.get("true_fraction") if statistic.get("kind") == "proportion"
             else statistic.get("mean"))
    if value is None:
        return "—"
    count = statistic.get("count", 0)
    if statistic.get("kind") == "proportion":
        true_count = statistic.get("true_count", round(value * count))
        return f"{format_number(float(value))}\n{true_count} of {count} groups"
    lines = [format_number(float(value))]
    if count >= 2 and statistic.get("std") is not None:
        lines[0] += f" ± {format_number(float(statistic['std']))}"
        if statistic.get("median") is not None:
            median, minimum, maximum = (format_number(statistic[key]) for key in ("median", "min", "max"))
            lines.append(f"median {median} [{minimum}, {maximum}]")
    return "\n".join([*lines, f"n = {count}"])


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


def _statistics(values: Sequence[Any], kind: str) -> dict[str, Any]:
    samples = [value for value in values if value is not None]
    if kind == "boolean":
        true_count = int(sum(samples))
        return {"kind": "proportion", "count": len(samples),
                "true_fraction": true_count / len(samples) if samples else None, "true_count": true_count}
    return {"kind": "number", "count": len(samples),
            "mean": float(np.mean(samples)) if samples else None,
            "std": float(np.std(samples, ddof=0)) if samples else None, "std_ddof": 0,
            "min": float(min(samples)) if samples else None,
            "median": float(np.median(samples)) if samples else None,
            "max": float(max(samples)) if samples else None}


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


__all__ = ['EvaluationConfiguration', 'EvaluationReport']
