"""Offline report records, formatting and statistics for recorded measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metric import Metric


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

    def record_error(self, source: str, error: Exception, **context: Any) -> None:
        """Record an evaluation failure and its group or pair context without losing results."""
        self.metric_errors.append({
            "metric": source, "error": f"{type(error).__name__}: {error}", **context,
        })

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
        parts = list(dict.fromkeys(row["part"] for row in self.treatment_summaries))
        for part in parts:
            rows = [row for row in self.treatment_summaries if row["part"] == part]
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
            if part is not None:
                lines.append(f"Part: {part}")
            lines.extend(_table(table))
            treatments = [row["treatment"] for row in rows]
            for field_name in ("DrivingMetrics.final_speed", "DrivingMetrics.brake_steps"):
                pairs = [row for row in self.pair_summaries if row["part"] == part]
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
                "part", "segment_index", "scenario_id", "seed", "treatment", "attack_name", "status", "error", *fields,
            ])
            writer.writeheader()
            for group in self.groups:
                for name, row in group["treatments"].items():
                    writer.writerow({key: group[key] for key in (
                        "part", "segment_index", "scenario_id", "seed",
                    )} | {"treatment": name, "attack_name": group["treatment_attacks"].get(name),
                         "status": row["status"], "error": row["error"] or ""}
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
