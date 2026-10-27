"""Structured, portable outputs produced by an attack's offline phase."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ReportValue:
    """One scalar measurement with its unit and explanation."""

    value: Any
    unit: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ReportSeries:
    """A finite numeric series and its axis labels, units and explanation."""

    x: list[float]
    y: list[float]
    x_label: str = "iteration"
    y_label: str | None = None
    x_unit: str | None = None
    y_unit: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class ReportTable:
    """Flat scalar rows and explanations keyed by existing column names."""

    rows: list[dict[str, Any]]
    description: str | None = None
    columns: dict[str, str] | None = None


@dataclass(frozen=True)
class ReportImage:
    """Copied RGB/RGBA pixels and their caption."""

    pixels: np.ndarray
    description: str | None = None


class OfflineReport:
    """Collect offline measurements and files independently of experiment execution.

    Names are shared across values, series, tables and images. Reusing a name within
    one kind replaces its contents. The optional directory is the attack workspace;
    persistence stores its contents under files/ beside the structured report.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = None if directory is None else Path(directory)
        self.status = "completed"
        self.values: dict[str, ReportValue] = {}
        self.series: dict[str, ReportSeries] = {}
        self.tables: dict[str, ReportTable] = {}
        self.images: dict[str, ReportImage] = {}

    def _name(self, name: str, kind: str) -> None:
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError("Report names must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$")
        for other in ("values", "series", "tables", "images"):
            if other != kind and name in getattr(self, other):
                raise ValueError(f"Report name {name!r} is already used by {other}")

    @staticmethod
    def _scalar(value: Any, *, nullable: bool = False) -> Any:
        if value is None and nullable:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real) and math.isfinite(float(value)):
            return float(value)
        raise ValueError("Report values must be finite JSON scalars")

    @staticmethod
    def _text(value: str | None, name: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Report {name} must be a string or None")
        return value

    def set_value(
        self, name: str, value: Any, *, unit: str | None = None,
        description: str | None = None,
    ) -> None:
        """Set a scalar with its optional unit and human-readable explanation."""
        self._name(name, "values")
        metadata = dict(unit=self._text(unit, "unit"),
                        description=self._text(description, "description"))
        self.values[name] = ReportValue(self._scalar(value), **metadata)

    def set_series(
        self, name: str, y: Sequence[float] | np.ndarray, *,
        x: Sequence[float] | np.ndarray | None = None,
        x_label: str = "iteration", y_label: str | None = None,
        x_unit: str | None = None, y_unit: str | None = None,
        description: str | None = None,
    ) -> None:
        """Set a finite one-dimensional series and optional matching x coordinates."""
        self._name(name, "series")
        try:
            yy = np.asarray(y)
            xx = np.arange(len(yy), dtype=float) if x is None else np.asarray(x)
            if yy.dtype.kind not in "iuf" or xx.dtype.kind not in "iuf":
                raise ValueError("Series values must be real numbers")
            yy, xx = yy.astype(float), xx.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Report series must contain finite numeric vectors") from exc
        if yy.ndim != 1 or xx.ndim != 1 or xx.shape != yy.shape:
            raise ValueError("Report series x and y must be vectors with equal lengths")
        if not np.isfinite(xx).all() or not np.isfinite(yy).all():
            raise ValueError("Report series must contain finite numbers")
        if not isinstance(x_label, str) or (y_label is not None and not isinstance(y_label, str)):
            raise ValueError("Report series labels must be strings")
        metadata = dict(x_unit=self._text(x_unit, "x unit"),
                        y_unit=self._text(y_unit, "y unit"),
                        description=self._text(description, "description"))
        self.series[name] = ReportSeries(xx.tolist(), yy.tolist(), x_label, y_label, **metadata)

    def set_table(
        self, name: str, rows: Sequence[Mapping[str, Any]], *,
        description: str | None = None, columns: Mapping[str, str] | None = None,
    ) -> None:
        """Set flat scalar rows and descriptions of existing columns.

        CSV column order follows first appearance across all rows. The optional
        columns mapping describes any subset of those columns, including units.
        """
        self._name(name, "tables")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("Report tables must be a sequence of flat mappings")
        result = []
        for row in rows:
            if not isinstance(row, Mapping) or not all(isinstance(key, str) for key in row):
                raise ValueError("Report table rows must be mappings with string column names")
            result.append({key: self._scalar(value, nullable=True) for key, value in row.items()})
        column_names = {key for row in result for key in row}
        if columns is not None:
            if not isinstance(columns, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in columns.items()
            ):
                raise ValueError("Report column descriptions must be a string mapping")
            unknown = set(columns) - column_names
            if unknown:
                raise ValueError(f"Report column descriptions require existing columns: {sorted(unknown)}")
        self.tables[name] = ReportTable(result, self._text(description, "description"), dict(columns or {}))

    def set_image(
        self, name: str, pixels: np.ndarray, *, description: str | None = None,
    ) -> None:
        """Set a copied uint8 RGB/RGBA image and optional caption."""
        self._name(name, "images")
        array = np.asarray(pixels)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Report images must be uint8 HxWx3 or HxWx4 arrays")
        if not array.shape[0] or not array.shape[1]:
            raise ValueError("Report images must be nonempty")
        self.images[name] = ReportImage(array.copy(), self._text(description, "description"))

    def save(self, directory: str | Path) -> None:
        """Write structured entries, CSV/PNG files and the optional workspace."""
        if self.status not in ("completed", "failed"):
            raise ValueError("Offline report status must be completed or failed")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        files = destination / "files"
        source = self.directory
        if source is not None and source.exists() and source.resolve() != files.resolve():
            if source.resolve() in files.resolve().parents:
                raise ValueError("Cannot copy an offline workspace into its own descendant")
            shutil.copytree(source, files, dirs_exist_ok=True)
        files.mkdir(exist_ok=True)
        tables = {}
        for name, table in self.tables.items():
            columns = list(dict.fromkeys(key for row in table.rows for key in row))
            filename = f"{name}.csv"
            with (destination / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(table.rows)
            tables[name] = dict(file=filename, **asdict(table))
        images = {}
        for name, image in self.images.items():
            filename = f"{name}.png"
            Image.fromarray(image.pixels).save(destination / filename)
            images[name] = dict(file=filename, description=image.description)
        payload = dict(status=self.status,
                       values={name: asdict(value) for name, value in self.values.items()},
                       series={name: asdict(series) for name, series in self.series.items()},
                       tables=tables, images=images)
        (destination / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.directory = files

    @classmethod
    def load(cls, directory: str | Path) -> OfflineReport:
        """Read a saved report without constructing an attack, simulator or model."""
        source = Path(directory)
        data = json.loads((source / "report.json").read_text(encoding="utf-8"))
        result = cls(source / "files")
        status = data["status"]
        if status not in ("completed", "failed"):
            raise ValueError("Offline report status must be completed or failed")
        result.status = status
        for name, value in data.get("values", {}).items():
            result.set_value(name, **value)
        for name, series in data.get("series", {}).items():
            result.set_series(name, **series)
        for name, table in data.get("tables", {}).items():
            result.set_table(name, table["rows"], description=table.get("description"),
                             columns=table.get("columns"))
        for name, image in data.get("images", {}).items():
            with Image.open(source / image["file"]) as pixels:
                result.set_image(name, np.asarray(pixels), description=image.get("description"))
        return result


__all__ = [
    'ReportValue',
    'ReportSeries',
    'ReportTable',
    'ReportImage',
    'OfflineReport',
]
