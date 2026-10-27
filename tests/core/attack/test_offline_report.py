"""OfflineReport's public collection and portable persistence."""
import csv
import json

import numpy as np
import pytest

from avsectester.attack import OfflineReport, ReportImage, ReportSeries, ReportTable, ReportValue


def test_offline_report(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence.txt").write_text("retained")
    report = OfflineReport(workspace)
    report.set_value("frames", 2)
    report.set_value("frames", 3, unit="frames", description="Captured training frames.")
    report.set_value("label", "<initial>")
    report.set_value("success", True)
    report.set_value("score", .25)
    report.set_series("objective", [1., .5], y_label="loss", y_unit="ratio (0–1)",
                      description="Loss before each update.")
    report.set_series("distance", [2., 3.], x=[10., 20.], x_label="frame",
                      x_unit="frames", y_unit="m")
    rows = [{"frame": 1, "note": "001"}, {"extra": None, "frame": 2, "valid": True}]
    report.set_table("samples", rows, description="Captured samples.",
                     columns={"frame": "Frame index (frames)", "valid": "Whether usable (yes/no)"})
    pixels = np.full((2, 3, 4), 120, np.uint8)
    report.set_image("patch", pixels, description="Initial replica face.")
    pixels[:] = 0
    assert report.images["patch"].pixels[0, 0, 0] == 120
    with pytest.raises(ValueError, match="names"):
        report.set_value("../escape", 0)
    with pytest.raises(ValueError, match="already used"):
        report.set_image("frames", pixels)
    with pytest.raises(ValueError, match="existing columns"):
        report.set_table("samples", rows, columns={"missing": "Unknown"})
    destination = tmp_path / "offline"
    report.status = "failed"
    report.save(destination)
    loaded = OfflineReport.load(destination)
    assert loaded.status == "failed"
    assert loaded.directory == destination / "files"
    assert (loaded.directory / "evidence.txt").read_text() == "retained"
    assert loaded.values == report.values
    assert loaded.values["frames"] == ReportValue(3, "frames", "Captured training frames.")
    assert isinstance(loaded.series["objective"], ReportSeries)
    assert loaded.series == report.series
    assert loaded.series["objective"].x == [0., 1.]
    assert isinstance(loaded.tables["samples"], ReportTable)
    assert loaded.tables == report.tables and loaded.tables["samples"].rows == rows
    assert loaded.tables["samples"].columns["frame"] == "Frame index (frames)"
    assert isinstance(loaded.images["patch"], ReportImage)
    np.testing.assert_array_equal(loaded.images["patch"].pixels, report.images["patch"].pixels)
    assert loaded.images["patch"].description == "Initial replica face."
    with (destination / "samples.csv").open() as stream:
        assert csv.DictReader(stream).fieldnames == ["frame", "note", "extra", "valid"]
    data = json.loads((destination / "report.json").read_text())
    assert data["values"]["frames"] == {"value": 3, "unit": "frames", "description": "Captured training frames."}
    assert data["tables"]["samples"]["rows"] == rows
    assert data["tables"]["samples"]["description"] == "Captured samples."
    assert data["tables"]["samples"]["columns"]["frame"] == "Frame index (frames)"
    assert data["images"]["patch"] == {"file": "patch.png", "description": "Initial replica face."}
    loaded.save(tmp_path / "copy")
    assert ((tmp_path / "copy/files/evidence.txt").read_text()) == "retained"
