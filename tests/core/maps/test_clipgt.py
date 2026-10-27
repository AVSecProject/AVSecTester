"""Clipgt readers preserve source geometry, associations and unavailable layers."""

import zipfile

import numpy as np
import pytest

from avsectester.maps.clipgt import read_clipgt


@pytest.fixture
def nurec_map_variants(nurec_usdz_path, tmp_path):
    """The same reconstruction with a light placeholder or only implicit wait lines."""
    import io
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    with zipfile.ZipFile(nurec_usdz_path) as archive:
        line_rows = parquet.read_table(io.BytesIO(archive.read("clipgt/wait_line.parquet"))).to_pylist()
    for row in line_rows:
        row["wait_line"]["is_implicit"] = True
    replacements = {
        "placeholder_lights": ("traffic_light", [{"key": {"map_id": None}, "traffic_light": {
            "center": None, "dimensions": None, "orientation": None, "category": None}}]),
        "implicit_only_lines": ("wait_line", line_rows),
    }
    paths = {}
    for variant, (layer, rows) in replacements.items():
        path = tmp_path / f"{variant}.usdz"
        member_name = f"clipgt/{layer}.parquet"
        output = arrow.BufferOutputStream()
        parquet.write_table(arrow.Table.from_pylist(rows), output)
        with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(path, "w") as destination:
            for member in source.infolist():
                if member.filename != member_name:
                    destination.writestr(member.filename, source.read(member.filename))
            destination.writestr(member_name, output.getvalue().to_pybytes())
        paths[variant] = str(path)
    unknown_units = tmp_path / "without-speed-unit.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(unknown_units, "w") as destination:
        for member in source.infolist():
            if member.filename != "map.xodr":
                destination.writestr(member.filename, source.read(member.filename))
    paths["without_speed_unit"] = unknown_units
    return paths


def test_read_clipgt(nurec_usdz_path, nurec_map_variants):
    with zipfile.ZipFile(nurec_usdz_path) as archive:
        scenario_map = read_clipgt(archive)
    lane = scenario_map.lanes["lane"]
    np.testing.assert_allclose(lane.centerline, [[-2., 0., 0.], [4., 0., 0.]])
    assert lane.speed_limit_mps == pytest.approx(25 * .44704)
    assert lane.lane_type == "driving" and lane.right_neighbor is None
    assert scenario_map.lanes["bike-lane"].lane_type == "bike"
    assert scenario_map.lanes["bike-lane"].speed_limit_mps is None
    assert scenario_map.unavailable_layers["lanes.speed_limit_mps"]
    assert scenario_map.lanes["shoulder-lane"].lane_type == "shoulder"
    assert scenario_map.lanes["parking-lane"].lane_type == "parking"
    assert scenario_map.lanes["bus-lane"].lane_type == "other"
    sign = scenario_map.traffic_controls["stop"]
    assert sign.kind == "stop_sign" and sign.native_category == "TRAFFIC_SIGN_REGULATORY_R1_STOP"
    assert sign.extent == (.1, .8, 1.2) and sign.lane_ids == ("lane",)
    assert {control.pose_reference for control in scenario_map.traffic_controls.values()} == {"face_center"}
    assert sign.stop_line_id == "stop-line"
    np.testing.assert_allclose(sign.pose[:3, 3], [3., 2., 1.5])
    assert scenario_map.traffic_controls["no-stopping"].kind == "other"
    assert scenario_map.traffic_controls["speed-limit"].kind == "speed_limit"
    assert scenario_map.traffic_controls["yield"].kind == "yield_sign"
    line = scenario_map.stop_lines["stop-line"]
    assert line.kind == "stop_sign" and line.lane_ids == ("lane",) and line.control_id == "stop"
    assert line.native_category == "STOP"
    signal = scenario_map.stop_lines["signal-line"]
    assert signal.kind == "traffic_light" and signal.control_id == "light"
    assert scenario_map.stop_lines["yield-line"].kind == "yield_sign"
    ambiguous = scenario_map.stop_lines["ambiguous-line"]
    assert ambiguous.kind == "traffic_light" and ambiguous.control_id is None
    unknown = scenario_map.stop_lines["unknown-line"]
    assert unknown.kind == "unknown" and unknown.control_id is None
    unbound = scenario_map.stop_lines["unbound-line"]
    assert unbound.kind == "unknown" and unbound.lane_ids == () and unbound.native_category == "STOP"
    assert set(scenario_map.stop_lines) == {"stop-line", "signal-line", "yield-line",
                                          "ambiguous-line", "unknown-line", "unbound-line"}
    assert scenario_map.drivable_area == ()
    assert scenario_map.unavailable_layers["drivable_area"]
    assert scenario_map.unavailable_layers["crosswalks"]

    with zipfile.ZipFile(nurec_map_variants["placeholder_lights"]) as archive:
        placeholder = read_clipgt(archive)
    assert len(placeholder.traffic_controls) == 4
    assert placeholder.unavailable_layers["traffic_controls.traffic_light"]
    assert "traffic_controls" not in placeholder.unavailable_layers
    with zipfile.ZipFile(nurec_map_variants["implicit_only_lines"]) as archive:
        implicit = read_clipgt(archive)
    assert implicit.stop_lines == {} and "stop_lines" not in implicit.unavailable_layers
    assert all(control.stop_line_id is None for control in implicit.traffic_controls.values())

    unknown_units = nurec_map_variants["without_speed_unit"]
    with zipfile.ZipFile(unknown_units) as archive:
        unscaled = read_clipgt(archive)
    assert unscaled.lanes["lane"].speed_limit_mps is None
    assert unscaled.unavailable_layers["lanes.speed_limit_mps"]
