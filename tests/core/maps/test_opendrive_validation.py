"""OpenDRIVE geographic metadata and isolated native validation."""

from pathlib import Path
import signal
import subprocess

import pytest

from avsectester.maps.opendrive import check_opendrive, opendrive_geo_origin


@pytest.mark.parametrize("altitude_key", ["+alt_0", "+=alt_0"])
def test_opendrive_geo_origin(altitude_key):
    xodr = f"""<OpenDRIVE><header><geoReference><![CDATA[
       +proj=tmerc +lon_0=-122.25
       +lat_0=47.5 {altitude_key}=110.75
    ]]></geoReference></header></OpenDRIVE>"""
    assert opendrive_geo_origin(xodr) == (47.5, -122.25, 110.75)
    assert opendrive_geo_origin(xodr.replace(f"{altitude_key}=110.75", "")) is None


def test_check_opendrive():
    pytest.importorskip("carla")
    xodr = (Path(__file__).parent / "fixtures" / "straight_stop.xodr").read_text()
    assert check_opendrive(xodr, 10.) is None


@pytest.mark.parametrize("returncode,stderr,reason", [
    (2, b"s <= road->GetLength()\n", "OpenDRIVE parser rejected the map: s <= road->GetLength()"),
    (-signal.SIGSEGV, b"", "OpenDRIVE parser crashed with SIGSEGV"),
])
def test_check_opendrive_returns_map_rejections(monkeypatch, returncode, stderr, reason):
    def parse(command, **_options):
        return subprocess.CompletedProcess(command, returncode, stdout=b"", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", parse)
    assert check_opendrive(f"<OpenDRIVE><header name='rejected {returncode}'/></OpenDRIVE>", 10.) == reason


@pytest.mark.parametrize("outcome,match", [
    (3, "needs the CARLA Python client"),
    (-signal.SIGKILL, "stopped by SIGKILL"),
    ("timeout", "timed out"),
])
def test_check_opendrive_raises_environment_failures(monkeypatch, outcome, match):
    xodr = f"<OpenDRIVE><header name='failure {outcome}'/></OpenDRIVE>"

    def parse(command, **options):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(command, outcome, stdout=b"",
                                           stderr=b"ModuleNotFoundError: No module named 'carla'\n")

    monkeypatch.setattr(subprocess, "run", parse)
    for _ in range(2):  # A failure is not cached as a map outcome.
        with pytest.raises(RuntimeError, match=match):
            check_opendrive(xodr, 10.)
