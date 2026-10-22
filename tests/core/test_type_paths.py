"""Explicit callable paths share one public resolver."""

from datetime import datetime
import math
from pathlib import Path

import pytest

from avsectester.type_paths import import_type_path, split_type_path


def test_import_type_path():
    assert import_type_path("pathlib:Path") is Path
    assert import_type_path("math:sqrt") is math.sqrt
    assert import_type_path("datetime:datetime.fromisoformat")("2026-10-22") == datetime(2026, 10, 22)
    with pytest.raises(TypeError, match="is not callable"):
        import_type_path("math:pi")


def test_split_type_path():
    assert split_type_path("datetime:datetime.fromisoformat") == ("datetime", "datetime.fromisoformat")
    assert split_type_path("uninstalled_package.types:Holder.Nested") == (
        "uninstalled_package.types", "Holder.Nested")
    with pytest.raises(ValueError, match="package.module:name"):
        split_type_path("datetime.datetime")
    with pytest.raises(ValueError, match="package.module:name"):
        split_type_path("datetime:datetime..now")
