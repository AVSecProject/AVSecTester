"""Configuration validates explicit native-hook registration imports."""
from pathlib import Path

import pytest

from avsectester.experiment.config import _hooks


@pytest.mark.parametrize("imports", ["user.registration", [None], [""]])
def test_native_imports_reject_non_module_names(imports):
    with pytest.raises((ValueError, TypeError), match="module names|must be a sequence"):
        _hooks([{"stage": "perception.post", "native_hook": {"type": "CustomHook"},
                 "imports": imports}], Path("."))
