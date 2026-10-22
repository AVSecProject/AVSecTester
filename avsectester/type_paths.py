"""Resolve callable Python objects from explicit module-and-attribute paths."""

from importlib import import_module
from typing import Any


def split_type_path(value: str) -> tuple[str, str]:
    """Validate ``package.module:name`` without importing it and return its two parts."""
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError(f"type {value!r} must be 'package.module:name'")
    module, name = value.split(":")
    if not all(part.isidentifier() for part in (*module.split("."), *name.split("."))):
        raise ValueError(f"type {value!r} must be 'package.module:name'")
    return module, name


def import_type_path(value: str) -> Any:
    """Import a callable ``package.module:name``, following dotted attributes in ``name``.

    Missing modules and attributes retain their normal Python import errors.
    """
    module, name = split_type_path(value)
    result = import_module(module)
    for part in name.split("."):
        result = getattr(result, part)
    if not callable(result):
        raise TypeError(f"type {value!r} is not callable")
    return result
