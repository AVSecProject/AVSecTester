"""Byte-exact copies and data descriptions of NumPy arrays, including record arrays.

This module depends only on NumPy, so processes outside the framework environment, such as a
hosted ROS relay, can import it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def item_bytes(array: np.ndarray[Any, Any]) -> bytes:
    """Copy every array item in row order, including the padding inside record items."""
    if array.dtype.itemsize == 0:
        return b""
    # NumPy may copy a strided record array field by field and drop its padding. A view with
    # one opaque field of the same item size copies whole items and keeps the strides.
    return np.asarray(array).view(np.dtype((np.void, array.dtype.itemsize))).tobytes()


def describe_dtype(dtype: np.dtype[Any]) -> str | dict[str, Any]:
    """Describe a dtype as JSON data, including record offsets, titles, item size and subarrays."""
    if dtype.hasobject:
        raise TypeError("Object dtypes cannot be described as data")
    if dtype.fields is not None:
        names = list(dtype.names or ())
        members = [dtype.fields[name] for name in names]
        return {"names": names, "formats": [describe_dtype(member[0]) for member in members],
                "offsets": [member[1] for member in members],
                "titles": [member[2] if len(member) == 3 else None for member in members],
                "itemsize": dtype.itemsize, "aligned": dtype.isalignedstruct}
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return {"base": describe_dtype(base), "shape": list(shape)}
    return dtype.str


def dtype_from_description(description: Any) -> np.dtype[Any]:
    """Rebuild a dtype described by ``describe_dtype``. Object dtypes are rejected."""
    if isinstance(description, str):
        dtype = np.dtype(description)
    elif "base" in description:
        dtype = np.dtype((dtype_from_description(description["base"]), tuple(description["shape"])))
    else:
        dtype = np.dtype({**description,
                          "formats": [dtype_from_description(item) for item in description["formats"]]})
    if dtype.hasobject:
        raise ValueError("Object dtypes cannot be rebuilt from data")
    return dtype


__all__ = ["describe_dtype", "dtype_from_description", "item_bytes"]
